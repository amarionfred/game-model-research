"""Bounded Phase 1 training, with no evaluation text in the input bundle.

Common parents and their optimizer state are reused across comparisons; each
standalone system must still be charged its common-parent training cost.
AI assistance: implemented by Codex under the user's approved protocol.
"""

import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .data import DATA_SEED, file_hash
from .model import BASE_MODELS, CALIBRATORS, TinyGPT
from .training import load_checkpoint, optimizer_for, phase_learning_rate, save_checkpoint, set_seed, update


TOKENS = 67108864
BATCH = 32
SEEDS = (17, 29, 43)


def schedule(family):
    tasks = []
    base = BASE_MODELS[family]
    for seed in SEEDS:
        parent = f"{family}-s{seed}-warm"
        for arm in ("warm", "dense"):
            tasks.append({"id": f"{family}-s{seed}-{arm}", "family": family,
                          "seed": seed, "arm": arm, "k": 1, "expert": 0,
                          "parent": parent if arm == "dense" else None,
                          "config": asdict(base), "steps": 4096})
        for method in ("clustered", "random"):
            for k in (2, 4, 8):
                for expert in range(k):
                    tasks.append({"id": f"{family}-s{seed}-{method}-k{k}-e{expert}",
                        "family": family, "seed": seed, "arm": method, "k": k,
                        "expert": expert, "parent": parent, "config": asdict(base),
                        "steps": 4096 // k})
        for k in (2, 4, 8):
            config = CALIBRATORS[f"{family}_{k}"]
            steps = (TOKENS * base.train_matmul_flops_per_token //
                     config.train_matmul_flops_per_token // (BATCH * base.context))
            tasks.append({"id": f"{family}-s{seed}-calibrator-k{k}", "family": family,
                "seed": seed, "arm": "calibrator", "k": k, "expert": 0,
                "parent": None, "config": asdict(config), "steps": steps})
    return tasks


def training_order(task, pools, groups):
    arm = task["arm"]
    if arm == "warm":
        indices = pools["warm"]
        salt = 0
    elif arm == "dense":
        indices = pools["continuation"]
        salt = 1
    elif arm == "calibrator":
        indices = np.concatenate([pools["warm"], pools["continuation"]])
        salt = 2
    else:
        partition = groups[f'{arm}-k{task["k"]}']
        indices = partition["block_ids"][partition["assignments"] == task["expert"]]
        salt = 100 + task["k"] * 10 + task["expert"]
    # Clustered/random arms use the same order seed for corresponding branches.
    order_seed = [DATA_SEED, task["seed"], salt]
    order = np.random.default_rng(np.random.SeedSequence(order_seed)).permutation(indices)
    needed = task["steps"] * BATCH
    if needed > len(order):
        raise ValueError("Task asks for more blocks than its permitted pool")
    return order[:needed].astype(np.int64), order_seed


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_input(path):
    manifest = json.loads((path / "training-manifest.json").read_text())
    allowed = {"train.npy", "warm-indices.npy", "continuation-indices.npy"}
    allowed |= {f"{method}-k{k}.npz" for method in ("clustered", "random") for k in (2, 4, 8)}
    if manifest["status"] != "passed" or set(manifest["files"]) != allowed:
        raise ValueError("Unapproved or incomplete training bundle")
    if {p.name for p in path.iterdir() if p.is_file()} - allowed - {"training-manifest.json"}:
        raise ValueError("Training input must contain only the reviewed allowlist")
    for name, sha in manifest["files"].items():
        if file_hash(path / name) != sha:
            raise ValueError(f"Input hash mismatch: {name}")
    blocks = np.load(path / "train.npy", mmap_mode="r", allow_pickle=False)
    if blocks.shape != (262144, 257) or blocks.dtype != np.uint16 or blocks.max() >= 8192:
        raise ValueError("Training blocks differ from the approved shape/vocabulary")
    pools = {name: np.load(path / f"{name}-indices.npy", allow_pickle=False)
             for name in ("warm", "continuation")}
    joined = np.concatenate(list(pools.values()))
    if any(len(v) != 131072 for v in pools.values()) or not np.array_equal(np.sort(joined), np.arange(len(blocks))):
        raise ValueError("Warm and continuation pools must partition the training corpus")
    groups = {}
    for method in ("clustered", "random"):
        for k in (2, 4, 8):
            name = f"{method}-k{k}"
            with np.load(path / f"{name}.npz", allow_pickle=False) as source:
                group = {key: source[key] for key in ("block_ids", "assignments")}
            labels = group["assignments"]
            if not np.array_equal(group["block_ids"], pools["continuation"]):
                raise ValueError("Partition references different continuation blocks")
            if labels.shape != (131072,) or labels.min() < 0 or labels.max() >= k:
                raise ValueError("Invalid specialist labels")
            if not np.array_equal(np.bincount(labels, minlength=k), np.full(k, 131072 // k)):
                raise ValueError("Specialists must receive equal block capacity")
            groups[name] = group
    return blocks, pools, groups


def run_task(task, *, blocks, pools, groups, output, identity_base, device, deadline):
    started = time.monotonic()
    directory = output / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    order, order_seed = training_order(task, pools, groups)
    identity = {**identity_base, "task": task,
                "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
    result_path = directory / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("A previous task used different data/code/configuration")
        if previous["status"] == "passed":
            if file_hash(directory / previous["artifact"]) != previous["artifact_sha256"]:
                raise ValueError("Completed artifact hash mismatch")
            return previous
    rolling = directory / "recovery.pt"
    parent_path = output / task["parent"] / "parent.pt" if task["parent"] else None
    if rolling.exists():
        model, optimizer, scaler, progress = load_checkpoint(
            rolling, device=device, expected_identity=identity, fused=True)
        completed = progress["completed_steps"]
    elif parent_path is not None:
        parent_result = json.loads((parent_path.parent / "result.json").read_text())
        if parent_result["status"] != "passed" or file_hash(parent_path) != parent_result["artifact_sha256"]:
            raise ValueError("Parent has not completed with a verified checkpoint")
        if {k: parent_result["identity"][k] for k in identity_base} != identity_base:
            raise ValueError("Parent code/data identity differs")
        model, optimizer, scaler, _ = load_checkpoint(parent_path, device=device,
            expected_identity=parent_result["identity"], fused=True)
        completed = 0
    else:
        from .model import ModelConfig
        set_seed(task["seed"])
        model = TinyGPT(ModelConfig(**task["config"])).to(device)
        optimizer = optimizer_for(model, fused=True)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        completed = 0
    if asdict(model.config) != task["config"]:
        raise ValueError("Parent architecture differs from branch")
    # Dropout is zero; explicit branch identity still prevents accidental RNG reuse.
    if completed == 0 and parent_path is not None:
        set_seed(int(np.random.SeedSequence(order_seed).generate_state(1)[0]))
    torch.cuda.reset_peak_memory_stats()
    report = {"status": "running", "identity": identity, "order_seed": order_seed,
              "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
              "resumed_from_step": completed, "completed_steps": completed,
              "target_steps": task["steps"], "parameters": model.config.parameters,
              "parent": task["parent"], "batch_tokens": 8192,
              "training_precision": "FP16 autocast; FP32 parameters and AdamW state",
              "strict_end_to_end_compute_matching": "not yet verified",
              "held_out_results": None}
    atomic_json(result_path, report)
    def checkpoint():
        required = 2 * 16 * model.config.parameters + 256 * 1024 ** 2
        if shutil.disk_usage(output).free < required:
            raise OSError("Insufficient reserve for atomic recovery checkpoint")
        save_checkpoint(rolling, model, optimizer, scaler=scaler,
                        progress={"completed_steps": completed}, identity=identity)
    try:
        with (directory / "updates.jsonl").open("a") as log:
            for step in range(completed, task["steps"]):
                if time.monotonic() > deadline - 120:
                    raise TimeoutError("Stopping with recovery time before the job deadline")
                learning_rate = phase_learning_rate(step, task["steps"])
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                batch = torch.from_numpy(np.array(blocks[order[step * BATCH:(step + 1) * BATCH]]))
                before = time.monotonic()
                measured = update(model, optimizer, batch, microbatch=8, scaler=scaler)
                if not measured["optimizer_step_applied"]:
                    raise FloatingPointError("A skipped optimizer step invalidates the exact token budget")
                completed = step + 1
                log.write(json.dumps({"step": completed, "learning_rate": learning_rate,
                    "wall_seconds": time.monotonic() - before, **measured}) + "\n")
                if completed % 128 == 0:
                    log.flush()
                    checkpoint()
                    report.update(completed_steps=completed, last_training_loss=measured["loss"],
                                  elapsed_seconds=time.monotonic() - started)
                    atomic_json(result_path, report)
                    print(json.dumps({"task": task["id"], "step": completed,
                                      "loss": measured["loss"]}), flush=True)
        if task["arm"] == "warm":
            checkpoint()
            artifact = directory / "parent.pt"
            rolling.replace(artifact)
            storage = "FP32 model plus optimizer/scaler/RNG for branching"
        else:
            artifact = directory / "weights.pt"
            temporary = artifact.with_name("weights.pt.tmp")
            torch.save({"format_version": 1, "config": task["config"], "identity": identity,
                        "model": {k: v.detach().half().cpu() for k, v in model.state_dict().items()}}, temporary)
            temporary.replace(artifact)
            storage = "FP16 weights for evaluation; common parents retain FP32 state"
            if rolling.exists():
                rolling.unlink()
        report.update(status="passed", artifact=artifact.name, artifact_sha256=file_hash(artifact),
                      artifact_storage=storage)
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        try:
            checkpoint()
        except BaseException as recovery_error:
            report["recovery_error"] = str(recovery_error)
        raise
    finally:
        report.update(completed_steps=completed, tokens=completed * 8192,
            executed_steps_this_attempt=completed - report["resumed_from_step"],
            elapsed_seconds=time.monotonic() - started,
            analytic_matmul_training_flops=completed * 8192 * model.config.train_matmul_flops_per_token,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(result_path, report)
        with (directory / "attempts.jsonl").open("a") as log:
            log.write(json.dumps(report) + "\n")
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(BASE_MODELS), required=True)
    parser.add_argument("--max-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not 300 <= args.max_seconds <= 21600:
        parser.error("Each device is bounded to at most six hours")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expose exactly one approved T4 per worker with CUDA_VISIBLE_DEVICES")
    if "T4" not in torch.cuda.get_device_name(0):
        parser.error("This recipe was calibrated on T4; hardware differs")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.max_seconds
    def timeout(*_):
        raise TimeoutError("Worker reached its six-hour wall-time cap")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    report = {"status": "running", "family": args.family, "max_seconds": args.max_seconds,
              "completed_tasks": [], "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "cash_budget": 0}
    try:
        blocks, pools, groups = verify_input(args.input)
        sources = ["run_grid.py", "training.py", "model.py", "data.py"]
        identity = {"training_manifest_sha256": file_hash(args.input / "training-manifest.json"),
                    "code_sha256": {name: file_hash(Path(__file__).parent / name) for name in sources}}
        report["identity"] = identity
        tasks = schedule(args.family)
        atomic_json(args.output / "schedule.json", tasks)
        for task in tasks:
            run_task(task, blocks=blocks, pools=pools, groups=groups, output=args.output,
                     identity_base=identity, device="cuda:0", deadline=deadline)
            report["completed_tasks"].append(task["id"])
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(args.output / "worker-status.json", report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "worker-status.json", report)


if __name__ == "__main__":
    main()
