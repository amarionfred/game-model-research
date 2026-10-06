"""Bounded CUDA engineering calibration; random tokens are NOT corpus evidence.

AI assistance: implemented by Codex. This command spends GPU time only when
explicitly launched. It neither downloads data nor starts a substantive run.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from .model import BASE_MODELS, CALIBRATORS, TinyGPT
from .training import optimizer_for, set_seed, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=420)
    args = parser.parse_args()
    started = time.monotonic()
    report = {"kind": "engineering_calibration_random_tokens_not_research_results",
              "started_at": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "torch": torch.__version__, "cuda": torch.version.cuda, "models": [],
              "status": "running", "source_commit": os.environ.get("SWARM_SOURCE_COMMIT", "uncommitted")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        torch.set_num_threads(2)
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda", args.device)
        props = torch.cuda.get_device_properties(device)
        report["hardware"] = {"name": props.name, "memory_bytes": props.total_memory}
        family = "small" if args.device == 0 else "medium"
        configs = [(family, BASE_MODELS[family])] + [
            (f"{family}_{k}", CALIBRATORS[f"{family}_{k}"]) for k in (2, 4, 8)
        ]
        for name, config in configs:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("Calibration reached its time budget")
            set_seed(17)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = TinyGPT(config).to(device)
            optimizer = optimizer_for(model, fused=True)
            scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
            blocks = torch.randint(config.vocabulary, (32, config.context + 1), device=device)
            row = {"name": name, "config": asdict(config), "parameters": config.parameters,
                   "train_matmul_flops_per_token": config.train_matmul_flops_per_token,
                   "updates": [], "microbatch": 8, "effective_batch_tokens": 8192}
            report["models"].append(row)
            for step in range(17):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Calibration reached its time budget")
                torch.cuda.synchronize(device)
                tick = time.monotonic()
                result = update(model, optimizer, blocks, microbatch=8, scaler=scaler)
                torch.cuda.synchronize(device)
                result.update(seconds=time.monotonic() - tick, warmup=step < 5, step=step)
                row["updates"].append(result)
                if not result["optimizer_step_applied"]:
                    raise FloatingPointError("Skipped optimizer update in calibration")
                save()
            times = [r["seconds"] for r in row["updates"] if not r["warmup"]]
            row.update(median_update_seconds=statistics.median(times),
                       mean_update_seconds=statistics.mean(times),
                       tokens_per_second=8192 / statistics.mean(times),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       status="passed")
            print(json.dumps({k: row[k] for k in ("name", "parameters", "tokens_per_second",
                                                 "peak_allocated_bytes")}), flush=True)
            del optimizer, model, blocks, scaler
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                      traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
