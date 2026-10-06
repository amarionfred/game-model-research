"""Training-only corpus preparation, duplicate exclusion and causal routing.

Public source: allenai/c4, revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2.
The dataset card declares ODC-By; preserve upstream attribution and document
provenance. This program does not publish raw text.
AI assistance: implemented by Codex; no learner authorship is implied.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import time
import unicodedata

import numpy as np


REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DATA_SEED = 20261006


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def document_key(text):
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def evaluation_split(key):
    return "validation" if int(key[-1], 16) % 2 == 0 else "test"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(text):
    from datasketch import MinHash
    words = normalized_text(text).lower().split()
    shingles = (" ".join(words[i:i + 5]).encode() for i in range(max(1, len(words) - 4)))
    result = MinHash(num_perm=128, seed=DATA_SEED)
    chunk = []
    for shingle in shingles:
        chunk.append(shingle)
        if len(chunk) == 256:
            result.update_batch(chunk)
            chunk = []
    if chunk:
        result.update_batch(chunk)
    return result


def prefix_texts(tokenizer, blocks, *, prefix_tokens=128):
    """Decode only already-observed positions; future targets never reach routing."""
    if blocks.ndim != 2 or not 0 < prefix_tokens < blocks.shape[1]:
        raise ValueError("Invalid causal prefix")
    return tokenizer.decode_batch(blocks[:, :prefix_tokens].astype(int).tolist())


def round_robin_documents(paths):
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(p, "rt", encoding="utf8")) for p in paths]
        active = list(enumerate(streams))
        ordinal = 0
        while active:
            next_active = []
            for index, stream in active:
                line = stream.readline()
                if line:
                    yield index, ordinal, json.loads(line)
                    ordinal += 1
                    next_active.append((index, stream))
            active = next_active


def ingest(raw, database, source_manifest, report, save):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS docs (key TEXT PRIMARY KEY, url TEXT, source TEXT, text TEXT)")
    connection.execute("CREATE INDEX IF NOT EXISTS source_key ON docs(source,key)")
    connection.execute("CREATE TABLE IF NOT EXISTS completed (source TEXT PRIMARY KEY, identity TEXT)")
    identity = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    for split, pattern in [("evaluation", "c4-validation.*.json.gz"), ("train", "c4-train.*.json.gz")]:
        old = connection.execute("SELECT identity FROM completed WHERE source=?", (split,)).fetchone()
        if old:
            if old[0] != identity:
                raise ValueError("Source manifest differs from the ingestion cache")
            continue
        report["stage"] = f"ingest_{split}"
        rows = []
        count = 0
        for _, _, record in round_robin_documents(sorted(raw.glob(pattern))):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = document_key(text)
            url = record.get("url") or key
            rows.append((key, url, split, text))
            count += 1
            if len(rows) == 2000:
                with connection:
                    connection.executemany("INSERT OR IGNORE INTO docs VALUES (?,?,?,?)", rows)
                rows.clear()
                if count % 20000 == 0:
                    report["ingest_documents_seen"] = count
                    save()
        with connection:
            connection.executemany("INSERT OR IGNORE INTO docs VALUES (?,?,?,?)", rows)
            connection.execute("INSERT INTO completed VALUES (?,?)", (split, identity))
        report[f"{split}_raw_documents_seen"] = count
        save()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return connection


def prepare(args):
    from datasketch import MinHashLSH
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stage": "verify_sources", "counts": {},
              "data_seed": DATA_SEED, "revision": REVISION,
              "training_blocks_requested": args.blocks, "evaluation_blocks_per_split": args.eval_blocks,
              "tokenizer_documents_requested": args.tokenizer_documents,
              "context": 256, "vocabulary": 8192}
    started = time.monotonic()
    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "preparation-status.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("stage", "elapsed_seconds", "counts")}), flush=True)
    def timeout(*_):
        raise TimeoutError("CPU preparation exceeded its recorded time budget")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        manifest = json.loads(args.source_manifest.read_text())
        if manifest["revision"] != REVISION or manifest["configuration"] != "en" or len(manifest["files"]) != 6:
            raise ValueError("Unexpected corpus manifest")
        for row in manifest["files"]:
            if file_hash(args.raw / row["file"]) != row["sha256"]:
                raise ValueError("Raw source hash mismatch")
        connection = ingest(args.raw, args.output / "documents.sqlite", manifest, report, save)
        report["stage"] = "reserve_and_deduplicate_evaluation"
        index = MinHashLSH(threshold=0.8, num_perm=128)
        signatures = {}
        eval_rows = []
        reserved_urls = {row[0] for row in connection.execute("SELECT url FROM docs WHERE source='evaluation'")}
        seen_eval_urls = set()
        counts = Counter()
        for key, url, text in connection.execute("SELECT key,url,text FROM docs WHERE source='evaluation' ORDER BY key"):
            if url in seen_eval_urls:
                counts["evaluation_duplicate_url"] += 1
                continue
            seen_eval_urls.add(url)
            sig = signature(text)
            if any(sig.jaccard(signatures[other]) >= 0.8 for other in index.query(sig)):
                counts["evaluation_near_duplicate"] += 1
                continue
            index.insert(key, sig)
            signatures[key] = sig
            eval_rows.append((key, url, evaluation_split(key)))
            counts["evaluation_reserved_documents"] += 1
            if len(eval_rows) % 2000 == 0:
                report["counts"] = dict(counts)
                save()
        # Evaluation text participates in exclusion fingerprints only. None enters
        # vocabulary fitting, feature learning, clustering or student batches.
        train_cursor = connection.execute("SELECT key,url,text FROM docs WHERE source='train' ORDER BY key")
        seen_train_urls = set()
        accepted = []
        def eligible():
            for key, url, text in train_cursor:
                counts["training_documents_screened"] += 1
                if url in reserved_urls or url in seen_train_urls:
                    counts["training_duplicate_or_reserved_url"] += 1
                    continue
                seen_train_urls.add(url)
                sig = signature(text)
                if any(sig.jaccard(signatures[other]) >= 0.8 for other in index.query(sig)):
                    counts["training_near_evaluation"] += 1
                    continue
                yield key, url, text
        iterator = eligible()
        report["stage"] = "select_training_only_tokenizer_documents"
        for key, url, text in iterator:
            accepted.append((key, url))
            if len(accepted) % 2000 == 0:
                report["counts"] = dict(counts)
                save()
            if len(accepted) == args.tokenizer_documents:
                break
        if len(accepted) < args.tokenizer_documents:
            raise ValueError("Insufficient eligible documents; no replacement corpus selected")
        report["stage"] = "fit_training_only_tokenizer"
        save()
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<eos>", "<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        def training_texts():
            for key, _ in accepted:
                yield connection.execute("SELECT text FROM docs WHERE key=?", (key,)).fetchone()[0]
        tokenizer.train_from_iterator(training_texts(), trainer=trainer, length=len(accepted))
        if tokenizer.get_vocab_size() != 8192:
            raise ValueError("Tokenizer did not reach the approved vocabulary size")
        tokenizer.save(str(args.output / "tokenizer.json"))
        (args.output / "tokenizer-training-document-ids.json").write_text(json.dumps([k for k,_ in accepted]) + "\n")
        report["stage"] = "encode_shared_training_pool"
        save()
        train = np.lib.format.open_memmap(args.output / "train.npy", mode="w+", dtype=np.uint16,
                                         shape=(args.blocks, 257))
        provenance = (args.output / "train-block-provenance.jsonl").open("w")
        cursor = 0
        used_train_ids = set()
        def encode_doc(key, url, text):
            nonlocal cursor
            ids = tokenizer.encode(text).ids + [tokenizer.token_to_id("<eos>")]
            counts["training_document_tokens"] += len(ids)
            blocks = (len(ids) - 1) // 256
            counts["training_short_remainder_tokens"] += (len(ids) - 1) % 256
            for block in range(blocks):
                if cursor == args.blocks:
                    break
                train[cursor] = ids[block * 256:block * 256 + 257]
                provenance.write(json.dumps({"block": cursor, "document_sha256": key,
                                             "url": url, "start_token": block * 256}) + "\n")
                used_train_ids.add(key)
                cursor += 1
        try:
            for key, url in accepted:
                encode_doc(key, url, connection.execute("SELECT text FROM docs WHERE key=?", (key,)).fetchone()[0])
                if cursor == args.blocks:
                    break
            for key, url, text in iterator:
                if cursor == args.blocks:
                    break
                encode_doc(key, url, text)
                if counts["training_documents_screened"] % 2000 == 0:
                    report["training_blocks_written"] = cursor
                    report["counts"] = dict(counts)
                    save()
        finally:
            provenance.close()
            train.flush()
        if cursor != args.blocks:
            raise ValueError(f"Only {cursor} training blocks; stop rather than change the source recipe")
        report["stage"] = "encode_separate_evaluation_documents"
        evaluations = {
            split: np.lib.format.open_memmap(args.output / f"{split}.npy", mode="w+", dtype=np.uint16,
                                            shape=(args.eval_blocks, 257))
            for split in ("validation", "test")
        }
        eval_keys = {"validation": [], "test": []}
        for key, url, split in eval_rows:
            if len(eval_keys[split]) >= args.eval_blocks:
                continue
            ids = tokenizer.encode(connection.execute("SELECT text FROM docs WHERE key=?", (key,)).fetchone()[0]).ids
            ids += [tokenizer.token_to_id("<eos>")]
            blocks = (len(ids) - 1) // 256
            if blocks == 0:
                continue
            block = int(key[:16], 16) % blocks
            evaluations[split][len(eval_keys[split])] = ids[block * 256:block * 256 + 257]
            eval_keys[split].append(key)
            if all(len(keys) == args.eval_blocks for keys in eval_keys.values()):
                break
        for split, matrix in evaluations.items():
            matrix.flush()
            if len(eval_keys[split]) != args.eval_blocks:
                raise ValueError(f"Insufficient {split} documents")
            (args.output / f"{split}-document-ids.json").write_text(json.dumps(eval_keys[split]) + "\n")
        assert not (used_train_ids & set(eval_keys["validation"]))
        assert not (used_train_ids & set(eval_keys["test"]))
        assert not (set(eval_keys["validation"]) & set(eval_keys["test"]))
        permutation = np.random.default_rng(DATA_SEED).permutation(args.blocks)
        np.save(args.output / "warm-indices.npy", permutation[:args.blocks // 2])
        np.save(args.output / "continuation-indices.npy", permutation[args.blocks // 2:])
        report.update(status="passed", stage="complete", counts=dict(counts),
                      training_blocks_written=cursor, unique_training_documents=len(used_train_ids),
                      tokenizer_training_documents=len(accepted), approximate_near_duplicate_check=True)
        output_names = ["train.npy", "validation.npy", "test.npy", "tokenizer.json",
                        "warm-indices.npy", "continuation-indices.npy",
                        "tokenizer-training-document-ids.json", "validation-document-ids.json",
                        "test-document-ids.json", "train-block-provenance.jsonl"]
        report["outputs"] = {name: {"sha256": file_hash(args.output / name),
                                   "bytes": (args.output / name).stat().st_size} for name in output_names}
        report["source_manifest_sha256"] = file_hash(args.source_manifest)
        (args.output / "data-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        connection.close()
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        signal.alarm(0)
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=262144)
    parser.add_argument("--eval-blocks", type=int, default=4096)
    parser.add_argument("--tokenizer-documents", type=int, default=100000)
    parser.add_argument("--max-seconds", type=int, default=14400)
    args = parser.parse_args()
    if args.blocks % 16 or min(args.blocks, args.eval_blocks, args.tokenizer_documents) <= 0:
        parser.error("Positive counts and training block count divisible by 16 required")
    prepare(args)


if __name__ == "__main__":
    main()
