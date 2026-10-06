"""Balanced specialization groups and prefix-only distance routing.

Inspired by c-BTM (2303.14177): TF-IDF, SVD and k-means. Block-level balancing
here is a declared small-scale adaptation, not the original released algorithm.
AI assistance: implemented by Codex.
"""

import argparse
import json
from pathlib import Path
import signal
import time

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from .data import DATA_SEED, file_hash, prefix_texts


def squared_distances(vectors, centers):
    return np.maximum(
        (vectors * vectors).sum(1, keepdims=True) + (centers * centers).sum(1)[None, :]
        - 2 * vectors @ centers.T, 0,
    )


def balanced_assignment(vectors, centers, block_ids):
    count, groups = len(vectors), len(centers)
    if count % groups:
        raise ValueError("Equal block capacity requires divisibility by K")
    distances = squared_distances(vectors, centers)
    choices = np.argsort(distances, axis=1, kind="stable")
    margin = distances[np.arange(count), choices[:, 1]] - distances[np.arange(count), choices[:, 0]]
    order = np.lexsort((block_ids, -margin))
    remaining = np.full(groups, count // groups)
    assignments = np.empty(count, dtype=np.int16)
    for row in order:
        group = next(int(g) for g in choices[row] if remaining[g] > 0)
        assignments[row] = group
        remaining[group] -= 1
    return assignments


def route_weights(prefix_vectors, centers, *, top_k=None, temperature=0.1):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scores = -squared_distances(prefix_vectors, centers) / temperature
    if top_k is not None:
        if not 1 <= top_k <= len(centers):
            raise ValueError("top_k outside the available experts")
        keep = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        mask = np.full(scores.shape, -np.inf)
        np.put_along_axis(mask, keep, np.take_along_axis(scores, keep, axis=1), axis=1)
        scores = mask
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)


def cluster(args):
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = json.loads((args.data / "data-manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("Data preparation did not pass")
    for name in ["train.npy", "tokenizer.json", "continuation-indices.npy"]:
        if file_hash(args.data / name) != manifest["outputs"][name]["sha256"]:
            raise ValueError("Training artifact hash mismatch")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    blocks = np.load(args.data / "train.npy", mmap_mode="r")
    continuation = np.load(args.data / "continuation-indices.npy")
    report = {"status": "running", "method": "training_only_tfidf_svd_balanced_block_kmeans",
              "data_manifest_sha256": file_hash(args.data / "data-manifest.json"), "groups": {}}
    def save(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started)
        (args.output / "clustering-status.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": stage, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    save("decode_training_blocks")
    texts = []
    for start in range(0, len(blocks), 512):
        texts.extend(tokenizer.decode_batch(blocks[start:start + 512, :256].astype(int).tolist()))
    with threadpool_limits(limits=2):
        vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=2,
                                    norm="l2", dtype=np.float32)
        features = vectorizer.fit_transform(texts)
        del texts
        save("training_only_svd")
        svd = TruncatedSVD(n_components=100, random_state=DATA_SEED)
        vectors = normalize(svd.fit_transform(features)).astype(np.float32)
        del features
        joblib.dump({"vectorizer": vectorizer, "svd": svd},
                    args.output / "training-feature-transform.joblib", compress=3)
        selected = vectors[continuation]
        for k in [2, 4, 8]:
            save(f"cluster_k{k}")
            centers = KMeans(n_clusters=k, n_init=10, max_iter=100, random_state=DATA_SEED).fit(selected).cluster_centers_
            for _ in range(5):
                assignments = balanced_assignment(selected, centers, continuation)
                centers = np.stack([selected[assignments == group].mean(0) for group in range(k)])
            random_order = np.random.default_rng(np.random.SeedSequence([DATA_SEED, k, 1])).permutation(len(selected))
            random_groups = np.empty(len(selected), dtype=np.int16)
            for group, indices in enumerate(np.split(random_order, k)):
                random_groups[indices] = group
            random_centers = np.stack([selected[random_groups == group].mean(0) for group in range(k)])
            for method, groups, means in [("clustered", assignments, centers),
                                           ("random", random_groups, random_centers)]:
                name = f"{method}-k{k}"
                np.savez(args.output / f"{name}.npz", block_ids=continuation,
                         assignments=groups, centers=means)
                report["groups"][name] = {"counts": np.bincount(groups).tolist(),
                    "sha256": file_hash(args.output / f"{name}.npz"),
                    "within_group_squared_distance": float(
                        ((selected - means[groups]) ** 2).sum(1).mean())}
        save("transform_observed_evaluation_prefixes")
        for split in ["validation", "test"]:
            if file_hash(args.data / f"{split}.npy") != manifest["outputs"][f"{split}.npy"]["sha256"]:
                raise ValueError("Evaluation artifact hash mismatch")
            evaluation = np.load(args.data / f"{split}.npy")
            prefix = prefix_texts(tokenizer, evaluation, prefix_tokens=128)
            prefix_vectors = normalize(svd.transform(vectorizer.transform(prefix))).astype(np.float32)
            np.save(args.output / f"{split}-prefix-vectors.npy", prefix_vectors)
    report["status"] = "passed"
    report["feature_transform_sha256"] = file_hash(args.output / "training-feature-transform.joblib")
    report["evaluation_prefix_sha256"] = {
        split: file_hash(args.output / f"{split}-prefix-vectors.npy")
        for split in ("validation", "test")
    }
    save("complete")
    (args.output / "clustering-manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("A positive CPU time limit is required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "clustering-manifest.json").exists():
        parser.error("Use a new output directory; preserve the completed run")
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError("Clustering exceeded its recorded wall-time limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        cluster(args)
    except BaseException as error:
        status = args.output / "clustering-status.json"
        report = json.loads(status.read_text()) if status.exists() else {}
        report.update(status="failed", elapsed_seconds=time.monotonic() - started,
                      error={"type": type(error).__name__, "message": str(error)})
        status.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
