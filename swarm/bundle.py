"""Build a private training-only input bundle; exclude evaluation text and notes.

Source attribution: allenai/c4, English, ODC-By dataset card; original document
provenance remains in the local preparation artifacts. AI assistance: Codex.
"""

import argparse
import json
from pathlib import Path
import shutil

from .data import REVISION, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh bundle directory")
    data = json.loads((args.data / "data-manifest.json").read_text())
    clusters = json.loads((args.clusters / "clustering-manifest.json").read_text())
    data_sha = file_hash(args.data / "data-manifest.json")
    if data["status"] != "passed" or clusters["status"] != "passed":
        raise ValueError("Preparation and clustering must both pass")
    if data["revision"] != REVISION or clusters["data_manifest_sha256"] != data_sha:
        raise ValueError("Source or data identity differs from the clustering run")
    sources = {}
    for name in ("train.npy", "warm-indices.npy", "continuation-indices.npy"):
        sources[name] = (args.data / name, data["outputs"][name]["sha256"])
    for method in ("clustered", "random"):
        for k in (2, 4, 8):
            name = f"{method}-k{k}"
            sources[name + ".npz"] = (args.clusters / (name + ".npz"), clusters["groups"][name]["sha256"])
    for source, expected in sources.values():
        if file_hash(source) != expected:
            raise ValueError("An input artifact changed after verification")
    args.output.mkdir(parents=True)
    manifest = {"status": "passed", "source": "allenai/c4", "revision": REVISION,
                "source_url": "https://huggingface.co/datasets/allenai/c4",
                "dataset_license": "ODC-By; preserve original document provenance",
                "data_manifest_sha256": data_sha,
                "clustering_manifest_sha256": file_hash(args.clusters / "clustering-manifest.json"),
                "tokenizer_sha256": data["outputs"]["tokenizer.json"]["sha256"],
                "contains_evaluation_text": False, "files": {}}
    for name, (source, expected) in sources.items():
        destination = args.output / name
        shutil.copyfile(source, destination)
        if file_hash(destination) != expected:
            raise OSError("Bundle copy failed integrity verification")
        manifest["files"][name] = expected
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(sources), "evaluation_text_included": False,
                      "training_manifest_sha256": file_hash(args.output / "training-manifest.json")}))


if __name__ == "__main__":
    main()
