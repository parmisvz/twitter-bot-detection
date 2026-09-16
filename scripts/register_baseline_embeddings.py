"""Explicitly register existing XLM-R vectors against Phase-2, without changing legacy files."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import read_json, register_legacy_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines_v2.json")
    args = parser.parse_args()
    print(register_legacy_embeddings(ROOT, read_json(ROOT / args.config)))
