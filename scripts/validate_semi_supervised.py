"""Fresh-kernel Notebook-08 execution and independent saved-run verification."""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis/08_semi_supervised_learning.ipynb"


def protected_hashes():
    files = []
    for folder in ("data", "processed_data", "final_data", "models", "results", "analysis/archive"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and not p.is_relative_to(ROOT / "results/versions/semi_supervised_v2"))
    files.extend(p for p in (ROOT / "analysis").glob("*.ipynb") if p != NOTEBOOK)
    files.extend(ROOT / p for p in ("src/preprocessing.py", "src/baselines.py", "src/graph_baseline.py",
        "src/graph_tabular.py", "src/multimodal.py", "src/ablation.py", "configs/preprocessing_v2.json",
        "configs/baselines_v2.json", "configs/multimodal_v2.json", "configs/ablation_v2.json",
        "configs/frozen_user_split_v1.csv"))
    result = {}
    for path in files:
        with path.open("rb") as stream:
            result[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-executed", action="store_true")
    parser.add_argument("--from-root", action="store_true")
    parser.add_argument("--kernel", default="bot_detection_env")
    parser.add_argument("--verify-run")
    args = parser.parse_args()
    if args.verify_run:
        sys.path.insert(0, str(ROOT))
        from src.semi_supervised import verify_run
        print(verify_run(ROOT, ROOT / args.verify_run))
        return
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    before = protected_hashes()
    cwd = ROOT if args.from_root else ROOT / "analysis"
    print(f"Fresh kernel from {cwd}; protecting {len(before)} prior files", flush=True)
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert verification['full_pseudo_pool'] == 18331
assert verification['selected_pseudo'] == 390
assert verification['selected_distribution'] == {0: 195, 1: 195}
assert verification['reloaded_checkpoints'] == 4
assert verification['verified_prediction_files'] == 9
assert verification['verified_metric_rows'] == 9
assert len(run.pool[run.pool.accepted_by_threshold & run.pool.pseudo_label.eq(0)]) == 13782
assert len(run.pool[run.pool.accepted_by_threshold & run.pool.pseudo_label.eq(1)]) == 195
assert all(len(history) == 50 for history in run.histories.values())
print('OUTPUT_VERSION:', run.output_dir)
print('VERIFICATION:', verification)
print(__import__('pandas').read_csv(run.output_dir / 'reports' / 'pseudo_label_distribution.csv').to_string(index=False))
print(__import__('pandas').read_csv(run.output_dir / 'reports' / 'training_populations.csv').to_string(index=False))
print(metrics[metrics.split.eq('test')].to_string(index=False))
"""))
    try:
        NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(cwd)}},
                       on_cell_execute=lambda **kw: print(f"Starting cell {kw['cell_index'] + 1}", flush=True),
                       on_cell_executed=lambda **kw: print(f"Executed cell {kw['cell_index'] + 1}", flush=True)).execute()
    finally:
        after = protected_hashes()
        if before != after:
            changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
            raise AssertionError(f"Protected artifacts changed: {changed}")
    for output in notebook.cells[-1].outputs:
        if output.output_type == "stream":
            print(output.text)
    notebook.cells.pop()
    if args.write_executed:
        nbformat.write(notebook, NOTEBOOK)
    print(f"PASS: {len(notebook.cells)} cells; all {len(before)} protected files unchanged")


if __name__ == "__main__":
    main()
