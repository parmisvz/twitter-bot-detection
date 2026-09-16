"""Fresh-kernel notebook-03 execution and independent checkpoint verification."""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis/03_baseline_models.ipynb"


def protected_hashes():
    files = []
    for folder in ("data", "processed_data", "final_data", "models", "results", "analysis/archive"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and not p.is_relative_to(ROOT / "results/versions/baselines_v2"))
    files.extend(p for p in (ROOT / "analysis").glob("*.ipynb") if p != NOTEBOOK)
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
    parser.add_argument("--verify-run", help="Verify one existing run directory without training or starting Jupyter")
    args = parser.parse_args()
    if args.verify_run:
        sys.path.insert(0, str(ROOT))
        from src.baselines import verify_run
        print(verify_run(ROOT, ROOT / args.verify_run))
        return
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    before = protected_hashes()
    cwd = ROOT if args.from_root else ROOT / "analysis"
    print(f"Fresh kernel from {cwd}; protecting {len(before)} files (including Phase-2 outputs)", flush=True)
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert verification['reloaded_checkpoints'] == 6
assert verification['verified_prediction_files'] == 18
assert verification['verified_metric_rows'] == 36
assert run.models['tabular_mlp']['selected'] == 'last'
assert run.models['text_mlp']['selected'] == 'best_validation_loss'
assert run.data['feature_order'] == run.data['manifest']['feature_order']
assert all(len(p['users']) == len(p['x']) for p in run.data['populations'].values())
print('OUTPUT_VERSION:', run.output_dir)
print('VERIFICATION:', verification)
print(run.coverage.to_string(index=False))
print(metrics[(metrics.primary_checkpoint) & metrics.split.eq('test')].to_string(index=False))
"""))
    try:
        NotebookClient(notebook, timeout=1200, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(cwd)}},
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
