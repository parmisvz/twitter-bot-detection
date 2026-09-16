"""Fresh-kernel notebook-02 validation while protecting every legacy artifact."""
import argparse
import asyncio
import hashlib
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis/02_data_preprocessing_and_feature_engineering.ipynb"


def protected_hashes():
    result = {}
    paths = []
    for directory in ["data", "processed_data", "final_data", "models", "results", "analysis/archive"]:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not p.is_relative_to(ROOT / "final_data/versions"))
    paths.extend(p for p in (ROOT / "analysis").glob("*.ipynb") if p != NOTEBOOK)
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        result[str(path.relative_to(ROOT))] = h.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-executed", action="store_true")
    parser.add_argument("--from-root", action="store_true")
    parser.add_argument("--kernel", default="bot_detection_env")
    args = parser.parse_args()
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    before = protected_hashes()
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert run.reports['validation']['passed'].all()
assert run.output_dir.is_relative_to(PROJECT_ROOT / 'final_data/versions')
assert set(run.preprocessor['training_user_keys']) == set(run.assignments.index[run.assignments.split.eq('train')])
assert run.users.index.is_unique
assert verification['reloaded_transform_populations'] == 4
print('OUTPUT_VERSION:', OUTPUT_DIR)
print(run.reports['population'].to_string(index=False))
print(run.reports['temporal_source_coverage'].to_string(index=False))
print(run.reports['temporal_by_split_label'].to_string(index=False))
print('VALIDATION_CHECKS:', len(run.reports['validation']))
print('VERIFIED_ARTIFACTS:', verification['verified_artifacts'])
"""))
    cwd = ROOT if args.from_root else ROOT / "analysis"
    print(f"Fresh kernel from {cwd}; protecting {len(before)} legacy files", flush=True)
    try:
        NotebookClient(notebook, timeout=600, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(cwd)}},
                       on_cell_executed=lambda **kw: print(f"Executed cell {kw['cell_index'] + 1}", flush=True)).execute()
    finally:
        after = protected_hashes()
        if before != after:
            changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
            raise AssertionError(f"Legacy artifacts changed: {changed}")
    for output in notebook.cells[-1].outputs:
        if output.output_type == "stream":
            print(output.text)
    notebook.cells.pop()
    if args.write_executed:
        nbformat.write(notebook, NOTEBOOK)
    print(f"PASS: {len(notebook.cells)} cells; all {len(before)} legacy files unchanged")


if __name__ == "__main__":
    main()
