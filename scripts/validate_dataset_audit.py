"""Execute notebook 01 in a fresh kernel and verify its read-only handoff.

Run with the project environment. --write-executed saves verified notebook outputs.
No preprocessing or model notebooks are executed.
"""

import argparse
import hashlib
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis/01_dataset_exploration_and_audit.ipynb"


def fingerprints():
    files = []
    for directory in ("data", "processed_data", "final_data", "models", "results"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    files.extend(p for p in (ROOT / "analysis").rglob("*.ipynb") if p != NOTEBOOK)
    result = {}
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.relative_to(ROOT).as_posix()] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-executed", action="store_true")
    parser.add_argument("--from-analysis", action="store_true")
    parser.add_argument("--kernel", default="bot_detection_env")
    args = parser.parse_args()
    # Rendering must work without a GUI or display server.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    before = fingerprints()
    check = nbformat.v4.new_code_cell("""
assert all(audit_checks.values()), audit_checks
assert len(TEMPORAL_COLUMNS) == 7
assert set(TEMPORAL_COLUMNS).isdisjoint(BEHAVIOR_COLUMNS)
for cohort, table in cohorts.items():
    assert table.index.is_unique, cohort
    assert not table.isna().any().any(), cohort
    assert table.groupby(pattern_columns, dropna=False).size().sum() == len(table)
assert coverage_report['users'].le(coverage_report['denominator']).all()
for row in population_compatibility:
    if row['status'] == 'compared':
        assert row['expected_only'] == row['artifact_only'] == row['duplicate_key_rows'] == 0, row
for row in compatibility_rows:
    if row['exists']:
        assert row['required_columns_present'], row
assert audit_key(pd.Series([' @Example ', None, '', 'EXAMPLE'])).tolist()[0] == 'example'
assert inspect_neighbors('[]') == ([], 'empty', 0)
assert inspect_neighbors('[broken')[1] == 'malformed'
assert inspect_neighbors('Alice, @BOB')[0] == ['alice', 'bob']
print('PASS: identity, populations, coverage denominators, graph inspection, and handoff schemas.')
print(population_reconciliation.to_string(index=False))
""")
    notebook.cells.append(check)
    cwd = ROOT / "analysis" if args.from_analysis else ROOT
    print(f"Executing from {cwd}; protecting {len(before)} files.", flush=True)
    try:
        NotebookClient(notebook, timeout=300, kernel_name=args.kernel,
                       on_cell_executed=lambda **kw: print(f"Executed cell {kw['cell_index'] + 1}", flush=True),
                       resources={"metadata": {"path": str(cwd)}}).execute()
    finally:
        after = fingerprints()
        if before != after:
            changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
            raise AssertionError(f"Audit changed protected data/artifacts: {changed}")
    for output in notebook.cells[-1].outputs:
        if output.output_type == "stream":
            print(output.text)
    notebook.cells.pop()
    nbformat.validate(notebook)
    figure_count = sum("image/png" in output.get("data", {})
                       for cell in notebook.cells for output in cell.get("outputs", []))
    if figure_count < 6:
        raise AssertionError(f"Expected at least six inline audit figures, found {figure_count}")
    if args.write_executed:
        nbformat.write(notebook, NOTEBOOK)
    print(f"PASS: {len(notebook.cells)} cells and {figure_count} figures executed from {cwd}; "
          f"{len(before)} protected files unchanged.")


if __name__ == "__main__":
    main()
