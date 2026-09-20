"""Run versioned validation-only Step 1, or independently verify its outputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.performance_diagnosis import run, verify


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-run")
    parser.add_argument("--execute-notebook", action="store_true")
    args = parser.parse_args()
    if args.execute_notebook:
        import asyncio
        import os
        import nbformat
        from nbclient import NotebookClient
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path = ROOT / "analysis/improvements/01_performance_diagnosis.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        notebook.cells.append(nbformat.v4.new_code_cell(
            "assert verification['validation_checkpoint_rows'] == 20\n"
            "assert verification['threshold_rows'] == 2020\n"
            "assert len(tables['learning_curve_summary']) == 10\n"
            "assert len(tables['class_weight_audit']) == 12\n"
            "print('OUTPUT_VERSION:', output_dir)\nprint(verification)"))
        NotebookClient(notebook, timeout=1200, kernel_name="bot_detection_env",
            resources={"metadata": {"path": str(path.parent)}},
            on_cell_execute=lambda **kw: print(f"Cell {kw['cell_index'] + 1}", flush=True)).execute()
        for out in notebook.cells[-1].outputs:
            if out.output_type == "stream":
                print(out.text)
        notebook.cells.pop()
        nbformat.write(notebook, path)
        print("PASS: fresh kernel; executed notebook saved")
    elif args.verify_run:
        print(verify(ROOT, ROOT / args.verify_run))
    else:
        output, tables = run(ROOT)
        print("OUTPUT_VERSION:", output)
        print(verify(ROOT, output))
        print(tables["checkpoint_threshold_summary"].to_string(index=False))
