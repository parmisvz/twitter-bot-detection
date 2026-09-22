"""Fresh-kernel scalar graph-scaling experiment and independent checkpoint verification."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph_scaling import run, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-notebook", action="store_true")
    parser.add_argument("--verify-run")
    parser.add_argument("--resume-run", help="Verify and display a completed run in a fresh notebook kernel, without retraining")
    args = parser.parse_args()
    if args.execute_notebook:
        import asyncio
        import os
        import nbformat
        from nbclient import NotebookClient

        class ProgressClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg.get("msg_type") == "stream":
                    print(msg["content"]["text"], end="", flush=True)
                return super().process_message(msg, cell, cell_index)

        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path = ROOT / "analysis/improvements/06_graph_input_scaling.ipynb"
        notebook = nbformat.read(path, as_version=4)
        if args.resume_run:
            output = (ROOT / args.resume_run).resolve()
            if not output.is_relative_to(ROOT / "results/versions/graph_scaling_v1"):
                raise ValueError("Resume directory must be a graph_scaling_v1 run")
            notebook.cells[1].source = notebook.cells[1].source.replace(
                "output_dir, metrics_table, seed_summary = run(ROOT)",
                "import pandas as pd\noutput_dir = ROOT / " + repr(output.relative_to(ROOT).as_posix())
                + "\nmetrics_table = pd.read_csv(output_dir / 'validation_metrics.csv')"
                + "\nseed_summary = pd.read_csv(output_dir / 'seed_summary.csv')")
        nbformat.validate(notebook)
        ProgressClient(notebook, timeout=7200, kernel_name="bot_detection_env",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        nbformat.write(notebook, path)
        print("PASS: fresh-kernel graph scaling experiment")
    elif args.verify_run:
        print(verify(ROOT, ROOT / args.verify_run))
    else:
        output, _, _ = run(ROOT)
        print("OUTPUT_VERSION:", output)
        print(verify(ROOT, output))
