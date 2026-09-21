"""Fresh-kernel runner and independent verifier for GCN representation integration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gcn_integration import run, verify

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

        class ProgressClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg.get("msg_type") == "stream":
                    print(msg["content"]["text"], end="", flush=True)
                return super().process_message(msg, cell, cell_index)
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path = ROOT / "analysis/improvements/04_gcn_representation_integration.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        ProgressClient(notebook, timeout=7200, kernel_name="bot_detection_env",
            resources={"metadata": {"path": str(path.parent)}},
            on_cell_execute=lambda **kw: print(f"Cell {kw['cell_index'] + 1}", flush=True)).execute()
        nbformat.write(notebook, path)
        print("PASS: fresh kernel; executed GCN integration notebook saved")
    elif args.verify_run:
        print(verify(ROOT, ROOT / args.verify_run))
    else:
        output, table, summary = run(ROOT)
        print("OUTPUT_VERSION:", output)
        print(verify(ROOT, output))
        print(summary.to_string(index=False))
