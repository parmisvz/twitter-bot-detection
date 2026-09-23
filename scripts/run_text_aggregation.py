"""Run or independently verify the text aggregation experiment."""
import argparse
import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text_aggregation import run, verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-notebook', action='store_true')
    parser.add_argument('--verify-run')
    parser.add_argument('--resume-run')
    args = parser.parse_args()
    if args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient

        class ProgressClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg.get('msg_type') == 'stream':
                    print(msg['content']['text'], end='', flush=True)
                return super().process_message(msg, cell, cell_index)

        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path = ROOT / 'analysis/improvements/08_cached_text_aggregation.ipynb'
        notebook = nbformat.read(path, as_version=4)
        if args.resume_run:
            output = (ROOT / args.resume_run).resolve()
            if not output.is_relative_to(ROOT / 'results/versions/text_aggregation_v1'):
                raise ValueError('Invalid resume directory')
            notebook.cells[1].source = notebook.cells[1].source.replace('output_dir = run(ROOT)', 'output_dir = ROOT / ' + repr(output.relative_to(ROOT).as_posix()))
        nbformat.validate(notebook)
        ProgressClient(notebook, timeout=7200, kernel_name='bot_detection_env', resources={'metadata': {'path': str(path.parent)}}).execute()
        nbformat.write(notebook, path)
        print('PASS: fresh-kernel text aggregation experiment')
    elif args.verify_run:
        print(verify(ROOT, ROOT / args.verify_run))
    else:
        output = run(ROOT)
        print('OUTPUT_VERSION:', output)
        print(verify(ROOT, output))
