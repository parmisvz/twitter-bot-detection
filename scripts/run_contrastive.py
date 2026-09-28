"""Run approved contrastive experiment in a fresh kernel, or independently verify."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-notebook',action='store_true')
    parser.add_argument('--verify-run')
    parser.add_argument('--resume-run',help='Replay completed bundle without training')
    args=parser.parse_args()
    if args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        class ProgressClient(NotebookClient):
            def process_message(self,msg,cell,cell_index):
                if msg.get('msg_type')=='stream':print(msg['content']['text'],end='',flush=True)
                return super().process_message(msg,cell,cell_index)
        if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path=ROOT/'analysis/improvements/10_supervised_contrastive.ipynb'
        n=nbformat.read(path,as_version=4)
        if args.resume_run:
            output=(ROOT/args.resume_run).resolve()
            if not output.is_relative_to(ROOT/'results/versions/contrastive_v1'):raise ValueError('Unexpected run path')
            n.cells[1].source=n.cells[1].source.replace('output_dir = run(ROOT)','output_dir = ROOT / '+repr(output.relative_to(ROOT).as_posix()))
        nbformat.validate(n)
        ProgressClient(n,timeout=3600,kernel_name='bot_detection_env',resources={'metadata':{'path':str(path.parent)}}).execute()
        nbformat.write(n,path);print('PASS: fresh-kernel contrastive_v1')
    elif args.verify_run:
        from src.contrastive import verify
        output=(ROOT/args.verify_run).resolve()
        result=verify(ROOT,output)
        print(json.dumps(result,indent=2))
    else:parser.error('Choose --execute-notebook or --verify-run')
