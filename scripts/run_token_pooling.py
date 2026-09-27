"""Fresh-kernel extraction/classifier experiment or independent verification."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-notebook',action='store_true')
    parser.add_argument('--extract-only',action='store_true',help='Standalone encoder process; use --reuse-embeddings for subsequent fresh-kernel execution')
    parser.add_argument('--verify-run')
    parser.add_argument('--reuse-embeddings',help='Completed embedding bundle only; verify before classifier execution')
    args=parser.parse_args()
    if args.extract_only:
        from src.xlmr_pooling import extract, verify as verify_embeddings
        embedding_dir=extract(ROOT)
        print('EMBEDDING_VERSION:',embedding_dir,flush=True)
        print(verify_embeddings(ROOT,embedding_dir),flush=True)
    elif args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        class ProgressClient(NotebookClient):
            def process_message(self,msg,cell,cell_index):
                if msg.get('msg_type')=='stream':print(msg['content']['text'],end='',flush=True)
                return super().process_message(msg,cell,cell_index)
        if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        path=ROOT/'analysis/improvements/09_controlled_token_pooling.ipynb';notebook=nbformat.read(path,as_version=4)
        if args.reuse_embeddings:
            ep=(ROOT/args.reuse_embeddings).resolve()
            if not ep.is_relative_to(ROOT/'final_data/embedding_versions/token_pooling_v1'):raise ValueError('Unexpected bundle')
            notebook.cells[1].source=notebook.cells[1].source.replace('embedding_dir = extract(ROOT)','embedding_dir = ROOT / '+repr(ep.relative_to(ROOT).as_posix()))
        nbformat.validate(notebook)
        ProgressClient(notebook,timeout=20000,kernel_name='bot_detection_env',resources={'metadata':{'path':str(path.parent)}}).execute()
        nbformat.write(notebook,path);print('PASS: fresh-kernel P0/P1 experiment')
    elif args.verify_run:
        import json
        from src.xlmr_pooling import verify as verify_embeddings
        from src.token_pooling import verify
        output=(ROOT/args.verify_run).resolve()
        config=json.loads((output/'provenance/resolved_config.json').read_text())
        print(verify_embeddings(ROOT,(ROOT/config['embedding_manifest']).parent))
        print(verify(ROOT,output))
    else:
        parser.error('Choose --extract-only, --execute-notebook or --verify-run')
