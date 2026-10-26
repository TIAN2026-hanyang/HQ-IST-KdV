#!/usr/bin/env python3
"""Run all three actual quantum stages. Default: 3 KdV cases, 4 snapshots each."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse,csv,json,hashlib,platform
from pathlib import Path
import numpy as np,scipy
from kdv_quantum.pipeline import run_case

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['single','double','triple','all'],default='all')
    ap.add_argument('--preset',choices=['smoke','snapshots','paper'],default='snapshots')
    ap.add_argument('--points',type=int,default=None)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'snapshots')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    names=['single','double','triple'] if args.case=='all' else [args.case]
    allrows=[]
    for name in names:
        arrays,rows,meta=run_case(name,args.preset,args.points)
        np.savez_compressed(args.out/f'{name}.npz',**arrays)
        (args.out/f'{name}_audit.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        allrows+=rows
    from summarize_results import summarize
    summarize(args.out)
if __name__=='__main__':main()
