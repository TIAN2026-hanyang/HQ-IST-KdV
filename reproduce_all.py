"""Single entry point: regenerate paper data, tables, panels and audit.

Run from any working directory: python reproduce_all.py
"""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parent
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')

def run(*args):
    print('EXEC:',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,env=ENV,check=True)

if __name__=='__main__':
    run('run_experiments.py','--case','all','--preset','paper','--out','data')
    run('scripts/export_csv.py')
    run('scripts/make_figures.py')
    run('scripts/make_table.py')
    run('scripts/verify_paper.py','--figures')
    run('-m','pytest','-q')
    print('COMPLETED: source -> 156 field arrays -> 21 manuscript panels -> 3 diagnostics -> acceptance tests')
