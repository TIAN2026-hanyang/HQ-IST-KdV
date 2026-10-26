"""Export full u-fields and all scattering data from NPZ without recomputation."""
from pathlib import Path
import sys,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def save_csv(name,x,t,A):
    p=ROOT/'data'/name
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['time']+[f'x={j:.12g}' for j in x])
        for ti,row in zip(t,A):w.writerow([format(float(ti),'.17g')]+[format(float(v),'.17g') for v in row])

for name in ('single','double','triple'):
    with np.load(ROOT/'data'/f'{name}.npz') as d:
        x=d['x'];t=d['times'];hyb=d['u_hybrid'];ex=d['u_exact']
        kk=d['k'];r=d['r_t'];kappa=d['kappa'];c=d['log_c_t']
    save_csv(f'{name}_u_hybrid.csv',x,t,hyb)
    save_csv(f'{name}_u_exact.csv',x,t,ex)
    with (ROOT/'data'/f'{name}_reflection_diagnostic.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time','k','reflection_real','reflection_imag'])
        for ti,row in zip(t,r):
            for ki,ri in zip(kk,row):w.writerow([f'{ti:.17g}',f'{ki:.17g}',f'{ri.real:.17g}',f'{ri.imag:.17g}'])
    with (ROOT/'data'/f'{name}_bound_spectrum.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time','kappa','log_c'])
        for ti,row in zip(t,c):
            for ki,ci in zip(kappa,row):w.writerow([f'{ti:.17g}',f'{ki:.17g}',f'{ci:.17g}'])
    print(f'Exported all space-time and spectral data to CSV for {name}')
