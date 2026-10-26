"""Produce the exact LaTeX rows and CSV for manuscript Table errors."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def fmt(v):
    if v==0:return '0'
    s=f'{v:.5e}';m,e=s.split('e')
    return rf'${float(m):.5f}\times10^{{{int(e)}}}$'

with (ROOT/'data/metrics.csv').open() as f:
    rows=list(csv.DictReader(f))
lines=['\\begin{tabular}{@{}lrrrr@{}}','\\toprule',
    'Case & $t$ & $E_2$ & $E_\\infty$ & $E_{I_2}$\\\\','\\midrule']
for i,name in enumerate(('single','double','triple')):
    current=[r for r in rows if r['case']==name]
    if i:lines.append('\\addlinespace')
    for j,r in enumerate((current[0],current[-1])):
        label=name.capitalize() if j==0 else ''
        lines.append(f"{label} & {float(r['time']):g} & {fmt(float(r['E2']))} & {fmt(float(r['Einf']))} & {fmt(float(r['EI2']))}\\\\")
lines.extend(['\\bottomrule','\\end{tabular}'])
out=ROOT/'tables';out.mkdir(exist_ok=True)
(out/'errors_rows.tex').write_text('\n'.join(lines)+'\n')
print('Generated tables/errors_rows.tex')
