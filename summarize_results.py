"""Merge independently completed cases without overwriting other case metrics."""
import argparse,csv,hashlib,json,platform
from pathlib import Path
import numpy as np,scipy
from kdv_quantum.metrics import three_metrics

def summarize(folder):
    folder=Path(folder);root=Path(__file__).resolve().parent;rows=[];audits=[];cases=[]
    for name in ('single','double','triple'):
        npz=folder/f'{name}.npz';auditfile=folder/f'{name}_audit.json'
        if not npz.exists() or not auditfile.exists():continue
        z=np.load(npz);a=json.loads(auditfile.read_text());cases.append(name)
        for i,t in enumerate(z['times']):
            rows.append({'case':name,'time':float(t),**three_metrics(z['x'],z['u_hybrid'][i],z['u_exact'][i])})
        for stage,details in a['audit']['stages'].items():
            audits.append({'case':name,'stage':stage,'circuit_executions':details['circuit_executions'],
                           'max_qubits':details['max_qubits'],'max_norm_error':details['max_norm_error']})
    if not rows:raise RuntimeError('No completed results found.')
    with (folder/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=['case','time','E2','Einf','EI2']);wr.writeheader();wr.writerows(rows)
    with (folder/'quantum_stage_audit.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=list(audits[0]));wr.writeheader();wr.writerows(audits)
    manifest={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'backend':'LocalStatevectorSimulator','cases':cases,
              'execution_scope':'completed ideal-statevector circuits; not a hardware run',
              'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'kdv_quantum').glob('*.py'))},
              'result_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.npz'))}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,audits
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent/'results'/'snapshots')
    args=p.parse_args();summarize(args.folder)
