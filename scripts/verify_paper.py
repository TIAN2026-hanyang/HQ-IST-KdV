"""Independent acceptance checks for the published single/double/triple benchmarks.

Reads only supplied numerical arrays and audits; never alters/re-solves them.
"""
from pathlib import Path
import sys,csv,json,hashlib,platform
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kdv_quantum.exact import CASES,exact_field
from kdv_quantum.metrics import three_metrics

EXPECTED_N={'single':51,'double':53,'triple':52}
EXPECTED_ENDPOINT={
 'single':{'E2':6.83229e-6,'Einf':1.22808e-5,'EI2':2.40015e-6},
 'double':{'E2':4.10163e-6,'Einf':7.20926e-6,'EI2':2.19999e-6},
 'triple':{'E2':4.13713e-5,'Einf':4.63254e-4,'EI2':1.84015e-5},
}


def verify(folder=ROOT/'data',require_figures=False):
    folder=Path(folder)
    items={};all_rows=[]
    stage_names=('direct_schrodingerisation','time_continuous','time_discrete','glm_qsvt')
    for name,case in CASES.items():
        p=folder/f'{name}.npz';meta_path=folder/f'{name}_audit.json'
        if not p.exists() or not meta_path.exists():raise FileNotFoundError(f'Missing real full-data output: {name}')
        with np.load(p,allow_pickle=False) as z:
            keys={'x','times','u_hybrid','u_exact','k','r0','r_t','kappa','log_c0','log_c_t'}
            if not keys.issubset(z.files):raise AssertionError((name,'missing archive keys',keys-set(z.files)))
            v={k:z[k].copy() for k in keys}
        m=json.loads(meta_path.read_text())
        x=v['x'];t=v['times'];uh=v['u_hybrid'];ue=v['u_exact']
        if len(t)!=EXPECTED_N[name] or len(x)!=case.points or uh.shape!=(len(t),len(x)) or ue.shape!=uh.shape:
            raise AssertionError((name,'data dimensions',len(t),len(x),uh.shape))
        if len(v['k'])!=4 or v['r_t'].shape!=(len(t),4) or v['log_c_t'].shape!=(len(t),len(case.kappa)):
            raise AssertionError((name,'spectral data dimensions'))
        if not np.all(np.isfinite(uh)) or not np.all(np.isfinite(ue)) or np.min(ue)<-1e-12:
            raise AssertionError((name,'invalid positive-polarity u'))
        if not np.isclose(x.min(),case.x_min) or not np.isclose(x.max(),case.x_max):
            raise AssertionError((name,'incorrect output interval'))
        if not np.all(np.diff(t)>0):raise AssertionError('times must increase')
        for tt in case.times:
            if not np.any(np.isclose(t,tt,atol=1e-12,rtol=0)):
                raise AssertionError((name,'missing profile time',tt))
        if m['model']!='u_t + 6*u*u_x + u_xxx = 0' or m['hierarchy_flow_coefficients']!=[0.,-1.]:
            raise AssertionError((name,'physical u or hierarchy metadata mismatch'))
        audit=m['audit']
        if audit['backend']!='LocalStatevectorSimulator':raise AssertionError((name,'backend missing'))
        for stage in stage_names:
            if audit['stages'].get(stage,{}).get('circuit_executions',0)<=0:
                raise AssertionError((name,'missing quantum stage',stage))
        if m.get('analytic_kappa_or_c_injected') or m.get('classical_glm_solve_used') or m.get('reference_used_in_solver'):
            raise AssertionError((name,'unallowed substitution of analytic or classical solve'))
        if not m.get('known_reflectionless_constraint_used'):raise AssertionError('reflectionless scope not declared')
        for i,ti in enumerate(t):
            predicted=exact_field(x,float(ti),case)
            if not np.allclose(predicted,ue[i],rtol=0,atol=2e-13):
                raise AssertionError((name,'analytic field mismatch',i))
            metrics=three_metrics(x,uh[i],ue[i])
            all_rows.append({'case':name,'time':float(ti),**metrics})
            if metrics['E2']>1e-3 or metrics['Einf']>2e-3:
                raise AssertionError((name,'excess field error',i,metrics))
        finish=three_metrics(x,uh[-1],ue[-1]);expected=EXPECTED_ENDPOINT[name]
        for key,value in expected.items():
            # Manuscript table values have been rounded to six significant figures.
            if not np.isclose(finish[key],value,rtol=1e-4,atol=1e-9):
                raise AssertionError((name,'does not match manuscript endpoint',key,finish[key],value))
        if not np.isclose(v['kappa'],case.kappa,rtol=0,atol=2e-6).all():
            raise AssertionError((name,'wrong soliton eigenvalues'))
        items[name]={'times':len(t),'positions':len(x),
                     'u_min':float(np.min(uh)),'u_max':float(np.max(uh)),
                     'kappa_numerical':v['kappa'].tolist(),
                     'c0_numerical':np.exp(v['log_c0']).tolist(),
                     'matching_points':[n['matching_x'] for n in m['norming_diagnostics']],
                     'max_abs_reflection_diagnostic':float(np.max(np.abs(v['r0']))),
                     'final_error':finish,
                     'max_over_times':{k:float(max(row[k] for row in all_rows if row['case']==name)) for k in ('E2','Einf','EI2')},
                     'quantum_counts':{k:audit['stages'][k]['circuit_executions'] for k in stage_names},
                     'max_qsvt_degree':int(max(d['max_degree'] for d in m['inverse_records']))}
        print(f"{name}: {len(t)} x {len(x)}, E2={finish['E2']:.9e}, max_degree={items[name]['max_qsvt_degree']}")
    if len(all_rows)!=156:raise AssertionError(('Expected 156 full fields',len(all_rows)))
    if require_figures:
        for name in CASES:
            for j in range(4):
                for ext in ('pdf','png'):
                    if not (ROOT/'figures'/f'{name}_profile_{j}.{ext}').exists():raise FileNotFoundError('Incomplete figures')
            for tag in ('exact','hybrid','error'):
                for ext in ('pdf','png'):
                    if not (ROOT/'figures'/f'{name}_spacetime_{tag}.{ext}').exists():raise FileNotFoundError('Incomplete figures')
    report={'status':'PASS','case_summary':items,'full_fields':len(all_rows),
            'backend':'bundled ideal-statevector (not Qiskit Aer/hardware)',
            'paper_compatibility':'standard KdV r(z)=-z, reflectionless N=1,2,3',
            'measured_metric_names':['E2','Einf','EI2'],
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'file_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in list((ROOT/'kdv_quantum').glob('*.py'))+list(folder.glob('*.npz'))}}
    path=ROOT/'MANUSCRIPT_DATA_VERIFICATION.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PASS: {len(all_rows)} independently reconstructed full fields; report: {path}')
    return report

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--figures',action='store_true');args=ap.parse_args()
    verify(require_figures=args.figures)
