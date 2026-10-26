"""Audited three-quantum-stage pipeline for exact reflectionless KdV cases."""
from __future__ import annotations
import time
import numpy as np
from .exact import CASES,exact_field
from .direct import DirectSettings,classical_bound_spectrum,quantum_norming,quantum_reflection
from .time_evolution import quantum_continuous_time,quantum_discrete_time
from .glm import reconstruct_field
from .metrics import three_metrics
from .simulator import SimulationAudit
from .hierarchy import STANDARD_KDV


def run_case(name,preset='snapshots',points=None,backend='local',progress=print):
    if name not in CASES:raise ValueError('Only single, double, triple KdV cases are supported.')
    if backend!='local':raise ValueError('This tested distribution uses its bundled LocalStatevectorSimulator; no silent backend substitution.')
    case=CASES[name];audit=SimulationAudit();start=time.time()
    if preset=='smoke':
        cfg=DirectSettings(step=.01,block=.08,n_p=256);nx=41
        times=np.array([0.,case.times[-1]]);realk=np.array([1.,1.5])
    elif preset in ('snapshots','paper'):
        cfg=DirectSettings(step=.005,block=.04,n_p=256);nx=case.points
        times=np.array(case.times) if preset=='snapshots' else np.unique(np.r_[np.linspace(0,case.times[-1],51),case.times])
        realk=np.array([.75,1.,1.5,2.])
    else:raise ValueError('preset must be smoke, snapshots, or paper.')
    if points is not None:nx=int(points)
    if nx<21:raise ValueError('At least 21 output points required.')
    x=np.linspace(case.x_min,case.x_max,nx)
    # exact_field is used here only as the t=0 input. Future-time exact fields
    # enter exclusively in the comparison section after the algorithm output.
    u0=lambda xx:exact_field(xx,0.,case)
    kappa,matches,disc=classical_bound_spectrum(u0,case.spectral_left,case.spectral_right,n=512)
    progress(f'{name}: classical eigenvalues obtained',flush=True)
    if len(kappa)!=len(case.kappa):raise RuntimeError('Unexpected number of resolved bound states.')
    logc0,norming_diagnostics=quantum_norming(u0,kappa,matches,case.spectral_left,case.spectral_right,cfg,audit)
    progress(f'{name}: quantum norming constants obtained',flush=True)
    r=[];direct_records=[]
    for kk in realk:
        rr,record=quantum_reflection(u0,kk,case.spectral_left,case.spectral_right,cfg,audit)
        r.append(rr);direct_records.append(record)
        progress(f'{name}: real-k direct circuit k={kk:g} completed',flush=True)
    r=np.array(r)
    # This program declares a reflectionless analytic-family benchmark. It
    # never silently promotes arbitrary input data into a reflectionless case.
    # Measured numerical leakage is retained in files and quantum time-evolved;
    # the exact finite-rank GLM uses the known zero-continuous-spectrum model.
    if np.max(np.abs(r))>5e-3:raise RuntimeError('Reflectionless-family consistency check failed: refine the direct circuit.')
    progress(f'{name}: direct quantum propagation completed; |r|_max={max(abs(r)):.3e}; elapsed={time.time()-start:.2f}s',flush=True)
    fields=[];refs=[];rows=[];r_times=[];logc_times=[];inverse_records=[]
    for tt in times:
        r_t=quantum_continuous_time(realk,r,float(tt),audit,flow=STANDARD_KDV)
        logc_t=quantum_discrete_time(kappa,logc0,float(tt),audit,flow=STANDARD_KDV)
        progress(f'{name}: inverse start t={tt:g}; kappa={kappa}; logc={logc_t}; elapsed={time.time()-start:.2f}s',flush=True)
        u,info=reconstruct_field(x,kappa,logc_t,audit,derivative_step=.003,tol=1e-8)
        # This exact evaluation is not passed back into the numerical solver.
        ref=exact_field(x,float(tt),case)
        metric=three_metrics(x,u,ref);rows.append({'case':name,'time':float(tt),**metric})
        fields.append(u);refs.append(ref);r_times.append(r_t);logc_times.append(logc_t);inverse_records.append(info)
        progress(f'{name}: t={tt:g}, E2={metric["E2"]:.3e}, Einf={metric["Einf"]:.3e}, EI2={metric["EI2"]:.3e}; elapsed={time.time()-start:.2f}s',flush=True)
    required=('direct_schrodingerisation','time_continuous','time_discrete','glm_qsvt')
    if any(audit.stages.get(key,{}).get('circuit_executions',0)==0 for key in required):
        raise RuntimeError('Quantum-stage audit failed; no partial/classical substitution is permitted.')
    meta={'case':name,'model':'u_t + 6*u*u_x + u_xxx = 0','hierarchy_flow_coefficients':[0.0,-1.0],'preset':preset,
          'scope':'exact_reflectionless_N_soliton_family',
          'direct_mode':'blockwise_Schrodingerisation_readout_and_reprepare',
          'discrete_eigenvalues':'classical sine DVR',
          'norming_constants':'quantum Jost propagation and classical quadrature',
          'time_continuous':'simulator cubic phase gate',
          'time_discrete':'simulator nonunitary growth block encoding with reference amplitude',
          'glm_method':'finite_rank_GLM + diagonal_equilibration + uniform_interval_QSVT',
          'continuous_numerical_leakage':'stored and phase-evolved; not interpreted as physical radiation in the declared reflectionless model',
          'analytic_kappa_or_c_injected':False,'known_reflectionless_constraint_used':True,'classical_glm_solve_used':False,
          'reference_used_in_solver':False,'external_quantum_hardware_used':False,
          'direct_settings':vars(cfg),'spectrum_settings':disc,
          'norming_diagnostics':norming_diagnostics,'direct_records':direct_records,
          'inverse_records':inverse_records,'audit':audit.to_dict(),'elapsed_seconds':time.time()-start}
    arrays={'x':x,'times':times,'u_hybrid':np.asarray(fields),'u_exact':np.asarray(refs),
            'k':realk,'r0':r,'r_t':np.asarray(r_times),'kappa':kappa,'log_c0':logc0,
            'log_c_t':np.asarray(logc_times)}
    return arrays,rows,meta
