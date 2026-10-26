"""Spectral time update obtained from simulator output, NOT classical update.

Continuous data: a real-k phase circuit. Discrete norming constants:
reference-amplitude encoding plus a controlled real contraction dilation.
The growth e^(8*kappa^3*t) is not unitary and is NEVER represented as a phase.
Classical gate-angle/coefficient calculation and state preparation are part
of the hybrid interface. Ideal readout of very small reference amplitudes
is not a finite-shot efficiency claim.
"""
import numpy as np
from .simulator import LocalStatevectorSimulator
from .hierarchy import STANDARD_KDV


def quantum_continuous_time(k,r0,t,audit=None,flow=STANDARD_KDV):
    k=np.asarray(k,float);r0=np.asarray(r0,complex)
    if k.ndim!=1 or r0.ndim!=1 or len(k)!=len(r0) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(r0)) or not np.isfinite(t):
        raise ValueError('k and reflection arrays must be equally long, finite 1D arrays.')
    out=[]
    for kk,rr in zip(np.asarray(k,float),np.asarray(r0,complex)):
        v=np.array([1.,rr],complex);v/=np.linalg.norm(v)
        sim=LocalStatevectorSimulator(v)
        sim.diagonal([1.,np.exp(2j*float(flow.omega(kk))*t)],'continuous_spectral_phase')
        rt=sim.state[1]/sim.state[0]
        if audit:audit.record('time_continuous',sim,{'k':float(kk),'t':float(t)})
        out.append(rt)
    return np.asarray(out)


def quantum_discrete_time(kappa,log_c0,t,audit=None,flow=STANDARD_KDV):
    kappa=np.asarray(kappa,float);log_c0=np.asarray(log_c0,float)
    if kappa.ndim!=1 or log_c0.ndim!=1 or len(kappa)!=len(log_c0) or not np.all(np.isfinite(kappa)) or not np.all(np.isfinite(log_c0)) or np.any(kappa<=0) or not np.isfinite(t):
        raise ValueError('kappa and log_c arrays must be equally long, finite and with positive kappa.')
    logs=[]
    for kap,lc in zip(np.asarray(kappa,float),np.asarray(log_c0,float)):
        offset=max(0.,float(lc))
        initial=np.array([np.exp(-offset),np.exp(lc-offset)],complex)
        initial/=np.linalg.norm(initial)
        growth=float(flow.norming_rate(kap))*t;shift=max(0.,float(growth))
        d=np.exp(np.array([-shift,growth-shift]))
        if d.min()==0:raise FloatingPointError('Time dilation underflow: reduce requested time or use chunked updates.')
        D=np.diag(d);S=np.diag(np.sqrt(np.maximum(0.,1-d*d)))
        # Keep small d entries explicitly. arccos/cos would round them away.
        dilation=np.block([[D,-S],[S,D]])
        sim=LocalStatevectorSimulator(np.r_[initial,0.,0.])
        sim.dense(dilation,'norming_growth_unitary_dilation')
        branch=sim.state[:2]
        if min(abs(branch))==0:raise FloatingPointError('Time reference amplitude unresolved.')
        result=float(np.log(abs(branch[1]))-np.log(abs(branch[0])))
        if audit:audit.record('time_discrete',sim,{'kappa':float(kap),'t':float(t),'growth_generator':float(flow.norming_rate(kap)),
                      'success_probability':float(np.vdot(branch,branch).real),
                      'reference_abs':float(abs(branch[0])),'classical_updated_c_used':False})
        logs.append(result)
    return np.asarray(logs)
