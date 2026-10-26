"""Quantum Jost propagation using blockwise Schrödingerisation.

A=H1+i H2; H_eff=eta*(H1-gamma I)-H2. Each spatial block is
simulated with QFT, controlled Strang rotations, inverse QFT and explicit
auxiliary postselection. Readout and state re-preparation between blocks make
this a HYBRID variant, not one coherent all-space circuit. No classical Jost
ODE propagation is used. Discrete eigenvalues alone are classical; their
norming constants are determined from quantum-propagated Jost functions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simpson
from .simulator import LocalStatevectorSimulator

@dataclass
class DirectSettings:
    step:float=.01
    block:float=.08
    n_p:int=256
    p_half_width:float=10.
    p_read:float=1.

@dataclass
class JostPath:
    x:np.ndarray
    states:np.ndarray
    log_norm:np.ndarray
    min_probability:float
    max_branch_mismatch:float=0.


def classical_bound_spectrum(u0,left,right,n=512,threshold=1e-7):
    """Sine-DVR self-adjoint spectral calculation; never a GLM solve."""
    j=np.arange(1,n+1,dtype=float)
    x=left+(right-left)*j/(n+1)
    S=np.sqrt(2/(n+1))*np.sin(np.pi*j[:,None]*j[None,:]/(n+1))
    energy=(np.pi*j/(right-left))**2
    H=(S*energy)@S.T
    u=np.asarray(u0(x),float);H.flat[::n+1]-=u
    e,V=eigh(H,subset_by_value=(float(-u.max()-1.),-threshold),check_finite=False)
    if not len(e):raise ValueError('No resolved bound states for a soliton benchmark.')
    kappa=np.sqrt(-e)
    # Deterministic rightmost maximum when an eigenstate has symmetric peaks.
    # This removes machine-dependent ±x selection for the shallow triple state.
    aa=np.abs(V)
    matches=[]
    for column in range(aa.shape[1]):
        candidates=np.flatnonzero(aa[:,column]>=aa[:,column].max()*(1-1e-8))
        matches.append(float(x[candidates[-1]]))
    matches=np.asarray(matches)
    return kappa,matches,{'classical_eigenvalue_method':'sine_DVR','nodes':n,'eigenvalues':e.tolist()}


def quantum_jost_path(u0,k,start,stop,initial,log_norm0=0.,settings=None,audit=None,representation='raw'):
    cfg=settings or DirectSettings()
    if start==stop:raise ValueError('Nonempty propagation interval required.')
    if cfg.n_p<32 or cfg.n_p&(cfg.n_p-1):raise ValueError('n_p must be a power of two >=32.')
    if cfg.step<=0 or cfg.block<=0:raise ValueError('Positive spatial steps required.')
    direction=np.sign(stop-start)
    nblocks=max(1,int(np.ceil(abs(stop-start)/cfg.block)))
    nodes=np.linspace(start,stop,nblocks+1)
    steps=max(1,int(np.ceil(abs(nodes[1]-nodes[0])/cfg.step)))
    h=abs(nodes[1]-nodes[0])/steps
    mids=nodes[:-1,None]+direction*h*(np.arange(steps)+.5)[None,:]
    uvals=np.asarray(u0(mids.reshape(-1)),float).reshape(nblocks,steps)
    if not np.all(np.isfinite(uvals)):raise ValueError("Nonfinite physical field u0.")
    p=np.linspace(-cfg.p_half_width,cfg.p_half_width,cfg.n_p,endpoint=False)
    eta=2*np.pi*np.fft.fftfreq(cfg.n_p,d=p[1]-p[0])
    # A C^7 continuation, exactly e^-p on the positive half-axis.
    env=np.exp(-p-np.minimum(p,0.)**8);envnorm=float(np.linalg.norm(env));env/=envnorm
    ridx=int(np.argmin(np.abs(p-cfg.p_read)))
    if p[ridx]<=0:raise ValueError('Recovery point must be positive.')
    target=np.asarray(initial,complex)
    norm=float(np.linalg.norm(target));target/=norm
    lognorm=float(log_norm0+np.log(norm))
    states=[target.copy()];logs=[lognorm];minprob=1.
    for row in uvals:
        sim=LocalStatevectorSimulator(np.kron(target,env));sim.auxiliary_qft()
        gamma_int=0.
        for u in row:
            if representation=='plane':
                if abs(complex(k).imag)>1e-15 or float(np.real(k))<=0:raise ValueError('Plane representation requires positive real k.')
                kk=float(np.real(k));h1=-direction*u/(2*kk);h2=direction*(kk+u/(2*kk));axis1='y';axis2='z'
            elif representation=='bound':
                kap=float(np.imag(k))
                if kap<=0 or abs(np.real(k))>1e-14:raise ValueError('Bound representation requires k=i*kappa.')
                coupling=-u/(2*kap)
                h1=-direction*(kap+coupling);h2=-direction*coupling;axis1='z';axis2='y'
            elif representation=='raw':
                kk2=complex(k)**2
                if abs(kk2.imag)>1e-12:raise ValueError('Raw benchmark supports real or purely imaginary k.')
                h1=direction*(1-u-kk2.real)/2;h2=direction*(1+u+kk2.real)/2;axis1='x';axis2='y'
            else:raise ValueError('Unknown representation.')
            gamma=abs(h1);gamma_int+=gamma*h
            phase=np.exp(.5j*eta*gamma*h)
            sim.auxiliary_phase(phase)
            sim.multiplexed_rotation(axis1,eta*h1*h)
            sim.multiplexed_rotation(axis2,-2*h2*h,'target_R'+axis2)
            sim.multiplexed_rotation(axis1,eta*h1*h)
            sim.auxiliary_phase(phase)
        sim.auxiliary_qft(inverse=True)
        branch=sim.state.reshape(2,-1)[:,ridx].copy()
        branch_norm=float(np.linalg.norm(branch));prob=branch_norm**2
        if not np.isfinite(prob) or prob<1e-18:raise FloatingPointError('Unresolved postselection; reduce spatial block or refine auxiliary register.')
        if audit:audit.record('direct_schrodingerisation',sim,{'k_real':float(np.real(k)),'k_imag':float(np.imag(k)),
                  'representation':representation,'block_success_probability':prob,'spatial_steps':steps,
                  'gamma_integral':gamma_int,'recovery_p':float(p[ridx])})
        target=branch/branch_norm
        lognorm+=np.log(branch_norm)+np.log(envnorm)+p[ridx]+gamma_int
        minprob=min(minprob,prob);states.append(target.copy());logs.append(lognorm)
    return JostPath(nodes,np.asarray(states),np.asarray(logs),minprob)


def _log_square_integral(path,kappa,side_endpoint):
    log_values=2*path.log_norm+2*np.log(np.maximum(abs(np.sum(path.states,axis=1)),1e-300))
    offset=float(np.max(log_values))
    s=abs(path.x-path.x[0]);integral=float(simpson(np.exp(log_values-offset),x=s))
    logcore=np.log(max(integral,1e-300))+offset
    logtail=log_values[0]-np.log(2*kappa)
    return float(np.logaddexp(logcore,logtail))


def quantum_norming(u0,kappa,matches,left,right,settings=None,audit=None):
    logs=[];diagnostics=[]
    for kap,match in zip(kappa,matches):
        pl=quantum_jost_path(u0,1j*kap,left,float(match),[0.,1.],kap*left,settings,audit,'bound')
        pr=quantum_jost_path(u0,1j*kap,right,float(match),[1.,0.],-kap*right,settings,audit,'bound')
        ov=np.vdot(pl.states[-1],pr.states[-1]);mismatch=float(max(0.,1-abs(ov)**2))
        if abs(ov)<.95:raise RuntimeError('Left/right bound-state match failed; refine eigenvalues and quantum propagation.')
        log_scale=pr.log_norm[-1]-pl.log_norm[-1]+np.log(abs(ov))
        logIL=_log_square_integral(pl,kap,left)+2*log_scale
        logIR=_log_square_integral(pr,kap,right)
        logN=np.logaddexp(logIL,logIR)
        logs.append(-logN)
        diagnostics.append({'kappa':float(kap),'matching_x':float(match),'direction_mismatch':mismatch,
                            'min_probability':min(pl.min_probability,pr.min_probability)})
    return np.asarray(logs),diagnostics


def quantum_reflection(u0,k,left,right,settings=None,audit=None):
    initial=np.array([0.,np.exp(-1j*k*left)],complex)
    path=quantum_jost_path(u0,k,left,right,initial,0.,settings,audit,'plane')
    z=path.states[-1]
    if abs(z[1])<1e-12:raise FloatingPointError('Unresolved Jost denominator.')
    r=np.exp(-2j*k*right)*z[0]/z[1]
    return complex(r),{'k':float(k),'reflection_abs':float(abs(r)),'min_probability':path.min_probability}
