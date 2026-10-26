"""Genuine uniform-interval QSVT inverse via symmetric QSP unitary sequences.

No classical solve/inverse of G is used to return a solution. Eigenspectrum
bounds and a square root construct the small dense block-encoding oracle;
the inverse polynomial does NOT use individual eigenvalues as interpolation
nodes. Classical preprocessing and ideal full amplitude readout are explicit.
"""
from pathlib import Path
import numpy as np
from .phases import inverse_phases
from .simulator import LocalStatevectorSimulator

DEFAULT_CACHE=Path(__file__).resolve().parent.parent/'phase_cache'

def qsvt_solve(G,b,audit=None,tol=1e-8,cache=DEFAULT_CACHE):
    G=np.asarray(G,float);b=np.asarray(b,float);n=len(b)
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(b)):raise ValueError('Nonfinite GLM input.')
    if G.shape!=(n,n) or not np.allclose(G,G.T,atol=1e-12):raise ValueError('G must be real symmetric.')
    eig,Q=np.linalg.eigh(G)
    if eig[0]<=0:raise ValueError('This benchmark inverse expects SPD GLM matrices.')
    norm=float(np.linalg.norm(b))
    if norm==0:raise ValueError('Zero rhs should be handled explicitly by caller.')
    dim=1<<int(np.ceil(np.log2(max(1,n))))
    # Keep nonphysical padding at a valid eigenvalue; rhs padding is zero.
    alpha=float(eig[-1]*1.01);delta=float(eig[0]/alpha)
    specification=inverse_phases(delta,tol,cache)
    B=np.eye(dim)*.5
    B[:n,:n]=G/alpha
    lam,V=np.linalg.eigh(B)
    S=(V*np.sqrt(np.maximum(0.,1-lam*lam)))@V.T
    W=np.block([[B,1j*S],[1j*S,B]])
    if np.linalg.norm(W.conj().T@W-np.eye(2*dim),ord=np.inf)>1e-9:raise RuntimeError('Invalid block encoding.')
    state=np.zeros(4*dim,complex);state[:n]=b/norm
    sim=LocalStatevectorSimulator(state);sim.hadamard_first()
    # QSP product is R0 W ... W Rd; applying the reversed sequence realizes it.
    phases=specification['phases']
    sim.qsp_phase_pair(phases[-1],dim)
    for ph in reversed(phases[:-1]):
        sim.qsp_signal_pair(W,dim);sim.qsp_phase_pair(ph,dim)
    # (-i U + i U*)/2 extracts the imaginary polynomial coherently.
    sim.diagonal(np.r_[np.full(2*dim,-1j),np.full(2*dim,1j)],'imaginary_part_extractor_phase')
    sim.hadamard_first()
    C=specification['scale']*alpha
    branch=sim.state[:n]
    solution=branch.real*norm/C
    record={'degree':specification['degree'],'alpha':alpha,'C':C,'delta_actual':delta,
            'delta_polynomial':specification['delta'],'phase_error':specification['phase_response_error'],
            'success_probability':float(np.vdot(branch,branch).real),
            'classical_G_inverse_used':False,'spectrum_interpolation_used':False}
    if audit:audit.record('glm_qsvt',sim,record)
    return solution,record
