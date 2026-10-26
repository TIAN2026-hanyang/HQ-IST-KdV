"""Uniform-interval odd inverse QSP, not interpolation at matrix eigenvalues.

Independent implementation of symmetric real-signal QSP with Newton phase
synthesis. Classical synthesis operates on a scalar target polynomial only.
QSP convention: U=exp(i phi0 Z) W(x) ... W(x) exp(i phid Z),
W(x)=[[x,i sqrt(1-x*x)],[i sqrt(1-x*x),x]]. The imaginary part of
U[0,0] represents the bounded real odd polynomial.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.linalg import solve


def response_and_jacobian(x, reduced, jacobian=True):
    x=np.asarray(x,dtype=float)
    p=np.asarray(reduced,dtype=float)
    full=np.r_[p[::-1],p]
    n=len(p); d=len(full)-1; m=len(x)
    W=np.zeros((m,2,2),complex)
    W[:,0,0]=W[:,1,1]=x
    W[:,0,1]=W[:,1,0]=1j*np.sqrt(np.maximum(0.,1-x*x))
    identity=np.broadcast_to(np.eye(2,dtype=complex),(m,2,2)).copy()
    pref=[]; U=identity.copy()
    for j,ph in enumerate(full):
        if jacobian:pref.append(U.copy())
        U=U*np.array([np.exp(1j*ph),np.exp(-1j*ph)])[None,None,:]
        if j<d:U=U@W
    values=U[:,0,0].imag
    if not jacobian:return values
    suffix=identity.copy(); jf=np.empty((m,len(full)))
    for j in range(d,-1,-1):
        ph=full[j]
        derivative=np.array([1j*np.exp(1j*ph),-1j*np.exp(-1j*ph)])
        # Only first row and first column of prefix*dR*suffix needed.
        jf[:,j]=np.sum(pref[j][:,0,:]*derivative[None,:]*suffix[:,:,0],axis=1).imag
        if j>0:
            R_suffix=np.array([np.exp(1j*ph),np.exp(-1j*ph)])[None,:,None]*suffix
            suffix=W@R_suffix
    J=jf[:,:n][:,::-1]+jf[:,n:]
    return values,J


def make_inverse_polynomial(delta:float, tol:float):
    if not (0<delta<=1) or not (0<tol<.1):raise ValueError('Invalid inverse interval/tolerance.')
    b=np.log(4/tol)/delta**2
    scale=.45/np.sqrt(b)
    ns=1<<int(np.ceil(np.log2(max(256,24*np.sqrt(b)))))
    x=np.cos(np.pi*(np.arange(ns)+.5)/ns)
    vals=scale*(-np.expm1(-b*x*x))/x
    coeff=dct(vals,type=2)/ns;coeff[0]/=2;coeff[::2]=0
    tail=np.cumsum(np.abs(coeff)[::-1])[::-1]
    target=scale*tol/30
    allowed=np.where(tail<target)[0]
    deg=(int(allowed[0])-1) if len(allowed) else ns-1
    if deg%2==0:deg+=1
    deg=max(1,deg)
    coeff=coeff[:deg+1]
    grid=np.r_[np.linspace(delta,1,4096),np.linspace(-1,1,4097)]
    pol=np.polynomial.chebyshev.chebval(grid,coeff)
    if np.max(np.abs(pol))>.5:raise RuntimeError('Target is not safely bounded.')
    xx=np.linspace(delta,1,4096)
    rel=np.max(np.abs(np.polynomial.chebyshev.chebval(xx,coeff)*xx/scale-1))
    if rel>tol:raise RuntimeError(f'Uniform inverse polynomial failed: {rel:g}')
    return coeff,float(scale),float(rel)


def synthesize(coeff,tol=3e-13,maxiter=20):
    coeff=np.asarray(coeff);n=(len(coeff))//2
    reduced=coeff[1::2]/2
    x=np.cos(np.pi*(np.arange(n)+.5)/(2*n))
    target=np.polynomial.chebyshev.chebval(x,coeff)
    previous=np.inf
    for iteration in range(maxiter):
        value,J=response_and_jacobian(x,reduced)
        residual=value-target;error=float(np.max(np.abs(residual)))
        if error<tol:break
        step=solve(J,residual,assume_a='gen',check_finite=False)
        damping=1.
        while damping>1/128:
            candidate=reduced-damping*step
            newerr=np.max(np.abs(response_and_jacobian(x,candidate,False)-target))
            if newerr<error:break
            damping/=2
        reduced=candidate;previous=error
    else:raise RuntimeError(f'QSP phase synthesis did not converge: {error:g}')
    dense=np.linspace(-1,1,max(1001,8*len(coeff)))
    err=float(np.max(np.abs(response_and_jacobian(dense,reduced,False)-np.polynomial.chebyshev.chebval(dense,coeff))))
    if err>5e-11:raise RuntimeError(f'QSP dense verification failed: {err:g}')
    return np.r_[reduced[::-1],reduced],err,iteration


def inverse_phases(delta,tol=1e-8,cache=None):
    # Powers-of-two binning provides reusable polynomials across matrices.
    bin_delta=2.**np.floor(np.log2(min(.95,delta)))
    key=f'inverse_delta_{bin_delta:.10g}_tol_{tol:.2g}.json'
    path=Path(cache)/key if cache is not None else None
    if path is not None and path.exists():
        data=json.loads(path.read_text())
        # Legacy phase files have no hash; independently validate the numerical
        # response for every loaded cache. Any corrupted scale or phase fails.
        ph=np.asarray(data['phases'],float)
        if len(ph)!=int(data['degree'])+1 or len(ph)%2!=0 or not np.all(np.isfinite(ph)):
            raise ValueError('Invalid QSP phase cache.')
        if abs(float(data['delta'])-bin_delta)>1e-14 or abs(float(data['tolerance'])-tol)>1e-15:
            raise ValueError('QSP phase cache parameter mismatch.')
        if len(ph)>4096:raise MemoryError('QSP polynomial exceeds configured resource limit.')
        x=np.linspace(bin_delta,1,257)
        rr=response_and_jacobian(x,ph[len(ph)//2:],False)
        residual=float(np.max(abs(rr*x/float(data['scale'])-1)))
        if not np.isfinite(residual) or residual>max(tol*3,3e-8):
            raise ValueError(f'QSP cache failed independent response check: {residual:g}')
        return data
    coeff,scale,err=make_inverse_polynomial(bin_delta,tol)
    if len(coeff)>4096:raise MemoryError(f"QSP polynomial degree {len(coeff)-1} exceeds 4095; refine conditioning/representation.")
    ph,perr,niter=synthesize(coeff)
    data={'delta':bin_delta,'tolerance':tol,'scale':scale,'degree':len(ph)-1,
          'coefficients':coeff.tolist(),'phases':ph.tolist(),
          'uniform_inverse_relative_bound_sampled':err,'phase_response_error':perr,
          'newton_iterations':niter,
          'method':'uniform_interval_smoothed_inverse_symmetric_QSP'}
    if path is not None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2))
    return data
