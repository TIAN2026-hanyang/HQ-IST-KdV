"""Exact finite-rank GLM reduction for the declared reflectionless N-solitons.

F(s)=sum c_j exp(-kappa_j s). G=I+v_i v_j/(kappa_i+kappa_j),
K(x,x)=-v^T G^-1 v. Every inverse application below is a QSVT circuit.
Diagonal equilibration and equivalent left/right data avoid large exponentials;
no inverse of G is used in preprocessing. The reducer is NOT a general
continuous-spectrum GLM solver; nonzero reflection benchmarks are rejected.
"""
import numpy as np
from scipy.special import expit
from .qsvt import qsvt_solve


def left_log_norming(kappa,log_c_right):
    k=np.asarray(kappa,float);lc=np.asarray(log_c_right,float)
    logA=np.zeros(len(k))
    for i in range(len(k)):
        for j in range(len(k)):
            if i!=j:logA[i]+=2*np.log(abs((k[i]-k[j])/(k[i]+k[j])))
    return 2*np.log(2*k)-lc-logA


def equilibrated_matrix(y,kappa,log_c):
    k=np.asarray(kappa,float)
    log_v=.5*np.asarray(log_c)-k*y
    h=np.sqrt(2*k*expit(2*log_v-np.log(2*k)))
    H=h[:,None]*h[None,:]/(k[:,None]+k[None,:])
    np.fill_diagonal(H,1.)
    return H,h


def kernel_diagonal(y,kappa,log_c,audit=None,tol=1e-8,cache=None):
    H,h=equilibrated_matrix(y,kappa,log_c)
    kw={} if cache is None else {'cache':cache}
    solution,record=qsvt_solve(H,h,audit,tol,**kw)
    return -float(h@solution),record


def reconstruct_field(x,kappa,log_c,audit=None,derivative_step=.003,tol=1e-8,cache=None):
    if derivative_step<=0:raise ValueError('derivative_step must be positive.')
    x=np.asarray(x,float);u=np.empty_like(x);left=left_log_norming(kappa,log_c)
    stats={'qsvt_calls':0,'max_degree':0,'max_condition_equilibrated':0.,'left_points':0,'right_points':0}
    for index,xx in np.ndenumerate(x):
        Hr,_=equilibrated_matrix(float(xx),kappa,log_c)
        Hl,_=equilibrated_matrix(-float(xx),kappa,left)
        er=np.linalg.eigvalsh(Hr);el=np.linalg.eigvalsh(Hl)
        cr=er[-1]/er[0];cl=el[-1]/el[0]
        if cl<cr:y=-float(xx);lc=left;stats['left_points']+=1;condition=cl
        else:y=float(xx);lc=log_c;stats['right_points']+=1;condition=cr
        # Same orientation for both derivative evaluations; no differentiation
        # of a discontinuous left/right selector.
        bp,rp=kernel_diagonal(y+derivative_step,kappa,lc,audit,tol,cache)
        bm,rm=kernel_diagonal(y-derivative_step,kappa,lc,audit,tol,cache)
        u[index]=(bp-bm)/derivative_step
        stats['qsvt_calls']+=2;stats['max_degree']=max(stats['max_degree'],rp['degree'],rm['degree'])
        stats['max_condition_equilibrated']=max(stats['max_condition_equilibrated'],float(condition))
    return u,stats
