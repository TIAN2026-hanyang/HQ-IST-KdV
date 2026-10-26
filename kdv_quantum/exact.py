"""Analytic positive-polarity KdV fields u_t+6u*u_x+u_xxx=0."""
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass(frozen=True)
class Case:
    name:str
    kappa:tuple
    c0:tuple
    x_min:float
    x_max:float
    points:int
    times:tuple
    spectral_left:float
    spectral_right:float

CASES={
 'single':Case('single',(1.,),(2.,),-20.,20.,501,(0.,.8,1.2,2.),-18.,18.),
 'double':Case('double',(1.,.5),(2.,float(np.exp(10.))),-20.,50.,501,(0.,1.5,3.,7.),-24.,38.),
 'triple':Case('triple',(3.,2.,1.),(60.,60.,12.),-21.,21.,421,(0.,.125,.25,.5),-18.,18.),
}

def exact_field(x,t,case):
    if isinstance(case,str):case=CASES[case]
    k=np.asarray(case.kappa);lc=np.log(case.c0);x=np.asarray(x,float)
    masks=np.asarray(list(product([0.,1.],repeat=len(k))))
    theta=lc[:,None]-np.log(2*k[:,None])-2*k[:,None]*x.reshape(1,-1)+8*k[:,None]**3*t
    logw=masks@theta
    for i in range(len(k)):
        for j in range(i+1,len(k)):
            logw+=(masks[:,i]*masks[:,j])[:,None]*2*np.log(abs((k[i]-k[j])/(k[i]+k[j])))
    logw-=logw.max(axis=0,keepdims=True)
    w=np.exp(logw);w/=w.sum(axis=0,keepdims=True)
    slopes=-2*masks@k;mean=np.sum(w*slopes[:,None],axis=0)
    return (2*np.sum(w*(slopes[:,None]-mean)**2,axis=0)).reshape(x.shape)
