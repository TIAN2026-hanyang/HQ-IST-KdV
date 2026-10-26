import numpy as np

def three_metrics(x,numerical,exact):
    x=np.asarray(x,float);u=np.asarray(numerical,float);e=np.asarray(exact,float)
    if x.ndim!=1 or u.shape!=x.shape or e.shape!=x.shape or len(x)<3 or np.any(np.diff(x)<=0):
        raise ValueError('Require matching arrays and an increasing grid.')
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(e)):raise ValueError('Nonfinite field.')
    Iex=float(np.trapezoid(e*e,x));Inum=float(np.trapezoid(u*u,x))
    if Iex<=0:raise ValueError('Exact squared norm must be positive.')
    return {'E2':float(np.sqrt(np.trapezoid((u-e)**2,x)/Iex)),
            'Einf':float(np.max(np.abs(u-e))),'EI2':float(abs(Inum-Iex)/Iex)}
