"""Polynomial spectral rates of the isospectral KdV hierarchy.

Only coefficients (0,-1), i.e. r(z)=-z, are numerically validated in
this package. The other polynomial rates are exposed for analytic interface
checks, *not* claimed as higher-flow numerical simulations.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class IsospectralFlow:
    coefficients: tuple[float, ...]=(0.0,-1.0)

    def __post_init__(self):
        if len(self.coefficients)<1 or not np.all(np.isfinite(self.coefficients)):
            raise ValueError('Coefficients of the flow must be finite.')

    def polynomial(self,z):
        return np.polynomial.polynomial.polyval(z,self.coefficients)

    def omega(self,k):
        k=np.asarray(k,float)
        return k*self.polynomial(-4*k*k)

    def norming_rate(self,kappa):
        kappa=np.asarray(kappa,float)
        return -2*kappa*self.polynomial(4*kappa*kappa)

STANDARD_KDV=IsospectralFlow((0.0,-1.0))
