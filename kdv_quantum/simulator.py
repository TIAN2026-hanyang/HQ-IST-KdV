"""A bundled, gate-level ideal statevector simulator (not Qiskit/Aer).

All physics modules call this backend. It applies only unitary instructions;
postselection and amplitude readout are explicit. Dense block-encoding oracles
are permitted for small validation circuits and are counted separately from
primitive gates. No ODE integrator or linear-system solver lives here.
"""
from __future__ import annotations
from collections import Counter
import numpy as np


class SimulationAudit:
    def __init__(self):
        self.stages={}
    def record(self,stage,sim,metadata=None):
        item=self.stages.setdefault(stage,{'circuit_executions':0,'gate_calls':{},'max_qubits':0,'max_norm_error':0.,'samples':[]})
        if not np.all(np.isfinite(sim.state)):raise FloatingPointError('Nonfinite state reached stage audit.')
        item['circuit_executions']+=1
        item['max_qubits']=max(item['max_qubits'],sim.n_qubits)
        item['max_norm_error']=max(item['max_norm_error'],abs(float(np.vdot(sim.state,sim.state).real)-1))
        for k,v in sim.counts.items():item['gate_calls'][k]=item['gate_calls'].get(k,0)+v
        if metadata and len(item['samples'])<8:item['samples'].append(metadata)
    def to_dict(self):
        return {'backend':'LocalStatevectorSimulator',
                'backend_type':'bundled gate-level ideal statevector simulator',
                'external_qiskit_executed':False,'stages':self.stages}


class LocalStatevectorSimulator:
    """Big-endian wire ordering. Matrix dimensions must be powers of two."""
    def __init__(self,initial):
        v=np.asarray(initial,dtype=complex).reshape(-1)
        n=len(v)
        if n<2 or n&(n-1):raise ValueError('State dimension must be a power of two >=2.')
        if not np.all(np.isfinite(v)):raise ValueError("Nonfinite quantum state.")
        if not np.isclose(np.linalg.norm(v),1,rtol=0,atol=2e-10):raise ValueError('Initial state must be normalized.')
        self.state=v.copy();self.n_qubits=int(np.log2(n));self.counts=Counter({'state_preparation':1})
    def dense(self,U,label='unitary_oracle',check=True):
        U=np.asarray(U,complex)
        if U.shape!=(len(self.state),len(self.state)):raise ValueError('Gate/state dimension mismatch.')
        if not np.all(np.isfinite(U)):raise ValueError('Nonfinite unitary gate.')
        if check and np.linalg.norm(U.conj().T@U-np.eye(len(U)),ord=np.inf)>2e-9:raise ValueError('Nonunitary gate rejected.')
        self.state=U@self.state;self.counts[label]+=1
    def diagonal(self,phases,label='diagonal_unitary'):
        p=np.asarray(phases,complex).reshape(-1)
        if len(p)!=len(self.state) or not np.all(np.isfinite(p)) or np.max(np.abs(np.abs(p)-1))>2e-12:raise ValueError('Diagonal gate must have unit-modulus entries.')
        self.state*=p;self.counts[label]+=1
    def auxiliary_qft(self,inverse=False):
        v=self.state.reshape(2,-1)
        self.state=(np.fft.ifft(v,axis=1,norm='ortho') if inverse else np.fft.fft(v,axis=1,norm='ortho')).reshape(-1)
        self.counts['inverse_QFT' if inverse else 'QFT']+=1
    def multiplexed_rotation(self,axis,angles,label=None):
        v=self.state.reshape(2,-1);a=np.asarray(angles,float)
        if a.ndim==0:a=np.full(v.shape[1],a)
        if a.shape!=(v.shape[1],):raise ValueError('One angle per auxiliary basis state required.')
        if not np.all(np.isfinite(a)):raise ValueError('Nonfinite rotation angle.')
        c=np.cos(a/2);s=np.sin(a/2);u=v[0].copy();w=v[1].copy()
        if axis=='x':v[0]=c*u-1j*s*w;v[1]=-1j*s*u+c*w
        elif axis=='y':v[0]=c*u-s*w;v[1]=s*u+c*w
        elif axis=='z':v[0]=np.exp(-.5j*a)*u;v[1]=np.exp(.5j*a)*w
        else:raise ValueError('Unknown rotation axis.')
        self.counts[label or ('multiplexed_R'+axis)]+=1
    def auxiliary_phase(self,phase):
        phase=np.asarray(phase,complex)
        if not np.all(np.isfinite(phase)) or np.max(np.abs(abs(phase)-1))>2e-12:raise ValueError('Nonunitary phase rejected.')
        self.state.reshape(2,-1)[:]*=phase[None,:];self.counts['auxiliary_phase']+=1
    def qsp_phase_pair(self,phi,data_dim):
        # branch 0 is U, branch 1 is U*; signal wire is second wire.
        v=self.state.reshape(2,2,data_dim)
        v[0,0]*=np.exp(1j*phi);v[0,1]*=np.exp(-1j*phi)
        v[1,0]*=np.exp(-1j*phi);v[1,1]*=np.exp(1j*phi)
        self.counts['controlled_projector_phase']+=1
    def qsp_signal_pair(self,W,data_dim):
        v=self.state.reshape(2,2*data_dim)
        v[0]=W@v[0];v[1]=W.conj()@v[1]
        self.counts['controlled_QSVT_signal']+=1
    def hadamard_first(self):
        v=self.state.reshape(2,-1);a=v[0].copy();b=v[1].copy()
        v[0]=(a+b)/np.sqrt(2);v[1]=(a-b)/np.sqrt(2);self.counts['H']+=1
