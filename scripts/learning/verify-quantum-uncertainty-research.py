"""Independent matrix checks for covariance and joint qubit readout supplements."""
import json
from fractions import Fraction as Q
import numpy as np
from itertools import product

X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]])
Z=np.diag([1.,-1.]);I=np.eye(2);pauli=[X,Y,Z]
rng=np.random.default_rng(20971)
def close(a,b):
    residual=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert residual<3e-12,(a,b,residual)
    return residual
def root(a):
    w,v=np.linalg.eigh(a)
    assert min(w)>-3e-15
    return (v*np.sqrt(np.maximum(w,0)))@v.conj().T
def gram(rho,observables):
    centered=[a-np.trace(rho@a)*np.eye(len(rho)) for a in observables]
    return np.array([[np.trace(rho@a@b) for b in centered] for a in centered]),centered
def bloch(v):return sum(x*a for x,a in zip(v,pauli))
counts=dict(finiteGramPreparations=0,pauliPreparations=0,pairPreparations=0,rotationPreparations=0,jointPreparations=0,sequentialPreparations=0,calibratedPreparations=0)
maximum=0.
for d in [2,3,4]:
    for _ in range(8):
        c=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));rho=c@c.conj().T;rho/=np.trace(rho)
        matrices=[]
        for _ in range(3):
            a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));matrices.append(a+a.conj().T)
        g,centered=gram(rho,matrices);sr=root(rho)
        columns=np.column_stack([(a@sr).ravel() for a in centered])
        maximum=max(maximum,close(g,columns.conj().T@columns));assert np.linalg.eigvalsh(g)[0]>-1e-12
        u=rng.normal(size=3);a=sum(t*m for t,m in zip(u,centered));close(u@g.real@u,np.trace(rho@a@a))
        counts['finiteGramPreparations']+=1
for radius in [0.,.25,.6,.9,1.]:
    for direction in [[1,0,0],[0,0,1],[3,4,0],[1,2,3]]:
        r=radius*np.array(direction)/np.linalg.norm(direction);rho=(I+bloch(r))/2;g,_=gram(rho,pauli)
        v=np.eye(3)-np.outer(r,r);close(g.real,v)
        maximum=max(maximum,close(np.linalg.eigvalsh(g),sorted([1+radius,1-radius,1-radius**2])))
        close(np.linalg.eigvalsh(v),[1-radius**2,1,1]);close(np.linalg.det(g),(1-radius**2)**2)
        for i,j in [(0,1),(0,2),(1,2)]:close(np.linalg.det(g[np.ix_([i,j],[i,j])]),2*(1-np.trace(rho@rho)))
        axis=np.array([1.,2.,-1.]);axis/=np.linalg.norm(axis);angle=.73
        cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        rotation=np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross
        gr,_=gram((I+bloch(rotation@r))/2,pauli);close(gr,rotation@g@rotation.T);counts['rotationPreparations']+=1
        for a,b in [([1,0,0],[0,1,0]),([1,2,0],[2,4,0]),([2,-1,3],[1,2,1]),([1,0,0],[1,.001,0])]:
            pair,_=gram(rho,[bloch(a),bloch(b)]);close(np.linalg.det(pair),(1-r@r)*np.linalg.norm(np.cross(a,b))**2);counts['pairPreparations']+=1
        counts['pauliPreparations']+=1
# Exact endpoint ranks do not depend on eigenvalue cutoffs.
import sympy as sp
pure=sp.Matrix([[1,sp.I,0],[-sp.I,1,0],[0,0,0]])
assert pure.rank()==1 and sp.re(pure).rank()==2
for k,l in product([0.,.2,.6,.8,.9,1.],repeat=2):
    effects={(s,t):(I+s*k*Z+t*l*X)/4 for s,t in product([-1,1],repeat=2)}
    for s,t in effects:
        close(np.linalg.eigvalsh(effects[s,t]),[(1-np.hypot(k,l))/4,(1+np.hypot(k,l))/4])
    for s in [-1,1]:close(sum(effects[s,t] for t in [-1,1]),(I+s*k*Z)/2)
    for t in [-1,1]:close(sum(effects[s,t] for s in [-1,1]),(I+t*l*X)/2)
    if k*k+l*l>1+1e-14:assert min(np.linalg.eigvalsh(effects[1,1]))<0
    else:
        assert min(np.linalg.eigvalsh(effects[1,1]))>=-1e-14
        if k>0 and l>0:
            # Reconstruct moment operators, not only the printed scalar formula.
            first_z=sum(s/k*e for (s,t),e in effects.items());second_z=sum((s/k)**2*e for (s,t),e in effects.items())
            first_x=sum(t/l*e for (s,t),e in effects.items());second_x=sum((t/l)**2*e for (s,t),e in effects.items())
            close(first_z,Z);close(first_x,X)
            noise_z=second_z-first_z@Z-Z@first_z+I;noise_x=second_x-first_x@X-X@first_x+I
            close(noise_z,(k**-2-1)*I);close(noise_x,(l**-2-1)*I)
            assert (k**-2-1)*(l**-2-1)>=1-1e-12
            counts['calibratedPreparations']+=1
    counts['jointPreparations']+=1
for k,mu in product([0.,.2,.6,.8,1.],[0.,.25,.6,1.]):
    h=np.sqrt(1-k*k)
    for s,t in product([-1,1],repeat=2):
        effect=(I+s*k*Z)/2;K=root(effect)
        actual=K@((I+t*mu*X)/2)@K
        maximum=max(maximum,close(actual,(I+s*k*Z+t*mu*h*X)/4))
        # Measure-and-prepare Z has identical first effects but coin X output.
        prepared_projector=(I+s*Z)/2
        close(np.trace(prepared_projector@((I+t*X)/2))*effect,effect/2)
    counts['sequentialPreparations']+=1
for u in [Q(0),Q(1,3),Q(1,2),Q(1)]:
    k,l=(1-u*u)/(1+u*u),2*u/(1+u*u)
    assert k*k+l*l==1
    if k and l:assert (1/(k*k)-1)*(1/(l*l)-1)==1
print(json.dumps(dict(status='passed',**counts,maxMatrixResidual=maximum,exactEndpointRanks=[1,2])))
