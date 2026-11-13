"""Direct matrix oracles for cutoff/variance and exterior-block research lessons."""
import json
from itertools import product
import numpy as np
import sympy as sp
rng=np.random.default_rng(631908)
counts=dict(templePreparations=0,intervalPreparations=0,referencePerturbations=0,blockPreparations=0,coordinatePreparations=0,schurPreparations=0,enrichmentPreparations=0,twoLevelPreparations=0)
maximum=0.
def close(a,b):
 global maximum
 residual=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum=max(maximum,residual);assert residual<3e-12,(a,b,residual)
def hermitian(n):
 a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));return (a+a.conj().T)/2
def unitary(n):return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
def projector(v):return np.outer(v,v.conj())
for spectrum,rank in [([-2,1],1),([-2,1,4],1),([-2,1,4,7],1),([-2,-2,1,4],2)]:
 spectrum=np.array(spectrum);n=len(spectrum)
 for q,cut_fraction,rotation in product([0.,.01,.1,.25],[0.,.1,.25],range(3)):
  u=unitary(n);h=(u*spectrum)@u.conj().T;ground=u[:,:rank]@u[:,:rank].conj().T
  weights=np.array([(1-q)/rank]*rank+[q/(n-rank)]*(n-rank));phase=np.exp(1j*rng.normal(size=n));psi=u@(np.sqrt(weights)*phase)
  for rho in [projector(psi),(u*weights)@u.conj().T]:
   energy=np.trace(rho@h).real;center=h-energy*np.eye(n)
   variance=np.vdot(center@psi,center@psi).real if np.linalg.norm(rho-projector(psi))<1e-14 else sum(weights[j]*np.linalg.norm(center@u[:,j])**2 for j in range(n))
   close(variance,np.trace(rho@h@h).real-energy*energy)
   beta=spectrum[rank]-cut_fraction*(spectrum[rank]-spectrum[0]);d=beta-energy;assert d>0
   lower=energy-variance/d;leak=variance/(variance+d*d);actual_leak=1-np.trace(rho@ground).real
   assert lower<=spectrum[0]+3e-12 and actual_leak<=leak+3e-12
   poly=(h-spectrum[0]*np.eye(n))@(h-beta*np.eye(n));close(np.trace(rho@poly),variance+(energy-spectrum[0])*(energy-beta))
   assert np.trace(rho@poly).real>=-3e-12
   el,eu=energy-.01*d,energy+.01*d;bl=beta-.01*d;vu=variance+.01*max(1,variance);dl=bl-eu
   assert el-vu/dl<=spectrum[0]+3e-12 and actual_leak<=vu/(vu+dl*dl)+3e-12
   counts['templePreparations']+=1;counts['intervalPreparations']+=1
for n,eta,rotation in product([2,3,4],[.1,.3,.5],range(3)):
 u=unitary(n);reference=np.array([-2+3*j for j in range(n)]);h0=(u*reference)@u.conj().T;v=hermitian(n);v*=eta/np.linalg.norm(v,2);values=np.linalg.eigvalsh(h0+v)
 assert np.all(values>=reference-eta-1e-12)and np.all(values<=reference+eta+1e-12);assert reference[1]-eta<=values[1]+1e-12;counts['referencePerturbations']+=1
for (k,m),separation,gamma,seed in product([(1,1),(2,1),(1,2),(2,2),(3,1)],[.5,2.,4.],[0.,.05,.5,2.],range(2)):
 n=k+m;a=hermitian(k);a-=np.eye(k)*np.linalg.eigvalsh(a)[0];a-=np.eye(k);mu=np.linalg.eigvalsh(a)[0]
 d=hermitian(m);d+=np.eye(m)*(mu+separation-np.linalg.eigvalsh(d)[0]);beta=np.linalg.eigvalsh(d)[0]
 c=rng.normal(size=(m,k))+1j*rng.normal(size=(m,k));c*=gamma/np.linalg.norm(c,2)
 block=np.block([[a,c.conj().T],[c,d]]);values,vectors=np.linalg.eigh(block);energy=values[0];x,y=vectors[:k,0],vectors[k:,0]
 width=beta-mu;error_bound=2*gamma*gamma/(width+np.sqrt(width*width+4*gamma*gamma))
 assert mu-error_bound<=energy+1e-12 and energy<=mu+1e-12
 correction=c.conj().T@np.linalg.solve(d-energy*np.eye(m),c)
 close(y,-np.linalg.solve(d-energy*np.eye(m),c@x));close((a-correction)@x,energy*x)
 assert np.linalg.eigvalsh(correction)[0]>-1e-12;assert np.linalg.eigvalsh(c.conj().T@c/(beta-energy)-correction)[0]>-1e-12
 counts['schurPreparations']+=1
 # Change both physical frame and nonorthogonal coordinates, then rebuild the projector independently.
 frame=unitary(n);h=frame@block@frame.conj().T;q=frame[:,:k];r=frame[:,k:];change=np.eye(k,dtype=complex)*2
 if k>1:change[0,1]=1+1j
 b=q@change;s=b.conj().T@b;p=b@np.linalg.solve(s,b.conj().T);close(p,q@q.conj().T)
 qnew=np.linalg.qr(b)[0];close(np.linalg.eigvalsh(qnew.conj().T@h@qnew),np.linalg.eigvalsh(a));close(np.linalg.norm(r.conj().T@h@qnew,2),gamma)
 counts['coordinatePreparations']+=1
 av,avec=np.linalg.eigh(a);psi=frame@np.r_[avec[:,0],np.zeros(m)];residual=h@psi-mu*psi;rn=np.linalg.norm(residual);close(p@residual,np.zeros(n))
 if gamma>0:
  assert rn>1e-10;zeta=residual/rn;nu=np.vdot(zeta,h@zeta).real;enriched=np.linalg.qr(np.column_stack([q,zeta]))[0];new=float(np.linalg.eigvalsh(enriched.conj().T@h@enriched)[0]);comparison=(mu+nu-np.sqrt((nu-mu)**2+4*rn*rn))/2
  assert new<=comparison+1e-12 and new<mu-1e-8;counts['enrichmentPreparations']+=1
 counts['blockPreparations']+=1
for delta,t in product([.01,.1,1.,10.],[0.,.001,.1,1.,3.]):
 h=np.array([[0,t],[t,delta]]);values,vectors=np.linalg.eigh(h);error=2*t*t/(np.sqrt(delta*delta+4*t*t)+delta);outside=(1-delta/np.sqrt(delta*delta+4*t*t))/2
 close(values[0],-error);close(abs(vectors[1,0])**2,outside);counts['twoLevelPreparations']+=1
# Exact interval arithmetic, sharp two-level bounds and failure of an uncertified cutoff.
for q in [sp.Rational(0),sp.Rational(1,4),sp.Rational(3,4),sp.Rational(99,100)]:
 e,v=2*q,4*q*(1-q);assert sp.simplify(e-v/(2-e))==0;assert sp.simplify(v/(v+(2-e)**2)-q)==0
el,eu,vu,bl=sp.Rational(49,100),sp.Rational(51,100),sp.Rational(76,100),sp.Rational(199,100);distance=bl-eu
assert el-vu/distance==-sp.Rational(87,3700)and vu/(vu+distance**2)==sp.Rational(475,1844)
wrong=sp.diag(0,1,9);trial=sp.Matrix([0,1,0]);assert (wrong-trial.dot(wrong*trial)*sp.eye(3))*trial==sp.zeros(3,1)
assert 1-sp.Rational(0,8)>min(wrong.eigenvals())
for u in [sp.Rational(0),sp.Rational(1,3),sp.Rational(1,2),sp.Rational(3,4)]:
 t=u/(1-u*u);root=(1+u*u)/(1-u*u);h=sp.Matrix([[0,t],[t,1]]);e=-(root-1)/2;assert sp.simplify((h-e*sp.eye(2)).det())==0
print(json.dumps(dict(status='passed',**counts,maxMatrixResidual=maximum,exactSharpCases=4,exactIntervalCases=1,exactRationalSchurCases=4,unsafeCutoffCounterexample=True)))
