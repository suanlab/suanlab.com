"""Independent matrices and exact finite-trial laws for the entanglement supplements."""
import json,math
from fractions import Fraction as Q
from itertools import product
import numpy as np
import sympy as sp
import mpmath as mp
mp.mp.dps=70
rng=np.random.default_rng(851031)
I=np.eye(2);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.]);axes=[X,Y,Z]
counts=dict(productGeometries=0,separableWitnessPreparations=0,filterPreparations=0,pptWitnessPreparations=0,wernerMeasurementPreparations=0,confidenceLaws=0,powerLaws=0,localAssignments=0,bellTailLaws=0,quantumGamePreparations=0)
maximum=0.
def close(a,b):
 global maximum
 residual=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum=max(maximum,residual)
 assert residual<3e-12,(a,b,residual)
def density(v):return np.outer(v,v.conj())
def pt(rho):return rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
def bob(rho):return np.trace(rho.reshape(2,2,2,2),axis1=0,axis2=2)
def unitary(d):
 a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d));return np.linalg.qr(a)[0]
def trace_distance(a,b):return np.abs(np.linalg.eigvalsh(a-b)).sum()/2
for da,db in [(2,2),(2,3),(3,4),(4,4)]:
 for _ in range(6):
  c=rng.normal(size=(da,db))+1j*rng.normal(size=(da,db));c/=np.linalg.norm(c);psi=c.ravel();rho=density(psi)
  u,s,vh=np.linalg.svd(c,full_matrices=False);largest=s[0]**2;alpha=u[:,0];beta=vh[0,:];pair=np.kron(alpha,beta)
  close(abs(np.vdot(pair,psi))**2,largest)
  close(trace_distance(rho,density(pair)),np.sqrt(1-largest))
  phase=np.angle(np.vdot(pair,psi));close(np.linalg.norm(psi-np.exp(1j*phase)*pair)**2,2-2*np.sqrt(largest))
  witness=largest*np.eye(da*db)-rho
  sep=np.zeros_like(rho)
  for weight in [.2,.3,.5]:
   a=unitary(da)[:,0];b=unitary(db)[:,0];sep+=weight*density(np.kron(a,b))
  assert np.trace(sep@witness).real>=-1e-13;counts['separableWitnessPreparations']+=1
  counts['productGeometries']+=1
bell=np.array([1.,0,0,1])/np.sqrt(2)
for q in [0.,1e-6,.1,.2,1/3,.5]:
 psi=np.array([np.sqrt(1-q),0,0,np.sqrt(q)]);ks=np.diag([np.sqrt(q/(1-q)),1]);kf=np.diag([np.sqrt((1-2*q)/(1-q)),0]);close(ks@ks+kf@kf,I)
 for _ in range(4):
  ua,ub=unitary(2),unitary(2);joint=np.kron(ua,ub);state=joint@psi
  outputs=[np.kron(ua@k@ua.conj().T,I)@state for k in [ks,kf]]
  close(np.vdot(outputs[0],outputs[0]),2*q);close(np.vdot(outputs[1],outputs[1]),1-2*q)
  close(density(outputs[0]),2*q*density(joint@bell))
  close(sum(bob(density(v)) for v in outputs),bob(density(state)))
  if q>0:close(bob(density(outputs[0]))/(2*q),I/2)
  counts['filterPreparations']+=1
 h=0 if q==0 else -(1-q)*math.log1p(-q)-q*math.log(q)
 assert 2*q*math.log(2)<=h+1e-14
# Exact rational-amplitude endpoint bookkeeping, independent of numerical branch normalization.
for u in [sp.Rational(0),sp.Rational(1,3),sp.Rational(1,2),sp.Rational(1)]:
 state=sp.Matrix([1,0,0,u])/sp.sqrt(1+u*u);ks=sp.diag(u,1);out=sp.kronecker_product(ks,sp.eye(2))*state
 assert sp.simplify((out.T*out)[0]-2*u*u/(1+u*u))==0
 if u==0:assert out==sp.zeros(4,1)
for _ in range(20):
 psi=unitary(4)[:,0];rho=density(psi);values,vectors=np.linalg.eigh(pt(rho));assert values[0]<-1e-8
 v=vectors[:,0];w=pt(density(v));close(np.trace(rho@w),values[0])
 a,b=unitary(2)[:,0],unitary(2)[:,0];assert np.trace(density(np.kron(a,b))@w).real>=-1e-13
 counts['pptWitnessPreparations']+=1
singlet=np.array([0,1.,-1.,0])/np.sqrt(2);w=np.eye(4)/2-density(singlet)
close(w,(np.eye(4)+sum(np.kron(a,a) for a in axes))/4)
for p in [0.,.2,1/3,.5,.6,1/np.sqrt(2),.8,.9,1.]:
 rho=p*density(singlet)+(1-p)*np.eye(4)/4;score=0;opposite=0;reported=0;bound=0
 for j,a in enumerate(axes):
  error_probability=[.01,.03,.07][j];ideal=np.trace(rho@np.kron(a,a)).real
  # Independently sum the complete measurement law, including a source-independent Alice-only detector flip.
  for s,t in product([-1,1],repeat=2):
   prob=np.trace(rho@np.kron((I+s*a)/2,(I+t*a)/2)).real/3
   score+=(1+3*s*t)/4*prob
   opposite+=prob*(s*t==-1)
   for flip,weight in [(1,1-error_probability),(-1,error_probability)]:reported+=(1+3*flip*s*t)/4*prob*weight
  bound+=2*error_probability/4
 close(score,np.trace(rho@w));close(score,(1-3*p)/4);close(opposite,(1+p)/2)
 assert abs(reported-score)<=bound+1e-14
 counts['wernerMeasurementPreparations']+=1

def real(q):return mp.mpf(q.numerator)/q.denominator
def binomial(n,probability):
 return [(k,Q(math.comb(n,k))*probability**k*(1-probability)**(n-k)) for k in range(n+1)]
def radius(n,delta):return mp.mpf('1.5')*mp.sqrt(mp.log(1/delta)/(2*n))
maximum_coverage_failure=mp.mpf(0)
for n,q,exponent in product([4,10,25,64],[Q(0),Q(1,4),Q(1,2),Q(2,3),Q(1)],[2,4]):
 delta=mp.exp(-exponent);r=radius(n,delta);mean=1-mp.mpf('1.5')*real(q)
 bad=sum((prob for k,prob in binomial(n,q) if mean>real(1-Q(3*k,2*n))+r),Q(0))
 assert real(bad)<=delta+mp.mpf('1e-65');maximum_coverage_failure=max(maximum_coverage_failure,real(bad));counts['confidenceLaws']+=1
for n,p in product([64,256,1024],[Q(1,2),Q(4,5),Q(1)]):
 delta=mp.exp(-4);beta=mp.exp(-2);gamma=real((3*p-1)/4)
 if gamma<=radius(n,delta)+radius(n,beta):continue
 fail=sum((prob for k,prob in binomial(n,(1+p)/2) if real(1-Q(3*k,2*n))+radius(n,delta)>=0),Q(0))
 assert real(fail)<=beta;counts['powerLaws']+=1
for a0,a1,b0,b1 in product([-1,1],repeat=4):
 s=a0*b0+a0*b1+a1*b0-a1*b1;wins=sum([a0,a1][x]*[b0,b1][y]==(-1)**(x*y) for x,y in product([0,1],repeat=2))
 assert abs(s)==2 and Q(wins,4)==Q(1,2)+Q(s,8) and wins<=3;counts['localAssignments']+=1
for n,q,exponent in product([10,40],[Q(1,4),Q(1,2),Q(3,4)],[2,4]):
 delta=mp.exp(-exponent);threshold=mp.mpf('.75')+mp.sqrt(mp.log(1/delta)/(2*n))
 tail=sum((prob for k,prob in binomial(n,q) if mp.mpf(k)/n>threshold),Q(0));assert real(tail)<=delta;counts['bellTailLaws']+=1
for p in [0.,1/3,.5,.6,1/np.sqrt(2),.9,1.]:
 rho=p*density(singlet)+(1-p)*np.eye(4)/4;alice=[Z,X];bs=[-(Z+X)/np.sqrt(2),-(Z-X)/np.sqrt(2)];win=0
 for x,y,s,t in product([0,1],[0,1],[-1,1],[-1,1]):
  prob=np.trace(rho@np.kron((I+s*alice[x])/2,(I+t*bs[y])/2)).real/4
  if s*t==(-1)**(x*y):win+=prob
 close(win,.5+np.sqrt(2)*p/4);counts['quantumGamePreparations']+=1
print(json.dumps(dict(status='passed',**counts,maxMatrixResidual=maximum,maximumEnumeratedCoverageFailure=float(maximum_coverage_failure),exactRationalAmplitudeCases=4)))
