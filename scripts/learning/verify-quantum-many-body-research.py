"""Independent dense spin/Fock matrices for finite response and Majorana claims."""
import json
from itertools import product
import numpy as np
import scipy.linalg as la
import sympy as sp
rng=np.random.default_rng(723109)
I=np.eye(2);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1.]);xx=np.kron(X,X);zz=np.kron(Z,Z);M=np.kron(Z,I)+np.kron(I,Z);Q=np.kron(X,X)
counts=dict(spinPreparations=0,longitudinalPairs=0,gaussianPreparations=0,quadraticEvolutions=0,majoranaEdgePreparations=0,interactionPreparations=0)
maximum=0.;derivative_error=0.
def close(a,b):
 global maximum
 r=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum=max(maximum,r);assert r<5e-10,(a,b,r)
def state(v):return np.outer(v,v.conj())
for j,x in product([.5,1.,2.],[.02,.1,.375,1.,3.]):
 h=x*j;ham=-j*zz-h*(np.kron(X,I)+np.kron(I,X));e,v=np.linalg.eigh(ham);g=v[:,0];rho=state(g);root=np.sqrt(j*j+4*h*h);gap=4*h*h/(root+j)
 close(e,[-root,-j,j,root]);close(Q@g,g);close(np.vdot(g,M@g),0);close(np.trace(rho@zz),j/root);close(np.trace(rho@np.kron(X,I)),2*h/root)
 red=np.trace(rho.reshape(2,2,2,2),axis1=1,axis2=3);close(red,(I+(2*h/root)*X)/2)
 response=sum(2*abs(np.vdot(v[:,n],M@g))**2/(e[n]-e[0]) for n in range(1,4));expected=4*(root+j)/(root*gap);assert abs(response/expected-1)<5e-12
 close(np.vdot(g,M@M@g),2*(1+j/root))
 for fraction in [.001,.01,.05]:
  b=fraction*gap;ep,vp=np.linalg.eigh(ham-b*M);em,vm=np.linalg.eigh(ham+b*M);mp=np.vdot(vp[:,0],M@vp[:,0]).real;mm=np.vdot(vm[:,0],M@vm[:,0]).real
  close(mp,-mm);assert ep[1]-ep[0]>=gap-4*b-1e-12;assert abs(mp)<=2+1e-12
  if fraction==.001:derivative_error=max(derivative_error,abs(mp/b/expected-1));assert abs(mp/b/expected-1)<2e-5
  counts['longitudinalPairs']+=1
 counts['spinPreparations']+=1

def fermions(n):
 size=2**n;cs=[]
 for j in range(n):
  c=np.zeros((size,size),complex)
  for col in range(size):
   bits=[(col>>(n-1-k))&1 for k in range(n)]
   if bits[j]:c[col-(1<<(n-1-j)),col]=(-1)**sum(bits[:j])
  cs.append(c)
 ms=[m for c in cs for m in [c+c.conj().T,-1j*(c-c.conj().T)]]
 parity=np.diag([(-1)**k.bit_count() for k in range(size)])
 return cs,ms,parity

def covariance(rho,ms):return np.array([[((1j/2)*np.trace(rho@(a@b-b@a))).real for b in ms] for a in ms])
vac_cs,vac_ms,vac_parity=fermions(1)
close(covariance(np.diag([1.,0.]),vac_ms)[0,1],-1)
for n in range(1,5):
 cs,ms,parity=fermions(n);size=2**n
 for a,b in product(range(2*n),repeat=2):close(ms[a]@ms[b]+ms[b]@ms[a],2*np.eye(size)*(a==b))
 for seed in range(4):
  raw=rng.normal(size=(2*n,2*n));a=raw-raw.T;ham=sum(1j/4*a[j,k]*(ms[j]@ms[k]) for j,k in product(range(2*n),repeat=2));close(ham,ham.conj().T);e,v=np.linalg.eigh(ham)
  gaussian=[]
  for temp in [.4,1.7]:
   rho=la.expm(-(ham-e[0]*np.eye(size))/temp);rho/=np.trace(rho);gaussian.append(rho)
  gaussian.append(state(v[:,0]))
  for rho in gaussian:
   gamma=covariance(rho,ms);close(np.trace(rho@ham),-np.trace(a@gamma)/4);assert np.linalg.eigvalsh(np.eye(2*n)-1j*gamma)[0]>-1e-12
   if rho is gaussian[-1]:close(gamma@gamma,-np.eye(2*n))
   one=np.array([[np.trace(rho@ci.conj().T@cj) for cj in cs]for ci in cs]);anomalous=np.array([[np.trace(rho@ci@cj)for cj in cs]for ci in cs])
   for i,j in product(range(n),repeat=2):
    if i==j:continue
    ni,nj=cs[i].conj().T@cs[i],cs[j].conj().T@cs[j];close(np.trace(rho@ni@nj),one[i,i]*one[j,j]-abs(one[i,j])**2+abs(anomalous[i,j])**2)
   counts['gaussianPreparations']+=1
  raw=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size));rho=raw@raw.conj().T;rho=(rho+parity@rho@parity)/2;rho/=np.trace(rho)
  for initial,t in product([gaussian[0],rho],[-.3,0.,.7]):
   u=la.expm(-1j*ham*t);o=la.expm(a*t);close(o.T@o,np.eye(2*n));close(u@parity,parity@u)
   for j in range(2*n):close(u.conj().T@ms[j]@u,sum(o[j,k]*ms[k] for k in range(2*n)))
   close(covariance(u@initial@u.conj().T,ms),o@covariance(initial,ms)@o.T);counts['quadraticEvolutions']+=1
cs,ms,p=fermions(4);cat=np.zeros(16);cat[0]=cat[-1]=1/np.sqrt(2);rho=state(cat);mixed=np.eye(16)/16;numbers=[c.conj().T@c for c in cs];number=sum(numbers)
for r in [rho,mixed]:close(covariance(r,ms),np.zeros((8,8)));close(np.trace(r@number),2)
close(np.trace(rho@numbers[0]@numbers[1]),.5);close(np.trace(mixed@numbers[0]@numbers[1]),.25);close(np.trace(rho@number@number)-4,4);close(np.trace(mixed@number@number)-4,1)
cs,ms,p=fermions(2)
for u in [0.,.01,.25,.5,1.,2.]:
 he=1j/2*ms[1]@ms[2];ham=he+1j*u/2*ms[0]@ms[3];close(ham@p,p@ham);close(np.linalg.eigvalsh(ham),sorted((s+r*u)/2 for s,r in product([-1,1],repeat=2)))
 for k in [0,3]:close(he@ms[k]-ms[k]@he,np.zeros((4,4)));close(p@ms[k]+ms[k]@p,np.zeros((4,4)))
 expansion=-.5*(cs[0].conj().T@cs[1]+cs[1].conj().T@cs[0]+cs[0].conj().T@cs[1].conj().T+cs[1]@cs[0]);close(expansion,he);counts['majoranaEdgePreparations']+=1
for n,u in product([2,3,4],[.5,1.,-2.]):
 cs,ms,p=fermions(n);n0,n1=[cs[k].conj().T@cs[k]for k in [0,1]];ham=u*n0@n1;derivative=1j*(ham@ms[0]-ms[0]@ham);close(derivative,u*n1@ms[1]);linear=sum(np.trace(m@derivative)/(2**n)*m for m in ms);assert np.linalg.norm(derivative-linear)>1e-3;counts['interactionPreparations']+=1
# Exact low-dimensional characteristic polynomials and covariance-sign convention.
a=sp.Matrix([[0,1],[0,0]]);one=sp.eye(2);z=sp.diag(1,-1);cs=[sp.kronecker_product(a,one),sp.kronecker_product(z,a)];ms=[m for c in cs for m in [c+c.adjoint(),-sp.I*(c-c.adjoint())]]
for ratio in [sp.Rational(0),sp.Rational(1,4),sp.Rational(1,2),sp.Rational(1)]:
 ham=sp.I*(ms[1]*ms[2]+ratio*ms[0]*ms[3])/2;lam=sp.Symbol('lam');expected=sp.prod(lam-(s+r*ratio)/2 for s,r in product([-1,1],repeat=2));assert sp.expand((lam*sp.eye(4)-ham).det()-expected)==0
for u in [sp.Rational(1,10),sp.Rational(1,3),sp.Rational(1,2),sp.Rational(3,4)]:
 h=u/(1-u*u);root=(1+u*u)/(1-u*u);assert sp.simplify(root*root-1-4*h*h)==0
print(json.dumps(dict(status='passed',**counts,maxMatrixResidual=maximum,maxResponseRelativeError=derivative_error,covarianceComparisonStates=2,exactSpinCases=4,exactEdgeCases=4)))
