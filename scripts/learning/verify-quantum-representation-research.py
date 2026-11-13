"""Independent oscillator evolutions, dense MPS amplitudes and transfer spectra."""
import json,math
from itertools import product
import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc
import sympy as sp
counts=dict(boundaryPreparations=0,dynamicalComparisons=0,analyticSchrodingerPreparations=0,finiteMpsStates=0,gaugeStateComparisons=0,gaugePowerComparisons=0,jordanComparisons=0)
maximum=0.
def close(a,b):
 global maximum
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maximum=max(maximum,e);assert e<3e-10,(a,b,e)
def ladder(c):
 a=np.zeros((c+1,c+1),complex)
 for n in range(1,c+1):a[n-1,n]=np.sqrt(n)
 return a
def trajectory(c,omega,f):
 a=ladder(c);h=omega*(a.conj().T@a)+f*a.conj().T+f.conjugate()*a;e,v=np.linalg.eigh(h)
 return h,lambda t:v@(np.exp(-1j*e*t)*v[0,:].conj())
def coherent(t,omega,f,c):
 alpha=-f/omega*(1-np.exp(-1j*omega*t));mu=abs(alpha)**2;phase=abs(f/omega)**2*(omega*t-np.sin(omega*t));coeff=np.array([alpha**n/np.sqrt(float(math.factorial(n)))for n in range(c+1)],complex)*np.exp(-mu/2+1j*phase)
 return coeff,alpha,mu,phase
for omega,f in product([.7,1.],[1+0j,.3+.4j,2j]):
 for t in [.1,.5,1.,2*np.pi/np.sqrt(5)]:
  coeff,alpha,mu,phase=coherent(t,omega,f,10);adot=-1j*f*np.exp(-1j*omega*t);mudot=2*(alpha.conjugate()*adot).real;phidot=abs(f)**2/omega*(1-np.cos(omega*t))
  for n in range(9):
   derivative=(1j*phidot-mudot/2)*coeff[n]
   if n:derivative+=np.exp(-mu/2+1j*phase)*n*alpha**(n-1)*adot/np.sqrt(float(math.factorial(n)))
   acted=omega*n*coeff[n]+f*np.sqrt(n)*(coeff[n-1]if n else 0)+f.conjugate()*np.sqrt(n+1)*coeff[n+1]
   close(1j*derivative,acted)
  counts['analyticSchrodingerPreparations']+=1
 for c in [0,1,3,6]:
  hc,evolve=trajectory(c,omega,f);a=ladder(c+1);full=omega*a.conj().T@a+f*a.conj().T+f.conjugate()*a
  for t in [.1,.5,1.,2*np.pi/np.sqrt(5)]:
   finite=evolve(t);embedded=np.r_[finite,0j];residual=full@embedded-np.r_[hc@finite,0j];expected=np.zeros(c+2,complex);expected[-1]=f*np.sqrt(c+1)*finite[-1];close(residual,expected)
   bound=abs(f)*np.sqrt(c+1)*quad(lambda s:abs(evolve(s)[-1]),0,t,epsabs=2e-12,epsrel=2e-12,limit=200)[0]
   exact,alpha,mu,phase=coherent(t,omega,f,c);tail=gammainc(c+1,mu);error=np.sqrt(np.linalg.norm(exact-finite)**2+tail)
   assert error<=bound+3e-10,(c,t,error,bound)
   overlap=np.vdot(exact,finite);aligned=finite*np.exp(-1j*np.angle(overlap));distance_sq=np.linalg.norm(exact-aligned)**2+tail;trace=np.sqrt(distance_sq*(1-distance_sq/4));assert trace<=min(1,bound)+3e-10
   counts['boundaryPreparations']+=1;counts['dynamicalComparisons']+=1
h,evolve=trajectory(1,1.,1+0j);end=2*np.pi/np.sqrt(5)
for t in np.linspace(0,end,9):
 close(abs(evolve(t)[1])**2,4/5*np.sin(np.sqrt(5)*t/2)**2)
 numeric=np.sqrt(2)*quad(lambda s:abs(evolve(s)[1]),0,t,epsabs=1e-12)[0];close(numeric,4*np.sqrt(2)/5*(1-np.cos(np.sqrt(5)*t/2)))
false_top=float(abs(evolve(end)[1])**2);false_distance=float(np.sqrt(1-np.exp(-4*np.sin(np.pi/np.sqrt(5))**2)));assert false_top<1e-27 and false_distance>.98
I=np.eye(2);X=np.array([[0,1],[1,0]]);Z=np.diag([1.,-1.])
def family(p):
 a,b=np.sqrt(p),np.sqrt(1-p);return [np.array([[a,b],[0,0]]),np.array([[0,0],[b,a]])],np.ones(2)/np.sqrt(2),np.ones(2)/(a+b)
def dense(as_,left,right,n):
 out=[]
 for bits in product([0,1],repeat=n):
  row=left.copy()
  for bit in bits:row=row@as_[bit]
  out.append(row@right)
 return np.array(out)
for p in [0.,.25,.5,.75,1.]:
 as_,left,right=family(p);second=2*p-1;ell=I/2+np.sqrt(p*(1-p))*X;transfer=lambda x:sum(a@x@a.conj().T for a in as_);insert=lambda x:as_[0]@x@as_[0].T-as_[1]@x@as_[1].T
 e=sum(np.kron(a.conj(),a)for a in as_);close(np.poly(e),np.poly([1,second,0,0]));close(transfer(I),I);close(transfer(Z),second*Z);close(sum(a.T@ell@a for a in as_),ell);close(insert(I),Z);close(insert(Z),second*I)
 for n in range(2,8):
  vector=dense(as_,left,right,n);close(np.vdot(vector,vector),1);expected=[]
  for bits in product([0,1],repeat=n):expected.append(np.sqrt(.5*np.prod([p if a==b else 1-p for a,b in zip(bits,bits[1:])])))
  close(vector,expected)
  probabilities=abs(vector)**2;bits=np.array(list(product([0,1],repeat=n)));signs=1-2*bits;close(probabilities@signs,np.zeros(n))
  for i,j in product(range(n),repeat=2):
   if i>=j:continue
   close(probabilities@(signs[:,i]*signs[:,j]),second**(j-i))
  for g in [np.diag([3.,1.]),np.array([[2,1+1j],[0,.5]],complex)]:
   inverse=np.linalg.inv(g);changed=[inverse@a@g for a in as_];close(dense(changed,left@g,inverse@right,n),vector);counts['gaugeStateComparisons']+=1
  counts['finiteMpsStates']+=1
 if 0<p<1:
  projection=np.outer(I.reshape(-1,order='F'),ell.T.reshape(-1,order='F'))
  for g,r in product([np.diag([3.,1.]),np.array([[2,1+1j],[0,.5]],complex)],[1,2,5,10]):
   inverse=np.linalg.inv(g);changed=[inverse@a@g for a in as_];ep=sum(np.kron(a.conj(),a)for a in changed);s=np.kron(g.conj(),g);si=np.linalg.inv(s);pp=si@projection@s;close(ep,si@e@s);close(np.linalg.cond(s),np.linalg.cond(g)**2)
   lhs=np.linalg.norm(np.linalg.matrix_power(ep,r)-pp,2);rhs=np.linalg.cond(s)*np.linalg.norm(np.linalg.matrix_power(e,r)-projection,2);assert lhs<=rhs+3e-10;counts['gaugePowerComparisons']+=1
for lam,r in product([.5,-.5,.3+.2j],range(1,9)):
 nilpotent=np.array([[0,1],[0,0]],complex);close(np.linalg.matrix_power(lam*I+nilpotent,r),lam**r*I+r*lam**(r-1)*nilpotent);counts['jordanComparisons']+=1
# Exact algebraic angles and transfer characteristic polynomials.
for v in [sp.Rational(0),sp.Rational(1,4),sp.Rational(1,2),sp.Rational(1)]:
 cosine,sine=(1-v*v)/(1+v*v),2*v/(1+v*v);assert sp.simplify(cosine**2+sine**2)==1;assert sp.Rational(4,5)*(1-cosine)==8*v*v/(5*(1+v*v))
assert sp.sin(sp.pi)==0
for p in [sp.Rational(1,4),sp.Rational(1,2),sp.Rational(3,4),sp.Rational(1)]:
 a,b=sp.sqrt(p),sp.sqrt(1-p);as_=[sp.Matrix([[a,b],[0,0]]),sp.Matrix([[0,0],[b,a]])];e=sum((sp.kronecker_product(v,v)for v in as_),sp.zeros(4));z=sp.Symbol('z');assert sp.expand((z*sp.eye(4)-e).det()-z*z*(z-1)*(z-(2*p-1)))==0
print(json.dumps(dict(status='passed',**counts,maxMatrixResidual=maximum,falseReturnNumericalTop=false_top,falseReturnTraceDistance=false_distance,exactFiniteReturn=True,exactAngleCases=4,exactTransferCases=4)))
