"""Independent SymPy local ladder/Kronecker reference; no product imports."""
import json
import random
from functools import lru_cache
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def prep(modes,c0,c1=0):
 p=dict(modes=modes,cutoff0=c0,cutoff1=c1,hDenominator=1,offset=0)
 p.update({key:0 for key in ['onsite0','onsite1','selfDensity0','selfDensity1','driveReal0','driveImag0','driveReal1','driveImag1','hoppingReal','hoppingImag','crossDensity','leftMode','rightMode','leftCreation']})
 p['rightCreation']=1
 p.update({f'state{part}{i}':0 for i in range(16) for part in ['Real','Imag']});p['stateReal0']=1
 return p

@lru_cache(None)
def rv(x):
 x=S.expand(S.radsimp(S.sympify(x)));coeff=[x.coeff(rad) for rad in [S.sqrt(2),S.sqrt(3),S.sqrt(6)]]
 a=S.simplify(x-sum(t*r for t,r in zip(coeff,[S.sqrt(2),S.sqrt(3),S.sqrt(6)])))
 values=[a]+coeff
 assert all(v.is_Rational for v in values),(x,values)
 value=mp.mpf(str(S.N(x,255)))
 return dict(coefficients=[str(v) for v in values],exactSign=int(S.sign(x)),observation=str(a) if all(v==0 for v in coeff) else 'decimal:'+mp.nstr(value,245))

def cv(x):
 x=S.expand(x);return dict(real=rv(S.re(x)),imaginary=rv(S.im(x)))
def matrix(a):return [[cv(a[i,j]) for j in range(a.cols)] for i in range(a.rows)]
def vec(a):return [cv(x) for x in a]
def ex(a):return a.applyfunc(S.expand)
def norm(a):return S.expand((a.conjugate().T*a)[0])
def frob(a):return S.expand(sum(S.expand(S.conjugate(v)*v) for v in a))
def normalized(z,N):
 def coordinate(v):return '0' if v==0 else 'decimal:'+mp.nstr(mp.mpf(str(S.N(v,255)))/mp.sqrt(mp.mpf(str(S.N(N,255)))),245)
 return [dict(real=coordinate(S.re(v)),imaginary=coordinate(S.im(v))) for v in z]

def reference(p, prepared_state=None):
 m=p['modes'];cuts=[p['cutoff0']]+([p['cutoff1']] if m==2 else []);dims=[c+1 for c in cuts];d=S.prod(dims)
 occ=[[i] for i in range(d)] if m==1 else [[i//dims[1],i%dims[1]] for i in range(d)]
 local=[]
 for c0 in cuts:
  a=S.zeros(c0+1)
  for n in range(1,c0+1):a[n-1,n]=S.sqrt(n)
  local.append(a)
 A=local if m==1 else [S.kronecker_product(local[0],S.eye(dims[1])),S.kronecker_product(S.eye(dims[0]),local[1])]
 C=[a.conjugate().T for a in A];numbers=[c*a for c,a in zip(C,A)];I=S.eye(d)
 z=prepared_state if prepared_state is not None else S.Matrix([p[f'stateReal{i}']+S.I*p[f'stateImag{i}'] for i in range(d)]);N=norm(z)
 expectation=lambda a:S.expand((z.conjugate().T*a*z)[0]/N)
 coefficient=lambda key:S.Rational(p[key],p['hDenominator'])
 terms=[]
 def put(kind,indices,coef,a,paired=False):
  if coef!=0:terms.append((kind,indices,coef,a,paired))
 put('offset',[],coefficient('offset'),I)
 for i in range(m):
  put('onsite',[i],coefficient(f'onsite{i}'),numbers[i])
  put('selfDensity',[i],coefficient(f'selfDensity{i}'),numbers[i]*(numbers[i]-I)/2)
  put('drive',[i],coefficient(f'driveReal{i}')+S.I*coefficient(f'driveImag{i}'),C[i],True)
 if m==2:
  put('hopping',[0,1],coefficient('hoppingReal')+S.I*coefficient('hoppingImag'),C[0]*A[1],True)
  put('crossDensity',[0,1],coefficient('crossDensity'),numbers[0]*numbers[1])
 H=S.zeros(d)
 for _,_,coef,a,paired in terms:H+=coef*a+(S.conjugate(coef)*a.conjugate().T if paired else S.zeros(d))
 H=ex(H);action=ex(H*z);E=expectation(H);E2=expectation(ex(H*H));var=S.expand(E2-E*E);res=ex(action-E*z)
 assert S.simplify(var-norm(res)/N)==0
 rho=ex(z*z.conjugate().T/N)
 total=[sum(v) for v in occ];TN=S.diag(*total);tn=expectation(TN);tv=S.expand(expectation(TN*TN)-tn*tn);tc=frob(ex(H*TN-TN*H))
 commutation=[]
 for i in range(m):
  for j in range(m):
   mixed=A[i]*C[j]-C[j]*A[i]
   expected=S.diag(*[1-(cuts[i]+1)*int(ns[i]==cuts[i]) for ns in occ]) if i==j else S.zeros(d)
   assert mixed==expected
   commutation.append(dict(leftMode=i,rightMode=j,mixed=matrix(mixed),projectedIdentityResidualFrobeniusSquared=rv(frob(mixed-expected)),infiniteCcrResidualFrobeniusSquared=rv(frob(mixed-(I if i==j else S.zeros(d)))),annihilationCommutatorFrobeniusSquared=rv(frob(A[i]*A[j]-A[j]*A[i])),creationCommutatorFrobeniusSquared=rv(frob(C[i]*C[j]-C[j]*C[i])),trace=cv(S.trace(mixed))))
 locals=[]
 for i,c0 in enumerate(cuts):
  top=S.diag(*[int(ns[i]==c0) for ns in occ]);tp=expectation(top);comm=expectation(A[i]*C[i]-C[i]*A[i]);X=(A[i]+C[i])/S.sqrt(2);P=(A[i]-C[i])/(S.I*S.sqrt(2));xm=expectation(X);pm=expectation(P)
  xv=S.expand(expectation(X*X)-xm*xm);pv=S.expand(expectation(P*P)-pm*pm);cov=S.expand(expectation((X*P+P*X)/2)-xm*pm);bound=comm*comm/4;det=S.expand(xv*pv-cov*cov);slack=S.expand(det-bound)
  assert S.sign(slack)>=0
  reduced=S.zeros(c0+1)
  for a,na in enumerate(occ):
   for b,nb in enumerate(occ):
    if all(na[k]==nb[k] for k in range(m) if k!=i):reduced[na[i],nb[i]]+=rho[a,b]
  reduced=ex(reduced)
  locals.append(dict(mode=i,cutoff=c0,density=matrix(reduced),purity=rv(S.trace(reduced*reduced)),occupation=rv(expectation(numbers[i])),annihilationMean=cv(expectation(A[i])),topProbability=rv(tp),finiteCommutatorExpectation=rv(comm),boundaryCorrection=rv((c0+1)*tp),quadratures=dict(xMean=rv(xm),pMean=rv(pm),xVariance=rv(xv),pVariance=rv(pv),covariance=rv(cov),robertsonBound=rv(bound),covarianceDeterminant=rv(det),uncertaintySlack=rv(slack))))
 L=(C if p['leftCreation'] else A)[p['leftMode']];R=(C if p['rightCreation'] else A)[p['rightMode']];forward=ex(L*R*z);reverse=ex(R*L*z)
 diagonal=all(H[i,j]==0 for i in range(d) for j in range(d) if i!=j);spectrum=None
 if diagonal:
  groups={}
  for i in range(d):groups.setdefault(H[i,i],[]).append(i)
  levels=sorted(groups,key=lambda v:float(v));g=levels[0];indices=groups[g];gp=S.expand(sum(S.conjugate(z[i])*z[i] for i in indices)/N);gap=levels[1]-g if len(levels)>1 else None;ratio=S.expand((E-g)/gap) if gap is not None else None
  spectrum=dict(scope='finite-projected-Fock-space-only',levels=[dict(energy=rv(e),basisIndices=groups[e],multiplicity=len(groups[e]),probability=rv(S.expand(sum(S.conjugate(z[i])*z[i] for i in groups[e])/N))) for e in levels],groundEnergy=rv(g),groundMultiplicity=len(indices),groundProbability=rv(gp),leakage=rv(1-gp),excitationGap=None if gap is None else rv(gap),excessEnergyToGapRatio=None if ratio is None else rv(ratio),certifiedLeakageUpperBound=None if ratio is None else rv(min(1,ratio)),selectedStateGround=bool(gp==1))
 return dict(modes=m,cutoffs=cuts,localDimensions=dims,dimension=int(d),basisOccupations=occ,state=vec(z),normalizedState=normalized(z,N),rawNormSquared=rv(N),hamiltonian=matrix(H),hamiltonianAction=vec(action),energy=rv(E),secondMoment=rv(E2),variance=rv(var),residual=vec(res),normalizedResidual=normalized(res,N),terms=[dict(type=kind,modes=indices,coefficient=cv(coef),expectation=cv(expectation(a)),includesAdjoint=paired,energyContribution=rv(S.expand(2*S.re(coef*expectation(a)) if paired else coef*expectation(a)))) for kind,indices,coef,a,paired in terms],density=matrix(rho),localStates=locals,gamma=[[cv(expectation(C[i]*A[j])) for j in range(m)] for i in range(m)],numberDistribution=[rv(S.expand(sum(S.conjugate(z[i])*z[i] for i in range(d) if total[i]==n)/N)) for n in range(sum(cuts)+1)],totalNumber=dict(expectation=rv(tn),variance=rv(tv),selectedStateSharp=bool(tv==0),conserved=bool(tc==0),commutatorFrobeniusSquared=rv(tc)),operators=[dict(mode=i,annihilation=matrix(a)) for i,a in enumerate(A)],commutation=commutation,probe=dict(forward=vec(forward),reverse=vec(reverse),commutatorAction=vec(forward-reverse),forwardOnNormalizedInput=normalized(forward,N),reverseOnNormalizedInput=normalized(reverse,N),forwardSquaredNormOnNormalizedInput=rv(norm(forward)/N),reverseSquaredNormOnNormalizedInput=rv(norm(reverse)/N),commutator=matrix(L*R-R*L)),spectralMode='finite-occupation-basis-enumeration' if diagonal else 'not-computed-for-nondiagonal-hamiltonian',diagonalSpectrum=spectrum,classification=dict(exactEigenstate=bool(var==0),selectedStateFiniteCutoffGround=None if spectrum is None else spectrum['selectedStateGround']))

if __name__ == '__main__':
 cases=[]
 def save(label,p):
  expected=reference(p);cases.append(dict(label=label,input=dict(p),expected=expected));print('Verified '+label,flush=True)
 rng=random.Random(20261109)
 for m,c0,c1 in [(1,c,0) for c in range(4)]+[(2,a,b) for a in range(4) for b in range(4)]:
  p=prep(m,c0,c1);p['hDenominator']=7;p['offset']=rng.randint(-3,3)
  for i in range(m):
   for prefix in ['onsite','selfDensity','driveReal','driveImag']:p[f'{prefix}{i}']=rng.randint(-3,3)
  if m==2:
   for key in ['hoppingReal','hoppingImag','crossDensity']:p[key]=rng.randint(-3,3)
  p['leftMode']=rng.randrange(m);p['rightMode']=rng.randrange(m);p['leftCreation']=rng.randrange(2);p['rightCreation']=rng.randrange(2)
  for i in range((c0+1)*(c1+1)):p[f'stateReal{i}']=rng.randint(-3,3);p[f'stateImag{i}']=rng.randint(-3,3)
  if not any(p[f'state{part}{i}'] for i in range((c0+1)*(c1+1)) for part in ['Real','Imag']):p['stateReal0']=1
  save(f'general complex {m} modes cutoff{c0}-{c1}',p)
 p=prep(1,3);save('vacuum cutoff3 uncertainty',p)
 p=prep(1,3);p.update(stateReal0=0,stateReal3=1);save('top occupation reverses finite commutator',p)
 p=prep(1,3);p['stateReal3']=1;save('vacuum top superposition finite commutator minus1',p)
 p=prep(1,2);p.update(stateReal0=0,stateReal1=1);save('number1 raises with sqrt2',p)
 p=prep(1,3);p.update(stateReal0=0,stateReal2=1);save('number2 raises with sqrt3',p)
 p=prep(1,0);p.update(driveReal0=2,driveImag0=3,onsite0=1);save('cutoff0 has vanishing quadratures',p)
 p=prep(2,3,2);p.update(hoppingReal=1,stateReal0=0,stateReal8=1);save('boson hopping ladder enhancement sqrt6',p)
 p=prep(2,3,3);p.update(hoppingImag=1,stateReal0=0,stateReal1=1,stateImag4=1);save('imaginary hopping phase orientation',p)
 p=prep(1,3);p.update(driveImag0=1,stateReal0=1,stateImag1=1);save('complex drive and p orientation',p)
 p=prep(2,3,3);p.update(stateReal15=1,onsite0=1,onsite1=2,selfDensity0=-1,crossDensity=3);save('diagonal interacting finite spectrum',p)
 p=prep(1,3);p.update(stateReal0=0,stateReal3=1,onsite0=-1);save('negative onsite finite ground has no infinite ground certificate',p)
 p=prep(1,3);p.update(stateReal0=1000000,stateReal1=1,onsite0=1);save('tiny positive number variance',p)
 p=prep(2,3,3)
 for key in ['offset','onsite0','onsite1','selfDensity0','selfDensity1','driveReal0','driveImag0','driveReal1','driveImag1','hoppingReal','hoppingImag','crossDensity']:p[key]=1000000
 for i in range(16):p[f'stateReal{i}']=1000000;p[f'stateImag{i}']=-1000000
 save('maximum complex integer preparation',p)
 output=Path('scripts/fixtures/boson-fock-reference.json');output.write_text(json.dumps(dict(method='independent SymPy local ladder/Kronecker,H squared,partial trace,commutator and mpmath250dps',seed=20261109,cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),output=str(output))),flush=True)
