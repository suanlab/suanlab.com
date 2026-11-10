"""Independent orthonormal square-root ladders and similarity-transformed coordinates.

Production constructs rational polynomial derivative matrices. Here standard
Hermitian angular momentum matrices are built first, then conjugated by the
basis map. SymPy exact moments and mpmath250dps physical coordinates are used.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp
import sympy as s
mp.mp.dps=250

def exact(x):return str(s.cancel(s.simplify(x)))
def dec(x):return 'decimal:'+mp.nstr(mp.mpf(str(s.N(s.re(x),270))),210,min_fixed=0,max_fixed=0)
def cv(x):return dict(real=dec(s.re(x)),imaginary=dec(s.im(x)))
def matrix(a,physical=False):return [[cv(a[i,j])if physical else dict(real=exact(s.re(a[i,j])),imaginary=exact(s.im(a[i,j])))for j in range(a.cols)]for i in range(a.rows)]
def put(n,z,h=(1,1)):
 return dict(twoJ=n,hbarNumerator=h[0],hbarDenominator=h[1],**{k:int(v)for i in range(9)for k,v in [(f'stateReal{i}',s.re(z[i])if i<len(z)else 0),(f'stateImag{i}',s.im(z[i])if i<len(z)else 0)]})
raw=[('spin one unequal metric',put(2,[1,1,1]))]
for n in range(9):
 raw.extend([(f'twoJ {n} highest',put(n,[1]+[0]*n)),(f'twoJ {n} lowest',put(n,[0]*n+[1])),(f'twoJ {n} middle',put(n,[int(k==n//2)for k in range(n+1)])),(f'twoJ {n} coherent X',put(n,[s.binomial(n,k)for k in range(n+1)])),(f'twoJ {n} complex',put(n,[k+1+s.I*(-1)**k*(k+2)for k in range(n+1)]))])
raw.extend([('spin one coherent Y',put(2,[1,2*s.I,-1])),('spin one negative X',put(2,[1,-2,1])),('half spin positive Y',put(1,[1,s.I])),('rational hbar',put(3,[1,2+s.I,3-2*s.I,-1],(5,7))),('small hbar',put(8,list(range(1,10)),(1,1000000))),('large hbar',put(8,list(range(1,10)),(1000000,1))),('large complex raw state',put(8,[1000000-k+s.I*(-1)**k*(999999-k)for k in range(9)])),('global imaginary raw scaling',put(2,[3*s.I,3*s.I,3*s.I]))])
seen=set();specs=[]
for label,p in raw:
 key=json.dumps(p,sort_keys=True)
 if key not in seen:seen.add(key);specs.append((label,p))
case=[];cache={}
for label,p in specs:
 n=p['twoJ'];d=n+1;j=s.Rational(n,2);h=s.Rational(p['hbarNumerator'],p['hbarDenominator']);z=s.Matrix([p[f'stateReal{k}']+s.I*p[f'stateImag{k}']for k in range(d)]);C=[s.binomial(n,k)for k in range(d)];B=s.diag(*[1/s.sqrt(v)for v in C]);G=B.conjugate().T*B;u=B*z;N=s.simplify((u.conjugate().T*u)[0]);euclidean=s.simplify((z.conjugate().T*z)[0]);
 key=(n,h)
 if key not in cache:
  raising=s.zeros(d)
  for k in range(1,d):raising[k-1,k]=h*s.sqrt(k*(n-k+1))
  lowering=raising.conjugate().T;Jz=s.diag(*[h*(j-k)for k in range(d)]);Jx=(raising+lowering)/2;Jy=(raising-lowering)/(2*s.I);casimir=s.simplify(Jx*Jx+Jy*Jy+Jz*Jz);physical={'x':Jx,'y':Jy,'z':Jz,'raising':raising,'lowering':lowering,'casimir':casimir};weighted={name:s.simplify(B.inv()*a*B)for name,a in physical.items()};cache[key]=(physical,weighted)
 physical,weighted=cache[key];probs=[s.simplify(s.conjugate(u[k])*u[k]/N)for k in range(d)];naive=[s.simplify(s.conjugate(z[k])*z[k]/euclidean)for k in range(d)]
 def amplitudes(v,mass):return None if mass==0 else[cv(s.simplify(x/s.sqrt(mass)))for x in v]
 def action(a,v):
  out=s.simplify(a*v);mass=s.simplify((out.conjugate().T*out)[0]);braw=s.simplify(B.inv()*out)
  return dict(weightedNumerators=[dict(real=exact(s.re(x)),imaginary=exact(s.im(x)))for x in braw],actionNormSquared=exact(mass/N),normalizedPhysicalAmplitudes=amplitudes(out,mass),normalizedZProbabilities=None if mass==0 else[exact(s.conjugate(x)*x/mass)for x in out],zeroActionExact=bool(mass==0))
 def chain(a):
  v=u;out=[]
  for k in range(d+1):out.append(dict(applications=k,**action(s.eye(d),v),normActionPower=2*k));v=s.simplify(a*v)
  return out
 moments={}
 for name in ['x','y','z']:
  a=physical[name];mean=s.simplify((u.conjugate().T*a*u)[0]/N);second=s.simplify((u.conjugate().T*a*a*u)[0]/N);moments[name]=dict(mean=dict(real=exact(s.re(mean)),imaginary=exact(s.im(mean))),secondMoment=exact(second),variance=exact(second-mean**2))
 mean_square=s.simplify(sum(s.sympify(m['mean']['real'])**2 for m in moments.values()));variance_sum=s.simplify(sum(s.sympify(m['variance'])for m in moments.values()));cv_value=h*h*j*(j+1)
 expected=dict(dimension=d,j=exact(j),hbar=exact(h),binomialCoefficients=[exact(x)for x in C],metricWeights=[exact(G[k,k])for k in range(d)],weightedRawNormSquared=exact(N),rawEuclideanNormSquared=exact(euclidean),weightedBasisMatrices={name:matrix(a)for name,a in weighted.items()},physicalMatrices={name:matrix(a,True)for name,a in physical.items()},state=dict(normalizedWeightedCoordinates=amplitudes(z,N),physicalAmplitudes=amplitudes(u,N),zProbabilities=[exact(x)for x in probs],naiveEuclideanZProbabilities=[exact(x)for x in naive],weightedMinusNaiveProbabilities=[exact(x-y)for x,y in zip(probs,naive)],spectrum=[dict(magneticQuantumNumber=exact(j-k),spinValue=exact(h*(j-k)),probability=exact(probs[k]))for k in range(d)],moments=moments,casimirMean=exact(cv_value),casimirVariance='0',meanVectorSquared=exact(mean_square),meanVectorDeficit=exact(h*h*j*j-mean_square),sumVariances=exact(variance_sum)),raising=action(physical['raising'],u),lowering=action(physical['lowering'],u),raisingChain=chain(physical['raising']),loweringChain=chain(physical['lowering']),ladderCoefficients=[dict(k=k,magneticQuantumNumber=exact(j-k),raisingSquaredCoefficient=exact(h*h*k*(n-k+1)),loweringSquaredCoefficient=exact(h*h*(n-k)*(k+1)))for k in range(d)],casimirEigenvalue=exact(cv_value),classification=dict(integerJ=bool(n%2==0),polynomialBasisIsOrthonormal=all(x==1 for x in C),ordinaryRaisingAdjointIsLowering=bool(weighted['raising'].conjugate().T==weighted['lowering']),thisStateNaiveHistogramAgrees=all(s.simplify(x-y)==0 for x,y in zip(probs,naive)),rawLadderChainRenormalized=False,raisingAnnihilatesState=bool(physical['raising']*u==s.zeros(d,1)),loweringAnnihilatesState=bool(physical['lowering']*u==s.zeros(d,1))))
 case.append(dict(label=label,input=p,expected=expected));print('prepared '+label,flush=True)
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/spin-ladder-reference.json');args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(method='SymPy standard orthonormal square-root ladder matrices,similarity transform by B=diag(1/sqrt(binomial(n,k))),exact direct moments and actions,250dps/210digit physical coordinates; no production modules or polynomial derivative matrices reused.',cases=case),indent=2)+'\n');print(json.dumps({'preparations':len(case),'twoJValues':sorted(set(c['input']['twoJ']for c in case)),'chainRecords':sum(len(c['expected']['raisingChain'])+len(c['expected']['loweringChain'])for c in case)}))
