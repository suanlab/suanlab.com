"""Independent exact coherent series + SymPy finite matrices + mpmath gamma tails."""
import importlib.util
import json
import random
from pathlib import Path
import sympy as S
import mpmath as mp
spec=importlib.util.spec_from_file_location('boson_reference',Path(__file__).with_name('generate-boson-fock-references.py'));ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
mp.mp.dps=250
rv,cv,vec,ex=ref.rv,ref.cv,ref.vec,ref.ex

def decimal(v):return '0' if v==0 else 'decimal:'+mp.nstr(v,245)
def mp_value(v):return mp.mpf(str(S.N(v,255)))
def interval_complex(v):return dict(real=decimal(mp_value(S.re(v))),imaginary=decimal(mp_value(S.im(v))))
def prep(m,c0,c1=0):
 p={k:v for k,v in ref.prep(m,c0,c1).items() if not k.startswith('state')}
 p.update(alphaReal0=1,alphaImag0=0,alphaDenominator0=1,alphaReal1=0,alphaImag1=0,alphaDenominator1=1)
 return p

def reference(p):
 m=p['modes'];cuts=[p['cutoff0']]+([p['cutoff1']] if m==2 else [])
 alphas=[S.Rational(p[f'alphaReal{i}'],p[f'alphaDenominator{i}'])+S.I*S.Rational(p[f'alphaImag{i}'],p[f'alphaDenominator{i}']) for i in range(m)];mus=[S.expand(S.conjugate(a)*a) for a in alphas]
 local=[S.Matrix([S.expand(a**n/S.sqrt(S.factorial(n))) for n in range(c+1)]) for a,c in zip(alphas,cuts)];weights=[[mu**n/S.factorial(n) for n in range(c+1)] for mu,c in zip(mus,cuts)];series=[sum(w) for w in weights];state=local[0] if m==1 else S.kronecker_product(*local);N=S.prod(series);total=sum(mus)
 controls={k:v for k,v in ref.prep(m,*([p['cutoff0'],p['cutoff1']] if m==2 else [p['cutoff0']])).items()};controls.update({k:v for k,v in p.items() if not k.startswith('alpha')})
 finite=ref.reference(controls,state)
 local_keeps=[];permode=[]
 for i,(mu,c0,alpha,z,Z,w) in enumerate(zip(mus,cuts,alphas,local,series,weights)):
  x=mp_value(mu);tail=mp.gammainc(c0+1,0,x)/mp.factorial(c0);keep=1-tail;local_keeps.append(keep);top=w[-1]/Z
  A=S.zeros(c0+1)
  for n in range(1,c0+1):A[n-1,n]=S.sqrt(n)
  residual=ex(A*z-alpha*z);norm2=S.simplify(ref.norm(residual)/Z);assert S.simplify(norm2-mu*top)==0
  factor=mp.exp(-x/2)
  permode.append(dict(mode=i,cutoff=c0,alpha=cv(alpha),mu=str(mu),rawCoefficients=vec(z),rawSeriesNorm=str(Z),infiniteState=dict(representation='exp(-mu/2)*sum(alpha^n/sqrt(n!),n>=0)',meanNumber=str(mu),numberVariance=str(mu),xMean=rv(S.sqrt(2)*S.re(alpha)),pMean=rv(S.sqrt(2)*S.im(alpha)),xVariance='1/2',pVariance='1/2'),unnormalizedProjection=dict(amplitudes=[interval_complex(v*mp_value(S.exp(-mu/2))) for v in z],probabilities=[decimal(mp.exp(-x)*mp_value(v)) for v in w],retainedProbability=decimal(keep),discardedProbability=decimal(tail)),normalizedProjection=dict(amplitudes=ref.normalized(z,Z),probabilities=[str(v/Z) for v in w],topProbability=str(top),finiteCommutatorExpectation=str(1-(c0+1)*top),annihilationEigenResidual=ref.normalized(residual,Z),annihilationEigenResidualSquared=rv(norm2)),exactResiduals=dict(coefficientSeriesNorm='0',eigenResidualAccount=rv(S.Integer(0)))))
 retained=mp.exp(-mp_value(total))*mp_value(N);tail=1-retained;overlap=mp.sqrt(retained)
 coef=lambda key:S.Rational(p[key],p['hDenominator'])
 E=coef('offset')
 for i in range(m):E+=coef(f'onsite{i}')*mus[i]+coef(f'selfDensity{i}')*mus[i]**2/2+2*S.re((coef(f'driveReal{i}')+S.I*coef(f'driveImag{i}'))*S.conjugate(alphas[i]))
 if m==2:E+=coef('crossDensity')*mus[0]*mus[1]+2*S.re((coef('hoppingReal')+S.I*coef('hoppingImag'))*S.conjugate(alphas[0])*alphas[1])
 finite.update(normalizationApplied=True,fidelitySquaredToInfinite=decimal(retained),overlapWithInfinite=decimal(overlap),distanceSquaredToInfinite=decimal(2*(1-overlap)))
 factor=S.exp(-total/2)
 return dict(preparationKind='product-coherent-rectangular-cutoff',modes=m,cutoffs=cuts,totalMu=str(total),perMode=permode,exactRawCoefficients=vec(state),rawSeriesNorm=str(N),infiniteState=dict(representation='tensor-product-of-infinite-coherent-series',normalized=True,meanTotalNumber=str(total),varianceTotalNumber=str(total),energyForSameNormallyOrderedHamiltonian=rv(E),noGroundCertificate=True),unnormalizedProjection=dict(normalizationApplied=False,amplitudes=[interval_complex(S.N(v*factor,255)) for v in state],normSquared=decimal(retained),distanceSquaredToInfinite=decimal(tail)),normalizedProjection=finite,truncation=dict(retainedProbability=decimal(retained),discardedProbability=decimal(tail),productOfModeRetained=decimal(mp.fprod(local_keeps))),exactResiduals=dict(coefficientSeriesNorm='0'))

cases=[]
def save(label,p):
 cases.append(dict(label=label,input=dict(p),expected=reference(p)));print('Verified '+label,flush=True)
rng=random.Random(20261109)
for m,c0,c1 in [(1,c,0) for c in range(4)]+[(2,a,b) for a in range(4) for b in range(4)]:
 p=prep(m,c0,c1);p['hDenominator']=7;p['offset']=rng.randint(-3,3)
 for i in range(m):
  p[f'alphaReal{i}']=rng.randint(-3,3);p[f'alphaImag{i}']=rng.randint(-3,3);p[f'alphaDenominator{i}']=3
  for prefix in ['onsite','selfDensity','driveReal','driveImag']:p[f'{prefix}{i}']=rng.randint(-3,3)
 if m==2:
  for key in ['hoppingReal','hoppingImag','crossDensity']:p[key]=rng.randint(-3,3)
 p['leftMode']=rng.randrange(m);p['rightMode']=rng.randrange(m);p['leftCreation']=rng.randrange(2);p['rightCreation']=rng.randrange(2)
 save(f'general coherent {m} modes cutoff{c0}-{c1}',p)
for c0 in range(4):
 p=prep(1,c0);p['alphaReal0']=0;save(f'vacuum coherent cutoff{c0}',p)
p=prep(1,3);save('unit real alpha cutoff3',p)
p=prep(1,3);p.update(alphaReal0=0,alphaImag0=1,driveImag0=1);save('unit imaginary alpha drive',p)
p=prep(1,3);p.update(alphaReal0=4,onsite0=1);save('maximum mu16 one mode',p)
p=prep(2,3,2);p.update(alphaReal0=4,alphaReal1=0,alphaImag1=-4,hoppingImag=2,crossDensity=-1);save('maximum total mu32 unequal cutoffs',p)
p=prep(1,3);p.update(alphaReal0=1,alphaDenominator0=1000000);save('tiny mu positive Poisson tail',p)
p=prep(2,3,3);p.update(alphaReal0=1,alphaDenominator0=1000000,alphaImag1=1,alphaDenominator1=1000000);save('two tiny modes retain positive joint tail',p)
p=prep(1,0);p.update(alphaReal0=3,alphaImag0=4,alphaDenominator0=2);save('cutoff0 differs from coherent state',p)
p=prep(2,3,3);p.update(alphaReal0=600000,alphaImag0=800000,alphaDenominator0=250000,alphaReal1=-600000,alphaImag1=800000,alphaDenominator1=250000)
for key in ['offset','onsite0','onsite1','selfDensity0','selfDensity1','driveReal0','driveImag0','driveReal1','driveImag1','hoppingReal','hoppingImag','crossDensity']:p[key]=1000000
save('maximum coefficient bounds coherent mu16 pair',p)
output=Path('scripts/fixtures/boson-coherent-reference.json');output.write_text(json.dumps(dict(method='independent SymPy coherent series and finite matrices;mpmath250dps incomplete gamma tail and exponential',seed=20261109,cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),output=str(output))),flush=True)
