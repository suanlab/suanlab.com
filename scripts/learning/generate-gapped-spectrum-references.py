"""Independent per-microstate enumeration and conditional entropy decomposition.

Ordinary/finite-ratio limits sum individual states for U and variance. Entropy
above the ground manifold uses binary mixing entropy plus the conditional state
counts, unlike production's positive logZ+xq expression. Extreme ratios use
explicit asymptotic logarithmic bounds rather than claiming direct enumeration.
"""
import argparse,json,math,random,sys
from pathlib import Path
import mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/gapped-spectrum-independent.json');out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(26110621);kfloat=1.380649e-23
base=dict(copies=1,groundDegeneracy=1,upperDegeneracy=1,gapTemperature=300.,temperature=300.)

def observation_log(log):
 if log is None:return dict(value='0',sign=0,logMagnitude=None,logNegativeLogMagnitude=None,logMagnitudeStatus='not-applicable',logNegativeLogMagnitudeStatus='not-applicable')
 lo=mp.log(mp.mpf(math.nextafter(0.,1.))/2);hi=mp.log(mp.mpf(sys.float_info.max))
 linear=None if log<lo or log>hi else float(mp.exp(log))
 if linear==0 or linear is not None and not math.isfinite(linear):linear=None
 lf=float(log);finite=math.isfinite(lf) and not(lf==0 and log!=0)
 return dict(value=None if linear is None else repr(linear),sign=1,logMagnitude=str(log)if finite else None,
  logNegativeLogMagnitude=None if log>=0 else str(mp.log(-log)),logMagnitudeStatus='finite'if finite else'outside-number-range',logNegativeLogMagnitudeStatus='not-applicable'if log>=0 else'finite')
def observation(v):return observation_log(None if v==0 else mp.log(v))
def case(p,digits,derivatives=False):
 with mp.workdps(digits):
  n,g0,g1=p['copies'],p['groundDegeneracy'],p['upperDegeneracy'];gap,T=mp.mpf(p['gapTemperature']),mp.mpf(p['temperature']);k=mp.mpf(kfloat)
  effective=g0+g1 if gap==0 else g0;res=mp.log(effective);scale=n*k
  expected=dict(gapEnergy=observation(k*gap),residualEntropy=observation(scale*res),residualEntropyPerCopyInKB=observation(res),positiveGapLowTemperatureEntropy=observation(scale*mp.log(g0)),zeroGapThenLowTemperatureEntropy=observation(scale*mp.log(g0+g1)))
  if T==0 or gap==0:
   upper=mp.mpf(g1)/(g0+g1)if gap==0 else mp.mpf(0);ground=1-upper
   expected.update(upperGroupProbability=observation(upper),groundGroupProbability=observation(ground),logPartitionPerCopy=observation(res),meanEnergy=observation(0),meanEnergyPerCopyInKB=observation(0),energyVariance=observation(0),entropy=observation(scale*res),entropyPerCopyInKB=observation(res),excitationEntropy=observation(0),excitationEntropyPerCopyInKB=observation(0),heatCapacity=observation(0),heatCapacityPerCopyInKB=observation(0),entropyTemperatureDerivative=None if T==0 else observation(0))
   partition=effective;method='exact fixed-spectrum equilibrium limit'if T==0 else'exact all-level degeneracy'
  else:
   x=gap/T;y=mp.log(mp.mpf(g1)/g0)-x
   if x>mp.mpf('1e6'):
    # With w=(g1/g0)*exp(-x), |log(q)-log(w)|<=w;
    # other excitation log errors here are bounded by4w. For x>1e6
    # this is far below the working precision. No huge exp(-x) is evaluated.
    lq=y;lC=2*mp.log(x)+lq;le=mp.log(1+x)+lq
    values=dict(upperGroupProbability=lq,groundGroupProbability=mp.mpf(0),meanEnergy=mp.log(scale*gap)+lq,meanEnergyPerCopyInKB=mp.log(gap)+lq,energyVariance=mp.log(n*(k*gap)**2)+lq,heatCapacity=mp.log(scale)+lC,heatCapacityPerCopyInKB=lC,entropyTemperatureDerivative=mp.log(scale/T)+lC,excitationEntropy=mp.log(scale)+le,excitationEntropyPerCopyInKB=le)
    if g0==1:values.update(logPartitionPerCopy=lq,entropy=mp.log(scale)+le,entropyPerCopyInKB=le)
    else:values.update(logPartitionPerCopy=mp.log(mp.log(g0)),entropy=mp.log(scale*mp.log(g0)),entropyPerCopyInKB=mp.log(mp.log(g0)))
    expected.update({key:observation_log(v)for key,v in values.items()});yf=float(y);expected['groundGroupProbability']=dict(value='1.0',sign=1,logMagnitude=None,logNegativeLogMagnitude=str(y)if math.isfinite(yf)else None,logMagnitudeStatus='outside-number-range',logNegativeLogMagnitudeStatus='finite'if math.isfinite(yf)else'outside-number-range');partition=g0;method='asymptotic logarithmic bounds; log-error bound below4w';error_bound=str(mp.log(4*mp.mpf(g1)/g0)-x)
   else:
    energies=[mp.mpf(0)]*g0+[k*gap]*g1
    def enumerate_at(t):
     weights=[mp.exp(-e/(k*t))for e in energies];Z=sum(weights);probs=[w/Z for w in weights];U=sum(q*e for q,e in zip(probs,energies));var=sum(q*(e-U)**2 for q,e in zip(probs,energies));upper=sum(probs[g0:])
     mixing=-upper*mp.log(upper)-(1-upper)*mp.log1p(-upper);exc=mixing+upper*mp.log(mp.mpf(g1)/g0)
     return U,var,upper,exc,Z
    U,var,upper,exc,Z=enumerate_at(T);C=var/(k*T*T);S=mp.log(g0)+exc
    expected.update(upperGroupProbability=observation(upper),groundGroupProbability=observation_log(mp.log1p(-upper)),logPartitionPerCopy=observation(mp.log(g0)+mp.log1p(mp.mpf(g1)/g0*mp.exp(-x))),meanEnergy=observation(n*U),meanEnergyPerCopyInKB=observation(U/k),energyVariance=observation(n*var),entropy=observation(scale*S),entropyPerCopyInKB=observation(S),excitationEntropy=observation(scale*exc),excitationEntropyPerCopyInKB=observation(exc),heatCapacity=observation(n*C),heatCapacityPerCopyInKB=observation(C/k),entropyTemperatureDerivative=observation(n*C/T))
    partition=float(Z);method='individual-state moments and conditional mixing entropy'
    if derivatives:
     dU=mp.diff(lambda t:enumerate_at(t)[0],T);dS=mp.diff(lambda t:k*(mp.log(g0)+enumerate_at(t)[3]),T)
     assert mp.almosteq(dU,C,rel_eps=mp.mpf('1e-120'));assert mp.almosteq(dS,C/T,rel_eps=mp.mpf('1e-120'))
  row=dict(parameters=p,digits=digits,method=method,temperatureDerivativesChecked=derivatives,partitionFunctionPerCopy=partition,effectiveGroundDegeneracy=effective,groundStateCount=str(effective**n),totalStateCount=str((g0+g1)**n),expected=expected)
  if 'error_bound'in locals():row['logAbsoluteLogErrorBound']=error_bound
  return row
ordinary=[]
fixed=[base,dict(base,copies=4,groundDegeneracy=2,upperDegeneracy=3,gapTemperature=100.,temperature=200.),dict(base,copies=3,groundDegeneracy=16,upperDegeneracy=1,gapTemperature=1.,temperature=800.),dict(base,copies=64,upperDegeneracy=16,gapTemperature=1000.,temperature=800.),dict(base,groundDegeneracy=2,upperDegeneracy=3,gapTemperature=0.),dict(base,groundDegeneracy=2,upperDegeneracy=3,temperature=0.)]
for i,p in enumerate(fixed):ordinary.append(case(p,160,derivatives=i<4))
for i in range(18):ordinary.append(case(dict(copies=rng.randint(1,64),groundDegeneracy=rng.randint(1,16),upperDegeneracy=rng.randint(1,16),gapTemperature=rng.uniform(.1,1000),temperature=rng.uniform(5,800)),160))
small=math.nextafter(0.,1.)
limits=[dict(base,gapTemperature=small,temperature=small),dict(base,gapTemperature=small,temperature=800.),dict(base,gapTemperature=750.,temperature=1.),dict(base,gapTemperature=1000.,temperature=1.),dict(base,groundDegeneracy=2,upperDegeneracy=3,gapTemperature=1000.,temperature=1.),dict(base,gapTemperature=small,temperature=0.),dict(base,gapTemperature=0.,temperature=small),dict(base,gapTemperature=0.,temperature=0.),dict(base,copies=64,groundDegeneracy=16,upperDegeneracy=16,temperature=0.),dict(base,gapTemperature=small,temperature=1.),dict(base,temperature=0.),dict(base,groundDegeneracy=2,upperDegeneracy=3,temperature=0.),dict(base,groundDegeneracy=16,upperDegeneracy=1,gapTemperature=small,temperature=800.),dict(base,groundDegeneracy=1,upperDegeneracy=16,gapTemperature=1e-200,temperature=800.),dict(base,copies=64,gapTemperature=750.,temperature=1.),dict(base,copies=64,gapTemperature=small,temperature=small)]
edge=[case(p,1000)for p in limits]
extreme=[case(p,1000)for p in [dict(base,gapTemperature=1000.,temperature=1e-305),dict(base,gapTemperature=1000.,temperature=1e-306),dict(base,gapTemperature=300.,temperature=small),dict(base,copies=64,groundDegeneracy=16,gapTemperature=1000.,temperature=small),dict(base,gapTemperature=1.,temperature=small)]]
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(method='individual microstates, conditional entropy and separate extreme log bounds',boltzmannConstantBinary=kfloat,ordinary=ordinary,limits=edge,extreme=extreme),indent=2)+'\n');print(f'Saved24 ordinary,16 finite-ratio limits and5 extreme log-bound cases to{out}')
