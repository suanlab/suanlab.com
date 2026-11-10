"""Exact rational oracle for an explicitly defined port budget, not a trajectory.

Construct cold heat from the entropy budget, then work from energy balance.
fractions.Fraction arithmetic is independent of the production binary arithmetic.
"""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/reservoir-cycle-independent.json');out=Path(parser.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(26110620)
base=dict(hotTemperature=600.,coldTemperature=300.,heatHot=1000.,relativeEntropyGeneration=0.)
def observation(v):
 x=mp.mpf(v.numerator)/v.denominator
 return dict(value=str(x),sign=int(mp.sign(x)),logMagnitude=None if x==0 else str(mp.log(abs(x))))
def case(p,digits):
 with mp.workdps(digits):
  th,tc,qh,sigma=[F(p[k])for k in ['hotTemperature','coldTemperature','heatHot','relativeEntropyGeneration']]
  hotEntropy=-qh/th;generation=qh/th*sigma;coldEntropy=generation-hotEntropy;qc=-tc*coldEntropy;work=qh+qc;reversible=qh*(1-tc/th)
  expected=dict(heatHotIntoDevice=qh,heatColdIntoDevice=qc,workByDevice=work,entropyGeneration=generation,hotReservoirEntropyChange=hotEntropy,coldReservoirEntropyChange=coldEntropy,reversibleWork=reversible,workDeficitFromReversible=reversible-work,carnotTemperatureBound=1-tc/th)
  if qh:expected['workFraction']=work/qh
  assert work==qh+qc;assert reversible-work==tc*generation
  regime='no-transfer'if qh==0 else'counterfactual'if generation<0 else'heat-engine'if work>0 else'work-driven-dissipation'if work<0 else'heat-transfer'
  return dict(parameters=p,digits=digits,regime=regime,secondLawAdmissible=generation>=0,expected={k:observation(v)for k,v in expected.items()})
ordinary=[]
for sigma in [0.,.5,2.,-.5]:ordinary.append(case(dict(base,relativeEntropyGeneration=sigma),160))
for i in range(20):
 cold=rng.uniform(250,550);p=dict(hotTemperature=rng.uniform(cold,800),coldTemperature=cold,heatHot=rng.uniform(.01,4000),relativeEntropyGeneration=rng.uniform(-1,4));ordinary.append(case(p,160))
small=math.nextafter(0.,1.)
limits=[dict(base,heatHot=0.,relativeEntropyGeneration=-.5),dict(base,relativeEntropyGeneration=-1.),dict(base,relativeEntropyGeneration=1.),dict(base,relativeEntropyGeneration=math.nextafter(1.,0.)),dict(base,relativeEntropyGeneration=math.nextafter(1.,2.)),dict(base,hotTemperature=300.,relativeEntropyGeneration=0.),dict(base,hotTemperature=300.,relativeEntropyGeneration=small),dict(base,hotTemperature=300.,relativeEntropyGeneration=-small),dict(base,heatHot=small,relativeEntropyGeneration=small),dict(base,heatHot=small,relativeEntropyGeneration=-small),dict(base,relativeEntropyGeneration=small),dict(base,hotTemperature=math.nextafter(300.,math.inf),relativeEntropyGeneration=0.)]
edge=[case(p,1000)for p in limits]
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(method='exact rational reservoir-entropy construction followed by physical energy balance',ordinary=ordinary,limits=edge),indent=2)+'\n');print(f'Saved{len(ordinary)} ordinary and{len(edge)} exact rational limit cases to{out}')
