"""Independent centered Decimal moments at the actual binary64 inputs.
No TypeScript code or interval/pair-variance algorithm is imported.
Recomputes at +80 decimal digits before writing the reproducible fixture.
"""
from decimal import Decimal as D, localcontext
from itertools import product
from pathlib import Path
import json, math

cases = []
def add(name, K=0., h1=0., h2=0., a1=1., a2=1., digits=400):
    cases.append(dict(id=name, parameters=dict(couplingK=K, fieldOne=h1, fieldTwo=h2, probeOne=a1, probeTwo=a2), precisionDigits=digits))

add('uniform-four-states')
add('ideal-ferro-actual-rounded-log', math.log(3)/2)
add('ideal-antiferro-actual-rounded-log', -math.log(3)/2)
add('unequal-fields', .7, -.4, .9, 2., -1.)
add('opposite-fields', 2., .8, -.8, 1., -1.)
add('both-fields-reversed', .7, .4, -.9, 2., -1.)
add('fields-and-probe-swapped', .7, .9, -.4, -1., 2.)
add('independent-biased', 0., 8., -8.)
add('strong-ferro-positive-fields', 8., 8., 8.)
add('strong-ferro-opposite-fields', 8., 8., -8., 1., -1.)
add('strong-antiferro', -8., 8., -8., 4., -4.)
add('strong-antiferro-opposite-probe', -8., 8., -8., 4., 4.)
add('zero-probe', .7, .3, -.9, 0., 0.)
add('scaled-probe', .7, -.4, .9, 4., -2.)
add('near-cancelled-mean-probe', 0., .5, .5, 1., math.nextafter(-1., 0.))
tiny = float.fromhex('0x0.0000000000001p-1022')
add('minimum-positive-coupling-biased', tiny, .8, -.4, digits=900)
add('minimum-negative-coupling-biased', -tiny, .8, -.4, digits=900)
add('minimum-field-independent', 0., tiny, 0., digits=900)
add('both-minimum-fields-ferro', 2., tiny, -tiny, digits=900)
add('subnormal-probe-variance', .7, -.4, .9, tiny, -tiny, digits=900)
add('minimum-coupling-zero-fields', tiny, 0., 0., digits=900)

spins = list(product([-1, 1], repeat=2))
def reference(p, digits):
    with localcontext() as c:
        c.prec = digits
        K,h1,h2,a1,a2 = [D.from_float(float(p[k])) for k in ['couplingK','fieldOne','fieldTwo','probeOne','probeTwo']]
        scores = [K*x*y+h1*x+h2*y for x,y in spins]
        w = [s.exp() for s in scores]; W = sum(w); prob = [v/W for v in w]
        means = [sum(p*xy[j] for p,xy in zip(prob,spins)) for j in [0,1]]
        cov = [[sum(p*(xy[i]-means[i])*(xy[j]-means[j]) for p,xy in zip(prob,spins)) for j in [0,1]] for i in [0,1]]
        # Independence is an exact symbolic identity, not Decimal residuals.
        if not K: cov[0][1] = cov[1][0] = D(0)
        if not h1 and not h2: means = [D(0), D(0)]
        if not K and not h1: means[0] = D(0)
        if not K and not h2: means[1] = D(0)
        A = [a1*x+a2*y for x,y in spins]; meanA = sum(p*v for p,v in zip(prob,A))
        if (not h1 and not h2) or (h1 == h2 and a1 == -a2): meanA = D(0)
        pair = sum(p*x*y for p,(x,y) in zip(prob,spins))
        if not K and (not h1 or not h2): pair = D(0)
        total = [D(x+y) for x,y in spins]; meanTotal = sum(p*v for p,v in zip(prob,total))
        result = dict(means=means, pairMean=pair, covariance=cov,
                      covarianceDeterminant=cov[0][0]*cov[1][1]-cov[0][1]**2,
                      logPartition=W.ln(), probeMean=meanA,
                      probeVariance=sum(p*(v-meanA)**2 for p,v in zip(prob,A)),
                      commonFieldResponsePerSpin=sum(p*(v-meanTotal)**2 for p,v in zip(prob,total))/2,
                      probabilities=prob, logProbabilities=[p.ln() for p in prob])
        def normalize(v):
            if isinstance(v,list):return [normalize(x) for x in v]
            return '0' if not v else str(v)
        return {k:normalize(v) for k,v in result.items()}

def walk(v):
    if isinstance(v,dict):
        for x in v.values():yield from walk(x)
    elif isinstance(v,list):
        for x in v:yield from walk(x)
    else:yield D(v)

for case in cases:
    digits = case['precisionDigits']; low=reference(case['parameters'],digits); high=reference(case['parameters'],digits+80)
    for a,b in zip(walk(low),walk(high)):
        assert abs(a-b) <= abs(b)*D('1e-180')+D(10)**(-digits+100), (case['id'],a,b)
    case['confirmationPrecisionDigits']=digits+80; case['reference']=high
out=Path('scripts/fixtures/binary-correlations-oracle.json')
out.write_text(json.dumps(dict(scope='independent actual-binary Decimal centered moments',cases=cases),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),observations=sum(len(list(walk(c['reference']))) for c in cases),confirmationExtraDigits=80)))
