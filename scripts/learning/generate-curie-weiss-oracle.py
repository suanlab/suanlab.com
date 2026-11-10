"""Independent Decimal sums and small-N microstate enumeration.
Uses actual binary64 K,b, never ideal exp(round(log(2)))=2.
Does not import the TypeScript interval implementation.
"""
from decimal import Decimal as D, localcontext
from itertools import product
from math import comb, log
from pathlib import Path
import json

cases = []
def add(id, N, K, b=0, precision=400):
    cases.append(dict(id=id, parameters=dict(spinCount=N, couplingK=K, fieldB=b), precisionDigits=precision))

add('one-spin-self-offset', 1, 8)
add('one-spin-biased', 1, 3, .7)
add('two-independent', 2, 0)
add('two-actual-ln2', 2, log(2))
add('two-next-above-ln2', 2, float.fromhex('0x1.62e42fefa39f0p-1'))
add('odd-symmetric', 5, 1.4)
add('small-biased', 7, .8, .35)
add('small-reversed-field', 7, .8, -.35)
add('independent-biased', 32, 0, -.6)
add('finite-critical-parameter', 32, 1)
add('finite-two-peaks', 32, 2)
add('beyond-safe-integer-multiplicity', 64, .4)
add('maximum-support', 128, 1)
add('maximum-coupling-field', 128, 8, 4)
add('maximum-coupling-reversed-field', 128, 8, -4)
tiny = float.fromhex('0x0.0000000000001p-1022')
add('subnormal-field-independent', 3, 0, tiny, 900)
add('subnormal-field-ordered', 128, 2, tiny, 900)
add('subnormal-coupling', 4, tiny, 0, 900)
add('both-subnormal', 4, tiny, -tiny, 900)

def reference(p, digits):
    with localcontext() as c:
        c.prec = digits
        N = p['spinCount']; K = D.from_float(float(p['couplingK'])); b = D.from_float(float(p['fieldB']))
        M = list(range(-N, N+1, 2)); g = [comb(N, i) for i in range(N+1)]
        m = [D(v)/N for v in M]
        score = [K*v*v/(2*N)+b*v for v in M]
        w = [D(count)*a.exp() for count,a in zip(g,score)]
        W = sum(w); prob = [v/W for v in w]; logZ = W.ln()
        meanM = sum(D(v)*r for v,r in zip(M,prob)) if b else D(0)
        mean = meanM/N
        m2 = sum(v*v*r for v,r in zip(m,prob)) if N != 1 else D(1)
        variance = sum((v-mean)**2*r for v,r in zip(m,prob))
        absolute = sum(abs(v)*r for v,r in zip(m,prob)) if N != 1 else D(1)
        meanM2 = sum(D(v*v)*r for v,r in zip(M,prob)) if N != 1 else D(1)
        macroH = -sum(r*r.ln() for r in prob)
        conditionalH = sum(r*D(count).ln() for r,count in zip(prob,g))
        microH = -sum(r*(a-logZ) for r,a in zip(prob,score))
        energy = -sum(r*a for r,a in zip(prob,score))
        pairEnergy = energy+K/2
        if N == 1 and not b: pairEnergy = D(0)
        logscores = [D(count).ln()+a for count,a in zip(g,score)]
        modes = [v for v,a in zip(M,logscores) if a==max(logscores)]
        if N <= 7:
            # Directly enumerate named configurations, without binomial weights.
            configurations = list(product([-1,1], repeat=N))
            brute = [(sum(s), (K*sum(s)**2/(2*N)+b*sum(s)).exp()) for s in configurations]
            bruteZ = sum(v for _,v in brute)
            assert abs((bruteZ-W)/W) < D(10)**(-digits+50)
            for total, target, multiplicity in zip(M,prob,g):
                hits = [v for t,v in brute if t==total]
                assert len(hits)==multiplicity
                assert abs(sum(hits)/bruteZ-target) < D(10)**(-digits+50)
        return dict(meanMagnetization=mean, meanAbsoluteMagnetization=absolute, meanSquareMagnetization=m2,
            magnetizationVariance=variance, fieldSusceptibility=N*variance,
            macroEntropyNats=macroH, conditionalMicrostateEntropyNats=conditionalH, microstateEntropyNats=microH,
            logPartition=logZ, pressurePerSpin=logZ/N, freeEnergyInThermalUnits=-logZ,
            freeEnergyPerSpinInThermalUnits=-logZ/N, meanEnergyInThermalUnits=energy,
            logPartitionWithoutSelfInteraction=logZ-K/2, meanEnergyWithoutSelfInteractionInThermalUnits=pairEnergy,
            logZFieldDerivative=meanM, logZCouplingDerivative=meanM2/(2*N),
            probabilities=prob, logProbabilities=[v.ln() for v in prob], multiplicities=[str(v) for v in g], modes=modes)

def flatten(v, path=()):
    if isinstance(v, dict):
        for k,x in v.items(): yield from flatten(x,path+(k,))
    elif isinstance(v,list):
        for i,x in enumerate(v): yield from flatten(x,path+(i,))
    else: yield path,v
def stringify(v):
    if isinstance(v,dict): return {k:stringify(x) for k,x in v.items()}
    if isinstance(v,list): return [stringify(x) for x in v]
    return str(v) if isinstance(v,D) else v

rows=[]
for case in cases:
    digits=case['precisionDigits']; lower=dict(flatten(reference(case['parameters'],digits)))
    truth=reference(case['parameters'],digits+80)
    with localcontext() as c:
        c.prec=digits+80
        for path,y in flatten(truth):
            x=lower[path]
            if isinstance(y,D):
                budget=abs(y)*D('1e-150')+D(10)**(-digits+40)
                assert abs(x-y)<=budget,(case['id'],path,str(x),str(y))
            else: assert x==y,(case['id'],path)
    rows.append(dict(**case, confirmationPrecisionDigits=digits+80, reference=stringify(truth)))
    print(json.dumps(dict(case=case['id'],precisionDigits=digits,confirmationDigits=digits+80)),flush=True)
Path('scripts/fixtures/curie-weiss-oracle.json').write_text(json.dumps(dict(
    scope='Actual binary64 independent Decimal sums and small-N named-microstate enumeration; not a reader or model verification claim', cases=rows),indent=2)+'\n')
print(json.dumps(dict(referenceCases=len(rows),referenceOnly=True)))
