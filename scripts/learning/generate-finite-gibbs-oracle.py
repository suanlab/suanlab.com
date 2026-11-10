"""Independent actual-binary64 reference for finite conditional Gibbs kernels.
No TypeScript imports. Rows are source states; systematic order is K1 @ K2.
Reference generation does not certify a model, path sampler or reader.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import json
import math

KB = 1.380649e-23
cases = []
def add(id, J, h1=0, h2=0, T=1, support='full-product', offset=0, precision=400):
    cases.append(dict(id=id, parameters=dict(couplingEnergy=J, firstFieldEnergy=h1,
        secondFieldEnergy=h2, temperature=T, support=support, energyOffset=offset), precisionDigits=precision))
add('near-ideal-ln2', KB * math.log(2))
add('antiferromagnetic', -KB * math.log(2))
add('unequal-fields', .8 * KB, .35 * KB, -.2 * KB)
add('independent-unequal-fields', 0, .35 * KB, -.2 * KB)
add('uniform-product', 0)
add('common-offset', KB * math.log(2), offset=2**60)
add('negative-temperature', .8 * KB, .35 * KB, -.2 * KB, T=-1)
add('fixed-zero-sum-uniform', 3 * KB, 100 * KB, 100 * KB, support='fixed-zero-sum')
add('fixed-zero-sum-biased', 3 * KB, .5 * KB, -.25 * KB, support='fixed-zero-sum')
add('fixed-zero-sum-strong-bias', 3 * KB, 400 * KB, -400 * KB, support='fixed-zero-sum', precision=1600)
add('weak-interaction', 1e-200, precision=600)
add('rare-conditional', 400 * KB, precision=1600)
add('global-energy-temperature-scale', 8 * KB * math.log(2), T=8)
add('hot-subnormal-interaction', float.fromhex('0x0.0000000000001p-1022'), T=float.fromhex('0x1.fffffffffffffp+1023'), precision=1600)

def dec(x): return D.from_float(float(x))
def matrix_product(A, B):
    return [[sum(a*b for a, b in zip(row, col)) for col in zip(*B)] for row in A]
def evolve(q, P): return [sum(q[i]*P[i][j] for i in range(len(q))) for j in range(len(q))]
def reference(p, digits):
    with localcontext() as c:
        c.prec = digits
        J, h1, h2, T, offset = map(dec, [p['couplingEnergy'], p['firstFieldEnergy'], p['secondFieldEnergy'], p['temperature'], p['energyOffset']])
        assert T != 0
        states = [(-1,-1), (-1,1), (1,-1), (1,1)]
        if p['support'] == 'fixed-zero-sum': states = [s for s in states if sum(s) == 0]
        E = [offset - J*a*b - h1*a - h2*b for a,b in states]
        anchor = min(E) if T > 0 else max(E)
        w = [(-(e-anchor)/(dec(KB)*T)).exp() for e in E]
        W = sum(w); pi = [x/W for x in w]; n = len(pi)
        kernels = []
        for site in range(2):
            K = []
            for s in states:
                candidates = [j for j,t in enumerate(states) if t[1-site] == s[1-site]]
                denominator = sum(w[j] for j in candidates)
                K.append([w[j]/denominator if j in candidates else D(0) for j in range(n)])
            kernels.append(K)
        K1, K2 = kernels
        random = [[(a+b)/2 for a,b in zip(r1,r2)] for r1,r2 in zip(K1,K2)]
        systematic = matrix_product(K1,K2)
        blocked = [list(pi) for _ in states]
        named = dict(first=K1, second=K2, random=random, systematic=systematic, blocked=blocked)
        initial = [D(int(i==0)) for i in range(n)]
        # Independently obtain the binary marginal contraction from Pearson
        # covariance, rather than the model's product of conditional contrasts.
        means = [sum(q*D(s[k]) for q,s in zip(pi,states)) for k in range(2)]
        variances = [sum(q*(D(s[k])-means[k])**2 for q,s in zip(pi,states)) for k in range(2)]
        covariance = sum(q*(D(s[0])-means[0])*(D(s[1])-means[1]) for q,s in zip(pi,states))
        contraction = D(1) if p['support']=='fixed-zero-sum' else D(0) if J==0 else covariance**2/(variances[0]*variances[1])
        first_sweep = evolve(initial,systematic)
        distance1 = sum(abs(a-b) for a,b in zip(first_sweep,pi))/2
        if p['support']=='full-product' and J==0: distance1 = D(0)
        out = dict(target=pi, logTarget=[x.ln() for x in pi], kernels=named, flows={}, distributions={},
            systematicContraction=contraction,totalVariationToTarget={})
        tolerance = D(10)**(-digits+30)
        for name,K in named.items():
            assert all(abs(sum(row)-1) <= tolerance for row in K), (name,'stochastic')
            assert all(abs(a-b) <= tolerance for a,b in zip(evolve(pi,K),pi)), (name,'stationary')
            flows = [[pi[i]*K[i][j] for j in range(n)] for i in range(n)]
            if name in ['first','second','random','blocked']:
                assert all(abs(flows[i][j]-flows[j][i]) <= tolerance for i in range(n) for j in range(n)), (name,'reversible')
            out['flows'][name] = flows
            q = initial
            observations = []
            for step in range(1,5):
                q = evolve(q,K)
                if step in [1,2,4]: observations.append(list(q))
            out['distributions'][name] = observations
            if name=='blocked': distances=[D(0)]*3
            elif name=='systematic': distances=[distance1 if t==1 else distance1*contraction**(t-1) for t in [1,2,4]]
            else: distances=[sum(abs(a-b) for a,b in zip(q,pi))/2 for q in observations]
            out['totalVariationToTarget'][name]=distances
        return out

def leaves(x, path=()):
    if isinstance(x, dict):
        for k,v in x.items(): yield from leaves(v, path+(k,))
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from leaves(v, path+(i,))
    else: yield path,x

def strings(x):
    if isinstance(x,dict): return {k:strings(v) for k,v in x.items()}
    if isinstance(x,list): return [strings(v) for v in x]
    return str(x)

rows = []
for case in cases:
    digits = case['precisionDigits']
    a = dict(leaves(reference(case['parameters'],digits)))
    truth = reference(case['parameters'],digits+80)
    with localcontext() as c:
        c.prec = digits+80
        for path,y in leaves(truth):
            x = a[path]
            if y == 0: assert x == 0, (case['id'],path,'structural zero')
            else: assert abs((x-y)/y) < D('1e-150'), (case['id'],path,x,y)
    rows.append(dict(**case,confirmationPrecisionDigits=digits+80,reference=strings(truth)))
    print(json.dumps(dict(case=case['id'],precision=digits,confirmation=digits+80)),flush=True)
Path('scripts/fixtures/finite-gibbs-oracle.json').write_text(json.dumps(dict(
    scope='Independent actual-binary64 conditional Gibbs kernel references only; no model/reader verification claimed',
    stateOrder='lexicographic (-1,-1),(-1,+1),(+1,-1),(+1,+1), filtered by support',
    convention='row-vector distribution, source-row transitions, systematic=K1 K2; random step is one coordinate update while a systematic sweep updates both',
    initialDistribution='point mass at the first allowed state',observationSteps=[1,2,4],cases=rows),indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),referenceOnly=True)),flush=True)
