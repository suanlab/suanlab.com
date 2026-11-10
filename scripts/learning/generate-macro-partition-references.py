"""Independent exact-Fraction distributions and high-precision direct log sums.

No production imports. Empty/zero-probability groups have no invented conditional
law. A target-support violation is infinite KL, not numerical overflow.
"""
import argparse
import json
import math
import random
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/macro-partition-independent.json')
out = Path(parser.parse_args().output)
if out.exists():
    raise SystemExit('Refusing to overwrite reference')

def m(v):
    return mp.mpf(v.numerator) / v.denominator if isinstance(v, F) else mp.mpf(v)

def obs(v):
    v = m(v)
    projected = float(v)
    if v and (not math.isfinite(projected) or projected == 0):
        projected = None
    return dict(value=projected, sign=int(mp.sign(v)), logMagnitude=float(mp.log(abs(v))) if v else None)

def entropy(p):
    return -mp.fsum(m(x) * mp.log(m(x)) for x in p if x)

def divergence(p, q):
    if any(a and not b for a, b in zip(p, q)):
        return mp.inf
    if p == q:
        return mp.mpf(0)
    return mp.fsum(m(a) * mp.log(m(a / b)) for a, b in zip(p, q) if a)

def klobs(v):
    return dict(status='infinite', value=None, sign=1, logMagnitude=None) if mp.isinf(v) else dict(status='finite', **obs(v))

def check(a, b):
    assert abs(a - b) < mp.mpf('1e-3000'), (a, b)

def case(name, weights, target, mapping, group_count):
    # 4096 decimal digits resolve cubic min/max-weight divergences after grouping; the
    # reference takes no difference of rounded binary64 probabilities/entropies.
    with mp.workdps(4096):
        w = list(map(F, weights)); v = list(map(F, target))
        p = [x / sum(w) for x in w]; q = [x / sum(v) for x in v]
        members = [[i for i, g in enumerate(mapping) if g == j] for j in range(group_count)]
        P = [sum((p[i] for i in ids), F(0)) for ids in members]
        Q = [sum((q[i] for i in ids), F(0)) for ids in members]
        reconstructed = [P[mapping[i]] / len(members[mapping[i]]) for i in range(len(p))]
        h = entropy(p); hm = entropy(P); hb = entropy(reconstructed)
        conditional = mp.mpf(0); reconstruction_gain = mp.mpf(0); hidden_kl = mp.mpf(0)
        groups = []
        for j, ids in enumerate(members):
            r = [p[i] / P[j] for i in ids] if P[j] else None
            s = [q[i] / Q[j] for i in ids] if Q[j] else None
            h_cond = entropy(r) if r is not None else None
            h_contribution = m(P[j]) * h_cond if P[j] else mp.mpf(0)
            conditional += h_contribution
            uniform = [F(1, len(ids))] * len(ids) if ids else None
            gain = m(P[j]) * divergence(r, uniform) if P[j] else mp.mpf(0)
            reconstruction_gain += gain
            # When P>0,Q=0 the conditional target does not exist. The macro
            # divergence is already infinite; report undefined, not infinity-
            # infinity or an arbitrary target conditional.
            if not P[j]:
                inside = mp.mpf(0)
            elif not Q[j]:
                inside = None
            else:
                inside = m(P[j]) * divergence(r, s)
            if inside is not None:
                hidden_kl += inside
            groups.append(dict(group=j, members=ids, count=len(ids), probability=obs(P[j]), targetProbability=obs(Q[j]),
                conditionalProbabilities=None if r is None else list(map(obs, r)),
                targetConditionalProbabilities=None if s is None else list(map(obs, s)),
                conditionalEntropy=None if h_cond is None else obs(h_cond),
                conditionalEntropyContribution=obs(h_contribution), reconstructionGainContribution=obs(gain),
                conditionalDivergenceContribution=None if inside is None else klobs(inside)))
        check(h, hm + conditional); check(hb - h, reconstruction_gain)
        d = divergence(p, q); dm = divergence(P, Q)
        if not mp.isinf(d):
            check(d, dm + hidden_kl)
        assert h >= hm - mp.mpf('1e-3000') and hb >= h - mp.mpf('1e-3000')
        return dict(name=name, weights=weights, targetWeights=target, mapping=mapping, groupCount=group_count,
            probabilities=list(map(obs, p)), reconstructedProbabilities=list(map(obs, reconstructed)), groups=groups,
            microEntropy=obs(h), macroEntropy=obs(hm), conditionalEntropy=obs(conditional),
            reconstructedEntropy=obs(hb), reconstructionGain=obs(reconstruction_gain),
            microDivergence=klobs(d), macroDivergence=klobs(dm),
            hiddenDivergence=None if any(g['conditionalDivergenceContribution'] is None for g in groups) else klobs(hidden_kl))

inputs = [
 ('hidden-bias', [6,2,1,1], [4,4,1,1], [0,0,1,1], 2),
 ('same-macros-different-micros', [8,0,1,1], [4,4,1,1], [0,0,1,1], 2),
 ('single-state-empty-labels', [1], [3], [2], 4),
 ('zero-mass-populated-group', [1,0,0], [1,0,0], [0,1,1], 3),
 ('hidden-infinite-divergence', [1,1], [1,0], [0,0], 1),
 ('macro-infinite-divergence', [1,1], [1,0], [0,1], 3),
 ('identity', [1,2,3,4], [4,3,2,1], [0,1,2,3], 4),
 ('one-group', [1,2,3,4], [4,3,2,1], [0,0,0,0], 1),
]
rng = random.Random(26110627)
for j in range(16):
    n = rng.randint(2,12); groups = rng.randint(1,8)
    w = [rng.randint(0,20) for _ in range(n)]; v = [rng.randint(0,20) for _ in range(n)]
    w[0] += 1; v[0] += 1
    inputs.append((f'integer-{j}',w,v,[rng.randrange(groups) for _ in range(n)],groups))
minimum = math.ulp(0.); maximum = float.fromhex('0x1.fffffffffffffp+1023')
limits = [
 ('overflowing-sums',[maximum]*4,[maximum,maximum/2,maximum/4,maximum/8],[0,0,1,1],3),
 ('underflowing-micro',[minimum,maximum],[minimum*2,maximum],[0,0],2),
 ('underflowing-macro',[minimum,maximum],[minimum*2,maximum],[0,1],3),
 ('nearly-equal',[1.,math.nextafter(1.,2.)],[1.,1.],[0,0],1),
 ('nearly-equal-macro',[1.,math.nextafter(1.,2.)],[1.,1.],[0,1],2),
 ('subnormal-only',[minimum,minimum*2,minimum*3],[minimum*2,minimum,minimum*3],[0,0,1],3),
 ('unobserved-positive-target',[0.,maximum,minimum],[minimum,maximum,minimum],[0,0,1],3),
 ('hidden-support',[minimum,maximum],[0.,maximum],[0,0],2),
 ('macro-support',[minimum,maximum],[0.,maximum],[0,1],2),
 ('uniform-in-groups',[maximum,maximum,minimum,minimum],[maximum,maximum,minimum,minimum],[0,0,1,1],4),
 ('ultratiny-normalization-change',[maximum,minimum,minimum],[maximum,minimum*2,minimum],[0,1,2],3),
 ('many-labels',[1.,0.,1.,0.],[0.,1.,0.,1.],[0,1,2,3],8),
]
result=dict(method='Exact Fraction normalization of supplied binary64 inputs; 4096-decimal-digit direct entropy/KL sums, with chain and reconstruction identities independently checked.',
            calculationDigits=4096, ordinary=[case(*x) for x in inputs], limits=[case(*x) for x in limits])
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(f'Wrote {len(inputs)} ordinary and {len(limits)} limiting cases at 4096 digits')
