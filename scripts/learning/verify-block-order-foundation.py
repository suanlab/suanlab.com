"""Independent row-transfer integer polynomial oracle; no producer imports."""
import json
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from math import prod


def text(x):
    return str(F(x))


def reference(inp):
    shape, bs = inp['shape'], inp['blockShape']
    n, v = prod(shape), prod(bs)
    b = n // v
    periodic = inp['boundary'] == 'periodic'
    x = F(inp['disagreementActivity'])
    ys = list(map(F, inp['blockUpActivities']))
    target = tuple(inp['selectedUpCounts'])
    nx = shape[0]
    # Contract along the longer axis: the transfer row has at most four spins.
    if len(shape) == 1:
        width, height = 1, nx
        site = lambda row, col: row
    elif shape[0] <= shape[1]:
        width, height = shape
        site = lambda row, col: col + nx * row
    else:
        height, width = shape
        site = lambda row, col: row + nx * col
    def block(i):
        bx = (i % nx) // bs[0]
        return bx if len(shape) == 1 else bx + (nx // bs[0]) * ((i // nx) // bs[1])
    block_of = [block(i) for i in range(n)]
    row_counts, internal_d = {}, {}
    for row in range(height):
        for state in range(1 << width):
            ks = [0] * b
            for col in range(width):
                ks[block(site(row, col))] += (state >> col) & 1
            row_counts[row, state] = tuple(ks)
    for state in range(1 << width):
        d = sum(((state >> c) & 1) != ((state >> (c + 1)) & 1) for c in range(width - 1))
        if periodic and width > 1:
            d += (state & 1) != ((state >> (width - 1)) & 1)
        internal_d[state] = d
    dp = {(first, first, row_counts[0, first], internal_d[first]): 1 for first in range(1 << width)}
    for row in range(1, height):
        following = defaultdict(int)
        for (first, previous, ks, d), count in dp.items():
            for state in range(1 << width):
                next_ks = tuple(a + c for a, c in zip(ks, row_counts[row, state]))
                next_d = d + (previous ^ state).bit_count() + internal_d[state]
                following[first, state, next_ks, next_d] += count
        dp = following
    hist = defaultdict(lambda: defaultdict(int))
    for (first, last, ks, d), count in dp.items():
        if periodic:
            d += (first ^ last).bit_count()
        hist[ks][d] += count
    edge_count = sum(n if periodic else n * (s - 1) // s for s in shape)
    bonds = {ks: sum((count * x**d for d, count in h.items()), F(0)) for ks, h in hist.items()}
    weights = {ks: bond * prod(y**k for y, k in zip(ys, ks)) for ks, bond in bonds.items()}
    total = sum(weights.values())
    mean_k = [sum(weight * ks[a] for ks, weight in weights.items()) / total for a in range(b)]
    mean_d = sum(count * x**d * prod(y**k for y, k in zip(ys, ks)) * d for ks, h in hist.items() for d, count in h.items()) / total
    mean_dd = sum(count * x**d * prod(y**k for y, k in zip(ys, ks)) * d*d for ks, h in hist.items() for d, count in h.items()) / total
    profiles = []
    for index in range((v + 1)**b):
        ks = tuple(index // (v + 1)**a % (v + 1) for a in range(b))
        h, bond = hist[ks], bonds[ks]
        cd = sum(count * x**d * d for d, count in h.items()) / bond
        cdd = sum(count * x**d * d*d for d, count in h.items()) / bond
        profiles.append(dict(upCounts=list(ks), magnetizations=[text(F(2*k-v, v)) for k in ks],
            multiplicity=sum(h.values()), disagreementHistogram=[[d, h[d]] for d in sorted(h)],
            bondPolynomial=text(bond), relativeWeight=text(weights[ks]), probability=text(weights[ks]/total),
            conditionalMeanDisagreements=text(cd), conditionalVarianceDisagreements=text(cdd-cd*cd)))
    # Separately construct original-axis edges, then directly enumerate the
    # selected constraint to check site observables (not the transfer histogram).
    edges = []
    for i in range(n):
        stride = 1
        for s in shape:
            coordinate = i // stride % s
            if coordinate < s - 1:
                edges.append((i, i + stride))
            elif periodic:
                edges.append((i, i - (s - 1)*stride))
            stride *= s
    spin_sums = [F(0)] * n
    examples = []
    for state in range(1 << n):
        ks = tuple(sum((state >> i) & 1 for i in range(n) if block_of[i] == a) for a in range(b))
        if ks != target:
            continue
        d = sum(((state >> i) & 1) != ((state >> j) & 1) for i, j in edges)
        weight = x**d
        spin_sums = [s + weight * (2*((state >> i) & 1)-1) for i, s in enumerate(spin_sums)]
        if len(examples) < 8:
            examples.append(dict(mask=state, disagreements=d, conditionalProbability=text(weight / bonds[target])))
    isolated = []
    for a in range(b):
        sites = [i for i in range(n) if block_of[i] == a]
        local = {i: j for j, i in enumerate(sites)}
        internal = [(local[i], local[j]) for i, j in edges if i in local and j in local]
        s = F(0)
        for spins in product((-1, 1), repeat=v):
            if sum(t == 1 for t in spins) == target[a]:
                s += x**sum(spins[i] != spins[j] for i, j in internal)
        isolated.append(s)
    ms = [F(2*k-v, v) for k in target]
    dbar = sum((F(1)-ms[block_of[i]]*ms[block_of[j]])/2 for i, j in edges if block_of[i] != block_of[j])
    dbar = F(dbar)
    residual_power = (bonds[target] / prod(isolated))**dbar.denominator / x**dbar.numerator
    locality = None
    if b > 1:
        a, c = inp['witnessPair']
        corners = []
        for ka, kc in [(v, v), (v, 0), (0, v), (0, 0)]:
            ks = list(target)
            ks[a], ks[c] = ka, kc
            corners.append(weights[tuple(ks)])
        odds = corners[0]*corners[3]/(corners[1]*corners[2])
        neighbor = any({block_of[i], block_of[j]} == {a, c} for i, j in edges)
        locality = dict(pair=[a, c], pairIsNeighbor=neighbor, crossRatio=text(odds),
            rulesOutNearestNeighborPairFactorization=not neighbor and odds != 1,
            conditioning='all other block up-counts fixed; global magnetization not conditioned')
    return dict(sites=n, blocks=b, blockVolume=v, edgeCount=edge_count, blockOf=block_of,
        profiles=profiles, relativePartition=text(total), absolutePartitionSquared=text(x**(-edge_count)*prod(y**(-v) for y in ys)*total*total),
        meanMagnetizations=[text(2*k/v-1) for k in mean_k],
        magnetizationCovariance=[[text(F(4, v*v)*(sum(w*ks[a]*ks[c] for ks, w in weights.items())/total-mean_k[a]*mean_k[c])) for c in range(b)] for a in range(b)],
        meanDisagreements=text(mean_d), varianceDisagreements=text(mean_dd-mean_d*mean_d),
        globalUpProbabilities=[text(sum((w for ks, w in weights.items() if sum(ks) == k), F(0))/total) for k in range(n+1)],
        selected=dict(profileIndex=sum(k*(v+1)**a for a, k in enumerate(target)),
            conditionalSpinMeans=[text(s/bonds[target]) for s in spin_sums], examples=examples,
            isolatedBlockPolynomials=list(map(text, isolated)), meanProductCrossDisagreements=text(dbar),
            meanProductResidualPower=text(residual_power), meanProductResidualDivisor=str(dbar.denominator)),
        localityWitness=locality)


def case(shape, bs, boundary='open', x='1/2', ys=None, target=None, pair=None):
    b, v = prod(shape)//prod(bs), prod(bs)
    return dict(shape=shape, blockShape=bs, boundary=boundary, disagreementActivity=x,
        blockUpActivities=ys or ['1']*b, selectedUpCounts=target if target is not None else [v//2]*b,
        witnessPair=pair if pair is not None else ([] if b == 1 else [0, b-1]))


def inputs():
    return [
        case([4], [2]), case([6], [2]), case([4], [4], target=[3]),
        case([6], [2], target=[2, 0, 2]), case([6], [2], target=[2, 1, 0]),
        case([4], [1]), case([2], [1], 'periodic'), case([2], [2], 'periodic'),
        case([4], [2], x='1'), case([8], [2], x='3/2', ys=['2','1/3','5/2','1'], target=[0,1,2,1]),
        case([6], [2], 'periodic'), case([6], [2], ys=['3','2/7','5']),
        case([10], [5], target=[1,4]), case([16], [8], target=[3,5]),
        case([4,4], [2,2]), case([4,4], [2,2], 'periodic'),
        case([8,2], [4,2], 'periodic', target=[3,6]),
        case([2,8], [2,4], 'periodic', target=[6,3]),
        case([2,2], [1,1], 'periodic', x='2/3', ys=['2','3','1/2','4']),
        case([3,3], [3,3], 'periodic', target=[4]),
        case([2,4], [2,1], 'periodic', target=[1,0,2,1]),
        case([4], [2], x='999983/999979', ys=['1/999983','999979/2']),
        case([12], [3], 'periodic', x='2/5', target=[0,1,2,3]),
        case([4,3], [2,3], 'open', x='5/3', ys=['7/2','3/11'], target=[2,4]),
    ]


if __name__ == '__main__':
    print(json.dumps(dict(method='integer row-transfer polynomial contraction; Fraction evaluation; independent selected-state enumeration',
        cases=[dict(input=i, expected=reference(i)) for i in inputs()]), separators=(',', ':')))
