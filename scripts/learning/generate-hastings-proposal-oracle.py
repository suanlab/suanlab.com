"""Independent pair-flux and three-state tree references for Hastings correction."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import gcd
from pathlib import Path
import json

BASE_TARGETS = [(1, 2, 4), (1, 1, 1), (4, 2, 1), (1, 8, 1), (100, 1, 3), (1, 1, 4)]
BASE_PROPOSALS = [(2, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1)]
CASES = [(f'target-{i}-proposal-{j}', target, proposal) for i, target in enumerate(BASE_TARGETS) for j, proposal in enumerate(BASE_PROPOSALS)]
CASES += [
    ('maximum-distinct', (1000000, 999999, 999983), (1000000, 999999, 0)),
    ('tiny-target', (1, 1000000, 1), (1, 1000000, 1)),
    ('large-target-contrast', (1000000, 1, 1), (1000000, 1, 0)),
    ('small-exact-drift', (1000000, 999999, 1000000), (1000000, 999999, 1000000)),
    ('tiny-accidental-invariance', (1000000, 999999, 999999), (1000000, 999999, 1000000)),
    ('scaled-target', (2, 4, 8), (2, 1, 1)),
    ('scaled-proposal', (1, 2, 4), (6, 3, 3)),
    ('maximum-directed-uniform', (1000000, 1000000, 1000000), (1000000, 0, 0)),
    ('symmetric-without-holding', (2, 3, 5), (1000000, 1000000, 0)),
    ('maximum-self-only', (1, 2, 4), (0, 0, 1000000)),
    ('independent-reset', (1, 1, 1), (1, 1, 1)),
    ('counterclockwise-directed-uniform', (1, 1, 1), (0, 1, 0)),
]


def advance(mu, p):
    return [sum(mu[i] * p[i][j] for i in range(3)) for j in range(3)]


def stationary_tree(p):
    # Enumerate all rooted directed spanning trees; each nonroot has one parent.
    totals = []
    for root in range(3):
        other = [i for i in range(3) if i != root]
        total = F(0)
        for parents in product(range(3), repeat=2):
            parent = dict(zip(other, parents))
            valid = True
            for start in other:
                node, seen = start, set()
                while node != root:
                    if node in seen:
                        valid = False
                        break
                    seen.add(node)
                    node = parent[node]
                if not valid:
                    break
            if valid:
                total += p[other[0]][parents[0]] * p[other[1]][parents[1]]
        totals.append(total)
    return [q / sum(totals) for q in totals] if sum(totals) else None


def stationary_rank(p):
    a = [[p[j][i] - F(i == j) for j in range(3)] for i in range(3)]
    if any(a[i][k] * a[j][l] != a[i][l] * a[j][k]
           for i, j in combinations(range(3), 2) for k, l in combinations(range(3), 2)):
        return 2
    return 1 if any(q for row in a for q in row) else 0


def structure(p):
    reach = []
    for start in range(3):
        seen, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for j in range(3):
                if p[i][j] and j not in seen:
                    seen.add(j)
                    stack.append(j)
        reach.append(seen)
    groups = sorted({tuple(j for j in range(3) if j in reach[i] and i in reach[j]) for i in range(3)})
    periods = [0, 0, 0]
    for group in groups:
        period = 0
        for length in range(1, len(group) + 1):
            for nodes in permutations(group, length):
                if all(p[nodes[i]][nodes[(i + 1) % length]] > 0 for i in range(length)):
                    period = gcd(period, length)
        for i in group:
            periods[i] = period
    return dict(irreducible=len(groups) == 1, communicatingClasses=[list(g) for g in groups],
                returnPeriods=periods, uniqueClassPeriod=periods[0] if len(groups) == 1 else None)


def quantity(q):
    with localcontext() as ctx:
        ctx.prec = 100
        logarithm = str((Decimal(abs(q.numerator)) / Decimal(q.denominator)).ln()) if q else None
    v = float(q)
    return dict(numerator=str(q.numerator), denominator=str(q.denominator),
                value=None if q and v == 0 else v, sign=1 if q > 0 else -1 if q < 0 else 0,
                logMagnitude=logarithm)


def serialize(v):
    if isinstance(v, F):
        return quantity(v)
    if isinstance(v, list):
        return [serialize(x) for x in v]
    if isinstance(v, dict):
        return {k: serialize(x) for k, x in v.items()}
    return v


endpoint_comparisons = 0
records = []
for name, target_weights, (forward, backward, stay) in CASES:
    total = forward + backward + stay
    target = [F(w, sum(target_weights)) for w in target_weights]
    proposal = [[F(0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for offset, w in [(0, stay), (1, forward), (-1, backward)]:
            proposal[i][(i + offset) % 3] += F(w, total)
    modes = {}
    for corrected in [True, False]:
        accepted = [[F(0) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if corrected:
                    accepted[i][j] = min(target[i] * proposal[i][j], target[j] * proposal[j][i]) / target[i]
                else:
                    accepted[i][j] = proposal[i][j] * F(min(target_weights[i], target_weights[j]), target_weights[i])
        acceptance = [[accepted[i][j] / proposal[i][j] if proposal[i][j] else None for j in range(3)] for i in range(3)]
        rejected = [[proposal[i][j] - accepted[i][j] for j in range(3)] for i in range(3)]
        rejection = [sum(row) for row in rejected]
        p = [[accepted[i][j] + (rejection[i] if i == j else 0) for j in range(3)] for i in range(3)]
        assert all(sum(row) == 1 and min(row) >= 0 for row in p)
        for i in range(3):
            # Independently enumerate accepted and rejected proposal outcomes.
            outcomes = [F(0)] * 3
            for proposed in range(3):
                if proposal[i][proposed]:
                    outcomes[proposed] += proposal[i][proposed] * acceptance[i][proposed]
                    outcomes[i] += proposal[i][proposed] * (1 - acceptance[i][proposed])
            assert outcomes == p[i]
            endpoint_comparisons += 3
        target_next = advance(target, p)
        residual = [a - b for a, b in zip(target_next, target)]
        flux = [[target[i] * p[i][j] for j in range(3)] for i in range(3)]
        current = [[flux[i][j] - flux[j][i] for j in range(3)] for i in range(3)]
        invariant = all(v == 0 for v in residual)
        db = all(v == 0 for row in current for v in row)
        rank = stationary_rank(p)
        stationary = stationary_tree(p)
        assert (stationary is not None) == (rank == 2)
        if stationary:
            assert advance(stationary, p) == stationary
            assert min(stationary) > 0 and sum(stationary) == 1
        else:
            assert p == [[F(i == j) for j in range(3)] for i in range(3)]
        if corrected:
            assert invariant and db
        actual_flux = [[stationary[i] * p[i][j] for j in range(3)] for i in range(3)] if stationary else None
        actual_current = [[actual_flux[i][j] - actual_flux[j][i] for j in range(3)] for i in range(3)] if actual_flux else None
        actual_residual = [a - b for a, b in zip(advance(stationary, p), stationary)] if stationary else None
        graph = structure(p)
        classification = dict(**graph, stationaryIsUnique=stationary is not None,
                              targetIsUniqueStationary=invariant and stationary is not None,
                              convergesToTargetFromEveryInitial=invariant and graph['irreducible'] and graph['uniqueClassPeriod'] == 1)
        movement = [sum(p[i][j] for j in range(3) if i != j) for i in range(3)]
        jump = [[p[i][j] / movement[i] if i != j else F(0) for j in range(3)] for i in range(3)] if min(movement) > 0 else None
        jump_stationary = stationary_tree(jump) if jump else None
        stationary_movement = sum(stationary[i] * movement[i] for i in range(3)) if stationary else None
        if jump_stationary:
            assert advance(jump_stationary, jump) == jump_stationary
            assert jump_stationary == [stationary[i] * movement[i] / stationary_movement for i in range(3)]
        jump_residual = [v - w for v, w in zip(advance(jump_stationary, jump), jump_stationary)] if jump_stationary else None
        delta = max(sum(abs(p[i][k] - p[j][k]) for k in range(3)) / 2 for i in range(3) for j in range(3))
        mode = dict(acceptance=acceptance, acceptedProposal=accepted, rejectedProposal=rejected,
                    rejectionByState=rejection, selfProposalByState=[proposal[i][i] for i in range(3)],
                    totalHoldingByState=[p[i][i] for i in range(3)], transition=p,
                    targetFlux=flux, targetCurrent=current, targetOneStep=target_next, targetResidual=residual,
                    targetInvariant=invariant, targetDetailedBalance=db, stationaryRank=rank,
                    stationaryFamilyDimension=2-rank, stationaryDistribution=stationary,
                    stationaryFlux=actual_flux, stationaryCurrent=actual_current, stationaryResidual=actual_residual,
                    stationaryTargetDistance=sum(abs(a - b) for a, b in zip(stationary, target)) / 2 if stationary else None,
                    movementByState=movement, jumpTransition=jump, jumpStationaryDistribution=jump_stationary,
                    jumpStationaryResidual=jump_residual,
                    jumpTargetDistance=sum(abs(v - w) for v, w in zip(jump_stationary, target)) / 2 if jump_stationary else None,
                    meanMovementAtStationary=stationary_movement,
                    contractionCoefficient=delta,
                    meanProposalAcceptanceAtTarget=sum(target[i] * sum(accepted[i]) for i in range(3)),
                    meanProposalRejectionAtTarget=sum(target[i] * rejection[i] for i in range(3)),
                    meanMovementAtTarget=sum(target[i] * sum(accepted[i][j] for j in range(3) if i != j) for i in range(3)),
                    classification=classification)
        assert mode['meanProposalAcceptanceAtTarget'] + mode['meanProposalRejectionAtTarget'] == 1
        modes['corrected' if corrected else 'uncorrected'] = mode
    reference = dict(target=target, proposal=proposal, **modes,
                     correctionChangesTransition=modes['corrected']['transition'] != modes['uncorrected']['transition'])
    records.append(dict(id=name, parameters=dict(targetWeights=target_weights, forwardWeight=forward,
                                                backwardWeight=backward, stayWeight=stay), reference=serialize(reference)))


def count(v):
    if isinstance(v, dict):
        return 1 if 'numerator' in v else sum(count(x) for x in v.values())
    if isinstance(v, list):
        return sum(count(x) for x in v)
    return 0


report = dict(scope='exact integer-input Fraction pair fluxes, rooted trees, minors and simple-cycle graph classification',
              cases=records, caseCount=len(records), exactObservationComparisons=sum(count(c['reference']) for c in records),
              elementaryOutcomeEndpointComparisons=endpoint_comparisons)
fixture = Path('scripts/fixtures/hastings-proposal-oracle.json')
fixture.write_text(json.dumps(report, indent=2) + '\n')
folder = Path('.runtime/learning/hastings-proposal')
folder.mkdir(parents=True, exist_ok=True)
(folder / 'oracle-verification.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
