"""Independent fixed drawing references; not a path-integral evaluator."""
from fractions import Fraction as F
import json
import math
import pathlib


def matrix_product(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1)**j * a[0][j] * determinant([
        row[:j] + row[j+1:] for row in a[1:]
    ]) for j in range(len(a)))


identity = [[F(1), F(0)], [F(0), F(1)]]
h = [[F(0), F(-1)], [F(-1), F(0)]]
assert determinant(h) == -1 and matrix_product(h, h) == identity
assert matrix_product(h, [[F(1)], [F(1)]]) == [[-1], [-1]]
assert matrix_product(h, [[F(1)], [F(-1)]]) == [[1], [-1]]

action = lambda q: -q*q/F(2) - q - F(1, 4)
assert action(F(-1)) == F(1, 4)
assert -F(-1)-1 == 0  # derivative at q*; second derivative is -1.
negative_step = [[F(-1, 2), F(1)], [F(-3, 4), F(-1, 2)]]
assert matrix_product(negative_step, negative_step) == [[F(-1, 2), F(-1)], [F(3, 4), F(-1, 2)]]
focus_step = [[F(0), F(1)], [F(-1), F(0)]]
assert matrix_product(focus_step, focus_step) == [[-1, 0], [0, -1]]

# Positive principal minors certify the three positive eigenvalues at free N=4.
free_h = [[F(2 if i == j else -1 if abs(i-j) == 1 else 0)
           for j in range(3)] for i in range(3)]
principal = [determinant([row[:k] for row in free_h[:k]]) for k in range(1, 4)]
assert principal == [2, 3, 4]
# Each sqrt(-i*lambda) has angle -pi/4; product and principal root differ by pi.
root_phase_pi, principal_phase_pi = F(-3, 4), F(1, 4)
assert root_phase_pi - principal_phase_pi == -1

initial = [[F(1), F(0)], [F(0), F(1, 4)]]
free_map = [[F(1), F(1)], [F(0), F(1)]]
transpose = [list(row) for row in zip(*free_map)]
covariance = matrix_product(matrix_product(free_map, initial), transpose)
assert covariance == [[F(5, 4), F(1, 4)], [F(1, 4), F(1, 4)]]
assert determinant(covariance) == F(1, 4)
variances = []
for hbar in [F(1, 2), F(1, 8)]:
    fixed = F(1) + hbar*hbar/4
    scaled = hbar + hbar/4
    variances.append({'hbar': str(hbar), 'fixedText': str(fixed), 'scaledText': str(scaled)})
    # A position focus at chirp -1 need not collapse momentum.
    assert (1 - 2 + 1 + hbar*hbar/4) == hbar*hbar/4
    assert (1 + hbar*hbar/4) > 1

phases = [{'x': float(x), 'linear': float(4*x), 'quadratic': float(2*x*x)}
          for x in [F(-1), F(-1, 2), F(0), F(1, 2), F(1)]]
action_points = [{'x': float(q), 'y': float(action(q))}
                 for q in [F(-3)+F(i, 20) for i in range(81)]]
folds = []
for u in [1, 0, -1]:
    def phase(q):
        return q*q*q/3-u*q
    points = [{'x': float(q), 'y': float(phase(q))}
              for q in [F(-2)+F(i, 20) for i in range(81)]]
    stationary = [-1, 1] if u == 1 else [0] if u == 0 else []
    for q in stationary:
        assert q*q-u == 0
    folds.append({'control': u, 'points': points, 'stationary': stationary})
assert F(1, 3)-1 == F(-2, 3) and -F(1, 3)+1 == F(2, 3)

density = []
for i in range(121):
    q = F(-4)+F(3*i, 40)
    x = float(q)
    density.append({'x': x, 'y': math.exp(-x*x/2)/math.sqrt(2*math.pi),
                    'propagated': math.exp(-(x-1)**2/F(5, 2))/math.sqrt(2*math.pi*1.25)})
report = {
    'scope': 'Fixed illustrative geometry; exact Fraction algebra and Python binary64 projections. No arbitrary-input numerical certificate.',
    'negativeStationaryPoint': '-1', 'negativeStationaryAction': '1/4',
    'zeroPivotDeterminant': '-1', 'freePrincipalMinors': list(map(str, principal)),
    'productRootPhasePi': str(root_phase_pi), 'principalRootPhasePi': str(principal_phase_pi),
    'packetCovariance': [[str(v) for v in row] for row in covariance],
    'variances': variances, 'phases': phases, 'action': action_points, 'folds': folds,
    'densities': density,
    'damping': [{'x': x, 'y': math.exp(-x*x/2)} for x in [-2, -1, 0, 1, 2]],
}
root = pathlib.Path('.runtime/learning/real-time-phase-concepts')
root.mkdir(parents=True, exist_ok=True)
(root/'reference-certificates.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
