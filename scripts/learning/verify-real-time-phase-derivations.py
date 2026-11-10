"""Independent exact finite-chain, regulator, packet and fold identities.

No production calculator is imported. Exact Fraction algebra is separated from
Decimal contour quadrature diagnostics, which do not certify integral errors.
"""
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
sys.set_int_max_str_digits(131072)
ROOT = Path('.runtime/learning/real-time-phase-foundation')
ROOT.mkdir(parents=True, exist_ok=True)
COUNTS = Counter()


def check(condition, category):
    assert condition, category
    COUNTS[category] += 1


@dataclass(frozen=True, eq=False)
class C:
    r: F = F(0)
    i: F = F(0)

    @staticmethod
    def of(x):
        return x if isinstance(x, C) else C(F(x))

    def __add__(self, x):
        x = C.of(x)
        return C(self.r + x.r, self.i + x.i)

    __radd__ = __add__

    def __neg__(self):
        return C(-self.r, -self.i)

    def __sub__(self, x):
        return self + (-C.of(x))

    def __rsub__(self, x):
        return C.of(x) - self

    def __mul__(self, x):
        x = C.of(x)
        return C(self.r*x.r-self.i*x.i, self.r*x.i+self.i*x.r)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = C.of(x)
        denominator = x.r*x.r+x.i*x.i
        return self*C(x.r, -x.i)*F(1, denominator)

    def __rtruediv__(self, x):
        return C.of(x)/self

    def __bool__(self):
        return bool(self.r or self.i)

    def __eq__(self, x):
        x = C.of(x)
        return self.r == x.r and self.i == x.i

    def conjugate(self):
        return C(self.r, -self.i)


I = C(F(0), F(1))


def mm(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def mv(a, b):
    return [sum(x*y for x, y in zip(row, b)) for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def power(a, count):
    result = eye(len(a))
    while count:
        if count % 2:
            result = mm(result, a)
        a = mm(a, a)
        count //= 2
    return result


def determinant(a):
    a = [row[:] for row in a]
    result = a[0][0]*0+1
    for k in range(len(a)):
        pivot = next((i for i in range(k, len(a)) if a[i][k]), None)
        if pivot is None:
            return result*0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            result = -result
        value = a[k][k]
        result *= value
        for i in range(k+1, len(a)):
            ratio = a[i][k]/value
            for j in range(k+1, len(a)):
                a[i][j] -= ratio*a[k][j]
    return result


def inverse(a):
    n = len(a)
    a = [row[:]+tail for row, tail in zip(a, eye(n))]
    for k in range(n):
        pivot = next((i for i in range(k, n) if a[i][k]), None)
        if pivot is None:
            return None
        a[k], a[pivot] = a[pivot], a[k]
        value = a[k][k]
        a[k] = [x/value for x in a[k]]
        for i in range(n):
            if i != k:
                ratio = a[i][k]
                a[i] = [x-ratio*y for x, y in zip(a[i], a[k])]
    return [row[n:] for row in a]


def solve(a, b):
    n = len(a)
    work = [row[:]+[rhs] for row, rhs in zip(a, b)]
    pivots = []
    for col in range(n):
        row = len(pivots)
        pivot = next((i for i in range(row, n) if work[i][col]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        value = work[row][col]
        work[row] = [x/value for x in work[row]]
        for i in range(n):
            if i != row:
                ratio = work[i][col]
                work[i] = [x-ratio*y for x, y in zip(work[i], work[row])]
        pivots.append(col)
    compatible = all(any(row[:n]) or not row[n] for row in work)
    q = [F(0)]*n
    if compatible:
        for i, col in enumerate(pivots):
            q[col] = work[i][n]
    return compatible, len(pivots), q


def inertia(a):
    """Dense symmetric congruences, independent of the Sturm sign formula."""
    a = [row[:] for row in a]
    positive = negative = zeros = two_by_two = 0
    while a:
        n = len(a)
        pivot = next((i for i in range(n) if a[i][i]), None)
        if pivot is not None:
            order = [pivot]+[i for i in range(n) if i != pivot]
            a = [[a[i][j] for j in order] for i in order]
            value = a[0][0]
            positive += value > 0
            negative += value < 0
            a = [[a[i][j]-a[i][0]*a[0][j]/value
                  for j in range(1, n)] for i in range(1, n)]
            continue
        pair = next(((i, j) for i in range(n) for j in range(i+1, n)
                     if a[i][j]), None)
        if pair is None:
            zeros += n
            break
        i, j = pair
        order = [i, j]+[k for k in range(n) if k not in pair]
        a = [[a[i][j] for j in order] for i in order]
        coupling = a[0][1]
        positive += 1
        negative += 1
        two_by_two += 1
        a = [[a[i][j]-(a[i][0]*a[1][j]+a[i][1]*a[0][j])/coupling
              for j in range(2, n)] for i in range(2, n)]
    return positive, negative, zeros, two_by_two


def continuants(v, d):
    values = [F(1)]
    for k in range(d):
        values.append((2-v)*values[-1]-(values[-2] if k else 0))
    return values


def encode(x):
    if isinstance(x, F):
        return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'
    if isinstance(x, C):
        return {'real': encode(x.r), 'imaginary': encode(x.i)}
    if isinstance(x, D):
        return str(x)
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    return x


def chain(case):
    n, tau, mass, hbar, omega, a, b = case['n'], *map(F, [case[k] for k in ['time', 'mass', 'hbar', 'omegaSquared', 'start', 'end']])
    d, delta = n-1, tau/n
    s, v = mass/delta, omega*delta*delta
    h = [[s*((2-v) if i == j else -1 if abs(i-j) == 1 else 0)
          for j in range(d)] for i in range(d)]
    forcing = [F(0)]*d
    forcing[0] += s*a
    forcing[-1] += s*b
    constant = s*(a*a+b*b)/2-delta*mass*omega*(a*a+b*b)/4
    theta = continuants(v, d)
    signed = [1 if x > 0 else -1 for x in theta if x]
    variations = sum(x != y for x, y in zip(signed, signed[1:]))
    ip, im, iz, blocks = inertia(h)
    check(im == variations, 'exact-sturm-vs-dense-inertia')
    check(ip+im+iz == d, 'exact-inertia-dimension')
    check(iz == int(theta[-1] == 0), 'exact-nullity')
    det_h = determinant(h)
    check(det_h == s**d*theta[-1], 'exact-factorized-determinant')
    step = [[1-v/2, delta/mass], [-mass*omega*delta*(1-v/4), 1-v/2]]
    total = power(step, n)
    aa, bb, cc, dd = total[0][0], total[0][1], total[1][0], total[1][1]
    check(aa*dd-bb*cc == 1, 'exact-symplectic-determinant')
    check(aa == dd, 'exact-time-reversal-diagonal')
    check(bb == delta/mass*theta[-1], 'exact-transfer-vs-Hessian-determinant')
    for multiplier in [2, 4, 8]:
        count = n*multiplier
        refined_delta = tau/count
        refined_v = omega*refined_delta**2
        refined_step = [[1-refined_v/2, refined_delta/mass], [-mass*omega*refined_delta*(1-refined_v/4), 1-refined_v/2]]
        refined = power(refined_step, count)
        check(determinant(refined) == 1, 'exact-refined-map-symplectic')
        check(refined[0][1] == refined_delta/mass*continuants(refined_v, count-1)[-1], 'exact-refined-map-continuant')
    def action(q):
        path = [a]+q+[b]
        return s*sum((x-y)**2 for x, y in zip(path[1:], path))/2-delta*mass*omega*(sum(x*x for x in q)+(a*a+b*b)/2)/2
    probe = [F(i-d, 2*n) for i in range(d)]
    quadratic = sum(x*y for x, y in zip(probe, mv(h, probe)))/2-sum(x*y for x, y in zip(forcing, probe))+constant
    check(action(probe) == quadratic, 'exact-action-and-endpoint-weights')
    gradient = [x-y for x, y in zip(mv(h, probe), forcing)]
    for i in range(d):
        plus, minus, epsilon = probe[:], probe[:], F(1, 7)
        plus[i] += epsilon
        minus[i] -= epsilon
        check((action(plus)-action(minus))/(2*epsilon) == gradient[i], 'exact-action-gradient')
    compatible, rank, q = solve(h, forcing)
    if theta[-1]:
        inv = inverse(h)
        check(compatible and rank == d and inv is not None, 'exact-unique-stationary-path')
        identity = mm(h, inv)
        for i in range(d):
            for j in range(d):
                check(identity[i][j] == F(i == j), 'exact-full-inverse-residual')
                green = theta[min(i, j)]*theta[d-1-max(i, j)]/(s*theta[d])
                check(inv[i][j] == green, 'exact-Green-vs-dense-inverse')
        check(mv(h, q) == forcing, 'exact-stationary-residual')
        classical = (dd*b*b+aa*a*a-2*a*b)/(2*bb)
        check(action(q) == classical, 'exact-generating-action')
        check(action(q) == constant-sum(x*y for x, y in zip(forcing, q))/2, 'exact-completed-action')
        check((-n+ip-im) == -1-2*im, 'exact-Fresnel-Maslov-phase')
        kernel_squared_times_pi = 1/(2*hbar*abs(bb))
        check(kernel_squared_times_pi == s/(2*hbar*abs(theta[-1])), 'exact-kernel-prefactor')
        status = 'unique'
    else:
        null = theta[:-1]
        check(mv(h, null) == [F(0)]*d, 'exact-full-null-vector')
        check(rank == d-1, 'exact-singular-rank')
        check(aa in [F(-1), F(1)] and cc == 0 and dd == aa, 'exact-caustic-map-parity')
        check(compatible == (b == aa*a), 'exact-caustic-endpoint-compatibility')
        check(compatible == (sum(x*y for x, y in zip(null, forcing)) == 0), 'exact-null-forcing-compatibility')
        k = im+1
        check(aa == (-1)**k, 'exact-caustic-Maslov-parity')
        if compatible:
            check(action(q) == 0, 'exact-caustic-stationary-action')
            for shift in [F(-2), F(1, 3), F(4)]:
                shifted = [x+shift*y for x, y in zip(q, null)]
                check(mv(h, shifted) == forcing and action(shifted) == 0, 'exact-affine-stationary-family')
        status = 'affine-family' if compatible else 'no-stationary-path'
    # Minimum-uncertainty Gaussians: exact covariance transport, no density
    # integral or phase-space collapse is inferred from a pointwise kernel.
    packet_rows = []
    for family in ['fixed-variance', 'hbar-scaled-variance']:
        width, chirp, q0, p0 = F(2, 3), F(7, 5), F(1, 3), F(-2, 5)
        variance = width if family == 'fixed-variance' else hbar*width
        sigma = [[variance, chirp*variance], [chirp*variance, chirp**2*variance+hbar**2/(4*variance)]]
        final = mm(mm(total, sigma), transpose(total))
        check(determinant(sigma) == hbar**2/4, 'exact-packet-initial-uncertainty')
        check(determinant(final) == hbar**2/4, 'exact-packet-propagated-uncertainty')
        check(final[0][0] > 0 and final[1][1] > 0, 'exact-packet-positive-variances')
        z0 = C(chirp, hbar/(2*variance))
        denominator = aa+bb*z0
        zt = (cc+dd*z0)/denominator
        check(zt.i > 0, 'exact-packet-positive-complex-width')
        check(hbar/(2*zt.i) == final[0][0], 'exact-packet-Mobius-vs-covariance')
        check(zt.r*final[0][0] == final[0][1], 'exact-packet-chirp-vs-covariance')
        center = mv(total, [q0, p0])
        central_action = (center[0]*center[1]-q0*p0)/2
        if bb:
            check(central_action == (dd*center[0]**2+aa*q0*q0-2*q0*center[0])/(2*bb), 'exact-packet-central-action')
            coefficient = C(hbar/(2*variance), -(aa/bb+chirp))
            # Squaring the branch-resolved Gaussian prefactor must give1/z.
            check(C(0, -1)*((-1)**im)/(abs(bb)*coefficient) == 1/denominator, 'exact-packet-prefactor-square')
        else:
            check(final == sigma and central_action == 0, 'exact-packet-through-caustic')
        if family == 'fixed-variance':
            classical_sigma = [[width, chirp*width], [chirp*width, chirp**2*width]]
            classical_final = mm(mm(total, classical_sigma), transpose(total))
            check(determinant(classical_final) == 0, 'exact-fixed-width-classical-ensemble')
            check(classical_final[0][0] == width*(aa+bb*chirp)**2, 'exact-fixed-width-limit-variance')
        else:
            scaled = [[x/hbar for x in row] for row in final]
            check(determinant(scaled) == F(1, 4), 'exact-hbar-scaled-localized-family')
        packet_rows.append({'family': family, 'variance': variance, 'final': final, 'center': center, 'centralAction': central_action, 'complexWidth': zt})
    if d <= 7 or (n == 32 and v == 2):
        for damping in [F(1), F(1, 3)]:
            damped = [[C(damping if i == j else 0, -h[i][j]) for j in range(d)] for i in range(d)]
            inv_c = inverse(damped)
            check(inv_c is not None, 'exact-regulator-invertibility')
            identity = mm(damped, inv_c)
            for i in range(d):
                for j in range(d):
                    check(identity[i][j] == F(i == j), 'exact-regulated-full-inverse-residual')
            pivots = []
            for i in range(d):
                pivot = damped[i][i]+s*s/pivots[-1] if i else damped[i][i]
                check(pivot.r > 0, 'exact-regulated-positive-real-pivots')
                pivots.append(pivot)
            product = C(F(1))
            for value in pivots:
                product *= value
            check(product == determinant(damped), 'exact-regulated-branch-determinant')
            mean = [(-I)*x for x in mv(inv_c, forcing)]
            check(mv(damped, mean) == [-I*x for x in forcing], 'exact-regulated-complex-mean')
            sample = [C(F(i, 3), F(1, 7)) for i in range(d)]
            centered = [x-y for x, y in zip(sample, mean)]
            lhs = -sum(x*y for x, y in zip(sample, mv(damped, sample)))/2-I*sum(x*y for x, y in zip(forcing, sample))+I*constant
            rhs = -sum(x*y for x, y in zip(centered, mv(damped, centered)))/2-sum(x*y for x, y in zip(forcing, mv(inv_c, forcing)))/2+I*constant
            check(lhs == rhs, 'exact-regulated-completed-square')
    row = {'id': case['id'], 'intervals': n, 'v': v, 'detA': theta[-1], 'inertia': [ip, im, iz], 'twoByTwoCongruenceBlocks': blocks, 'stationaryStatus': status, 'map': total, 'packetFamilies': packet_rows}
    return row


def packet_limit_counterexamples():
    free = [[F(1), F(1)], [F(0), F(1)]]
    variance = F(2, 3)
    for hbar in [F(1, 2), F(1, 4), F(1, 8), F(1, 16)]:
        fixed = [[variance, F(0)], [F(0), hbar*hbar/(4*variance)]]
        propagated = mm(mm(free, fixed), transpose(free))
        check(propagated[0][0] == variance+hbar*hbar/(4*variance), 'exact-fixed-width-free-packet-counterexample')
        focused = [[variance, -variance], [-variance, variance+hbar*hbar/(4*variance)]]
        propagated = mm(mm(free, focused), transpose(free))
        check(propagated[0][0] == hbar*hbar/(4*variance), 'exact-chirped-focus-position-limit')
        check(propagated[1][1] == variance+hbar*hbar/(4*variance), 'exact-chirped-focus-momentum-ensemble')
        check(determinant(propagated) == hbar*hbar/4, 'exact-chirped-focus-uncertainty')


def fold_checks():
    for r in [F(1, 2), F(1), F(2), F(4)]:
        u = r*r
        check(r*r-u == 0 and (-r)**2-u == 0, 'exact-fold-two-stationary-points')
        check(r**3/3-u*r == -2*r**3/3, 'exact-fold-positive-saddle-action')
        check((-r)**3/3-u*(-r) == 2*r**3/3, 'exact-fold-negative-saddle-action')
    for length in [F(1, 3), F(1, 2), F(1), F(2)]:
        hbar = length**3
        for control in map(F, [-4, -1, 0, 1, 4]):
            u = control*length*length
            for z in map(F, [-2, F(-1, 2), 0, 1, 2]):
                x = length*z
                check((x**3/3-u*x)/hbar == z**3/3-control*z, 'exact-fold-uniform-rescaling')
    coefficients = [F(1)]
    for k in range(1, 81):
        coefficients.append(coefficients[-1]*F((6*k-5)*(6*k-1), 48*k))
    dlmf = F(1)
    for k, value in enumerate(coefficients):
        moment = F(factorial(6*k), factorial(3*k)*factorial(2*k)*576**k)
        check(value == moment, 'exact-positive-Airy-Gaussian-moment-coefficients')
        if k:
            dlmf *= F((6*k-5)*(6*k-3)*(6*k-1), (2*k-1)*216*k)
        check(value == F(3, 2)**k*dlmf, 'exact-Airy-DLMF-coefficient-match')
    # Independent polynomial ODE coefficient construction for both origin data.
    for initial in [(F(1), F(0)), (F(0), F(1))]:
        coefficients = [initial[0], initial[1], F(0)]
        for n in range(1, 190):
            coefficients.append(coefficients[n-1]/((n+2)*(n+1)))
        for n in range(190):
            check((n+2)*(n+1)*coefficients[n+2] == (coefficients[n-1] if n else 0), 'exact-Airy-ODE-series')
    # An exactly rational sufficient precision/error check at the proposed
    # large-argument boundary x64: zeta=1024/3, central branch handles<=64.
    coefficient = F(factorial(6*160), factorial(3*160)*factorial(2*160)*576**160)
    check(coefficient/F(64)**(3*160//2) < F(1, 2**540), 'exact-Airy-first-omitted-large-argument-budget')


def decimal_diagnostics():
    """Contour quadrature is a diagnostic, never a certified error enclosure."""
    with localcontext() as context:
        context.prec = 65
        def atan_inverse(q):
            x, term, total = D(1)/q, D(1)/q, D(1)/q
            for k in range(1, 180):
                term *= -x*x
                total += term/(2*k+1)
            return total
        pi = 16*atan_inverse(5)-4*atan_inverse(239)
        def cos(x):
            x = x % (2*pi)
            term = total = D(1)
            for k in range(1, 160):
                term *= -x*x/((2*k)*(2*k-1))
                total += term
            return total
        def sin(x):
            return cos(pi/2-x)
        def simpson(fun, cutoff, count):
            step = cutoff/count
            return step/3*(fun(D(0))+fun(cutoff)+sum((4 if i % 2 else 2)*fun(step*i) for i in range(1, count)))
        rows = []
        sqrt3 = D(3).sqrt()
        for control in map(D, ['-2', '-1', '0', '1', '2', '4']):
            def integrand(r):
                envelope = (-r**3/3+control*r/2).exp()
                angle = sqrt3*control*r/2
                return envelope*(sqrt3*cos(angle)+sin(angle))
            coarse = simpson(integrand, D(8), 512)
            fine = simpson(integrand, D(8), 1024)
            finer = simpson(integrand, D(8), 2048)
            check(abs(finer-fine) < abs(fine-coarse), 'decimal-contour-grid-diagnostic')
            rows.append({'scaledControl': control, 'foldIntegralDiagnostic': finer, 'lastGridDifference': abs(finer-fine), 'certified': False})
        check(rows[0]['foldIntegralDiagnostic'] > 0, 'decimal-no-real-saddle-is-not-zero')
        return rows


def branch_diagnostics():
    with localcontext() as context:
        context.prec = 75
        def dec(x):
            return D(x.numerator)/D(x.denominator)
        def multiply(a, b):
            return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
        def root(z):
            length = (z[0]*z[0]+z[1]*z[1]).sqrt()
            return (((length+z[0])/2).sqrt(), (D(1) if z[1] >= 0 else D(-1))*((length-z[0])/2).sqrt())
        def distance(a, b):
            return ((a[0]-b[0])**2+(a[1]-b[1])**2).sqrt()
        expected = (D('.5').sqrt(), -D('.5').sqrt())
        rows, prior = [], None
        for damping in [F(1, 100), F(1, 10000), F(1, 1000000)]:
            pivots = []
            for i in range(3):
                pivots.append(C(damping, F(-2))+(1/pivots[-1] if i else 0))
            lifted = (D(1), D(0))
            determinant_value = C(F(1))
            for pivot in pivots:
                lifted = multiply(lifted, root((dec(pivot.r), dec(pivot.i))))
                determinant_value *= pivot
            principal = root((dec(determinant_value.r), dec(determinant_value.i)))
            check(distance(lifted, (-principal[0], -principal[1])) < D('1e-60'), 'decimal-regulator-principal-root-sign-counterexample')
            length = (lifted[0]**2+lifted[1]**2).sqrt()
            # N4 free-step factor is exp(-i*pi)=-1.
            normalized_kernel_phase = (-lifted[0]/length, lifted[1]/length)
            error = distance(normalized_kernel_phase, expected)
            if prior is not None:
                check(error < prior, 'decimal-regulator-Abel-phase-convergence')
            prior = error
            rows.append({'damping': damping, 'liftedRoot': lifted, 'principalRoot': principal, 'phaseErrorDiagnostic': error, 'certified': False})
        check(prior < D('0.000002'), 'decimal-free-kernel-phase-limit')
        return rows


def main():
    cases = []
    for n in [2, 3, 4, 6, 8, 16, 31, 32]:
        delta = F(1) if n <= 16 else F(1, 2)
        for v in map(F, ['0', '1/4', '1', '2', '3', '4', '-1']):
            cases.append({'id': f'n{n}-v{v}', 'n': n, 'time': str(delta*n), 'mass': '1', 'hbar': '1/2', 'omegaSquared': str(v/delta**2), 'start': '1/3', 'end': '-1/3' if n % 2 else '1/3'})
    cases += [
        {'id': 'maximum-coarse-hyperbolic', 'n': 2, 'time': '16', 'mass': '16', 'hbar': '1/999999999989', 'omegaSquared': '16', 'start': '-16', 'end': '16'},
        {'id': 'near-caustic-left', 'n': 32, 'time': '16', 'mass': '1/999999999989', 'hbar': '1/999999999983', 'omegaSquared': str(F(8*99999999989-4, 99999999989)), 'start': '-16', 'end': '16'},
        {'id': 'near-caustic-right', 'n': 32, 'time': '16', 'mass': '1/999999999989', 'hbar': '1/999999999983', 'omegaSquared': str(F(8*99999999989+4, 99999999989)), 'start': '-16', 'end': '16'},
        {'id': 'maximum-mixed-fractions', 'n': 32, 'time': '999999999983/999999999989', 'mass': '999999999979/999999999961', 'hbar': '999999999953/999999999929', 'omegaSquared': '-999999999893/999999999883', 'start': '-999999999757/999999999749', 'end': '999999999727/999999999719'},
        {'id': 'caustic-compatible-N2', 'n': 2, 'time': '2', 'mass': '1', 'hbar': '1', 'omegaSquared': '2', 'start': '1', 'end': '-1'},
        {'id': 'caustic-incompatible-N2', 'n': 2, 'time': '2', 'mass': '1', 'hbar': '1', 'omegaSquared': '2', 'start': '1', 'end': '1'},
    ]
    rows = []
    for case in cases:
        rows.append(chain(case))
        (ROOT/'progress.json').write_text(json.dumps({'completed': len(rows), 'total': len(cases), 'lastId': case['id'], 'checks': dict(COUNTS)}, indent=2)+'\n')
        print(json.dumps({'completed': len(rows), 'id': case['id']}), flush=True)
    packet_limit_counterexamples()
    fold_checks()
    diagnostics = decimal_diagnostics()
    branches = branch_diagnostics()
    exact = sum(value for category, value in COUNTS.items() if category.startswith('exact-'))
    report = {'finiteModels': len(rows), 'exactChecksPassed': exact, 'decimalDiagnosticChecksPassed': sum(COUNTS.values())-exact, 'counts': dict(COUNTS), 'rows': rows, 'decimalDiagnostics': diagnostics, 'branchDiagnostics': branches, 'decimalPrecision': {'contour':65,'branch':75}, 'decimalQuadratureCertified': False, 'productionCalculatorImplemented': False, 'wholeGoalComplete': False}
    (ROOT/'derivation-verification.json').write_text(json.dumps(encode(report), indent=2)+'\n')
    print(json.dumps({'finiteModels': len(rows), 'exactChecksPassed': exact, 'decimalDiagnosticChecksPassed': sum(COUNTS.values())-exact}), flush=True)


if __name__ == '__main__':
    main()
