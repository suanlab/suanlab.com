"""Independent finite-dimensional saddle identities; no TypeScript/runtime imports.

Exact Fraction identities and certified algebraic bounds are separated from
Decimal quadrature, whose observed grid convergence is not a certified integral.
Run from repository root; evidence is written to the ignored .runtime directory.
"""
from collections import Counter
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
from pathlib import Path
import json

ROOT = Path('.runtime/learning/saddle-fluctuation-scope')
COUNTS = Counter()

def check(condition, category):
    assert condition, category
    COUNTS[category] += 1

def decimal(x):
    x = F(x)
    return D(x.numerator) / D(x.denominator)

def encode(x):
    if isinstance(x, F):
        return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'
    if isinstance(x, D):
        return str(x)
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    return x

def inverse(a):
    n = len(a)
    v = [[F(x) for x in row] + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    determinant = F(1)
    for i in range(n):
        pivot = v[i][i]
        assert pivot > 0  # all matrices here are positive definite, no row swaps
        determinant *= pivot
        v[i] = [x / pivot for x in v[i]]
        for j in range(n):
            if j != i:
                factor = v[j][i]
                v[j] = [a - factor*b for a, b in zip(v[j], v[i])]
    return [row[n:] for row in v], determinant

def model(n, time=F(1), mass=F(1), omega_squared=F(1), coupling=F(1), start=F(0), end=F(0)):
    return dict(n=n, time=F(time), mass=F(mass), omegaSquared=F(omega_squared), coupling=F(coupling), start=F(start), end=F(end))

def action(p, q):
    delta = p['time'] / p['n']
    x = [p['start'], *q, p['end']]
    potential = lambda a: p['mass']*p['omegaSquared']*a*a/2 + p['coupling']*a**4/4
    return p['mass']*sum((b-a)**2 for a, b in zip(x, x[1:]))/(2*delta) + delta*(potential(x[0])/2 + sum(potential(a) for a in q) + potential(x[-1])/2)

def derivatives(p, q):
    delta = p['time']/p['n']; s = p['mass']/delta
    x = [p['start'], *q, p['end']]
    g = [s*(2*a-x[i]-x[i+2]) + delta*(p['mass']*p['omegaSquared']*a+p['coupling']*a**3) for i, a in enumerate(q)]
    h = [[(2*s+delta*p['mass']*p['omegaSquared']+3*delta*p['coupling']*q[i]**2) if i == j else -s if abs(i-j) == 1 else F(0) for j in range(len(q))] for i in range(len(q))]
    return g, h

def moment(c, powers):
    @lru_cache(None)
    def visit(e):
        if not any(e):
            return F(1)
        if sum(e) % 2:
            return F(0)
        i = next(i for i, x in enumerate(e) if x)
        a = list(e); a[i] -= 1
        value = F(0)
        for j, multiplicity in enumerate(a):
            if multiplicity:
                b = a.copy(); b[j] -= 1
                value += multiplicity*c[i][j]*visit(tuple(b))
        return value
    return visit(tuple(powers))

def pair_moment(c, i, j, power):
    return moment([[c[i][i], c[i][j]], [c[j][i], c[j][j]]], (power, power))

def coefficients(p, q, c):
    n = len(q); a = p['time']/p['n']*p['coupling']
    quartic = 3*a*sum(c[i][i]**2 for i in range(n))/4
    cubic = a*a*sum(q[i]*q[j]*(9*c[i][i]*c[j][j]*c[i][j]+6*c[i][j]**3) for i in range(n) for j in range(n))/2
    w2 = sum(9*c[i][i]**2*c[j][j]**2+72*c[i][i]*c[j][j]*c[i][j]**2+24*c[i][j]**4 for i in range(n) for j in range(n))
    return dict(cubicContribution=cubic, quarticContribution=-quartic, firstNonGaussian=cubic-quartic, zeroSaddleSecondOrderUpperCoefficient=a*a*w2/32)

def exact_cases():
    rows = []
    for n in [2, 3, 4, 7, 8, 16, 32]:
        for omega in [F(0), F(1), F(7, 3)]:
            p = model(n, time=F(3, 2), mass=F(5, 4), omega_squared=omega, coupling=F(2, 3))
            q = [F(0)]*(n-1); g, h = derivatives(p, q); c, det = inverse(h); co = coefficients(p, q, c)
            check(all(v == 0 for v in g), 'zero-saddle-stationarity')
            check(det > 0 and co['cubicContribution'] == 0 and co['quarticContribution'] < 0, 'gaussian-and-correction-signs')
            for i in range(n-1):
                for j in range(n-1):
                    check(sum(h[i][k]*c[k][j] for k in range(n-1)) == F(i == j), 'full-inverse-identity')
                    if n <= 8 or (i, j) in [(0, 0), (0, n-2), (n//2-1, n//2-1)]:
                        check(pair_moment(c, i, j, 4) == 9*c[i][i]**2*c[j][j]**2+72*c[i][i]*c[j][j]*c[i][j]**2+24*c[i][j]**4, 'eighth-moment-independent-recursion')
                        check(pair_moment(c, i, j, 3) == 9*c[i][i]*c[j][j]*c[i][j]+6*c[i][j]**3, 'sixth-moment-independent-recursion')
            if omega == 0:
                expected = p['coupling']*p['time']**3/(40*p['mass']**2)*(1-F(1, n**4))
                check(-co['quarticContribution'] == expected, 'free-chain-first-correction-closed-form')
            rows.append(dict(input=p, minimumAction=action(p, q), determinant=det, coefficients=co))
    # Quartic expansion about an arbitrary point retains a linear residual.
    for n in [2, 3, 5, 8]:
        p = model(n, F(4, 3), F(2, 3), F(3, 2), F(5, 7), F(-2, 3), F(5, 4))
        q = [F(i-2, 5) for i in range(n-1)]
        for shift in [F(1, 7), F(-2, 9), F(3, 2)]:
            u = [(-1)**i*shift*(i+1) for i in range(n-1)]; g, h = derivatives(p, q)
            expansion = action(p, q)+sum(a*b for a, b in zip(g, u))+sum(u[i]*h[i][j]*u[j] for i in range(n-1) for j in range(n-1))/2+p['time']/n*p['coupling']*sum(q[i]*u[i]**3+u[i]**4/4 for i in range(n-1))
            check(action(p, [a+b for a, b in zip(q, u)]) == expansion, 'complete-shift-polynomial-identity')
    # A manufactured exact stationary point for the uniform, nonlinear potential.
    p = model(2, end=F(37, 32)); q = [F(1, 2)]; g, h = derivatives(p, q); c, det = inverse(h); co = coefficients(p, q, c)
    check(g == [0] and h == [[F(39, 8)]], 'nonzero-exact-saddle')
    check(co['firstNonGaussian'] == F(-232, 19773) and co['cubicContribution'] == F(80, 19773), 'nonzero-cubic-correction')
    check(p['time']/p['n']*p['coupling']*(q[0]*F(-1, 2)**3+F(-1, 2)**4/4) == F(-3, 128), 'shifted-remainder-can-be-negative')
    # Residual certificates, action bounds and inverse/determinant brackets against an exact root.
    certificate_rows = []
    mu = p['mass']/(p['time']/p['n']*p['n']**2) + p['time']/p['n']*p['mass']*p['omegaSquared']
    for perturbation in [F(1, 10), F(-1, 10), F(1, 1000), F(-1, 10**8)]:
        qhat = [q[0]+perturbation]; residual, hh = derivatives(p, qhat); rr, dhat = inverse(hh); rho = max(map(abs, residual)); error = len(q)*rho/mu
        action_error = len(q)*rho*rho/(2*mu)
        eta = 3*(p['time']/p['n'])*p['coupling']*error*(2*max(map(abs, qhat))+error)
        check(abs(qhat[0]-q[0]) <= error, 'strong-convexity-position-certificate')
        check(0 <= action(p, qhat)-action(p, q) <= action_error, 'strong-convexity-action-certificate')
        check(abs(rr[0][0]-c[0][0]) <= eta/(mu*mu), 'resolvent-covariance-certificate')
        if eta < mu:
            check(hh[0][0]-eta <= h[0][0] <= hh[0][0]+eta, 'loewner-determinant-bracket')
        certificate_rows.append(dict(perturbation=perturbation, residual=residual, radius=error, actionErrorBound=action_error, hessianErrorBound=eta))
    rows.append(dict(input=p, exactSaddle=q, minimumAction=action(p, q), determinant=det, coefficients=co, residualCertificates=certificate_rows))
    # Coupled nonlinear exact saddles, manufactured from a symmetric midpoint.
    # Their endpoint fractions still fit the proposed 12-digit input grammar.
    for n in [3, 4, 6]:
        middle = n//2
        xs = [F(0)]*(n+1); xs[middle] = F(1, 2)
        if n % 2:
            xs[middle+1] = xs[middle]
        else:
            xs[middle+1] = xs[middle]+xs[middle]**3/2
        for i in range(middle, 0, -1):
            xs[i-1] = 2*xs[i]-xs[i+1]+xs[i]**3
        for i in range(middle+1, n+1):
            xs[i] = xs[n-i]
        pp = model(n, time=F(n), omega_squared=F(0), start=xs[0], end=xs[-1]); qq = xs[1:-1]
        gg, hh = derivatives(pp, qq); rr, dd = inverse(hh); cc = coefficients(pp, qq, rr)
        check(all(v == 0 for v in gg), 'coupled-nonlinear-exact-saddle')
        check(len(str(xs[0].numerator)) <= 12 and len(str(xs[0].denominator)) <= 12 and abs(xs[0]) <= 16, 'manufactured-endpoint-within-input-grammar')
        for i in range(n-1):
            for j in range(n-1):
                check(sum(hh[i][k]*rr[k][j] for k in range(n-1)) == F(i == j), 'coupled-nonlinear-full-inverse')
                check(pair_moment(rr, i, j, 3) == 9*rr[i][i]*rr[j][j]*rr[i][j]+6*rr[i][j]**3, 'coupled-nonlinear-cubic-contractions')
        certificates = []
        mu = F(1, n*n)
        for shift in [F(1, 10**6), F(-1, 10**9)]:
            qhat = [a+(i+1)*shift for i, a in enumerate(qq)]; residual, hhat = derivatives(pp, qhat); rhat, _ = inverse(hhat)
            rho = max(map(abs, residual)); error = (n-1)*rho/mu; error_action = (n-1)*rho*rho/(2*mu)
            eta = 3*error*(2*max(map(abs, qhat))+error)
            check(max(abs(a-b) for a, b in zip(qq, qhat)) <= error, 'coupled-position-certificate')
            check(0 <= action(pp, qhat)-action(pp, qq) <= error_action, 'coupled-action-certificate')
            check(all(abs(rr[i][j]-rhat[i][j]) <= eta/(mu*mu) for i in range(n-1) for j in range(n-1)), 'coupled-full-covariance-certificate')
            if eta < mu:
                minus = [[v-F(i == j)*eta for j, v in enumerate(row)] for i, row in enumerate(hhat)]
                plus = [[v+F(i == j)*eta for j, v in enumerate(row)] for i, row in enumerate(hhat)]
                _, lower = inverse(minus); _, upper = inverse(plus)
                check(lower <= dd <= upper, 'coupled-determinant-certificate')
            certificates.append(dict(residualRadius=error, actionErrorBound=error_action, hessianErrorBound=eta))
        rows.append(dict(input=pp, exactSaddle=qq, minimumAction=action(pp, qq), determinant=dd, coefficients=cc, residualCertificates=certificates))
    p_positive = model(2, end=F(13, 2)); q_positive = [F(2)]
    g_positive, h_positive = derivatives(p_positive, q_positive); r_positive, det_positive = inverse(h_positive)
    positive = coefficients(p_positive, q_positive, r_positive)
    check(g_positive == [0] and positive['firstNonGaussian'] > 0, 'convex-nonzero-saddle-positive-first-correction')
    rows.append(dict(input=p_positive, exactSaddle=q_positive, determinant=det_positive, coefficients=positive))
    # N=2 Gaussian reference connects to preceding Euclidean kernel normalization.
    p0 = model(2); _, h0 = derivatives(p0, [F(0)]); c0, det0 = inverse(h0); co0 = coefficients(p0, [F(0)], c0)
    check(c0 == [[F(2, 9)]] and co0['firstNonGaussian'] == F(-1, 54) and co0['zeroSaddleSecondOrderUpperCoefficient'] == F(35, 17496), 'harmonic-one-variable-hand-coefficients')
    check(p0['mass']**2/(2*(p0['time']/2)**2*det0) == F(4, 9), 'preceding-kernel-prefactor-normalization')
    # Nondegenerate minima/maximum of the one-coordinate double well.
    for kappa in [F(-4), F(-1), F(0), F(1, 4), F(1), F(4)]:
        if kappa > 0:
            square = kappa; curvature = 2*kappa
            check(square* square/4-kappa*square/2 == -kappa*kappa/4, 'double-well-minimum-action')
            check(F(15, 2)*square/curvature**3-F(3, 4)/curvature**2 == F(3, 4)/(kappa*kappa), 'double-well-first-correction')
            check(-kappa < 0 and curvature > 0, 'maximum-excluded-from-positive-gaussian')
        elif kappa < 0:
            check(-kappa > 0, 'single-well-positive-curvature')
        else:
            check(-kappa == 0, 'coalescence-gaussian-unavailable')
    # Quartic zero-dimensional coefficients and term-growth identity, not an infinite-sum interchange.
    df = lambda n: 1 if n < 1 else n*df(n-2)
    terms = [F((-1)**n*df(4*n-1), 4**n*factorial(n)) for n in range(17)]
    for n in range(16):
        check(abs(terms[n+1]/terms[n]) == F((4*n+1)*(4*n+3), 4*(n+1)), 'quartic-asymptotic-term-ratio')
    return rows

PI = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
def simpson(function, radius, count):
    step = 2*radius/count
    return step/3*(function(-radius)+function(radius)+sum((4 if i % 2 else 2)*function(-radius+step*i) for i in range(1, count)))

def quadrature_cases():
    rows = []
    with localcontext() as context:
        context.prec = 80
        # No Decimal output here is advertised as an outward-rounded integral bound.
        for saddle in [F(0), F(1, 2), F(2)]:
            nonzero = bool(saddle)
            p = model(2, end=(9*saddle+saddle**3)/4); q = [saddle]; _, hessian = derivatives(p, q); cov, _ = inverse(hessian); co = coefficients(p, q, cov)
            errors = []
            for hb in [F(1, 4), F(1, 16), F(1, 64)]:
                h = decimal(hb); r = decimal(cov[0][0]); a = decimal(p['time']/p['n']*p['coupling']); center = decimal(q[0]); b3 = a*center*h.sqrt()*r*r.sqrt(); b4 = a*h*r*r/4
                f = lambda z: (-z*z/2-b3*z**3-b4*z**4).exp()
                coarse = simpson(f, D(12), 1024)/(2*PI).sqrt(); fine = simpson(f, D(12), 2048)/(2*PI).sqrt()
                check(abs(fine-coarse) < D('1e-24'), 'decimal-quadrature-grid-agreement')
                coefficient_error = abs((fine-1)/h-decimal(co['firstNonGaussian'])); errors.append(coefficient_error)
                row = dict(kind='nonzero-cubic-saddle' if nonzero else 'zero-saddle-positive-quartic', saddle=saddle, hbar=hb, ratioApproximation=fine, gridDifference=abs(fine-coarse), firstCoefficient=co['firstNonGaussian'], observedCoefficientError=coefficient_error)
                if not nonzero:
                    lower = max(F(0), 1+hb*co['firstNonGaussian']); upper = min(F(1), 1+hb*co['firstNonGaussian']+hb*hb*co['zeroSaddleSecondOrderUpperCoefficient'])
                    check(decimal(lower)-D('1e-24') <= fine <= decimal(upper)+D('1e-24'), 'positive-quartic-bound-numerical-crosscheck')
                    row.update(exactRatioLower=lower, exactRatioUpper=upper)
                rows.append(row)
            check(errors[2] < errors[1] < errors[0], 'nonGaussian-coefficient-small-hbar-convergence')
        for kappa in [F(-1), F(0), F(1)]:
            errors = []
            for hb in [F(1, 20), F(1, 40), F(1, 80)]:
                h = decimal(hb); k = decimal(kappa); minimum = -k*k/4 if kappa > 0 else D(0)
                f = lambda x: (-(x**4/4-k*x*x/2-minimum)/h).exp()
                coarse = simpson(f, D(12), 2048); fine = simpson(f, D(12), 4096)
                check(abs(fine-coarse) < D('1e-22'), 'well-quadrature-grid-agreement')
                row = dict(kind='one-coordinate-well', hbar=hb, kappa=kappa, integralTimesExpMinimumApproximation=fine, gridDifference=abs(fine-coarse))
                if kappa:
                    minima = 2 if kappa > 0 else 1; curvature = 2*k if kappa > 0 else -k; leading = minima*(2*PI*h/curvature).sqrt(); ratio = fine/leading; c1 = F(3, 4)/(kappa*kappa) if kappa > 0 else -F(3, 4)/(kappa*kappa)
                    errors.append(abs((ratio-1)/h-decimal(c1))); row.update(minimaIncluded=minima, gaussianRatioApproximation=ratio, firstCoefficient=c1)
                else:
                    row.update(quarticScaledIntegralApproximation=fine/h.sqrt().sqrt(), gaussianAvailable=False)
                rows.append(row)
            if kappa:
                check(errors[2] < errors[1] < errors[0], 'well-first-coefficient-small-hbar-convergence')
        critical = [x['quarticScaledIntegralApproximation'] for x in rows if x.get('kappa') == 0]
        check(max(critical)-min(critical) < D('1e-22'), 'degenerate-quartic-hbar-quarter-scaling')
    return rows

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    exact = exact_cases(); numerical = quadrature_cases()
    numerical_categories = {'decimal-quadrature-grid-agreement', 'positive-quartic-bound-numerical-crosscheck', 'nonGaussian-coefficient-small-hbar-convergence', 'well-quadrature-grid-agreement', 'well-first-coefficient-small-hbar-convergence', 'degenerate-quartic-hbar-quarter-scaling'}
    decimal_checks = sum(COUNTS[k] for k in numerical_categories)
    output = dict(exactChecks=sum(COUNTS.values())-decimal_checks, decimalChecks=decimal_checks, stage='mathematical-foundation-verified-production-pending', exactModels=len(exact), decimalRows=len(numerical), checks=sum(COUNTS.values()), checksByCategory=dict(COUNTS), decimalPrecision=80, decimalQuadratureCertified=False, positiveQuarticBoundBasis='Pointwise Taylor inequality for nonnegative quartic remainder; derivation in model contract.', formalProofAssistantUsed=False, generalShiftedRemainderNonnegative=False, productionNumericsImplemented=False, readerRegistered=False, originalTopicComplete=False, wholeGoalComplete=False, exactCases=exact, decimalCases=numerical)
    (ROOT/'derivations.json').write_text(json.dumps(encode(output), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ['exactCases', 'decimalCases']}, indent=2))

if __name__ == '__main__':
    main()
