"""Independent Fraction/Taylor-ring oracle. No TypeScript or production arithmetic imports.

Dense incidence matrices and Gauss-Jordan elimination differentiate the absolute
partition function itself; the producer instead uses LDL and Gaussian moments.
"""
from fractions import Fraction as F
from itertools import product
import json
import sys
from decimal import Decimal, localcontext


class Taylor:
    """Coefficients through deltaT squared, not raw second derivatives."""
    def __init__(self, a=0, b=0, c=0):
        self.v = tuple(map(F, (a, b, c)))

    @staticmethod
    def cast(x):
        return x if isinstance(x, Taylor) else Taylor(x)

    def __add__(self, x):
        x = self.cast(x)
        return Taylor(*(a+b for a, b in zip(self.v, x.v)))

    __radd__ = __add__

    def __neg__(self):
        return Taylor(*(-x for x in self.v))

    def __sub__(self, x):
        return self + -self.cast(x)

    def __rsub__(self, x):
        return self.cast(x) + -self

    def __mul__(self, x):
        x = self.cast(x)
        return Taylor(*(sum(self.v[k]*x.v[n-k] for k in range(n+1)) for n in range(3)))

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self.cast(x)
        a, b, c = x.v
        return self * Taylor(1/a, -b/a**2, b*b/a**3-c/a**2)

    def __rtruediv__(self, x):
        return self.cast(x) / self


def jet(value, first='0', second='0'):
    return dict(value=str(value), first=str(first), second=str(second))


def read(x):
    return Taylor(F(x['value']), F(x['first']), F(x['second'])/2)


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def inverse(a):
    n = len(a)
    aug = [row[:] + [Taylor(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    det, positive = Taylor(1), True
    for k in range(n):
        if aug[k][k].v[0] == 0:
            positive = False
            pivot_row = next((i for i in range(k+1, n) if aug[i][k].v[0]), None)
            if pivot_row is None:
                return None
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
            det = -det
        pivot = aug[k][k]
        positive = positive and pivot.v[0] > 0
        det *= pivot
        aug[k] = [x/pivot for x in aug[k]]
        for i in range(n):
            if i != k:
                factor = aug[i][k]
                aug[i] = [x-factor*y for x, y in zip(aug[i], aug[k])]
    return [row[n:] for row in aug], det, positive


def oracle(spec):
    shape = spec['shape']
    # First coordinate changes fastest, matching the declared public geometry.
    coords = [tuple(reversed(c)) for c in product(*(range(n) for n in reversed(shape)))]
    index = {c: i for i, c in enumerate(coords)}
    n = len(coords)
    incidence = []
    for c in coords:
        for axis, length in enumerate(shape):
            if c[axis]+1 < length or spec['boundary'] == 'periodic':
                other = list(c)
                other[axis] = (c[axis]+1) % length
                row = [F(0)]*n
                row[index[c]] -= 1
                row[index[tuple(other)]] += 1
                incidence.append(row)
            if spec['boundary'] == 'dirichlet':
                for face in [0, length-1]:
                    if c[axis] == face:
                        row = [F(0)]*n
                        row[index[c]] = F(1)
                        incidence.append(row)
    lap = [[sum(row[i]*row[j] for row in incidence) for j in range(n)] for i in range(n)]
    spacing, temperature = F(spec['spacing']), F(spec['temperature'])
    cell = spacing**len(shape)
    mass, stiffness = read(spec['mass']), read(spec['stiffness'])
    K = [[cell*(mass*int(i == j)+stiffness*lap[i][j]/spacing**2) for j in range(n)] for i in range(n)]
    source = [[cell*read(j)] for j in spec['sourceDensity']]
    constrained = spec['constraint'] == 'mean-zero'
    q = n-int(constrained)
    basis = [[Taylor(int(i == j)-int(constrained and i == n-1)) for j in range(q)] for i in range(n)]
    bt = list(map(list, zip(*basis)))
    A = matmul(matmul(bt, K), basis)
    result = inverse(A)
    if result is None or not result[2]:
        return dict(kind='not-normalizable')
    inv, determinant, _ = result
    determinant /= n if constrained else 1
    G = matmul(matmul(basis, inv), bt)
    mean = matmul(G, source)
    background = n*cell*read(spec['backgroundDensity'])
    minimum = background-sum(source[i][0]*mean[i][0] for i in range(n))/2
    # Differentiate log Z directly in the Taylor ring.
    m_over_T = minimum/Taylor(temperature, 1)
    d0, d1, d2 = determinant.v
    log_d_first = d1/d0
    log_d_second = 2*d2/d0-(d1/d0)**2
    log_z_first = F(q, 2)/temperature-log_d_first/2-m_over_T.v[1]
    log_z_second = -F(q, 2)/temperature**2-log_d_second/2-2*m_over_T.v[2]
    U = temperature**2*log_z_first
    Cv = 2*temperature*log_z_first+temperature**2*log_z_second
    base = lambda m: [[str(x.v[0]) for x in row] for row in m]
    return dict(kind='normalizable', dimension=q, laplacian=[[str(x) for x in row] for row in lap],
                precision=base(K), inverseOnSpace=base(G), mean=[str(row[0].v[0]) for row in mean],
                covariance=[[str(temperature*x.v[0]) for x in row] for row in G],
                restrictedDeterminant=str(d0), minimumEnergy=str(minimum.v[0]),
                expectedEnergy=str(minimum.v[0]+F(q, 2)*temperature), thermalEnergy=str(U), heatCapacity=str(Cv),
                logPartition=dict(dimension=q, temperature=str(temperature), restrictedDeterminant=str(d0),
                                  referenceScale=str(F(spec['referenceScale'])), constant=str(-minimum.v[0]/temperature)))


def spec(shape, boundary='open', constraint='none', **changes):
    n = 1
    for size in shape:
        n *= size
    result = dict(shape=shape, boundary=boundary, constraint=constraint, spacing='3/2', temperature='4/3', referenceScale='5/4',
                  mass=jet('2/3', '-1/5', '1/7'), stiffness=jet('3/5', '1/11', '-1/13'),
                  backgroundDensity=jet('1/3', '-2/7', '3/11'),
                  sourceDensity=[jet(str(F(i % 3-1, 3)), str(F(i % 2, 5)), str(F(i % 4-2, 7))) for i in range(n)])
    result.update(changes)
    return result


def cases():
    rows = []
    for shape in [[2], [3], [4], [7], [2, 2], [2, 3], [3, 3]]:
        for boundary in ['open', 'periodic', 'dirichlet']:
            for constraint in ['none', 'mean-zero']:
                rows.append(spec(shape, boundary, constraint))
    rows.append(spec([4, 8], 'periodic', 'mean-zero'))
    for constraint in ['none', 'mean-zero']:
        rows.append(spec([3], 'periodic', constraint, mass=jet('0')))
        rows.append(spec([2], 'periodic', constraint, spacing='1', mass=jet('-1'), stiffness=jet('1')))
    rows.append(spec([2], 'open', spacing='1', mass=jet('-1'), stiffness=jet('1')))  # zero pivot but invertible indefinite K
    for r in [jet('1'), jet('2', '1'), jet('4', '4', '2')]:
        rows.append(spec([2], spacing='1', temperature='2', mass=r, stiffness=jet('0'),
                         backgroundDensity=jet('0'), sourceDensity=[jet('0'), jet('0')]))
    # Source derivatives and background offsets also exercise completion of the square.
    rows.append(spec([2], 'dirichlet', mass=jet('-1/2'), stiffness=jet('2')))
    return [dict(input=x, expected=oracle(x)) for x in rows]


def observation_diagnostic(exact):
    """High-precision Decimal cross-check, not an interval proof."""
    if exact['kind'] != 'normalizable':
        return None
    with localcontext() as ctx:
        ctx.prec = 230
        # Independent Gauss-Legendre pi iteration, rather than producer/checker arctangents.
        a, b, t, p = Decimal(1), Decimal(1)/Decimal(2).sqrt(), Decimal(1)/4, Decimal(1)
        for _ in range(9):
            next_a = (a+b)/2
            b = (a*b).sqrt()
            t -= p*(a-next_a)**2
            a = next_a
            p *= 2
        pi = (a+b)**2/(4*t)
        def decimal(s):
            q = F(s)
            return Decimal(q.numerator)/Decimal(q.denominator)
        form = exact['logPartition']
        T = decimal(form['temperature'])
        log_z = Decimal(form['dimension'])/2*((2*pi).ln()+T.ln())-decimal(form['restrictedDeterminant']).ln()/2
        log_z -= Decimal(form['dimension'])*decimal(form['referenceScale']).ln()
        log_z += decimal(form['constant'])
        return dict(logPartition=str(log_z), freeEnergy=str(-T*log_z),
                    thermodynamicEntropy=str(log_z+decimal(exact['thermalEnergy'])/T),
                    gibbsEntropy=str(log_z+decimal(exact['expectedEnergy'])/T))


if __name__ == '__main__':
    # Guard the independent Taylor implementation before it generates references.
    t = Taylor(2, 1)
    assert ((t*t)/t).v == t.v
    assert (1/t).v == (F(1, 2), F(-1, 4), F(1, 8))
    rows = [dict(input=x, expected=oracle(x)) for x in json.load(sys.stdin)] if '--inputs' in sys.argv else cases()
    if '--observations' in sys.argv:
        for row in rows:
            row['observationDiagnostic'] = observation_diagnostic(row['expected'])
    print(json.dumps(dict(method='Fraction Taylor-ring Gauss-Jordan and absolute log-Z derivatives', cases=rows), indent=2))
