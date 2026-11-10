"""Independent input-only references, with exact a-posteriori certificates.

Decimal dense Newton uses a zero seed and a squared-residual line search.
Gauss-Jordan constructs an inverse candidate. Fraction residual/resolvent and
Loewner bounds certify the candidates; Decimal convergence alone is not proof.
No TypeScript calculator or producer result is executed/read.
"""
from decimal import Decimal as D, localcontext, ROUND_HALF_EVEN
from fractions import Fraction as F
from math import factorial, isqrt
from pathlib import Path
import json
import sys
import time

sys.dont_write_bytecode = True
sys.set_int_max_str_digits(131072)
BITS = 1024
SCALE = 1 << BITS
ULP = F(1, SCALE)
ROOT = Path('.runtime/learning/saddle-fluctuation-numerics')

def text(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f'{x.numerator}/{x.denominator}'

def fixed(x):
    return int((x*D(SCALE)).to_integral_value(rounding=ROUND_HALF_EVEN))

def fixed_fraction(x):
    return (2*x.numerator*SCALE+x.denominator)//(2*x.denominator)

def dec(x):
    x = F(x)
    return D(x.numerator)/D(x.denominator)

def rounded_range(lower, upper):
    lo = (lower.numerator*SCALE)//lower.denominator
    hi = -((-upper.numerator*SCALE)//upper.denominator)
    assert lo <= hi
    return dict(lower=text(F(lo, SCALE)), upper=text(F(hi, SCALE)))

def dense_solve(matrix, rhs):
    n = len(rhs)
    a = [list(row)+[rhs[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = a[i][i]
        assert pivot > 0
        for j in range(i+1, n):
            if not a[j][i]:
                continue
            factor = a[j][i]/pivot
            for k in range(i+1, n+1):
                a[j][k] -= factor*a[i][k]
            a[j][i] = D(0)
    answer = [D(0)]*n
    for i in reversed(range(n)):
        answer[i] = (a[i][-1]-sum(a[i][j]*answer[j] for j in range(i+1, n)))/a[i][i]
    return answer

def dense_inverse(matrix):
    n = len(matrix)
    a = [list(row)+[D(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = a[i][i]
        assert pivot > 0
        a[i] = [x/pivot for x in a[i]]
        for j in range(n):
            if j != i:
                factor = a[j][i]
                a[j] = [x-factor*y for x, y in zip(a[j], a[i])]
    return [row[n:] for row in a]

def exact_linear(diag, start, end, n):
    # Ordinary forward row elimination, not the producer's LDL representation.
    a = [diag]*(n-1); rhs = [F(0)]*(n-1)
    rhs[0] += start; rhs[-1] += end
    for i in range(1, n-1):
        a[i] -= 1/a[i-1]
        rhs[i] += rhs[i-1]/a[i-1]
    q = [F(0)]*(n-1)
    for i in reversed(range(n-1)):
        q[i] = (rhs[i]+(q[i+1] if i+1<n-1 else 0))/a[i]
    return q

def gaussian_pair(a, b, c, left, right):
    # Coefficients of exp(a*t^2/2+c*t*u+b*u^2/2), not hard-coded Wick polynomials.
    total = F(0)
    for cross in range(min(left, right)+1):
        if (left-cross) % 2 or (right-cross) % 2:
            continue
        i, j = (left-cross)//2, (right-cross)//2
        count = F(factorial(left)*factorial(right), 2**(i+j)*factorial(i)*factorial(j)*factorial(cross))
        total += count*a**i*b**j*c**cross
    return total

def chain(input):
    n = input['intervals']; d = n-1
    tau, mass, hbar, omega, lam, start, end = [F(input[k]) for k in ['time','mass','hbar','omegaSquared','quartic','start','end']]
    delta = tau/n; physical_scale = mass/delta; v = omega*delta**2; w = lam*delta**2/mass; mu = F(1,n*n)+v
    def residual(q):
        x = [start,*q,end]
        return [2*a-x[i]-x[i+2]+v*a+w*a**3 for i,a in enumerate(q)]
    iterations = 0
    if lam == 0:
        q = exact_linear(2+v,start,end,n)
    else:
        vd, wd, ad, bd = map(dec,[v,w,start,end])
        def dg(q):
            x=[ad,*q,bd]
            return [2*a-x[i]-x[i+2]+vd*a+wd*a**3 for i,a in enumerate(q)]
        qd=[D(0)]*d
        for iterations in range(256):
            g=dg(qd)
            if max(map(abs,g)) < D('1e-310'):
                break
            h=[[2+vd+3*wd*qd[i]**2 if i==j else D(-1) if abs(i-j)==1 else D(0) for j in range(d)] for i in range(d)]
            direction=dense_solve(h,[-x for x in g]); old=sum(x*x for x in g); alpha=D(1)
            for _ in range(128):
                nextq=[a+alpha*b for a,b in zip(qd,direction)]; new=sum(x*x for x in dg(nextq))
                if new <= (1-alpha/D(4))*old:
                    qd=nextq; break
                alpha/=2
            else:
                raise AssertionError('Independent squared-residual search failed')
        else:
            raise AssertionError('Independent Newton budget exhausted')
        q=[F(fixed(x),SCALE) for x in qd]
    g=residual(q); rho=max(map(abs,g)); error=d*rho/mu
    assert error < F(1,1<<880), 'Reference root certificate too wide'
    diagonal=[2+v+3*w*x*x for x in q]
    h=[[diagonal[i] if i==j else F(-1) if abs(i-j)==1 else F(0) for j in range(d)] for i in range(d)]
    inverse_decimal=dense_inverse([[dec(x) for x in row] for row in h])
    ints=[[fixed(x) for x in row] for row in inverse_decimal]
    rhat=[[F(x,SCALE) for x in row] for row in ints]
    defects=[[sum(h[i][k]*rhat[k][j] for k in range(max(0,i-1),min(d,i+2)))-F(i==j) for j in range(d)] for i in range(d)]
    operator_defect=max(max(sum(abs(x) for x in row) for row in defects),max(sum(abs(defects[i][j]) for i in range(d)) for j in range(d)))
    eta=3*w*error*(2*max(map(abs,q))+error)
    inverse_error=operator_defect/mu+eta/(mu*mu)
    assert inverse_error < F(1,1<<780), 'Reference inverse certificate too wide'
    # Exact continuants at the rational reference candidate; a Loewner
    # perturbation and a binomial/geometric bound enclose the true determinant.
    previous, determinant=F(1),diagonal[0]
    pivots=[determinant]
    for a in diagonal[1:]:
        previous,determinant=determinant,a*determinant-previous
        pivots.append(determinant/previous)
    relative=eta/mu
    assert d*relative <= F(1,2)
    determinant_lower=determinant*(1-d*relative)
    determinant_upper=determinant*(1+2*d*relative)
    prefactor_lower=physical_scale/(2*hbar*determinant_upper)
    prefactor_upper=physical_scale/(2*hbar*determinant_lower)
    def potential(x):return mass*omega*x*x/2+lam*x**4/4
    def action(q):
        path=[start,*q,end]
        return physical_scale*sum((b-a)**2 for a,b in zip(path,path[1:]))/2+delta*((potential(start)+potential(end))/2+sum(potential(x) for x in q))
    upper_action=action(q); action_error=physical_scale*d*rho*rho/(2*mu)
    lower_action=max(mass*(end-start)**2/(2*tau)+delta*(potential(start)+potential(end))/2,upper_action-action_error)
    rr=[[x/physical_scale for x in row] for row in rhat]; re=inverse_error/physical_scale
    qmax=max(map(abs,q))+error; rmax=max(abs(x) for row in rr for x in row)+re; coupling=delta*lam
    m1=sum(gaussian_pair(rr[i][i],rr[i][i],rr[i][i],4,0) for i in range(d)); m2=F(0); c3=F(0)
    for i in range(d):
        for j in range(d):
            c3+=coupling**2*q[i]*q[j]*gaussian_pair(rr[i][i],rr[j][j],rr[i][j],3,3)/2
            m2+=gaussian_pair(rr[i][i],rr[j][j],rr[i][j],4,4)
    c4=-coupling*m1/4
    m1e=6*d*rmax*re; m2e=420*d*d*rmax**3*re
    c3e=coupling**2*d*d*(30*qmax*rmax**3*error+45*qmax*qmax*rmax*rmax*re)/2
    c4e=coupling*m1e/4
    zero_saddle=rho==0 and all(x==0 for x in q)
    bounds=None
    if lam==0:
        bounds=dict(lower='1',upper='1')
    elif zero_saddle:
        epsilon=hbar*coupling/4
        lower=max(F(0),1-epsilon*(m1+m1e))
        upper=min(F(1),1-epsilon*max(F(0),m1-m1e)+epsilon**2*(m2+m2e)/2)
        bounds=rounded_range(lower,upper)
    return dict(kind='chain',referenceIterations=iterations,stationary=dict(point=list(map(text,q)),errorUpper=text(error),exact=error==0,action=dict(lower=text(lower_action),upper=text(upper_action)),actionExact=text(upper_action) if error==0 else None),
                gaussian=dict(normalizedDiagonal=[str(fixed_fraction(x)) for x in diagonal],normalizedPivots=[str(fixed_fraction(x)) for x in pivots],normalizedPivotErrorUpper=text(ULP+eta*(1+1/(mu*mu))),normalizedDiagonalErrorUpper=text(eta+ULP),inverseNormalized=[[str(x) for x in row] for row in ints],inverseNormalizedErrorUpper=text(inverse_error),determinantNormalized=rounded_range(determinant_lower,determinant_upper),prefactorSquaredTimesPi=rounded_range(prefactor_lower,prefactor_upper)),
                correction=dict(cubic=rounded_range(max(F(0),c3-c3e),c3+c3e),quartic=rounded_range(c4-c4e,min(F(0),c4+c4e)),first=rounded_range(c3+c4-c3e-c4e,c3+c4+c3e+c4e),gaussianFourthSum=rounded_range(max(F(0),m1-m1e),m1+m1e),gaussianFourthSumSquared=rounded_range(max(F(0),m2-m2e),m2+m2e)),ratio=bounds)

def root_range(value, degree):
    target=value.numerator*SCALE**degree//value.denominator
    low=isqrt(target) if degree==2 else isqrt(isqrt(target))
    high=low if low**degree*value.denominator==value.numerator*SCALE**degree else low+1
    return dict(lower=text(F(low,SCALE)),upper=text(F(high,SCALE)))

def well(input):
    h,k=F(input['hbar']),F(input['kappa']); positive=k>0
    c1=F(3 if positive else -3,4)/(k*k) if k else None
    return dict(kind='well',regime='double-well' if positive else 'single-well' if k<0 else 'degenerate-quartic',positiveRoot=root_range(k,2) if positive else None,minimumAction=text(-k*k/4 if positive else 0),firstCoefficient=text(c1) if c1 is not None else None,sumPrefactorSquaredDividedByPi=text((4 if positive else 2)*h/abs(k)) if k else None,criticalQuarticWidthScale=root_range(h,4))

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path('scripts/fixtures/saddle-fluctuation-reference-inputs.json').read_text())['cases']; rows=[]
    with localcontext() as context:
        context.prec=350
        for case in cases:
            started=time.perf_counter(); expected=chain(case['input']) if case['input']['mode']=='chain' else well(case['input'])
            rows.append(dict(id=case['id'],input=case['input'],expected=expected))
            progress=dict(completed=len(rows),total=len(cases),lastId=case['id'],seconds=time.perf_counter()-started)
            (ROOT/'python-reference-progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    output=dict(method='Independent Decimal350 dense Newton/zero seed/squared-residual search and Gauss-Jordan;exact Fraction root residual,resolvent,inverse residual and Loewner certificates;Gaussian generating-function coefficients.',referenceBits=BITS,decimalPrecision=350,certificatesAreExactRational=True,cases=rows)
    target=Path('scripts/fixtures/saddle-fluctuation-references.json');target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),bytes=target.stat().st_size)),flush=True)

if __name__=='__main__':main()
