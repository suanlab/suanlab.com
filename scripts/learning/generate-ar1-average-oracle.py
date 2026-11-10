"""Exact Fraction covariance lag sums at actual binary64 rho inputs.
Uses the covariance quadratic form, never the TypeScript innovation formula.
Small n also have a direct double-sum check. Decimal is only a projection
of exact fractions and an independent square root, confirmed at +80 digits.
"""
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
from pathlib import Path
import json, math

cases=[]
def add(name,n,rho):cases.append(dict(id=name,parameters=dict(sampleCount=n,correlationRho=rho),precisionDigits=400,directDoubleSumChecked=n<=17))
for n in [1,2,4]:
    for sign in [-1,1]:add(f'half-rho-n{n}-{sign}',n,sign*.5)
for n,rho,name in [(17,.7,'moderate-positive'),(17,-.7,'moderate-negative'),(512,.9,'long-positive'),(512,-.9,'long-negative'),(512,0.,'long-independent'),(512,1.,'constant-even'),(512,-1.,'alternating-even'),(511,-1.,'alternating-odd'),(1,-1.,'alternating-single'),(1,1.,'constant-single'),(512,math.nextafter(1.,0.),'near-constant'),(512,math.nextafter(-1.,0.),'near-alternating-even'),(511,math.nextafter(-1.,0.),'near-alternating-odd'),(512,float.fromhex('0x0.0000000000001p-1022'),'minimum-positive-long'),(511,-float.fromhex('0x0.0000000000001p-1022'),'minimum-negative-long')]:add(name,n,rho)

def finite_variance(n,rho):
    # Count n-k ordered positions above the diagonal and reflect below it.
    total=F(n);power=F(1)
    for k in range(1,n):
        power*=rho;total+=2*(n-k)*power
    result=total/(n*n)
    assert result>=0
    return result

def exact_case(case):
    n=case['parameters']['sampleCount'];rho=F.from_float(float(case['parameters']['correlationRho']))
    variance=finite_variance(n,rho)
    if case['directDoubleSumChecked']:
        assert variance==sum((rho**abs(i-j) for i in range(n) for j in range(n)),F(0))/(n*n)
    interior=abs(rho)<1;tau=(1+rho)/(1-rho) if interior else None
    result=dict(meanExpectation=F(0),innovationVariance=1-rho*rho,independentMeanVariance=F(1,n),
                meanVariance=variance,totalSumVariance=variance*n*n,varianceInflation=variance*n,
                varianceEquivalentSampleCount=1/variance if variance else 'infinity',
                asymptoticIntegratedCorrelationTime=tau,asymptoticMeanVariance=tau/n if interior else None,
                finiteToAsymptoticVarianceRatio=variance*n/tau if interior else None)
    lags=sorted({0,1,2,3,n//2,n-1}&set(range(n)))
    prefixes=sorted({1,2,3,n//2,n-1,n}&set(range(1,n+1)))
    return result,[(k,rho**k) for k in lags],[(m,finite_variance(m,rho)) for m in prefixes]

def project(exact,digits):
    data,lags,prefixes=exact
    with localcontext() as ctx:
        ctx.prec=digits
        def decimal(v):return D(v.numerator)/D(v.denominator)
        def text(v):
            if v is None or isinstance(v,str):return v
            return '0' if not v else str(decimal(v))
        result={k:text(v) for k,v in data.items()};var=data['meanVariance'];result['standardError']='0' if not var else str(decimal(var).sqrt())
        result['autocovariances']=[dict(lag=k,reference=text(v)) for k,v in lags]
        result['varianceBySampleCount']=[dict(sampleCount=m,reference=text(v)) for m,v in prefixes]
        return result

def values(v):
    if isinstance(v,dict):
        for k,x in v.items():
            if k not in ['lag','sampleCount']:yield from values(x)
    elif isinstance(v,list):
        for x in v:yield from values(x)
    else:yield v

for case in cases:
    exact=exact_case(case);digits=case['precisionDigits'];low=project(exact,digits);high=project(exact,digits+80)
    for x,y in zip(values(low),values(high)):
        if x is None or x=='infinity':assert x==y;continue
        x=D(x);y=D(y);assert abs(x-y)<=abs(y)*D('1e-200')+D(10)**(-digits+100),(case['id'],x,y)
    case.update(confirmationPrecisionDigits=digits+80,reference=high)
result=dict(scope='Exact actual-binary Fraction covariance lag sums; small direct double sums; Decimal projections and square roots',cases=cases)
Path('scripts/fixtures/ar1-average-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),observations=sum(sum(v is not None for v in values(c['reference'])) for c in cases),directDoubleSumCases=sum(c['directDoubleSumChecked'] for c in cases),confirmationExtraDigits=80)))
