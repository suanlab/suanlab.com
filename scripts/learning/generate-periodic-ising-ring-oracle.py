"""Actual-binary independent Decimal transfer traces and ring references.
Transfer traces are checked against a closed formula where cancellation fits
within the stated decimal budget. Extreme tiny long-range tails use the
Decimal closed formula explicitly; they are not claimed as transfer checks.
"""
from decimal import Decimal as D, localcontext
from itertools import product
from pathlib import Path
import json, math

cases=[]
def add(name,N,K,digits=400,transfer=True,enumerate_states=False):
    cases.append(dict(id=name,parameters=dict(spinCount=N,couplingK=K),precisionDigits=digits,transferChecked=transfer,enumerationChecked=enumerate_states))
for N in [3,4,5,8]:
    for sign in [-1,1]:add(f'actual-ln3-half-N{N}-{sign}',N,sign*math.log(3)/2,enumerate_states=True)
add('independent-small',3,0.,enumerate_states=True)
add('independent-maximum-support',128,0.)
add('ferro-maximum-input',128,8.)
add('antiferro-even-maximum-input',128,-8.)
add('antiferro-odd-maximum-input',127,-8.)
add('weak-long-ring',128,.001)
add('negative-weak-long-ring',127,-.001)
add('decimal-small-coupling-long-ring',128,1e-10,900)
tiny=float.fromhex('0x0.0000000000001p-1022')
add('minimum-positive-three',3,tiny,900,enumerate_states=True)
add('minimum-negative-five',5,-tiny,1200,enumerate_states=True)
add('minimum-positive-maximum-ring',128,tiny,900,transfer=False)
add('minimum-negative-odd-large-ring',127,-tiny,900,transfer=False)

def matmul(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def decimal_reference(case,digits):
    with localcontext() as ctx:
        ctx.prec=digits
        N=case['parameters']['spinCount'];K=D.from_float(float(case['parameters']['couplingK']))
        e=(2*K).exp();t=(e-1)/(e+1);pn=t**N
        C=[D(1) if r==0 else (t**r+t**(N-r))/(1+pn) for r in range(N)]
        chi=sum(C);xi=-1/abs(t).ln() if K else D(0)
        Z=((K.exp()+(-K).exp())**N)*(1+pn)
        result=dict(meanSpin=D(0),correlations=C,logAbsoluteCorrelations=[abs(c).ln() if c else '-infinity' for c in C],
                    commonFieldResponsePerSpin=chi,magnetizationMeanVariance=chi/N,totalSpinVariance=chi*N,
                    logPartition=Z.ln(),infiniteChainCorrelationLength=xi,
                    infiniteChainCommonFieldResponse=(2*K).exp(),infiniteChainOneStepCorrelation=t,
                    logAbsoluteOneStepCorrelation=abs(t).ln() if t else '-infinity')
        def close(a,b):assert abs(a-b)<=abs(b)*D('1e-140')+D(10)**(-digits+110),(case['id'],a,b)
        if case['transferChecked']:
            # No eigendecomposition, tanh or geometric sums in this path.
            T=[[K.exp(),(-K).exp()],[(-K).exp(),K.exp()]];I=[[D(1),D(0)],[D(0),D(1)]];powers=[I]
            for _ in range(N):powers.append(matmul(powers[-1],T))
            trace=powers[N][0][0]+powers[N][1][1];close(trace,Z)
            for r in range(N):
                A=powers[r];B=powers[N-r]
                numerator=sum(D((1 if i==j else -1))*A[i][j]*B[j][i] for i in range(2) for j in range(2))
                close(numerator/trace,C[r])
        if case['enumerationChecked']:
            states=list(product([-1,1],repeat=N));weights=[(K*sum(s[i]*s[(i+1)%N] for i in range(N))).exp() for s in states];W=sum(weights)
            close(W,Z)
            for r in range(N):close(sum(w*s[0]*s[r] for s,w in zip(states,weights))/W,C[r])
            close(sum(w*D(sum(s))**2 for s,w in zip(states,weights))/W/N,chi)
        def strings(v):
            if isinstance(v,list):return [strings(x) for x in v]
            if isinstance(v,str):return v
            return '0' if not v else str(v)
        return {k:strings(v) for k,v in result.items()}

def walk(v):
    if isinstance(v,dict):
        for x in v.values():yield from walk(x)
    elif isinstance(v,list):
        for x in v:yield from walk(x)
    else:yield v
for case in cases:
    digits=case['precisionDigits'];a=decimal_reference(case,digits);b=decimal_reference(case,digits+80)
    for x,y in zip(walk(a),walk(b)):
        if x=='-infinity':assert y==x;continue
        x=D(x);y=D(y)
        assert abs(x-y)<=abs(y)*D('1e-150')+D(10)**(-digits+100),(case['id'],x,y)
    case.update(confirmationPrecisionDigits=digits+80,reference=b)
result=dict(scope='Independent actual-binary Decimal references; transfer trace and enumeration coverage explicitly per case',cases=cases)
Path('scripts/fixtures/periodic-ising-ring-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),observations=sum(len(list(walk(c['reference']))) for c in cases),transferCases=sum(c['transferChecked'] for c in cases),enumerationCases=sum(c['enumerationChecked'] for c in cases),enumeratedConfigurations=sum(2**c['parameters']['spinCount'] for c in cases if c['enumerationChecked']))))
