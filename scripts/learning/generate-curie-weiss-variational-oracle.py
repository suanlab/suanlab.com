"""Independent high-precision stationary roots using direct Decimal logarithms.
No TypeScript interval or factored critical-series implementation is imported.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
from math import log, nextafter, inf
import json

cases=[]
def add(id,K,b=0,precision=400):
    cases.append(dict(id=id,parameters=dict(couplingK=K,fieldB=b),precisionDigits=precision))
add('independent-zero',0)
add('subcritical-biased',.5,.2)
add('critical-exact',1)
add('critical-biased',1,.05)
add('ordered-coexistence',2)
add('ordered-biased',2,.2)
add('ordered-reflected',2,-.2)
add('above-spinodal',2,1)
add('maximum-positive',8,4)
add('actual-log-three',log(3))
add('next-below-critical',nextafter(1,-inf))
add('next-above-critical',nextafter(1,inf))
with localcontext() as c:
    c.prec=100
    rounded=float(D(3)/8-D(2).ln()/2)
add('below-rounded-spinodal',9/8,nextafter(rounded,-inf))
add('rounded-spinodal',9/8,rounded)
add('above-rounded-spinodal',9/8,nextafter(rounded,inf))
tiny=float.fromhex('0x0.0000000000001p-1022')
add('critical-subnormal-field',1,tiny,1200)
add('subcritical-subnormal-field',.5,tiny,1200)
add('ordered-subnormal-field',2,tiny,1200)
add('independent-subnormal-field',0,tiny,1200)
add('near-critical-subnormal-field',nextafter(1,inf),tiny,1200)
add('critical-subnormal-reflected',1,-tiny,1200)

def reference(p,digits):
    with localcontext() as c:
        c.prec=digits
        K=D.from_float(float(p['couplingK'])); b=D.from_float(float(p['fieldB']))
        r=D(1)-D(2)**(-64); log2=D(2).ln()
        def f(m): return ((1+m)/(1-m)).ln()/2-K*m-b
        def root(a,z,increasing=True):
            fa,fz=f(a),f(z)
            assert (fa<0<fz) if increasing else (fa>0>fz)
            for _ in range(6000):
                mid=(a+z)/2; v=f(mid)
                assert v!=0, 'Reference precision insufficient for its requested bracket'
                if (v<0)==increasing:a=mid
                else:z=mid
                near=min(abs(a),abs(z))
                if (a>0 or z<0) and z-a<=near*D('1e-300'):return (a+z)/2
            raise AssertionError('Reference bisection exhausted')
        spinodal=None
        if K<=1:
            ms=[D(0)] if not b else [root(-r,r)]
        else:
            s=(1-1/K).sqrt(); B=K*s-((1+s)/(1-s)).ln()/2
            spinodal=dict(turningMagnetization=s,fieldMagnitude=B)
            if not b:
                x=root(s,r);ms=[-x,D(0),x]
            elif abs(b)<B:ms=[root(-r,-s),root(-s,s,False),root(s,r)]
            elif b>0:ms=[root(s,r)]
            else:ms=[root(-r,-s)]
        roots=[]
        for m in ms:
            plus,minus=(1+m)/2,(1-m)/2
            entropy=-plus*plus.ln()-minus*minus.ln()
            psi=-K*m*m/2-b*m-entropy
            curve=1/(1-m*m)-K
            if K==1 and not b:
                role='critical-minimum';derivative='positive-infinity';multiplicity=3
            else:
                role='unstable-maximum' if curve<0 else 'global-minimum' if not b or m*b>0 else 'metastable-minimum'
                derivative=1/curve;multiplicity=1
            roots.append(dict(magnetization=m,curvature=curve,branchFieldDerivative=derivative,
                constrainedFreeEnergy=psi,freeEnergyDifferenceToZero=psi+log2 if m else D(0),entropyNats=entropy,
                role=role,globalMinimum=role in ['global-minimum','critical-minimum'],stationaryMultiplicity=multiplicity))
        global_indices=[i for i,v in enumerate(roots) if v['globalMinimum']]
        pressure=-roots[global_indices[0]]['constrainedFreeEnergy']
        barriers=[];gap=None
        if len(roots)==3:
            barriers=[dict(minimumIndex=i,barrier=roots[1]['constrainedFreeEnergy']-roots[i]['constrainedFreeEnergy']) for i in [0,2]]
            gap=D(0) if not b else roots[0 if b>0 else 2]['constrainedFreeEnergy']-roots[global_indices[0]]['constrainedFreeEnergy']
        return dict(rootCount=len(ms),roots=roots,spinodal=spinodal,globalMinimumIndices=global_indices,
            thermodynamicLimitPressure=pressure,barriers=barriers,minimumFreeEnergyGap=gap)

def flatten(v,path=()):
    if isinstance(v,dict):
        for k,x in v.items():yield from flatten(x,path+(k,))
    elif isinstance(v,list):
        for i,x in enumerate(v):yield from flatten(x,path+(i,))
    else:yield path,v
def stringify(v):
    if isinstance(v,dict):return {k:stringify(x) for k,x in v.items()}
    if isinstance(v,list):return [stringify(x) for x in v]
    return str(v) if isinstance(v,D) else v

rows=[]
for case in cases:
    digits=case['precisionDigits']; low=dict(flatten(reference(case['parameters'],digits)))
    truth=reference(case['parameters'],digits+80)
    with localcontext() as c:
        c.prec=digits+80
        for path,y in flatten(truth):
            x=low[path]
            if isinstance(y,D):
                assert abs(x-y)<=abs(y)*D('1e-150')+D(10)**(-digits+60),(case['id'],path,str(x),str(y))
            else:assert x==y,(case['id'],path)
    rows.append(dict(**case,confirmationPrecisionDigits=digits+80,reference=stringify(truth)))
    print(json.dumps(dict(case=case['id'],digits=digits,confirmationDigits=digits+80)),flush=True)
Path('scripts/fixtures/curie-weiss-variational-oracle.json').write_text(json.dumps(dict(
    scope='Actual binary64 direct-log Decimal stationary roots at independent precision; model and reader evidence recorded separately',cases=rows),indent=2)+'\n')
print(json.dumps(dict(referenceCases=len(rows),referenceOnly=True)))
