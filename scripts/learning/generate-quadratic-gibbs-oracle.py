"""Independent exact-binary input classification and Decimal Gaussian integrals.
Positive T, dimensionless x in R², Lebesgue dx. Reference generation alone
claims no TypeScript model, manuscript, teaching components or reader.
"""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from pathlib import Path
import json
import math
KB=1.380649e-23
cases=[]
def add(id,A11,A12,A22,h1=0,h2=0,c=0,T=1,scale=(1,1),shift=(0,0),precision=400):
    cases.append(dict(id=id,parameters=dict(firstStiffness=A11,crossStiffness=A12,secondStiffness=A22,
        firstSourceEnergy=h1,secondSourceEnergy=h2,energyOffset=c,temperature=T,coordinateScale=list(scale),coordinateShift=list(shift)),precisionDigits=precision))
add('baseline-quadratic',2*KB,KB,2*KB)
add('linear-source',2*KB,KB,2*KB,h1=KB)
add('density-above-one',100*KB,0,100*KB)
add('coordinate-jacobian',2*KB,KB,2*KB,h1=KB,scale=(2,3),shift=(1,-2))
add('common-energy-offset',2*KB,KB,2*KB,h1=KB,c=2**60)
add('barely-positive-determinant',1,1,math.nextafter(1,math.inf))
add('singular-flat-direction',1,1,1)
add('indefinite-near-boundary',1,1,math.nextafter(1,-math.inf))
add('negative-definite',-1,0,-1)
add('overflowing-determinant',float.fromhex('0x1.fffffffffffffp+1023'),0,float.fromhex('0x1.fffffffffffffp+1023'))
add('subnormal-stiffness',float.fromhex('0x0.0000000000001p-1022'),0,2*float.fromhex('0x0.0000000000001p-1022'))
add('outside-range-mean',float.fromhex('0x0.0000000000001p-1022'),0,1,h1=float.fromhex('0x1.fffffffffffffp+1023'),precision=1200)
add('subnormal-positive-temperature',2*KB,KB,2*KB,T=float.fromhex('0x0.0000000000001p-1022'))
add('maximum-temperature',2*KB,KB,2*KB,T=float.fromhex('0x1.fffffffffffffp+1023'))
add('common-energy-temperature-scale',16*KB,8*KB,16*KB,h1=8*KB,T=8)
add('orientation-reversing-coordinates',2*KB,KB,2*KB,h1=KB,scale=(-2,3),shift=(1,-2))
add('outside-range-jacobian',2*KB,KB,2*KB,scale=(float.fromhex('0x1.fffffffffffffp+1023'),float.fromhex('0x1.fffffffffffffp+1023')))

def dec(x):return D.from_float(float(x))
def pi_decimal(digits):
    with localcontext() as ctx:
        ctx.prec=digits+30
        def atan_inverse(k):
            x=D(1)/k;power=x;total=D(0);j=0
            while True:
                term=power/(2*j+1)
                total+=term if j%2==0 else -term
                if abs(term)<D(10)**(-digits-20):break
                power*=x*x;j+=1
            return total
        result=16*atan_inverse(5)-4*atan_inverse(239)
    return result

def reference(p,digits):
    # Sign decisions use exact rational arithmetic, independent of Decimal
    # rounding and without an eigenvalue tolerance or pseudoinverse.
    a,b,d=map(lambda x:F.from_float(float(x)),[p['firstStiffness'],p['crossStiffness'],p['secondStiffness']])
    determinant=a*d-b*b
    if a<=0 or determinant<=0:
        return dict(normalizable=False,classification='not-positive-definite',exactDeterminantSign=(determinant>0)-(determinant<0))
    with localcontext() as ctx:
        ctx.prec=digits
        A,B,C=map(dec,[p['firstStiffness'],p['crossStiffness'],p['secondStiffness']])
        h1,h2,c,T=map(dec,[p['firstSourceEnergy'],p['secondSourceEnergy'],p['energyOffset'],p['temperature']])
        tau=dec(KB)*T;det=D(determinant.numerator)/D(determinant.denominator)
        inv=[[C/det,-B/det],[-B/det,A/det]]
        mean=[inv[0][0]*h1+inv[0][1]*h2,inv[1][0]*h1+inv[1][1]*h2]
        covariance=[[tau*v for v in row] for row in inv]
        minimum=c-(h1*mean[0]+h2*mean[1])/2
        shape=(2*pi_decimal(digits)*tau).ln()-det.ln()/2
        logZ=shape-minimum/tau
        entropy=1+shape
        peak=(-shape).exp()
        scales=list(map(dec,p['coordinateScale']));shifts=list(map(dec,p['coordinateShift']))
        jacobian=abs(scales[0]*scales[1]);logjac=jacobian.ln()
        transformed=dict(mean=[(m-d)/s for m,d,s in zip(mean,shifts,scales)],
            covariance=[[v/(scales[i]*scales[j]) for j,v in enumerate(row)] for i,row in enumerate(covariance)],
            logAbsoluteJacobian=logjac,logPartition=logZ-logjac,differentialEntropyNats=entropy-logjac,densityAtMean=peak*jacobian)
        return dict(normalizable=True,classification='positive-definite',exactDeterminantSign=1,
            determinant=det,thermalEnergy=tau,mean=mean,covariance=covariance,
            precisionMatrix=[[A/tau,B/tau],[B/tau,C/tau]],minimumEnergy=minimum,meanEnergy=minimum+tau,
            heatCapacityInKB=D(1),logPartition=logZ,freeEnergy=-tau*logZ,
            differentialEntropyNats=entropy,densityAtMean=peak,transformedCoordinates=transformed)

def leaves(x,path=()):
    if isinstance(x,dict):
        for k,v in x.items():yield from leaves(v,path+(k,))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from leaves(v,path+(i,))
    else:yield path,x

def strings(x):
    if isinstance(x,dict):return {k:strings(v) for k,v in x.items()}
    if isinstance(x,list):return [strings(v) for v in x]
    return str(x) if isinstance(x,D) else x
rows=[]
for case in cases:
    digits=case['precisionDigits'];a=dict(leaves(reference(case['parameters'],digits)));truth=reference(case['parameters'],digits+80)
    with localcontext() as ctx:
        ctx.prec=digits+80
        for path,y in leaves(truth):
            x=a[path]
            if isinstance(y,D):
                if y==0:assert x==0,(case['id'],path,'structural zero')
                else:assert abs((x-y)/y)<D('1e-150'),(case['id'],path,x,y)
            else:assert x==y,(case['id'],path)
    rows.append(dict(**case,confirmationPrecisionDigits=digits+80,reference=strings(truth)))
    print(json.dumps(dict(case=case['id'],precision=digits,confirmation=digits+80)),flush=True)
Path('scripts/fixtures/quadratic-gibbs-oracle.json').write_text(json.dumps(dict(
    scope='Independent actual binary64 quadratic Gibbs references only; no model/manuscript/reader verification claimed',
    referenceMeasure='Lebesgue dx on dimensionless R²; positive finite temperature',
    piMethod='Machin identity 16 atan(1/5)-4 atan(1/239), Decimal Taylor with 30 guard digits',
    piDecimalReferenceDigits=400,piDecimalReference=str(pi_decimal(400)),
    coordinateConvention='x_i=coordinateScale_i*z_i+coordinateShift_i; reference dz',cases=rows),indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),referenceOnly=True)),flush=True)
