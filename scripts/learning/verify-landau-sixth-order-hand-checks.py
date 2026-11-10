"""Exact rational identities for the upcoming Landau family, not model evidence.
Use y=m^2 so irrational stationary magnetizations retain rational certificates.
"""
from fractions import Fraction as F
from pathlib import Path
import json

a,b,c=F(3,16),F(-1),F(1)
def potential(y):return a*y/2+b*y*y/4+c*y*y*y/6
def curvature(y):return a+3*b*y+5*c*y*y
# Coefficients of y(y-3/4)^2/6 in increasing powers of y.
factored=[F(0),F(3,4)**2/6,-2*F(3,4)/6,F(1,6)]
original=[F(0),a/2,b/4,c/6]
assert original==factored
rows=[]
for y,role,expected_curvature in [(F(0),'global-minimum',F(3,16)),(F(1,4),'unstable-maximum',F(-1,4)),(F(3,4),'global-minimum',F(3,4))]:
    if y:assert a+b*y+c*y*y==0
    assert curvature(y)==expected_curvature
    assert potential(y)==(F(1,96) if y==F(1,4) else 0)
    rows.append(dict(magnetizationSquared=str(y),magnetizationSigns=[0] if not y else [-1,1],
        role=role,freeEnergy=str(potential(y)),curvature=str(curvature(y))))
assert 3*b*b/(16*c)==a
assert -3*b/(4*c)==F(3,4)
# At the ordered spinodal, the y polynomial has a repeated root.
a_sp=b*b/(4*c);y_sp=-b/(2*c)
assert a_sp+b*y_sp+c*y_sp*y_sp==0
assert a_sp+3*b*y_sp+5*c*y_sp*y_sp==0
assert potential(F(1,4))-potential(F(0))==F(1,96)
def multiply(p,q):
    out=[F(0)]*(len(p)+len(q)-1)
    for i,x in enumerate(p):
        for j,y in enumerate(q):out[i+j]+=x*y
    return out
tilted=[]
for P in [F(3,16),F(1,8)]:
    S=F(1);C=F(3)
    A=C*(S**4-S*S*P+3*P*P)/3
    B=2*C*(-2*S*S+3*P)/3
    H=C*S**3*P/3
    E=-C*(S*S+P)*P*P/6
    factor=multiply(multiply([P,-S,F(1)],[P,-S,F(1)]),[S*S+P,2*S,F(1)])
    assert [C*x/6 for x in factor]==[-E,-H,A/2,F(0),B/4,F(0),C/6]
    assert S*S-4*P>0 and (2*S)**2-4*(S*S+P)<0
    tilted.append(dict(a=str(A),b=str(B),c=str(C),H=str(H),minimumValue=str(E),
        minimumQuadraticCoefficients=[str(P),str(-S),'1'],globalMinima=2,positiveQuadraticFactor=True))
result=dict(scope='exact hand identities only; not an actual-input root isolation, reader, or whole-topic verification',
    coefficients=dict(a=str(a),b=str(b),c=str(c),H='0'),factorizationVerified=True,
    distinctStationaryRoots=5,distinctGlobalMinima=3,rows=rows,
    orderedSpinodal=dict(a=str(a_sp),magnetizationSquared=str(y_sp)),
    disorderedStabilityBoundaryA='0',coexistenceA=str(a),barrier='1/96',tiltedCoexistenceCertificates=tilted,
    next='Implement and independently verify all roots for actual binary64 coefficients and tilted fields; compare global free energies, multiplicities and precision limits.')
path=Path('.runtime/learning/landau-sixth-order/hand-checks.json')
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
