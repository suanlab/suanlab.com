"""Independent symbolic protocol identities and180-digit two-bath references.
No imports from production. Existing references are immutable; choose a new output.
"""
from pathlib import Path
import argparse,json,sympy as s,mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);output=Path(parser.parse_args().output)
if output.exists():raise SystemExit('Refusing to overwrite an existing reference')
mp.mp.dps=180
p,r0,r1,r2,e1,e2=s.symbols('p r0 r1 r2 e1 e2',positive=True)
pi=lambda r:s.Matrix([[r/(1+r),1/(1+r)]])
P=lambda r,e:s.Matrix([[(r+e)/(1+r),(1-e)/(1+r)],[r*(1-e)/(1+r),(1+r*e)/(1+r)]])
D=lambda a,b:s.diag(1,a/b)
M=(pi(r0)*D(r0,r1)*P(r1,e1)*D(r1,r2)*P(r2,e2)*s.ones(2,1))[0]
assert s.factor(M-(1+1/r2)/(1+1/r0))==0
for x in range(2):
 for y in range(2):
  forward=pi(r0)[x]*P(r1,e1)[x,y];reverse=pi(r2)[y]*P(r1,e1)[y,x]
  expected=(1+1/r2)/(1+1/r0)*(r1/r0)**x*(r2/r1)**y
  assert s.factor(forward-reverse*expected)==0
for x in range(2):
 for y in range(2):
  for z in range(2):
   work=(r1/r0)**x*(r2/r1)**y;heat=r1**(y-x)*r2**(z-y);delta=r2**z/r0**x
   assert s.factor(work*heat-delta)==0
# Arbitrary normalized full-support final law in the reverse construction.
v=s.symbols('v',positive=True);pf=s.Matrix([[1-v,v]])
reverseMass=sum(pf[z]*P(r2,e2)[z,y]*P(r1,e1)[y,x]for x in range(2)for y in range(2)for z in range(2));assert s.factor(reverseMass-1)==0
# Independent two-bath family; channel identity is retained, not a production model.
rows=[]
for label,r,bh,bc,nh,nc in [('worked hot/cold',s.Rational(3),1,2,1,1),('equal temperatures',s.Rational(3),1,1,1,1),('zero gap',s.Rational(1),1,2,1,1),('negative gap',s.Rational(1,3),1,2,1,1),('reverse bath names',s.Rational(3),2,1,1,1),('unequal clocks',s.Rational(3),1,2,7,2),('almost equal odds',s.Rational(1000001,1000000),1,2,1,1),('cold contrast',s.Rational(9),1,4,1,3),('fractional odds',s.Rational(7,5),1,3,s.Rational(2,3),s.Rational(5,7)),('clock scaling',s.Rational(3),1,2,2,2),('hot weak coupling',s.Rational(3),1,2,s.Rational(1,1000000),1),('cold weak coupling',s.Rational(3),1,2,1,s.Rational(1,1000000))]:
 nh,nc=map(s.Rational,[nh,nc]);ah=nh/(1+r**bh);dh=nh-ah;ac=nc/(1+r**bc);dc=nc-ac
 Q=s.Matrix([[-ah-ac,ah+ac],[dh+dc,-dh-dc]]);v=Q.T.nullspace()[0];law=v/sum(v);jh=s.factor(law[0]*ah-law[1]*dh);jc=s.factor(law[0]*ac-law[1]*dc);assert jh+jc==0
 currentFormula=nh*nc/(nh+nc)*(1/(1+r**bh)-1/(1+r**bc));assert s.factor(jh-currentFormula)==0
 m=lambda x:mp.mpf(str(s.numer(x)))/mp.mpf(str(s.denom(x)))
 energy=mp.log(m(r));qh=energy*m(jh);qc=energy*m(jc);sigma=-bh*qh-bc*qc
 paired=sum(m(law[0]*a-law[1]*d)*mp.log(m(law[0]*a/(law[1]*d)))for a,d in [(ah,dh),(ac,dc)])
 assert abs(sigma-paired)<mp.mpf('1e-160');assert sigma>=0;assert qh+qc==0
 rows.append(dict(label=label,odds=str(r),betaHot=bh,betaCold=bc,hotClock=str(nh),coldClock=str(nc),stationary=[str(x)for x in law],hotCurrent=str(jh),coldCurrent=str(jc),aggregateCurrent='0',aggregateProduction='0',energyInReferenceThermalUnits=mp.nstr(energy,150),hotHeatIntoSystemPerSecond=mp.nstr(qh,150),coldHeatIntoSystemPerSecond=mp.nstr(qc,150),physicalEntropyRate=mp.nstr(sigma,150)))
report=dict(passed=True,symbolic=dict(canonicalJarzynski=True,crooksPairedEndpointIdentities=4,endpointFirstLawIdentities=8,arbitraryFinalLawReverseNormalization=True),reservoirResolvedCases=len(rows),referenceDigits=mp.mp.dps,rows=rows,scope='Source derivation of two-quench work/endpoint identities and reservoir-resolved two-state heat conduction. No registered reader or empirical experiment.')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='rows'}))
