"""Implementation-independent derivation checks; not the future calculator/reference fixture."""
import json,math
from fractions import Fraction as Q
from pathlib import Path
import sympy as s
import mpmath as mp

ROOT=Path('.runtime/learning/spatial-spectrum-foundation')
rows=[]
def record(name,kind,details=None):rows.append(dict(name=name,kind=kind,passed=True,details=details or {}))
r,k,T=s.symbols('r k T',positive=True);N,d=s.symbols('N d',integer=True,positive=True)
M=k*(1-r)**2/r;amplitude=T*r/(k*(1-r**2))
assert s.factor(k*r**2-(M+2*k)*r+k)==0;record('infinite-characteristic-root','symbolic')
assert s.factor((M+2*k)*amplitude-2*k*amplitude*r-T)==0;record('infinite-source-jump','symbolic')
g=lambda x:amplitude*(r**x+r**N/r**x)/(1-r**N)
assert s.simplify(s.expand_power_exp((M+2*k)*g(d)-k*(g(d-1)+g(d+1))))==0;record('periodic-image-interior-recurrence','symbolic')
assert s.factor((M+2*k)*g(0)-2*k*g(1)-T)==0;record('periodic-image-origin-jump','symbolic')
u=s.symbols('u',positive=True)
assert s.limit(s.sqrt(u)/s.acosh(1+u/2),u,0,dir='+')==1;record('continuum-length-ratio-limit','symbolic')
assert s.simplify(s.limit((s.sqrt(u)/s.acosh(1+u/2)-1)/u,u,0,dir='+')-s.Rational(1,24))==0;record('continuum-length-leading-correction','symbolic')
positive_mass=s.symbols('positive_mass',positive=True)
assert s.limit(2*k/(positive_mass+2*k+s.sqrt(positive_mass*(positive_mass+4*k))),k,0,dir='+')==0;record('zero-coupling-decay-limit','symbolic')
assert s.limit(T/s.sqrt(positive_mass*(positive_mass+4*k)),k,0,dir='+')==T/positive_mass;record('zero-coupling-onsite-variance','symbolic')

temperature=Q(2,3);stiffness=Q(5,7)
for n in range(3,17):
 for decay in [Q(1,8),Q(1,2),Q(3,4),Q(99999,100000),Q(1,999999)]:
  mass=stiffness*(1-decay)**2/decay;A=temperature*decay/(stiffness*(1-decay**2));row=[A*(decay**j+decay**(n-j))/(1-decay**n)for j in range(n)]
  assert sum(row)==temperature/mass
  for i in range(n):
   for j in range(n):
    delta=(j-i)%n;value=(mass+2*stiffness)*row[delta]-stiffness*(row[(delta-1)%n]+row[(delta+1)%n]);assert value==(temperature if i==j else 0)
  constrained=[v-temperature/(n*mass)for v in row];assert sum(constrained)==0
  for j in range(n):assert (mass+2*stiffness)*constrained[j]-stiffness*(constrained[(j-1)%n]+constrained[(j+1)%n])==temperature*((1 if j==0 else 0)-Q(1,n))
  record(f'periodic-images-n{n}-r{decay}','exact-fraction',dict(sites=n,decay=str(decay),allMatrixEntries=n*n,meanZeroConstantSubtraction=True))
 row=[temperature/stiffness*(Q(n*n-1,12*n)-Q(j*(n-j),2*n))for j in range(n)];assert sum(row)==0 and any(v<0 for v in row)
 for j in range(n):assert stiffness*(2*row[j]-row[(j-1)%n]-row[(j+1)%n])==temperature*((1 if j==0 else 0)-Q(1,n))
 record(f'massless-constrained-green-n{n}','exact-fraction',dict(sites=n,negativeCovariancePresent=True))

mp.mp.dps=200
num=lambda q:mp.mpf(str(s.numer(q)))/mp.mpf(str(s.denom(q)))
tol=mp.mpf('1e-150')
for n in [4,5,8]:
 for boundary in ['periodic','open','dirichlet']:
  for constrained in [False,True]:
   spacing=s.Rational(1,2);mass=s.Rational(3,2);speed=s.Rational(5,3);temperature=s.Rational(7,5);L=s.zeros(n)
   edges=[(i,i+1)for i in range(n-1)]+([(n-1,0)]if boundary=='periodic'else[])
   for i,j in edges:
    incidence=s.zeros(n,1);incidence[i]=1;incidence[j]=-1;L+=incidence*incidence.T
   if boundary=='dirichlet':L[0,0]+=1;L[n-1,n-1]+=1
   K=spacing*mass*s.eye(n)+speed/spacing*L;ones=s.ones(n,1);P=s.eye(n)-ones*ones.T/n if constrained else s.eye(n)
   G=K.row_join(ones).col_join(ones.T.row_join(s.zeros(1))).inv(method='GE')[:n,:n]if constrained else K.inv(method='GE');C=temperature*G;assert P*K*C==temperature*P and C==C.T and P*C==C
   F=mp.matrix([[mp.exp(-2j*mp.pi*l*j/n)/mp.sqrt(n)for j in range(n)]for l in range(n)]);cm=mp.matrix([[num(v)for v in C.row(i)]for i in range(n)]);km=mp.matrix([[num(v)for v in K.row(i)]for i in range(n)]);pm=mp.matrix([[num(v)for v in P.row(i)]for i in range(n)]);spectral=F*cm*F.H;precision=F*km*F.H;projection=F*pm*F.H
   off=max(abs(spectral[i,j])for i in range(n)for j in range(n)if i!=j)
   assert mp.norm(spectral-spectral.H)<tol and abs(sum(spectral[i,i]for i in range(n))-sum(cm[i,i]for i in range(n)))<tol
   assert mp.norm(F.H*spectral*F-cm)<tol and mp.norm(projection*precision*spectral-num(temperature)*projection)<tol
   assert (off<tol)if boundary=='periodic'else(off>mp.mpf('1e-4'))
   if constrained:assert abs(spectral[0,0])<tol
   if boundary=='periodic':
    for l in range(n):
     expected=0 if constrained and l==0 else num(temperature)/(num(spacing*mass)+4*num(speed/spacing)*mp.sin(mp.pi*l/n)**2)
     assert abs(spectral[l,l]-expected)<tol
   record(f'fourier-n{n}-{boundary}-'+('constrained'if constrained else'full'),'exact-matrix-plus-200-digit-cross-check',dict(sites=n,offDiagonalMagnitude=mp.nstr(off,12),tolerance=str(tol),referencePointIsCertifiedInterval=False))

# Same physical mode and box, not the same raw unitary-coordinate variance.
length=mp.mpf(8);wave=1;mass=mp.mpf(1);speed=mp.mpf(1);temperature=mp.mpf(1);continuum=mass+speed*(2*mp.pi*wave/length)**2;last=None;refinements=[]
for n in [8,16,32,64]:
 a=length/n;energy=a*mass+4*speed/a*mp.sin(mp.pi*wave/n)**2;power=temperature/energy;normalized=a*power;gap=continuum-energy/a
 assert gap>0 and (last is None or gap<last);last=gap
 refinements.append(dict(sites=n,spacing=mp.nstr(a,8),rawUnitaryPower=mp.nstr(power,20),continuumNormalizedPower=mp.nstr(normalized,20),operatorGap=mp.nstr(gap,20)))
record('fixed-box-physical-mode-refinement','200-digit-cross-check',dict(rows=refinements,continuumPower=mp.nstr(temperature/continuum,20)))
massless_row=[Q(8*8-1,12*8)-Q(j*(8-j),2*8)for j in range(8)];assert massless_row[0]>massless_row[1]>0>massless_row[2]
record('positive-local-ratio-without-infinite-stationary-field','exact-fraction',dict(firstThree=list(map(str,massless_row[:3]))))
independent_constrained=[[Q(int(i==j))-Q(1,4)for j in range(4)]for i in range(4)];assert independent_constrained[0][1]<0 and all(sum(row)==0 for row in independent_constrained)
record('zero-edge-coupling-is-not-independence-after-conditioning','exact-fraction',dict(firstRow=list(map(str,independent_constrained[0]))))
aliases=0
for n in range(3,17):
 for index in range(n):
  signed=index if 2*index<=n else index-n
  for factor in [2,4,8]:
   refined=n*factor;new_index=signed%refined;new_signed=new_index if 2*new_index<=refined else new_index-refined
   assert new_signed==signed and Q(new_signed,refined)*factor==Q(signed,n);aliases+=1
record('refinement-preserves-folded-physical-mode-not-raw-index','exact-integer',dict(aliasMappings=aliases,maximumVirtualSites=128))
report=dict(passed=len(rows),symbolic=sum(x['kind']=='symbolic'for x in rows),exactFraction=sum(x['kind']=='exact-fraction'for x in rows),exactInteger=sum(x['kind']=='exact-integer'for x in rows),highPrecisionCrossChecks=sum('cross-check'in x['kind']for x in rows),rows=rows,productionCalculatorImplemented=False,readerRegistered=False,wholeGoalComplete=False)
ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'derivations.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['passed','symbolic','exactFraction','highPrecisionCrossChecks']}))
