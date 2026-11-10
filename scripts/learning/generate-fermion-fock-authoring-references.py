"""Independent exact JW matrices and density algebra for the fermion manuscript."""
import json
from pathlib import Path
import sympy as S
I,Z=S.eye(2),S.diag(1,-1);a=S.Matrix([[0,1],[0,0]])
def ops(m):return [S.kronecker_product(*[Z if j<i else a if j==i else I for j in range(m)]) for i in range(m)]
def vector(m,values):return S.Matrix([values.get(i,0) for i in range(2**m)])
def fraction(x):x=S.simplify(x);assert x.is_Rational,x;return str(x)
def cv(x):return dict(real=fraction(S.re(x)),imaginary=fraction(S.im(x)))
def matrix(A):return [[cv(A[i,j]) for j in range(A.cols)] for i in range(A.rows)]
def vec(z):return [cv(v) for v in z]
def frob(A):return S.simplify(S.trace(A.H*A))
def stats(m,z,H=None):
 C=ops(m);n=[v.H*v for v in C];number=sum(n,S.zeros(2**m));parity=S.diag(*[(-1)**i.bit_count() for i in range(2**m)]);N=(z.H*z)[0];rho=z*z.H/N;twirl=(rho+parity*rho*parity)/2
 expect=lambda A:S.simplify(S.trace(rho*A));H=H if H is not None else S.zeros(2**m);E=expect(H);v=S.simplify(expect(H*H)-E*E);nm=expect(number);nv=S.simplify(expect(number*number)-nm*nm)
 return dict(norm=fraction(N),hamiltonian=matrix(H),energy=fraction(E),variance=fraction(v),numberMean=fraction(nm),numberVariance=fraction(nv),numberSecond=fraction(expect(number*number)),numberConserved=bool(H*number-number*H==S.zeros(2**m)),parityConserved=bool(H*parity-parity*H==S.zeros(2**m)),gamma=[[cv(expect(ci.H*cj)) for cj in C] for ci in C],anomalous=[[cv(expect(ci*cj)) for cj in C] for ci in C],oddMeans=[cv(expect(ci)) for ci in C],parityMean=fraction(expect(parity)),purePurity=fraction(S.trace(rho*rho)),twirlPurity=fraction(S.trace(twirl*twirl)),coherenceSquared=fraction(frob(rho-twirl)),twirlEnergy=fraction(S.trace(twirl*H)),twirlOddMeans=[cv(S.trace(twirl*ci)) for ci in C])
C=ops(2);vac=vector(2,{0:1});odd=vector(2,{2:1});T=C[0].H*C[1].H
assert T*vac==vector(2,{3:1});assert C[1].H*C[0].H*vac==-T*vac;assert C[0].H*C[0].H*vac==S.zeros(4,1)
forward=C[0]*C[1].H*odd;reverse=C[1].H*C[0]*odd;naive=[S.kronecker_product(a,I),S.kronecker_product(I,a)]
assert forward==-reverse and forward==vector(2,{1:-1})
wrong_car=(naive[0]*naive[1].H+naive[1].H*naive[0])*odd;assert wrong_car==vector(2,{1:2})
hop=C[0].H*C[1]+C[1].H*C[0];X=S.Matrix([[0,1],[1,0]]);Y=S.Matrix([[0,-S.I],[S.I,0]])
assert hop==(S.kronecker_product(X,X)+S.kronecker_product(Y,Y))/2
hop_i=S.I*C[0].H*C[1]-S.I*C[1].H*C[0];one_i=vector(2,{1:1,2:S.I});hop_stats=stats(2,one_i,hop_i);assert hop_stats['energy']=='1' and hop_stats['gamma'][0][1]['imaginary']=='-1/2'
C3=ops(3);longH=C3[0].H*C3[2]+C3[2].H*C3[0];longz=vector(3,{3:1,6:1});assert C3[0].H*C3[2]*vector(3,{3:1})==vector(3,{6:-1})
assert longH*longz==-longz
naive3=[S.kronecker_product(*[a if j==i else I for j in range(3)]) for i in range(3)];wrongH=naive3[0].H*naive3[2]+naive3[2].H*naive3[0];assert wrongH*longz==longz
pair=T+T.H;pair_i=S.I*T-S.I*T.H;plus=vector(2,{0:1,3:1});minus=vector(2,{0:1,3:-1});complex_pair=vector(2,{0:1,3:S.I});assert pair*plus==plus and pair*minus==-minus and pair_i*complex_pair==complex_pair
wrong_pair=S.I*T-S.I*C[0]*C[1];assert frob(wrong_pair-wrong_pair.H)==8
pair_mean=S.simplify((complex_pair.H*T*complex_pair)[0]/2);assert pair_mean==-S.I/2
one_ops=ops(1);n=one_ops[0].H*one_ops[0];formal=stats(1,vector(1,{0:1,1:1}),n);assert formal['coherenceSquared']=='1/2' and formal['twirlPurity']=='1/2'
cat=stats(3,vector(3,{0:1,7:1}));assert all(v==dict(real='0',imaginary='0') for v in cat['oddMeans']);assert cat['coherenceSquared']=='1/2'
u=stats(4,vector(4,{3:1,12:1}));v=stats(4,vector(4,{0:1,15:1}));assert u['gamma']==v['gamma'] and u['anomalous']==v['anomalous'];assert u['numberVariance']=='0' and v['numberVariance']=='4'
A,B=S.symbols('A B',real=True,nonzero=True);rho=S.Matrix([[A*A,A*B],[A*B,B*B]])/(A*A+B*B);tw=(rho+Z*rho*Z)/2;assert S.simplify(frob(rho-tw)-2*A*A*B*B/(A*A+B*B)**2)==0
p0,p1,q0,q1=S.symbols('p0 p1 q0 q1');w=p0*q1-p1*q0;assert w.subs({p0:p1,p1:p0,q0:q1,q1:q0},simultaneous=True)==-w
result=dict(method='independent exact SymPy JW Kronecker paths,H squared,actual commutators,outer density/parity twirl and symbolic pure coherence',annihilators=[matrix(c) for c in C],creationOrder=dict(forward=vec(T*vac),reverse=vec(C[1].H*C[0].H*vac),excluded=vec(C[0].H*C[0].H*vac)),mixedCar=dict(forward=vec(forward),reverse=vec(reverse),naiveSum=vec(wrong_car)),imaginaryHopping=hop_stats,longRange=stats(3,longz,longH),longRangeNaiveEnergy='1',pairPlus=stats(2,plus,pair),pairMinus=stats(2,minus,pair),pairVacuum=stats(2,vac,pair),complexPair=stats(2,complex_pair,pair_i),complexCreationPairMean=cv(pair_mean),wrongPairHermiticityResidualSquared='8',formalOneMode=formal,formalThreeCat=cat,sameGammaFixed=u,sameGammaCat=v,sectorMinima=dict(full='-1',even='0',odd='-1',vacuumFullGround=False,vacuumEvenGround=True),vacuumSector=stats(1,vector(1,{0:1}),-n),symbolicPureCoherenceIdentity=True,exchangeAntisymmetry=True)
energies=[(-n)[i,i] for i in range(2)]
full=min(energies);even=energies[0];odd_energy=energies[1]
result['sectorMinima']=dict(full=fraction(full),even=fraction(even),odd=fraction(odd_energy),vacuumFullGround=bool(energies[0]==full),vacuumEvenGround=bool(energies[0]==even))
output=Path('scripts/fixtures/fermion-fock-authoring-reference.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(output),checks='CAR/order,hopping/pairing,complex adjoints,number/parity,SSR twirl,one-body insufficiency,sector minima')))
