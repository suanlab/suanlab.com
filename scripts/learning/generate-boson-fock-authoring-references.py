"""Independent exact cutoff/embedded operators and high-precision coherent tails."""
import json
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def fraction(x):x=S.simplify(x);assert x.is_Rational,x;return str(x)
def real(x):
 x=S.expand(S.radsimp(S.sympify(x)));basis=[S.sqrt(2),S.sqrt(3),S.sqrt(6)];b=[x.coeff(r) for r in basis];a=S.simplify(x-sum(v*r for v,r in zip(b,basis)));return dict(coefficients=[fraction(v) for v in [a]+b])
def cv(x):return dict(real=real(S.re(x)),imaginary=real(S.im(x)))
def matrix(A):return [[cv(A[i,j]) for j in range(A.cols)] for i in range(A.rows)]
def ladder(C):
 A=S.zeros(C+1)
 for n in range(1,C+1):A[n-1,n]=S.sqrt(n)
 return A

def stats(A,z,H=None):
 N=S.simplify((z.H*z)[0]);expect=lambda B:S.simplify((z.H*B*z)[0]/N);number=A.H*A;comm=A*A.H-A.H*A;X=(A+A.H)/S.sqrt(2);P=(A-A.H)/(S.I*S.sqrt(2));xm=expect(X);pm=expect(P);xv=S.simplify(expect(X*X)-xm*xm);pv=S.simplify(expect(P*P)-pm*pm);cov=S.simplify(expect((X*P+P*X)/2)-xm*pm);nm=expect(number);nv=S.simplify(expect(number*number)-nm*nm);H=H if H is not None else number;E=expect(H);variance=S.simplify(expect(H*H)-E*E)
 return dict(norm=real(N),annihilationMean=cv(expect(A)),numberMean=real(nm),numberSecond=real(expect(number*number)),numberVariance=real(nv),commutatorMean=real(expect(comm)),xMean=real(xm),pMean=real(pm),xVariance=real(xv),pVariance=real(pv),covariance=real(cov),robertsonBound=real(expect(comm)**2/4),uncertaintyProduct=real(xv*pv),energy=real(E),variance=real(variance))
cutoffs=[]
for C in range(4):
 A=ladder(C);Ae=ladder(C+1);X=(A+A.H)/S.sqrt(2);P=(A-A.H)/(S.I*S.sqrt(2));Xe=(Ae+Ae.H)/S.sqrt(2);Pe=(Ae-Ae.H)/(S.I*S.sqrt(2));top=S.zeros(C+1);top[C,C]=1;comm=A*A.H-A.H*A
 assert comm==S.eye(C+1)-(C+1)*top and S.trace(comm)==0
 assert (Xe*Xe)[:C+1,:C+1]-X*X==(C+1)*top/2
 assert (Pe*Pe)[:C+1,:C+1]-P*P==(C+1)*top/2
 assert A**(C+1)==S.zeros(C+1)
 H=A.H*A+S.eye(C+1)/2;quad=(X*X+P*P)/2;assert H-quad==(C+1)*top/2
 vacuum=S.zeros(C+1,1);vacuum[0]=1;zt=S.zeros(C+1,1);zt[C]=1
 cutoffs.append(dict(cutoff=C,annihilation=matrix(A),numberDiagonal=[fraction((A.H*A)[i,i]) for i in range(C+1)],reverseNumberDiagonal=[fraction((A*A.H)[i,i]) for i in range(C+1)],commutator=matrix(comm),commutatorTrace='0',nilpotent=True,projectedXSquareCorrection=matrix((Xe*Xe)[:C+1,:C+1]-X*X),projectedPSquareCorrection=matrix((Pe*Pe)[:C+1,:C+1]-P*P),projectedHamiltonian=matrix(H),recombinedFiniteHamiltonian=matrix(quad),vacuum=stats(A,vacuum),top=stats(A,zt)))
C=3;A=ladder(C);raw=S.Matrix([1,1,S.sqrt(2)/2,S.sqrt(6)/6]);norm=S.simplify((raw.H*raw)[0]);assert norm==S.Rational(8,3)
finite=stats(A,raw);embedded=stats(ladder(C+1),S.Matrix(list(raw)+[0]));assert finite['xVariance']==real(S.Rational(39,128)) and embedded['xVariance']==real(S.Rational(55,128));assert finite['numberVariance']==real(S.Rational(207,256))
res=(A-S.eye(4))*raw;residual=S.simplify((res.H*res)[0]/norm);assert residual==S.Rational(1,16)
cat=stats(A,S.Matrix([1,0,0,1]));assert cat['commutatorMean']==real(-1)
hopping=[]
for c0,c1 in [(3,2),(3,3)]:
 a0=S.kronecker_product(ladder(c0),S.eye(c1+1));a1=S.kronecker_product(S.eye(c0+1),ladder(c1));H=a0.H*a1+a1.H*a0;z=S.zeros((c0+1)*(c1+1),1);z[2*(c1+1)+2]=1;action=H*z;E=(z.H*H*z)[0];var=S.simplify((z.H*H*H*z)[0]-E**2);assert E==0 and var==(6 if c1==2 else 12)
 hopping.append(dict(cutoffs=[c0,c1],hamiltonianAction=[cv(v) for v in action],energy=real(E),variance=real(var)))

def decimal(x):return 'decimal:'+mp.nstr(x,245)
tails=[]
for mu,C in [(S.Rational(0),0),(S.Rational(1),0),(S.Rational(1),1),(S.Rational(1),3),(S.Rational(16),3),(S.Rational(1,10**12),3)]:
 Z=sum(mu**n/S.factorial(n) for n in range(C+1));x=mp.mpf(str(S.N(mu,255)));tail=mp.gammainc(C+1,0,x)/mp.factorial(C);retained=mp.exp(-x)*mp.mpf(str(S.N(Z,255)));assert abs(retained+tail-1)<mp.mpf('1e-240')
 tails.append(dict(mu=fraction(mu),cutoff=C,seriesNorm=fraction(Z),retained=decimal(retained),tail=decimal(tail),fidelitySquared=decimal(retained),normalizedDistanceSquared=decimal(2*(1-mp.sqrt(retained)))))
S0=mp.exp(-1)*mp.mpf(8)/3;S1=mp.exp(-1)*mp.mpf(5)/2;two=dict(retained=decimal(S0*S1),tail=decimal(1-S0*S1))
A1=ladder(1);D=(A1.H-A1).exp();target=S.Matrix([1,1])/S.sqrt(2);assert S.simplify(D.H*D-S.eye(2))==S.zeros(2)
assert S.simplify(D[0,0]-S.cos(1))==0 and S.simplify(D[1,0]-S.sin(1))==0
fidelity=S.simplify((target.H*D*S.Matrix([1,0]))[0]**2)
e=S.symbols('epsilon',positive=True);difference=S.Matrix([[-e,S.sqrt(e*(1-e))],[S.sqrt(e*(1-e)),e]]);lam=S.symbols('lambda');assert S.simplify(difference.charpoly(lam).as_expr()-(lam**2-e))==0
result=dict(method='independent SymPy finite/extended ladder matrices,H squared,quadrature products,exact coherent series and mpmath250dps incomplete-gamma tails',cutoffs=cutoffs,boundaryCat=cat,coherentUnit=dict(rawCoefficients=[cv(v) for v in raw],seriesNorm=fraction(norm),topProbability='1/16',annihilatorResidualSquared=fraction(residual),finite=finite,embeddedOriginalOperators=embedded),hopping=hopping,tails=tails,twoMode=two,finiteDisplacement=dict(vacuumReal0=decimal(mp.cos(1)),vacuumReal1=decimal(mp.sin(1)),fidelityToNormalizedProjection=decimal(mp.mpf(str(S.N(fidelity,255)))),unitary=True,differsFromCoherentProjection=True),boundedObservableTraceDistanceSquared='epsilon',unboundedNumberCounterexample=[dict(tail='1/100',occupation=100,meanNumber='1'),dict(tail='1/100',occupation=10000,meanNumber='100')])
output=Path('scripts/fixtures/boson-fock-authoring-reference.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(output),cutoffs=4,tailPreparations=len(tails),coherentVariance='207/256',finiteXVariance='39/128',embeddedXVariance='55/128')))
