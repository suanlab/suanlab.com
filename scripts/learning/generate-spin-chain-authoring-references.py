"""Independent symbolic hand calculations for the spin-chain chapter."""
import json
from pathlib import Path
import sympy as S
I=S.eye(2);X=S.Matrix([[0,1],[1,0]]);Y=S.Matrix([[0,-S.I],[S.I,0]]);Z=S.diag(1,-1)
def op(n,operators):return S.kronecker_product(*[operators.get(i,I) for i in range(n)])
def fraction(x):x=S.simplify(x);assert x.is_Rational;return str(x)
def cv(x):return dict(real=fraction(S.re(x)),imaginary=fraction(S.im(x)))
def matrix(A):return [[cv(A[i,j]) for j in range(A.cols)] for i in range(A.rows)]
def moments(H,z):
 N=(z.conjugate().T*z)[0];E=S.simplify((z.conjugate().T*H*z)[0]/N);second=S.simplify((z.conjugate().T*H*H*z)[0]/N);return N,E,second,S.expand(second-E*E)
def exact_moments(H,z):return dict(zip(['norm','energy','second','variance'],[fraction(v) for v in moments(H,z)]))
XX,YY,ZZ=[S.kronecker_product(P,P) for P in [X,Y,Z]];M=op(2,{0:Z})+op(2,{1:Z});h=S.symbols('h',real=True);b,c=S.symbols('b c',real=True)
z=S.zeros(16,1);z[0]=z[15]=1
H0=-sum((op(4,{i:Z,i+1:Z}) for i in range(3)),S.zeros(16));field=sum((op(4,{i:X}) for i in range(4)),S.zeros(16));H=H0-h*field
N,E,E2,var=moments(H,z);assert E==-3 and var==4*h*h
assert S.simplify(moments(b*H+c*S.eye(16),z)[1]-(-3*b+c))==0
assert S.simplify(moments(b*H+c*S.eye(16),z)[3]-4*b*b*h*h)==0
images=S.Matrix.hstack(*[op(4,{i:X})*z for i in range(4)]);gram=images.conjugate().T*images/N;assert gram==S.eye(4)
rho=z*z.conjugate().T/N;mixture=S.zeros(16);mixture[0,0]=mixture[15,15]=S.Rational(1,2)
Xall=op(4,{i:X for i in range(4)})
assert S.trace(rho*Xall)==1 and S.trace(mixture*Xall)==0
pair_z=[]
for i in range(4):
 for j in range(i+1,4):
  joint=op(4,{i:Z,j:Z});pure=S.trace(rho*joint);mixed=S.trace(mixture*joint)
  assert pure==mixed==1
  assert S.trace(rho*op(4,{i:Z}))==S.trace(mixture*op(4,{i:Z}))==0
  assert S.trace(rho*op(4,{j:Z}))==S.trace(mixture*op(4,{j:Z}))==0
  pair_z.append(dict(sites=[i,j],pure=fraction(pure),mixture=fraction(mixed)))
Q=Xall;P=op(4,{i:Z for i in range(4)})
assert H*Q-Q*H==S.zeros(16)
assert H*P-P*H!=S.zeros(16)
locals=[]
for site in range(4):
 mask=2**(3-site);local=S.zeros(2)
 for a in range(16):
  for j in range(16):
   if a&~mask==j&~mask:local[int(bool(a&mask)),int(bool(j&mask))]+=rho[a,j]
 assert local==S.eye(2)/2
 locals.append(matrix(local))
triangle=sum((op(3,{i:Z,j:Z}) for i,j in [(0,1),(1,2),(0,2)]),S.zeros(8));energies=[fraction(triangle[i,i]) for i in range(8)];assert energies.count('-1')==6 and energies.count('3')==2
commXX=M*XX-XX*M;commYY=M*YY-YY*M;expected=2*S.I*(S.kronecker_product(Y,X)+S.kronecker_product(X,Y));assert commXX==expected and commYY==-expected
phase=S.Matrix([1,S.I,0,0]);coefficients=S.Matrix(2,2,list(phase));rhoRight=coefficients.T*coefficients.conjugate()/2
assert S.trace(rhoRight*Y)==1
singlet=S.Matrix([0,1,-1,0]);triplet=S.Matrix([0,1,1,0]);assert (XX+YY+ZZ)*singlet==-3*singlet and (XX+YY+ZZ)*triplet==triplet
result=dict(method='independent SymPy Kronecker/H squared/symbolic field/partial trace and actual commutators',basisActions={key:matrix(A) for key,A in [('X0',op(2,{0:X})),('Y1',op(2,{1:Y})),('Z0Z1',ZZ)]},rawNormalization=exact_moments(-ZZ,S.Matrix([2,2*S.I,0,0])),xySpectrum={str(v):m for v,m in (XX+YY).eigenvals().items()},heisenbergSpectrum={str(v):m for v,m in (XX+YY+ZZ).eigenvals().items()},ghz=dict(norm=fraction(N),energy=fraction(E),variancePolynomial=str(var),transverseImagesGram=matrix(gram),halfField=exact_moments(H.subs(h,S.Rational(1,2)),z),doubleField=exact_moments(H.subs(h,2),z),localDensities=locals,xProductPure='1',xProductMixture='0',globalPurityPure=fraction(S.trace(rho*rho)),globalPurityMixture=fraction(S.trace(mixture*mixture)),xParityConserved=True,zParityBrokenForNonzeroField=True),triangle=dict(basisEnergies=energies,groundEnergy='-1',groundMultiplicity=6,gap='4'),commutators=dict(MwithXX=matrix(commXX),MwithYY=matrix(commYY),MwithSumZero=True),complexRightDensity=matrix(rhoRight),complexRightY='1',physicalSpinCoefficient='J_S*hbar^2/4')
result['ghz']['pairZComparison']=pair_z
output=Path('scripts/fixtures/spin-chain-authoring-reference.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(output),symbolicVariance=str(var),checks='basis,moments,phase,singlet/triplet,GHZ,mixture,conservation,frustration,shift/scale')))
