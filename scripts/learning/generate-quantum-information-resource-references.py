"""Dense SymPy code/projectors; exact convex hull of target permutations for LOCC.
Vidal rates use theorem tail ratios; entropy centers are130-decimal mpmath, not interval proofs.
"""
import itertools,json
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import sympy as s
import mpmath as mp
mp.mp.dps=130
I=s.eye(2);X=s.Matrix([[0,1],[1,0]]);Y=s.Matrix([[0,-s.I],[s.I,0]]);Z=s.diag(1,-1);paulis=dict(I=I,X=X,Y=Y,Z=Z);root=s.sqrt(2)
def wire(z):
 re,im=s.expand_complex(s.simplify(z)).as_real_imag()
 def pair(x):
  x=s.expand(s.radsimp(x));b=x.coeff(root);a=s.simplify(x-b*root)
  assert a.is_Rational and b.is_Rational
  return [str(a),str(b)]
 return dict(real=pair(re),imaginary=pair(im))
def matrix_wire(a):return [[wire(x) for x in row] for row in a.tolist()]
def norm(v):return s.simplify((v.H*v)[0])
def density(v):return s.simplify(v*v.H/norm(v))
@lru_cache(None)
def operator(word):return s.kronecker_product(*[paulis[c] for c in word])
V=s.zeros(8,2);V[0,0]=1;V[7,1]=1
S0=s.kronecker_product(Z,Z,I);S1=s.kronecker_product(I,Z,Z)
projectors={(a,b):(s.eye(8)+(-1)**a*S0)*(s.eye(8)+(-1)**b*S1)/4 for a,b in itertools.product([0,1],repeat=2)}
corrections={(0,0):'III',(1,0):'XII',(1,1):'IXI',(0,1):'IIX'}
recoveries={key:V.H*operator(corrections[key])*P for key,P in projectors.items()}
inputs=[s.Matrix([1,0]),s.Matrix([0,1]),s.Matrix([1,1]),s.Matrix([1,s.I]),s.Matrix([s.Rational(3,5),s.I*s.Rational(4,5)]),s.Matrix([1+root,s.I])]
words=[''.join(word) for word in itertools.product('IXYZ',repeat=3)]
cases=[]
for index,psi in enumerate(inputs):
 selected=words if index==4 else ['III','XII','IXI','IIX','XXI','XXX','ZII','YII']
 for word in selected:
  E=operator(word);encoded=V*psi;errored=E*encoded;branches=[]
  for (a,b),R in recoveries.items():
   K=R*E*V;raw=K*psi;prob=s.simplify(norm(raw)/norm(psi))
   branches.append(dict(syndrome=[a,b],correction=corrections[a,b],probability=[str(prob),'0'],logicalOperator=matrix_wire(K),rawRecovered=[wire(z) for z in raw],conditionalLogicalDensity=matrix_wire(density(raw)) if prob else None))
  cases.append(dict(name=f'{index}/{word}',input=[wire(z) for z in psi],error=word,encodedDensity=matrix_wire(density(encoded)),erroredDensity=matrix_wire(density(errored)),branches=branches))
noise=[]
bell=s.Matrix([1,0,0,1])/root;bell_density=density(bell)
for index,psi in enumerate(inputs):
 for p in map(s.Rational,['0','1/10','1/2','1']):
  output=s.zeros(2);choi=s.zeros(4)
  for flags in itertools.product([0,1],repeat=3):
   word=''.join('X' if bit else 'I' for bit in flags);weight=p**sum(flags)*(1-p)**(3-sum(flags));E=operator(word)
   for R in recoveries.values():
    K=R*E*V
    output+=weight*K*density(psi)*K.H
    full=s.kronecker_product(I,K);choi+=weight*full*bell_density*full.H
  noise.append(dict(name=f'{index}/{p}',input=[wire(z) for z in psi],p=str(p),logicalOutput=matrix_wire(output),normalizedLogicalChoi=matrix_wire(choi),inputSquaredFidelity=wire(s.trace(density(psi)*output)),entanglementFidelity=wire(s.trace(bell_density*choi))))
families=[['III','XII','IXI','IIX'],['III','ZII'],['III','YII'],['III','XXX'],['ZII','IZI','IIZ'],['XII','IXI'],['III','ZZI','IZZ','ZIZ'],['XII','YII'],['III','ZZZ'],['XXX'],['III','XII','IXI','IIX','XXI','XIX','IXX','XXX']]
kl=[]
for family in families:
 pairs=[]
 for i,a in enumerate(family):
  for j,b in enumerate(family):
   compressed=V.H*operator(a).H*operator(b)*V;scalar=compressed==compressed[0,0]*s.eye(2)
   pairs.append(dict(a=i,b=j,compressed=matrix_wire(compressed),scalarOnCode=scalar,c=wire(compressed[0,0]) if scalar else None))
 kl.append(dict(words=family,conditionHolds=all(p['scalarOnCode'] for p in pairs),pairs=pairs))
# Exact Birkhoff convex-hull feasibility, rather than the product prefix test.
def convex_possible(source,target):
 points=list(set(itertools.permutations(target)));rhs=s.Matrix(list(source))
 for count in range(1,min(3,len(points))+1):
  for subset in itertools.combinations(points,count):
   A=s.Matrix.hstack(*[s.Matrix(list(p)) for p in subset])
   if A.rank()!=count:continue
   try:solution,_=A.gauss_jordan_solve(rhs)
   except ValueError:continue
   if all(x>=0 for x in solution):return True
 return False
vectors=sorted(set(tuple(sorted((F(a,6),F(b,6),F(6-a-b,6)),reverse=True)) for a in range(7) for b in range(7-a)))
vectors+= [(F(1,2),F(1,4),F(1,4)),(F(2,5),F(2,5),F(1,5)),(F(3,5),F(3,10),F(1,10))]
vectors=list(dict.fromkeys(vectors))
def vidal(a,b):
 return min(sum(a[k:])/sum(b[k:]) for k in range(3) if sum(b[k:]))
def entropy(a):
 return mp.nstr(-sum(mp.mpf(x.numerator)/x.denominator*mp.log(mp.mpf(x.numerator)/x.denominator,2) for x in a if x),120)
major=[]
for a,b in itertools.product(vectors,repeat=2):
 major.append(dict(source=list(map(str,a)),target=list(map(str,b)),forward=convex_possible(a,b),reverse=convex_possible(b,a),pForward=str(vidal(a,b)),pReverse=str(vidal(b,a)),entropySource=entropy(a),entropyTarget=entropy(b)))
output=Path('scripts/fixtures/quantum-information-resources-independent.json')
output.write_text(json.dumps(dict(method='SymPy dense code/projectors and exact permutation-convex-hull feasibility;Vidal theorem tail rates;130-decimal entropy centers',cases=cases,noise=noise,kl=kl,majorization=major),indent=2)+'\n')
print(json.dumps(dict(selectedPauliCases=len(cases),iidNoiseCases=len(noise),klFamilies=len(kl),majorizationPairs=len(major),file=str(output))))
