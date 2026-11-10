"""Independent dense SymPy circuits, ket Born probabilities and posterior entropies."""
import json,itertools
from fractions import Fraction as F
from collections import defaultdict
from pathlib import Path
import sympy as s
root=s.sqrt(2);I=s.eye(2);H=s.Matrix([[1,1],[1,-1]])/root;X=s.Matrix([[0,1],[1,0]]);Z=s.diag(1,-1)
def wire(z):
    re,im=s.expand_complex(s.simplify(z)).as_real_imag()
    def pair(x):
        x=s.expand(s.radsimp(x));b=x.coeff(root);a=s.simplify(x-b*root)
        assert a.is_Rational and b.is_Rational,(a,b)
        return [str(a),str(b)]
    return dict(real=pair(re),imaginary=pair(im))
def matrix_wire(a):return [[wire(z) for z in row] for row in a.tolist()]
def vector_wire(a):return [wire(z) for z in a]
def norm(a):return s.simplify((a.H*a)[0])
def density(a):return s.simplify(a*a.H/norm(a))
def single(g,n,k):return s.kronecker_product(*[g if j==k else I for j in range(n)])
def cx(n,c,t):
    a=s.zeros(2**n)
    for j in range(2**n):a[j^(1<<(n-1-t)) if j&(1<<(n-1-c)) else j,j]=1
    return a
teleports=[];copies=[]
inputs=[s.Matrix([a+s.I*b,c+s.I*d]) for a,b,c,d in itertools.product([-1,0,1],repeat=4) if a or b or c or d]
inputs += [s.Matrix([s.Rational(3,5),s.I*s.Rational(4,5)]),s.Matrix([1+root,s.I]),s.Matrix([root/2,root/2]),s.Matrix([1,root+s.I])]
U=single(H,3,0)*cx(3,0,1)*cx(3,1,2)*single(H,3,1)
for index,psi in enumerate(inputs):
    pre=U*s.kronecker_product(psi,s.Matrix([1,0,0,0]));branches=[]
    for a,b in itertools.product([0,1],repeat=2):
        raw=pre[4*a+2*b:4*a+2*b+2,0];corrected=(Z**a)*(X**b)*raw
        branches.append(dict(a=a,b=b,probability=[str(s.simplify(norm(raw)/norm(psi))),'0'],rawBob=vector_wire(raw),correctedBob=vector_wire(corrected),bobDensity=matrix_wire(density(raw)),correctedDensity=matrix_wire(density(corrected))))
    full=density(pre);bob=s.Matrix(2,2,lambda i,j:sum(full[2*k+i,2*k+j] for k in range(4)))
    teleports.append(dict(name=f'input-{index}',input=vector_wire(psi),branches=branches,bobBeforeMessage=matrix_wire(bob),inputDensity=matrix_wire(density(psi))))
    copied=cx(2,0,1)*s.kronecker_product(psi,s.Matrix([1,0]));joint=density(copied)
    first=s.Matrix(2,2,lambda i,j:sum(joint[2*i+k,2*j+k] for k in range(2)))
    second=s.Matrix(2,2,lambda i,j:sum(joint[2*k+i,2*k+j] for k in range(2)))
    target=s.kronecker_product(density(psi),density(psi))
    copies.append(dict(name=f'input-{index}',joint=matrix_wire(joint),first=matrix_wire(first),second=matrix_wire(second),isExactProductClone=s.simplify(joint-target)==s.zeros(4),singleCopySquaredFidelity=wire(s.trace(density(psi)*first))))
dense=[]
for a,b in itertools.product([0,1],repeat=2):
    bell=cx(2,0,1)*single(H,2,0)*s.Matrix([1,0,0,0])
    for order in ['x-then-z','z-then-x']:
        encoding=(Z**a)*(X**b) if order=='x-then-z' else (X**b)*(Z**a)
        encoded=single(encoding,2,0)*bell;decoded=single(H,2,0)*cx(2,0,1)*encoded
        dense.append(dict(a=a,b=b,order=order,encodedState=vector_wire(encoded),decodedState=vector_wire(decoded),decodedDensity=matrix_wire(density(decoded))))
bb=[]
ket=lambda basis,bit:s.Matrix([1,0]) if basis=='z' and bit==0 else s.Matrix([0,1]) if basis=='z' else s.Matrix([1,(-1)**bit])/root
born=lambda a,b:F(str(s.simplify(abs((a.H*b)[0])**2)))
for attack,eve_z in itertools.product([F(0),F(1,4),F(1,2),F(3,4),F(1)],repeat=2):
    rows=[]
    for ab,bit,bbasis in itertools.product(['z','x'],[0,1],['z','x']):
        for bobbit in [0,1]:rows.append(dict(aliceBasis=ab,aliceBit=bit,bobBasis=bbasis,bobBit=bobbit,attacked=False,eveBasis=None,eveBit=None,weight=F(1,8)*(1-attack)*born(ket(bbasis,bobbit),ket(ab,bit))))
        for eb,ebit,bobbit in itertools.product(['z','x'],[0,1],[0,1]):rows.append(dict(aliceBasis=ab,aliceBit=bit,bobBasis=bbasis,bobBit=bobbit,attacked=True,eveBasis=eb,eveBit=ebit,weight=F(1,8)*attack*(eve_z if eb=='z' else 1-eve_z)*born(ket(eb,ebit),ket(ab,bit))*born(ket(bbasis,bobbit),ket(eb,ebit))))
    sifted=[r for r in rows if r['aliceBasis']==r['bobBasis']];mass=sum(r['weight'] for r in sifted);error=sum(r['weight'] for r in sifted if r['aliceBit']!=r['bobBit'])/mass
    groups=defaultdict(lambda:[F(0),F(0)])
    for r in sifted:groups[(r['aliceBasis'],r['attacked'],r['eveBasis'],r['eveBit'])][r['aliceBit']]+=r['weight']/mass
    conditional=F(0);guess=F(0)
    for weights in groups.values():
        total=sum(weights)
        if not total:continue
        guess+=max(weights)
        if all(weights):
            assert weights[0]==weights[1]
            conditional+=total
    for r in rows:r['weight']=str(r['weight'])
    bb.append(dict(attack=str(attack),eveZ=str(eve_z),rows=rows,qber=str(error),eveInformation=str(1-conditional),eveGuess=str(guess)))
output=Path('scripts/fixtures/quantum-information-protocols-independent.json')
output.write_text(json.dumps(dict(method='SymPy dense tensor circuits, ket Born probabilities and posterior conditional entropy',teleports=teleports,copies=copies,dense=dense,bb84=bb),indent=2)+'\n')
print(json.dumps(dict(teleports=len(teleports),copies=len(copies),dense=len(dense),bb84=len(bb),file=str(output))))
