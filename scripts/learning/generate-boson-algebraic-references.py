"""Independent number-field arithmetic/sign/interval references via SymPy."""
import json
import random
from pathlib import Path
import sympy as S
roots=[S.Integer(1),S.sqrt(2),S.sqrt(3),S.sqrt(6)]
def expression(a):return sum(S.Rational(x)*r for x,r in zip(a,roots))
def encode(x):
 x=S.expand(S.radsimp(x));b=[x.coeff(r) for r in roots[1:]];a=S.simplify(x-sum(v*r for v,r in zip(b,roots[1:])));assert a.is_Rational and all(v.is_Rational for v in b)
 return dict(coefficients=[str(v) for v in [a]+b],exactSign=int(S.sign(x)),observation=str(a) if all(v==0 for v in b) else 'decimal:'+str(S.N(x,245)))
rng=random.Random(20261109);cases=[]
def save(label,a,b):
 x,y=expression(a),expression(b);assert y!=0
 cases.append(dict(label=label,a=a,b=b,expected=dict(a=encode(x),b=encode(y),add=encode(x+y),subtract=encode(x-y),multiply=encode(x*y),divide=encode(x/y))))
for i in range(48):
 a=[str(S.Rational(rng.randint(-100,100),rng.randint(1,50))) for _ in roots];b=[str(S.Rational(rng.randint(-100,100),rng.randint(1,50))) for _ in roots];save(f'general four-coordinate field case{i}',a,b)
# Very small values from exact Pell cancellation, requiring exact sign decisions.
x,y=1,1
for _ in range(140):x,y=x+2*y,x+y
save('Pell near cancellation positive or negative',[str(x),str(-y),'0','0'],['1','1','1','1'])
save('nested quadratic cancellation',['0','1','-1','0'],['0','1','1','0'])
save('mixed two radical components',['-1','2','3','-4'],['5','-6','-7','8'])
save('zero numerator',['0','0','0','0'],['1','-1','1','-1'])
output=Path('scripts/fixtures/boson-algebraic-reference.json');output.write_text(json.dumps(dict(method='independent SymPy radsimp/exact signs/245 decimal digits',cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),output=str(output))))
