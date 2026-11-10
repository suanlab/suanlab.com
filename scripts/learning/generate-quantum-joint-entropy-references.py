"""Exact SymPy density/partial-transpose spectra,130-decimal entropy centers."""
import itertools,json
from pathlib import Path
import sympy as s
import mpmath as m
m.mp.dps=130
bell=s.Matrix([1,0,0,1])/s.sqrt(2);corr=s.diag(s.Rational(1,2),0,0,s.Rational(1,2));I=s.eye(4)
D=10**30
parameters=list(itertools.product(map(s.Rational,['0','1/3','1/2','4/5','1']),map(s.Rational,['0','1/4','1/2','3/4','1'])))
parameters += [(s.Rational(D-1,3*D),s.Integer(1)),(s.Rational(D+1,3*D),s.Integer(1))]
cases=[]
for c,t in parameters:
 rho=c*(t*bell*bell.H+(1-t)*corr)+(1-c)*I/4
 eigenvalues=[v for v,count in rho.eigenvals().items() for _ in range(count)]
 pt=s.Matrix(4,4,lambda i,j:rho[(i//2)*2+j%2,(j//2)*2+i%2])
 pe=[v for v,count in pt.eigenvals().items() for _ in range(count)]
 assert all(v>=0 for v in eigenvalues) and s.trace(rho)==1
 local=s.Matrix(2,2,lambda i,j:sum(rho[2*i+k,2*j+k] for k in range(2)))
 assert local==s.eye(2)/2
 entropy=-sum(m.mpf(str(v.p))/int(v.q)*m.log(m.mpf(str(v.p))/int(v.q),2) for v in eigenvalues if v)
 cases.append(dict(c=str(c),t=str(t),density=[[str(x) for x in row] for row in rho.tolist()],partialTransposeEigenvalues=list(map(str,pe)),negativity=str(-sum(v for v in pe if v<0)),entangled=any(v<0 for v in pe),jointEntropy=m.nstr(entropy,120)))
output=Path('scripts/fixtures/quantum-joint-entropy-independent.json')
output.write_text(json.dumps(dict(method='Exact SymPy4x4 density and partial-transpose eigensystems;130-decimal entropy centers',cases=cases),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),file=str(output))))
