"""Manuscript examples: explicit virtual paths and dense physical algebra.

No product implementation or product numerical helpers are imported.
"""
import itertools
import json
from pathlib import Path
import sympy as S


def rational(x):
    x = S.simplify(x)
    assert x.is_Rational, x
    return str(x)


def cv(z):
    return dict(real=rational(S.re(z)), imaginary=rational(S.im(z)))


def matrix(m):
    return [[cv(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def frob(m):
    return S.simplify(S.trace(m.H * m))


def paths(tensors, start, end):
    dims = [tensors[0][0].rows] + [t[0].cols for t in tensors]
    result = {}
    for bits in itertools.product(range(2), repeat=end-start):
        for a in range(dims[start]):
            for b in range(dims[end]):
                value = 0
                for internal in itertools.product(*(range(dims[k]) for k in range(start+1, end))):
                    route = (a,) + internal + (b,)
                    value += S.prod(tensors[start+i][s][route[i], route[i+1]] for i, s in enumerate(bits))
                result[(bits, a, b)] = S.simplify(value)
    return result


def evaluate(tensors):
    n = len(tensors)
    values = paths(tensors, 0, n)
    z = S.Matrix([values[(bits, 0, 0)] for bits in itertools.product(range(2), repeat=n)])
    norm = S.simplify((z.H*z)[0])
    left, right, closures, cuts = [], [], [], []
    for k in range(n+1):
        dim = tensors[k][0].rows if k < n else 1
        lp, bp = paths(tensors, 0, k), paths(tensors, k, n)
        lb = list(itertools.product(range(2), repeat=k))
        rb = list(itertools.product(range(2), repeat=n-k))
        L = S.Matrix([[lp[(bits, 0, a)] for a in range(dim)] for bits in lb])
        B = S.Matrix([[bp[(bits, a, 0)] for bits in rb] for a in range(dim)])
        E, R = L.H*L, B.conjugate()*B.T
        closure = S.simplify(sum(E[a,b]*R[a,b] for a in range(dim) for b in range(dim)))
        assert closure == norm
        left.append(matrix(E)); right.append(matrix(R)); closures.append(rational(closure))
        if 0 < k < n:
            C = S.Matrix(2**k, 2**(n-k), list(z)); assert S.simplify(C-L*B) == S.zeros(C.rows,C.cols)
            cuts.append(dict(coefficientMatrix=matrix(C), coefficientRank=C.rank(), schmidtRank=C.rank() if norm else None))
    local = []
    if norm:
        for site in range(n):
            rho = S.zeros(2); mask = 2**(n-site-1)
            for a in range(2**n):
                for b in range(2**n):
                    if a & ~mask == b & ~mask:
                        rho[int(bool(a & mask)),int(bool(b & mask))] += z[a]*S.conjugate(z[b])/norm
            local.append(dict(density=matrix(rho),bloch=[rational(2*S.re(rho[0,1])),rational(-2*S.im(rho[0,1])),rational(rho[0,0]-rho[1,1])],purity=rational(S.trace(rho*rho))))
    canonicality = []
    for tensor in tensors:
        lg = sum((a.H*a for a in tensor),S.zeros(tensor[0].cols))
        rg = sum((a*a.H for a in tensor),S.zeros(tensor[0].rows))
        canonicality.append(dict(leftGram=matrix(lg),rightGram=matrix(rg),leftIsometry=lg==S.eye(lg.rows),rightIsometry=rg==S.eye(rg.rows)))
    return dict(tensors=[[matrix(a) for a in t] for t in tensors],state=[cv(v) for v in z],normSquared=rational(norm),tensorParameterSquaredNorm=rational(sum(frob(a) for t in tensors for a in t)),leftEnvironments=left,rightEnvironments=right,cutClosureNorms=closures,cuts=cuts,canonicality=canonicality,physical=dict(localStates=local) if norm else None,stateDefined=bool(norm))


def encode(tensors,G):
    n=len(tensors);dims=[1]+[t[0].cols for t in tensors]
    p=dict(sites=n,tensorDenominator=1,**{f'bond{i}':dims[i+1] if i<n-1 else 0 for i in range(3)},gaugeBond=0 if n>1 else -1,gaugeRow=0,gaugeColumn=1 if n>1 and dims[1]>1 else 0,gaugeScaleNumerator=1,gaugeScaleDenominator=1,gaugeShearReal=0,gaugeShearImag=0,gaugeShearDenominator=1)
    p.update({f't{i}{part}{a}{s}{b}':0 for i in range(4) for a in range(4) for s in range(2) for b in range(4) for part in ['Real','Imag']})
    for i,t in enumerate(tensors):
        for s,A in enumerate(t):
            for a in range(A.rows):
                for b in range(A.cols):
                    p[f't{i}Real{a}{s}{b}']=int(S.re(A[a,b]));p[f't{i}Imag{a}{s}{b}']=int(S.im(A[a,b]))
    if G is not None:
        lam=G[0,0];eta=G[0,1]/lam if G.cols>1 else S.Rational(0)
        p.update(gaugeScaleNumerator=int(S.numer(lam)),gaugeScaleDenominator=int(S.denom(lam)),gaugeShearDenominator=int(S.ilcm(S.denom(S.re(eta)),S.denom(S.im(eta)))))
        p.update(gaugeShearReal=int(S.re(eta)*p['gaugeShearDenominator']),gaugeShearImag=int(S.im(eta)*p['gaugeShearDenominator']))
    return p


GHZ=[[S.Matrix([[1,0]]),S.Matrix([[0,1]])],[S.diag(1,0),S.diag(0,1)],[S.Matrix([1,0]),S.Matrix([0,1])]]
G=S.Matrix([[2,1],[0,1]])
zero_complex=[[S.Matrix([[1,S.I]]),S.zeros(1,2)],[S.Matrix([1,S.I]),S.zeros(2,1)]]
zero_real=[[S.Matrix([[1,1]]),S.zeros(1,2)],[S.Matrix([1,-1]),S.zeros(2,1)]]
phase=[[S.Matrix([[1]]),S.zeros(1,1)],[S.Matrix([[1]]),S.Matrix([[S.I]])]]
product4=[[S.Matrix([[1,0,0,0]]),S.zeros(1,4)],[S.diag(1,0,0,0),S.zeros(4)],[S.Matrix([1,0,0,0]),S.zeros(4,1)]]
preparations=[('GHZ3 real scale shear',GHZ,G),('GHZ3 complex shear',GHZ,S.Matrix([[-S.Rational(3,2),-S.Rational(3,5)+S.I*S.Rational(9,10)],[0,1]])),('complex cancellation closure orientation',zero_complex,G),('real cancellation',zero_real,G),('right Y positive product',phase,S.Matrix([[2]])),('bond4 redundant product',product4,S.eye(4)),('one site complex vector',[[S.Matrix([[1]]),S.Matrix([[S.I]])]],None)]
cases=[]
for label,tensors,gauge in preparations:
    changed=[[a.copy() for a in t] for t in tensors]
    if gauge is not None:
        assert gauge.det()!=0
        changed[0]=[a*gauge for a in changed[0]];changed[1]=[gauge.inv()*a for a in changed[1]]
    a,b=evaluate(tensors),evaluate(changed)
    assert a['state']==b['state'] and a['physical']==b['physical']
    cases.append(dict(label=label,input=encode(tensors,gauge),expected=dict(original=a,gauged=b)))
assert cases[0]['expected']['original']['tensorParameterSquaredNorm']=='6'
assert cases[0]['expected']['gauged']['tensorParameterSquaredNorm']=='19/2'
E=S.Matrix([[1,S.I],[-S.I,1]]);assert S.trace(E*E)==4 and sum(v*v for v in E)==0
X=S.Matrix([[0,1],[1,0]]);Z=S.diag(1,-1);I=S.eye(2)
z=S.Matrix([1,0,0,0,0,0,0,1]);rho=z*z.H/2;mix=S.diag(S.Rational(1,2),0,0,0,0,0,0,S.Rational(1,2))
for i,j in itertools.combinations(range(3),2):
    for Oi,Oj in itertools.product([X,S.Matrix([[0,-S.I],[S.I,0]]),Z],repeat=2):
        ops=[I,I,I];ops[i]=Oi;ops[j]=Oj;O=S.kronecker_product(*ops);assert S.trace(rho*O)==S.trace(mix*O)
XXX=S.kronecker_product(X,X,X);ZZI=S.kronecker_product(Z,Z,I)
assert S.trace(rho*XXX)==1 and S.trace(mix*XXX)==0 and S.trace(rho*ZZI)==1
output=Path('scripts/fixtures/mps-authoring-reference.json')
output.write_text(json.dumps(dict(method='SymPy explicit virtual-path sums,direct Gram/dense partial trace and physical Pauli matrices',cases=cases,analytic=dict(wrongTraceClosure='4',correctComplexCancellationNorm='0',ghzXXX='1',mixtureXXX='0',ghzPairZZ='1',mixturePairZZ='1',ghzPairObservationsEqualToMixture=True,ghzActiveEntries=16,redundantProductActiveEntries=48,denseAmplitudes=8)),indent=2)+'\n')
print(json.dumps(dict(output=str(output),independentPreparations=len(cases),ghzParameterLengthBefore='6',ghzParameterLengthAfter='19/2',complexCancellationNorm='0',wrongClosure='4')))
