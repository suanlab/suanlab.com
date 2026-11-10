"""Independent exact eigenspaces and high-precision manuscript comparisons.

Uses SymPy eigenvectors/Gram-Schmidt, not product polynomial projectors.
"""
import json
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250


def fraction(x):
    x=S.simplify(x);assert x.is_Rational,x
    return str(x)


def cv(x):return dict(real=fraction(S.re(x)),imaginary=fraction(S.im(x)))
def matrix(A):return [[cv(A[i,j]) for j in range(A.cols)] for i in range(A.rows)]
def scalar(x):return mp.mpf(str(S.N(S.simplify(x),255)))
def dec(x):return 'decimal:'+mp.nstr(scalar(x),245)
def cdec(x):return dict(real=dec(S.re(x)),imaginary=dec(S.im(x)))
def mdec(A):return [[cdec(A[i,j]) for j in range(A.cols)] for i in range(A.rows)]


def prep(sites,budget,entries,cut=None):
    p=dict(sites=sites,retainedRank=budget,selectedCut=cut or sites//2,**{f'state{part}{i}':0 for i in range(16) for part in ['Real','Imag']})
    for i,v in entries.items():p[f'stateReal{i}']=int(S.re(v));p[f'stateImag{i}']=int(S.im(v))
    return p


def reference(p):
    z=S.Matrix([p[f'stateReal{i}']+S.I*p[f'stateImag{i}'] for i in range(2**p['sites'])]);N=(z.H*z)[0];cuts=[]
    for k in range(1,p['sites']):
        nl,nr=2**k,2**(p['sites']-k);C=S.Matrix(nl,nr,list(z));L=(C*C.H/N).applyfunc(S.simplify);R=(C.T*C.conjugate()/N).applyfunc(S.simplify);side='left' if nl<=nr else 'right';rho=L if side=='left' else R;rank=C.rank();groups=[]
        for weight,multiplicity,vectors in rho.eigenvects():
            orth=S.GramSchmidt(vectors,orthonormal=True);assert len(orth)==multiplicity
            P=sum((v*v.H for v in orth),S.zeros(rho.rows)).applyfunc(S.simplify)
            assert S.simplify(P*P-P)==S.zeros(rho.rows)
            groups.append(dict(weight=weight,multiplicity=multiplicity,zero=bool(weight==0),projector=P))
        groups.sort(key=lambda g:scalar(g['weight']),reverse=True)
        remaining=min(p['retainedRank'],rank);kept=S.Rational(0);Pkeep=S.zeros(rho.rows);boundary=False
        for g in groups:
            n=0 if g['zero'] else min(remaining,g['multiplicity']);remaining-=n;kept+=n*g['weight'];boundary|=0<n<g['multiplicity']
            if n==g['multiplicity'] and not g['zero']:Pkeep+=g['projector']
        kept=S.simplify(kept);discarded=S.simplify(1-kept);chosen=None
        if not boundary:
            T=(Pkeep*C if side=='left' else C*Pkeep.T).applyfunc(S.simplify);rest=C-T;direct=S.simplify(S.trace(rest.H*rest)/N);norm=S.simplify(S.trace(T.H*T)/N)
            assert S.simplify(direct-discarded)==0 and S.simplify(norm-kept)==0
            assert S.simplify(S.trace(T.H*rest))==0
            chosen=dict(retainedPositiveSupportProjector=mdec(Pkeep),truncatedRawCoefficientMatrix=mdec(T),amplitudesOnNormalizedInput=[cdec(v/S.sqrt(N)) for v in T],normalizedAmplitudes=[cdec(v/S.sqrt(N*kept)) for v in T] if kept else None,directSquaredError=dec(direct),directRetainedNormSquared=dec(norm))
        entropy=-sum(g['multiplicity']*g['weight']*S.log(g['weight']) for g in groups if not g['zero'])
        cuts.append(dict(cut=k,leftDimension=nl,rightDimension=nr,smallerSide=side,schmidtRank=rank,rhoLeft=matrix(L),rhoRight=matrix(R),purity=fraction(S.trace(L*L)),leftZeroMultiplicity=nl-rank,rightZeroMultiplicity=nr-rank,entropyNats=dec(entropy),entropyBits=dec(entropy/S.log(2)),spectrum=[dict(weight=dec(g['weight']),singularValue=dec(S.sqrt(g['weight'])),multiplicity=g['multiplicity'],zeroWeight=g['zero'],projectorOnSmallerPhysicalSide=mdec(g['projector'])) for g in groups],truncation=dict(retainedPositiveRank=min(p['retainedRank'],rank),boundaryDegenerate=boundary,optimalUnnormalizedStateUnique=not boundary,normalizedStateDefinedForEveryOptimum=bool(kept),retainedWeight=dec(kept),discardedWeight=dec(discarded),optimalUnnormalizedSquaredError=dec(discarded),optimalNormalizedFidelitySquared=dec(kept) if kept else None,optimalNormalizedDistanceSquared=dec(2*(1-S.sqrt(kept))) if kept else None,chosen=chosen)))
    return dict(rawNormSquared=fraction(N),cuts=cuts)


cases=[]
for budget in [0,1,4]:cases.append(dict(label=f'unequal two-mode budget{budget}',input=prep(2,budget,{0:2,3:1})))
cases.append(dict(label='Bell split positive group',input=prep(2,1,{0:1,3:1})))
for budget in range(5):cases.append(dict(label=f'diagonal3-2-2-1 budget{budget}',input=prep(4,budget,{0:3,5:2,10:2,15:1})))
cases.extend([dict(label='right phase projector transpose',input=prep(3,1,{0:2,1:2*S.I,2:1,3:-S.I},2)),dict(label='tiny positive second weight',input=prep(2,1,{0:1000000,1:999999,2:999999,3:999998})),dict(label='adjacent Bell pairs',input=prep(4,4,{0:1,3:1,12:1,15:1})),dict(label='crossing Bell pairs',input=prep(4,4,{0:1,5:1,10:1,15:1}))])
for case in cases:
    case['expected']=reference(case['input']);print('Verified '+case['label'],flush=True)
phase=S.Matrix([[2,2*S.I],[1,-S.I],[0,0],[0,0]]);P=S.Matrix([[1,-S.I],[S.I,1]])/2;wrong=phase-phase*P;wrongError=S.trace(wrong.H*wrong)/10;assert wrongError==S.Rational(4,5)
phi=S.Matrix([1,0,0,1])/S.sqrt(2);choices=[S.Matrix([1,0,0,0])/S.sqrt(2),S.Matrix([0,0,0,1])/S.sqrt(2)]
for c in choices:assert S.simplify(((phi-c).H*(phi-c))[0])==S.Rational(1,2)
M=S.Integer(10**6);C=S.Matrix([[M,M-1],[M-1,M-2]]);assert C.det()==-1
# Pure-state trace distance bound: rho - |00><00| has eigenvalues +/-sqrt(epsilon).
e=S.symbols('epsilon',positive=True);difference=S.Matrix([[-e,S.sqrt(e*(1-e))],[S.sqrt(e*(1-e)),e]]);t=S.symbols('t');assert S.simplify(difference.charpoly(t).as_expr()-(t*t-e))==0
result=dict(method='independent exact SymPy eigenvectors and Gram-Schmidt eigenspaces;exact partial traces and projected coefficients;250dps decimal evaluation',cases=cases,analytic=dict(wrongRightProjectionSquaredError=fraction(wrongError),bellDistinctOptimaSquaredError='1/2',tinyRawDeterminant='-1',tinyRawNorm=fraction(S.trace(C.H*C)),boundedObservableTraceDistanceSquared='epsilon'))
output=Path('scripts/fixtures/schmidt-authoring-reference.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(output),preparations=len(cases))),flush=True)
