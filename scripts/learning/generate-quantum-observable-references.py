"""Independent SymPy eigenspaces, normalized kets and density trace laws.

No production imports. References include zero branches and merged spectra.
"""
import argparse
import json
from pathlib import Path
import sympy as s


def vec(p, prefix, n):
    return s.Matrix([p[f'{prefix}Real{i}'] + s.I*p[f'{prefix}Imag{i}'] for i in range(n)])


def put(prefix, values):
    result = {}
    for i, value in enumerate(values):
        result[f'{prefix}Real{i}'] = int(s.re(value))
        result[f'{prefix}Imag{i}'] = int(s.im(value))
    return result


def ratio(p, prefix):
    return s.Rational(p[prefix+'Numerator'], p[prefix+'Denominator'])


def encode(x):
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    x = s.simplify(x)
    return str(x) if x.is_Rational else 'decimal:'+str(s.N(x, 170))


def cv(x):
    return dict(real=s.re(x).simplify(), imaginary=s.im(x).simplify())


def mat(x):
    return [[cv(x[i,j]) for j in range(x.cols)] for i in range(x.rows)]


def amps(x):
    return [cv(v) for v in x]


def outer(v):
    return s.simplify(v*v.H)


def spectral(A):
    # Independent eigensolver, not the production rank-one/complement split.
    A = A.applyfunc(s.simplify)
    result = []
    for eigenvalue, _, vectors in sorted(A.eigenvects(), key=lambda e:e[0]):
        orthogonal = s.GramSchmidt(vectors, True)
        P = sum((outer(v) for v in orthogonal), s.zeros(A.rows))
        result.append((eigenvalue, P))
    return result


def measurement(psi, parts):
    branches = []
    rho = s.zeros(psi.rows)
    mean = second = s.S.Zero
    for value, P in parts:
        projected = s.simplify(P*psi)
        probability = s.simplify((projected.H*projected)[0])
        conditional = None if probability == 0 else amps(projected/s.sqrt(probability))
        branches.append(dict(eigenvalue=value, rank=s.trace(P), projector=mat(P), probability=probability, conditionalAmplitudes=conditional, expectationContribution=value*probability))
        rho += outer(projected)
        mean += value*probability
        second += value**2*probability
    A = sum((value*P for value,P in parts), s.zeros(psi.rows))
    return dict(matrix=mat(A), outcomes=branches, mean=mean, secondMoment=second, variance=s.simplify(second-mean**2), nonselectiveDensity=mat(rho), nonselectivePurity=s.trace(rho*rho)), rho


def sequence(psi, first, second):
    # Trace of an unnormalized post-measurement density, independently of norms.
    joints = [[s.simplify(s.trace(R*outer(P*psi))) for _,R in second] for _,P in first]
    rows = [sum(row) for row in joints]
    columns = [sum(row[j] for row in joints) for j in range(len(second))]
    return dict(jointProbabilities=joints, firstMarginal=rows, secondMarginal=columns, conditionalSecondProbabilities=[None if rows[i]==0 else [v/rows[i] for v in row] for i,row in enumerate(joints)])


finite_base = {**put('state',[1,s.I]), **put('axis',[1,0]), **put('basis',[1,0]), 'centerNumerator':0,'centerDenominator':1,'gapNumerator':1,'gapDenominator':1}
finite_specs = [
 ('Z with imaginary relative phase',{}),
 ('Z eigenstate',put('state',[1,0])),
 ('negative Z eigenstate',put('state',[0,1])),
 ('X eigenstate',{**put('state',[1,1]),**put('axis',[1,1])}),
 ('Y eigenstate',put('axis',[1,s.I])),
 ('negative Y eigenstate',put('axis',[1,-s.I])),
 ('complex direction',put('axis',[2+3*s.I,-4+5*s.I])),
 ('complex raw state',put('state',[-3+2*s.I,5-7*s.I])),
 ('passive complex basis',put('basis',[2+3*s.I,-4+5*s.I])),
 ('imaginary passive basis',put('basis',[1,s.I])),
 ('fully degenerate zero',{'gapNumerator':0}),
 ('fully degenerate nonzero',{'gapNumerator':0,'centerNumerator':7,'centerDenominator':3}),
 ('negative gap',{'gapNumerator':-3,'gapDenominator':7}),
 ('shift and scale',{'centerNumerator':-2,'centerDenominator':3,'gapNumerator':5,'gapDenominator':11}),
 ('same mean zero spectrum',{'gapNumerator':0,**put('axis',[1,1])}),
 ('global imaginary phase',put('state',[s.I,-1])),
 ('global negative scale',put('state',[-7,-7*s.I])),
 ('axis negative scale',put('axis',[-5,0])),
 ('axis complex phase',put('axis',[3+4*s.I,0])),
 ('small first state coordinate',put('state',[1,1000000])),
 ('small second state coordinate',put('state',[1000000,1])),
 ('maximum gap',{'gapNumerator':1000000,'gapDenominator':1}),
 ('minimum rational gap',{'gapNumerator':1,'gapDenominator':1000000}),
 ('large complex coordinates',{**put('state',[1000000-999999*s.I,-999999+1000000*s.I]),**put('axis',[-1000000+999999*s.I,1000000+1000000*s.I]),**put('basis',[999999+1000000*s.I,-1000000+999999*s.I])}),
]
finite_cases=[]
for label, patch in finite_specs:
    p={**finite_base,**patch}; z,v,b=map(lambda k:vec(p,k,2),['state','axis','basis']);N=(z.H*z)[0];B=(b.H*b)[0];psi=z/s.sqrt(N)
    P=v*v.H/(v.H*v)[0];a,beta=ratio(p,'center'),ratio(p,'gap');A=a*s.eye(2)+beta*(2*P-s.eye(2));parts=spectral(A);m,_=measurement(psi,parts)
    u,w=b;U=s.Matrix([[u,-s.conjugate(w)],[w,s.conjugate(u)]])/s.sqrt(B);coordinates=U.H*psi;passive=s.simplify(U.H*A*U);nilpotent=s.Matrix([[0,1],[0,0]])
    expected=dict(rawNormSquared=N,stateAmplitudes=amps(psi),**m,passiveBasis=dict(rawNormSquared=B,columns=[amps(U[:,i])for i in range(2)],coordinates=amps(coordinates),matrix=mat(passive),mean=cv((coordinates.H*passive*coordinates)[0])),nonHermitianCounterexample=dict(matrix=mat(nilpotent),expectation=cv((psi.H*nilpotent*psi)[0]),hermiticityDefect=mat(nilpotent-nilpotent.H)))
    finite_cases.append(dict(label=label,input=p,expected=encode(expected)))

seq_base={**put('state',[0,1,1]),**put('axisA',[1,0,0]),**put('axisB',[0,1,1]),'aPositiveNumerator':1,'aPositiveDenominator':1,'aNegativeNumerator':2,'aNegativeDenominator':1,'bPositiveNumerator':1,'bPositiveDenominator':1,'bNegativeNumerator':0,'bNegativeDenominator':1}
seq_specs=[
 ('degenerate coherence preserved',{}),
 ('rank-one zero complementary branch',put('state',[1,0,0])),
 ('rank-two eigenvector',put('state',[0,1,0])),
 ('imaginary coherence',{**put('state',[0,1,s.I]),**put('axisB',[0,1,s.I])}),
 ('negative coherence',{**put('state',[0,1,-1]),**put('axisB',[0,1,-1])}),
 ('order-dependent first plane',{**put('state',[1,0,0]),**put('axisB',[1,1,0])}),
 ('order-dependent superposition',{**put('state',[1,1,1]),**put('axisB',[1,1,0])}),
 ('commuting identical axes',put('axisB',[1,0,0])),
 ('commuting orthogonal axes',put('axisB',[0,1,0])),
 ('same state statistics do not imply commuting',{**put('state',[0,0,1]),**put('axisB',[1,1,0])}),
 ('fully degenerate A',{'aNegativeNumerator':1}),
 ('fully degenerate B',{'bNegativeNumerator':1}),
 ('both fully degenerate',{'aNegativeNumerator':1,'bNegativeNumerator':1}),
 ('equal rational eigenvalues',{'aPositiveNumerator':2,'aPositiveDenominator':6,'aNegativeNumerator':1,'aNegativeDenominator':3}),
 ('negative eigenvalue sorting',{'aPositiveNumerator':-3,'aNegativeNumerator':-7,'bPositiveNumerator':-2}),
 ('first standard axis survives',put('axisA',[0,1,0])),
 ('third coordinate axis',put('axisA',[0,0,1])),
 ('complex rank-one projector',put('axisA',[1,s.I,1-s.I])),
 ('complex second projector',put('axisB',[-2+s.I,3-2*s.I,1+s.I])),
 ('complex preparation',put('state',[2+3*s.I,-1+2*s.I,4-s.I])),
 ('global imaginary scale',put('state',[0,3*s.I,3*s.I])),
 ('axis imaginary scale',put('axisA',[5*s.I,0,0])),
 ('small rank-one probability',put('state',[1,1000000,1000000])),
 ('near standard axis',put('axisA',[1000000,1,0])),
 ('large complex coordinates',{**put('state',[1000000+999999*s.I,-1000000+1000000*s.I,999999-1000000*s.I]),**put('axisA',[1000000,999999*s.I,-1000000]),**put('axisB',[999999+s.I,1000000,-999999*s.I])}),
 ('maximum rational eigenvalues',{'aPositiveNumerator':1000000,'aPositiveDenominator':999999,'aNegativeNumerator':-999999,'aNegativeDenominator':1000000}),
]
seq_cases=[]
for label,patch in seq_specs:
    p={**seq_base,**patch};z,v,w=map(lambda k:vec(p,k,3),['state','axisA','axisB']);N=(z.H*z)[0];psi=z/s.sqrt(N);P=s.simplify(v*v.H/(v.H*v)[0]);Q=s.eye(3)-P;R=s.simplify(w*w.H/(w.H*w)[0]);A=ratio(p,'aPositive')*P+ratio(p,'aNegative')*Q;B=ratio(p,'bPositive')*R+ratio(p,'bNegative')*(s.eye(3)-R);pa,pb=spectral(A),spectral(B)
    # Independent Gram-Schmidt starting with v then the standard basis.
    chosen=next(i for i in range(3)if (Q[:,i].H*Q[:,i])[0]!=0);orth=s.GramSchmidt([v,s.eye(3)[:,chosen]],True);S=outer(orth[1]);T=s.simplify(Q-S);fine=[(ratio(p,'aPositive'),P),(ratio(p,'aNegative'),S),(ratio(p,'aNegative'),T)]
    ma,da=measurement(psi,pa);mb,db=measurement(psi,pb);mf,df=measurement(psi,fine);diff=s.simplify(da-df);comm=s.simplify(A*B-B*A)
    expected=dict(rawNormSquared=N,stateAmplitudes=amps(psi),a=ma,b=mb,refinedA=mf,aThenB=sequence(psi,pa,pb),bThenA=sequence(psi,pb,pa),refinedAThenB=sequence(psi,fine,pb),commutator=mat(comm),commutatorSquaredNorm=s.trace(comm.H*comm),originalDensity=mat(outer(psi)),coarseMinusRefinedDensity=mat(diff),coarseRefinementSquaredDistance=s.trace(diff.H*diff),secondMarginalCoarseMinusRefined=[s.trace(r*diff)for _,r in pb])
    seq_cases.append(dict(label=label,input=p,expected=encode(expected)))

result=dict(method='Independent SymPy eigenvects and orthonormalized eigenspaces; normalized complex kets, density trace joint laws, Gram-Schmidt refinement; exact rational matrices and170-digit irrational coordinates.',finiteCases=finite_cases,sequentialCases=seq_cases)
a=argparse.ArgumentParser();a.add_argument('--output',default='scripts/fixtures/quantum-observable-reference.json');args=a.parse_args();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(finitePreparations=len(finite_cases),sequentialPreparations=len(seq_cases))))
