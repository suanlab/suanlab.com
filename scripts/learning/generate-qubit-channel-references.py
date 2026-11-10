"""Independent symbolic channel, environment and instrument references.

SymPy characteristic-polynomial root counts and symbolic eigenvalue signs
certify positivity; the product uses principal minors and its validator uses
Schur complements. No product TypeScript is imported.
"""
import argparse
import json
from pathlib import Path
import sympy as S

I=S.eye(2); X=S.Matrix([[0,1],[1,0]]); Y=S.Matrix([[0,-S.I],[S.I,0]]); Z=S.diag(1,-1)
P0=S.diag(1,0); P1=S.diag(0,1)
H=S.Matrix([[1,1],[1,-1]])/S.sqrt(2); T=S.diag(1,(1+S.I)/S.sqrt(2))
F=S.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
clean=lambda a:a.applyfunc(S.simplify)


def scalar(z):
    def pair(v):
        v=S.expand(S.simplify(v));b=v.coeff(S.sqrt(2));a=S.simplify(v-b*S.sqrt(2))
        assert a.is_Rational and b.is_Rational,(a,b)
        return [str(a),str(b)]
    return dict(real=pair(S.re(z)),imaginary=pair(S.im(z)))


def wire(a):return [[scalar(v) for v in a.row(i)] for i in range(a.rows)]
def unit(i,j):
    a=S.zeros(2);a[i,j]=1;return a
def partial(a,keep):
    return clean(S.Matrix(2,2,lambda i,j:sum(a[2*i+b,2*j+b] if keep==0 else a[2*b+i,2*b+j] for b in range(2))))


def psd(a):
    if clean(a-a.H)!=S.zeros(a.rows):return False
    polynomial=a.charpoly().as_poly()
    if all(c.is_Rational for c in polynomial.all_coeffs()):
        x=polynomial.gen
        while polynomial.degree()>0 and polynomial.eval(0)==0:
            polynomial=polynomial.exquo(S.Poly(x,x))
        return polynomial.degree()==0 or polynomial.count_roots(-S.oo,0)==0
    eigenvalues=a.eigenvals()
    assert sum(eigenvalues.values())==a.rows
    for value in eigenvalues:
        decision=S.simplify(value).is_nonnegative
        assert decision is not None,value
        if not decision:return False
    return True


def summed(terms):return lambda rho:clean(sum((w*K*rho*K.H for w,K in terms),S.zeros(2)))
def choi(phi):return clean(sum((S.kronecker_product(unit(i,j),phi(unit(i,j))) for i in range(2) for j in range(2)),S.zeros(4)))
def from_choi(a):return lambda rho:clean(sum((rho[i,j]*a[2*i:2*i+2,2*j:2*j+2] for i in range(2) for j in range(2)),S.zeros(2)))
def term_wire(terms):return dict(kind='operator-sum',terms=[dict(weight=str(w),operator=wire(K)) for w,K in terms])
def image_wire(phi):return dict(kind='matrix-units',images=[wire(phi(unit(i,j))) for i in range(2) for j in range(2)])


def map_base(phi):
    J=choi(phi);marginal=partial(J,0);dual=marginal.conjugate();unital=phi(I);cp=psd(J);tp=clean(marginal-I)==S.zeros(2)
    minors=[]
    for mask in range(1,16):
        indices=[i for i in range(4) if mask&(1<<i)]
        minors.append(dict(mask=mask,value=scalar(J.extract(indices,indices).det())))
    return dict(schemaVersion=1,choiConvention='reference-output-unnormalized',choi=wire(J),principalMinors=minors,
                properties=dict(hermiticityPreserving=clean(J-J.H)==S.zeros(4),completelyPositive=cp,tracePreserving=tp,
                                traceNonincreasing=psd(I-dual),unital=unital==I,quantumChannel=cp and tp),
                choiOutputPartialTrace=wire(marginal),dualIdentity=wire(dual),unitalImage=wire(unital),
                bellOutput=wire(J/2),bellOutputIsDensity=cp and J.trace()==2,referenceMarginal=wire(marginal/2))


def damping(t):
    c=S.cancel((1-t*t)/(1+t*t));s=S.cancel(2*t/(1+t*t))
    return c,s,S.diag(1,c),S.Matrix([[0,s],[0,0]])


def environment(rho,t):
    c,s,K0,K1=damping(t);V=S.Matrix([[1,0],[0,s],[0,c],[0,0]])
    U=S.Matrix([[1,0,0,0],[0,c,s,0],[0,-s,c,0],[0,0,0,1]])
    joint=clean(V*rho*V.H);system=partial(joint,0);env=partial(joint,1)
    mixed=[clean((K0+K1)/S.sqrt(2)),clean((K0-K1)/S.sqrt(2))]
    phi=summed([(1,K0),(1,K1)]);mix=summed([(1,k) for k in mixed]);ref=choi(phi)/2
    assert V.H*V==I and U.H*U==S.eye(4)
    assert clean(system-phi(rho))==S.zeros(2) and choi(phi)==choi(mix)
    return dict(schemaVersion=1,tensorOrder='system-environment',isometry=wire(V),unitaryCompletion=wire(U),
                isometryResidual=wire(V.H*V-I),unitaryResidual=wire(U.H*U-S.eye(4)),joint=wire(joint),
                system=wire(system),environment=wire(env),reducedEqualsKraus=True,mixedKraus=[wire(a) for a in mixed],
                mixedChannelEqualsOriginal=True,referenceOutput=wire(ref),referenceMarginal=wire(partial(ref,0)))


def instrument(family,t):
    c,s,K0,K1=damping(t)
    if family=='projective-z':return [[(1,P0)],[(1,P1)]]
    if family=='reset-z':return [[(1,P0)],[(1,S.Matrix([[0,1],[0,0]]))]]
    if family=='coarse-pauli-record':return [[(S.Rational(1,2),I),(S.Rational(1,4),Z)],[(S.Rational(1,4),X)]]
    if family=='damping-record':return [[(1,K0)],[(1,K1)]]
    return [[(1,clean((K0+K1)/S.sqrt(2)))],[(1,clean((K0-K1)/S.sqrt(2)))]]


def instrument_report(outcomes,rho,correct):
    outputs=[summed(terms)(rho) for terms in outcomes];effects=[clean(sum((w*k.H*k for w,k in terms),S.zeros(2))) for terms in outcomes]
    assert sum(effects,S.zeros(2))==I
    assert all(psd(choi(summed(terms))) for terms in outcomes)
    branches=[]
    for i,(out,E) in enumerate(zip(outputs,effects)):
        p=S.simplify(out.trace());assert p.is_nonnegative
        branches.append(dict(label=i,probability=scalar(p),effect=wire(E),unnormalized=wire(out),conditional=None if p==0 else wire(clean(out/p))))
    cq=clean(S.kronecker_product(P0,outputs[0])+S.kronecker_product(P1,outputs[1]))
    forgotten=clean(outputs[0]+outputs[1]);feedback=clean(outputs[0]+X*outputs[1]*X) if correct else forgotten
    assert S.simplify(cq.trace()-1)==0 and clean(partial(cq,1)-forgotten)==S.zeros(2)
    return dict(schemaVersion=1,recordOrder='record-system',correctZ=correct,branches=branches,
                classicalQuantum=wire(cq),forgotten=wire(forgotten),feedforward=wire(feedback))


def deferred(rho):
    a=S.Matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    b=S.Matrix([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    U=b*a;coherent=clean(U*S.kronecker_product(rho,P0)*U.H)
    dephased=clean(sum((S.kronecker_product(I,p)*coherent*S.kronecker_product(I,p) for p in [P0,P1]),S.zeros(4)))
    record_system=F*dephased*F;classical=instrument_report(instrument('projective-z',S.Rational(1,2)),rho,True)
    assert clean(partial(coherent,0)-P0)==S.zeros(2) and clean(partial(record_system,1)-P0)==S.zeros(2)
    return dict(schemaVersion=1,tensorOrder='system-record',coherent=wire(coherent),dephased=wire(dephased),
                recordSystem=wire(record_system),classicalCorrected=classical['feedforward'],
                system=wire(partial(coherent,0)),record=wire(partial(coherent,1)))


def generate():
    states=[]
    vectors=[('zero',[1,0]),('one',[0,1]),('plus',[1,1]),('minus',[1,-1]),('plus-i',[1,S.I]),
             ('complex',[1+2*S.I,3-S.I]),('T-plus',[1,(1+S.I)/S.sqrt(2)]),('irrational-norm',[1+S.sqrt(2),S.I])]
    for name,values in vectors:
        v=S.Matrix(values);rho=clean(v*v.H/(v.H*v)[0]);assert psd(rho) and rho.trace()==1;states.append((name,rho))
    states.append(('mixed',S.Matrix([[S.Rational(2,3),S.I/6],[-S.I/6,S.Rational(1,3)]])))
    params=[S.Rational(0),S.Rational(1,3),S.Rational(1,2),S.Rational(2,3),S.Rational(1)]
    maps=[]
    maps.append(('identity',term_wire([(1,I)]),summed([(1,I)])))
    for p in params:
        c,s,K0,K1=damping(p)
        families=[('damping',[(1,K0),(1,K1)]),('dephasing',[(1-p,I),(p,Z)]),
                  ('depolarizing',[(1-3*p/4,I),(p/4,X),(p/4,Y),(p/4,Z)])]
        for name,terms in families:maps.append((f'{name}-{p}',term_wire(terms),summed(terms)))
    for name,terms in [('postselect-zero',[(1,P0)]),('complex-unitary',[(1,clean(H*T))]),
                       ('trace-growing',[(2,I)]),('negative-map',[(-1,I)]),('signed-cancellation',[(2,I),(-1,I)])]:
        maps.append((name,term_wire(terms),summed(terms)))
    maps.append(('transpose',dict(kind='transpose'),lambda rho:rho.T))
    # Full-rank rational CP map without assuming TP or a special Pauli form.
    B=S.Matrix([[1,S.I,0],[2,0,1],[0,1,1-S.I],[1,2,3]])/5
    cpmap=from_choi(clean(B*B.H));maps.append(('generic-choi-cp',image_wire(cpmap),cpmap))
    bad=S.eye(4);bad[0,1]=S.I;bad[0,2]=S.I
    badmap=from_choi(bad);maps.append(('non-hermiticity-preserving',image_wire(badmap),badmap))
    cases=[]
    for name,description,phi in maps:
        base=map_base(phi)
        if name in ['identity','complex-unitary','signed-cancellation'] or name.startswith(('damping-','dephasing-','depolarizing-')):
            assert base['properties']['quantumChannel'],name
        for state,rho in states:
            output=phi(rho);report=dict(base,input=wire(rho),output=wire(output),outputTrace=scalar(output.trace()),outputIsDensity=psd(output) and output.trace()==1)
            cases.append(dict(name=f'{name}/{state}',map=description,state=wire(rho),expected=report))
    environments=[dict(name=f'{t}/{name}',parameter=str(t),state=wire(rho),expected=environment(rho,t)) for t in params for name,rho in states]
    instruments=[]
    for family in ['projective-z','reset-z','damping-record','damping-mixed-record','coarse-pauli-record']:
        for t in params if family.startswith('damping') else [S.Rational(1,2)]:
            for name,rho in states:
                outcomes=instrument(family,t)
                for correct in [False,True]:
                    instruments.append(dict(name=f'{family}/{t}/{name}/{correct}',family=family,parameter=str(t),state=wire(rho),correctZ=correct,
                                            outcomes=[term_wire(terms)['terms'] for terms in outcomes],expected=instrument_report(outcomes,rho,correct)))
    deferrals=[dict(name=name,state=wire(rho),expected=deferred(rho)) for name,rho in states]
    # Explicit manuscript witnesses, independent of reference serialization.
    anti=S.Matrix([0,1,-1,0])/S.sqrt(2)
    assert (anti.H*choi(lambda a:a.T)*anti)[0]==-1
    assert psd(choi(summed([(1,I)]))) and not psd(choi(lambda a:a.T))
    return dict(schemaVersion=1,scalarConvention='[a,b] means a+b*sqrt(2)',channelCases=cases,environmentCases=environments,instrumentCases=instruments,deferredCases=deferrals)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('scripts/fixtures/qubit-channel-independent.json'));args=parser.parse_args()
    data=generate();args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:len(v) for k,v in data.items() if k.endswith('Cases')}))
