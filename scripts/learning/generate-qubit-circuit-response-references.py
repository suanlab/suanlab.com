"""Exact dense SymPy observables; tensor contraction supplies reduced states.

Imports only an independent Python gate reference, never product TypeScript.
No tolerance-based equality or floating-point reference values are used.
"""
import argparse
import functools
import importlib.util
import json
from pathlib import Path
import sympy as S

spec=importlib.util.spec_from_file_location('circuit_reference',Path(__file__).with_name('generate-qubit-circuit-references.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
scalar=functools.lru_cache(maxsize=32768)(base.scalar)
clean=base.clean;I=base.I;axes=[base.X,base.Y,base.Z]


def wire(a):return [[scalar(a[i,j]) for j in range(a.cols)] for i in range(a.rows)]
def reduce_one(rho,sites,site):
    array=S.MutableDenseNDimArray(list(rho),[2]*(2*sites))
    contraction=[(k,sites+k) for k in range(sites) if k!=site]
    reduced=S.tensorcontraction(array,*contraction) if contraction else array
    return clean(S.Matrix(2,2,lambda i,j:reduced[i,j]))


def pauli(sites,positions):return S.kronecker_product(*[axes[positions[k]] if k in positions else I for k in range(sites)])
def expected(sites,gates,values):
    z0=S.Matrix(values);N=S.simplify((z0.H*z0)[0]);assert N.is_positive
    z=z0;U=S.eye(2**sites);states=[z]
    for op in gates:
        A=base.embedded(sites,op);z=clean(A*z);states.append(z);U=clean(A*U)
    prefixes=[]
    for step,z in enumerate(states):
        rho=clean(z*z.H/N);reductions=[reduce_one(rho,sites,k) for k in range(sites)]
        bloch=[[S.simplify((d*a).trace()) for a in axes] for d in reductions]
        pairs=[]
        for first in range(sites):
            for second in range(first+1,sites):
                corr=S.Matrix(3,3,lambda a,b:S.simplify((rho*pauli(sites,{first:a,second:b})).trace()))
                conn=S.Matrix(3,3,lambda a,b:S.simplify(corr[a,b]-bloch[first][a]*bloch[second][b]))
                pairs.append(dict(first=first,second=second,expectations=wire(corr),connected=wire(conn)))
        norm=S.simplify((z.H*z)[0]);assert S.simplify(norm-N)==0
        assert S.simplify(rho.trace()-1)==0 and clean(rho*rho-rho)==S.zeros(rho.rows)
        prefixes.append(dict(step=step,rawAmplitudes=[scalar(v) for v in z],normSquared=scalar(norm),normResidual=scalar(norm-N),
                             density=wire(rho),probabilities=[scalar(S.simplify(S.conjugate(v)*v/N)) for v in z],purity=scalar(S.simplify((rho*rho).trace())),
                             singleQubits=[dict(site=k,density=wire(d),purity=scalar(S.simplify((d*d).trace())),bloch=[scalar(v) for v in bloch[k]]) for k,d in enumerate(reductions)],pauliPairs=pairs))
    residual=clean(U.H*U-S.eye(U.rows));assert residual==S.zeros(U.rows)
    restored=clean(U.H*states[-1]);assert clean(restored-z0)==S.zeros(z0.rows,1)
    overlap=S.simplify((z0.H*states[-1])[0]/N);inverse_overlap=S.simplify((z0.H*restored)[0]/N)
    return dict(schemaVersion=1,basisOrder='site-zero-most-significant',gateOrder='chronological',amplitudeConvention='raw-with-exact-squared-norm',fidelityConvention='squared-pure-state-overlap',
                sites=sites,gates=gates,pauliOrder=['X','Y','Z'],inputNormSquared=scalar(N),prefixes=prefixes,
                unitary=wire(U),unitarityResidual=wire(residual),unitaryVerified=True,overlapWithInput=scalar(overlap),
                fidelityWithInput=scalar(S.simplify(overlap*S.conjugate(overlap))),sameAmplitudesAsInput=clean(states[-1]-z0)==S.zeros(z0.rows,1),
                samePhysicalStateAsInput=clean(states[-1]*states[-1].H/N-z0*z0.H/N)==S.zeros(z0.rows),
                inverseRawAmplitudes=[scalar(v) for v in restored],inverseResidual=[scalar(v) for v in restored-z0],
                inverseFidelity=scalar(S.simplify(inverse_overlap*S.conjugate(inverse_overlap))),inverseRecovered=True)


def generate():
    cases=[]
    def add(name,n,gates,values=None):
        if values is None:values=[(j%3)-1+S.I*(j%2) for j in range(2**n)]
        cases.append(dict(name=name,sites=n,gates=gates,input=[scalar(v) for v in values],expected=expected(n,gates,values)))
    for n in range(1,4):
        add(f'empty-{n}',n,[])
        for gate in base.GATES:
            for target in range(n):add(f'{gate}-{n}-{target}',n,[dict(gate=gate,target=target)])
        for gate in ['CX','CZ']:
            for control in range(n):
                for target in range(n):
                    if target!=control:add(f'{gate}-{n}-{control}-{target}',n,[dict(gate=gate,control=control,target=target)])
        for a in range(n):
            for b in range(a+1,n):add(f'SWAP-{n}-{a}-{b}',n,[dict(gate='SWAP',target=a,other=b)])
        add(f'global-minus-{n}',n,[dict(gate=g,target=n-1) for g in ['X','Z','X','Z']])
    for name,gates in [('HZH',['H','Z','H']),('ZHH',['Z','H','H']),('HTH',['H','T','H'])]:
        add(name,1,[dict(gate=g,target=0) for g in gates],[1,0])
    for control in [0,1]:
        add(f'bell-control-{control}',2,[dict(gate='H',target=control),dict(gate='CX',control=control,target=1-control)],[1,0,0,0])
    add('GHZ',3,[dict(gate='H',target=0),dict(gate='CX',control=0,target=1),dict(gate='CX',control=1,target=2)],[1,0,0,0,0,0,0,0])
    add('global-T',1,[dict(gate='T',target=0)],[0,1])
    add('positive-Y',1,[],[1,S.I])
    add('irrational-norm-one',1,[dict(gate=g,target=0) for g in ['H','T','Sdg']],[1+S.sqrt(2),S.I])
    add('irrational-norm-three',3,[dict(gate='CX',control=2,target=0),dict(gate='H',target=1),dict(gate='Tdg',target=0)],
        [1+S.sqrt(2),S.I,0,1,0,0,S.sqrt(2)*S.I,2])
    sequence=[dict(gate='H',target=0),dict(gate='T',target=2),dict(gate='CX',control=2,target=0),dict(gate='Y',target=1),
              dict(gate='SWAP',target=0,other=1),dict(gate='CZ',control=0,target=2),dict(gate='Sdg',target=2),dict(gate='Tdg',target=0)]
    add('sixteen-gates',3,sequence+sequence)
    lookup={c['name']:c for c in cases}
    assert lookup['HTH']['expected']['prefixes'][-1]['probabilities'][1]==scalar((2-S.sqrt(2))/4)
    assert lookup['positive-Y']['expected']['prefixes'][0]['singleQubits'][0]['bloch'][1]==scalar(1)
    for n in range(1,4):
        report=lookup[f'global-minus-{n}']['expected']
        assert report['samePhysicalStateAsInput'] and not report['sameAmplitudesAsInput'] and report['fidelityWithInput']==scalar(1)
    bell=lookup['bell-control-0']['expected']['prefixes'][-1]
    assert bell['singleQubits'][0]['purity']==scalar(S.Rational(1,2))
    assert bell['pauliPairs'][0]['expectations']==wire(S.diag(1,-1,1))
    return dict(schemaVersion=1,cases=cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('scripts/fixtures/qubit-circuit-response-independent.json'));args=parser.parse_args()
    data=generate();args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(cases=len(data['cases']),prefixes=sum(len(c['expected']['prefixes']) for c in data['cases']))))
