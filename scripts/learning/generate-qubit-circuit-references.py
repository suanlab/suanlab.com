"""Independent SymPy dense-matrix circuit and channel hand checks.

No product TypeScript imports. The product kernel propagates amplitudes;
this reference constructs full matrices before multiplying states.
"""
import argparse
import json
from pathlib import Path
import sympy as S

I = S.eye(2)
X = S.Matrix([[0, 1], [1, 0]])
Y = S.Matrix([[0, -S.I], [S.I, 0]])
Z = S.diag(1, -1)
H = S.Matrix([[1, 1], [1, -1]]) / S.sqrt(2)
T = S.diag(1, (1 + S.I) / S.sqrt(2))
GATES = dict(I=I, X=X, Y=Y, Z=Z, H=H, S=S.diag(1, S.I),
             Sdg=S.diag(1, -S.I), T=T, Tdg=T.H)


def clean(a):
    return a.applyfunc(S.simplify)


def scalar(z):
    def pair(v):
        v = S.expand(S.simplify(v))
        b = v.coeff(S.sqrt(2))
        a = S.simplify(v - b * S.sqrt(2))
        assert a.is_Rational and b.is_Rational, v
        return [str(a), str(b)]
    return dict(real=pair(S.re(z)), imaginary=pair(S.im(z)))


def wire(a):
    return [[scalar(a[r, c]) for c in range(a.cols)] for r in range(a.rows)]


def embedded(n, op):
    name, target = op['gate'], op['target']
    if name in GATES:
        return S.kronecker_product(*[GATES[name] if k == target else I for k in range(n)])
    result = S.zeros(2 ** n)
    for col in range(2 ** n):
        bits = list(map(int, format(col, f'0{n}b')))
        sign = 1
        if name == 'SWAP':
            bits[target], bits[op['other']] = bits[op['other']], bits[target]
        elif bits[op['control']]:
            if name == 'CX':
                bits[target] = 1 - bits[target]
            elif name == 'CZ':
                sign = (-1) ** bits[target]
        row = int(''.join(map(str, bits)), 2)
        result[row, col] = sign
    return result


def circuit_case(name, n, ops, raw):
    state = S.Matrix([a + S.I * b for a, b in raw])
    norm = S.simplify((state.H * state)[0])
    U = S.eye(2 ** n)
    prefixes = [list(state)]
    for op in ops:
        A = embedded(n, op)
        assert clean(A.H * A) == S.eye(2 ** n)
        U = clean(A * U)
        prefixes.append(list(clean(U * state)))
    assert S.simplify((U * state).norm() ** 2 - norm) == 0
    return dict(name=name, sites=n, gates=ops, input=raw,
                prefixes=[[scalar(v) for v in prefix] for prefix in prefixes],
                normSquared=str(norm))


def channel_checks():
    # Reference/output order: J = sum |i><j|_R tensor Phi(|i><j|)_B.
    K0, K1 = S.diag(1, S.Rational(3, 5)), S.Matrix([[0, S.Rational(4, 5)], [0, 0]])
    K = [K0, K1]
    phi = lambda rho: clean(sum((a * rho * a.H for a in K), S.zeros(2)))
    choi = S.zeros(4)
    for i in range(2):
        for j in range(2):
            E = S.zeros(2)
            E[i, j] = 1
            choi += S.kronecker_product(E, phi(E))
    partial_output = S.Matrix(2, 2, lambda i, j: sum(choi[2*i+b, 2*j+b] for b in range(2)))
    assert partial_output == I
    assert sorted(choi.eigenvals().items()) == [(S.Integer(0), 2), (S.Rational(16, 25), 1), (S.Rational(34, 25), 1)]
    assert sum((a.H*a for a in K), S.zeros(2)) == I
    assert phi(I) != I
    swap = S.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    anti = S.Matrix([0,1,-1,0]) / S.sqrt(2)
    assert (anti.H * swap * anti)[0] == -1
    # Stinespring output order B then environment: V|1> = 3/5|10>+4/5|01>.
    V = S.Matrix([[1,0],[0,S.Rational(4,5)],[0,S.Rational(3,5)],[0,0]])
    assert V.H*V == I
    rho = S.Matrix([[S.Rational(1,2), -S.I/2],[S.I/2,S.Rational(1,2)]])
    joint = V*rho*V.H
    reduced = S.Matrix(2,2,lambda a,b:sum(joint[2*a+e,2*b+e] for e in range(2)))
    assert clean(reduced-phi(rho)) == S.zeros(2)
    # Mix Kraus labels with a Hadamard. Same forgotten-record channel,
    # different branch probabilities and conditional states for |0>.
    mixed = [(K0+K1)/S.sqrt(2),(K0-K1)/S.sqrt(2)]
    assert clean(sum((a*rho*a.H for a in mixed),S.zeros(2))-phi(rho)) == S.zeros(2)
    plus = S.Matrix([1,1])/S.sqrt(2)
    P0, P1 = S.diag(1,0), S.diag(0,1)
    measured = sum((a*plus*plus.H*a.H for a in [P0,P1]),S.zeros(2))
    corrected = P0*plus*plus.H*P0 + X*P1*plus*plus.H*P1*X
    assert measured == I/2 and corrected == P0
    assert phi(I) == S.diag(S.Rational(41,25),S.Rational(9,25))
    assert reduced == S.Matrix([[S.Rational(41,50),-3*S.I/10],[3*S.I/10,S.Rational(9,50)]])
    assert reduced.det() == S.Rational(36,625)
    assert phi(X*P0*X) == S.diag(S.Rational(16,25),S.Rational(9,25))
    assert X*phi(P0)*X == P1
    # Fixed unitary completion acts on B,E in that order.
    U = S.Matrix([[1,0,0,0],[0,S.Rational(3,5),S.Rational(4,5),0],
                  [0,-S.Rational(4,5),S.Rational(3,5),0],[0,0,0,1]])
    assert U.H*U == S.eye(4) and U[:,0] == V[:,0] and U[:,2] == V[:,1]
    zero = S.Matrix([1,0])
    record_probs = [[S.simplify((a*zero).norm()**2) for a in operators] for operators in [K,mixed]]
    assert record_probs == [[1,0],[S.Rational(1,2),S.Rational(1,2)]]
    # Same effects, different forgotten outputs.
    reset = [P0,S.Matrix([[0,1],[0,0]])]
    assert all(a.H*a == b.H*b for a,b in zip([P0,P1],reset))
    assert sum((a*plus*plus.H*a.H for a in reset),S.zeros(2)) == P0
    cq = S.kronecker_product(P0,P0/2)+S.kronecker_product(P1,P1/2)
    assert cq.trace() == 1
    # Coherent deferral: input order B,C. Final C dephasing equals measured
    # feedforward, but a coherent intermediate record retains X expectation1.
    entangle = embedded(2,dict(gate='CX',control=0,target=1))
    correct = embedded(2,dict(gate='CX',control=1,target=0))
    coherent = clean(correct*entangle*S.kronecker_product(plus,zero))
    assert coherent == S.kronecker_product(zero,plus)
    dephased = sum((S.kronecker_product(I,p)*coherent*coherent.H*S.kronecker_product(I,p) for p in [P0,P1]),S.zeros(4))
    assert dephased == S.kronecker_product(P0,I/2)
    assert (coherent.H*S.kronecker_product(I,X)*coherent)[0] == 1
    assert (dephased*S.kronecker_product(I,X)).trace() == 0
    # Preserve the reference marginal for damping on the second Bell wire.
    bell = S.Matrix([1,0,0,1])/S.sqrt(2)
    bell_out = clean(sum((S.kronecker_product(I,a)*bell*bell.H*S.kronecker_product(I,a.H) for a in K),S.zeros(4)))
    assert bell_out == choi/2
    assert S.Matrix(2,2,lambda a,b:sum(bell_out[2*a+v,2*b+v] for v in range(2))) == I/2
    return dict(amplitudeDampingKraus=[wire(a) for a in K], choi=wire(choi),
                choiEigenvalues=['0','0','16/25','34/25'], partialOutput=wire(partial_output),
                unitalImage=wire(phi(I)), transposeChoi=wire(swap),
                transposeBellNegativeExpectation='-1/2', stinespring=wire(V),
                reducedExample=wire(reduced), forgottenRecord=wire(measured),
                feedforwardOutput=wire(corrected), mixedKraus=[wire(a) for a in mixed],
                unitaryCompletion=wire(U), recordProbabilities=[[str(v) for v in row] for row in record_probs],
                recordSystemCQ=wire(cq), coherentCorrection=wire(coherent),
                finalDephasedSystemRecord=wire(dephased), bellOutput=wire(bell_out))


def generate():
    cases = []
    for n in range(1,4):
        raw = [[(j % 3)-1, (j % 2)] for j in range(2 ** n)]
        for name in GATES:
            for target in range(n):
                cases.append(circuit_case(f'{name}-{n}-{target}', n, [dict(gate=name,target=target)], raw))
        for name in ['CX','CZ','SWAP']:
            for a in range(n):
                for b in range(n):
                    if a != b:
                        op = dict(gate=name,target=b,**({'other':a} if name=='SWAP' else {'control':a}))
                        cases.append(circuit_case(f'{name}-{n}-{a}-{b}',n,[op],raw))
    demos = [
        ('HZH',1,[dict(gate=g,target=0) for g in ['H','Z','H']],[[1,0],[0,0]]),
        ('ZHH',1,[dict(gate=g,target=0) for g in ['Z','H','H']],[[1,0],[0,0]]),
        ('HTH',1,[dict(gate=g,target=0) for g in ['H','T','H']],[[1,0],[0,0]]),
        ('bell',2,[dict(gate='H',target=0),dict(gate='CX',target=1,control=0)],[[1,0],[0,0],[0,0],[0,0]]),
        ('three-wire',3,[dict(gate='H',target=2),dict(gate='T',target=2),dict(gate='CX',target=0,control=2),dict(gate='Y',target=1),dict(gate='SWAP',target=0,other=1),dict(gate='CZ',target=2,control=0)],[[1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])]
    cases += [circuit_case(*d) for d in demos]
    # Every matrix column independently exercises wire orientation and phases.
    for n in range(1,4):
        ops = [dict(gate=name,target=target) for name in GATES for target in range(n)]
        ops += [dict(gate=name,target=b,**({'other':a} if name=='SWAP' else {'control':a}))
                for name in ['CX','CZ','SWAP'] for a in range(n) for b in range(n) if a != b]
        for index,op in enumerate(ops):
            for basis in range(2**n):
                raw = [[int(j==basis),0] for j in range(2**n)]
                cases.append(circuit_case(f'basis-{n}-{index}-{basis}',n,[op],raw))
    # Hand-derived interference probabilities, not only dense-engine agreement.
    state = clean(H*T*H*S.Matrix([1,0]))
    assert S.simplify(S.Abs(state[1])**2-(2-S.sqrt(2))/4) == 0
    assert clean(H*Z*H) == X and clean(H*H*Z) == Z
    return dict(schemaVersion=1, scalarConvention='[a,b] means a+b*sqrt(2)',
                basisOrder='site zero is most significant; gates chronological',
                circuitCases=cases, channelHandChecks=channel_checks())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('scripts/fixtures/qubit-circuit-independent.json'))
    args=parser.parse_args()
    data=generate()
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(f"Verified {len(data['circuitCases'])} dense circuit references and channel/environment/instrument hand checks")
