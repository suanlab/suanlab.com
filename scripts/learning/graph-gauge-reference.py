"""Independent exact oracle: quadratic-form polarization and polynomial differences.

No TypeScript implementation is imported. Python Fraction uses reduced rationals.
Run from repository root: python3 scripts/learning/graph-gauge-reference.py
"""
import json
import random
from fractions import Fraction as F
from pathlib import Path

Z = (F(0), F(0))
O = (F(1), F(0))
I = (F(0), F(1))
def plus(z, w): return (z[0]+w[0], z[1]+w[1])
def negative(z): return (-z[0], -z[1])
def minus(z, w): return plus(z, negative(w))
def times(z, w): return (z[0]*w[0]-z[1]*w[1], z[0]*w[1]+z[1]*w[0])
def conj(z): return (z[0], -z[1])
def norm(z): return z[0]**2+z[1]**2
def wire(z): return dict(real=str(z[0]), imaginary=str(z[1]))
def phase(p):
    t = F(p['cayley'])
    return (-F(1), F(0)) if p['minusOne'] else ((1-t*t)/(1+t*t), 2*t/(1+t*t))
def product(links, steps):
    z = O
    for step in steps:
        u = links[step['edge']]
        z = times(z, u if step['forward'] else conj(u))
    return z

def reference(p):
    psi = [(F(z['real']), F(z['imaginary'])) for z in p['matter']]
    frames = list(map(phase, p['frames']))
    links = list(map(phase, p['edges']))
    transformed_psi = [times(z,g) for z,g in zip(psi,frames)]
    transformed_links = [times(links[k],times(frames[e['from']],conj(frames[e['to']]))) for k,e in enumerate(p['edges'])]
    n = len(psi)
    a,b,kappa = map(F,[p['quadratic'],p['quartic'],p['loopCoupling']])
    def hopping(z, connection):
        return sum((F(e['weight'])*norm(minus(z[e['from']],times(connection[k],z[e['to']]))) for k,e in enumerate(p['edges'])),F(0))
    def quadratic(z, connection): return hopping(z,connection)+a*sum(map(norm,z))
    def quartic(z): return b*sum(norm(x)**2 for x in z)/2
    def loops(connection): return sum((kappa*(1-product(connection,w['steps'])[0]) for w in p['walks']),F(0))
    def energy(z,connection): return quadratic(z,connection)+quartic(z)+loops(connection)
    def matrix(connection):
        # Recover every Hermitian entry by probing the scalar quadratic form.
        # This is independent of a covariant-incidence matrix construction.
        basis = [[O if i==j else Z for i in range(n)] for j in range(n)]
        diagonal = [quadratic(x,connection) for x in basis]
        h = [[Z for _ in range(n)] for _ in range(n)]
        for i in range(n):
            h[i][i] = (diagonal[i],F(0))
            for j in range(i+1,n):
                real_probe = [plus(x,y) for x,y in zip(basis[i],basis[j])]
                imag_probe = [plus(x,times(I,y)) for x,y in zip(basis[i],basis[j])]
                real = (quadratic(real_probe,connection)-diagonal[i]-diagonal[j])/2
                imag = -(quadratic(imag_probe,connection)-diagonal[i]-diagonal[j])/2
                h[i][j],h[j][i] = (real,imag),(real,-imag)
        return h
    def gradient(z,connection):
        # Energy has degree <=4. Central differences of its cubic derivative
        # have exactly the known 2*b*coordinate*h^2 error, removed here.
        h = F(2,7)
        values=[]
        for j in range(n):
            derivatives=[]
            for axis in range(2):
                step=(h,F(0)) if axis==0 else (F(0),h)
                right,left=list(z),list(z)
                right[j],left[j]=plus(z[j],step),minus(z[j],step)
                derivative=(energy(right,connection)-energy(left,connection))/(2*h)-2*b*z[j][axis]*h*h
                derivatives.append(derivative/2)
            values.append(wire(tuple(derivatives)))
        return values
    H,T=matrix(links),matrix(transformed_links)
    edge_rows=[]
    for k,e in enumerate(p['edges']):
        i,j=e['from'],e['to']
        d=minus(psi[i],times(links[k],psi[j]))
        td=minus(transformed_psi[i],times(transformed_links[k],transformed_psi[j]))
        fd=minus(transformed_psi[i],times(links[k],transformed_psi[j]))
        edge_rows.append(dict(link=wire(links[k]),transformedLink=wire(transformed_links[k]),difference=wire(d),transformedDifference=wire(td),energy=str(F(e['weight'])*norm(d)),transformedEnergy=str(F(e['weight'])*norm(td)),fieldOnlyEnergy=str(F(e['weight'])*norm(fd)),dressed=wire(times(conj(psi[i]),times(links[k],psi[j])))))
    walk_rows=[]
    for w in p['walks']:
        z=product(links,w['steps'])
        reverse=[dict(edge=s['edge'],forward=not s['forward']) for s in reversed(w['steps'])]
        walk_rows.append(dict(holonomy=wire(z),transformedHolonomy=wire(product(transformed_links,w['steps'])),reverseHolonomy=wire(product(links,reverse)),penalty=str(kappa*(1-z[0]))))
    unseen=set(range(n));components=0
    while unseen:
        components+=1; frontier=[unseen.pop()]
        while frontier:
            node=frontier.pop()
            for e in p['edges']:
                for u,v in [(e['from'],e['to']),(e['to'],e['from'])]:
                    if u==node and v in unseen: unseen.remove(v);frontier.append(v)
    return dict(componentCount=components,cycleRank=len(links)-n+components,
      energy=dict(hopping=str(hopping(psi,links)),transformedHopping=str(hopping(transformed_psi,transformed_links)),fieldOnlyHopping=str(hopping(transformed_psi,links)),onsiteQuadratic=str(a*sum(map(norm,psi))),onsiteQuartic=str(quartic(psi)),loops=str(loops(links)),transformedLoops=str(loops(transformed_links)),total=str(energy(psi,links)),transformedTotal=str(energy(transformed_psi,transformed_links)),fieldOnlyTotal=str(energy(transformed_psi,links))),
      operator=dict(original=[[wire(z) for z in row] for row in H],transformed=[[wire(z) for z in row] for row in T]),
      gradients=gradient(psi,links),transformedGradients=gradient(transformed_psi,transformed_links),edges=edge_rows,walks=walk_rows)

preparations=json.loads(Path('scripts/fixtures/graph-gauge-preparations.json').read_text())
cases=[dict(label=x['id'],input=x['input']) for x in preparations]
rng=random.Random(20261109)
# Include trees, rings, disconnected graphs, self-loops and parallel edges.
for k in range(36):
    n=3+k%5
    edges=[]
    for _ in range(k%13):
        edges.append(dict(from_=rng.randrange(n),to=rng.randrange(n),cayley=str(F(rng.randint(-3,3),rng.randint(1,5))),minusOne=False,weight=str(F(rng.randint(1,5),rng.randint(1,5)))))
    edges=[{('from' if key=='from_' else key):value for key,value in e.items()} for e in edges]
    walks=[]
    if edges:
        walks=[dict(start=edges[0]['from'],steps=[dict(edge=0,forward=True),dict(edge=0,forward=False)])]
    p=dict(matter=[dict(real=str(F(rng.randint(-3,3),rng.randint(1,5))),imaginary=str(F(rng.randint(-3,3),rng.randint(1,5)))) for _ in range(n)],frames=[dict(cayley=str(F(rng.randint(-4,4),rng.randint(1,5))),minusOne=False) for _ in range(n)],edges=edges,walks=walks,quadratic=str(F(k%5-2,3)),quartic=str(F(k%4,5)),loopCoupling='2/3')
    cases.append(dict(label=f'graph-family-{k}',input=p))
for case in cases: case['expected']=reference(case['input'])
path=Path('scripts/fixtures/graph-gauge-independent.json')
path.write_text(json.dumps(dict(method='Python Fraction; scalar quadratic polarization and exact quartic central differences',seed=20261109,cases=cases),indent=2)+'\n')
print(f'{len(cases)} independent cases written to {path}')
