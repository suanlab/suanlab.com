"""Independent finite U(1) orbit oracle.

Equivalence is solved on ALL graph edges,without the selected tree.
Anchored forests are selected by exhaustive subsets,not greedy union-find.
Normal coordinates use unique simple-path enumeration,not breadth-first phases.
Run from the repository root.
"""
import itertools
import json
import random
from fractions import Fraction as F
from pathlib import Path
Z=(F(0),F(0));O=(F(1),F(0))
def add(z,w):return(z[0]+w[0],z[1]+w[1])
def mul(z,w):return(z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0])
def conj(z):return(z[0],-z[1])
def norm(z):return z[0]**2+z[1]**2
def divide(z,w):return tuple(x/norm(w) for x in mul(z,conj(w)))
def wire(z):return dict(real=str(z[0]),imaginary=str(z[1]))
def phase(p):
 t=F(p['cayley']);return(-F(1),F(0)) if p['minusOne'] else((1-t*t)/(1+t*t),2*t/(1+t*t))
def chart(z):
 assert norm(z)==1
 return dict(cayley='0',minusOne=True) if z==(-F(1),F(0)) else dict(cayley=str(z[1]/(1+z[0])),minusOne=False)
def matter(rows):return[(F(z['real']),F(z['imaginary'])) for z in rows]
def groups(n,edges,selected):
 unseen=set(range(n));out=[]
 while unseen:
  todo=[min(unseen)];unseen.remove(todo[0]);sites=[]
  while todo:
   i=todo.pop();sites.append(i)
   for k in selected:
    e=edges[k]
    for a,b in [(e['from'],e['to']),(e['to'],e['from'])]:
     if a==i and b in unseen:unseen.remove(b);todo.append(b)
  out.append(sorted(sites))
 return out

def reference(p):
 n=p['sites'];edges=p['edges'];m=len(edges);forest=p['forestEdges'];roots=p['roots'];anchors=p['anchors'];components=groups(n,edges,range(m));C=len(components);A=len(anchors);unanch=sum(not set(g)&set(anchors) for g in components)
 # A maximal allowed subset has N-A-C_unanchored edges. The first valid
 # combination in lexicographic order is the earliest-edge greedy basis.
 target=n-A-unanch
 for positions in itertools.combinations(range(len(forest)),target):
  anchored=[forest[i] for i in positions]
  if all(len(set(g)&set(anchors))<=1 for g in groups(n,edges,anchored)):break
 else:raise AssertionError('No anchor-compatible forest')
 subgroups=groups(n,edges,anchored)
 aroots=sorted(next(iter(set(g)&set(anchors))) if set(g)&set(anchors) else next(r for r in roots if r in g) for g in subgroups)
 if not anchors:aroots=roots
 def paths(start,end,selected,seen):
  if start==end:return []
  for k in selected:
   e=edges[k]
   if start!=e['from'] and start!=e['to']:continue
   forward=start==e['from'];next_site=e['to'] if forward else e['from']
   if next_site in seen:continue
   tail=paths(next_site,end,selected,seen|{next_site})
   if tail is not None:return[dict(edge=k,forward=forward)]+tail
  return None
 def product(links,steps):
  z=O
  for s in steps:z=mul(z,links[s['edge']] if s['forward'] else conj(links[s['edge']]))
  return z
 def normal(links,psi,selected,origins):
  h=[];tree=[]
  for i in range(n):
   for root in origins:
    steps=paths(root,i,selected,{root})
    if steps is not None:break
   else:raise AssertionError('Missing path')
   h.append(product(links,steps));parent=None
   if steps:
    e=edges[steps[-1]['edge']];parent=e['from'] if steps[-1]['forward'] else e['to']
   tree.append(dict(site=i,root=root,parent=parent,viaEdge=steps[-1]['edge'] if steps else None,forward=steps[-1]['forward'] if steps else None,steps=steps))
  fixed=[mul(mul(h[e['from']],links[k]),conj(h[e['to']])) for k,e in enumerate(edges)]
  fixed_psi=[mul(h[i],psi[i]) for i in range(n)];coordinates=[]
  for k,e in enumerate(edges):
   if k in selected:continue
   left,right=tree[e['from']],tree[e['to']]
   reverse=[dict(edge=s['edge'],forward=not s['forward']) for s in reversed(right['steps'])]
   steps=left['steps']+[dict(edge=k,forward=True)]+reverse
   z=product(links,steps);assert z==fixed[k]
   coordinates.append(dict(edge=k,start=left['root'],end=right['root'],kind='closed-cycle' if left['root']==right['root'] else 'anchored-path',steps=steps,product=wire(z),canonicalLink=wire(fixed[k]),productResidual=wire(Z)))
  return dict(phases=list(map(wire,h)),links=list(map(wire,fixed)),matter=list(map(wire,fixed_psi)),nodeTree=tree,coordinates=coordinates,treeResiduals=[dict(edge=k,value=wire(Z)) for k in selected],anchorResiduals=[dict(site=i,value=wire(add(h[i],(-F(1),F(0))))) for i in anchors])
 U,V=list(map(phase,p['leftLinks'])),list(map(phase,p['rightLinks']));L,R=matter(p['leftMatter']),matter(p['rightMatter'])
 # Independent direct constraint solver. No selected forest or normal form used.
 connection=joint=anchored_connection=anchored_joint=True
 for group in components:
  potential={group[0]:O};queue=[group[0]];consistent=True
  while queue:
   i=queue.pop(0)
   for k,e in enumerate(edges):
    for a,b,ratio in [(e['from'],e['to'],mul(U[k],conj(V[k]))),(e['to'],e['from'],mul(conj(U[k]),V[k]))]:
     if a!=i:continue
     required=mul(potential[i],ratio)
     if b in potential:consistent &= potential[b]==required
     else:potential[b]=required;queue.append(b)
  anchor_values=[conj(potential[i]) for i in group if i in anchors]
  matter_values=[];supports=True
  for i in group:
   if norm(L[i]):matter_values.append(divide(R[i],mul(potential[i],L[i])))
   elif norm(R[i]):supports=False
  def compatible(values):return all(norm(z)==1 for z in values) and all(z==values[0] for z in values)
  connection &= consistent;joint &= consistent and supports and compatible(matter_values)
  anchored_connection &= consistent and compatible(anchor_values)
  anchored_joint &= consistent and supports and compatible(anchor_values+matter_values)
 side=[]
 for links,psi in [(U,L),(V,R)]:
  full=normal(links,psi,forest,roots);anch=normal(links,psi,anchored,aroots)
  values=matter(full['matter']);gram=[[wire(mul(conj(z),w)) if any(i in g and j in g for g in components) else None for j,w in enumerate(values)] for i,z in enumerate(values)]
  side.append(dict(full=full,anchored=anch,treeGram=gram))
 left_zero=[g for g in components if all(not norm(L[i]) for i in g)];right_zero=[g for g in components if all(not norm(R[i]) for i in g)];left_free=sum(not set(g)&set(anchors) for g in left_zero);right_free=sum(not set(g)&set(anchors) for g in right_zero)
 counts=dict(fullGaugeGroupDimension=n,anchoredGaugeGroupDimension=n-A,connectionStabilizerDimension=C,anchoredConnectionStabilizerDimension=unanch,leftJointStabilizerDimension=len(left_zero),rightJointStabilizerDimension=len(right_zero),leftAnchoredJointStabilizerDimension=left_free,rightAnchoredJointStabilizerDimension=right_free,fullConnectionOrbitDimension=n-C,anchoredConnectionOrbitDimension=n-A-unanch,leftJointOrbitDimension=n-len(left_zero),rightJointOrbitDimension=n-len(right_zero),leftAnchoredJointOrbitDimension=n-A-left_free,rightAnchoredJointOrbitDimension=n-A-right_free,freeConnectionInvariantCount=m-n+C,anchoredConnectionInvariantCount=m-n+A+unanch)
 return dict(componentCount=C,cycleRank=m-n+C,anchoredForest=dict(edges=anchored,roots=aroots,removedFromFullForest=[k for k in forest if k not in anchored]),counts=counts,left=side[0],right=side[1],comparison=dict(full=dict(connectionsEquivalent=connection,jointEquivalent=joint),anchored=dict(connectionsEquivalent=anchored_connection,jointEquivalent=anchored_joint)))

examples=json.loads(Path('scripts/fixtures/gauge-orbits-preparations.json').read_text());cases=[dict(label=x['id'],input=x['input']) for x in examples];rng=random.Random(20261110)
for k in range(60):
 n=3+k%6;edges=[dict(from_=rng.randrange(n),to=rng.randrange(n)) for _ in range(k%15)];edges=[{('from' if key=='from_' else key):v for key,v in e.items()} for e in edges];selected=[]
 for i,e in enumerate(edges):
  before=groups(n,edges,selected)
  if not any(e['from'] in g and e['to'] in g for g in before):selected.append(i)
 rng.shuffle(selected);components=groups(n,edges,selected);roots=[rng.choice(g) for g in components]
 link=[dict(cayley=str(F(rng.randint(-3,3),rng.randint(1,4))),minusOne=False) for _ in edges];psi=[(F(rng.randint(-2,2),rng.randint(1,3)),F(rng.randint(-2,2),rng.randint(1,3))) for _ in range(n)];g=[phase(dict(cayley=str(F(rng.randint(-2,2),3)),minusOne=False)) for _ in range(n)]
 if k%5==0:
  new_links=[chart(mul(mul(g[e['from']],phase(link[i])),conj(g[e['to']]))) for i,e in enumerate(edges)];new_psi=[mul(g[i],psi[i]) for i in range(n)]
 elif k%5==1:
  new_links=[dict(cayley=str(F(rng.randint(-3,3),rng.randint(1,4))),minusOne=False) for _ in edges];new_psi=list(reversed(psi))
 elif k%5==2:new_links=link;new_psi=psi
 elif k%5==3:
  new_links=link;new_psi=list(psi)
  for group in components:
   for i in group:new_psi[i]=mul(g[group[0]],psi[i])
 else:
  new_links=link;new_psi=list(psi);new_psi[0]=add(psi[0],O)
 anchors=rng.sample(range(n),k%(n+1))
 p=dict(sites=n,edges=edges,leftLinks=link,rightLinks=new_links,leftMatter=list(map(wire,psi)),rightMatter=list(map(wire,new_psi)),forestEdges=selected,roots=roots,anchors=anchors)
 cases.append(dict(label=f'graph-family-{k}',input=p))
for x in cases:x['expected']=reference(x['input'])
Path('scripts/fixtures/gauge-orbits-independent.json').write_text(json.dumps(dict(seed=20261110,method='All-edge phase-constraint solving; exhaustive anchored-forest subsets; unique simple-path products; Python Fraction',cases=cases),indent=2)+'\n')
print(f'{len(cases)} independent orbit cases written')
