"""Independent canonical-graph reference: SymPy linear systems/Poisson matrices,
Fraction complex products and explicit product-rule potential derivatives.
The anchored forest is chosen by subset enumeration,not the producer's greedy scan.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import json,random,copy
import sympy as S
Z=(F(0),F(0));O=(F(1),F(0))
def add(a,b):return(a[0]+b[0],a[1]+b[1])
def mul(a,b):return(a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def conj(a):return(a[0],-a[1])
def prod(xs):
 z=O
 for x in xs:z=mul(z,x)
 return z
def wire(z):return dict(real=str(z[0]),imaginary=str(z[1]))
def phase(t='0',minus=False):return dict(cayley=str(t),minusOne=minus)
def complex_phase(p):
 if p['minusOne']:return(-F(1),F(0))
 t=F(p['cayley']);return((1-t*t)/(1+t*t),2*t/(1+t*t))
def components(n,edges,subset=None):
 adj=[[]for _ in range(n)]
 for e in range(len(edges))if subset is None else subset:
  a,b=edges[e]['from'],edges[e]['to'];adj[a].append(b);adj[b].append(a)
 seen=set();groups=[]
 for i in range(n):
  if i in seen:continue
  group=[];stack=[i];seen.add(i)
  while stack:
   x=stack.pop();group.append(x)
   for y in adj[x]:
    if y not in seen:seen.add(y);stack.append(y)
  groups.append(sorted(group))
 return groups
def spanning(n,edges):
 out=[]
 for e,edge in enumerate(edges):
  groups=components(n,edges,out)
  if not any(edge['from']in g and edge['to']in g for g in groups):out.append(e)
 return out
def path(n,edges,forest,start,end):
 stack=[(start,[],{start})]
 while stack:
  at,steps,seen=stack.pop()
  if at==end:return steps
  for e in forest:
   edge=edges[e]
   for a,b,forward in [(edge['from'],edge['to'],True),(edge['to'],edge['from'],False)]:
    if a==at and b not in seen:stack.append((b,steps+[dict(edge=e,forward=forward)],seen|{b}))
 raise ValueError('No forest path')
def reference(p):
 n=p['sites'];edges=p['edges'];m=len(edges);A=set(p['anchors']);I=[i for i in range(n)if i not in A];groups=components(n,edges);C0=sum(not A.intersection(g)for g in groups);r=n-len(A)-C0
 forest=next(list(sub)for sub in combinations(p['forestEdges'],r)if all(len(A.intersection(g))<=1 for g in components(n,edges,sub)))
 subgroups=components(n,edges,forest);roots=[]
 for group in subgroups:
  roots.append(next(iter(A.intersection(group)))if A.intersection(group)else next(x for x in p['roots']if x in group))
 rootof={i:root for g,root in zip(subgroups,roots)for i in g};chords=[e for e in range(m)if e not in forest];k=len(chords)
 P=[];coordinate_steps=[]
 for e in chords:
  edge=edges[e];first=path(n,edges,forest,rootof[edge['from']],edge['from']);last=path(n,edges,forest,rootof[edge['to']],edge['to']);steps=first+[dict(edge=e,forward=True)]+[dict(edge=s['edge'],forward=not s['forward'])for s in reversed(last)];row=[0]*m
  for s in steps:row[s['edge']]+=1 if s['forward']else -1
  P.append(row);coordinate_steps.append(steps)
 B=S.Matrix(n,m,lambda i,e:int(edges[e]['from']==i)-int(edges[e]['to']==i));BI=B.extract(I,list(range(m)));rho=S.Matrix([S.Rational(p['charges'][i])for i in I])if I else S.zeros(0,1);E=S.Matrix([S.Rational(x)for x in p['electric']])if m else S.zeros(0,1);W=S.diag(*[S.Rational(x)for x in p['electricWeights']])if m else S.zeros(0,0);PM=S.Matrix(P)if k else S.zeros(0,m);K=PM.T
 rank=BI.rank();compatible=BI.row_join(rho).rank()==rank;admissible=BI*E==rho
 affine=None
 if compatible:
  affine=S.zeros(m,1)
  if forest:
   solution,parameters=BI.extract(list(range(len(I))),forest).gauss_jordan_solve(rho)
   assert parameters.rows==0
   for e,x in zip(forest,solution):affine[e]=x
  assert BI*affine==rho
 U=[complex_phase(x)for x in p['links']];g=[complex_phase(x)for x in p['frames']];Up=[mul(mul(g[e['from']],U[i]),conj(g[e['to']]))for i,e in enumerate(edges)]
 def product(links,steps):return prod([links[s['edge']]if s['forward']else conj(links[s['edge']])for s in steps])
 terms=[];force=[F(0)]*m
 for term in p['magneticTerms']:
  factors=[U[s['edge']]if s['forward']else conj(U[s['edge']])for s in term['steps']];z=prod(factors);gradient=[]
  for e in range(m):
   derivative=Z
   for at,step in enumerate(term['steps']):
    if step['edge']!=e:continue
    dz=mul((F(0),F(1 if step['forward']else -1)),factors[at]);derivative=add(derivative,prod(factors[:at]+[dz]+factors[at+1:]))
   gradient.append(F(term['weight'])*derivative[0]);force[e]+=gradient[-1]
  terms.append(dict(product=wire(z),transformedProduct=wire(product(Up,term['steps'])),potential=str(F(term['weight'])*(1-z[0])),force=list(map(str,gradient))))
 FM=S.Matrix([S.Rational(x.numerator,x.denominator)for x in force])if m else S.zeros(0,1);selector=S.Matrix(k,m,lambda i,e:int(chords[i]==e));Jq=PM.row_join(S.zeros(k,m));Jp=S.zeros(k,m).row_join(selector);JG=S.zeros(len(I),m).row_join(BI);Omega=S.zeros(m,m).row_join(S.eye(m)).col_join((-S.eye(m)).row_join(S.zeros(m,m)))
 vals=lambda x:[str(v)for v in x];mat=lambda x:[[int(v)for v in x.row(i)]for i in range(x.rows)];rmat=lambda x:[[str(v)for v in x.row(i)]for i in range(x.rows)];mom=selector*E;lift=affine+K*mom if affine is not None else None;pot=sum((F(x['potential'])for x in terms),F(0));he=(E.T*W*E)[0]/2
 return dict(forestEdges=forest,nonForestEdges=chords,incidence=mat(B),interiorSites=I,interiorIncidence=mat(BI),rank=rank,chargesCompatible=compatible,inputAdmissible=admissible,physicalPhaseSpaceDimension=2*(m-rank)if compatible else None,pathMatrix=P,electricKernel=mat(K),coordinateProducts=[wire(product(U,s))for s in coordinate_steps],transformedCoordinateProducts=[wire(product(Up,s))for s in coordinate_steps],terms=terms,divergence=vals(B*E),gaussResiduals=[None if i in A else str((B*E)[i]-S.Rational(p['charges'][i]))for i in range(n)],affineElectric=vals(affine)if affine is not None else None,liftElectric=vals(lift)if lift is not None else None,electricMetric=rmat(K.T*W*K),electricLinear=vals(K.T*W*affine)if affine is not None else None,electricConstant=str((affine.T*W*affine)[0]/2)if affine is not None else None,coordinateMomentumBrackets=mat(Jq*Omega*Jp.T),pathConstraintBrackets=mat(Jq*Omega*JG.T),gaussBrackets=mat(JG*Omega*JG.T),angleRates=vals(W*E),electricRates=vals(FM),gaussRates=vals(BI*FM),coordinateRates=vals(PM*W*E),momentumRates=vals(selector*FM),energy=dict(electric=str(he),potential=str(pot),total=str(he+S.Rational(pot.numerator,pot.denominator))),liftHamiltonian=str((lift.T*W*lift)[0]/2+S.Rational(pot.numerator,pot.denominator))if lift is not None else None)
def base(n=3,edges=None):
 if edges is None:edges=[dict(from_=0,to=1),dict(from_=1,to=2),dict(from_=2,to=0)]
 edges=[dict(**{'from':e.get('from',e.get('from_'))},to=e['to'])for e in edges];forest=spanning(n,edges);groups=components(n,edges)
 return dict(sites=n,edges=edges,links=[phase()for _ in edges],frames=[phase()for _ in range(n)],electric=['0']*len(edges),charges=['0']*n,forestEdges=forest,roots=[g[0]for g in groups],anchors=[],electricWeights=['1']*len(edges),magneticTerms=[])
def term(start,steps,weight='1'):return dict(start=start,steps=[dict(edge=e,forward=f)for e,f in steps],weight=weight)
def preparations():
 rows=[]
 def put(id,ko,en,p):rows.append(dict(id=id,label=dict(ko=ko,en=en),input=copy.deepcopy(p)))
 p=base();p.update(links=[phase(1),phase(),phase()],frames=[phase(),phase(1),phase(0,True)],electric=['1']*3,magneticTerms=[term(0,[(0,True),(1,True),(2,True)])]);put('physical-triangle','Gauss를 만족하는 삼각형','A physical triangle',p)
 p['electric']=['1','0','0'];put('constraint-violating-triangle','Gauss를 어긴 현재 입력','Current Gauss violation',p)
 p['charges']=['1','-1','0'];put('charged-triangle','중성인 고정 전하 쌍','A neutral fixed-charge pair',p)
 p['charges']=['1','0','0'];p['electric']=['0']*3;put('incompatible-total-charge','전하 조건 자체가 불가능','An incompatible charge level',p)
 p=base(edges=[dict(from_=0,to=1),dict(from_=1,to=2)]);put('neutral-open-chain','중성 열린 사슬의 영 자유도','A neutral open chain with no pair',p)
 p.update(charges=['1','0','-1'],electric=['1','1']);put('charged-open-chain','사슬에서 전하가 정하는 전기장','Charge-determined electric field on a chain',p)
 p.update(anchors=[0,2],charges=[None,'0',None],electric=['2','2'],links=[phase(1),phase()],magneticTerms=[term(0,[(0,True),(1,True)],'2')]);put('anchored-open-chain','앵커 사이의 열린 정준 쌍','An open canonical pair between anchors',p)
 p.update(charges=[None,'1',None],electric=['1','2']);put('boundary-charge-flux','내부 전하와 경계 선속','Interior charge and boundary flux',p)
 p=base();p.update(anchors=[0],charges=[None,'1','0'],electric=['0','1','1']);put('one-anchor-nonneutral-interior','한 앵커가 받는 내부 순전하','Net interior charge with one anchor',p)
 p=base();p.update(anchors=[0,1,2],charges=[None]*3,links=[phase(1),phase(),phase(0,True)],electric=['1','2','-1'],magneticTerms=[term(0,[(0,True)]),term(2,[(2,True)],'2')]);put('every-site-anchored','모든 노드가 외부 기준','Every site anchored',p)
 p=base(6,[dict(from_=a,to=b)for a,b in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]]);p.update(charges=['1','-1','0','0','1','-1'],electric=['1','0','0','0','1','0']);put('disconnected-neutral-components','성분마다 중성인 전하','Componentwise neutral charges',p)
 p['charges']=['1','0','0','-1','0','0'];put('global-neutrality-insufficient','전체 중성만으로 부족','Global neutrality is insufficient',p)
 p=base(edges=[dict(from_=0,to=1),dict(from_=0,to=1),dict(from_=1,to=1)]);p.update(links=[phase(),phase(1),phase(0,True)],electric=['1','-1','2'],magneticTerms=[term(0,[(1,True),(0,False)]),term(1,[(2,True)],'3')]);put('parallel-links-self-loop','평행 간선과 자기 루프','Parallel links and a self-loop',p)
 p=base(edges=[]);p.update(anchors=[1],charges=['0',None,'0'],roots=[2,0,1]);put('isolated-sites','고립 노드와 빈 위상공간 좌표','Isolated sites with no canonical pairs',p)
 p=base();p.update(links=[phase('1/2'),phase('-2/3'),phase(0,True)],frames=[phase('2/7'),phase('-3/5'),phase(1)],electric=['2/3']*3,electricWeights=['1/2','2/3','3/4'],magneticTerms=[term(0,[(0,True),(1,True),(2,True)],'3/5')]);put('rational-canonical-data','정확한 분수 계수','Exact rational coefficients',p)
 p=copy.deepcopy(rows[2]['input']);p.update(forestEdges=[1,2],roots=[1]);put('alternative-tree-root','같은 Hamiltonian의 다른 숲','Another forest for the same Hamiltonian',p)
 p=copy.deepcopy(rows[0]['input']);p['magneticTerms']=[term(0,[(0,True),(0,False),(2,False),(2,True)],'3')];put('backtracking-potential','왕복 항의 정확한 영 힘','Exactly zero force for a backtrack',p)
 return rows
def with_terms(p,rng):
 forest=p['forestEdges'];n=p['sites'];edges=p['edges'];groups=components(n,edges);terms=[]
 for e,edge in enumerate(edges):
  if e in forest:continue
  root=next(g[0]for g in groups if edge['from']in g);left=path(n,edges,forest,root,edge['from']);right=path(n,edges,forest,root,edge['to']);steps=left+[dict(edge=e,forward=True)]+[dict(edge=s['edge'],forward=not s['forward'])for s in reversed(right)];terms.append(dict(start=root,steps=steps,weight=str(F(rng.randrange(0,5),rng.randrange(1,6)))))
 for group in groups:
  anchors=[a for a in p['anchors']if a in group]
  for a,b in zip(anchors,anchors[1:]):terms.insert(0,dict(start=a,steps=path(n,edges,forest,a,b),weight='2/3'))
 p['magneticTerms']=terms[:8]
 return p

def main():
 rng=random.Random(20261110);presets=preparations();cases=[dict(label=x['id'],input=x['input'])for x in presets]
 for j in range(48):
  n=rng.randrange(3,9);m=rng.randrange(0,15);p=base(n,[dict(from_=rng.randrange(n),to=rng.randrange(n))for _ in range(m)]);rng.shuffle(p['forestEdges']);p['roots']=[rng.choice(g)for g in components(n,p['edges'])];p['anchors']=rng.sample(range(n),rng.randrange(n+1));p['links']=[phase(str(F(rng.randrange(-3,4),rng.randrange(1,6))),False)for _ in range(m)];p['frames']=[phase()if i in p['anchors']else phase(str(F(rng.randrange(-3,4),rng.randrange(1,6))))for i in range(n)];p['electric']=[str(F(rng.randrange(-3,4),rng.randrange(1,5)))for _ in range(m)];p['electricWeights']=[str(F(rng.randrange(1,5),rng.randrange(1,6)))for _ in range(m)];divergence=[sum((int(e['from']==i)-int(e['to']==i))*F(p['electric'][k])for k,e in enumerate(p['edges']))for i in range(n)];p['charges']=[None if i in p['anchors']else str(divergence[i])for i in range(n)]
  if j%3==1:
   groups=[g for g in components(n,p['edges'])if not set(p['anchors']).intersection(g)]
   if groups:p['charges'][groups[0][0]]=str(F(p['charges'][groups[0][0]])+1)
  elif j%3==2 and m:p['electric'][0]=str(F(p['electric'][0])+1)
  with_terms(p,rng);cases.append(dict(label=f'random-{j}',input=p))
 budgets=[]
 for count in [0,1,3,12,0]:
  p=base(12,[dict(from_=i,to=(i+1)%12)for i in range(12)]+[dict(from_=(5*i+1)%12,to=(7*i+3)%12)for i in range(20)]);p['anchors']=list(range(count));p['charges']=[None if i in p['anchors']else'0'for i in range(12)];p['electric']=['999999999999999971/999999999999999989']*12+['0']*20;p['links']=[phase(f'{999999999999999971-i}/{999999999999999989-i}')for i in range(32)];p['frames']=[phase()if i in p['anchors']else phase(f'{999999999999999941-i}/{999999999999999983-i}')for i in range(12)];p['electricWeights']=[f'{999999999999999937-i}/{999999999999999967-i}'for i in range(32)];with_terms(p,rng)
  label=f'maximum-{count}-anchors'+('-reversed-forest'if len(budgets)==4 else'')
  if len(budgets)==4:
   p['forestEdges']=list(reversed(p['forestEdges']));p['roots']=[11]
   long_walk=[(e,True)for e in range(12)]*2+[(0,True),(0,False)]*4
   p['magneticTerms']=[term(0,long_walk,f'{999999999999999937-i}/{999999999999999967-i}')for i in range(8)]
  budgets.append(dict(label=label,input=p))
 cases+=budgets
 out=Path('scripts/fixtures');(out/'graph-gauss-preparations.json').write_text(json.dumps(presets,ensure_ascii=False,indent=2)+'\n');(out/'graph-gauss-budget-preparations.json').write_text(json.dumps(budgets,indent=2)+'\n');rows=[]
 for case in cases:rows.append(dict(**case,expected=reference(case['input'])))
 (out/'graph-gauss-independent.json').write_text(json.dumps(dict(method='Independent SymPy linear systems and canonical Poisson matrices; Fraction complex products and explicit product-rule differentiation',cases=rows),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(presets=len(presets),referenceCases=len(rows),maximumCases=len(budgets))))
if __name__=='__main__':main()
