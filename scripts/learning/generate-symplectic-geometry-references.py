"""Independent symbolic generators, pullbacks and path integrals;no product imports."""
from pathlib import Path
import json,random
import sympy as s
rng=random.Random(20261104)
q1,q2,p1,p2,a,b,h,c,t=s.symbols('q1 q2 p1 p2 a b h c t',real=True)
z=s.Matrix([q1,q2,p1,p2]);J=s.Matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
F=h*(q1*q1-q2*q2)/2+c*q1*q2;G=h*p1*p1/2+c*p1*p2
K=s.Matrix([q1,q2,p1+s.diff(F,q1),p2+s.diff(F,q2)]).jacobian(z)
T=s.Matrix([q1+s.diff(G,p1),q2+s.diff(G,p2),p1,p2]).jacobian(z)
D=s.diag(a,b,1/a,1/b);Db=s.diag(a,b,1/b,1/a)
Ms=[D,K*D,T*K*D,T*K*Db]
for M in Ms[:3]:assert s.simplify(M.T*J*M-J)==s.zeros(4)
assert s.simplify(Ms[3].det())==1
assert s.simplify((Ms[3].T*J*Ms[3])[0,2])==a/b
assert s.simplify((Ms[3].T*J*Ms[3])[1,3])==b/a
probes=[([1,0,0,0],[0,0,1,0]),([0,1,0,0],[0,0,0,1]),([1,0,0,0],[0,1,0,0]),([1,1,0,0],[0,0,1,-1]),([1,0,0,1],[0,1,-1,0]),([1,1,1,1],[2,2,2,2])]
bases=[[0,0,0,0],[1,1,1,0],[1,-1,0,1],[-1,s.Rational(1,2),1,-s.Rational(1,2)]]
def text(v):return str(s.factor(v))
def vec(v):return [text(x)for x in v]
def mat(M):return [[text(M[i,j])for j in range(M.cols)]for i in range(M.rows)]
def areas(u,v):return [u[0]*v[2]-u[2]*v[0],u[1]*v[3]-u[3]*v[1]]
# Integrate the symbolic pullback of theta on each edge,not the product code's81sample trapezoids.
def edge_action(start,end,up):
 path=start+t*(end-start);integrand=sum(path[2+i]*s.diff(path[i],t)for i in range(2));return s.integrate(integrand,(t,0,up))
def partial_action(vertices,code):
 edge=min(3,code//20);fraction=s.Rational(code-20*edge,20)
 return sum(edge_action(vertices[j],vertices[j+1],1)for j in range(edge))+edge_action(vertices[edge],vertices[edge+1],fraction)
def vertices(base,u,v):return[base,base+u,base+u+v,base+v,base]
map_params=[[3,2,1,0,0,j,0,k]for j in range(6)for k in [0,10,20,40,60,80]]+[[2,1,1,0,2,0,0,40],[3,2,2,3,-2,3,1,17],[0,3,4,2,-1,4,3,31]]
while len(map_params)<300:map_params.append([rng.randrange(4),rng.randint(1,4),rng.randint(1,4),rng.randint(-4,4),rng.randint(-4,4),rng.randrange(6),rng.randrange(4),rng.randrange(81)])
map_rows=[]
for p in map_params:
 kind,aa,bb,hh,cc,pc,bi,code=p;M=Ms[kind].subs({a:aa,b:bb,h:s.Rational(hh,2),c:s.Rational(cc,2)});u,v=map(s.Matrix,probes[pc]);base=s.Matrix(bases[bi]);Mu=M*u;Mv=M*v;verts=vertices(base,u,v);mapped=[M*x for x in verts];edge=min(3,code//20);fraction=s.Rational(code-edge*20,20);query=verts[edge]+fraction*(verts[edge+1]-verts[edge]);pul=M.T*J*M;push=M*J*M.T;before=areas(u,v);after=areas(Mu,Mv)
 map_rows.append(dict(p=p,matrix=mat(M),inverse=mat(M.inv()),pullback=mat(pul),pushforward=mat(push),determinant=text(M.det()),isSymplectic=pul==J,probeRank=s.Matrix.hstack(u,v).rank(),originalPairAreas=vec(before),transformedPairAreas=vec(after),originalArea=text(sum(before)),transformedArea=text(sum(after)),queryOriginal=vec(query),queryTransformed=vec(M*query),queryOriginalAction=text(partial_action(verts,code)),queryTransformedAction=text(partial_action(mapped,code)),closedOriginalAction=text(partial_action(verts,80)),closedTransformedAction=text(partial_action(mapped,80))))
a,d,b,c=s.symbols('a d b c',real=True);S=(a*q1*q1+2*b*q1*q2+d*q2*q2)/2
field=s.Matrix([s.diff(S,q1)+c*q2,s.diff(S,q2)-c*q1]);embed=s.Matrix([q1,q2,*field]);tangent=embed.jacobian([q1,q2]);pull=tangent.T*J*tangent;curl=s.diff(field[1],q1)-s.diff(field[0],q2);assert s.simplify(pull[0,1]+curl)==0
line_dirs=[[1,0],[0,1],[1,1],[1,-1]]
graph_params=[[2,2,0,0,0,0,2,2,0,0],[-2,-2,0,0,0,0,2,2,0,40],[2,-2,0,0,0,0,2,2,0,40],[0,0,0,2,0,0,2,2,0,0],[0,0,0,0,0,0,2,2,1,40],[2,0,0,0,2,-2,1,3,2,17],[-2,0,0,0,2,-2,1,3,2,17]]
while len(graph_params)<240:
 p=[rng.randint(-4,4)for _ in range(6)]+[rng.randint(1,4),rng.randint(1,4),rng.randrange(4),rng.randrange(81)]
 if len(graph_params)%4==0:p[3]=0
 graph_params.append(p)
graph_rows=[]
for p in graph_params:
 aa,dd,bb,cc,qc1,qc2,wc,hc,line,code=p;subs={a:s.Rational(aa,2),d:s.Rational(dd,2),b:s.Rational(bb,2),c:s.Rational(cc,2)};A=field.jacobian([q1,q2]).subs(subs);Sym=s.hessian(S,[q1,q2]).subs(subs);base=s.Matrix([s.Rational(qc1,2),s.Rational(qc2,2)]);u=s.Matrix([s.Rational(wc,2),0]);v=s.Matrix([0,s.Rational(hc,2)]);qverts=vertices(base,u,v);phases=[s.Matrix([*q,*(A*q)])for q in qverts];edge=min(3,code//20);fraction=s.Rational(code-edge*20,20);q=qverts[edge]+fraction*(qverts[edge+1]-qverts[edge]);act=partial_action(phases,code);startS=(base.T*Sym*base)[0]/2;queryS=(q.T*Sym*q)[0]/2;embedding=s.eye(2).col_join(A);eig=[]
 for value,multiplicity in Sym.eigenvals().items():eig.extend([value]*multiplicity)
 signs=[int(s.sign(x))for x in eig];nPos=signs.count(1);nNeg=signs.count(-1);signature={(2,0):'positive-definite',(0,2):'negative-definite',(1,1):'indefinite',(1,0):'positive-semidefinite',(0,1):'negative-semidefinite',(0,0):'zero'}[(nPos,nNeg)]
 graph_rows.append(dict(p=p,matrix=mat(A),pullback=mat(embedding.T*J*embedding),graphRank=embedding.rank(),momentumRank=A.rank(),momentumDeterminant=text(A.det()),isLagrangian=embedding.T*J*embedding==s.zeros(2),signature=signature,eigenvalues=[str(s.N(x,55))for x in sorted(eig,key=lambda x:float(x))],queryPosition=vec(q),queryMomentum=vec(A*q),queryGradient=vec(Sym*q),querySymmetricAction=text(queryS),queryCumulativeAction=text(act),queryPotentialChange=text(queryS-startS),closedLoopAction=text(partial_action(phases,80)),curl=text(curl.subs(subs)),lineTangent=vec(embedding*s.Matrix(line_dirs[line]))))
out=dict(provenance='SymPy Hessian-generated canonical shears,exact4Dpullbacks/determinants/ranks,symbolic theta edge integrals,andgraph field derivatives/eigensigns. No product imports.',maps=map_rows,graphs=graph_rows)
Path('scripts/fixtures/symplectic-geometry-independent.json').write_text(json.dumps(out,indent=2)+'\n')
print('Generated300exact4Dmap cases and240exactLagrangian graph/path-integral cases.')
