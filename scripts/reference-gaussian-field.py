"""Independent exact SymPy bordered-matrix Gaussian reference. No TS numerical code."""
import json,random
from pathlib import Path
import sympy as s
examples=json.loads(Path('scripts/fixtures/gaussian-field-examples.json').read_text())
cases=[{'id':c['id'],'input':c['input']} for c in examples]
rng=random.Random(20261110)
for index in range(24):
 n=rng.randint(3,8);boundary=rng.choice(['open','periodic','dirichlet']);constraint=rng.choice(['none','mean-zero']);mass=rng.choice(['0','1/11','1','16']);stiffness=rng.choice(['0','1/7','1','16'])
 if mass=='0' and (stiffness=='0' or (boundary!='dirichlet' and constraint=='none')):mass='1/11'
 cases.append({'id':f'seeded-{index}','input':dict(sites=n,boundary=boundary,constraint=constraint,massSquared=mass,stiffness=stiffness,temperature=rng.choice(['1/3','1','2','16']),source=[str(s.Rational(rng.randint(-3,3),rng.randint(1,5))) for _ in range(n)],probes=[rng.randrange(n) for _ in range(4)])})
tiny='1/999999999989'
for label,mass,stiffness,T,boundary,constraint,source in [
 ('all-tiny',tiny,tiny,tiny,'open','none',[tiny]*8),('soft-large',tiny,'16','16','periodic','none',['16','-16']*4),('massless-tiny-stiffness','0',tiny,'16','periodic','mean-zero',['16','-16']*4),('uniform-dirichlet-constrained','16','16','1','dirichlet','mean-zero',['16']*8)]:
 cases.append({'id':label,'input':dict(sites=8,boundary=boundary,constraint=constraint,massSquared=mass,stiffness=stiffness,temperature=T,source=source,probes=[0,7,0,7])})
denominators=[];prime=10**12
for _ in range(8):
 prime=int(s.prevprime(prime));denominators.append(prime)
cases.append({'id':'pairwise-prime-source-denominators','input':dict(sites=8,boundary='periodic',constraint='none',massSquared=tiny,stiffness='999999999983/62499999999',temperature='999999999961/62499999999',source=[str(s.Rational((-1)**i*999999999997,d))for i,d in enumerate(denominators)],probes=[0,7,0,7])})
def wire(v):
 if isinstance(v,s.MatrixBase):return [[str(v[i,j]) for j in range(v.cols)]for i in range(v.rows)]
 return str(v)
def reference(x):
 n=x['sites'];mass=s.Rational(x['massSquared']);stiff=s.Rational(x['stiffness']);T=s.Rational(x['temperature']);J=s.Matrix(list(map(s.Rational,x['source'])));L=s.zeros(n)
 # Construct by incidence outer products rather than editing diagonal/off-diagonal entries.
 edges=[(i,i+1) for i in range(n-1)]+([(n-1,0)]if x['boundary']=='periodic' else [])
 for i,j in edges:
  v=s.zeros(n,1);v[i]=1;v[j]=-1;L+=v*v.T
 if x['boundary']=='dirichlet':L[0,0]+=1;L[n-1,n-1]+=1
 K=mass*s.eye(n)+stiff*L;constrained=x['constraint']=='mean-zero';ones=s.ones(n,1);P=s.eye(n)-ones*ones.T/n if constrained else s.eye(n)
 if constrained:
  bordered=K.row_join(ones).col_join(ones.T.row_join(s.zeros(1)));inverse=bordered.inv(method='GE');G=inverse[:n,:n];det=-bordered.det(method='berkowitz')/n
 else:G=K.inv(method='GE');det=K.det(method='berkowitz')
 B=s.eye(n)[:,:n-1]if constrained else s.eye(n)
 if constrained:
  for j in range(n-1):B[n-1,j]=-1
 A=B.T*K*B;lower,diag=A.LDLdecomposition(hermitian=False);C=T*G;mu=G*J;raw=C+mu*mu.T;probes=x['probes']
 def moment(indices):
  if not indices:return s.Integer(1)
  a,*rest=indices
  return mu[a]*moment(rest)+sum(C[a,b]*moment(rest[:j]+rest[j+1:])for j,b in enumerate(rest))
 i,j,k,l=probes;pairs=[(i,j,k,l),(i,k,j,l),(i,l,j,k)];wick=[C[a,b]*C[c,d]for a,b,c,d in pairs];mp=mu[i]*mu[j]*mu[k]*mu[l];r4=moment(probes)
 assert P*K*G==P and G==G.T and all(d>0 for d in diag.diagonal())
 return dict(integrationDimension=B.cols,precision=wire(K),laplacian=wire(L),basis=wire(B),projector=wire(P),reducedPrecision=wire(A),factorization=dict(lower=wire(lower),diagonal=[str(v)for v in diag.diagonal()],reducedDeterminant=str(A.det(method='berkowitz'))),normalization=dict(restrictedPrecisionDeterminant=str(det),euclideanMeasureGramDeterminant=str(n if constrained else 1),formula='Z(0)=(2πT)^(dimension/2)/sqrt(restrictedPrecisionDeterminant)',logZRatio=str((J.T*mu)[0]/(2*T))),mean=[str(v)for v in mu],susceptibility=wire(G),connectedTwo=wire(C),rawTwo=wire(raw),sourceDerivatives=dict(logZGradient=[str(v/T)for v in mu],logZHessian=wire(C/T**2)),fourPoint=dict(indices=probes,meanProduct=str(mp),wickPairProducts=[str(v)for v in wick],centered=str(sum(wick)),raw=str(r4),connectedCumulant='0',incorrectRawPairSubtraction=str(r4-sum(raw[a,b]*raw[c,d]for a,b,c,d in pairs)),incorrectSubtractionDifference=str(-2*mp)),certificates=dict(ambientMeanResidual=[str(v)for v in K*mu-J],constraintMultiplier=str(-sum(K*mu-J)/n)if constrained else None,projectedMeanResidual=[str(v)for v in P*(K*mu-J)],projectedGreenResidual=wire(P*K*G-P),constraintMeanSum=str(sum(mu))if constrained else None,covarianceRowSums=[str(sum(C[i,j]for j in range(n)))for i in range(n)]if constrained else None,precisionPositiveOnIntegrationSpace=True))
for c in cases:c['expected']=reference(c['input'])
Path('scripts/fixtures/gaussian-field-reference.json').write_text(json.dumps({'method':'Independent SymPy Gauss-Jordan bordered inverse, incidence precision, Berkowitz determinant and recursive Gaussian moments','exact':True,'cases':cases},indent=2)+'\n')
print(json.dumps({'cases':len(cases),'examples':len(examples),'seeded':24,'extremes':5,'matrixEntries':sum(c['input']['sites']**2 for c in cases)}))
