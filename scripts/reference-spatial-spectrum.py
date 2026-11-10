"""Independent incidence/bordered inverse and 300-digit direct complex DFT.
High precision point comparisons are cross-checks, not certified intervals.
"""
import json,random
from pathlib import Path
import sympy as s
import mpmath as mp
mp.mp.dps=300
num=lambda q:mp.mpf(str(s.numer(q)))/mp.mpf(str(s.denom(q)))
wire=lambda a:[[str(v)for v in a.row(i)]for i in range(a.rows)]
def reference(x):
 n=x['sites'];a=s.Rational(x['spacing']);mass=s.Rational(x['massSquared']);c=s.Rational(x['speedSquared']);T=s.Rational(x['temperature']);M=a*mass;k=c/a;periodic=x['boundary']=='periodic';pinned=x['boundary']=='dirichlet';constrained=x['constraint']=='mean-zero';L=s.zeros(n)
 edges=[(i,i+1)for i in range(n-1)]+([(n-1,0)]if periodic else[])
 for i,j in edges:
  v=s.zeros(n,1);v[i]=1;v[j]=-1;L+=v*v.T
 if pinned:L[0,0]+=1;L[n-1,n-1]+=1
 K=M*s.eye(n)+k*L;ones=s.ones(n,1);P=s.eye(n)-ones*ones.T/n if constrained else s.eye(n);B=s.eye(n)[:,:n-1]if constrained else s.eye(n)
 if constrained:
  for j in range(n-1):B[n-1,j]=-1
  G=K.row_join(ones).col_join(ones.T.row_join(s.zeros(1))).inv(method='GE')[:n,:n]
 else:G=K.inv(method='GE')
 C=T*G;A=B.T*K*B
 # Recover LDL pivots by consecutive leading principal determinants, not a factor solve.
 dets=[s.Integer(1)]+[A[:j,:j].det(method='berkowitz')for j in range(1,A.rows+1)]
 exact=dict(integrationDimension=B.cols,spacing=str(a),boxLength=str(a*(n+1 if pinned else n)),positions=[str(a*(j+1 if pinned else j if periodic else s.Rational(2*j+1,2)))for j in range(n)],onsiteCoefficient=str(M),edgeCoefficient=str(k),precision=wire(K),laplacian=wire(L),basis=wire(B),projector=wire(P),reducedPrecision=wire(A),factorization=dict(diagonal=[str(dets[i+1]/dets[i])for i in range(A.rows)],reducedDeterminant=str(dets[-1])),covariance=wire(C))
 exact['masslessPeriodicGreen']=[str(T/k*(s.Rational(n*n-1,12*n)-s.Rational(d*(n-d),2*n)))for d in range(n)]if periodic and mass==0 and constrained else None
 curve=[]
 for d in range(n):
  site=(x['origin']+d)%n if periodic else d;steps=d if periodic else site-x['origin'];curve.append(dict(site=site,steps=steps,distance=str(a*steps),covariance=str(C[x['origin'],site])))
 exact['curve']=curve
 exact['certificates']={'projectedGreenResidual':wire(P*K*C-T*P),'projectionResidual':wire(P*C-C),'trace':str(s.trace(C)),'covarianceRowSums':[str(sum(C.row(i)))for i in range(n)],'periodicTranslationInvariant':bool(periodic and all(C[i,j]==C[0,(j-i)%n]for i in range(n)for j in range(n)))}
 assert P*K*C==T*P and P*C==C and C==C.T and all(dets[i+1]/dets[i]>0 for i in range(A.rows))
 intervals=[]
 def put(path,value):intervals.append(dict(path=path,value=mp.nstr(value,285)))
 F=mp.matrix([[mp.exp(-2j*mp.pi*l*j/n)/mp.sqrt(n)for j in range(n)]for l in range(n)]);cm=mp.matrix([[num(C[i,j])for j in range(n)]for i in range(n)]);km=mp.matrix([[num(K[i,j])for j in range(n)]for i in range(n)]);spectral=F*cm*F.H;operator=F*km*F.H
 for name,mat in [('fourierCovariance',spectral),('fourierPrecision',operator)]:
  for i in range(n):
   for j in range(n):
    for component,v in [('real',mp.re(mat[i,j])),('imaginary',mp.im(mat[i,j]))]:put([name,i,j,component],v)
 for index in range(n):
  signed=index if 2*index<=n else index-n
  put(['modes',index,'wavenumber'],2*mp.pi*signed/(n*num(a)));put(['modes',index,'power'],mp.re(spectral[index,index]));put(['modes',index,'continuumNormalizedPower'],num(a)*mp.re(spectral[index,index]))
  if periodic:put(['modes',index,'periodicEnergyEigenvalue'],num(M)+4*num(k)*mp.sin(mp.pi*index/n)**2)
 for j in range(n):put(['selectedReconstruction',j,'real'],num(C[x['origin'],j]));put(['selectedReconstruction',j,'imaginary'],0)
 put(['certificates','traceResidual'],0);put(['certificates','hermitianResidualMaximum'],0)
 infinity={}
 if mass==0:infinity['status']='massless-no-stationary-field'
 elif c==0:
  infinity['status']='independent-infinite-reference'
  for name,v in [('decay',0),('amplitude',num(T/M)),('correlationLength',0),('continuumLength',0)]:put(['infinite',name],v)
  for d in range(n):put(['infinite','curve',d],num(T/M)if d==(0 if periodic else x['origin'])else 0)
 else:
  infinity['status']='massive-correlated';alpha=mp.acosh(1+num(M/k)/2);r=mp.exp(-alpha);amp=num(T)/mp.sqrt(num(M*(M+4*k)))
  for name,v in [('decay',r),('amplitude',amp),('correlationLength',num(a)/alpha),('continuumLength',mp.sqrt(num(c/mass)))]:put(['infinite',name],v)
  for d in range(n):
   put(['infinite','curve',d],amp*r**(d if periodic else abs(d-x['origin'])))
   if periodic:put(['infinite','periodicImage',d],amp*(r**d+r**(n-d))/(1-r**n)-(num(T/(n*M))if constrained else 0))
 infinity['periodicImageApplicable']=bool(periodic and mass>0 and c>0)
 ratios=[]
 for d in range(n-1):
  p,q=curve[d:d+2];v=s.Rational(p['covariance']);w=s.Rational(q['covariance']);status='boundary-nonuniform'if not periodic else'nonpositive'if v<=0 or w<=0 else'equal'if v==w else'increasing'if v<w else'positive-local-ratio-only'
  ratios.append(dict(fromSite=p['site'],toSite=q['site'],status=status))
  if status=='positive-local-ratio-only':put(['localRatios',d,'length'],num(a)/mp.log(num(v/w)))
 if periodic:
  signed=x['selectedMode']if 2*x['selectedMode']<=n else x['selectedMode']-n;wave=2*mp.pi*signed/(n*num(a));continuum=num(mass)+num(c)*wave**2
  for level in range(4):
   sites=n*2**level;spacing=a/2**level;energy=num(spacing*mass)+4*num(c/spacing)*mp.sin(mp.pi*signed/sites)**2;power=0 if constrained and signed==0 else num(T)/energy
   for key,v in [('wavenumber',wave),('operatorEigenvalue',energy/num(spacing)),('continuumOperatorEigenvalue',continuum),('operatorGap',continuum-energy/num(spacing)),('unitaryPower',power),('continuumNormalizedPower',num(spacing)*power)]:put(['refinements',level,key],v)
 return dict(exact=exact,infinity=infinity,localRatios=ratios,intervals=intervals)
def main():
 cases=json.loads(Path('scripts/fixtures/spatial-spectrum-examples.json').read_text());rng=random.Random(20261110)
 for j in range(6):
  n=rng.choice([3,4,5,6,9,12,16]);boundary=rng.choice(['periodic','open','dirichlet']);constraint=rng.choice(['none','mean-zero']);mass=rng.choice(['0','1/7','1','16']);speed=rng.choice(['0','1/11','1','16'])
  if mass=='0'and(speed=='0'or(boundary!='dirichlet'and constraint=='none')):mass='1/7'
  cases.append(dict(id=f'seeded-{j}',input=dict(sites=n,boundary=boundary,constraint=constraint,massSquared=mass,speedSquared=speed,spacing=rng.choice(['1/16','1/2','1','16']),temperature=rng.choice(['1/3','1','16']),origin=rng.randrange(n),selectedMode=rng.randrange(n))))
 for name,patch in [('tiny-coupling-long-distance',dict(sites=16,boundary='dirichlet',massSquared='16',speedSquared='1/999999999989',spacing='16',origin=15,selectedMode=15)),('massless-soft-gradient',dict(sites=16,massSquared='0',speedSquared='1/999999999989',constraint='mean-zero')),('soft-mass-fine-grid',dict(sites=16,massSquared='1/999999999989',spacing='1/16',speedSquared='16'))]:cases.append(dict(id=name,input={**cases[0]['input'],**patch}))
 for c in cases:c['expected']=reference(c['input'])
 Path('scripts/fixtures/spatial-spectrum-reference.json').write_text(json.dumps(dict(method='Independent SymPy incidence/bordered Gauss-Jordan inverse and principal determinants;300-digit mpmath complex DFT/acosh; point comparisons are not certified intervals',workingDecimalDigits=300,comparisonAbsoluteTolerance='1e-250',cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),intervalValues=sum(len(c['expected']['intervals'])for c in cases),matrixEntries=sum(c['input']['sites']**2 for c in cases))))
if __name__=='__main__':main()
