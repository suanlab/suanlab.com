"""Independent explicit virtual-path sums and dense physical observables."""
import itertools
import json
import random
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def preparation(sites,bonds=None):
 bonds=bonds if bonds is not None else [2]*(sites-1)
 p=dict(sites=sites,tensorDenominator=1,**{f'bond{i}':bonds[i] if i<len(bonds) else 0 for i in range(3)},gaugeBond=0 if sites>1 else -1,gaugeRow=0,gaugeColumn=1 if sites>1 and bonds[0]>1 else 0,gaugeScaleNumerator=2 if sites>1 else 1,gaugeScaleDenominator=1,gaugeShearReal=1 if sites>1 and bonds[0]>1 else 0,gaugeShearImag=0,gaugeShearDenominator=1)
 p.update({f't{i}{part}{l}{s}{r}':0 for i in range(4) for l in range(4) for s in range(2) for r in range(4) for part in ['Real','Imag']})
 return p

def fraction(x):
 x=S.simplify(x);assert x.is_Rational,x;return str(x)
def cv(x):return dict(real=fraction(S.re(x)),imaginary=fraction(S.im(x)))
def matrix(a):return [[cv(a[i,j]) for j in range(a.cols)] for i in range(a.rows)]
def norm(a):return S.expand((a.conjugate().T*a)[0])
def frob(a):return S.expand(S.trace(a.conjugate().T*a))
def normalized(a,N):
 def coordinate(v):return '0' if v==0 else 'decimal:'+mp.nstr(mp.mpf(str(S.N(v,255)))/mp.sqrt(mp.mpf(str(S.N(N,255)))),245)
 return [dict(real=coordinate(S.re(v)),imaginary=coordinate(S.im(v))) for v in a]

def amplitudes(tensors,dimensions,start,end):
 # Sum every virtual assignment explicitly, rather than using transfer recurrences.
 length=end-start
 result=S.zeros(2**length,dimensions[end]) if start==0 else S.zeros(dimensions[start],2**length)
 for bits in itertools.product(range(2),repeat=length):
  physical=sum(s*2**(length-i-1) for i,s in enumerate(bits))
  for left in range(dimensions[start]):
   for right in range(dimensions[end]):
    value=0
    for middle in itertools.product(*(range(dimensions[i]) for i in range(start+1,end))):
     path=(left,*middle,right);term=S.Integer(1)
     for k,s in enumerate(bits):term*=tensors[start+k][s][path[k],path[k+1]]
     value+=term
    if start==0:result[physical,right]=S.expand(value)
    else:result[left,physical]=S.expand(value)
 return result

pauli=[S.Matrix([[0,1],[1,0]]),S.Matrix([[0,-S.I],[S.I,0]]),S.diag(1,-1)]
def evaluate(tensors,dimensions):
 n=len(tensors);d=2**n
 prefixes=[S.Matrix([[1]])]+[amplitudes(tensors,dimensions,0,k) for k in range(1,n+1)]
 suffixes=[amplitudes(tensors,dimensions,k,n) if k<n else S.Matrix([[1]]) for k in range(n+1)]
 # k=0 amplitudes has physical-first convention; transpose to virtual-first.
 suffixes[0]=prefixes[-1].T
 z=prefixes[-1];N=norm(z)
 left=[S.expand_complex(L.conjugate().T*L).applyfunc(S.expand) for L in prefixes]
 right=[(B.conjugate()*B.T).applyfunc(S.expand) for B in suffixes]
 closures=[S.expand(sum(E[i,j]*R[i,j] for i in range(E.rows) for j in range(E.cols))) for E,R in zip(left,right)]
 assert all(S.simplify(v-N)==0 for v in closures)
 physical=None
 if N!=0:
  rho=(z*z.conjugate().T/N).applyfunc(S.expand);locals=[];blochs=[]
  for site in range(n):
   mask=2**(n-site-1);local=S.zeros(2)
   for a in range(d):
    for b in range(d):
     if a&~mask==b&~mask:local[int(bool(a&mask)),int(bool(b&mask))]+=rho[a,b]
   local=local.applyfunc(S.expand);bloch=[S.simplify(S.trace(local*P)) for P in pauli];blochs.append(bloch)
   locals.append(dict(site=site,density=matrix(local),bloch=[fraction(v) for v in bloch],purity=fraction(S.trace(local*local))))
  correlations=[]
  for i in range(n):
   for j in range(i+1,n):
    values=[]
    for a,Pa in enumerate(pauli):
     row=[]
     for b,Pb in enumerate(pauli):
      O=S.kronecker_product(*[Pa if k==i else Pb if k==j else S.eye(2) for k in range(n)]);v=S.expand((z.conjugate().T*O*z)[0]/N)
      row.append(dict(value=fraction(v),connected=fraction(v-blochs[i][a]*blochs[j][b])))
     values.append(row)
    correlations.append(dict(sites=[i,j],values=values))
  physical=dict(normalizedState=normalized(z,N),localStates=locals,correlations=correlations)
 cuts=[]
 for k in range(1,n):
  C=S.Matrix(2**k,2**(n-k),list(z));rank=C.rank()
  cuts.append(dict(cut=k,bondDimension=dimensions[k],leftDimension=C.rows,rightDimension=C.cols,coefficientRank=rank,schmidtRank=rank if N!=0 else None,coefficientMatrix=matrix(C)))
 canon=[]
 for site,T in enumerate(tensors):
  L=sum((a.conjugate().T*a for a in T),S.zeros(dimensions[site+1]));R=sum((a*a.conjugate().T for a in T),S.zeros(dimensions[site]));L=L.applyfunc(S.expand);R=R.applyfunc(S.expand)
  l=frob(L-S.eye(L.rows));r=frob(R-S.eye(R.rows))
  canon.append(dict(site=site,leftGram=matrix(L),rightGram=matrix(R),leftIsometry=bool(l==0),rightIsometry=bool(r==0),leftIdentityResidualSquared=fraction(l),rightIdentityResidualSquared=fraction(r)))
 return dict(tensors=[[matrix(a) for a in T] for T in tensors],state=[cv(v) for v in z],normSquared=fraction(N),tensorParameterSquaredNorm=fraction(sum(frob(a) for T in tensors for a in T)),prefixes=[matrix(a) for a in prefixes],leftEnvironments=[matrix(a) for a in left],rightEnvironments=[matrix(a) for a in right],cutClosureNorms=[fraction(v) for v in closures],cuts=cuts,canonicality=canon,physical=physical,stateDefined=bool(N!=0)),dict(z=z,N=N,left=left,right=right)

def reference(p):
 n=p['sites'];dims=[1]+[p[f'bond{i}'] for i in range(n-1)]+[1]
 tensors=[[S.Matrix(dims[i],dims[i+1],lambda l,r:S.Rational(p[f't{i}Real{l}{s}{r}'],p['tensorDenominator'])+S.I*S.Rational(p[f't{i}Imag{l}{s}{r}'],p['tensorDenominator'])) for s in range(2)] for i in range(n)]
 original,internal=evaluate(tensors,dims);changed=[[a.copy() for a in T] for T in tensors];gauge=None;cov=None
 if p['gaugeBond']>=0:
  k=p['gaugeBond'];size=dims[k+1];row,col=p['gaugeRow'],p['gaugeColumn'];D=S.eye(size);D[row,row]=S.Rational(p['gaugeScaleNumerator'],p['gaugeScaleDenominator']);shear=S.eye(size)
  if row!=col:shear[row,col]=S.Rational(p['gaugeShearReal'],p['gaugeShearDenominator'])+S.I*S.Rational(p['gaugeShearImag'],p['gaugeShearDenominator'])
  G=D*shear;inverse=G.inv();changed[k]=[(a*G).applyfunc(S.expand) for a in tensors[k]];changed[k+1]=[(inverse*a).applyfunc(S.expand) for a in tensors[k+1]]
  gauge=dict(bond=k,order='G=diagonal-scale-times-shear',matrix=matrix(G),inverse=matrix(inverse),inverseResidualSquared=fraction(frob(G*inverse-S.eye(size))));cov=(k+1,G,inverse)
 gauged,after=evaluate(changed,dims);assert internal['z']==after['z'];assert internal['N']==after['N']
 residual=dict(stateDifferenceSquared=fraction(norm(internal['z']-after['z'])),normAccount=fraction(after['N']-internal['N']),leftEnvironmentCovariance=None,rightEnvironmentCovariance=None)
 if cov:
  k,G,inv=cov;residual['leftEnvironmentCovariance']=fraction(frob(after['left'][k]-G.conjugate().T*internal['left'][k]*G));residual['rightEnvironmentCovariance']=fraction(frob(after['right'][k]-inv.conjugate()*internal['right'][k]*inv.T))
 return dict(sites=n,physicalDimensionPerSite=2,dimension=2**n,bondDimensions=dims[1:-1],virtualDimensionsIncludingBoundaries=dims,original=original,gauged=gauged,gauge=gauge,validity=dict(structurallyValid=True,physicalStateDefined=original['stateDefined'],reason='nonzero-contracted-vector' if original['stateDefined'] else 'zero-contracted-vector-cannot-normalize'),cost=dict(activeComplexTensorEntries=sum(2*dims[i]*dims[i+1] for i in range(n)),paddedComplexTensorEntries=128,denseComplexStateAmplitudes=2**n,prefixDenseMultiplyAccumulateSlots=sum(2**(i+1)*dims[i]*dims[i+1] for i in range(n)),normTransferDenseSummandSlots=sum(2*dims[i]**2*dims[i+1]**2 for i in range(n)),countsAreAlgorithmicSlotsNotRuntimeOrCompressionGuarantees=True),exactResiduals=residual)

cases=[]
def save(label,p):
 cases.append(dict(label=label,input=dict(p),expected=reference(p)));print('Verified '+label,flush=True)
rng=random.Random(20261109)
profiles=[(1,[])]+[(n,[d]*(n-1)) for n in range(2,5) for d in range(1,5)]+[(3,[1,4]),(3,[4,1]),(3,[2,3]),(4,[1,2,3]),(4,[4,1,2]),(4,[3,4,1]),(4,[4,2,3])]
for index,(n,bonds) in enumerate(profiles):
 p=preparation(n,bonds);p['tensorDenominator']=3
 dims=[1]+bonds+[1]
 for i in range(n):
  for l in range(dims[i]):
   for s in range(2):
    for r in range(dims[i+1]):
     for part in ['Real','Imag']:p[f't{i}{part}{l}{s}{r}']=rng.randint(-2,2)
 if n>1:
  k=rng.randrange(n-1);p.update(gaugeBond=k,gaugeRow=0,gaugeColumn=bonds[k]-1,gaugeScaleNumerator=-3,gaugeScaleDenominator=2,gaugeShearReal=2 if bonds[k]>1 else 0,gaugeShearImag=-1 if bonds[k]>1 else 0,gaugeShearDenominator=3)
 save(f'general complex MPS profile{index} sites{n} bonds{bonds}',p)
# A GHZ has nontrivial canonical blocks and gauge-dependent tensor lengths.
p=preparation(3,[2,2]);p.update(t0Real000=1,t0Real011=1,t1Real000=1,t1Real111=1,t2Real000=1,t2Real110=1,gaugeShearDenominator=2);save('GHZ3 scale shear parameter norm changes',p)
p=preparation(2,[2]);p.update(t0Real000=1,t0Real011=1,t1Real000=1,t1Imag110=1,gaugeShearImag=2);save('complex Bell phase fixes environment orientation',p)
p=preparation(2,[1]);p.update(t0Real000=1,t1Real000=1,t1Imag010=1);save('product right Y positive',p)
p=preparation(2,[2]);p.update(t0Real000=1,t0Real001=1,t1Real000=1,t1Real100=-1);save('nonzero tensors cancel to global zero',p)
p=preparation(4,[4,4,4]);save('all tensors zero no normalized state',p)
p=preparation(3,[4,4]);p.update(t0Real000=1,t1Real000=1,t2Real000=1);save('redundant bond4 product rank1',p)
p=preparation(4,[2,2,2]);p.update(tensorDenominator=1000000,t0Real000=1,t0Real011=1,t1Real000=1,t1Real111=1,t2Real000=1,t2Real111=1,t3Real000=1,t3Real110=1);save('smallest amplitude scale irrational normalization',p)
p=preparation(1,[]);p.update(t0Real000=3,t0Imag010=4);save('one site no virtual gauge',p)
p=preparation(4,[4,4,4]);dims=[1,4,4,4,1]
for i in range(4):
 for l in range(dims[i]):
  for s in range(2):
   for r in range(dims[i+1]):p[f't{i}Real{l}{s}{r}']=1000000;p[f't{i}Imag{l}{s}{r}']=-1000000
p.update(gaugeScaleNumerator=1000000,gaugeScaleDenominator=1,gaugeShearReal=1000000,gaugeShearImag=-1000000)
save('maximum integer tensors and gauge',p)
output=Path('scripts/fixtures/mps-reference.json');output.write_text(json.dumps(dict(method='independent SymPy explicit virtual-path sums,direct prefix/suffix Gram,dense partial traces and Pauli operators;mpmath250dps normalization',seed=20261109,cases=cases),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),output=str(output))),flush=True)
