"""Independent exact SymPy and400dps Cholesky/Hermitian Rayleigh-Ritz references."""
import json
import random
from pathlib import Path
import sympy as s
import mpmath as mp
mp.mp.dps=400
CASES=[]

def prep(diagonal,basis,entries=None,coefficients=None,denominator=1,shear=(1,1),scale=(2,1)):
 n=len(diagonal);k=len(basis[0]);entries=entries or {};p=dict(dimension=n,columns=k,hDenominator=denominator)
 for i in range(4):
  for j in range(i,4):
   v=diagonal[i] if i==j and i<n else entries.get((i,j),0);re,im=v if isinstance(v,tuple) else (v,0);p[f'hReal{i}{j}']=re
   if i!=j:p[f'hImag{i}{j}']=im
 for i in range(4):
  for j in range(4):
   v=basis[i][j] if i<n and j<k else 0;re,im=v if isinstance(v,tuple)else(v,0);p[f'basisReal{i}{j}'],p[f'basisImag{i}{j}']=re,im
 coefficients=coefficients or [1]+[0]*(k-1)
 for i in range(4):
  v=coefficients[i] if i<k else 0;re,im=v if isinstance(v,tuple)else(v,0);p[f'coefficientReal{i}'],p[f'coefficientImag{i}']=re,im
 p['shearReal'],p['shearImag']=shear if k>1 else (0,0);p['scaleNumerator'],p['scaleDenominator']=scale
 assert len(p)==63
 return p

def add(label,**kwargs):CASES.append((label,prep(**kwargs)))
add('hand computed plane',diagonal=[0,1,3],basis=[[1,0],[0,1],[1,1]])
add('Ritz stationary but not ambient eigenstate',diagonal=[0,2,4],basis=[[1,0],[0,1],[1,0]],coefficients=[1,1])
add('nonorthogonal full space',diagonal=[-1,2],entries={(0,1):(1,2)},basis=[[1,(1,1)],[0,1]],coefficients=[1,(1,-1)])
add('constant H degenerate subspace',diagonal=[-3]*4,basis=[[1,1],[0,1],[(0,1),1],[1,0]],coefficients=[(1,1),2])
add('exact ambient ground',diagonal=[-1,2,5],basis=[[1,0],[0,1],[0,0]])
add('zero projected mean nonzero variance',diagonal=[-1,1],basis=[[1,0],[0,1]],coefficients=[1,1])
add('nearly dependent two columns',diagonal=[-1,3,5],entries={(0,2):(1,1),(1,2):2},basis=[[1000000,999999],[999999,999998],[0,0]],coefficients=[1,-1],scale=(-1,3))
add('ill conditioned full four space',diagonal=[-5,-1,2,7],entries={(0,1):(1,2),(0,3):1,(1,2):(-1,1),(2,3):(2,-1)},basis=[[1,1000000,0,0],[0,1,1000000,0],[0,0,1,1000000],[0,0,0,1]],coefficients=[1,-1,(0,1),2],scale=(-2,3))
add('maximum complex input',diagonal=[1000000,-1000000,999999,-999999],entries={(0,1):(1000000,999999),(1,2):(-999999,1000000),(2,3):(1000000,-1000000)},basis=[[(1000000,1000000),999999],[999998,(-1000000,999999)],[(999999,-999999),999997],[999996,(1000000,-999999)]],coefficients=[(1000000,-1000000),(999999,1000000)],denominator=1000000,shear=(1000000,-1000000),scale=(-1000000,999999))
add('one column negative scale',diagonal=[-2,3,7],basis=[[(1,1)],[2],[(1,-2)]],scale=(-3,5))
rng=random.Random(141108)
for n in range(1,5):
 for k in range(1,n+1):
  for repeat in range(3):
   basis=[]
   for i in range(n):
    row=[]
    for j in range(k):
     row.append((rng.randint(-3,3),rng.randint(-3,3)) if i>j or i>=k else (rng.randint(1,3) if i==j else 0))
    basis.append(row)
   diagonal=[rng.randint(-7,7) for _ in range(n)];entries={(i,j):(rng.randint(-3,3),rng.randint(-3,3))for i in range(n)for j in range(i+1,n)}
   coefficients=[(rng.randint(-3,3),rng.randint(-3,3))for _ in range(k)]
   if all(x==(0,0)for x in coefficients):coefficients[0]=1
   add(f'general complex {n}x{k} case{repeat}',diagonal=diagonal,basis=basis,entries=entries,coefficients=coefficients,denominator=repeat+1,scale=((-2 if repeat==1 else 3),repeat+1))
# An independent invertible complex integer basis change of the same hand plane.
p=dict(CASES[0][1]);p.update(basisReal00=2,basisReal01=1,basisImag01=1,basisReal10=0,basisReal11=1,basisReal20=2,basisReal21=2,basisImag21=1,coefficientReal0=1,coefficientReal1=0)
CASES.append(('same plane changed coordinates',p))

def exact(x):return str(s.cancel(x))
def ec(x):return {'real':exact(s.re(x)),'imaginary':exact(s.im(x))}
def em(a):return [[ec(a[i,j])for j in range(a.cols)]for i in range(a.rows)]
def mv(x):
 x=s.expand(x);re=s.cancel(s.re(x));im=s.cancel(s.im(x));return mp.mpc(mp.mpf(int(re.p))/int(re.q),mp.mpf(int(im.p))/int(im.q))
def mm(a):return mp.matrix([[mv(a[i,j])for j in range(a.cols)]for i in range(a.rows)])
def dec(x):return 'decimal:'+mp.nstr(mp.re(x),215)
def dm(a):return [[{'real':dec(a[i,j].real),'imaginary':dec(a[i,j].imag)}for j in range(a.cols)]for i in range(a.rows)]
def intervals(poly):return s.polys.polytools.intervals(poly,eps=s.Rational(1,10**330))
def midpoint(ab):
 a,b=ab;return (mp.mpf(str(a.p))/int(a.q)+mp.mpf(str(b.p))/int(b.q))/2

def generalized(H,B):
 S=(B.H*B).applyfunc(lambda x:s.expand(x));A=(B.H*H*B).applyfunc(lambda x:s.expand(x));C=(S.inv()*A).applyfunc(lambda x:s.cancel(s.expand(x)));poly=C.charpoly().as_poly();roots=intervals(poly)
 L=mp.cholesky(mm(S));Li=L**-1;values,U=mp.eighe(Li*mm(A)*Li.H);X=Li.H*U;physical=mm(B)*X
 parts=[];offset=0
 for ab,multiplicity in roots:
  lam=midpoint(ab);P=mp.zeros(B.rows)
  for j in range(offset,offset+multiplicity):
   assert abs(values[j]-lam)<mp.mpf('1e-290')*max(1,abs(lam))
   v=physical[:,j];P+=v*v.H
  offset+=multiplicity;parts.append((lam,multiplicity,P))
 return S,A,C,poly,parts

def reference(label,p):
 n=p['dimension'];k=p['columns'];H=s.zeros(n)
 for i in range(n):
  for j in range(i,n):
   v=(s.Integer(p[f'hReal{i}{j}'])+(s.I*p[f'hImag{i}{j}']if i!=j else 0))/p['hDenominator'];H[i,j]=v;H[j,i]=s.conjugate(v)
 B=s.Matrix(n,k,lambda i,j:p[f'basisReal{i}{j}']+s.I*p[f'basisImag{i}{j}']);coeff=s.Matrix([p[f'coefficientReal{i}']+s.I*p[f'coefficientImag{i}']for i in range(k)])
 S,A,C,poly,parts=generalized(H,B);Sinv=S.inv();PV=s.simplify(B*Sinv*B.H);z=B*coeff;N=s.expand((z.H*z)[0]);E=s.cancel((z.H*H*z)[0]/N);second=s.cancel((z.H*H*H*z)[0]/N);r=s.simplify(H*z-E*z);pr=s.simplify(PV*r);outside=s.simplify(r-pr);var=s.cancel((r.H*r)[0]/N);pvar=s.cancel((pr.H*pr)[0]/N);ovar=s.cancel((outside.H*outside)[0]/N);mz=mm(z);mn=mp.re((mz.H*mz)[0]);spectra=[]
 for lam,multiplicity,P in parts:
  prob=mp.re((mz.H*P*mz)[0])/mn
  if abs(prob)<mp.mpf('1e-280'):prob=mp.mpf(0)
  if abs(prob-1)<mp.mpf('1e-280'):prob=mp.mpf(1)
  residual=(mm(H)-lam*mp.eye(n))*P
  norm=sum(abs(residual[i,j])**2 for i in range(n)for j in range(n))
  spectra.append(dict(energy=dec(lam),multiplicity=multiplicity,physicalProjector=dm(P),probability=dec(prob),fullResidualOperator=dm(residual),fullResidualFrobeniusSquared=dec(norm)))
 ambient=[]
 for ab,m in intervals(H.charpoly().as_poly()):ambient.append(dict(energy=dec(midpoint(ab)),multiplicity=m))
 nested=[];last=None
 for j in range(k):
  _,_,_,pol,ps=generalized(H,B[:,:j+1]);lam=ps[0][0];nested.append(dict(columns=j+1,characteristicPolynomial=[exact(v)for v in pol.all_coeffs()],minimumEnergy=dec(lam),minimumMultiplicity=ps[0][1],improvement=None if last is None else dec(last-lam)));last=lam
 gram=[];gvalues=[]
 for ab,m in intervals(S.charpoly().as_poly()):v=midpoint(ab);gram.append(dict(eigenvalue=dec(v),multiplicity=m));gvalues.append(v)
 M=s.eye(k);M[0,0]=s.Rational(p['scaleNumerator'],p['scaleDenominator'])
 if k>1:M[0,1]=p['shearReal']+s.I*p['shearImag']
 CB=B*M;cc=M.inv()*coeff;CS=(CB.H*CB).applyfunc(lambda x:s.cancel(s.expand(x)));CA=(CB.H*H*CB).applyfunc(lambda x:s.cancel(s.expand(x)))
 expected=dict(dimension=n,columns=k,hamiltonian=em(H),basis=em(B),gram=em(S),gramInverse=em(Sinv),gramDeterminant=ec(S.det()),projectedHamiltonian=em(A),coefficientOperator=em(C),spaceProjector=em(PV),coefficients=[ec(v)for v in coeff],coefficientEuclideanNormSquared=exact((coeff.H*coeff)[0]),state=[ec(v)for v in z],rawNormSquared=exact(N),energy=exact(E),secondMoment=exact(second),variance=exact(var),compressedSecondMoment=exact((H*z).H.multiply(PV*(H*z))[0]/N),residualNumerators=[ec(v)for v in r],projectedResidualNumerators=[ec(v)for v in pr],outsideResidualNumerators=[ec(v)for v in outside],projectedVariance=exact(pvar),outsideVariance=exact(ovar),coordinateStationarityResidual=[ec(v)for v in s.simplify(A*coeff-E*S*coeff)],characteristicPolynomial=[exact(v)for v in poly.all_coeffs()],spectrum=spectra,ambientSpectrum=ambient,nestedSubspaces=nested,gramSpectrum=gram,gramConditionNumber=dec(gvalues[-1]/gvalues[0]),basisChange=dict(matrix=em(M),basis=em(CB),coefficients=[ec(v)for v in cc],gram=em(CS),projectedHamiltonian=em(CA),coefficientEuclideanNormSquared=exact((cc.H*cc)[0]),state=[ec(v)for v in s.simplify(CB*cc)],characteristicPolynomial=[exact(v)for v in (CS.inv()*CA).applyfunc(lambda x:s.cancel(s.expand(x))).charpoly().all_coeffs()],spaceProjector=em(s.simplify(CB*CS.inv()*CB.H))))
 return dict(label=label,input=p,expected=expected)

references=[]
for label,p in CASES:
 print('START',label,flush=True);references.append(reference(label,p));print('PASS',label,flush=True)

result=dict(precisionDecimalDigits=400,reportedDigits=215,method='SymPy exact Gram/inverse/moments/charpoly/Sturm multiplicities and mpmath Cholesky plus Hermitian eighe in the whitened metric',spectralGrouping='Exact polynomial real-root multiplicities;no floating gap grouping',referenceZeroPolicy='Fixture numerical probabilities within1e-280 of0/1 use that endpoint;comparison tolerance1e-205;not production classification',cases=references)
path=Path('scripts/fixtures/ritz-reference.json');path.write_text(json.dumps(result,indent=2)+'\n');print(f'{len(CASES)} independent Ritz references written to {path}')
