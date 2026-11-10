#!/usr/bin/env python3
"""Independent SymPy traces/Gaussian ranks and mpmath Hermitian eigensolvers."""
import argparse,json,random
from pathlib import Path
import sympy as S
import mpmath as mp
mp.mp.dps=250

def blank(d):
 p={'dimension':d,**{f'weight{i}':int(i==0)for i in range(4)}}
 for i in range(4):
  for j in range(4):
   p[f'component{i}Real{j}']=int(i==0 and j==0);p[f'component{i}Imag{j}']=0
 for prefix in ['a','b']:
  for i in range(4):
   for j in range(i,4):p[f'{prefix}Real{i}{j}']=0
   for j in range(i+1,4):p[f'{prefix}Imag{i}{j}']=0
  p[f'{prefix}Denominator']=1
 p.update(mixUNumerator=1,mixUDenominator=1,mixVNumerator=1,mixVDenominator=1)
 assert len(p)==75
 return p

def state(p,components,weights):
 for i in range(4):
  p[f'weight{i}']=weights[i]if i<len(weights)else 0
  for j in range(4):
   z=S.sympify(components[i][j])if i<len(components)and j<len(components[i])else S.Integer(0)
   p[f'component{i}Real{j}']=int(S.re(z));p[f'component{i}Imag{j}']=int(S.im(z))
 return p

def operator(p,prefix,matrix,den=1):
 n=p['dimension'];p[f'{prefix}Denominator']=den
 for i in range(n):
  for j in range(i,n):
   z=S.sympify(matrix[i][j]);p[f'{prefix}Real{i}{j}']=int(S.re(z))
   if i<j:p[f'{prefix}Imag{i}{j}']=int(S.im(z))
 return p
X=[[0,1],[1,0]];Y=[[0,-S.I],[S.I,0]];Z=[[1,0],[0,-1]]
def pauli(components,weights):return operator(operator(state(blank(2),components,weights),'a',X),'b',Y)
def cases():
 result=[];put=lambda name,p:result.append({'label':name,'input':p.copy()})
 for d in range(1,5):
  rng=random.Random(100+d);a=[[0]*d for _ in range(d)];b=[[0]*d for _ in range(d)]
  for i in range(d):
   a[i][i]=i-2;b[i][i]=2*i+1
   for j in range(i+1,d):a[i][j]=i+j+1+S.I*(i-j);b[i][j]=2*i-j+S.I*(i+j+2)
  p=operator(operator(blank(d),'a',a,2),'b',b,3)
  basis=[[int(i==j)for j in range(d)]for i in range(d)]
  put(f'dimension {d} uniform',state(p.copy(),basis,[1]*d))
  put(f'dimension {d} pure',state(p.copy(),[[1]+[0]*(d-1)],[1]))
  put(f'dimension {d} complex mixed',state(p.copy(),[[rng.randint(-7,7)+S.I*rng.randint(-7,7)for _ in range(d)]for _ in range(4)],[1,2,3,4]))
  q=state(p.copy(),[[j+1+S.I*(d-j)for j in range(d)]],[1]);put(f'dimension {d} complex pure',q)
  q=operator(q.copy(),'b',a,2);q['mixVNumerator']=-1;put(f'dimension {d} identical observables',q)
  q=operator(p.copy(),'a',[[5 if i==j else 0 for j in range(d)]for i in range(d)]);put(f'dimension {d} constant A',q)
  q=operator(q.copy(),'b',[[0]*d for _ in range(d)]);put(f'dimension {d} constant pair',q)
 put('pure Z Pauli XY',pauli([[1,0]],[1]))
 put('pure XY covariance saturation',pauli([[5,3+4*S.I]],[1]))
 put('pure tilted XZ',pauli([[3,1]],[1]))
 put('mixed Pauli XY',pauli([[1,0],[0,1]],[1,1]))
 put('Z ensemble maximally mixed',pauli([[1,0],[0,1]],[1,1]))
 put('X ensemble maximally mixed',pauli([[1,1],[1,-1]],[1,1]))
 put('Y ensemble maximally mixed',pauli([[1,S.I],[1,-S.I]],[1,1]))
 put('unequal nonorthogonal mixture',pauli([[1,0],[1,1]],[2,3]))
 put('opposite rays pure',pauli([[5,3+4*S.I],[-5,-3-4*S.I]],[2,7]))
 put('zero weight raw zero',pauli([[1,1],[0,0]],[1,0]))
 put('zero weight nonzero raw',pauli([[1,1],[4,3*S.I]],[1,0]))
 p=operator(operator(state(blank(2),[[1,1]],[1]),'a',Z),'b',Z);p['mixVNumerator']=-1;put('commuting identical Z',p)
 p=pauli([[5,3+4*S.I]],[1]);q=p.copy()
 for prefix,shift in [('a',7),('b',-11)]:
  for i in range(2):q[f'{prefix}Real{i}{i}']+=shift*q[f'{prefix}Denominator']
 put('shifted pure XY',q)
 q=p.copy()
 for k in q:
  if k.startswith('aReal')or k.startswith('aImag'):q[k]*=2
  if k.startswith('bReal')or k.startswith('bImag'):q[k]*=-3
 put('scaled pure XY',q)
 p=operator(operator(blank(3),'a',[[0,1,0],[1,0,0],[0,0,0]]),'b',[[0,0,1],[0,0,0],[1,0,0]]);put('qutrit pure strict',p)
 p=operator(operator(state(blank(3),[[1,1,1]],[1]),'a',[[-1,0,0],[0,0,0],[0,0,1]]),'b',[[1,0,0],[0,0,0],[0,0,1]]);put('commuting qutrit strict',p)
 p=operator(operator(blank(3),'a',[[0,1,0],[1,0,0],[0,0,0]]),'b',[[0,1000000,1],[1000000,0,0],[1,0,0]],1000000);put('nearly dependent qutrit',p)
 p=operator(operator(blank(3),'a',[[0,1000000,1],[1000000,0,0],[1,0,0]],1000000),'b',[[0,0,1],[0,0,0],[1,0,0]]);put('small covariance ellipse tilt',p)
 p=pauli([[1000000,1],[1000000,-1]],[1,1]);put('near pure mixed',p)
 p=pauli([[1,0],[0,1]],[1000000,1]);put('thin mixed component',p)
 p=pauli([[1000000+999999*S.I,-999998+999997*S.I]],[1]);p.update(aDenominator=1000000,bDenominator=999999,mixUNumerator=-7,mixUDenominator=11,mixVNumerator=13,mixVDenominator=17);put('large complex rational scales',p)
 p=pauli([[1,0]],[1]);p.update(mixUNumerator=0,mixVNumerator=0);put('zero linear combination',p)
 p=pauli([[1,0]],[1]);p=operator(p,'a',[[0,0],[0,0]]);put('zero variance A nonzero B',p)
 return result

def simp(x):return S.simplify(x)
def number(x):return str(simp(x))
def complex_value(x):return {'real':number(S.re(simp(x))),'imaginary':number(S.im(simp(x)))}
def matrix(x):return [[complex_value(x[i,j])for j in range(x.cols)]for i in range(x.rows)]
def to_mp(x):
 z=S.N(x,255);return mp.mpc(str(S.re(z)),str(S.im(z)))
def decimal(x):return 'decimal:'+mp.nstr(x,210)
def display_matrix(m):return [[{'real':decimal(mp.re(m[i,j])),'imaginary':decimal(mp.im(m[i,j]))}for j in range(m.cols)]for i in range(m.rows)]
def spectral(m):
 eigs=m.eigenvals();eigenvalues=sorted(eigs,key=lambda e:mp.re(to_mp(e)));M=mp.matrix([[to_mp(m[i,j])for j in range(2)]for i in range(2)]);w,U=mp.eighe(M)
 rows=[]
 if len(eigs)==1:rows=[dict(eigenvalue=number(eigenvalues[0]),rank=2,projector=display_matrix(mp.eye(2)),zeroEigenvalueExact=eigenvalues[0]==0)]
 else:
  for j,e in enumerate(eigenvalues):
   v=U[:,j];P=v*v.H
   rows.append(dict(eigenvalue=number(e)if e.is_Rational else decimal(w[j]),rank=1,projector=display_matrix(P),zeroEigenvalueExact=e==0))
 return dict(trace=number(S.trace(m)),determinant=number(m.det()),gapSquared=number(S.trace(m)**2-4*m.det()),positiveRank=m.rank(),eigenspacesMerged=len(eigs)==1,parts=rows)

def evaluate(p):
 d=p['dimension'];I=S.eye(d);z=[S.Matrix([p[f'component{i}Real{j}']+S.I*p[f'component{i}Imag{j}']for j in range(d)])for i in range(4)];N=[(v.H*v)[0]for v in z];W=sum(p[f'weight{i}']for i in range(4));weights=[S.Rational(p[f'weight{i}'],W)for i in range(4)]
 rho=sum((weights[i]*(z[i]*z[i].H)/N[i]for i in range(4)if weights[i]),S.zeros(d));rho=rho.applyfunc(simp)
 def op(prefix):
  m=S.zeros(d)
  for i in range(d):
   for j in range(i,d):
    v=S.Rational(p[f'{prefix}Real{i}{j}'],p[f'{prefix}Denominator'])
    if i!=j:v+=S.I*S.Rational(p[f'{prefix}Imag{i}{j}'],p[f'{prefix}Denominator'])
    m[i,j]=v;m[j,i]=S.conjugate(v)
  return m
 A=op('a');B=op('b');E=lambda M:simp(S.trace(rho*M));a=E(A);b=E(B);DA=A-a*I;DB=B-b*I;va=E(DA*DA);vb=E(DB*DB);cross=E(DA*DB);s=S.re(cross);k=S.im(cross);product=simp(va*vb);strong=simp(cross*S.conjugate(cross));det=simp(product-strong);comm=A*B-B*A;jordan=(DA*DB+DB*DA)/2
 G=S.Matrix([[va,cross],[S.conjugate(cross),vb]]);V=S.Matrix([[va,s],[s,vb]])
 comp=[];withinA=withinB=withinCross=betweenA=betweenB=betweenCross=S.Integer(0)
 for i in range(4):
  if N[i]==0:mom=None
  else:
   ei=lambda M:simp((z[i].H*M*z[i])[0]/N[i]);ma=ei(A);mb=ei(B);a0=A-ma*I;b0=B-mb*I;vva=ei(a0*a0);vvb=ei(b0*b0);c0=ei(a0*b0)
   mom=dict(meanA=number(ma),meanB=number(mb),varianceA=number(vva),varianceB=number(vvb),centeredCross=complex_value(c0))
   withinA+=weights[i]*vva;withinB+=weights[i]*vvb;withinCross+=weights[i]*c0;betweenA+=weights[i]*(ma-a)**2;betweenB+=weights[i]*(mb-b)**2;betweenCross+=weights[i]*(ma-a)*(mb-b)
  comp.append(dict(index=i,occupied=bool(weights[i]),moments=mom))
 slope=None if va==0 else simp(cross/va);R=None if slope is None else DB-slope*DA;residual=vb if R is None else E(R.H*R)
 def weighted(M):
  cols=[]
  for i in range(4):
   v=S.zeros(d,1)if weights[i]==0 else S.sqrt(weights[i]/N[i])*M*z[i]
   cols.append([{'real':decimal(mp.re(to_mp(simp(x)))),'imaginary':decimal(mp.im(to_mp(simp(x))))}for x in v])
  return cols
 u=S.Rational(p['mixUNumerator'],p['mixUDenominator']);v=S.Rational(p['mixVNumerator'],p['mixVDenominator']);C=u*A+v*B;mean=E(C);variance=simp(E(C*C)-mean**2)
 std=lambda x:number(S.sqrt(x))if S.sqrt(x).is_Rational else decimal(mp.sqrt(mp.re(to_mp(x))))
 return dict(dimension=d,weights=[number(w)for w in weights],rawNormsSquared=[number(n)for n in N],density=matrix(rho),purity=number(E(rho)),observables=dict(a=matrix(A),b=matrix(B),centeredA=matrix(DA),centeredB=matrix(DB),commutator=matrix(comm),centeredJordan=matrix(jordan)),moments=dict(meanA=complex_value(a),meanB=complex_value(b),secondA=complex_value(E(A*A)),secondB=complex_value(E(B*B)),varianceA=number(va),varianceB=number(vb),centeredCross=complex_value(cross),symmetricCovariance=number(s),commutatorHalfOverI=number(k),commutatorMean=complex_value(E(comm)),centeredJordanMean=complex_value(E(jordan))),bounds=dict(varianceProduct=number(product),robertsonSquaredBound=number(k*k),covarianceSquared=number(s*s),schrodingerSquaredBound=number(strong),robertsonSlack=number(product-k*k),schrodingerSlack=number(det),stdA=std(va),stdB=std(vb),stdProduct=std(product),robertsonStdProductBound=number(abs(k)),schrodingerStdProductBound=std(strong),standardizedCovariance=None if product==0 else decimal(mp.re(to_mp(s))/mp.sqrt(mp.re(to_mp(product)))),standardizedCommutator=None if product==0 else decimal(mp.re(to_mp(k))/mp.sqrt(mp.re(to_mp(product)))),complexCorrelationSquared=None if product==0 else number(strong/product)),quantumCovariance=matrix(G),realCovariance=matrix(V),quantumSpectrum=spectral(G),realSpectrum=spectral(V),components=comp,varianceDecomposition=dict(withinA=number(withinA),betweenA=number(betweenA),withinB=number(withinB),betweenB=number(betweenB),withinCross=complex_value(withinCross),betweenCross=complex_value(betweenCross)),weightedFluctuationColumns=dict(a=weighted(DA),b=weighted(DB),optimalResidual=None if R is None else weighted(R)),complexProjection=dict(coefficientBOnA=None if slope is None else complex_value(slope),coefficientUnique=slope is not None,minimumResidualVariance=number(residual),orthogonality=None if R is None else complex_value(E(DA*R))),linearCombination=dict(u=number(u),v=number(v),matrix=matrix(C),mean=complex_value(mean),variance=number(variance),quadraticVariance=number(u*u*va+v*v*vb+2*u*v*s)),classification=dict(pureState=E(rho)==1,zeroVarianceA=va==0,zeroVarianceB=vb==0,operatorsCommute=comm==S.zeros(d),commutatorMeanZero=k==0,symmetricCovarianceZero=s==0,robertsonEquality=simp(product-k*k)==0,schrodingerEquality=det==0,quantumCovarianceRank=G.rank(),robertsonBoundTrivial=k==0,standardizedCorrelationDefined=product!=0,jointMeasurementDistributionSupplied=False,measurementInstrumentSupplied=False,measurementErrorSupplied=False,measurementDisturbanceSupplied=False,canonicalPositionMomentumSupplied=False,physicalTimeSpecified=False))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/uncertainty-covariance-reference.json');args=parser.parse_args();rows=cases()
 for row in rows:
  assert len(row['input'])==75 and all(type(x)is int and abs(x)<=1000000 for x in row['input'].values()),row['label']
  row['expected']=evaluate(row['input'])
 Path(args.output).write_text(json.dumps(dict(source='SymPy exact operator traces,Gaussian ranks and mpmath250dps Hermitian eigenvectors; no production imports',cases=rows),indent=2)+'\n');print(f'Generated {len(rows)} cases')
if __name__=='__main__':main()
