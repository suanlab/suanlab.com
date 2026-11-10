"""Independent symbolic spin algebra and high-precision general matrix exponentials.

No production modules, turn reduction, pi enclosure or SU2 trigonometric
implementation are imported. Eigenvectors are independently computed by SymPy.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp
import sympy as s
mp.mp.dps=250

def cv(z):return dict(real='decimal:'+mp.nstr(mp.re(z),210,min_fixed=0,max_fixed=0),imaginary='decimal:'+mp.nstr(mp.im(z),210,min_fixed=0,max_fixed=0))
def dec(x):return 'decimal:'+mp.nstr(mp.re(x),210,min_fixed=0,max_fixed=0)
def matrix(m):return [[cv(m[i,j])for j in range(m.cols)]for i in range(m.rows)]
def amps(z):return [cv(x)for x in z]
def rat(x):return str(s.cancel(x))
def put(prefix,values):return {k:int(v)for i,x in enumerate(values)for k,v in [(f'{prefix}Real{i}',s.re(x)),(f'{prefix}Imag{i}',s.im(x))]}
def vector(p,prefix):return s.Matrix([p[f'{prefix}Real{i}']+s.I*p[f'{prefix}Imag{i}']for i in range(2)])
def ratio(p,prefix):return s.Rational(p[prefix+'Numerator'],p[prefix+'Denominator'])
def mc(x):return mp.mpc(mp.mpf(str(s.re(x).p))/int(s.re(x).q),mp.mpf(str(s.im(x).p))/int(s.im(x).q))
def mm(a):return mp.matrix([[mc(s.simplify(a[i,j]))for j in range(a.cols)]for i in range(a.rows)])
base={**put('state',[1,0]),**put('axis',[1,0]),**put('rotation',[1,1]),'turnNumerator':1,'turnDenominator':1,'hbarNumerator':1,'hbarDenominator':1,'sampleSteps':4}
specs=[('X full turn',{}),('zero angle',{'turnNumerator':0}),('X half turn',{'turnDenominator':2}),('X quarter turn',{'turnDenominator':4}),('two full turns',{'turnNumerator':2}),('negative full turn',{'turnNumerator':-1}),('negative quarter turn',{'turnNumerator':-1,'turnDenominator':4}),('Z rotation',put('rotation',[1,0])),('Y rotation',put('rotation',[1,s.I])),('X eigenstate',put('state',[1,1])),('imaginary input',put('state',[1,s.I])),('negative imaginary input',put('state',[1,-s.I])),('complex input',put('state',[-3+2*s.I,5-7*s.I])),('measure X',put('axis',[1,1])),('measure Y',put('axis',[1,s.I])),('nonorthogonal axes',{**put('axis',[2,1]),**put('rotation',[1,2])}),('complex axes',{**put('axis',[2+3*s.I,-4+5*s.I]),**put('rotation',[-2+s.I,3-4*s.I])}),('hbar twice',{'hbarNumerator':2}),('rational hbar',{'hbarNumerator':5,'hbarDenominator':7}),('third turn',{'turnDenominator':3}),('seven thirteenths',{'turnNumerator':7,'turnDenominator':13}),('large positive angle',{'turnNumerator':1000000,'turnDenominator':3}),('large negative angle',{'turnNumerator':-1000000,'turnDenominator':7}),('near integer turn',{'turnNumerator':999999,'turnDenominator':1000000}),('tiny nonzero turn',{'turnDenominator':1000000}),('raw imaginary scaling',put('state',[3*s.I,0])),('axis raw phase scaling',put('rotation',[5*s.I,5*s.I])),('opposite rotation axis',put('rotation',[1,-1])),('large complex coordinates',{**put('state',[1000000+999999*s.I,-999998+999997*s.I]),**put('axis',[999999+1000000*s.I,-999997+999998*s.I])}),('sixteen samples',{'sampleSteps':16}),('large hbar',{'hbarNumerator':1000000}),('small hbar',{'hbarDenominator':1000000})]
SX=s.Matrix([[0,1],[1,0]]);SY=s.Matrix([[0,-s.I],[s.I,0]]);SZ=s.diag(1,-1);sigmas=[SX,SY,SZ]
cases=[]
for label,changes in specs:
 p=base|changes;z=vector(p,'state');norm2=s.simplify((z.conjugate().T*z)[0]);v=vector(p,'axis');v=v/s.sqrt((v.conjugate().T*v)[0]);vR=vector(p,'rotation');vR=vR/s.sqrt((vR.conjugate().T*vR)[0]);P=s.simplify(v*v.conjugate().T);K=s.simplify(2*vR*vR.conjugate().T-s.eye(2));hb=ratio(p,'hbar');A=hb*(2*P-s.eye(2))/2;spins=[hb*m/2 for m in sigmas];zm=mm(z);N=mp.re(mc(norm2));root=mp.sqrt(N);km=mm(K);am=mm(A);h=mp.re(mc(hb))/2;sigm=[mm(m)for m in sigmas]
 eig=sorted(A.eigenvects(),key=lambda x:x[0]);branches=[]
 for energy,rank,vectors in eig:
  e=vectors[0];proj=s.simplify(e*e.conjugate().T/(e.conjugate().T*e)[0]);prob=s.simplify((z.conjugate().T*proj*z)[0]/norm2);branches.append((energy,proj,prob))
 initial={'matrix':[[{'real':rat(s.re(A[i,j])),'imaginary':rat(s.im(A[i,j]))}for j in range(2)]for i in range(2)],'outcomes':[{'eigenvalue':rat(E),'rank':1,'probability':rat(prob),'projector':[[{'real':rat(s.re(proj[i,j])),'imaginary':rat(s.im(proj[i,j]))}for j in range(2)]for i in range(2)]}for E,proj,prob in branches]}
 initial['mean']=rat((z.conjugate().T*A*z)[0]/norm2);initial['secondMoment']=rat((z.conjugate().T*A*A*z)[0]/norm2);initial['variance']=rat(s.sympify(initial['secondMoment'])-s.sympify(initial['mean'])**2)
 exact_matrix=lambda m:[[{ 'real':rat(s.re(m[i,j])),'imaginary':rat(s.im(m[i,j]))}for j in range(m.cols)]for i in range(m.rows)]
 rotation_axis=[s.simplify(s.trace(K*m)/2)for m in sigmas];measure_axis=[s.simplify(s.trace((2*P-s.eye(2))*m)/2)for m in sigmas];r0=[s.simplify((z.conjugate().T*m*z)[0]/norm2)for m in sigmas]
 records=[]
 for i in range(p['sampleSteps']+1):
  fraction=s.Rational(i,p['sampleSteps']);t=ratio(p,'turn')*fraction;tm=mp.mpf(str(t.p))/int(t.q);u=mp.expm(-mp.j*mp.pi*tm*km);uz=u*zm;up=mp.expm(-mp.j*mp.pi*(tm+1)*km);zp=up*zm;b=[mp.re((uz.H*m*uz)[0]/N)for m in sigm];mean=(uz.H*am*uz)[0]/N;second=mp.re((uz.H*am*am*uz)[0]/N);prob=[mp.re((uz.H*mm(proj)*uz)[0]/N)for _,proj,_ in branches];shiftprob=[mp.re((zp.H*mm(proj)*zp)[0]/N)for _,proj,_ in branches]
  records.append({'index':i,'fraction':rat(fraction),'turns':rat(t),'physicalAngleRadians':dec(2*mp.pi*tm),'propagator':matrix(u),'amplitudes':amps(uz/root),'bloch':[dec(x)for x in b],'rodriguesBloch':[dec(x)for x in b],'spinMeans':[dec(h*x)for x in b],'spinVariances':[dec(h*h*(1-x*x))for x in b],'outcomeProbabilities':[dec(x)for x in prob],'measurementMean':cv(mean),'measurementSecondMoment':dec(second),'measurementVariance':dec(second-mp.re(mean)**2),'computedNormSquared':dec(mp.re((uz.H*uz)[0]/N)),'fidelityToInitial':dec(abs((zm.H*uz)[0])**2/N**2),'representativeSquaredDifference':dec(mp.re(((uz-zm).H*(uz-zm))[0]/N)),'shiftedByOneTurn':{'propagator':matrix(up),'amplitudes':amps(zp/root),'outcomeProbabilities':[dec(x)for x in shiftprob],'stateFidelity':dec(abs((uz.H*zp)[0])**2/N**2),'representativeSquaredDifference':dec(mp.re(((zp-uz).H*(zp-uz))[0]/N))}})
 expected={'hbar':rat(hb),'rawNormSquared':rat(norm2),'rotationAxis':[rat(x)for x in rotation_axis],'measurementAxis':[rat(x)for x in measure_axis],'initialBloch':[rat(x)for x in r0],'spinMatrices':[exact_matrix(m)for m in spins],'casimir':exact_matrix(sum((m*m for m in spins),s.zeros(2))),'casimirEigenvalue':rat(3*hb*hb/4),'initialMeasurement':initial,'records':records,'classification':{'integerEndpointTurns':ratio(p,'turn').q==1,'integerEndpointSpinorSign':int((-1)**int(ratio(p,'turn')))if ratio(p,'turn').q==1 else None}}
 cases.append({'label':label,'input':p,'expected':expected});print('prepared '+label,flush=True)
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/spin-half-reference.json');args=parser.parse_args();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'method':'SymPy Pauli spin algebra and independent eigenspaces; mpmath250dps general matrix exponentials exp(-i*pi*turn*K),210-digit coordinates. No production imports,Machin pi or Taylor/doubling trigonometry.','pi':dec(mp.pi),'cases':cases},indent=2)+'\n');print(json.dumps({'preparations':len(cases),'angleSamples':sum(len(c['expected']['records'])for c in cases)}))
