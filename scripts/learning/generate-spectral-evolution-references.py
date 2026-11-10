"""Independent matrix exponentials and exact direct-matrix Euler iterations.

Production uses directed Taylor/doubling and spectral Euler factors; no production
module, fixed-point arithmetic or trigonometric implementation is imported here.
"""
import argparse
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
import sympy as s

sys.set_int_max_str_digits(0)
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
def fc(x):return(F(str(s.re(x))),F(str(s.im(x))))
def plus(a,b):return(a[0]+b[0],a[1]+b[1])
def times(a,b):return(a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def matvec(A,z):return [sumc([times(a,b)for a,b in zip(row,z)])for row in A]
def sumc(xs):
 out=(F(0),F(0))
 for x in xs:out=plus(out,x)
 return out

def norm(z):return sum(a*a+b*b for a,b in z)
def mpvec(z):return mp.matrix([mp.mpc(mp.mpf(a.numerator)/a.denominator,mp.mpf(b.numerator)/b.denominator)for a,b in z])
base={**put('state',[1,0]),**put('axis',[1,1]),'centerNumerator':0,'centerDenominator':1,'gapNumerator':1,'gapDenominator':1,'timeNumerator':1,'timeDenominator':1,'hbarNumerator':1,'hbarDenominator':1,'sampleSteps':4,'eulerSteps':16}
specs=[
 ('X evolution',{}),('zero duration',{'timeNumerator':0}),
 ('Z eigenstate',put('axis',[1,0])),('X eigenstate',put('state',[1,1])),
 ('imaginary relative phase',put('state',[1,s.I])),('negative relative phase',put('state',[1,-s.I])),
 ('complex state',put('state',[-3+2*s.I,5-7*s.I])),('complex Hamiltonian axis',put('axis',[2+3*s.I,-4+5*s.I])),
 ('energy shift only',{'centerNumerator':1}),('negative energy shift',{'centerNumerator':-3,'centerDenominator':7}),
 ('negative gap',{'gapNumerator':-1}),('scalar Hamiltonian',{'gapNumerator':0,'centerNumerator':2}),('zero Hamiltonian',{'gapNumerator':0}),
 ('zero eigenvalue stationary state',{**put('axis',[1,0]),**put('state',[0,1]),'centerNumerator':1}),
 ('hbar twice as large',{'hbarNumerator':2}),('hbar half as large',{'hbarDenominator':2}),
 ('rational spectrum',{'centerNumerator':-2,'centerDenominator':3,'gapNumerator':5,'gapDenominator':11,'timeNumerator':7,'timeDenominator':4}),
 ('small physical time',{'timeDenominator':1000000}),('small nonzero gap',{'gapDenominator':1000000}),
 ('near pi revival',{'timeNumerator':355,'timeDenominator':113}),('near half pi transfer',{'timeNumerator':104348,'timeDenominator':66430}),
 ('global imaginary raw scaling',put('state',[3*s.I,0])),('axis imaginary raw scaling',put('axis',[5*s.I,5*s.I])),
 ('fine Euler grid',{'eulerSteps':64}),('eight physical samples',{'sampleSteps':8,'eulerSteps':64}),
 ('single Euler step',{'sampleSteps':1,'eulerSteps':1}),
 ('maximum complex integers',{**put('state',[1000000-999999*s.I,-999999+1000000*s.I]),**put('axis',[-1000000+999999*s.I,1000000+1000000*s.I])}),
 ('large spectral phase',{'gapNumerator':1000000,'timeNumerator':1000000,'hbarNumerator':1,'hbarDenominator':1000000,'sampleSteps':2,'eulerSteps':2}),
 ('large common and relative phases',{'centerNumerator':-1000000,'gapNumerator':1000000,'timeNumerator':1000000,'hbarNumerator':1,'hbarDenominator':1000000,'sampleSteps':2,'eulerSteps':4}),
 ('large exact denominators',{**put('state',[1000000-999999*s.I,-999999+1000000*s.I]),**put('axis',[-1000000+999999*s.I,1000000+1000000*s.I]),'centerNumerator':-999999,'centerDenominator':999983,'gapNumerator':1000000,'gapDenominator':999979,'timeNumerator':1000000,'timeDenominator':999983,'hbarNumerator':999979,'hbarDenominator':1000000,'sampleSteps':4,'eulerSteps':64}),
 ('Euler overflow retained exactly',{'gapNumerator':1000000,'timeNumerator':1000000,'hbarNumerator':1,'hbarDenominator':1000000,'sampleSteps':4,'eulerSteps':64}),
]
cases=[]
for label,patch in specs:
 p={**base,**patch};z=vector(p,'state');v=vector(p,'axis');N=s.simplify((z.H*z)[0]);P=s.simplify(v*v.H/(v.H*v)[0]);a,b=ratio(p,'center'),ratio(p,'gap');H=(a*s.eye(2)+b*(2*P-s.eye(2))).applyfunc(s.simplify);T,hbar=ratio(p,'time'),ratio(p,'hbar');dt=T/p['eulerSteps'];psi=mm(z)/mp.sqrt(int(N));mh=mm(H);ht=hbar.evalf(270);mhbar=mp.mpf(str(ht));spectra=[]
 for value,_,vectors in sorted(H.eigenvects(),key=lambda e:e[0]):
  basis=s.GramSchmidt(vectors,True);E=sum((s.simplify(w*w.H)for w in basis),s.zeros(2));spectra.append((value,E))
 initial_mean=s.simplify((z.H*H*z)[0]/N);initial_second=s.simplify(((H*z).H*(H*z))[0]/N)
 exact_H=[[fc(H[i,j])for j in range(2)]for i in range(2)];qdt=F(str(dt/hbar));euler_kernel=[[plus((F(int(i==j)),F(0)),times((F(0),-qdt),exact_H[i][j]))for j in range(2)]for i in range(2)];ez=[fc(x)for x in z];euler={0:ez};
 for j in range(1,p['eulerSteps']+1):ez=matvec(euler_kernel,ez);euler[j]=ez
 expected=dict(hbar=rat(hbar),duration=rat(T),eulerTimeStep=rat(dt),rawNormSquared=rat(N),hamiltonian=[[dict(real=rat(s.re(H[i,j])),imaginary=rat(s.im(H[i,j])))for j in range(2)]for i in range(2)],initialEnergyMean=rat(initial_mean),initialEnergySecondMoment=rat(initial_second),initialEnergyVariance=rat(initial_second-initial_mean**2),energySpectrum=[dict(eigenvalue=rat(value),rank=int(s.trace(E)),initialProbability=rat(s.simplify((z.H*E*z)[0]/N)))for value,E in spectra],records=[])
 for i in range(p['sampleSteps']+1):
  t=T*i/p['sampleSteps'];mt=mp.mpf(str(t.evalf(270)));U=mp.expm(-mp.j*mh*mt/mhbar);out=U*psi;without=mp.expm(-mp.j*(mh-mc(a)*mp.eye(2))*mt/mhbar)*psi;k=i*p['eulerSteps']//p['sampleSteps'];eu=mpvec(euler[k])/mp.sqrt(int(N));enorm=norm(euler[k])/F(int(N));fidelity=abs((psi.H*out)[0])**2
  record=dict(index=i,eulerStep=k,time=rat(t),phaseAngle=rat(a*t/hbar),relativePhaseHalfAngle=rat(b*t/hbar),propagator=matrix(U),amplitudes=amps(out),amplitudesWithoutGlobalPhase=amps(without),standardProbabilities=[dec(abs(x)**2)for x in out],energyProbabilities=[dec((out.H*mm(E)*out)[0])for _,E in spectra],computedNormSquared='1',computedEnergyMean=dict(real=rat(initial_mean),imaginary='0'),computedEnergySecondMoment=rat(initial_second),fidelityToInitial=dec(fidelity),fidelityDefect=dec(1-fidelity),eulerNormSquared=str(enorm),eulerNormExcess=str(enorm-1),eulerSpectralNormSquared=str(enorm),eulerNormIdentityResidual='0',eulerAmplitudes=amps(eu),squaredEulerError=dec(sum(abs(x-y)**2 for x,y in zip(out,eu))))
  expected['records'].append(record)
 cases.append(dict(label=label,input=p,expected=expected))
 print('prepared '+label,flush=True)
# Trigonometric references include exact rational arguments far beyond binary64 phase resolution.
angles=[('zero','0'),('eighth','1/8'),('negative eighth','-1/8'),('one','1'),('negative one','-1'),('near pi','355/113'),('near half pi','104348/66430'),('tiny','1/1000000000000000000'),('negative tiny','-1/1000000000000000000'),('large','1000000000000000000'),('large plus rational','1000000000000000001/1'),('large fractional','999999999999999999999999/1000000'),('negative large','-1000000000000000000'),('phase limit',str(2**64)),('negative phase limit',str(-2**64))]
trig=[]
for label,raw in angles:
 x=F(raw);mx=mp.mpf(x.numerator)/x.denominator;trig.append(dict(label=label,numerator=str(x.numerator),denominator=str(x.denominator),sin=dec(mp.sin(mx)),cos=dec(mp.cos(mx))))
interval_specs=[('cross zero',F(-1),F(1)),('cross sine maximum',F(1),F(2)),('cross cosine minimum',F(3),F(4)),('wide range',F(-10),F(10)),('near phase limit',F(2**64-1),F(2**64))]
interval_cases=[]
for label,lower,upper in interval_specs:
 samples=[]
 for j in range(9):
  x=lower+(upper-lower)*F(j,8);mx=mp.mpf(x.numerator)/x.denominator;samples.append(dict(sin=dec(mp.sin(mx)),cos=dec(mp.cos(mx))))
 interval_cases.append(dict(label=label,lower=str(lower),upper=str(upper),samples=samples))
ode_cases=[]
for label in ['X evolution','imaginary relative phase','complex state','complex Hamiltonian axis','energy shift only','negative gap','rational spectrum','hbar half as large','near pi revival']:
 case=next(c for c in cases if c['label']==label);p=case['input'];v=vector(p,'axis');P=s.simplify(v*v.H/(v.H*v)[0]);H=(ratio(p,'center')*s.eye(2)+ratio(p,'gap')*(2*P-s.eye(2))).applyfunc(s.simplify);ham=np.array([[complex(s.N(H[i,j],18))for j in range(2)]for i in range(2)]);z=np.array([complex(x)for x in vector(p,'state')]);psi=z/np.linalg.norm(z);end=float(ratio(p,'time'));hb=float(ratio(p,'hbar'));times=np.linspace(0,end,p['sampleSteps']+1)
 solution=solve_ivp(lambda t,y:-1j*ham@y/hb,(0,end),psi,method='DOP853',t_eval=times,rtol=1e-12,atol=1e-14);assert solution.success
 ode_cases.append(dict(label=label,amplitudes=[[dict(real=float(z.real),imaginary=float(z.imag))for z in row]for row in solution.y.T]))
result=dict(method='mpmath250dps direct matrix exponentials; exact Python Fraction complex Euler matrix-vector iterations; SymPy Hermitian eigenspaces;210-digit reference coordinates and trigonometry. Separate complex DOP853 integrations use rtol1e-12,atol1e-14.',cases=cases,trigonometricCases=trig,intervalCases=interval_cases,odeCases=ode_cases)
a=argparse.ArgumentParser();a.add_argument('--output',default='scripts/fixtures/spectral-evolution-reference.json');args=a.parse_args();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(spectralPreparations=len(cases),timeSamples=sum(len(c['expected']['records'])for c in cases),trigonometricPreparations=len(trig),intervalPreparations=len(interval_cases),odeCases=len(ode_cases))))
