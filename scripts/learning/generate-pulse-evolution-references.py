"""Independent direct matrix exponentials and segmented complex DOP853 pulses.
No production imports, trigonometric enclosures or SU2 radius formula.
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
base={**put('state',[1,0]),**put('axisA',[1,1]),**put('axisB',[1,0]),'aCenterNumerator':0,'aCenterDenominator':1,'aGapNumerator':1,'aGapDenominator':1,'bCenterNumerator':0,'bCenterDenominator':1,'bGapNumerator':1,'bGapDenominator':1,'durationANumerator':1,'durationADenominator':1,'durationBNumerator':1,'durationBDenominator':1,'hbarNumerator':1,'hbarDenominator':1,'sampleSteps':4}
specs=[
 ('X then Z',{}),
 ('unequal durations',{'durationANumerator':2}),
 ('off-grid switch times',{'durationANumerator':2,'durationADenominator':3,'durationBNumerator':4,'durationBDenominator':5,'sampleSteps':4}),
 ('nonorthogonal axes',put('axisB',[2,1])),
 ('complex state',put('state',[-3+2*s.I,5-7*s.I])),
 ('complex axes',{**put('axisA',[2+3*s.I,-4+5*s.I]),**put('axisB',[-2+s.I,3-4*s.I])}),
 ('imaginary relative phase',put('state',[1,s.I])),
 ('identical Hamiltonians unequal durations',{**put('axisB',[1,1]),'durationANumerator':2}),
 ('commuting unequal schedule',{**put('axisA',[1,0]),**put('axisB',[1,0]),**put('state',[1,1]),'bGapNumerator':2}),
 ('opposite axes cancellation',put('axisB',[1,-1])),
 ('scalar mean with cancelled traceless action',{**put('axisB',[1,-1]),'aCenterNumerator':3,'bCenterNumerator':-1}),
 ('zero first duration',{'durationANumerator':0}),
 ('zero second duration',{'durationBNumerator':0}),
 ('zero total duration',{'durationANumerator':0,'durationBNumerator':0}),
 ('scalar A pulse',{'aGapNumerator':0,'aCenterNumerator':2}),
 ('scalar B pulse',{'bGapNumerator':0,'bCenterNumerator':-3}),
 ('both scalar pulses',{'aGapNumerator':0,'aCenterNumerator':2,'bGapNumerator':0,'bCenterNumerator':-3}),
 ('zero A Hamiltonian',{'aGapNumerator':0}),
 ('negative gaps',{'aGapNumerator':-1,'bGapNumerator':-2}),
 ('common energy shift',{'aCenterNumerator':1,'bCenterNumerator':1}),
 ('one pulse energy shift',{'aCenterNumerator':3,'aCenterDenominator':7}),
 ('rational spectrum and hbar',{'aCenterNumerator':-2,'aCenterDenominator':3,'aGapNumerator':5,'aGapDenominator':11,'bCenterNumerator':3,'bCenterDenominator':7,'bGapNumerator':-4,'bGapDenominator':13,'hbarNumerator':3,'hbarDenominator':2}),
 ('half hbar',{'hbarDenominator':2}),
 ('small pulses',{'durationADenominator':100,'durationBDenominator':100}),
 ('half small pulses',{'durationADenominator':200,'durationBDenominator':200}),
 ('tiny nonzero actions',{'durationADenominator':1000000,'durationBDenominator':1000000,'aGapDenominator':1000000,'bGapDenominator':1000000,'hbarNumerator':1000000}),
 ('near opposite axes',{**put('axisA',[1000000,999999]),**put('axisB',[-999998,999999])}),
 ('near opposite tiny actions',{**put('axisA',[1000000,999999]),**put('axisB',[-999998,999999]),'durationADenominator':1000000,'durationBDenominator':1000000,'aGapDenominator':1000000,'bGapDenominator':1000000,'hbarNumerator':1000000}),
 ('global imaginary raw scaling',put('state',[3*s.I,0])),
 ('raw axis scaling',put('axisA',[5*s.I,5*s.I])),
 ('sixteen intervals',{'sampleSteps':16}),
 ('large pulse phases',{'aGapNumerator':1000000,'bGapNumerator':1000000,'durationANumerator':1000000,'durationBNumerator':1000000,'hbarDenominator':1000000,'sampleSteps':2}),
 ('large scalar and pulse phases',{'aCenterNumerator':-1000000,'bCenterNumerator':1000000,'aGapNumerator':1000000,'bGapNumerator':1000000,'durationANumerator':1000000,'durationBNumerator':1000000,'hbarDenominator':1000000,'sampleSteps':2}),
 ('large complex coordinates',{**put('state',[1000000-999999*s.I,-999999+1000000*s.I]),**put('axisA',[-1000000+999999*s.I,1000000+1000000*s.I]),**put('axisB',[999999+1000000*s.I,-1000000+999999*s.I])}),
 ('large exact denominators',{**put('state',[1000000-999999*s.I,-999999+1000000*s.I]),**put('axisA',[-1000000+999999*s.I,1000000+1000000*s.I]),**put('axisB',[999999+1000000*s.I,-1000000+999999*s.I]),'aCenterNumerator':-999999,'aCenterDenominator':999983,'aGapNumerator':1000000,'aGapDenominator':999979,'bCenterNumerator':999983,'bCenterDenominator':999979,'bGapNumerator':-999979,'bGapDenominator':999983,'durationANumerator':1000000,'durationADenominator':999983,'durationBNumerator':999979,'durationBDenominator':1000000,'hbarNumerator':999979,'hbarDenominator':1000000}),
]
def H_for(p,prefix):
 v=vector(p,'axis'+prefix.upper());P=s.simplify(v*v.H/(v.H*v)[0]);return(ratio(p,prefix+'Center')*s.eye(2)+ratio(p,prefix+'Gap')*(2*P-s.eye(2))).applyfunc(s.simplify)
def exact_matrix(a):return[[dict(real=rat(s.re(a[i,j])),imaginary=rat(s.im(a[i,j])))for j in range(a.cols)]for i in range(a.rows)]
def frob(a):return sum(abs(x)**2 for x in a)
def prob_path(U,psi,ha,hb,g):
 out=U*psi;ea=(out.H*ha*out)[0];eb=(out.H*hb*out)[0];eg=(out.H*g*out)[0]
 return dict(propagator=matrix(U),amplitudes=amps(out),standardProbabilities=[dec(abs(x)**2)for x in out],plusXProbability=dec(abs(out[0]+out[1])**2/2),plusYProbability=dec(abs(out[0]-mp.j*out[1])**2/2),computedNormSquared='1',energyMeanA=cv(ea),energyMeanB=cv(eb),totalActionMean=cv(eg))
def compare_path(a,b,psi):
 za,zb=a*psi,b*psi;fidelity=abs((za.H*zb)[0])**2;process=abs(sum((a.H*b)[i,i]for i in range(2))/2)**2
 return dict(stateFidelity=dec(fidelity),stateFidelityDefect=dec(1-fidelity),stateRepresentativeSquaredDifference=dec(sum(abs(x-y)**2 for x,y in zip(za,zb))),propagatorSquaredDifference=dec(frob(a-b)),processFidelityDefect=dec(1-process))
cases=[]
for label,patch in specs:
 p={**base,**patch};z=vector(p,'state');N=s.simplify((z.H*z)[0]);psi=mm(z)/mp.sqrt(int(N));ha,hb=H_for(p,'a'),H_for(p,'b');ta,tb,hbar=ratio(p,'durationA'),ratio(p,'durationB'),ratio(p,'hbar');T=ta+tb;ga,gb=ha*ta/hbar,hb*tb/hbar;g=(ga+gb).applyfunc(s.simplify);commH=(ha*hb-hb*ha).applyfunc(s.simplify);commG=(ga*gb-gb*ga).applyfunc(s.simplify);mha,mhb,mg=mm(ha),mm(hb),mm(g);initial=s.simplify((z.H*g*z)[0]/N);mt=lambda x:mp.mpf(str(s.sympify(x).evalf(270)));mhbar=mt(hbar)
 times=sorted(set([T*s.Rational(i,p['sampleSteps'])for i in range(p['sampleSteps']+1)]+[ta,tb]));observations=[]
 def U(h,t):return mp.expm(-mp.j*h*mt(t)/mhbar)
 def phase(t,first_duration,first,second):return 'finished'if t==T else second if t>=first_duration else first
 for i,t in enumerate(times):
  af,bf=min(t,ta),max(0,t-ta);br,ar=min(t,tb),max(0,t-tb);uf=U(mhb,bf)*U(mha,af);ur=U(mha,ar)*U(mhb,br);um=mp.eye(2)if T==0 else mp.expm(-mp.j*mg*mt(t/T));action_f=(ha*af+hb*bf)/hbar;action_r=(ha*ar+hb*br)/hbar
  observations.append(dict(index=i,time=rat(t),forwardDurations=dict(a=rat(af),b=rat(bf)),reverseDurations=dict(a=rat(ar),b=rat(br)),forwardPhase=phase(t,ta,'A','B'),reversePhase=phase(t,tb,'B','A'),forward=prob_path(uf,psi,mha,mhb,mg),reverse=prob_path(ur,psi,mha,mhb,mg),average=prob_path(um,psi,mha,mhb,mg),accumulatedActionForward=exact_matrix(action_f),accumulatedActionReverse=exact_matrix(action_r),comparisons=dict(order=compare_path(uf,ur,psi),forwardAverage=compare_path(uf,um,psi),reverseAverage=compare_path(ur,um,psi)),averageActionMeanResidual=dict(real='0',imaginary='0')))
 uf=U(mhb,tb)*U(mha,ta);ur=U(mha,ta)*U(mhb,tb);um=mp.expm(-mp.j*mg);undo=U(mha,-ta)*U(mhb,-tb);commutes=commH==s.zeros(2);gcommutes=commG==s.zeros(2);same=ha==hb
 expected=dict(hbar=rat(hbar),durationA=rat(ta),durationB=rat(tb),totalDuration=rat(T),rawNormSquared=rat(N),hamiltonianA=exact_matrix(ha),hamiltonianB=exact_matrix(hb),averageHamiltonian=None if T==0 else exact_matrix((ha*ta+hb*tb)/T),integratedActionA=exact_matrix(ga),integratedActionB=exact_matrix(gb),totalIntegratedAction=exact_matrix(g),initialTotalActionMean=rat(initial),hamiltonianCommutator=exact_matrix(commH),hamiltonianCommutatorSquaredNorm=rat(s.trace(commH.H*commH)),integratedActionCommutator=exact_matrix(commG),integratedActionCommutatorSquaredNorm=rat(s.trace(commG.H*commG)),records=observations,endpointUndoPropagator=matrix(undo),endpointLocalExpansion=dict(leadingOrderDifference=exact_matrix(commG),leadingForwardAverageDifference=exact_matrix(commG/2),orderRemainderSquaredNorm=dec(frob(uf-ur-mm(commG))),forwardAverageRemainderSquaredNorm=dec(frob(uf-um-mm(commG)/2))),classification=dict(physicalTimeSpecified=True,closedPureStateOnly=True,hamiltoniansCommute=commutes,integratedActionsCommute=gcommutes,endpointEqualityProved=gcommutes,identicalHamiltonians=same,allThreePrefixesEqualProved=T==0 or ta==0 or tb==0 or same,zeroDuration=T==0,zeroPulseA=ta==0,zeroPulseB=tb==0,meanTracelessActionExactlyZero=s.simplify(g-s.trace(g)*s.eye(2)/2)==s.zeros(2),intermediateMeasurementApplied=False,stateRenormalized=False,positiveReorderingIsNotTimeReversal=True,infiniteDimensionalDomainSpecified=False))
 cases.append(dict(label=label,input=p,expected=expected));print('prepared '+label,flush=True)
ode_cases=[]
for label in ['X then Z','unequal durations','off-grid switch times','nonorthogonal axes','complex state','complex axes','commuting unequal schedule','opposite axes cancellation','rational spectrum and hbar','negative gaps']:
 case=next(c for c in cases if c['label']==label);p=case['input'];ha,hb=H_for(p,'a'),H_for(p,'b');ta,tb=map(float,[ratio(p,'durationA'),ratio(p,'durationB')]);T=ta+tb;hbv=float(ratio(p,'hbar'));A=np.array([[complex(s.N(ha[i,j],18))for j in range(2)]for i in range(2)]);B=np.array([[complex(s.N(hb[i,j],18))for j in range(2)]for i in range(2)]);z=np.array([complex(x)for x in vector(p,'state')]);psi=z/np.linalg.norm(z);times=[float(s.Rational(r['time']))for r in case['expected']['records']]
 def integrate(segments):
  elapsed=0.;start=psi.copy();solutions=[]
  for H,duration in segments:
   if duration==0:continue
   solution=solve_ivp(lambda t,y:-1j*H@y/hbv,(0,duration),start,method='DOP853',dense_output=True,rtol=1e-12,atol=1e-14);assert solution.success;solutions.append((elapsed,elapsed+duration,solution));elapsed+=duration;start=solution.y[:,-1]
  out=[]
  for time in times:
   found=next((x for x in solutions if time<=x[1]+1e-14),None)
   if found:lo,hi,solution=found;value=solution.sol(min(max(time-lo,0),hi-lo))
   else:value=psi if not solutions else start
   out.append([dict(real=float(x.real),imaginary=float(x.imag))for x in value])
  return out
 mean=(A*ta+B*tb)/T if T else np.zeros((2,2));ode_cases.append(dict(label=label,forward=integrate([(A,ta),(B,tb)]),reverse=integrate([(B,tb),(A,ta)]),average=integrate([(mean,T)])))
result=dict(method='mpmath250dps direct exponentials of complete Hermitian matrices at210-digit output; SymPy exact actions and commutators; independent segmented complex DOP853 forward/reverse/constant-average trajectories (rtol1e-12,atol1e-14). No production imports or SU2 sine/radius implementation.',cases=cases,odeCases=ode_cases)
a=argparse.ArgumentParser();a.add_argument('--output',default='scripts/fixtures/pulse-evolution-reference.json');args=a.parse_args();Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(preparations=len(cases),physicalTimeSamples=sum(len(c['expected']['records'])for c in cases),odeCases=len(ode_cases))))
