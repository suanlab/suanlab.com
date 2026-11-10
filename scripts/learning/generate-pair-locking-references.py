#!/usr/bin/env python3
"""Independent220-digit2x2 exponential and analytic drift-period phase lift."""
import argparse,json
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
mp.mp.dps=220
base=dict(omega1Numerator=3,omega1Denominator=1,omega2Numerator=0,omega2Denominator=1,couplingNumerator=5,couplingDenominator=1,initialHalfA=0,initialHalfB=1,meanPhaseNumerator=0,meanPhaseDenominator=1,timeNumerator=4,timeDenominator=1,sampleSteps=8)
specs=[('locking from alignment',{}),('stable exact fixed phase',dict(initialHalfA=1,initialHalfB=3)),('unstable exact fixed phase',dict(initialHalfA=3,initialHalfB=1)),('locking from the upper cut',dict(initialHalfA=1,initialHalfB=0)),('reverse locking',dict(omega1Numerator=-3)),('reverse stable fixed phase',dict(omega1Numerator=-3,initialHalfA=-1,initialHalfB=3)),('reverse unstable fixed phase',dict(omega1Numerator=-3,initialHalfA=-3,initialHalfB=1)),('drift positive',dict(omega1Numerator=6,couplingNumerator=3)),('drift negative',dict(omega1Numerator=-6,couplingNumerator=3)),('drift other initial phase',dict(omega1Numerator=6,couplingNumerator=3,initialHalfA=2,initialHalfB=1)),('critical from alignment',dict(omega1Numerator=1,couplingNumerator=1)),('critical fixed phase',dict(omega1Numerator=1,couplingNumerator=1,initialHalfA=1,initialHalfB=1)),('critical from beyond the bottleneck',dict(omega1Numerator=1,couplingNumerator=1,initialHalfA=2,initialHalfB=1)),('critical exact cut at two seconds',dict(omega1Numerator=1,couplingNumerator=1,initialHalfA=2,initialHalfB=1,timeNumerator=2)),('reverse critical exact cut',dict(omega1Numerator=-1,couplingNumerator=1,initialHalfA=-2,initialHalfB=1,timeNumerator=2)),('critical initial upper cut',dict(omega1Numerator=1,couplingNumerator=1,initialHalfA=1,initialHalfB=0)),('reverse critical initial upper cut',dict(omega1Numerator=-1,couplingNumerator=1,initialHalfA=1,initialHalfB=0)),('neutral aligned',dict(omega1Numerator=2,omega2Numerator=2,couplingNumerator=0)),('neutral antiphase',dict(omega1Numerator=2,omega2Numerator=2,couplingNumerator=0,initialHalfA=1,initialHalfB=0)),('identical unstable antiphase',dict(omega1Numerator=2,omega2Numerator=2,couplingNumerator=5,initialHalfA=1,initialHalfB=0)),('free different frequencies',dict(couplingNumerator=0)),('free reverse from upper cut',dict(omega1Numerator=-3,couplingNumerator=0,initialHalfA=1,initialHalfB=0)),('zero time',dict(timeNumerator=0,initialHalfA=1,initialHalfB=0)),('dense records',dict(omega1Numerator=6,couplingNumerator=3,sampleSteps=64)),('near threshold locked',dict(omega1Numerator=1000000000,omega1Denominator=1000000000,couplingNumerator=1000000000,couplingDenominator=999999999)),('near threshold drifting',dict(omega1Numerator=1000000000,omega1Denominator=999999999,couplingNumerator=1)),('tiny positive time',dict(timeNumerator=1,timeDenominator=1000000000)),('weak coupling',dict(couplingNumerator=1,couplingDenominator=1000000000)),('fractional protocol',dict(omega1Numerator=7,omega1Denominator=3,omega2Numerator=-2,omega2Denominator=5,couplingNumerator=4,couplingDenominator=3,initialHalfA=-2,initialHalfB=3,meanPhaseNumerator=1,meanPhaseDenominator=3,timeNumerator=3,timeDenominator=2,sampleSteps=13)),('homogeneous gauge negative',dict(initialHalfA=-2,initialHalfB=-6)),('mean frame shifted',dict(omega1Numerator=5,omega2Numerator=2)),('maximum rate drifting',dict(omega1Numerator=8,omega2Numerator=-8,couplingNumerator=15)),('maximum rate hyperbolic',dict(omega1Numerator=0,couplingNumerator=16,initialHalfA=999999999,initialHalfB=1)),('almost unstable fixed phase',dict(initialHalfA=1000000000,initialHalfB=333333333)),('critical zero time',dict(omega1Numerator=1,couplingNumerator=1,initialHalfA=2,initialHalfB=1,timeNumerator=0)),('neutral general phase',dict(omega1Numerator=0,omega2Numerator=0,couplingNumerator=0,initialHalfA=-3,initialHalfB=2)),('drift near period below',dict(omega1Numerator=2,couplingNumerator=0,timeNumerator=314159265,timeDenominator=100000000)),('drift near period above',dict(omega1Numerator=2,couplingNumerator=0,timeNumerator=314159266,timeDenominator=100000000))]
def val(x):return mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(x)
def out(x):return str(x) if isinstance(x,F) else mp.nstr(x,180)
a,b,d,k=sp.symbols('a b Delta K',real=True);norm=a*a+b*b;da=(d*b-k*a)/2;db=(k*b-d*a)/2;assert sp.simplify(2*(b*da-a*db)/norm-(d-k*2*a*b/norm))==0
cases=[];record_count=0
for label,patch in specs:
 p={**base,**patch};ratio=lambda s:F(p[s+'Numerator'],p[s+'Denominator']);w1,w2,K,mean0,T=map(ratio,['omega1','omega2','coupling','meanPhase','time']);delta=w1-w2;mean=(w1+w2)/2;disc=K*K-delta*delta
 ai,bi=p['initialHalfA'],p['initialHalfB']
 if bi<0 or bi==0 and ai<0:ai,bi=-ai,-bi
 scale=max(abs(ai),abs(bi));aa,bb=F(ai,scale),F(bi,scale);norm=aa*aa+bb*bb;cs=(bb*bb-aa*aa)/norm;sn=2*aa*bb/norm;speed=delta-K*sn;fixed=speed==0;neutral=K==0 and delta==0;critical=disc==0 and not neutral
 phase0=mp.pi if bi==0 else 2*mp.atan(val(aa)/val(bb));mm,dd,kk,pp=map(val,[mean,delta,K,mean0]);A=mp.matrix([[kk/2,-dd/2],[dd/2,-kk/2]]);v0=mp.matrix([val(bb),val(aa)])
 spectral=mp.sqrt(abs(val(disc)));asym=mp.sign(dd)*spectral if disc<0 else mp.mpf(0);rows=[]
 for step in range(p['sampleSteps']+1):
  tq=T*F(step,p['sampleSteps']);t=val(tq);M=mp.expm(t*A);v=M*v0
  if fixed or not tq:change=mp.mpf(0)
  elif not K:change=val(delta*tq)
  elif disc<0:
   periods=int(mp.floor(spectral*t/(2*mp.pi)));z=((-1)**periods)*mp.mpc(v[0],v[1])*mp.conj(mp.mpc(v0[0],v0[1]));change=mp.sign(dd)*2*mp.pi*periods+2*mp.arg(z)
  else:change=2*mp.arg(mp.mpc(v[0],v[1])*mp.conj(mp.mpc(v0[0],v0[1])))
  phase=phase0+change;pole=critical and bb+(K*bb-delta*aa)*tq/2==0
  aligned=critical and aa+(delta*bb-K*aa)*tq/2==0
  if fixed or not tq:branch=0
  elif aligned:branch=int(mp.nint(phase/(2*mp.pi)));phase=2*branch*mp.pi;change=phase-phase0
  elif pole:branch=int(mp.nint((phase/mp.pi-1)/2));phase=(2*branch+1)*mp.pi;change=phase-phase0
  else:branch=int(mp.ceil((phase-mp.pi)/(2*mp.pi)))
  if fixed:co,si=val(cs),val(sn)
  elif pole:co,si=mp.mpf(-1),mp.mpf(0)
  else:co,si=mp.cos(phase),mp.sin(phase)
  velocity=mp.mpf(0) if fixed else dd-kk*si;phi=pp+mm*t;theta1=phi+phase/2;theta2=phi-phase/2;turns=0 if disc>=0 else int(mp.sign(dd))*int(mp.floor(spectral*t/(2*mp.pi)))
  # Explicit phase identities avoid numerical branch-cut artifacts in the reference.
  wrapped=mp.mpf(0) if aligned else mp.pi if pole or fixed and bi==0 else phase-2*mp.pi*branch
  rows.append(dict(step=step,time=out(tq),relativeCos=out(co),relativeSin=out(si),relativePhase=out(phase),phaseChange=out(change),wrappedRelativePhase=out(wrapped),phaseBranch={'signedCount':branch},completedRelativeTurns={'signedCount':turns},relativeAngularVelocity=out(velocity),finiteWindowRelativeFrequency=out(change/t) if tq else None,meanPhase=out(mean0+mean*tq),phase1=out(theta1),phase2=out(theta2),cos1=out(mp.cos(theta1)),sin1=out(mp.sin(theta1)),cos2=out(mp.cos(theta2)),sin2=out(mp.sin(theta2)),angularFrequency1=out(mm+velocity/2),angularFrequency2=out(mm-velocity/2),firstHarmonicSquared=out((1+co)/2),firstHarmonic=out(mp.sqrt(max(mp.mpf(0),(1+co)/2))),secondHarmonicSquared=out(co*co)))
 expected=dict(detuning=out(delta),totalCoupling=out(K),edgeCoupling=out(K/2),discriminant=out(disc),meanAngularFrequency=out(mean),initialRelativeCos=out(cs),initialRelativeSin=out(sn),initialRelativePhase=out(phase0),initialRelativeAngularVelocity=out(speed),asymptoticRelativeFrequency=out(asym),localLockingDecayRate=out(spectral) if disc>0 else None,localLockingRelaxationTime=out(1/spectral) if disc>0 else None,driftPeriod=out(2*mp.pi/spectral) if disc<0 else None,history=rows)
 cases.append(dict(label=label,input=p,expected=expected));record_count+=len(rows)
ode_labels={'locking from alignment','locking from the upper cut','drift positive','drift negative','critical from alignment','critical from beyond the bottleneck','critical exact cut at two seconds','reverse critical exact cut','free different frequencies'}
max_ode_error=0.0;ode_checks=0
for case in cases:
 if case['label'] not in ode_labels:continue
 pp=case['input'];dd=pp['omega1Numerator']/pp['omega1Denominator']-pp['omega2Numerator']/pp['omega2Denominator'];kk=pp['couplingNumerator']/pp['couplingDenominator'];rows=case['expected']['history'];times=[float(val(F(row['time']))) for row in rows];start=float(mp.mpf(case['expected']['initialRelativePhase']))
 sol=solve_ivp(lambda time,y:[dd-kk*np.sin(y[0])],(0,times[-1]),[start],t_eval=times,method='DOP853',rtol=2.5e-14,atol=1e-14)
 assert sol.success and len(sol.t)==len(times)
 error=max(abs(float(mp.mpf(row['relativePhase']))-actual) for row,actual in zip(rows,sol.y[0]));assert error<1e-10,(case['label'],error);max_ode_error=max(max_ode_error,error);ode_checks+=1
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'provenance':'Independent220-digit general2x2 matrix exponential; drift lift uses analytic full periods plus a principal argument, not production local-atan partition. Exact rational critical-pole and fixed-phase identities. SymPy verifies the projective Riccati phase equation.180digits stored; no production imports.','cases':cases},indent=2)+'\n');print(json.dumps({'preparations':len(cases),'timeRecords':record_count,'precisionDigits':220,'independentODEChecks':ode_checks,'maxDOP853AbsoluteError':max_ode_error}))
