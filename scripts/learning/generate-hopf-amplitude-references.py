#!/usr/bin/env python3
"""Independent reciprocal-amplitude matrix exponential,220digits; no product imports."""
import argparse,json
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
import sympy as sp
mp.mp.dps=220
base=dict(muNumerator=-1,muDenominator=1,gNumerator=1,gDenominator=1,shearNumerator=0,shearDenominator=1,omegaNumerator=2,omegaDenominator=1,xNumerator=1,xDenominator=1,yNumerator=0,yDenominator=1,timeNumerator=4,timeDenominator=1,sampleSteps=8)
specs=[('below onset',{}),('critical',dict(muNumerator=0)),('on periodic circle',dict(muNumerator=1)),('inside attracting circle',dict(muNumerator=1,xDenominator=2)),('outside attracting circle',dict(muNumerator=1,xNumerator=2)),('critical with shear',dict(muNumerator=0,shearNumerator=3)),('positive mu with shear',dict(muNumerator=1,shearNumerator=2,xDenominator=2)),('circle of equilibria',dict(muNumerator=1,shearNumerator=2)),('reverse rotation on circle',dict(muNumerator=1,shearNumerator=3)),('negative shear',dict(muNumerator=1,shearNumerator=-2)),('circle from two nonzero coordinates',dict(muNumerator=2,xNumerator=1,yNumerator=1)),('zero frequency at onset',dict(muNumerator=0,omegaNumerator=0)),('zero omega nonlinear rotation',dict(muNumerator=1,omegaNumerator=0,shearNumerator=1)),('origin below onset',dict(xNumerator=0)),('origin at onset',dict(muNumerator=0,xNumerator=0)),('origin above onset',dict(muNumerator=1,xNumerator=0)),('near onset below',dict(muNumerator=-1,muDenominator=1000000000)),('near onset above',dict(muNumerator=1,muDenominator=1000000000)),('negative initial coordinates',dict(xNumerator=-1,yNumerator=-1,shearNumerator=1)),('imaginary initial coordinate',dict(xNumerator=0,yNumerator=-1)),('rational coefficients',dict(muNumerator=2,muDenominator=3,gNumerator=5,gDenominator=7,shearNumerator=-2,shearDenominator=3,omegaNumerator=5,omegaDenominator=2,xNumerator=2,xDenominator=3,yNumerator=1,yDenominator=5,timeNumerator=1,timeDenominator=3,sampleSteps=13)),('zero time',dict(timeNumerator=0,muNumerator=1,shearNumerator=1)),('dense records',dict(sampleSteps=64)),('tiny initial amplitude',dict(muNumerator=0,xDenominator=1000000000)),('tiny initial growth',dict(muNumerator=8,xDenominator=1000000000)),('weak saturation huge accumulated phase',dict(muNumerator=8,gDenominator=1000000000,shearNumerator=8,omegaNumerator=-8,xNumerator=4,yNumerator=4)),('strong damping maximum initial radius',dict(muNumerator=-8,gNumerator=8,shearNumerator=8,omegaNumerator=-8,xNumerator=4,yNumerator=-4)),('weak saturation critical',dict(muNumerator=0,gDenominator=1000000000,shearNumerator=8)),('near invariant circle',dict(muNumerator=1,xNumerator=1000000000,xDenominator=999999999)),('critical squared amplitude half time',dict(muNumerator=0,timeNumerator=1,timeDenominator=2)),('critical amplitude half time',dict(muNumerator=0,timeNumerator=3,timeDenominator=2)),('nonreduced equivalent values',dict(muNumerator=-2,muDenominator=2,gNumerator=2,gDenominator=2,omegaNumerator=4,omegaDenominator=2,xNumerator=2,xDenominator=2,timeNumerator=8,timeDenominator=2)),('slow circle radial rate',dict(muNumerator=1,muDenominator=1000000000,gDenominator=1000000000)),('critical tiny initial weak saturation',dict(muNumerator=0,xDenominator=1000000000,gDenominator=1000000000,shearNumerator=8))]
def number(x):return mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(x)
def out(x):return str(x) if isinstance(x,F) else mp.nstr(x,180)
u,m,g,t=sp.symbols('u m g t',real=True);expr=u*sp.exp(-2*m*t)+g/m*(1-sp.exp(-2*m*t));assert sp.simplify(sp.diff(expr,t)+2*m*expr-2*g)==0
specs.append(('worked critical shear',dict(muNumerator=0,shearNumerator=1)))
xx,yy,ww,cc=sp.symbols('x y omega shear',real=True)
rho=xx**2+yy**2;dx=(m-g*rho)*xx-(ww-cc*rho)*yy;dy=(ww-cc*rho)*xx+(m-g*rho)*yy
assert sp.expand(2*xx*dx+2*yy*dy-(2*m*rho-2*g*rho**2))==0
assert sp.expand(xx*dy-yy*dx-(ww-cc*rho)*rho)==0
cases=[];records=0
for label,change in specs:
 p={**base,**change};ratio=lambda k:F(p[k+'Numerator'],p[k+'Denominator']);mu,gg,c,w,x0,y0,T=[ratio(k) for k in ['mu','g','shear','omega','x','y','time']];rho0=x0*x0+y0*y0
 history=[]
 for step in range(p['sampleSteps']+1):
  tq=T*F(step,p['sampleSteps']);tt=number(tq);mm,gm,cm,wm=map(number,[mu,gg,c,w]);r0=number(rho0)
  if rho0:
   lift=mp.expm(mp.matrix([[-2*mm,2*gm],[0,0]])*tt)*mp.matrix([1/r0,1]);rho=1/lift[0]
   A=(mm*tt-mp.log(rho/r0)/2)/gm
   phi=wm*tt-cm*A;gain=mp.sqrt(rho/r0);x=gain*(number(x0)*mp.cos(phi)-number(y0)*mp.sin(phi));y=gain*(number(x0)*mp.sin(phi)+number(y0)*mp.cos(phi));sensitivity=(rho/r0)**2*mp.exp(-2*mm*tt)
  else:rho=A=x=y=mp.mpf(0);phi=None;sensitivity=mp.exp(2*mm*tt)
  history.append(dict(step=step,time=out(tq),squaredAmplitude=out(rho),amplitude=out(mp.sqrt(rho)),integratedSquaredAmplitude=out(A),phaseShift=out(phi) if phi is not None else None,instantaneousAngularFrequency=out(wm-cm*rho) if rho0 else None,x=out(x),y=out(y),squaredAmplitudeDerivative=out(2*rho*(mm-gm*rho)),sensitivityToInitialSquaredAmplitude=out(sensitivity)))
 records+=len(history)
 expected={'initialSquaredAmplitude':out(rho0),'originLinearAmplitudeGrowthRate':out(mu),'originLinearSquaredAmplitudeGrowthRate':out(2*mu),'attractingCircleSquaredAmplitude':out(mu/gg) if mu>0 else None,'attractingCircleAngularFrequency':out(w-c*mu/gg) if mu>0 else None,'attractingCircleRadialDecayRate':out(2*mu) if mu>0 else None,'originLinearAmplitudeRelaxationTime':out(-1/mu) if mu<0 else None,'attractingCircleRadialRelaxationTime':out(1/(2*mu)) if mu>0 else None,'criticalSquaredAmplitudeHalfTime':out(1/(2*gg*rho0)) if not mu and rho0 else None,'criticalAmplitudeHalfTime':out(3/(2*gg*rho0)) if not mu and rho0 else None,'periodDividedByTwoPi':out(1/abs(w-c*mu/gg)) if mu>0 and w-c*mu/gg else None,'history':history}
 cases.append(dict(label=label,input=p,expected=expected))
rotations=[dict(n=str(n),d=str(d),sin=out(mp.sin(mp.mpf(n)/d)),cos=out(mp.cos(mp.mpf(n)/d))) for n,d in [(0,1),(1,3),(-1,3),(16,1),(2**48,1),(-2**48,1),(3*2**48-1,3)]]
args=argparse.ArgumentParser();args.add_argument('--output',required=True);a=args.parse_args();path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'provenance':'Independent reciprocal squared-amplitude affine2x2 matrix exponential at220digits; Cartesian reconstruction and initial-amplitude sensitivity;180digits stored. SymPy affine-flow differential identity checked. No production modules imported.','cases':cases,'rotationReferences':rotations},indent=2)+'\n');print(json.dumps({'preparations':len(cases),'timeRecords':records,'precisionDigits':220}))
