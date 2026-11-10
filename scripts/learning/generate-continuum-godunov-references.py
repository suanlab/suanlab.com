"""Independent high-precision Burgers oracle.

Godunov uses min/max of the convex physical flux; Roe uses the central flux
minus its viscosity. The production code selects traces and factors changes.
Exact cell moments and integrated continuum boundary fluxes are independent.
"""
import argparse, json, random, math
from pathlib import Path
import mpmath as m
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-godunov-independent.json');target=Path(parser.parse_args().output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
def obs(x):
 x=m.mpf(x);return dict(value=m.nstr(x,70),sign=int(m.sign(x)),logMagnitude=None if x==0 else m.nstr(m.log(abs(x)),70))
def flux(a,b,method):
 f=lambda q:q*q/2
 if method=='roe':return (f(a)+f(b))/2-abs(a+b)*(b-a)/4
 return (m.mpf(0)if a<=0<=b else min(f(a),f(b)))if a<=b else max(f(a),f(b))
def face(a,b,method,stable=False):
 F=flux(a,b,method);D=(b**3-a**3)/6-(b-a)*F;H=(a+b)*F/2-(a**3+b**3)/12
 if stable:
  jump=b-a
  D=(b**3-a**3)/6 if method=='godunov' and a<0<b else jump**2*(abs(a+b)/4-jump/12)
 return F,D,H

def moments(j,left,right,t,dx):
 lo=m.mpf(j);hi=lo+1
 if t==0 or left==right:return left if j<0 else right,(left if j<0 else right)**2/2
 if left>right:
  front=(left+right)*t/(2*dx);wl=max(m.mpf(0),min(hi,front)-lo);return left*wl+right*(1-wl),(left**2*wl+right**2*(1-wl))/2
 a=left*t/dx;b=right*t/dx;wl=max(m.mpf(0),min(hi,a)-lo);wr=max(m.mpf(0),hi-max(lo,b));u=max(lo,a);v=min(hi,b)
 fan1=dx/t*(v*v-u*u)/2 if v>u else m.mpf(0)
 fan2=(dx/t)**2*(v**3-u**3)/6 if v>u else m.mpf(0)
 return left*wl+right*wr+fan1,left**2*wl/2+right**2*wr/2+fan2

def reconstruction_error(j,q,left,right,t,dx):
 lo=m.mpf(j);hi=lo+1
 if not t or left==right:return dx*abs(q-(left if j<0 else right))
 if left>right:
  front=(left+right)*t/(2*dx);wl=max(m.mpf(0),min(hi,front)-lo)
  return dx*(wl*abs(q-left)+(1-wl)*abs(q-right))
 a=left*t/dx;b=right*t/dx;wl=max(m.mpf(0),min(hi,a)-lo);wr=max(m.mpf(0),hi-max(lo,b));u=max(lo,a);v=min(hi,b);fan=m.mpf(0)
 if v>u:
  da=q-u*dx/t;db=q-v*dx/t
  average=(abs(da)+abs(db))/2 if da*db>=0 else (da*da+db*db)/(2*abs(da-db))
  fan=(v-u)*average
 return dx*(wl*abs(q-left)+wr*abs(q-right)+fan)

def integrated_boundary(y,left,right,T,power):
 if not T:return m.mpf(0)
 speeds=[(left+right)/2]if left>right else[left,right]
 cuts=sorted(set([m.mpf(0),T]+[y/v for v in speeds if v and 0<y/v<T]))
 result=m.mpf(0)
 for a,b in zip(cuts,cuts[1:]):
  middle=(a+b)/2;ray=y/middle
  if left>right:q=left if ray<(left+right)/2 else right
  else:q=left if ray<left else right if ray>right else ray
  if left<right and left<ray<right:
   assert a>0
   result+=y**power/(power*(power-1))*(a**(1-power)-b**(1-power))
  else:result+=(b-a)*q**power/power
 return result

def run(p):
 N=2**p['cellCountExponent'];L=m.mpf(p['length']);dx=L/N;cut=int(N*p['interfaceFraction']);left=m.mpf(p['leftState']);right=m.mpf(p['rightState']);dt=m.mpf(p['timeStep']);steps=p['steps'];initial=[left if i<cut else right for i in range(N)];mass0=dx*sum(initial);entropy0=dx*sum(q*q/2 for q in initial);T=steps*dt
 exact=[moments(i-cut,left,right,T,dx)for i in range(N)]
 exactMassChange=dx*sum(q-initial[i]for i,(q,e)in enumerate(exact));exactEntropyChange=dx*sum(e-initial[i]**2/2 for i,(q,e)in enumerate(exact))
 massBoundary=integrated_boundary(-cut*dx,left,right,T,2)-integrated_boundary((N-cut)*dx,left,right,T,2)
 entropyBoundary=integrated_boundary(-cut*dx,left,right,T,3)-integrated_boundary((N-cut)*dx,left,right,T,3)
 shockTime=T
 if left>right:
  speed=(left+right)/2
  if speed>0:shockTime=min(T,(N-cut)*dx/speed)
  elif speed<0:shockTime=min(T,-cut*dx/speed)
  entropyBoundary-=(left-right)**3*shockTime/12
 scale=max(m.mpf(1),abs(mass0),abs(entropy0));tol=scale*m.mpf(10)**(-m.mp.dps+40)
 assert abs(massBoundary-exactMassChange)<tol
 assert abs(entropyBoundary-exactEntropyChange)<tol,(p,entropyBoundary,exactEntropyChange)
 result=dict(parameters=p,exactMassChange=obs(exactMassChange),exactContinuumEntropyChange=obs(exactEntropyChange),integratedContinuumBoundaryMass=obs(massBoundary),continuumEntropyBudget=obs(entropyBoundary))
 for method in ['godunov','roe']:
  q=initial.copy();massLedger=m.mpf(0);boundaryEntropy=m.mpf(0);spatial=m.mpf(0);positive=m.mpf(0);allCfl=True;allLocal=True;allFaces=True
  for step in range(steps+1):
   t=dt*step;gl=moments(-cut-1,left,right,t,dx)[0];gr=moments(N-cut,left,right,t,dx)[0];extended=[gl,*q,gr];faces=[face(a,b,method,stable=True)for a,b in zip(extended,extended[1:])];rates=[(faces[i][0]-faces[i+1][0])/dx for i in range(N)]
   local=[]
   for i in range(N):
    F,D,H=faces[i];Fr,Dr,Hr=faces[i+1];inc=dt*rates[i];timeCorrection=dt**2*dx*rates[i]**2/2;entropyChange=dx*((q[i]+inc)**2-q[i]**2)/2;residual=entropyChange+dt*(Hr-H);margin=D+Dr-dt*dx*rates[i]**2
    local.append(dict(entropyChange=obs(entropyChange),positiveEulerCorrection=obs(timeCorrection),entropyResidualFromFaces=obs(-dt*(D+Dr)/2+timeCorrection),sufficientMargin=obs(margin)))
   cfl=dt/dx*max(map(abs,extended))
   if step==steps:break
   allCfl &= cfl<=1;allLocal &= all(m.mpf(v['sufficientMargin']['value'])>=0 for v in local);allFaces &= all(f[1]>=0 for f in faces)
   massLedger+=dt*(faces[0][0]-faces[-1][0]);boundaryEntropy+=dt*(faces[0][2]-faces[-1][2]);spatial-=dt/2*sum(faces[i][1]+faces[i+1][1]for i in range(N));positive+=dt**2*dx*sum(r*r for r in rates)/2
   q=[value+dt*r for value,r in zip(q,rates)]
  changes=[q[i]-initial[i]for i in range(N)];entropyChange=dx*sum(q[i]**2-initial[i]**2 for i in range(N))/2;massChange=dx*sum(changes);errors=[q[i]-exact[i][0]for i in range(N)];condition=max(m.mpf(1),dx*sum(map(abs,q)),abs(boundaryEntropy),abs(spatial),abs(positive))
  assert abs(massChange-massLedger)<tol*condition
  assert abs(entropyChange-boundaryEntropy-spatial-positive)<tol*condition
  summary=dict(massFromBoundaryLedger=mass0+massChange,massChangeFromBoundary=massChange,assembledMass=mass0+massChange,assembledMassChange=massChange,absoluteCellMassSum=dx*sum(map(abs,q)),quadraticEntropy=entropy0+entropyChange,stableEntropyChange=entropyChange,accumulatedEntropyChangeFromCells=entropyChange,boundaryEntropyLedger=boundaryEntropy,spatialEntropyLedger=spatial,positiveEulerLedger=positive,entropyChangeFromFaceLedger=entropyChange,exactMassChange=exactMassChange,exactContinuumEntropyChange=exactEntropyChange,l1CellAverageError=dx*sum(map(abs,errors)),l1ReconstructedProfileError=sum(reconstruction_error(i-cut,q[i],left,right,T,dx)for i in range(N)),linfinityCellAverageError=max(map(abs,errors)))
  if dt==0 or steps==0:
   for key in ['massChangeFromBoundary','assembledMassChange','stableEntropyChange','accumulatedEntropyChangeFromCells','boundaryEntropyLedger','spatialEntropyLedger','positiveEulerLedger','entropyChangeFromFaceLedger','exactMassChange','exactContinuumEntropyChange','l1CellAverageError','l1ReconstructedProfileError','linfinityCellAverageError']:summary[key]=m.mpf(0)
  result[method]=dict(cells=[dict(value=obs(q[i]),change=obs(changes[i]),exactValue=obs(exact[i][0]),exactChange=obs(exact[i][0]-initial[i]),error=obs(errors[i]))for i in range(N)],final={k:obs(v)for k,v in summary.items()},nextStep=dict(faces=[dict(flux=obs(f),entropyDefect=obs(d),numericalEntropyFlux=obs(h))for f,d,h in faces],rates=list(map(obs,rates)),local=local,cfl=obs(cfl)),allExecutedCflAtMostOne=allCfl,allExecutedLocalEntropySufficient=allLocal,allExecutedFacesSemidiscreteAdmissible=allFaces)
 return result
base=dict(cellCountExponent=3,length=2.,leftState=1.5,rightState=.5,interfaceFraction=.5,timeStep=.01,steps=12)
params=[base,{**base,'leftState':1.,'rightState':-1.},{**base,'leftState':-1.,'rightState':1.},{**base,'leftState':-.5,'rightState':-1.5},{**base,'leftState':.5,'rightState':1.5},{**base,'leftState':-1.5,'rightState':-.5},{**base,'rightState':1.5},{**base,'leftState':0.,'rightState':0.},{**base,'timeStep':0.},{**base,'steps':0},{**base,'leftState':1.,'rightState':.5,'timeStep':.25,'steps':16},{**base,'leftState':-.5,'rightState':-1.,'timeStep':.25,'steps':16},{**base,'leftState':-2.,'rightState':-.5,'timeStep':.1,'steps':32},{**base,'leftState':.5,'rightState':2.,'timeStep':.1,'steps':32},{**base,'leftState':2.,'rightState':0.,'timeStep':.25,'steps':12}]
rng=random.Random(261111)
for i in range(9):params.append({**base,'cellCountExponent':3+i%2,'length':rng.uniform(.5,4),'leftState':rng.uniform(-2,2),'rightState':rng.uniform(-2,2),'interfaceFraction':[.25,.5,.75][i%3],'timeStep':rng.uniform(.001,.005),'steps':i+1})
tiny=float.fromhex('0x0.0000000000001p-1022')
limits=[{**base,'timeStep':tiny},{**base,'leftState':tiny,'rightState':0.},{**base,'leftState':-tiny,'rightState':tiny},{**base,'leftState':tiny,'rightState':-tiny},{**base,'leftState':tiny,'rightState':0.,'timeStep':tiny},{**base,'leftState':-tiny,'rightState':tiny,'timeStep':tiny},{**base,'leftState':1e-200,'rightState':-1e-200},{**base,'leftState':-1e-200,'rightState':1e-200,'timeStep':tiny},{**base,'leftState':1.,'rightState':math.nextafter(1.,2.)},{**base,'leftState':1e-200,'rightState':math.nextafter(1e-200,2.)},{**base,'steps':0,'timeStep':tiny},{**base,'leftState':-tiny,'rightState':0.}]
result=dict(method='independent convex-flux extrema/central Roe flux, high-precision actual Euler updates, analytic piecewise cell moments and time-integrated continuum boundary fluxes',ordinaryDigits=160,limitDigits=1600,outputDigits=70,ordinaryCases=[],limitCases=[])
for digits,inputs,key in [(160,params,'ordinaryCases'),(1600,limits,'limitCases')]:
 m.mp.dps=digits
 for i,p in enumerate(inputs):result[key].append(run(p));print(f'{key} {i+1}/{len(inputs)}',flush=True)
m.mp.dps=1600;interfaces=[]
for a in [-2.,-1.,-.5,0.,.5,1.,2.,tiny,-tiny]:
 for b in [-2.,-1.,-.5,0.,.5,1.,2.,tiny,-tiny]:
  for method in ['godunov','roe']:
   F,D,H=face(m.mpf(a),m.mpf(b),method);interfaces.append(dict(left=a,right=b,method=method,flux=obs(F),entropyDefect=obs(D),numericalEntropyFlux=obs(H)))
result['interfaces']=interfaces
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n');print(f'Wrote {target}:24 ordinary,12 limits,162 interface references.',flush=True)
