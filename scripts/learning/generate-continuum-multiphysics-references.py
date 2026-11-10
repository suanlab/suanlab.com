"""Independent physical-coordinate differentiation, matrix exponential, eigensystem and integrals."""
import argparse,json,random
from pathlib import Path
import mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-multiphysics-independent.json');target=Path(parser.parse_args().output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
mp.mp.dps=160;M=mp.mpf;rng=random.Random(261107)
def obs(v):return dict(value=str(mp.re(v)),sign=int(mp.sign(mp.re(v))),logMagnitude=None if mp.re(v)==0 else str(mp.log(abs(mp.re(v)))))
def thermal_case(p,time,phase,mode,integrate=True):
 E,rho,C,T,beta,K,k,A,a0,v0,b0=[M(p[n])for n in ['elasticModulus','density','heatCapacity','referenceTemperature','coupling','conductivity','wavenumber','area','initialDisplacement','initialVelocity','initialTemperatureDeviation']]
 # Physical a,v,b variables, unlike the normalized production system.
 B=mp.matrix([[0,1,0],[-E*k*k/rho,0,beta*k/rho],[0,-T*beta*k/C if mode=='reciprocal'else 0,-K*k*k/C]])
 initial=mp.matrix([a0,v0,b0]);t=M(time);phase=M(phase);cell=2*mp.pi/k;V=A*cell
 eig,R=mp.eig(B);coeff=mp.lu_solve(R,initial)
 def state(s):return initial if s==0 else mp.matrix([mp.re(x)for x in R*mp.matrix([mp.exp(eig[i]*s)*coeff[i]for i in range(3)])])
 direct=mp.expm(B*t)*initial;at=state(t)
 for i in range(3):assert mp.almosteq(direct[i],at[i],abs_eps=mp.mpf('1e-130') if mp.mp.dps==160 else mp.mpf('1e-1100'))
 a,v,b=at;da,dv,db=B*at;x=phase/k
 u=lambda z:a*mp.sin(k*z);vel=lambda z:v*mp.sin(k*z);theta=lambda z:b*mp.cos(k*z);strain=mp.diff(u,x)
 mechanical=lambda s:V/4*(rho*state(s)[1]**2+E*k*k*state(s)[0]**2)
 thermal=lambda s:V*C/(4*T)*state(s)[2]**2
 availability=lambda s:mechanical(s)+thermal(s)
 exchange=lambda s:V*beta*k/2*state(s)[1]*state(s)[2]
 loss=lambda s:V*K*k*k/(2*T)*state(s)[2]**2
 expected=dict(displacementAmplitude=a,velocityAmplitude=v,temperatureAmplitude=b,displacementChange=a-a0,velocityChange=v-v0,temperatureChange=b-b0,accelerationAmplitude=dv,temperatureRateAmplitude=db,
               displacement=u(x),velocity=vel(x),strain=strain,strainRate=mp.diff(vel,x),temperatureDeviation=theta(x),absoluteTemperature=T+theta(x),temperatureGradient=mp.diff(theta,x),heatFlux=-K*mp.diff(theta,x),stress=E*strain-beta*theta(x),mechanicalAvailability=mechanical(t),thermalAvailability=thermal(t),totalAvailability=availability(t),initialAvailability=availability(0),mechanicalExchangeRate=exchange(t),thermalExchangeRate=-exchange(t)if mode=='reciprocal'else M(0),conductionAvailabilityLoss=loss(t),totalAvailabilityRate=-loss(t)if mode=='reciprocal'else exchange(t)-loss(t))
 out=dict(parameters=p,time=time,phase=float(phase),mode=mode,expected={n:obs(v)for n,v in expected.items()})
 if integrate:
  ex=mp.quad(exchange,[0,t]);diss=mp.quad(loss,[0,t]);change=availability(t)-availability(0)
  assert mp.almosteq(change,(-diss if mode=='reciprocal'else ex-diss),abs_eps=mp.mpf('1e-125'))
  mech_integral=A*mp.quad(lambda z:rho*vel(z)**2/2+E*mp.diff(u,z)**2/2,[0,cell/2,cell]);heat_integral=A*mp.quad(lambda z:C*theta(z)**2/(2*T),[0,cell/2,cell])
  assert mp.almosteq(mech_integral,mechanical(t),rel_eps=M('1e-125'),abs_eps=M('1e-140'));assert mp.almosteq(heat_integral,thermal(t),rel_eps=M('1e-125'),abs_eps=M('1e-140'))
  out['window']={n:obs(v)for n,v in dict(mechanicalExchange=ex,conductionAvailabilityLoss=diss,balanceDerivedAvailabilityChange=change).items()}
 return out

def electric_case(p,xi,integrate=True):
 L,A,r,K,S,TL,TR,j,reference=[M(p[n])for n in ['length','area','resistivity','conductivity','seebeck','leftTemperature','rightTemperature','currentDensity','referencePotential']];x=M(xi)*L;delta=TR-TL
 fieldT=lambda z:TL+delta*z/L+r*j*j*z*(L-z)/(2*K)
 voltage=lambda z:reference-r*j*z-S*(fieldT(z)-TL)
 E=lambda z:-mp.diff(voltage,z);heat=lambda z:S*fieldT(z)*j-K*mp.diff(fieldT,z)
 dv=voltage(0)-voltage(L);power=A*j*dv;left=A*heat(0);right=A*heat(L)
 expected=dict(position=x,current=A*j,currentDensity=j,resistance=r*L/A,temperature=fieldT(x),stableJouleTemperatureRise=r*j*j*x*(L-x)/(2*K),temperatureGradient=mp.diff(fieldT,x),temperatureCurvature=mp.diff(fieldT,x,2),electricField=E(x),potential=voltage(x),stablePotentialChange=voltage(x)-reference,terminalVoltageDrop=dv,peltierHeatFlux=S*fieldT(x)*j,fourierHeatFlux=-K*mp.diff(fieldT,x),totalHeatFlux=heat(x),localJouleHeating=r*j*j,localElectricalWork=j*E(x),heatFluxGradient=mp.diff(heat,x),leftHeatInput=left,rightHeatOutput=right,electricalPowerInput=power,joulePower=A*r*j*j*L,reversibleElectricalPower=A*S*j*delta,stableBoundaryHeatDifference=right-left,partialElectricalPower=A*j*(voltage(0)-voltage(x)),partialJoulePower=A*r*j*j*x)
 out=dict(parameters=p,xi=xi,expected={n:obs(v)for n,v in expected.items()})
 if left>0 and power>0:out['cop']=obs(left/power)
 if integrate:
  work=A*mp.quad(lambda z:j*E(z),[0,L/2,L]);joule=A*mp.quad(lambda z:r*j*j,[0,L/2,L]);assert mp.almosteq(work,right-left,rel_eps=M('1e-125'),abs_eps=M('1e-140'));assert mp.almosteq(work,power,rel_eps=M('1e-125'),abs_eps=M('1e-140'))
  out['integrals']=dict(electricalWork=obs(work),jouleHeating=obs(joule),partialWork=obs(A*mp.quad(lambda z:j*E(z),[0,x])))
 return out
thermal=[];electric=[]
for i in range(24):
 p=dict(elasticModulus=rng.uniform(.6,3.5),density=rng.uniform(.6,1.8),heatCapacity=rng.uniform(.6,1.8),referenceTemperature=rng.uniform(260,340),coupling=rng.uniform(.005,.045),conductivity=rng.uniform(.02,.5),wavenumber=rng.uniform(.3,1.5),area=rng.uniform(.3,3),initialDisplacement=rng.uniform(-.0008,.0008),initialVelocity=rng.uniform(-.0008,.0008),initialTemperatureDeviation=rng.uniform(-.018,.018))
 thermal.append(thermal_case(p,rng.uniform(.1,1.5),rng.uniform(.1,6), 'reciprocal'if i%2==0 else'one-way'))
 p=dict(length=rng.uniform(.3,3),area=rng.uniform(.3,3),resistivity=rng.uniform(.02,2),conductivity=rng.uniform(.2,3),seebeck=rng.uniform(-.04,.04),leftTemperature=rng.uniform(260,340),rightTemperature=rng.uniform(260,340),currentDensity=rng.uniform(-4,4),referencePotential=rng.uniform(-3,3))
 electric.append(electric_case(p,rng.uniform(.05,.95)))
thermal_limits=[];electric_limits=[]
with mp.workdps(1200):
 p=dict(elasticModulus=2.,density=1.,heatCapacity=1.,referenceTemperature=300.,coupling=.02,conductivity=.1,wavenumber=1.,area=1.,initialDisplacement=.001,initialVelocity=.001,initialTemperatureDeviation=.02)
 for changes,t,mode in [({},0.,'reciprocal'),({'coupling':0.},1.,'reciprocal'),({'conductivity':0.},1.,'reciprocal'),({'initialDisplacement':0.,'initialTemperatureDeviation':0.},1e-20,'reciprocal'),({'initialDisplacement':0.,'initialTemperatureDeviation':0.},5e-324,'reciprocal'),({'coupling':5e-324,'initialTemperatureDeviation':0.},1.,'reciprocal'),({'initialDisplacement':1e-200,'initialVelocity':0.,'initialTemperatureDeviation':0.},1.,'reciprocal'),({'initialDisplacement':0.,'initialVelocity':0.,'initialTemperatureDeviation':5e-324},1.,'one-way')]:
  thermal_limits.append(thermal_case(dict(p,**changes),t,0.,mode,False))
 p=dict(length=1.,area=1.,resistivity=1.,conductivity=1.,seebeck=.01,leftTemperature=300.,rightTemperature=310.,currentDensity=.9,referencePotential=0.)
 for changes,xi in [({'currentDensity':0.},.5),({'seebeck':0.,'currentDensity':5e-324,'rightTemperature':300.},.5),({'currentDensity':1e-200,'rightTemperature':300.},.5),({'seebeck':5e-324,'currentDensity':5e-324},.5),({'resistivity':.01,'conductivity':.1,'seebeck':.05,'currentDensity':10.},.5),({'currentDensity':-.05},.5),({'referencePotential':3.},0.),({},1.)]:
  electric_limits.append(electric_case(dict(p,**changes),xi,False))
teaching=[]
base=dict(elasticModulus=2.,density=1.,heatCapacity=1.,referenceTemperature=300.,coupling=.02,conductivity=.1,wavenumber=1.,area=1.,initialDisplacement=.001,initialVelocity=.001,initialTemperatureDeviation=.02)
for changes,time,phase,mode in [({},1.,.7,'reciprocal'),({'conductivity':0.},1.,.3,'reciprocal'),({'coupling':0.},2.,1.,'reciprocal'),({'initialTemperatureDeviation':-.02},1.,.5,'reciprocal'),({},1.,.7,'one-way'),({'elasticModulus':4.,'coupling':.05,'conductivity':1.,'wavenumber':2.},4.,1.,'reciprocal'),({'initialDisplacement':0.,'initialTemperatureDeviation':0.},1e-20,.7,'reciprocal'),({'coupling':0.,'conductivity':1.,'wavenumber':2.},4.,1.,'reciprocal')]:
 teaching.append(thermal_case(dict(base,**changes),time,phase,mode))
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(dict(method='physical-variable matrix exponential and independent eigensystem; physical-space derivatives and quadrature; time quadrature',ordinaryDigits=160,limitDigits=1200,thermalCases=thermal,electricCases=electric,thermalLimits=thermal_limits,electricLimits=electric_limits,teachingCases=teaching),indent=2)+'\n');print('Wrote72 independent multiphysics references to',target)
