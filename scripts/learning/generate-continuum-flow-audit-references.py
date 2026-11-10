"""1600-digit point fields, derivative observations and physical cell budgets.
No product imports; energy is formed from velocity/pressure, derivatives by diff.
"""
import argparse,json,math,random
from pathlib import Path
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('--output',default='scripts/fixtures/continuum-flow-audit-independent.json');out=Path(p.parse_args().output)
if out.exists():raise SystemExit('Refusing to overwrite references')
m.mp.dps=1600;R=random.Random(20261107);tiny=float.fromhex('0x0.0000000000001p-1022')
def obs(x):
 x=m.mpf(x);v=float(x)
 return dict(value=v if x==0 or v else None,sign=int(m.sign(x)))
base=dict(density=.25,soundSpeed=.25,meanVelocity=1.,gamma=1.4,waveNumber=1.,rightAmplitude=tiny,leftAmplitude=0.,contactAmplitude=0.,phase=0.)
acoustic=[];params=[]
for a in [tiny,4*tiny,1e-200,1e-162,.01,0.]:
 for kind in ['right','contact','opposite']:
  p={**base,'rightAmplitude':0. if kind=='contact' else a,'leftAmplitude':a if kind=='opposite' else 0.,'contactAmplitude':a if kind=='contact' else 0.};params.append((p,0.,0.))
for _ in range(12):params.append((dict(density=R.uniform(.25,4),soundSpeed=R.uniform(.25,4),meanVelocity=R.uniform(-4,4),gamma=R.uniform(1.1,1.67),waveNumber=R.uniform(.25,4),rightAmplitude=R.uniform(-.03,.03),leftAmplitude=R.uniform(-.03,.03),contactAmplitude=R.uniform(-.03,.03),phase=R.uniform(-3,3)),R.uniform(-2,2),R.uniform(0,4)))
for p,x,t in params:
 r,c,U,g,k,ap,am,ae,phi=[m.mpf(p[key])for key in ['density','soundSpeed','meanVelocity','gamma','waveNumber','rightAmplitude','leftAmplitude','contactAmplitude','phase']];X,T=m.mpf(x),m.mpf(t)
 def modes(x,t):return [a*m.cos(k*(x-speed*t)+phi) for a,speed in zip([ap,am,ae],[U+c,U-c,U])]
 def field(x,t):
  A,B,C=modes(x,t);return [r*(A-B+C),c*(A+B),r*c*c*(A-B)]
 v=field(X,T);dx=[m.diff(lambda xx:field(xx,T)[i],X)for i in range(3)];dt=[m.diff(lambda tt:field(X,tt)[i],T)for i in range(3)]
 energy=r*v[1]**2/2+v[2]**2/(2*r*c*c)
 values=dict(zip(['perturbation.density','perturbation.velocity','perturbation.pressure'],v));values.update(zip(['density','velocity','pressure'],[r+v[0],U+v[1],r*c*c/g+v[2]]));values.update({f'dx.{key}':value for key,value in zip(['density','velocity','pressure'],dx)});values.update({f'dt.{key}':value for key,value in zip(['density','velocity','pressure'],dt)})
 values.update(contactVariable=v[2]-c*c*v[0],rightInvariant=v[1]+v[2]/(r*c),leftInvariant=v[1]-v[2]/(r*c),acousticEnergy=energy,energyFlux=U*energy+v[2]*v[1])
 values.update({'nonlinearResidual.mass':v[1]*dx[0]+v[0]*dx[1],'nonlinearResidual.velocity':v[1]*dx[1]+(1/(r+v[0])-1/r)*dx[2],'nonlinearResidual.pressure':v[1]*dx[2]+g*v[2]*dx[1]})
 acoustic.append(dict(parameters=p,x=x,time=t,expected={key:obs(value)for key,value in values.items()}))
vbase=dict(density=1.,amplitude=4.,waveNumber=.5,viscosity=tiny,pressureOffset=1.,depth=1.)
params=[({**vbase,'amplitude':a},t,.4,.8)for a in [0.,1.,4.,1e-162,1e-200,tiny]for t in [0.,1.,3.]]
params += [({**vbase,'amplitude':1.,'waveNumber':1.,'viscosity':1e-200},1e-200,.4,.8)]
params += [({**vbase,'density':4.,'amplitude':4.,'waveNumber':.25,'depth':2.},.25,.4,.8)]
params += [(dict(density=R.uniform(.25,4),amplitude=R.uniform(0,4),waveNumber=R.uniform(.25,4),viscosity=R.uniform(0,1),pressureOffset=R.uniform(-4,4),depth=R.uniform(.25,2)),R.uniform(0,4),R.uniform(-1,1),R.uniform(-1,1))for _ in range(12)]
vortex=[]
for p,t,x,y in params:
 r,A,k,nu,offset,depth=[m.mpf(p[key])for key in ['density','amplitude','waveNumber','viscosity','pressureOffset','depth']];T,X,Y=map(m.mpf,[t,x,y]);z=2*nu*k*k*T
 def vel(x,y,t):
  amp=A*m.exp(-2*nu*k*k*t);return[-amp*m.cos(k*x)*m.sin(k*y),amp*m.sin(k*x)*m.cos(k*y)]
 def pressure(x,y,t):return offset-r*A*A*m.exp(-4*nu*k*k*t)*(m.cos(2*k*x)+m.cos(2*k*y))/4
 v=vel(X,Y,T);grad=[[m.diff(lambda xx:vel(xx,Y,T)[i],X),m.diff(lambda yy:vel(X,yy,T)[i],Y)]for i in range(2)];vt=[m.diff(lambda tt:vel(X,Y,tt)[i],T)for i in range(2)];lap=[m.diff(lambda xx:vel(xx,Y,T)[i],X,2)+m.diff(lambda yy:vel(X,yy,T)[i],Y,2)for i in range(2)];pg=[m.diff(lambda xx:pressure(xx,Y,T),X),m.diff(lambda yy:pressure(X,yy,T),Y)];conv=[sum(v[j]*grad[i][j]for j in range(2))for i in range(2)]
 # Geometry is the explicitly declared ordinary binary64 side/volume, not a new cell.
 side=2*math.pi/p['waveNumber'];volume=side*side*p['depth'];initial=r*A*A*m.mpf(volume)/4;E=initial*m.exp(-2*z)
 values={'pressure':pressure(X,Y,T),'pressureChange':pressure(X,Y,T)-offset,'kineticEnergyDensity':r*sum(a*a for a in v)/2,'localDissipation':2*r*nu*(grad[0][0]**2+grad[1][1]**2+(grad[0][1]+grad[1][0])**2/2),'decay.stableDecayChange':m.expm1(-z),'decay.stableEnergyChange':m.expm1(-2*z),'decay.amplitudeChange':A*m.expm1(-z),'cell.meanKineticEnergy':r*A*A*m.exp(-2*z)/4,'cell.kineticEnergy':E,'cell.stableEnergyChange':initial*m.expm1(-2*z),'cell.dissipation':4*nu*k*k*E}
 for key,vs in [('velocity',v),('timeDerivative',vt),('laplacian',lap),('pressureGradient',pg),('convection',conv),('pressureAcceleration',[-q/r for q in pg])]:values.update({f'{key}.{i}':q for i,q in enumerate(vs)})
 vortex.append(dict(parameters=p,time=t,x=x,y=y,expected={key:obs(value)for key,value in values.items()}))
out.write_text(json.dumps(dict(method='mpmath1600: direct fields, numerical differentiation, velocity/pressure energy; declared binary64 cell geometry',acousticCases=acoustic,vortexCases=vortex),indent=2)+'\n');print(f'Wrote {len(acoustic)} acoustic and {len(vortex)} vortex references')
