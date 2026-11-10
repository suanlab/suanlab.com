"""Independent complex convolution primitive, differentiation, physical quadrature and matrix mechanics."""
import argparse,json,random
from pathlib import Path
import mpmath as mp
mp.mp.dps=160
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-constitutive-independent.json');path=Path(parser.parse_args().output)
if path.exists():raise SystemExit('Refusing to overwrite an existing reference')
M=mp.mpf;rng=random.Random(261109)
def serial(v):
 if isinstance(v,dict):return{k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return[serial(x)for x in v]
 return float(v)
def signed(v):return dict(value=str(v),sign=int(mp.sign(v)),logMagnitude=None if v==0 else str(mp.log(abs(v))))
def strain(p,t):return M(p['strainAmplitude'])*mp.sin(M(p['angularFrequency'])*t)
def maxwell(p,t):
 e,eta,amp,w,e0=[M(p[k])for k in ['elasticModulus','viscosity','strainAmplitude','angularFrequency','initialElasticStrain']];a=e/eta
 return e*e0*mp.exp(-a*t)+e*amp*w*mp.re(mp.exp(-a*t)*mp.expm1((a+1j*w)*t)/(a+1j*w))
def state(p,t):
 e,eta,amp,w,e0=[M(p[k])for k in ['elasticModulus','viscosity','strainAmplitude','angularFrequency','initialElasticStrain']];eps=lambda s:strain(p,s);rate=mp.diff(eps,t)
 funcs=[lambda s:e*eps(s)+eta*mp.diff(eps,s),lambda s:maxwell(p,s)];models=[]
 for i,f in enumerate(funcs):
  stress=f(t);energy=(lambda s:e*eps(s)**2/2)if i==0 else(lambda s:f(s)**2/(2*e))
  models.append(dict(stress=stress,stressRate=mp.diff(f,t),energy=energy(t),energyRate=mp.diff(energy,t),dissipation=eta*rate**2 if i==0 else stress**2/eta,power=stress*rate))
 return dict(strain=eps(t),strainRate=rate,kelvin=models[0],maxwell=models[1])
def window(p,start,end):
 e,eta=M(p['elasticModulus']),M(p['viscosity']);out={}
 for name in ['kelvin','maxwell']:
  def stress(t):return e*strain(p,t)+eta*mp.diff(lambda s:strain(p,s),t)if name=='kelvin'else maxwell(p,t)
  def energy(t):return e*strain(p,t)**2/2 if name=='kelvin'else stress(t)**2/(2*e)
  power=lambda t:stress(t)*mp.diff(lambda s:strain(p,s),t)
  diss=lambda t:eta*mp.diff(lambda s:strain(p,s),t)**2 if name=='kelvin'else stress(t)**2/eta
  out[name]=dict(work=mp.quad(power,[start,end]),dissipation=mp.quad(diss,[start,end]),energyChange=energy(end)-energy(start))
 return out
visco=[]
for i in range(24):
 p=dict(elasticModulus=rng.uniform(.3,3.5),viscosity=rng.uniform(.3,3.5),strainAmplitude=rng.uniform(-.004,.004),angularFrequency=rng.uniform(.5,3),initialElasticStrain=rng.uniform(-.004,.004));t=M(rng.uniform(.1,5));start=M(rng.uniform(0,.8));end=M(rng.uniform(1,3))
 e,eta,amp,w,e0=[M(p[k])for k in ['elasticModulus','viscosity','strainAmplitude','angularFrequency','initialElasticStrain']]
 direct=e*e0*mp.exp(-e*t/eta)+e*amp*w*t*mp.quad(lambda u:mp.exp(-e*t*(1-u)/eta)*mp.cos(w*t*u),[0,1]);assert mp.almosteq(direct,maxwell(p,t))
 visco.append(dict(parameters=p,time=float(t),start=float(start),end=float(end),expected=serial(state(p,t)),window=serial(window(p,start,end))))
def neo(p):
 mu,ell,angle,normal,area,length=[M(p[k])for k in ['shearModulus','logStretch','rotation','normalAngle','referenceArea','referenceLength']];lam=mp.exp(ell);r=mp.matrix([[mp.cos(angle),-mp.sin(angle),0],[mp.sin(angle),mp.cos(angle),0],[0,0,1]]);F=r*mp.diag([lam,1/mp.sqrt(lam),1/mp.sqrt(lam)]);B=F*F.T;pressure=mu/lam;stress=mu*B-pressure*mp.eye(3);piola=stress*F.T**-1;n=mp.matrix([mp.cos(normal),mp.sin(normal),0]);t=stress*n;m=mp.matrix([-mp.sin(normal),mp.cos(normal),0]);energy=mu/2*(sum(B[i,i]for i in range(3))-3)
 def energy_lambda(x):return mu/2*(x*x+2/x-3)
 return dict(stretch=lam,energyDensity=energy,nominalStress=mp.diff(energy_lambda,lam),cauchyStress=mu*(lam*lam-1/lam),nominalTangent=mp.diff(energy_lambda,lam,2),pressure=pressure,currentArea=area/lam,force=area/lam*(mu*(lam*lam-1/lam)),storedEnergy=energy*area*length,F=F.tolist(),stress=stress.tolist(),firstPiola=piola.tolist(),traction=list(t),normalStress=(n.T*t)[0],signedShearStress=(m.T*t)[0])
neos=[]
for i in range(24):
 p=dict(shearModulus=rng.uniform(.3,3.5),logStretch=rng.uniform(-1.2,1.2),rotation=rng.uniform(-3,3),normalAngle=rng.uniform(-3,3),referenceArea=rng.uniform(.3,3),referenceLength=rng.uniform(.3,3));a=M(rng.uniform(-1,1));b=M(rng.uniform(-1,1));mu=M(p['shearModulus']);W=lambda x:mu/2*(x*x+2/x-3)
 work=mp.quad(lambda x:mp.diff(W,x),[mp.exp(a),mp.exp(b)]);wrong=mp.quad(lambda x:mu*(x*x-1/x),[mp.exp(a),mp.exp(b)])
 neos.append(dict(parameters=p,expected=serial(neo(p)),start=float(a),end=float(b),path=serial(dict(workDensity=work,energyDensityChange=W(mp.exp(b))-W(mp.exp(a)),wrongCauchyWorkDensity=wrong))))
visco_limits=[];neo_limits=[]
with mp.workdps(1200):
 for t,e0,w,amp in [(1e-20,0,1,.002),(1e-20,.002,1,.002),(1e-20,.001,1,.002),(1e-200,.001,1,.002),(5e-324,.001,1,.002),(16,.001,0,0),(1,0,1e-200,.002),(1,0,5e-324,.002)]:
  p=dict(elasticModulus=2,viscosity=.0625 if t==16 else 1,strainAmplitude=amp,angularFrequency=w,initialElasticStrain=e0);tt=M(t);sigma=maxwell(p,tt);change=sigma-M(p['elasticModulus'])*M(e0);energy_change=change*(sigma+M(p['elasticModulus'])*M(e0))/(2*M(p['elasticModulus']))
  visco_limits.append(dict(parameters=p,time=t,expected=dict(stress=signed(sigma),change=signed(change),stressRate=signed(mp.diff(lambda s:maxwell(p,s),tt)),energyChange=signed(energy_change))))
 for ell,mu,area,length in [(1e-20,2,1,1),(-1e-20,2,1,1),(1e-200,2,1,1),(5e-324,.25,.25,.25),(5e-163,1,4,4),(0,2,1,1),(1e-10,2,1,1),(-1e-10,2,1,1)]:
  p=dict(shearModulus=mu,logStretch=ell,rotation=.3,normalAngle=.7,referenceArea=area,referenceLength=length);r=neo(p);neo_limits.append(dict(parameters=p,expected={k:signed(r[k])for k in ['energyDensity','nominalStress','cauchyStress','force','storedEnergy']}))
payload=dict(method='160-digit complex convolution primitive;24 direct convolution checks;actual differentiation and physical quadrature;full matrix stresses;1200-digit tiny limits',viscoCases=visco,neoCases=neos,viscoLimits=visco_limits,neoLimits=neo_limits)
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2)+'\n');print('Wrote24 viscoelastic,24 hyperelastic,8 viscoelastic limits and8 hyperelastic limits')
