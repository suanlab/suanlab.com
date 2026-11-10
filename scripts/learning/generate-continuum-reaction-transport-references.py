"""Independent 160-digit differentiation, physical quadrature and stable limiting cases."""
import argparse,json,random
from pathlib import Path
import mpmath as mp
mp.mp.dps=160
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-reaction-transport-independent.json');path=Path(parser.parse_args().output)
if path.exists():raise SystemExit('Refusing to overwrite an existing reference')
M=mp.mpf;rng=random.Random(261108)
def serial(v):
 if isinstance(v,dict):return{k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return[serial(x)for x in v]
 return float(v)
def carrier(p,x,t):
 mass,area,sigma,mu,u,d=[M(p[k])for k in ['mass','area','initialSigma','initialMean','velocity','diffusivity']];v=sigma*sigma+2*d*t
 return mass/(area*mp.sqrt(2*mp.pi*v))*mp.exp(-(x-mu-u*t)**2/(2*v))
def fraction(p,t):
 kf,kb,initial=[M(p[k])for k in ['forwardRate','reverseRate','initialFractionA']];rate=kf+kb
 return initial if rate==0 else kb/rate+(initial-kb/rate)*mp.exp(-rate*t)
def species(p,x,t,j):
 f=fraction(p,t);return carrier(p,x,t)*(f if j==0 else 1-f)
reaction=[]
for i in range(24):
 p=dict(mass=rng.uniform(.2,3),area=rng.uniform(.4,2),initialSigma=rng.uniform(.3,1.2),initialMean=rng.uniform(-1,1),velocity=rng.uniform(-1,1),diffusivity=rng.uniform(.02,.5),forwardRate=rng.uniform(.05,2),reverseRate=rng.uniform(.05,2),initialFractionA=rng.uniform(.1,.9))
 x,t=M(rng.uniform(-2,2)),M(rng.uniform(.1,2));left,right=M('-1.3'),M('1.1');area=M(p['area']);values=[]
 def source(xx):return -M(p['forwardRate'])*species(p,xx,t,0)+M(p['reverseRate'])*species(p,xx,t,1)
 for j in range(2):
  f=lambda xx,tt:species(p,xx,tt,j);c=f(x,t);dx=mp.diff(lambda xx:f(xx,t),x);dt=mp.diff(lambda tt:f(x,tt),t)
  flux=lambda xx:M(p['velocity'])*f(xx,t)-M(p['diffusivity'])*mp.diff(lambda yy:f(yy,t),xx)
  mass=area*mp.quad(lambda xx:f(xx,t),[left,right]);rate=area*mp.quad(lambda xx:mp.diff(lambda tt:f(xx,tt),t),[left,right]);integrated_source=area*mp.quad(lambda xx:source(xx)*(1 if j==0 else -1),[left,right])
  values.append(dict(concentration=c,dx=dx,dt=dt,flux=flux(x),fluxGradient=mp.diff(flux,x),source=source(x)*(1 if j==0 else -1),window=dict(massInside=mass,massRate=rate,netOutflow=area*(flux(right)-flux(left)),integratedSource=integrated_source)))
 reaction.append(dict(parameters=p,x=float(x),time=float(t),left=float(left),right=float(right),expected=serial(dict(fractionA=fraction(p,t),species=values))))
def front(p,x,t):
 d,r,k,u,x0=[M(p[s])for s in ['diffusivity','growthRate','carryingConcentration','bulkVelocity','initialFront']];lam=mp.sqrt(r/(6*d));speed=u+5*mp.sqrt(d*r/6)
 return k/(1+mp.exp(lam*(x-x0-speed*t)))**2
fronts=[]
for i in range(24):
 p=dict(diffusivity=rng.uniform(.03,1),growthRate=rng.uniform(.03,2),carryingConcentration=rng.uniform(.3,3),bulkVelocity=rng.uniform(-1,1),initialFront=rng.uniform(-1,1),area=rng.uniform(.4,2))
 x,t=M(rng.uniform(-3,3)),M(rng.uniform(.1,2));left,right=M('-1.1'),M('1.3');area=M(p['area'])
 c=front(p,x,t);dx=mp.diff(lambda xx:front(p,xx,t),x);dxx=mp.diff(lambda xx:front(p,xx,t),x,2);dt=mp.diff(lambda tt:front(p,x,tt),t)
 flux=lambda xx:M(p['bulkVelocity'])*front(p,xx,t)-M(p['diffusivity'])*mp.diff(lambda yy:front(p,yy,t),xx)
 source=lambda xx:M(p['growthRate'])*front(p,xx,t)*(1-front(p,xx,t)/M(p['carryingConcentration']))
 rate=area*mp.quad(lambda xx:mp.diff(lambda tt:front(p,xx,tt),t),[left,right]);mass=area*mp.quad(lambda xx:front(p,xx,t),[left,right]);integrated=area*mp.quad(source,[left,right])
 fronts.append(dict(parameters=p,x=float(x),time=float(t),left=float(left),right=float(right),expected=serial(dict(concentration=c,dx=dx,dxx=dxx,dt=dt,flux=flux(x),source=source(x),window=dict(massInside=mass,massRate=rate,netOutflow=area*(flux(right)-flux(left)),integratedSource=integrated)))))
kinetic=[]
for kf,kb,f0,t in [(1e-20,0,1,1),(1e-200,0,1,1e-200),(128,128,1,4),(1,1,.5,2),(0,0,.3,1),(128,5e-324,0,4),(5e-324,5e-324,.5,1)]:
 kf0,kb0,f00,t0=kf,kb,f0,t;kf,kb,f0,t=map(M,[kf,kb,f0,t]);rate=kf+kb
 if rate==0:a,b,df,change=f0,1-f0,M(0),M(0)
 else:
  z=rate*t;mix=-mp.expm1(-z);a=f0*mp.exp(-z)+(kb/rate)*mix;b=(1-f0)*mp.exp(-z)+(kf/rate)*mix;df=(kb*(1-f0)-kf*f0)*mp.exp(-z);change=(kb*(1-f0)-kf*f0)*mix/rate
 kinetic.append(dict(parameters=dict(forwardRate=kf0,reverseRate=kb0,initialFractionA=f00),time=t0,expected={k:str(v)for k,v in dict(fractionA=a,fractionB=b,logFractionA=mp.log(a)if a else None,fractionRateA=df,logRateMagnitude=mp.log(abs(df))if df else None,stableChangeA=change).items()}))
tails=[];p=dict(diffusivity=.0001,growthRate=4,carryingConcentration=1.3,bulkVelocity=0,initialFront=0,area=2);lam=mp.sqrt(M(p['growthRate'])/(6*M(p['diffusivity'])))
for requested in [-1000,-400,-40,40,373.1,400,1000]:
 x=float(M(requested)/lam);z=lam*M(x);e=mp.exp(z);k=M(p['carryingConcentration']);r=M(p['growthRate']);c=k/(1+e)**2;deficit=k*e*(2+e)/(1+e)**2;source=r*c*deficit/k;dx=-2*lam*k*e/(1+e)**3
 tails.append(dict(parameters=p,x=x,requestedPhase=requested,expected={key:str(v)for key,v in dict(concentration=c,logConcentration=mp.log(c),deficit=deficit,logDeficit=mp.log(deficit),source=source,logSource=mp.log(source),dx=dx).items()}))
result=dict(method='Independent mpmath160 actual-field differentiation and physical-window quadrature; stable rational/expm1 references for limiting states',seed=261108,reactionCases=reaction,frontCases=fronts,kineticCases=kinetic,frontTailCases=tails)
path.write_text(json.dumps(result,indent=2)+'\n');print('Wrote24 reversible,24 Fisher,7 kinetic and7 front-tail references')
