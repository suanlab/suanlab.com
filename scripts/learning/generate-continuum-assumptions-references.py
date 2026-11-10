"""Independent definition-level quadrature/differentiation; never imports the product model."""
import argparse,json,random,math
from pathlib import Path
import mpmath as mp
mp.mp.dps=55
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/continuum-assumptions-independent.json');args=parser.parse_args();path=Path(args.output)
if path.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(8675309)
def decimal(x):return mp.mpf(x) # Exact binary input conversion.
def weight(x,position,h):
 q=(x-position)/h
 return 15*(1-q*q)**2/(16*h) if abs(q)<1 else mp.mpf(0)
rows=[]
for i in range(48):
 particles=[dict(mass=rng.uniform(.1,3),position=rng.uniform(-2,2),velocity=rng.uniform(-1.5,1.5)) for _ in range(2+i%7)]
 x=rng.uniform(-2,2);time=rng.uniform(-.7,.7);h=rng.uniform(.2,1.4);rate=rng.uniform(-.3,.3);left=x-rng.uniform(.05,1);right=x+rng.uniform(.05,1)
 ps=[{k:decimal(v)for k,v in p.items()}for p in particles]
 def field(xx,tt,hh,power=0):return sum(p['mass']*p['velocity']**power*weight(xx,p['position']+p['velocity']*tt,hh) for p in ps)
 xx,tt,hh,rr,ll,uu=map(decimal,[x,time,h,rate,left,right]);density=field(xx,tt,hh);flux=field(xx,tt,hh,1);velocity=flux/density if density else None
 breaks=sorted(set([ll,uu]+[v for p in ps for v in [p['position']+p['velocity']*tt-hh,p['position']+p['velocity']*tt+hh] if ll<v<uu]))
 mass=mp.quad(lambda z:field(z,tt,hh),breaks)
 vals=dict(density=density,flux=flux,momentumFlux=field(xx,tt,hh,2),velocity=velocity,dispersion=sum(p['mass']*(p['velocity']-velocity)**2*weight(xx,p['position']+p['velocity']*tt,hh)for p in ps)if velocity is not None else None,
 densityX=mp.diff(lambda z:field(z,tt,hh),xx),fluxX=mp.diff(lambda z:field(z,tt,hh,1),xx),densityT=mp.diff(lambda dt:field(xx,tt+dt,hh+rr*dt),0),momentumT=mp.diff(lambda dt:field(xx,tt+dt,hh+rr*dt,1),0),smoothingRate=rr*mp.diff(lambda b:field(xx,tt,b),hh),intervalMass=mass)
 rows.append(dict(particles=particles,x=x,time=time,h=h,bandwidthRate=rate,left=left,right=right,expected={k:float(v) if v is not None else None for k,v in vals.items()}))
counts=[]
for n in range(2,21):
 for p in [.05,.2,.5,.9,1]:
  probs=[mp.binomial(n,k)*decimal(p)**k*(1-decimal(p))**(n-k) for k in range(n+1)]
  mean=sum(k*q for k,q in enumerate(probs));variance=sum((k-mean)**2*q for k,q in enumerate(probs))
  counts.append(dict(n=n,p=p,mean=float(mean),variance=float(variance),relativeStd=float(mp.sqrt(variance)/mean)))
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(precision=55,particleCases=rows,binomialCases=counts),indent=2)+'\n');print(json.dumps(dict(path=str(path),particleCases=len(rows),binomialCases=len(counts))))
