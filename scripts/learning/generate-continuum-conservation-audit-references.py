"""Stage-one independent1600-digit spatial integrals and endpoint differences.
The source model is never imported; ordinary float output is a separate projection.
"""
import argparse,json
from pathlib import Path
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('--output',default='scripts/fixtures/continuum-conservation-audit-independent.json');target=Path(p.parse_args().output)
if target.exists():raise SystemExit('Refusing to overwrite reference')
m.mp.dps=1600
tiny=float.fromhex('0x0.0000000000001p-1022')
S=dict(meanDensity=1.,contrast=.5,wavenumber=1.,velocity=1.,area=1.)
W=dict(center=0.,halfWidth=.5,speed=0.)
def obs(x):return dict(value=m.nstr(x,80),sign=int(m.sign(x)),logMagnitude=None if x==0 else m.nstr(m.log(abs(x)),80),rounded=float(x)if x==0 or float(x)!=0 else None)
def mass_integral(s,w,t):
 r,alpha,k,v,A=[m.mpf(s[x])for x in ['meanDensity','contrast','wavenumber','velocity','area']];c,h,speed=[m.mpf(w[x])for x in ['center','halfWidth','speed']];lo=c+speed*t-h;hi=c+speed*t+h
 if not k:return A*r*(1+alpha)*(hi-lo)
 return A*r*((hi-lo)+alpha*(m.sin(k*(hi-v*t))-m.sin(k*(lo-v*t)))/k)
transport=[]
params=[(S,W,0.,tiny),(S,W,0.,2*tiny),(S,W,-tiny,tiny),(S,{**W,'center':.3},0.,tiny),({**S,'area':4.,'meanDensity':4.,'contrast':.8},{**W,'center':.5,'halfWidth':1.},0.,tiny),({**S,'area':4.,'meanDensity':4.,'contrast':.8,'wavenumber':8.,'velocity':2.},{**W,'center':.25,'halfWidth':1.,'speed':-2.},0.,tiny),({**S,'wavenumber':tiny},W,0.,1.),({**S,'wavenumber':tiny},{**W,'halfWidth':tiny},0.,tiny),(S,{**W,'halfWidth':tiny,'center':.3},-.4,.7),(S,{**W,'speed':1.},0.,tiny),({**S,'contrast':0.},W,0.,tiny),(S,{**W,'halfWidth':0.},0.,tiny),(S,W,.2,.2),(S,W,-.4,.7)]
for s,w,t0,t1 in params:
 change=mass_integral(s,w,m.mpf(t1))-mass_integral(s,w,m.mpf(t0))
 if t0==t1 or s['contrast']==0 or s['wavenumber']==0 or s['velocity']==w['speed']or w['halfWidth']==0 or w['center']==0 and t0==-t1:change=m.mpf(0)
 transport.append(dict(system=s,window=w,start=t0,end=t1,massChange=obs(change)))
G=dict(initialDensity=1.,initialPressure=1.,expansionRate=tiny,drift=0.,gamma=1.5,area=1.)
C=dict(center=0.,halfWidth=.5,speed=0.,mode='material')
def gas_internal(s,w,t):
 A=m.mpf(s['area']);h=m.mpf(w['halfWidth']);P=m.mpf(s['initialPressure']);gamma=m.mpf(s['gamma']);stretch=1+m.mpf(s['expansionRate'])*t
 # Integrate the spatially uniform internal-energy density over actual boundaries.
 width=2*h*stretch if w['mode']=='material'else 2*h
 return A*width*P*stretch**(-gamma)/(gamma-1)
gas=[]
params=[(G,C,0.,1.),({**G,'expansionRate':-tiny},C,0.,1.),(G,{**C,'mode':'fixed'},0.,1.),(G,{**C,'mode':'translating'},0.,1.),({**G,'gamma':1.1,'area':4.,'initialPressure':4.},{**C,'halfWidth':1.},.4,.7),({**G,'expansionRate':.25,'gamma':1.1,'area':4.,'initialPressure':4.},{**C,'halfWidth':1.},0.,tiny),({**G,'expansionRate':tiny},C,0.,tiny),({**G,'expansionRate':.5},C,-.4,.7),({**G,'expansionRate':-.5},{**C,'mode':'fixed'},-.4,.7),({**G,'initialPressure':0.},C,0.,1.),({**G,'expansionRate':0.},C,0.,1.),(G,{**C,'halfWidth':0.},0.,1.),(G,C,.2,.2),({**G,'area':.25,'initialPressure':4.,'gamma':1.1},{**C,'halfWidth':tiny},0.,1.)]
for s,w,t0,t1 in params:
 change=gas_internal(s,w,m.mpf(t1))-gas_internal(s,w,m.mpf(t0))
 if t0==t1 or s['initialPressure']==0 or s['expansionRate']==0 or w['halfWidth']==0:change=m.mpf(0)
 gas.append(dict(system=s,window=w,start=t0,end=t1,internalChange=obs(change)))
target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(dict(digits=1600,outputDigits=80,method='independent spatial primitive integrals and high-precision endpoint subtraction on exact binary inputs',transportCases=transport,gasCases=gas),indent=2)+'\n');print('Wrote14 transport and14 internal-energy cases.')
