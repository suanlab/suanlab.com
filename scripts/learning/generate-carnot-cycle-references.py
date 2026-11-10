"""Independent physical-volume integrations along a constructed Carnot path.

No production imports. Exact Fraction input coordinates define temperature
interpolation; volumes follow pV=nRT and the adiabatic equation. Ordinary cases
integrate p(V) directly. Limits use high-precision physical antiderivatives.
"""
import argparse,json,math,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
parser=argparse.ArgumentParser();parser.add_argument('--output',default='scripts/fixtures/carnot-cycle-independent.json')
output=Path(parser.parse_args().output)
if output.exists():raise SystemExit('Refusing to overwrite an existing reference')
rng=random.Random(26110619);Rfloat=8.31446261815324
base=dict(moles=1.,hotTemperature=600.,coldTemperature=300.,initialVolume=.01,expansionRatio=2.)
def val(f):return mp.mpf(f.numerator)/f.denominator
def obs(v):return dict(value=str(v),sign=int(mp.sign(v)),logMagnitude=None if v==0 else str(mp.log(abs(v))))
def case(p,direction,progress,digits,quadrature):
 with mp.workdps(digits):
  n,R=mp.mpf(p['moles']),mp.mpf(Rfloat);th,tc,va,r=[mp.mpf(p[k])for k in ['hotTemperature','coldTemperature','initialVolume','expansionRatio']];cv=mp.mpf('1.5')*R;alpha=(th/tc)**mp.mpf('1.5')
  fth,ftc,fs=F(p['hotTemperature']),F(p['coldTemperature']),F(progress);index=min(3,int(4*fs));u=4*fs-index
  forward=direction=='forward';d=1 if forward else -1
  corners=[(th,va),(th,r*va),(tc,alpha*r*va),(tc,alpha*va),(th,va)]
  if not forward:corners=[corners[i]for i in [0,3,2,1,4]]
  kinds=['isothermal','adiabatic','isothermal','adiabatic']if forward else ['adiabatic','isothermal','adiabatic','isothermal']
  qh=mp.mpf(0);qc=mp.mpf(0);work=mp.mpf(0);legs=[];T,V=th,va
  for j in range(index+1):
   z=F(1)if j<index else u;(ta,a),(tb,b)=corners[j],corners[j+1];kind=kinds[j]
   if z==0:T,V=ta,a
   elif z==1:T,V=tb,b
   elif kind=='isothermal':T=ta;V=a*mp.exp(val(z)*mp.log(b/a))
   else:
    start=fth if ta==th else ftc;end=fth if tb==th else ftc
    T=val(start+(end-start)*z);V=a*(ta/T)**mp.mpf('1.5')
   if a==V:w=mp.mpf(0)
   elif kind=='isothermal':w=mp.quad(lambda x:n*R*ta/x,[a,V])if quadrature else n*R*ta*mp.log(V/a)
   else:
    # p(V)=nR*Tstart*Vstart^(2/3)*V^(-5/3).
    w=mp.quad(lambda x:n*R*ta*(a/x)**(mp.mpf(2)/3)/x,[a,V])if quadrature else n*R*ta*a**(mp.mpf(2)/3)*(V**(-mp.mpf(2)/3)-a**(-mp.mpf(2)/3))/(-mp.mpf(2)/3)
   du=n*cv*(T-ta);q=du+w;entropy=mp.mpf(0)
   if kind=='isothermal':
    entropy=w/ta
    if (forward and j==0)or(not forward and j==3):qh+=w
    else:qc+=w
   work+=w;legs.append(dict(work=obs(w),heat=obs(w if kind=='isothermal'else mp.mpf(0)),energyChange=obs(du),entropyChange=obs(entropy)))
  if forward:fh=u if index==0 else F(1);fc=F(0)if index<2 else u if index==2 else F(1)
  else:fh=u if index==3 else F(0);fc=F(0)if index==0 else u if index==1 else F(1)
  # Recognize algebraic zero cases only; never threshold numerical residuals.
  net=qh+qc
  if fth*fh-ftc*fc==0 or r==1:net=mp.mpf(0)
  du=n*cv*(T-th)
  if du==0 and (fth*fh-ftc*fc==0 or r==1):work=mp.mpf(0)
  entropy=sum(mp.mpf(x['entropyChange']['value'])for x in legs)
  if fh==fc or r==1:entropy=mp.mpf(0)
  deltaT=T-th;deltaV=V-va
  expected=dict(temperature=T,volume=V,pressure=n*R*T/V,temperatureChange=deltaT,volumeChange=deltaV,logVolumeRatio=mp.log(V/va),energyChange=du,heatHotIntoGas=qh,heatColdIntoGas=qc,netHeat=net,workByGas=work,entropyChangeGas=entropy,hotReservoirEntropyChange=-qh/th,coldReservoirEntropyChange=-qc/tc)
  # Independent physical integration and stored energy must agree at high precision.
  assert mp.almosteq(work,net-du,rel_eps=mp.mpf('1e-'+str(digits-40)),abs_eps=mp.mpf('1e-'+str(digits-40)))
  result=dict(parameters=p,direction=direction,progress=progress,digits=digits,physicalQuadrature=quadrature,expected={k:obs(v)for k,v in expected.items()})
  if progress==1:result['physicalFullLegs']=legs
  return result
ordinary=[]
for direction in ['forward','reverse']:
 for j in range(9):ordinary.append(case(base,direction,j/8,160,True))
for i in range(12):
 cold=rng.uniform(250,550);p=dict(moles=rng.uniform(.25,2),hotTemperature=rng.uniform(cold,800),coldTemperature=cold,initialVolume=rng.uniform(.005,.04),expansionRatio=rng.uniform(1,4))
 ordinary.append(case(p,'forward'if i%2==0 else'reverse',rng.random(),160,True))
limits=[(base,'forward',math.nextafter(0.,1.)),(base,'reverse',math.nextafter(0.,1.)),
 (dict(base,expansionRatio=1.),'forward',.375),(dict(base,expansionRatio=1.),'reverse',1.),
 (dict(base,hotTemperature=300.),'forward',1.),(dict(base,hotTemperature=300.),'reverse',.875),
 (dict(base,hotTemperature=math.nextafter(300.,math.inf),expansionRatio=math.nextafter(1.,math.inf)),'forward',1.),
 (dict(base,hotTemperature=math.nextafter(300.,math.inf),expansionRatio=math.nextafter(1.,math.inf)),'reverse',1.),
 (base,'reverse',.875),(base,'reverse',math.nextafter(.875,0.)),(base,'reverse',math.nextafter(.875,1.)),
 (dict(base,moles=2.,hotTemperature=800.,coldTemperature=250.,initialVolume=.04,expansionRatio=4.),'forward',.5),
 (dict(base,hotTemperature=790.718,coldTemperature=math.nextafter(250.,math.inf)),'forward',1.),
 (dict(base,hotTemperature=790.718,coldTemperature=math.nextafter(250.,math.inf)),'reverse',1.),
 (dict(base,expansionRatio=math.nextafter(1.,math.inf)),'forward',math.nextafter(0.,1.)),
 (dict(base,expansionRatio=math.nextafter(1.,math.inf)),'reverse',math.nextafter(0.,1.))]
edge=[case(p,d,s,1000,False)for p,d,s in limits]
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dict(method='physical-volume p(V) quadrature and independent high-precision antiderivatives; exact binary temperature interpolation',gasConstantBinary=Rfloat,ordinary=ordinary,limits=edge),indent=2)+'\n');print(f'Saved{len(ordinary)} physical-integral cases and{len(edge)} limits to{output}',flush=True)
