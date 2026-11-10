"""Independent 160-digit physical field differentiation and scaled quadrature.

Never imports product code; never overwrites an existing reference fixture.
"""
import argparse
import json
import random
from pathlib import Path
import mpmath as mp
mp.mp.dps = 160
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-diffusion-independent.json')
path = Path(parser.parse_args().output)
if path.exists(): raise SystemExit('Refusing to overwrite an existing reference')
M = mp.mpf
rng = random.Random(261107)
def serial(x):
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    return float(x)

def heat(p, x, t):
    cap, cond, mean, contrast, k, phase = [M(p[s]) for s in ['heatCapacity','conductivity','meanTemperature','contrast','waveNumber','phase']]
    return mean+mean*contrast*mp.exp(-cond*k*k*t/cap)*mp.cos(k*x+phase)
heat_cases = []
for i in range(24):
    p = dict(heatCapacity=rng.uniform(.3,3),conductivity=rng.uniform(.01,2),meanTemperature=rng.uniform(270,350),contrast=rng.uniform(-.18,.18),waveNumber=rng.uniform(.3,3),phase=rng.uniform(-3,3),area=rng.uniform(.3,3))
    x, t, center, half = [M(rng.uniform(a,b)) for a,b in [(-2,2),(.1,2),(-1,1),(.1,1)]]
    c, kappa, area = [M(p[s]) for s in ['heatCapacity','conductivity','area']]
    dx=mp.diff(lambda z:heat(p,z,t),x);dxx=mp.diff(lambda z:heat(p,z,t),x,2);dt=mp.diff(lambda z:heat(p,x,z),t)
    left,right=center-half,center+half
    mode=area*c*mp.quad(lambda z:heat(p,z,t)-M(p['meanTemperature']),[left,right])
    rate=area*c*mp.quad(lambda z:mp.diff(lambda tt:heat(p,z,tt),t),[left,right])
    flux=lambda z:-kappa*mp.diff(lambda xx:heat(p,xx,t),z)
    net=area*(flux(right)-flux(left));side=2*mp.pi/M(p['waveNumber'])
    norm=area*mp.quad(lambda z:(heat(p,z,t)-M(p['meanTemperature']))**2/2,[0,side])
    dissipation=area*kappa/c*mp.quad(lambda z:mp.diff(lambda xx:heat(p,xx,t),z)**2,[0,side])
    heat_cases.append(dict(parameters=p,x=float(x),time=float(t),center=float(center),halfWidth=float(half),expected=serial(dict(temperature=heat(p,x,t),gradient=dx,laplacian=dxx,timeDerivative=dt,heatFlux=-kappa*dx,modeHeat=mode,energyRate=rate,netPower=net,quadraticNorm=norm,gradientNormDissipation=dissipation))))

def gaussian(p,x,t):
    mass,area,sigma,mean,u,d = [M(p[s]) for s in ['mass','area','initialSigma','initialMean','velocity','diffusivity']]
    variance=sigma*sigma+2*d*t;mu=mean+u*t
    return mass/(area*mp.sqrt(2*mp.pi*variance))*mp.exp(-(x-mu)**2/(2*variance))
def window(p,left,right,t):
    mass,area,sigma,mean,u,d = [M(p[s]) for s in ['mass','area','initialSigma','initialMean','velocity','diffusivity']]
    variance=sigma*sigma+2*d*t;sd=mp.sqrt(variance);mu=mean+u*t
    logpdf=lambda x:-mp.log(sd)-mp.log(2*mp.pi)/2-(x-mu)**2/(2*variance)
    closest=min(max(mu,left),right);scale=logpdf(closest)
    scaled=mp.quad(lambda x:mp.exp(logpdf(x)-scale),[left,right]);logfrac=scale+mp.log(scaled)
    rate=area*mp.quad(lambda x:mp.diff(lambda tt:gaussian(p,x,tt),t),[left,right])
    flux=lambda x:u*gaussian(p,x,t)-d*mp.diff(lambda xx:gaussian(p,xx,t),x)
    return dict(fraction=mp.exp(logfrac),logFraction=logfrac,massInside=mass*mp.exp(logfrac),massRate=rate,netOutflow=area*(flux(right)-flux(left)))
gaussian_cases=[]
for i in range(24):
    p=dict(mass=rng.uniform(.1,3.5),area=rng.uniform(.3,3),initialSigma=rng.uniform(.2,1.5),initialMean=rng.uniform(-1,1),velocity=rng.uniform(-1.5,1.5),diffusivity=rng.uniform(.01,.8))
    x,t,left,width=[M(rng.uniform(a,b))for a,b in [(-2,2),(.1,2),(-2,0),(.2,2)]];right=left+width
    # The exported physical endpoints are IEEE doubles; re-use exactly these values.
    left,right=M(float(left)),M(float(right));area=M(p['area'])
    dx=mp.diff(lambda z:gaussian(p,z,t),x);dxx=mp.diff(lambda z:gaussian(p,z,t),x,2);dt=mp.diff(lambda z:gaussian(p,x,z),t)
    g=gaussian(p,x,t);s=M(p['initialSigma']);v=s*s+2*M(p['diffusivity'])*t;mu=M(p['initialMean'])+M(p['velocity'])*t
    # Full-domain moments via an independent standardized physical-position integral.
    mass=M(p['mass']);norm=lambda z:mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
    moments=[mass*mp.quad(lambda z:(mu+mp.sqrt(v)*z)**n*norm(z),[-mp.inf,0,mp.inf])for n in range(3)]
    gaussian_cases.append(dict(parameters=p,x=float(x),time=float(t),left=float(left),right=float(right),expected=serial(dict(concentration=g,logConcentration=mp.log(g),spatialDerivative=dx,secondDerivative=dxx,timeDerivative=dt,advectiveFlux=M(p['velocity'])*g,diffusiveFlux=-M(p['diffusivity'])*dx,window=window(p,left,right,t),moments=moments))))

tails=[]
for sigma,left,right in [(1,9,9.1),(1,9,9+1e-12),(.125,4.8,4.85),(.125,5,5.1),(.125,15,15.1),(1,0,5e-324)]:
    p=dict(mass=4,area=1,initialSigma=sigma,initialMean=0,velocity=0,diffusivity=0)
    e=window(p,M(left),M(right),M(0))
    tails.append(dict(parameters=p,left=left,right=right,expected=dict(fraction=str(e['fraction']),logFraction=str(e['logFraction']),massInside=str(e['massInside']))))
tiny=[]
for d,t in [('1e-20','1'),('1e-200','1e-100'),('1e-200','1e-200')]:
    d,t=M(d),M(t);z=d*t;dv=2*d*t;v0=M('.25')
    # Rational identities avoid loss even at 160 digits for a 1e-400 change.
    root=mp.sqrt(1+dv/v0);ds=dv/(mp.sqrt(v0+dv)+mp.sqrt(v0));peak=-(dv/v0)/(root*(root+1))
    tiny.append(dict(diffusivity=str(d),time=str(t),heatDecayChange=str(mp.expm1(-z)),varianceChange=str(dv),sigmaChange=str(ds),relativePeakChange=str(peak)))
result=dict(method='Independent mpmath160 field differentiation, scaled physical-window and full-domain moment quadrature',seed=261107,heatCases=heat_cases,gaussianCases=gaussian_cases,tailCases=tails,tinyCases=tiny)
path.write_text(json.dumps(result,indent=2)+'\n')
print('Wrote24 heat,24 Gaussian,6 tail-window and3 tiny-change references')
