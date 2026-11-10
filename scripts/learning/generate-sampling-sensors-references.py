"""Independent NumPy/SciPy references: box/time quadrature, Poisson sums, Gaussian CDF.
No product TypeScript imports; no seeded-sample accuracy surrogate.
"""
from pathlib import Path
import json,itertools
import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson,norm
from scipy.constants import h,c
pixels=[];sensors=[]
for i,(fraction,width,phase) in enumerate(itertools.product([0,.25,.5,.8,1.25,2.],[0,.25,.7,1.],[0,.7])):
 pitch=[5e-6,10e-6,20e-6][i%3]; exposure=0 if i%13==0 else .002; freq=fraction/pitch
 speed=[0,.0003,-.0007][i%3]
 s=dict(pitch=pitch,apertureWidth=width*pitch,pixelHeight=7e-6,wavelength=550e-9,irradiance=.003,modulation=.8,frequency=freq,phase=phase,velocity=speed,exposure=exposure,centerTime=.003,origin=.3*pitch,radius=4,queryCoordinate=.7*pitch)
 samples=[]
 for n in range(-4,5):
  x=s['origin']+n*pitch
  fun=lambda u,v:s['irradiance']*(1+s['modulation']*np.cos(2*np.pi*freq*(x+u*s['apertureWidth']-speed*(s['centerTime']+v*exposure))+phase))
  average=quad(lambda v:quad(lambda u:fun(u,v),-.5,.5,epsabs=1e-13,epsrel=1e-12)[0],-.5,.5,epsabs=1e-13,epsrel=1e-12)[0]
  energy=average*s['apertureWidth']*s['pixelHeight']*exposure
  samples.append(dict(index=n,coordinate=x,limitAverage=average,energy=energy,incidentPhotonMean=energy*s['wavelength']/(h*c)))
 pixels.append(dict(system=s,samples=samples))
for i in range(36):
 photons=[0,.1,2,50,2048,4096][i%6];qe=[0,.3,1][(i//6)%3];exposure=.01
 s=dict(incidentPhotonMean=photons,quantumEfficiency=qe,exposure=exposure,darkRate=[0,12,128][i%3],readSigma=[0,.3,2,16][i%4],gain=[.05,.7,2,8][(i//3)%4],offset=[-2,.25,8,250][(i//2)%4],bits=[4,6,8][i%3],fullWell=[1,12,80,8192][(i//4)%4])
 lam=photons*qe+s['darkRate']*exposure;ns=np.arange(0,int(np.ceil(lam+16*np.sqrt(lam)+128))+1);ps=poisson.pmf(ns,lam);stored=np.minimum(ns,s['fullWell']);mu=s['offset']+s['gain']*stored;sd=s['gain']*s['readSigma'];M=2**s['bits']-1
 probs=[]
 if sd==0:
  codes=np.clip(np.floor(mu+.5),0,M).astype(int)
  probs=np.bincount(codes,weights=ps,minlength=M+1)
  lower=float(np.sum(ps[mu<-.5]));upper=float(np.sum(ps[mu>=M+.5]))
 else:
  for k in range(M+1):
   lo=0 if k==0 else norm.cdf((k-.5-mu)/sd)
   hi=1 if k==M else norm.cdf((k+.5-mu)/sd)
   probs.append(float(np.sum(ps*(hi-lo))))
  lower=float(np.sum(ps*norm.cdf((-.5-mu)/sd)));upper=float(np.sum(ps*norm.sf((M+.5-mu)/sd)))
 probs=np.array(probs);mean=float(np.dot(np.arange(M+1),probs));var=float(np.dot((np.arange(M+1)-mean)**2,probs));sm=float(np.dot(stored,ps));sv=float(np.dot((stored-sm)**2,ps))
 sensors.append(dict(system=s,bins=probs.tolist(),adcMean=mean,adcVariance=var,storedMean=sm,storedVariance=sv,fullWellOverflowProbability=float(np.sum(ps[ns>s['fullWell']])),adcLowerClippingProbability=lower,adcUpperClippingProbability=upper,coveredProbability=float(np.sum(ps))))
z=np.unique(np.r_[np.linspace(-9,9,73),[-1.00001,-1,-.99999,0,.99999,1,1.00001]])
out=dict(provenance=dict(productImports=False,method='SciPy nested box/time quadrature; scipy.stats.poisson and normal CDF; direct clipped-code sums',generator='scripts/learning/generate-sampling-sensors-references.py'),pixels=pixels,sensors=sensors,normal=[dict(z=float(x),cdf=float(norm.cdf(x))) for x in z])
path=Path(__file__).resolve().parent.parent/'fixtures/sampling-sensors-independent.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(pixelCases=len(pixels),pixelObservations=sum(len(x['samples']) for x in pixels),sensorCases=len(sensors),normalCDFQueries=len(z))))
