"""Product-free continuous 1D reference calculations. numpy/scipy required.
Direct pupil quadrature, direct shifted-pupil autocorrelation quadrature and
real-period field-square integration. No DFT and no product TypeScript imports.
"""
from pathlib import Path
import itertools,json
import numpy as np
from scipy.integrate import quad
from scipy.special import sici
rng=np.random.default_rng(89411)
def pack(z):return [float(z.real),float(z.imag)]
def cq(fun,lo,hi):return quad(lambda t:float(fun(t).real),lo,hi,epsabs=2e-12,epsrel=2e-12,limit=300)[0]+1j*quad(lambda t:float(fun(t).imag),lo,hi,epsabs=2e-12,epsrel=2e-12,limit=300)[0]
def intervals(p):return [(-1,-p['obstructionFraction']),(p['obstructionFraction'],1)] if p['obstructionFraction']<1 else []
def setup(p):
 ref=p['referenceHalfWidth']/(p['wavelength']*p['focalLength1']);fc=ref*p['apertureFraction'];return ref,fc

def mask(p,nu):
 ref,fc=setup(p)
 if fc==0 or abs(nu)>=fc or abs(nu)<p['obstructionFraction']*fc:return 0j
 t=nu/fc;return p['amplitudeTransmission']*np.exp(1j*(p['quadraticPhase']*t*t+p['linearPhase']*t))
def field(p,x):
 ref,fc=setup(p)
 if fc==0 or p['amplitudeTransmission']==0:return 0j
 return fc*p['amplitudeTransmission']*sum(cq(lambda t:np.exp(1j*(p['quadraticPhase']*t*t+(p['linearPhase']+2*np.pi*fc*x)*t)),lo,hi) for lo,hi in intervals(p))
def energy(p):
 ref,fc=setup(p)
 return fc*sum(quad(lambda t:abs(p['amplitudeTransmission']*np.exp(1j*(p['quadraticPhase']*t*t+p['linearPhase']*t)))**2,lo,hi)[0] for lo,hi in intervals(p))
def otf(p,nu):
 ref,fc=setup(p)
 if fc==0:return 0j
 d=nu/fc;answer=0j
 for a in intervals(p):
  for b in intervals(p):
   lo=max(a[0],b[0]+d);hi=min(a[1],b[1]+d)
   if hi>lo:
    answer+=fc*cq(lambda t:p['amplitudeTransmission']**2*np.exp(1j*(p['quadraticPhase']*(t*t-(t-d)**2)+p['linearPhase']*d)),lo,hi)
 return answer
combinations=list(itertools.product([0,.4,1],[0,.3,1],[0,.7,1],[0,2,-6],[0,.8,-1.1]))
chosen=[(1,0,1,0,0),(1,0,1,6,0),(1,.3,1,-6,.8),(0,0,1,0,0),(1,1,1,0,0),(1,0,0,0,0)]+[combinations[i] for i in rng.choice(len(combinations),42,replace=False)]
pupils=[];coherent=[];incoherent=[]
for index,(ap,ob,amp,a,b) in enumerate(chosen):
 p=dict(wavelength=[500e-9,800e-9][index%2],focalLength1=[.05,.1][index%2],focalLength2=[.05,.1,.2][index%3],referenceHalfWidth=.001,apertureFraction=ap,obstructionFraction=ob,amplitudeTransmission=amp,quadraticPhase=a,linearPhase=b)
 ref,fc=setup(p);e=energy(p);raw_transfer=[dict(frequency=float(nu),value=pack(otf(p,nu))) for nu in np.linspace(-2.2*ref,2.2*ref,25)]
 psfs=[dict(coordinate=float(x),field=pack(field(p,x)),intensity=float(abs(field(p,x))**2)) for x in np.linspace(-4/ref,4/ref,17)]
 pupils.append(dict(system=p,energy=float(e),psf=psfs,otf=raw_transfer))
 period=[.0001,.0004,.001][index%3];global_phase=[0,.31,-np.pi/2][index%3]
 modes=[dict(order=n,amplitude=float(v),phase=float(ph)) for n,v,ph in zip([-4,-1,0,1,5,1],[.23,.4,1,.5,.17,.2],[.2,-.3,0,.4,-.6,np.pi])]
 def modefield(x,filtered):
  return np.exp(1j*global_phase)*sum(m['amplitude']*np.exp(1j*m['phase'])*np.exp(2j*np.pi*m['order']*x/period)*(mask(p,m['order']/period) if filtered else 1) for m in modes)
 observations=[dict(coordinate=float(x),before=pack(modefield(x,False)),after=pack(modefield(x,True))) for x in np.linspace(-period/2,period/2,17)]
 # Direct integral includes cross terms and duplicate preparation orders.
 mean_in=quad(lambda t:float(abs(modefield(t*period,False))**2),-.5,.5,epsabs=2e-12)[0]
 mean_out=quad(lambda t:float(abs(modefield(t*period,True))**2),-.5,.5,epsabs=2e-12)[0]
 coherent.append(dict(system={**p,'period':period,'modes':modes,'globalPhase':global_phase,'queryCoordinate':period*.13},observations=observations,inputMean=float(mean_in),outputMean=float(mean_out)))
 sources=[dict(position=-.8/ref,weight=1),dict(position=.6/ref,weight=.4)]
 frequency=.75*ref;phase=.4;mean=1.2;mod=.8;tr=e/(2*ref);h=otf(p,frequency);cond=h/e if e>0 else None
 observations=[]
 for x in np.linspace(-2/ref,2/ref,9):
  point=sum(s['weight']*abs(field(p,x-s['position']))**2/(2*ref) for s in sources)
  grating=mean*tr*(1+mod*np.real((cond if cond is not None else 0)*np.exp(1j*(2*np.pi*frequency*x+phase))))
  observations.append(dict(coordinate=float(x),pointIntensity=float(point),gratingIntensity=float(grating)))
 incoherent.append(dict(system={**p,'sources':sources,'queryCoordinate':0,'gratingMean':mean,'gratingModulation':mod,'gratingFrequency':frequency,'gratingPhase':phase},observations=observations,pointIntegral=float(tr*1.4),gratingMean=float(mean*tr)))
# Continuous finite-window sinc-square integral, not a claim of finite-window unit mass.
windows=[]
for radius in [.1,.5,1,2,8]:
 actual=quad(lambda t:2*np.sinc(2*t)**2,-radius,radius,epsabs=2e-12)[0]
 analytic=2/np.pi*sici(4*np.pi*radius)[0]+(np.cos(4*np.pi*radius)-1)/(2*np.pi**2*radius)
 assert abs(actual-analytic)<2e-12
 windows.append(dict(scaledRadius=radius,numericalMass=float(actual),analyticMass=float(analytic)))
report=dict(provenance=dict(productImports=False,generator='scripts/learning/generate-fourier-optics-references.py',method='continuous QUADPACK pupil fields and shifted-pupil autocorrelation; direct real-period intensity integration; positive point-source intensity sum; finite-window sinc-square integral',coordinateConvention='1D continuous FT cycles/m, actual magnification=-f2/f1, fixed reference slit throughput'),pupils=pupils,coherent=coherent,incoherent=incoherent,finiteWindowMasses=windows)
Path('scripts/fixtures/fourier-optics-independent.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(pupils=len(pupils),pupilFieldObservations=17*len(pupils),pupilOTFObservations=25*len(pupils),coherentSystems=len(coherent),coherentFieldObservations=17*len(coherent),incoherentSystems=len(incoherent),positiveImageObservations=9*len(incoherent),finiteWindowIntegrals=len(windows)),indent=2))
