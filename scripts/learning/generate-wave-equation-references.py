"""Independent SI references: matrix exponentials and polynomial image data.
Run from repository root. Requires NumPy, SciPy and SymPy; never imports product code.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import expm
from scipy.integrate import quad, solve_ivp
import sympy as sp

rng=np.random.default_rng(20261105)
chains=[]
for case in range(24):
    n=[8,16,32][case%3]; length=[4.,8.,12.,16.][case%4]
    density=[.25,1.,2.,4.][case%4]; tension=[4.,2.,1.,.25][(case//3)%4]
    modes=[0,1,min(3,n//2-1)] if case%4 else [0]
    harmonics=[]
    for mode in modes:
        coeff=rng.uniform(-.05,.05,4)
        if mode==0: coeff[1]=coeff[3]=0
        harmonics.append(dict(zip(['mode','positionCos','positionSin','velocityCos','velocitySin'],[mode,*map(float,coeff)])))
    x=np.arange(n)*length/n; mass=density*length/n; stiffness=tension*n/length
    lap=-2*np.eye(n)+np.roll(np.eye(n),1,axis=0)+np.roll(np.eye(n),-1,axis=0)
    acc=stiffness/mass*lap
    matrix=np.block([[np.zeros((n,n)),np.eye(n)],[acc,np.zeros((n,n))]])
    q0=np.zeros(n);v0=np.zeros(n)
    for h in harmonics:
        phase=2*np.pi*h['mode']*x/length
        q0+=h['positionCos']*np.cos(phase)+h['positionSin']*np.sin(phase)
        v0+=h['velocityCos']*np.cos(phase)+h['velocitySin']*np.sin(phase)
    initial=np.r_[q0,v0];states=[]
    fft_q=np.fft.fft(q0);fft_v=np.fft.fft(v0);wave_numbers=2*np.pi*np.fft.fftfreq(n,d=length/n);cw=np.sqrt(tension/density)*np.abs(wave_numbers)
    times=[0.,.13,.75,2.,7.]
    ode=solve_ivp(lambda t,y:matrix@y,(0,7),initial,method='DOP853',rtol=2e-12,atol=2e-14,t_eval=times)
    assert ode.success
    difference=0.
    for j,t in enumerate(times):
        actual=expm(matrix*t)@initial; difference=max(difference,float(np.max(np.abs(actual-ode.y[:,j]))))
        q,v=actual[:n],actual[n:];force=mass*acc@q
        cosine=np.cos(cw*t);sine_ratio=t*np.sinc(cw*t/np.pi)
        continuum_q_coeff=fft_q*cosine+fft_v*sine_ratio
        continuum_v_coeff=fft_v*cosine-fft_q*cw*np.sin(cw*t)
        continuum_q=np.fft.ifft(continuum_q_coeff).real;continuum_v=np.fft.ifft(continuum_v_coeff).real
        continuum_gx=np.fft.ifft(1j*wave_numbers*continuum_q_coeff).real
        continuum_energy=float(length/n*np.sum((density*continuum_v**2+tension*continuum_gx**2)/2))
        link=stiffness*(np.roll(q,-1)-q)**2/2
        assigned=mass*v*v/2+(link+np.roll(link,1))/2
        flux=-stiffness*(np.roll(q,-1)-q)*(np.roll(v,-1)+v)/2
        states.append({'time':t,'position':q.tolist(),'velocity':v.tolist(),'force':force.tolist(),'energy':float(np.sum(assigned)),'assignedEnergy':assigned.tolist(),'linkFlux':flux.tolist(),'momentum':float(mass*np.sum(v)),'continuumPosition':continuum_q.tolist(),'continuumVelocity':continuum_v.tolist(),'continuumGradient':continuum_gx.tolist(),'continuumEnergy':continuum_energy,'rmsDisplacementError':float(np.sqrt(np.mean((q-continuum_q)**2)))})
    assert difference<2e-9
    chains.append({'medium':{'density':density,'tension':tension,'length':length,'particles':n},'harmonics':harmonics,'states':states,'odeExpmDifference':difference})

poly=Polynomial([1,0,-3,0,3,0,-1]); dp=poly.deriv();ddp=dp.deriv()
def profile(z):return (float(poly(z)),float(dp(z)),float(ddp(z))) if abs(z)<1 else (0.,0.,0.)
def images(s):
    d={'left':1,'right':-1,'split':0}[s['direction']]
    positive=(s['center'],s['amplitude'],s['velocityAmplitude'],d)
    if s['boundary']=='whole-line':return [positive]
    parity=-1 if s['boundary']=='fixed' else 1
    return [positive,(-s['center'],parity*s['amplitude'],parity*s['velocityAmplitude'],-d)]
def initial(s,x):
    c=np.sqrt(s['tension']/s['density']);w=s['width'];result=np.zeros(5)
    for center,a,b,d in images(s):
        p,p1,p2=profile((x-center)/w)
        result+=np.array([a*p,a*p1/w,a*p2/w**2,d*c*a*p1/w+b*p,d*c*a*p2/w**2+b*p1/w])
    return result

def pieces(s,a,b):
    cuts=[center+sign*s['width'] for center,*_ in images(s) for sign in [-1,1]]
    return sorted(set([a,*[x for x in cuts if a<x<b],b]))
def integral(f,a,b,cuts):
    points=sorted(set([a,*[x for x in cuts if a<x<b],b]))
    return sum(quad(f,x,y,epsabs=2e-12,epsrel=2e-12)[0] for x,y in zip(points,points[1:]))
def point(s,x,t,displacement=True):
    c=np.sqrt(s['tension']/s['density']); l=x-c*t;r=x+c*t
    f1,p1,pp1,g1,gp1=initial(s,l);f2,p2,pp2,g2,gp2=initial(s,r)
    v=c*(p2-p1)/2+(g2+g1)/2;gx=(p1+p2)/2+(g2-g1)/(2*c)
    out={'x':x,'time':t,'velocity':float(v),'gradient':float(gx),'acceleration':float(c*c*(pp1+pp2)/2+c*(gp2-gp1)/2),'energyDensity':float((s['density']*v*v+s['tension']*gx*gx)/2),'flux':float(-s['tension']*v*gx)}
    if displacement:
        cuts=pieces(s,l,r);out['displacement']=float((f1+f2)/2+integral(lambda y:initial(s,y)[3],l,r,cuts)/(2*c))
    return out

def balance(s,window,t):
    c=np.sqrt(s['tension']/s['density']);edges=[center+sign*s['width'] for center,*_ in images(s) for sign in [-1,1]]
    def energy(a,b,at):return integral(lambda x:point(s,x,at,False)['energyDensity'],a,b,[y+sign*c*at for y in edges for sign in [-1,1]])
    def flow(x):return integral(lambda at:point(s,x,at,False)['flux'],0,t,[(x-y)/c for y in edges]+[(y-x)/c for y in edges])
    a,b=window;e0=energy(a,b,0);e=energy(a,b,t);inflow=flow(a);outflow=flow(b)
    radius=abs(s['center'])+s['width']+c*t+1;left=-radius if s['boundary']=='whole-line' else 0
    return {'time':t,'window':window,'energy0':e0,'energyNow':e,'integratedInflow':inflow,'integratedOutflow':outflow,'totalEnergy0':energy(left,radius,0),'totalEnergyNow':energy(left,radius,t),'independentResidual':e-e0-inflow+outflow}

pulses=[]
for case in range(27):
    boundary=['whole-line','fixed','free'][case%3]
    s={'density':[.25,1.,4.][(case//3)%3],'tension':[4.,1.,.25][(case//9)%3],'amplitude':[-.08,0.,.12][(case//3)%3],'velocityAmplitude':[0.,.2,-.3][case%3],'width':[.5,1.,1.5][(case//3)%3],'center':2.,'direction':['left','right','split'][(case//9)%3],'boundary':boundary}
    c=np.sqrt(s['tension']/s['density']);coordinates=[-2.,0.,.75,2.,3.5,6.] if boundary=='whole-line' else [0.,.2,.75,2.,3.5,6.]
    states=[point(s,x,t) for t in [0.,.37,2/c,4/c] for x in coordinates]
    balances=[balance(s,[-1.,3.] if boundary=='whole-line' else [0.,3.],t) for t in [0.,.37,2/c,4/c]]
    assert max(abs(a['independentResidual']) for a in balances)<1e-9
    pulses.append({'medium':s,'states':states,'balances':balances})
x,t,c,mu=sp.symbols('x t c mu',positive=True);F=sp.Function('F');G=sp.Function('G');u=F(x-c*t)+G(x+c*t)
v=sp.diff(u,t);gx=sp.diff(u,x);e=mu*(v*v+c*c*gx*gx)/2;flux=-mu*c*c*v*gx
assert sp.simplify(sp.diff(u,t,2)-c*c*sp.diff(u,x,2))==0
assert sp.simplify(sp.diff(e,t)+sp.diff(flux,x))==0
z=sp.symbols('z');assert sp.integrate((1-z*z)**3,(z,-1,1))==sp.Rational(32,35)
out={'provenance':{'generator':'scripts/learning/generate-wave-equation-references.py','independence':'No product imports. Chain matrix exponential independently refined by DOP853; continuum from a separate sampled-data FFT propagator and Parseval/trapezoidal physical energy. Pulse data are polynomial image sources; displacement uses real scipy quad of independently extended velocity. Energy and time flux use separately split QUADPACK integrals.','symbolicPde':True,'symbolicEnergyConservation':True,'chainSystems':len(chains),'chainStates':sum(len(a['states']) for a in chains),'pulseSystems':len(pulses),'pulsePoints':sum(len(a['states']) for a in pulses),'pulseBalances':sum(len(a['balances']) for a in pulses)},'chains':chains,'pulses':pulses}
path=Path('scripts/fixtures/wave-equation-independent.json');path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['provenance'],indent=2))
