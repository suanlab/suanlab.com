"""Independent spatial/temporal quadrature, differences, particle IVP and 100-digit inputs.
No product imports. Selected smooth prescribed families, not rigorous global bounds.
Run with --output for a separate repeat; existing reference files are never overwritten.
"""
from pathlib import Path
import argparse, itertools, json, math
import numpy as np
from scipy.integrate import quad, solve_ivp
import mpmath as mp

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-conservation-independent.json')
target = Path(parser.parse_args().output)
if target.exists(): raise FileExistsError(target)
transport, gases, particles = [], [], []
derivative = lambda f, t: (-f(t+2e-4)+8*f(t+1e-4)-8*f(t-1e-4)+f(t-2e-4))/(12e-4)
for rho, alpha, k, v, material in itertools.product([1.,2.],[-.5,0.,.6],[0.,.7,2.],[-.8,1.2],[False,True]):
    w = v if material else -.3
    s = dict(meanDensity=rho,contrast=alpha,wavenumber=k,velocity=v,area=1.3)
    window = dict(center=.3,halfWidth=.6,speed=w)
    density = lambda x,t: rho*(1+alpha*math.cos(k*(x-v*t)))
    mass = lambda t: 1.3*quad(lambda x:density(x,t),.3+w*t-.6,.3+w*t+.6,epsabs=2e-13)[0]
    flow = lambda t: 1.3*(v-w)*(density(.3+w*t-.6,t)-density(.3+w*t+.6,t))
    change = mass(.7)-mass(-.4)
    integral = quad(flow,-.4,.7,epsabs=2e-13)[0]
    assert abs(change-integral)<2e-12
    transport.append(dict(system=s,window=window,time=.2,mass=mass(.2),massRate=derivative(mass,.2),change=change,integral=integral))

for rho,p0,a,v,gamma in itertools.product([1.,2.],[0.,1.5],[-.4,0.,.3],[-.6,.7],[1.4,5/3]):
    s = dict(initialDensity=rho,initialPressure=p0,expansionRate=a,drift=v,gamma=gamma,area=1.3)
    def primitive(x,t):
        stretch=1+a*t; density=rho/stretch; u=(v+a*x)/stretch; p=p0*stretch**(-gamma)
        E=p/(gamma-1)+density*u*u/2
        return np.array([density,density*u,E]),np.array([density*u,density*u*u+p,(E+p)*u]),p,u
    x,t=.23,.2
    qt=derivative(lambda tau:primitive(x,tau)[0],t)
    fx=derivative(lambda z:primitive(z,t)[1],x)
    assert np.max(np.abs(qt+fx))<2e-10
    for mode in ['fixed','translating','material']:
        window=dict(center=.2,halfWidth=.6,speed=-.3,mode=mode)
        def geometry(t):
            if mode=='material': return ((1+a*t)*(-.4)+v*t,(1+a*t)*.8+v*t,v-.4*a,v+.8*a)
            w=-.3 if mode=='translating' else 0.
            return (-.4+w*t,.8+w*t,w,w)
        def storage(t):
            L,R,_,_=geometry(t)
            return np.array([1.3*quad(lambda x:primitive(x,t)[0][i],L,R,epsabs=2e-12)[0]for i in range(3)])
        def flow(t,pressure=True):
            L,R,wL,wR=geometry(t);UL,FL,pL,uL=primitive(L,t);UR,FR,pR,uR=primitive(R,t)
            result=1.3*((FL-wL*UL)-(FR-wR*UR))
            if not pressure: result[2]-=1.3*(pL*uL-pR*uR)
            return result
        values=storage(.2);rate=derivative(storage,.2)
        integrated=np.array([quad(lambda t:flow(t)[i],-.4,.7,epsabs=2e-12)[0]for i in range(3)])
        change=storage(.7)-storage(-.4)
        assert np.max(np.abs(rate-flow(.2)))<2e-10 and np.max(np.abs(change-integrated))<2e-11
        gases.append(dict(system=s,window=window,time=.2,storage=values.tolist(),rate=rate.tolist(),change=change.tolist(),integrated=integrated.tolist(),withoutPressureWork=flow(.2,False).tolist(),localConserved=primitive(x,t)[0].tolist(),localFlux=primitive(x,t)[1].tolist(),localTimeDerivative=qt.tolist(),localFluxDerivative=fx.tolist()))
    sol=solve_ivp(lambda t,x:[primitive(float(x[0]),t)[3]],(0.,.7),[.8],rtol=2e-12,atol=2e-14)
    assert abs(sol.y[0,-1]-((1+.7*a)*.8+.7*v))<2e-11
    particles.append(dict(system=s,initialPosition=.8,time=.7,position=float(sol.y[0,-1])))

mp.mp.dps=100
tiny=[]
for a in [1e-20,-1e-20,1e-12]:
    A, rho, p, gamma, h, t = map(mp.mpf,[1.,1.,1.,1.4,.5,.5])
    rate=mp.mpf(a);stretch=1+rate*t
    # Integrate the actual material-volume energy; constant particle speeds leave KE unchanged.
    initial=2*A*h*p/(gamma-1)
    change=2*A*h*stretch*(p*stretch**(-gamma))/(gamma-1)-initial
    tiny.append(dict(expansionRate=a,time=.5,materialInternalChange=float(change),exact=str(change)))
rho,alpha,k,v,w,c,h,t0,t1=map(mp.mpf,[1.,.6,1e-12,1.,-.3,.3,.6,-.4,.7])
mass=lambda t:mp.quad(lambda x:rho*(1+alpha*mp.cos(k*(x-v*t))),[c+w*t-h,c+w*t+h])
tiny_transport=dict(wavenumber=1e-12,change=float(mass(t1)-mass(t0)),exact=str(mass(t1)-mass(t0)))
result=dict(method='Independent SciPy spatial/time quadrature,five-point conservation derivatives,particle IVP,and100-digit exact-binary-input differences. No product imports;selected observations,not global error bounds.',transportCases=transport,gasCases=gases,particleCases=particles,tinyMaterialEnergyChanges=tiny,tinyTransportChange=tiny_transport)
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:len(v)for k,v in result.items()if isinstance(v,list)}))
