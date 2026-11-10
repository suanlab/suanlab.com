"""Independent portable references. No product imports. SciPy Fourier quadrature,
real-space moments, matrix exponential of physical equations, and FFT analytic
reconstruction. Run from repository root with Python + numpy/scipy/sympy.
"""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
import sympy as sp


def pair(z):
    return [float(z.real), float(z.imag)]


def cq(f):
    return quad(lambda y: float(f(y).real), -14, 14, epsabs=2e-12, epsrel=2e-12, limit=400)[0] + 1j * quad(
        lambda y: float(f(y).imag), -14, 14, epsabs=2e-12, epsrel=2e-12, limit=400)[0]


gaussians = []
for index, (sigma, beta, t) in enumerate(itertools.product([.5, .9, 1.25], [-.7, 0, .9], [-1, 0, .6, 1.8])):
    s = dict(amplitude=0 if index == 35 else .7, sigma=sigma, center=.3,
        carrierWaveNumber=0 if index % 7 == 0 else 3.2,
        carrierFrequency=5.3, groupVelocity=1.1, dispersion=beta, phase=.2)
    width = sigma*np.hypot(1, beta*t/sigma**2)
    center = s['center'] + s['groupVelocity']*t
    states = []
    for offset in [-2.1, -.4, 0, .8, 2.2]:
        x = center + offset*width
        def integrand(y):
            p = y/sigma
            return s['amplitude']/np.sqrt(2*np.pi) * np.exp(-y*y/2 + 1j*((s['carrierWaveNumber']+p)*x
                - (s['carrierFrequency']+s['groupVelocity']*p+beta*p*p/2)*t + s['phase'] - p*s['center']))
        field = cq(integrand)
        dx = cq(lambda y: integrand(y)*1j*y/sigma)
        dxx = cq(lambda y: -integrand(y)*(y/sigma)**2)
        dt = cq(lambda y: -1j*integrand(y)*(s['groupVelocity']*y/sigma + beta*(y/sigma)**2/2))
        states.append(dict(x=x, field=pair(field), envelopeDerivativeX=pair(dx),
            envelopeDerivativeXX=pair(dxx), envelopeDerivativeT=pair(dt)))
    density = lambda u: s['amplitude']**2*sigma/width*np.exp(-u*u/width**2)
    norm = quad(density, -np.inf, np.inf, epsabs=2e-12)[0]
    variance = None if norm == 0 else quad(lambda u: u*u*density(u), -np.inf, np.inf, epsabs=2e-12)[0]/norm
    gaussians.append(dict(system=s, time=t, states=states, norm=norm, intensityVariance=variance))

chains = []
for index, (n, carrier, bandwidth, t) in enumerate(itertools.product([32, 64], [4, 10.3], [.2, 2, 5], [0, 1.3, 4])):
    s = dict(particles=n, length=8, density=.8, tension=1.7, amplitude=0 if index == 35 else .08,
        modeCenter=carrier, bandwidth=bandwidth, center=2.96, phase=.4)
    a = s['length']/n
    m, stiffness = s['density']*a, s['tension']/a
    c = np.sqrt(s['tension']/s['density'])
    indices = np.arange(1, n//2)
    k = 2*np.pi*indices/s['length']
    omega = 2*c/a*np.sin(k*a/2)
    weight = np.exp(-.5*((indices-carrier)/bandwidth)**2)
    amplitude = s['amplitude']*weight/sum(weight)
    phase = s['phase'] - k*s['center']
    x = a*np.arange(n)
    q0 = np.sum(amplitude[:,None]*np.cos(k[:,None]*x + phase[:,None]), axis=0)
    v0 = np.sum((omega*amplitude)[:,None]*np.sin(k[:,None]*x + phase[:,None]), axis=0)
    lap = -2*np.eye(n)+np.roll(np.eye(n),1,axis=1)+np.roll(np.eye(n),-1,axis=1)
    matrix = np.block([[np.zeros((n,n)), np.eye(n)], [stiffness/m*lap, np.zeros((n,n))]])
    state = expm(t*matrix) @ np.r_[q0,v0]
    q, v = state[:n], state[n:]
    dq = np.roll(q,-1)-q
    kinetic = m*v*v/2
    link_energy = stiffness*dq*dq/2
    assigned = kinetic+(link_energy+np.roll(link_energy,1))/2
    flux = -stiffness*dq*(np.roll(v,-1)+v)/2
    energy = float(sum(assigned))
    coeff = np.fft.fft(q)
    positive = np.zeros(n, dtype=complex)
    positive[1:n//2] = 2*coeff[1:n//2]
    analytic = np.fft.ifft(positive)
    modal_energy = m*n*omega**2*amplitude**2/2
    group = c*np.cos(k*a/2)
    def circular(w):
        total = sum(w)
        moment = 0j if total == 0 else sum(w*np.exp(2j*np.pi*x/s['length']))/total
        return dict(resultant=float(abs(moment)), position=None if abs(moment)<1e-10 else float((np.angle(moment)/(2*np.pi)+1)%1*s['length']))
    chains.append(dict(system=s,time=t,position=q.tolist(),velocity=v.tolist(),
        analyticField=[pair(z) for z in analytic],assignedEnergy=assigned.tolist(),linkFlux=flux.tolist(),
        energy=energy,modalEnergy=float(sum(modal_energy)),energyTransportVelocity=None if energy==0 else float(a*sum(flux)/energy),
        energyWeightedGroupVelocity=None if energy==0 else float(sum(modal_energy*group)/sum(modal_energy)),
        analyticNorm=float(a*sum(abs(analytic)**2)),analyticCenter=circular(abs(analytic)**2),energyCenter=circular(assigned)))

# The logarithmic PDE residual is rational, so checking it symbolically does not
# inherit branch choices from a symbolic square root.
y,r,b,beta,vg=sp.symbols('y r b beta vg',real=True)
D=r+sp.I*b
logx=-y/D
logxx=y*y/D**2-1/D
logt=-sp.I*beta/(2*D)+vg*y/D+sp.I*beta*y*y/(2*D**2)
residual=sp.simplify(sp.I*(logt+vg*logx)+beta*logxx/2)
assert residual==0
output=dict(provenance=dict(generator='scripts/learning/generate-dispersion-references.py',productImports=False,
    gaussian='QUADPACK initial Fourier inversion and derivative integrals; real-space moments',
    chain='matrix exponential of physical real equations; independent FFT analytic-field reconstruction',
    symbolicEnvelopePdeResidual=str(residual)),gaussians=gaussians,chains=chains)
Path('scripts/fixtures/dispersion-independent.json').write_text(json.dumps(output,indent=2)+'\n')
print(f'Wrote{len(gaussians)} Gaussian systems/{sum(len(r["states"]) for r in gaussians)} points and{len(chains)} physical chain states. Symbolic PDE residual={residual}.')
