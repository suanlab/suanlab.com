"""Independent Maxwell references. Requires NumPy/SciPy; imports no site code.

Run from any directory. Reproducible fixture; differences are observations,
not certified bounds. SI dimensions; normalized positive coefficients for FD.
"""
from pathlib import Path
import itertools
import json
import math
import random
import numpy as np
from scipy.integrate import quad

rng = random.Random(82431)
waves, charges = [], []


def derivative(f, x, h=1e-4):
    return (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)


for i, (eps, mu, amplitude, k) in enumerate(itertools.product(
        [.5, 1., 2.], [.25, 1., 4.], [-2., 0., 1.], [.4, 1.3])):
    # Both propagation directions; detuned fields are explicit negative controls.
    direction = [-1, 1][i % 2]
    frequency_ratio = [1., 1., 1.2][i % 3]
    phase_offset = rng.uniform(-1, 1)
    c = 1/math.sqrt(eps*mu)
    omega = frequency_ratio*c*k
    t, z = rng.uniform(-1, 1), rng.uniform(-2, 2)
    z0, z1, span = rng.uniform(-2, 0), rng.uniform(.2, 2), .7

    def E(z, t):
        return amplitude*math.cos(direction*k*z-omega*t+phase_offset)

    def B(z, t):
        return direction*E(z, t)/c

    def flux_b(time):
        return -span*quad(lambda u: B(u, time), z0, z1, epsabs=1e-12)[0]

    def flux_e(time):
        return span*quad(lambda u: E(u, time), z0, z1, epsabs=1e-12)[0]

    curl_e = derivative(lambda u: E(u, t), z)
    curl_b = -derivative(lambda u: B(u, t), z)
    et, bt = derivative(lambda u: E(z, u), t), derivative(lambda u: B(z, u), t)
    waves.append(dict(permittivity=eps, permeability=mu, amplitude=amplitude,
                      wavenumber=k, direction=direction, phaseOffset=phase_offset,
                      frequencyRatio=frequency_ratio, time=t, z=z, bottom=z0,
                      top=z1, span=span, electric=E(z, t), magnetic=B(z, t),
                      electricCurlY=curl_e, magneticCurlX=curl_b,
                      electricTimeDerivative=et, magneticTimeDerivative=bt,
                      faradayLine=span*(E(z0, t)-E(z1, t)),
                      ampereLine=span*(B(z0, t)-B(z1, t)),
                      magneticFlux=flux_b(t), electricFlux=flux_e(t),
                      negativeMagneticFluxRate=-derivative(flux_b, t),
                      displacementFluxRate=eps*mu*derivative(flux_e, t)))

for i, (eps, radius, a0, a1) in enumerate(itertools.product(
        [.5, 1., 2.], [.5, 1., 2.], [-1., 0., 1.], [-.3, 0., .4])):
    t, offset = .7, [0., .3, -.2][i % 3]
    point = np.array([.2, -.3, .4])

    def E(p, time):
        return (a0+a1*time)*p

    def J(p):
        return -eps*a1*p

    def rho(time):
        return 3*eps*(a0+a1*time)+offset

    dt_e = np.array([derivative(lambda time: E(point, time)[j], t) for j in range(3)])
    div_j = 0
    for j in range(3):
        def component(x):
            p = point.copy()
            p[j] = x
            return J(p)[j]
        div_j += derivative(component, point[j])
    # Integrate actual Cartesian vectors against 200 sphere normals, not rho/eps.
    flux_e, current = 0., 0.
    for u in range(10):
        zz = -1+(u+.5)*2/10
        hh = math.sqrt(1-zz*zz)
        for v in range(20):
            phi = 2*math.pi*(v+.5)/20
            n = np.array([hh*math.cos(phi), hh*math.sin(phi), zz])
            weight = 4*math.pi*radius*radius/200
            flux_e += float(E(radius*n, t)@n)*weight
            current += float(J(radius*n)@n)*weight
    volume = 4*math.pi*radius**3/3
    q, qt = rho(t)*volume, derivative(lambda time: rho(time)*volume, t)
    charges.append(dict(permittivity=eps, radius=radius, initialGradient=a0,
                        gradientRate=a1, chargeOffset=offset, time=t,
                        point=point.tolist(), electric=E(point, t).tolist(),
                        electricTimeDerivative=dt_e.tolist(),
                        conductionCurrent=J(point).tolist(), currentDivergence=div_j,
                        chargeDensity=rho(t), chargeTimeDerivative=derivative(rho, t),
                        electricFlux=flux_e, enclosedCharge=q,
                        outwardConductionCurrent=current, chargeRate=qt))

# Thin-loop reference from integrating the spatial derivative, avoiding the
# endpoint cosine subtraction and avoiding the product implementation's sinc.
thin = []
for width in [1e-8, 1e-12, 1e-15]:
    bottom, top, phase = .5, .5+width, .7
    value = (top-bottom)*quad(lambda u: math.sin(bottom+(top-bottom)*u+phase), 0, 1)[0]
    thin.append(dict(bottom=bottom, top=top, phaseOffset=phase, line=value))

result = dict(provenance='Independent SciPy quadrature and five-point spatial/time differences; no site model imports.',
              planeWaves=waves, chargeBalances=charges, thinLoops=thin,
              limitations=['Finite selected normalized-coefficient cases, not a certified error bound.',
                           'Detuning and constant charge offsets deliberately violate Maxwell constraints.',
                           'Spherical flux uses actual 3D vectors; radial symmetry makes this cubature especially simple.'])
target = Path(__file__).resolve().parents[1]/'fixtures/maxwell-independent.json'
target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(planeReferences=len(waves), radialReferences=len(charges), thinReferences=len(thin))))
