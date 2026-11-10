"""Independent field differentiation/quadrature (140 digits) and DOP853 tracers.

No imports from the product. Existing fixtures are immutable; use --output to compare.
"""
import argparse
import json
import random
from pathlib import Path
import mpmath as mp
from scipy.integrate import solve_ivp
import math

mp.mp.dps = 140
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='scripts/fixtures/continuum-euler-navier-stokes-independent.json')
path = Path(parser.parse_args().output)
if path.exists():
    raise SystemExit('Refusing to overwrite an existing reference')
rng = random.Random(261106)
M = mp.mpf
def serial(v):
    if isinstance(v, dict): return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [serial(x) for x in v]
    return float(v)

def acoustic(p, x, t):
    r, c, u, k, gamma = [M(p[s]) for s in ['density', 'soundSpeed', 'meanVelocity', 'waveNumber', 'gamma']]
    a, b, e = [M(p[s]) * mp.cos(k * (x - v*t) + M(p['phase'])) for s, v in
               [('rightAmplitude', u+c), ('leftAmplitude', u-c), ('contactAmplitude', u)]]
    return [r*(a-b+e), c*(a+b), r*c*c*(a-b)]

acoustics = []
for i in range(24):
    p = dict(density=rng.uniform(.3, 3), soundSpeed=rng.uniform(.3, 3), meanVelocity=rng.uniform(-3, 3),
             gamma=rng.uniform(1.15, 1.65), waveNumber=rng.uniform(.3, 3), phase=rng.uniform(-3, 3),
             **{s: rng.uniform(-.025, .025) for s in ['rightAmplitude', 'leftAmplitude', 'contactAmplitude']})
    x, t = M(rng.uniform(-3, 3)), M(rng.uniform(.1, 2))
    q = acoustic(p, x, t)
    dx = [mp.diff(lambda z: acoustic(p, z, t)[j], x) for j in range(3)]
    dt = [mp.diff(lambda z: acoustic(p, x, z)[j], t) for j in range(3)]
    r, c, u, gamma = [M(p[s]) for s in ['density', 'soundSpeed', 'meanVelocity', 'gamma']]
    totals = [r+q[0], u+q[1], r*c*c/gamma+q[2]]
    nonlinear = [dt[0]+totals[1]*dx[0]+totals[0]*dx[1],
                 dt[1]+totals[1]*dx[1]+dx[2]/totals[0],
                 dt[2]+totals[1]*dx[2]+gamma*totals[2]*dx[1]]
    # Fixed physical window, integral of the *differentiated actual field* vs face flux.
    lo, hi = M('-.7'), M('1.1')
    rate = [mp.quad(lambda z: mp.diff(lambda tt: acoustic(p, z, tt)[j], t), [lo, hi]) for j in range(3)]
    def flux(z):
        rr, vv, pp = acoustic(p, z, t)
        return [u*rr+r*vv, u*vv+pp/r, u*pp+r*c*c*vv]
    boundary = [flux(hi)[j]-flux(lo)[j] for j in range(3)]
    acoustics.append(dict(parameters=p, x=float(x), time=float(t), expected=serial(dict(
        perturbation=q, dx=dx, dt=dt, nonlinear=nonlinear, windowRate=rate, windowFlux=boundary))))

def tg(p, x, y, t):
    r, a, k, nu = [M(p[s]) for s in ['density', 'amplitude', 'waveNumber', 'viscosity']]
    a *= mp.exp(-2*nu*k*k*t)
    return [-a*mp.cos(k*x)*mp.sin(k*y), a*mp.sin(k*x)*mp.cos(k*y),
            -r*a*a*(mp.cos(2*k*x)+mp.cos(2*k*y))/4]

vortices = []
for i in range(24):
    p = dict(density=rng.uniform(.3, 3), amplitude=rng.uniform(.2, 3), waveNumber=rng.uniform(.3, 3),
             viscosity=rng.uniform(0, .3), pressureOffset=rng.uniform(-2, 2), depth=rng.uniform(.3, 1.7))
    x, y, t = [M(rng.uniform(.1, 2)) for _ in range(3)]
    values = tg(p, x, y, t)
    grad = [[mp.diff(lambda z: tg(p, z, y, t)[j], x), mp.diff(lambda z: tg(p, x, z, t)[j], y)] for j in range(3)]
    dt = [mp.diff(lambda z: tg(p, x, y, z)[j], t) for j in range(2)]
    lap = [mp.diff(lambda z: tg(p, z, y, t)[j], x, 2)+mp.diff(lambda z: tg(p, x, z, t)[j], y, 2) for j in range(2)]
    # Integrate sampled kinetic energy and strain dissipation over a real period cell.
    k, r, nu, depth = [M(p[s]) for s in ['waveNumber', 'density', 'viscosity', 'depth']]
    side = 2*mp.pi/k
    energy, dissipation = M(0), M(0)
    for ix in range(12):
        for iy in range(12):
            xx, yy = side*ix/12, side*iy/12
            u, v, _ = tg(p, xx, yy, t)
            ux = mp.diff(lambda z: tg(p, z, yy, t)[0], xx)
            vy = mp.diff(lambda z: tg(p, xx, z, t)[1], yy)
            uy = mp.diff(lambda z: tg(p, xx, z, t)[0], yy)
            vx = mp.diff(lambda z: tg(p, z, yy, t)[1], xx)
            energy += r*(u*u+v*v)/2
            dissipation += 2*r*nu*(ux*ux+vy*vy+(uy+vx)**2/2)
    weight = side*side*depth/144
    vortices.append(dict(parameters=p, x=float(x), y=float(y), time=float(t), expected=serial(dict(
        velocity=values[:2], pressureChange=values[2], gradient=grad[:2], pressureGradient=grad[2], dt=dt, laplacian=lap,
        energy=energy*weight, dissipation=dissipation*weight))))

particles = []
for nu in [0, .07, .3]:
    p = dict(density=1.2, amplitude=1.3, waveNumber=1.1, viscosity=nu, pressureOffset=0, depth=1)
    for start in [[.35, .7], [1.1, -.4]]:
        def f(t, xy):
            x, y = xy; k = p['waveNumber']; a = p['amplitude']*math.exp(-2*nu*k*k*t)
            return [-a*math.cos(k*x)*math.sin(k*y), a*math.sin(k*x)*math.cos(k*y)]
        solution = solve_ivp(f, [0, 1.7], start, method='DOP853', rtol=3e-14, atol=3e-15, max_step=.02)
        assert solution.success
        particles.append(dict(parameters=p, start=start, time=1.7, endpoint=solution.y[:, -1].tolist()))

tiny = []
for nu, t in [('1e-20', '1'), ('1e-200', '1e-100'), ('1e-200', '1e-200')]:
    q = 2*M(nu)*M(t)
    tiny.append(dict(viscosity=nu, time=t, decayChange=str(mp.expm1(-q)), energyChange=str(mp.expm1(-2*q))))
result = dict(method='Independent mpmath140 field differentiation and physical-cell quadrature; SciPy DOP853 tracers',
              seed=261106, acousticCases=acoustics, vortexCases=vortices, particles=particles, tinyDecayCases=tiny)
path.write_text(json.dumps(result, indent=2)+'\n')
print(f'Wrote {len(acoustics)} acoustic, {len(vortices)} vortex, {len(particles)} tracer and {len(tiny)} tiny-decay references')
