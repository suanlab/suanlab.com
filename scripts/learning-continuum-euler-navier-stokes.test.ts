import test from 'node:test';
import assert from 'node:assert/strict';
import { linearizedEulerAt as acoustic, taylorGreenAt as vortex, taylorGreenCell as cell, taylorGreenDecay as decay,
  taylorGreenParticle as particle, type AcousticParameters, type TaylorGreenParameters, type FlowValue } from '../src/lib/learning-continuum-euler-navier-stokes';
import references from './fixtures/continuum-euler-navier-stokes-independent.json';
const a: AcousticParameters = { density: 1.2, soundSpeed: 2, meanVelocity: .5, gamma: 1.4, waveNumber: 1,
  rightAmplitude: .02, leftAmplitude: -.01, contactAmplitude: .015, phase: .3 };
const g: TaylorGreenParameters = { density: 1.2, amplitude: 2, waveNumber: 1, viscosity: .1, pressureOffset: .7, depth: 1.5 };
function near(actual: FlowValue, expected: number, tolerance = 5e-11) {
  assert.notEqual(actual, null); assert.ok(Math.abs(actual! - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);
}
function relative(actual: FlowValue, expected: number) {
  assert.notEqual(actual, null); assert.ok(Math.abs(actual! / expected - 1) < 2e-12, `${actual} vs ${expected}`);
}
function finiteOrMissing(v: unknown): void {
  if (typeof v === 'number') assert.ok(Number.isFinite(v));
  else if (v && typeof v === 'object') Object.values(v).forEach(finiteOrMissing);
}
test('24 independent acoustic field values and differentiated components match', () => {
  assert.equal(references.acousticCases.length, 24);
  for (const r of references.acousticCases) {
    const q = acoustic(r.parameters, r.x, r.time);
    for (const [j, key] of ['density', 'velocity', 'pressure'].entries()) {
      near(q.perturbation[key as keyof typeof q.perturbation], r.expected.perturbation[j]);
      near(q.dx[key as keyof typeof q.dx], r.expected.dx[j]); near(q.dt[key as keyof typeof q.dt], r.expected.dt[j]);
    }
    Object.values(q.linearResidual).forEach(v => near(v, 0)); near(q.energyResidual, 0);
    Object.values(q.nonlinearResidual).forEach((v, j) => near(v, r.expected.nonlinear[j]));
    assert.ok(q.positiveBackgroundAndTotals);
  }
});
test('24 fixed-window rates balance actual linear face flux, without periodic-window assumptions', () => {
  for (const r of references.acousticCases) for (let j = 0; j < 3; j++) near(r.expected.windowRate[j] + r.expected.windowFlux[j], 0, 1e-12);
  // Product fields, numerical midpoint integration, independent finite difference in time.
  const lo = -.7, hi = 1.1, h = 1e-5, t = .4, count = 4096;
  const integral = (time: number, key: 'density' | 'velocity' | 'pressure') => {
    let value = 0; for (let i = 0; i < count; i++) value += acoustic(a, lo + (i + .5) * (hi - lo) / count, time).perturbation[key]!;
    return value * (hi - lo) / count;
  };
  const flux = (x: number) => { const q = acoustic(a, x, t).perturbation; return [a.meanVelocity*q.density! + a.density*q.velocity!, a.meanVelocity*q.velocity! + q.pressure!/a.density, a.meanVelocity*q.pressure! + a.density*a.soundSpeed**2*q.velocity!]; };
  for (const [j, key] of (['density', 'velocity', 'pressure'] as const).entries()) near((integral(t+h,key)-integral(t-h,key))/(2*h)+flux(hi)[j]-flux(lo)[j],0,3e-8);
});
test('three characteristics translate at U+c, U-c and U rather than all following the fluid', () => {
  const q = acoustic(a, .3, .2), dt = .17;
  q.modes.forEach((m, j) => near(acoustic(a, .3 + m.speed * dt, .2 + dt).modes[j].value, m.value!));
  assert.deepEqual(q.characteristicSpeeds, [2.5, -1.5, .5]);
  near(q.rightInvariant, q.perturbation.velocity! + q.perturbation.pressure!/(a.density*a.soundSpeed));
  near(q.leftInvariant, q.perturbation.velocity! - q.perturbation.pressure!/(a.density*a.soundSpeed));
});
test('pressure-free contact mode has density transport but zero acoustic energy', () => {
  const q = acoustic({ ...a, rightAmplitude: 0, leftAmplitude: 0 }, .7, .2);
  near(q.perturbation.velocity, 0); near(q.perturbation.pressure, 0); near(q.acousticEnergy, 0);
  assert.notEqual(q.perturbation.density, 0); Object.values(q.nonlinearResidual).forEach(v => near(v, 0));
  near(q.contactVariable, -(a.soundSpeed**2)*q.perturbation.density!);
});
test('nonlinear Euler residuals have the expected quadratic small-amplitude scale', () => {
  const small = acoustic({ ...a, rightAmplitude:a.rightAmplitude/2,leftAmplitude:a.leftAmplitude/2,contactAmplitude:a.contactAmplitude/2 }, .7, .2);
  const large = acoustic(a, .7, .2);
  near(large.nonlinearResidual.mass, 4*small.nonlinearResidual.mass!);
  near(large.nonlinearResidual.pressure, 4*small.nonlinearResidual.pressure!);
  assert.ok(Math.abs(large.nonlinearResidual.velocity!/small.nonlinearResidual.velocity! - 4) < .2);
  assert.ok(Math.abs(large.nonlinearResidual.mass!) > 1e-6);
});
test('quadratic acoustic energy has its own flux Uq+p-prime u-prime', () => {
  const x=.4,t=.3,h=1e-5;
  const rate=(acoustic(a,x,t+h).acousticEnergy!-acoustic(a,x,t-h).acousticEnergy!)/(2*h);
  const divergence=(acoustic(a,x+h,t).energyFlux!-acoustic(a,x-h,t).energyFlux!)/(2*h);
  near(rate+divergence,0,1e-10);
});
test('supported amplitudes bound positive density and pressure globally, including both signs', () => {
  for (const c of [.25,4]) for (const r of [.25,4]) for (const gamma of [1.1,1.67]) for (const sign of [-1,1]) {
    const q=acoustic({...a,density:r,soundSpeed:c,gamma,rightAmplitude:sign*.03,leftAmplitude:-sign*.03,contactAmplitude:sign*.03},.5,1);
    assert.ok(q.positiveBackgroundAndTotals); near(q.maxRelativeDensityBound,.09); near(q.maxRelativePressureBound,gamma*.06);
  }
});
test('24 independently differentiated Taylor-Green vector fields match every momentum term', () => {
  assert.equal(references.vortexCases.length,24);
  for (const r of references.vortexCases) {
    const q=vortex(r.parameters,r.x,r.y,r.time),e=r.expected;
    near(q.pressureChange,e.pressureChange); near(q.divergence,0);
    for(let j=0;j<2;j++) {
      near(q.velocity[j],e.velocity[j]); near(q.pressureGradient[j],e.pressureGradient[j]);
      near(q.timeDerivative[j],e.dt[j]); near(q.laplacian[j],e.laplacian[j]);
      for(let k=0;k<2;k++)near(q.gradient[j][k],e.gradient[j][k]);
      near(q.residual[j],0); near(q.omittedPressureResidual[j],q.pressureAcceleration[j]!);
    }
  }
});
test('24 independently integrated physical period cells match mass, energy and strain dissipation', () => {
  for(const r of references.vortexCases){const q=cell(r.parameters,r.time);near(q.kineticEnergy,r.expected.energy);near(q.dissipation,r.expected.dissipation);near(q.energyTimeDerivative,-q.dissipation!);near(q.mass,r.parameters.density*q.side**2*r.parameters.depth);}
});
test('actual finite differences verify incompressibility, viscosity and pressure signs', () => {
  const x=.4,y=.8,t=.3,h=2e-4,q=vortex(g,x,y,t);
  const xx=[vortex(g,x-h,y,t),vortex(g,x+h,y,t)],yy=[vortex(g,x,y-h,t),vortex(g,x,y+h,t)],tt=[vortex(g,x,y,t-h),vortex(g,x,y,t+h)];
  near((xx[1].velocity[0]!-xx[0].velocity[0]!)/(2*h)+(yy[1].velocity[1]!-yy[0].velocity[1]!)/(2*h),0,2e-9);
  for(let j=0;j<2;j++) {
    const ux=(xx[1].velocity[j]!-xx[0].velocity[j]!)/(2*h),uy=(yy[1].velocity[j]!-yy[0].velocity[j]!)/(2*h);
    const dp=((j===0?xx:yy)[1].pressure!-(j===0?xx:yy)[0].pressure!)/(2*h);
    const lap=(xx[1].velocity[j]!+xx[0].velocity[j]!+yy[1].velocity[j]!+yy[0].velocity[j]!-4*q.velocity[j]!)/(h*h);
    near((tt[1].velocity[j]!-tt[0].velocity[j]!)/(2*h)+q.velocity[0]!*ux+q.velocity[1]!*uy+dp/g.density-g.viscosity*lap,0,6e-8);
  }
});
test('zero viscosity is a steady nonlinear Euler vortex with nonzero pressure acceleration', () => {
  const p={...g,viscosity:0},q=vortex(p,.4,.8,3),initial=vortex(p,.4,.8,0);
  assert.deepEqual(q.velocity,initial.velocity); q.timeDerivative.forEach(v=>near(v,0)); q.viscousAcceleration.forEach(v=>near(v,0));
  q.residual.forEach(v=>near(v,0)); assert.ok(Math.abs(q.convection[0]!)>.01); near(cell(p,3).dissipation,0);
});
test('pressure gauge changes no force, particle motion, energy or dissipation', () => {
  const q=vortex(g,.4,.8,.3),r=vortex({...g,pressureOffset:-2},.4,.8,.3);
  near(r.pressure!-q.pressure!,-2-g.pressureOffset);assert.deepEqual(q.pressureGradient,r.pressureGradient);assert.deepEqual(q.residual,r.residual);
  assert.deepEqual(cell(g,.3),cell({...g,pressureOffset:-2},.3));
});
test('physical cell length, vorticity sign and periodic field are explicit', () => {
  const side=cell(g,.3).side, q=vortex(g,.2,.4,.3),r=vortex(g,.2+side,.4+side,.3);
  q.velocity.forEach((v,j)=>near(r.velocity[j],v!));near(q.pressureChange,r.pressureChange!);
  near(vortex(g,0,0,0).vorticity,2*g.amplitude*g.waveNumber);
  // Pure local rotation need not dissipate: local strain, rather than vorticity alone, enters Phi.
  near(vortex(g,0,0,0).localDissipation,0);
});
test('six RK4 particle endpoints match independent DOP853 and preserve their streamline label', () => {
  assert.equal(references.particles.length,6);
  for(const r of references.particles){const q=particle(r.parameters,r.start as[number,number],r.time,256);assert.equal(q.status,'integrated');const end=q.points.at(-1)!;near(end.x,r.endpoint[0],2e-10);near(end.y,r.endpoint[1],2e-10);
    const k=r.parameters.waveNumber,label=Math.cos(k*r.start[0])*Math.cos(k*r.start[1]);near(Math.cos(k*end.x)*Math.cos(k*end.y),label,2e-10);
  }
});
test('RK4 time refinement reduces tracer endpoint error rather than substituting streamlines', () => {
  const r=references.particles[0],error=(steps:number)=>{const end=particle(r.parameters,r.start as[number,number],r.time,steps).points.at(-1)!;return Math.hypot(end.x-r.endpoint[0],end.y-r.endpoint[1]);};
  assert.ok(error(16)/error(32)>12);assert.ok(error(32)/error(64)>12);
  const q=particle({...g,amplitude:0},[.4,.8],2,8);q.points.forEach(p=>{near(p.x,.4);near(p.y,.8);});
});
test('stable expm1 changes survive rounded equality and agree with 140-digit references', () => {
  for(const r of references.tinyDecayCases.slice(0,2)){const p={...g,amplitude:1,waveNumber:1,viscosity:Number(r.viscosity)},q=decay(p,Number(r.time));assert.equal(q.roundedDecay,1);relative(q.stableDecayChange,Number(r.decayChange));relative(q.stableEnergyChange,Number(r.energyChange));assert.equal(q.status,'change-below-rounded-equality');assert.equal(cell(p,Number(r.time)).roundedEnergyChange,0);assert.notEqual(cell(p,Number(r.time)).stableEnergyChange,0);}
});
test('nonzero sub-range changes and energy are missing rather than physical zero', () => {
  const q=decay({...g,waveNumber:1,viscosity:1e-200},1e-200);assert.equal(q.roundedDecay,1);assert.equal(q.stableDecayChange,null);assert.equal(q.status,'change-arithmetic-unavailable');
  assert.equal(cell({...g,amplitude:Number.MIN_VALUE},1).kineticEnergy,null);
  const tiny=acoustic({...a,rightAmplitude:1e-200,leftAmplitude:0,contactAmplitude:0,phase:0},0,0);assert.equal(tiny.acousticEnergy,null);assert.notEqual(tiny.perturbation.velocity,0);
  assert.equal(decay({...g,viscosity:0},1).stableDecayChange,0);assert.equal(decay(g,0).stableDecayChange,0);
});
test('zero amplitude is a rest state even when a nonzero decay change is unavailable', () => {
  const q=vortex({...g,amplitude:0,viscosity:1e-200},.4,.8,1e-200);assert.deepEqual(q.velocity,[0,0]);near(q.kineticEnergyDensity,0);near(q.localDissipation,0);q.residual.forEach(v=>near(v,0));
});
test('tracer distinguishes zero elapsed time, unrepresentable steps and rounded coordinate updates', () => {
  const zero=particle(g,[.4,.8],0);assert.equal(zero.status,'integrated');assert.equal(zero.points.length,1);
  assert.equal(particle(g,[.4,.8],Number.MIN_VALUE,256).status,'increment-arithmetic-unavailable');
  const tiny=particle({...g,amplitude:1e-20},[.4,.8],1,8);assert.equal(tiny.status,'integrated');assert.equal(tiny.roundedCoordinateUpdates,16);
  assert.equal(particle({...g,amplitude:Number.MIN_VALUE},[.4,.8],1).status,'arithmetic-unavailable');
});
test('invalid inputs fail before observations and accepted extremes never leak NaN or infinity', () => {
  for(const value of [NaN,Infinity,-1,5])assert.throws(()=>vortex({...g,density:value},0,0,0),RangeError);
  assert.throws(()=>acoustic({...a,rightAmplitude:.04},0,0),RangeError);assert.throws(()=>vortex({...g,viscosity:-1},0,0,0),RangeError);
  assert.throws(()=>particle(g,[0,0],1,1.5),RangeError);assert.throws(()=>vortex(g,33,0,0),RangeError);
  for(const amplitude of [0,Number.MIN_VALUE,1e-200,4])for(const viscosity of [0,Number.MIN_VALUE,1])finiteOrMissing(vortex({...g,amplitude,viscosity,waveNumber:4},1.2,-2.1,4));
});
