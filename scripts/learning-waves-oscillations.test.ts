import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/waves-oscillations-independent.json';
import {forcedOscillatorState as state, oscillatorFrequencyResponse as response, sinusoidalForcingTransient as transient} from '../src/lib/learning-waves-oscillations';
const system = {mass: 1, stiffness: 1, damping: 1, forceAmplitude: 1, frequency: 1};
const zero = {position: 0, velocity: 0};
const close = (a: number, b: number, tolerance = 2e-9) => assert.ok(Number.isFinite(a) && Math.abs(a - b) <= tolerance * (1 + Math.abs(b)), `${a} != ${b}`);
function finite(value: unknown): void {
  if (typeof value === 'number') assert.ok(Number.isFinite(value));
  else if (Array.isArray(value)) value.forEach(finite);
  else if (value && typeof value === 'object') Object.values(value).forEach(finite);
}
test('58 independent refined trajectories agree at all 522 queried times including resonance and stiff endpoints', () => {
  assert.equal(refs.trajectories.length, 58);
  for (const row of refs.trajectories) {
    assert.ok(row.refinementDifference < 2e-7);
    assert.ok(row.independentExpmDifference < 2e-9);
    for (const expected of row.samples) {
      const actual = state(row.system, row.initial, expected.time);
      close(actual.position, expected.position); close(actual.velocity, expected.velocity);
    }
  }
});
test('adaptive startup work and heat match independent integration even at maximum damping and duration', () => {
  for (const row of refs.trajectories) {
    const r = transient(row.system, row.initial, row.duration);
    assert.equal(r.quadrature.completed, true);
    assert.ok(r.quadrature.evaluations <= r.quadrature.evaluationBudget);
    close(r.work, row.work, 2e-8); close(r.heat, row.heat, 2e-8);
    assert.ok(Math.abs(r.balanceResidual) < 2e-7);
    assert.ok(r.heat >= 0);
    finite(r);
  }
});
test('frequency response agrees with independent real force/velocity cycle quadratures', () => {
  assert.ok(refs.periodic.length >= 20);
  for (const row of refs.periodic) {
    const r = response(row.system);
    close(r.meanPower!, row.meanPower, 1e-11); close(r.meanPower!, row.meanHeat, 1e-11);
    close(r.gain!, row.gain, 1e-12); close(r.phaseLag!, row.lag, 1e-12);
  }
});
test('exact undamped resonance has finite secular evolution and no bounded periodic particular', () => {
  const s = {...system, damping: 0}, t = Math.PI / 2, r = state(s, zero, t);
  close(r.position, Math.PI / 4, 1e-12); close(r.velocity, .5, 1e-12);
  close(r.energy, .125 + Math.PI ** 2 / 32, 1e-12);
  assert.equal(response(s).amplitude, null);
  assert.equal(response(s).status, 'no-bounded-periodic-response');
  assert.equal(transient(s, zero, t).particularInitial, null);
});
test('near resonance is continuous without clamping the frequency to resonance', () => {
  const t = 12, resonant = state({...system, damping: 0}, zero, t);
  for (const delta of [-1e-9, 1e-9]) {
    const s = {...system, damping: 0, frequency: 1 + delta}, r = state(s, zero, t);
    close(r.position, resonant.position, 1e-7);
    assert.equal(response(s).status, 'nonattracting-particular');
    assert.notEqual(response(s).gain, null);
  }
});
test('three characteristic frequencies are distinct and the half-power width is exact', () => {
  const r = response(system);
  close(r.displacementPeakFrequency!, Math.sqrt(.5)); close(r.dampedFreeFrequency!, Math.sqrt(.75));
  assert.equal(r.powerPeakFrequency, 1);
  close(r.halfPower!.upper - r.halfPower!.lower, 1);
  close(r.halfPower!.upper * r.halfPower!.lower, 1);
  for (const w of [r.halfPower!.lower, r.halfPower!.upper]) close(response({...system, frequency: w}).meanPower!, .25);
});
test('underdamped free motion need not have an interior displacement resonance peak', () => {
  const r = response({...system, stiffness: 2, damping: 2});
  assert.equal(r.dampingRegime, 'underdamped'); assert.equal(r.displacementPeakFrequency, null);
  assert.equal(r.displacementPeakStatus, 'maximum-at-dc'); close(r.dampedFreeFrequency!, 1);
});
test('critical and overdamped systems still admit attracting forced periodic responses', () => {
  for (const damping of [2, 4]) {
    const r = response({...system, damping});
    assert.equal(r.dampedFreeFrequency, null); assert.equal(r.status, 'attracting-particular');
    assert.ok(r.amplitude! > 0); assert.ok(r.meanPower! > 0);
  }
});
test('selected initial data isolate sin(t) and its real mean input power', () => {
  const r = transient(system, {position: 0, velocity: 1}, 2 * Math.PI);
  assert.deepEqual(r.homogeneousInitial, zero);
  for (const p of r.states) {close(p.position, Math.sin(p.time), 1e-11); close(p.velocity, Math.cos(p.time), 1e-11);}
  close(r.work, Math.PI, 1e-11); close(r.heat, Math.PI, 1e-11);
  close(r.finiteWindowMeanInputPower!, .5, 1e-11);
});
test('startup power differs from the periodic asymptotic mean', () => {
  const r = transient(system, zero, 1);
  assert.ok(Math.abs(r.finiteWindowMeanInputPower! - r.response.meanPower!) > .1);
  assert.ok(Math.abs(r.balanceResidual) > 1e-15);
  assert.ok(Math.abs(r.balanceResidual) < 1e-8);
});
test('instantaneous and finite-window drive work can be negative while dissipated heat is positive', () => {
  const r = transient(system, {position: 0, velocity: -1}, .1);
  assert.ok(r.work < 0); assert.ok(r.heat > 0); assert.ok(r.terminal.inputPower < 0);
});
test('DC and zero duration have no cycle or finite-window mean; static equilibrium energy is explicit', () => {
  const s = {...system, frequency: 0}, r = transient(s, {position: 1, velocity: 0}, 0);
  assert.equal(r.response.period, null); assert.equal(r.response.meanStoredEnergy, null);
  assert.equal(r.response.phaseLag, null); assert.equal(r.response.intrinsicLag, 0);
  assert.equal(r.response.heatPerCycle, null); assert.equal(r.finiteWindowMeanInputPower, null);
  assert.equal(r.work, 0); assert.equal(r.heat, 0); assert.equal(r.terminal.energy, .5);
  const later = state(s, {position: 1, velocity: 0}, 8); close(later.position, 1); close(later.velocity, 0);
});
test('zero forcing removes measured phase without deleting the intrinsic transfer or free motion', () => {
  const s = {...system, forceAmplitude: 0}, r = response(s);
  assert.equal(r.amplitude, 0); assert.equal(r.phaseLag, null); assert.equal(r.intrinsicLag, Math.PI / 2);
  assert.equal(r.powerPeakFrequency, null); assert.equal(r.meanPower, 0);
  assert.ok(state(s, {position: 1, velocity: 0}, .1).position > .9);
  const singular = response({...s, damping: 0});
  assert.equal(singular.status, 'singular-transfer-zero-input'); assert.equal(singular.amplitude, 0);
  assert.equal(singular.gain, null); assert.equal(singular.meanPower, 0);
  const free = transient({...s, damping: 0}, {position: 1, velocity: 0}, 2);
  assert.deepEqual(free.particularInitial, zero);
  assert.deepEqual(free.homogeneousInitial, {position: 1, velocity: 0});
  close(free.terminal.position, Math.cos(2), 1e-11);
});
test('the exp(+iwt) convention gives a negative imaginary susceptibility and lag in the correct quadrant', () => {
  const r = response({...system, frequency: 2});
  assert.ok(r.susceptibility!.imaginary < 0); assert.ok(r.phaseLag! > Math.PI / 2);
  close(r.susceptibility!.real, -3 / 13); close(r.susceptibility!.imaginary, -2 / 13);
});
test('undamped off-resonant free oscillation persists rather than attracting the particular solution', () => {
  const s = {...system, damping: 0, frequency: 2}, r = response(s), t = 1;
  assert.equal(r.qualityFactor, null); assert.equal(r.halfPower, null);
  close(state(s, zero, t).position, (Math.cos(2 * t) - Math.cos(t)) / -3, 1e-12);
});
test('an actual state derivative satisfies the forced equation independently of reported acceleration', () => {
  const h = 1e-4;
  for (const damping of [0, 1, 2, 4]) {
    const s = {...system, damping}, t = .8, a = state(s, zero, t - h), b = state(s, zero, t), c = state(s, zero, t + h);
    close((c.position - a.position) / (2 * h), b.velocity, 1e-8);
    const acceleration = (c.velocity - a.velocity) / (2 * h);
    close(s.mass * acceleration + s.damping * b.velocity + s.stiffness * b.position, Math.cos(t), 1e-8);
  }
});
test('invalid, missing and nonfinite physical inputs are rejected without repairs', () => {
  for (const key of Object.keys(system)) for (const value of [NaN, Infinity, -Infinity, undefined]) assert.throws(() => response({...system, [key]: value} as typeof system));
  for (const mass of [0, -.1, 5]) assert.throws(() => response({...system, mass}));
  for (const time of [-1, 33, NaN, Infinity]) assert.throws(() => state(system, zero, time));
  assert.throws(() => state(system, {position: 5, velocity: 0}, 1));
});
test('analytic peak claims survive neighboring-frequency checks for several damping strengths', () => {
  for (const damping of [.25, .5, 1, 2, 4, 8]) {
    const s = {...system, damping}, r = response(s), delta = 1e-3;
    for (const frequency of [1 - delta, 1 + delta]) assert.ok(response({...s, frequency}).meanPower! < r.meanPower!);
    if (r.displacementPeakFrequency !== null) {
      const w = r.displacementPeakFrequency, peak = response({...s, frequency: w}).gain!;
      for (const frequency of [w - delta, w + delta]) assert.ok(response({...s, frequency}).gain! < peak);
    } else assert.ok(response({...s, frequency: 0}).gain! > response({...s, frequency: delta}).gain!);
    close(r.halfPower!.width, damping);
    close(r.halfPower!.lower * r.halfPower!.upper, 1);
  }
});
test('the energy-to-cycle-heat quality ratio equals Q at resonance but differs away from it', () => {
  const r = response(system), off = response({...system, frequency: 2});
  close(2 * Math.PI * r.meanStoredEnergy! / r.heatPerCycle!, r.qualityFactor!);
  assert.ok(Math.abs(2 * Math.PI * off.meanStoredEnergy! / off.heatPerCycle! - off.qualityFactor!) > .1);
});
for (const id of ['sinusoidal-forcing-transient-response', 'oscillator-frequency-response-power-resonance']) {
  for (const lang of ['ko', 'en'] as const) test(`${id}.${lang}: full manuscript renders strict math and a unique bilingual outline`, async () => {
    const {getLearningContent} = await import('../src/lib/learning-content');
    const {readFile} = await import('node:fs/promises');
    const raw = await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
    const rendered = await getLearningContent(id);
    assert.ok(rendered.outline[lang].length >= 14);
    assert.equal(rendered.outline.ko.length, rendered.outline.en.length);
    assert.equal(new Set(rendered.outline[lang].map(h => h.id)).size, rendered.outline[lang].length);
    assert.ok(rendered[lang].includes('katex-display'));
    assert.ok(!rendered[lang].includes('katex-error'));
    assert.equal((raw.match(/^\d\. \*\*/gm) || []).length, 6);
    assert.equal((raw.match(lang === 'en' ? /\*\*Answer:\*\*/g : /\*\*해설:\*\*/g) || []).length, 6);
    assert.ok(raw.length > 5500);
    assert.doesNotMatch(raw, /TODO|TBD|coming soon/i);
    if (lang === 'en') assert.doesNotMatch(raw, /[가-힣]/);
  });
}
