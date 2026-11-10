import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { observeQubitReal, observeQubitComplex, type QubitDisplayReal } from '../src/lib/learning-qubit-display';
import { validateQubitDisplayReal } from '../src/lib/learning-qubit-display-validator';
import { observeQubitDensity } from '../src/lib/learning-qubit-density-display';
import { validateQubitDensityObservation } from '../src/lib/learning-qubit-density-display-validator';
import { decodeQubitFraction } from '../src/lib/learning-qubit-decode';
import { f, quadratic, complex, real, Z, O, type Q, type M } from '../src/lib/learning-qubit-channel-algebra';
import { qubitWorkerCases } from './learning/qubit-worker-cases';

const fixtures = JSON.parse(readFileSync('scripts/fixtures/qubit-display-independent.json', 'utf8')) as {
  cases: { name: string; coefficients: [string, string]; limit: number; expected: QubitDisplayReal }[];
};
for (const c of fixtures.cases) test(`independent interval: ${c.name}`, () => {
  const value = c.coefficients.map(decodeQubitFraction) as unknown as Q;
  const observed = observeQubitReal(value, c.limit);
  assert.deepEqual(observed, c.expected);
  assert(validateQubitDisplayReal(value, observed, c.limit));
  for (const tamper of [
    { ...observed, sign: observed.sign === 0 ? 1 : 0 },
    { ...observed, exact: !observed.exact },
    { ...observed, coordinate: NaN },
    { ...observed, status: observed.status === 'outside' ? 'enclosed' : 'outside' },
  ]) assert.equal(validateQubitDisplayReal(value, tamper as QubitDisplayReal, c.limit), false);
});

for (const kind of ['circuit', 'channel', 'environment', 'instrument', 'deferred'])
  test(`certified plotting for every independent ${kind} response scalar`, t => {
    let count = 0;
    const visit = (value: unknown) => {
      if (!value || typeof value !== 'object') return;
      const record = value as Record<string, unknown>;
      if (Array.isArray(record.real) && Array.isArray(record.imaginary)) {
        for (const coefficients of [record.real, record.imaginary]) {
          const q = coefficients.map(decodeQubitFraction) as unknown as Q;
          const observation = observeQubitReal(q);
          assert(validateQubitDisplayReal(q, observation));
          if (observation.sign === 0) assert.equal(observation.coordinate, 0);
          if (observation.status === 'enclosed') assert(observation.upper! - observation.lower! <= 2 ** -40);
          count++;
        }
      } else for (const child of Object.values(record)) visit(child);
    };
    for (const c of qubitWorkerCases.filter(c => c.input.kind === kind)) visit(c.expected);
    assert(count > 0);
    t.diagnostic(JSON.stringify({ kind, realComponents: count }));
  });

test('one-sided nonzero intervals never imply exact zero, including subnormal inputs', () => {
  for (const sign of [-1, 1]) {
    const observed = observeQubitReal(quadratic(f(BigInt(sign), BigInt(10) ** BigInt(500))));
    assert.equal(observed.sign, sign);
    assert.equal(observed.exact, false);
    assert.equal(Math.sign(observed.coordinate!), sign);
    assert(observed.lower === 0 || observed.upper === 0);
  }
});

test('independent validator rejects displaced, loose, negative-zero and forged outside certificates', () => {
  const value = quadratic(f(BigInt(1), BigInt(3))), good = observeQubitReal(value);
  for (const bad of [
    { ...good, lower: good.lower! + 2 ** -40, upper: good.upper! + 2 ** -40 },
    { ...good, lower: good.lower! - 2 ** -40 },
    { ...good, coordinate: good.upper },
    { ...good, lower: -0 },
    { ...good, status: 'outside', lower: null, upper: null, coordinate: null, exact: false },
  ]) assert.equal(validateQubitDisplayReal(value, bad as QubitDisplayReal), false);
  const zero = quadratic(f(BigInt(0)));
  assert.equal(validateQubitDisplayReal(zero, { ...observeQubitReal(zero), upper: 2 ** -40, coordinate: 2 ** -41, exact: false }), false);
});

test('range limits and malformed arithmetic inputs cannot create plot coordinates', () => {
  const value = quadratic(f(BigInt(1)));
  for (const limit of [0, 3, 1025, Infinity, NaN, 1.5]) {
    assert.throws(() => observeQubitReal(value, limit), RangeError);
    assert.equal(validateQubitDisplayReal(value, observeQubitReal(value), limit), false);
  }
  for (const malformed of [[], [value[0]], [{ n: BigInt(1), d: BigInt(0) }, value[1]], [{ n: 1, d: 1 }, value[1]]]) {
    assert.throws(() => observeQubitReal(malformed as unknown as Q), RangeError);
    assert.equal(validateQubitDisplayReal(malformed as unknown as Q, observeQubitReal(value)), false);
  }
});

test('positive Y state has Y=+1, and the mixed state has zero Bloch vector and purity 1/2', () => {
  const positiveY: M = [[real(1, 2), complex(quadratic(f(BigInt(0))), quadratic(f(BigInt(-1), BigInt(2))))],
    [complex(quadratic(f(BigInt(0))), quadratic(f(BigInt(1), BigInt(2)))), real(1, 2)]];
  const observed = observeQubitDensity(positiveY);
  assert(observed.isDensity && observed.bloch && observed.purity);
  assert.deepEqual(observed.bloch.map(z => z.display.real.coordinate), [0, 1, 0]);
  assert.equal(observed.purity.display.real.coordinate, 1);
  const mixed = observeQubitDensity([[real(1, 2), Z], [Z, real(1, 2)]]);
  assert(mixed.isDensity && mixed.bloch && mixed.purity);
  assert.deepEqual(mixed.bloch.map(z => z.display.real.coordinate), [0, 0, 0]);
  assert.equal(mixed.purity.display.real.coordinate, 0.5);
  assert.deepEqual(observeQubitComplex(Z).real, { sign: 0, status: 'enclosed', lower: 0, upper: 0, coordinate: 0, exact: true });
});

test('joint density has purity but no single Bloch vector; nonphysical images have neither', () => {
  const half = real(1, 2);
  const bell = observeQubitDensity([[half, Z, Z, half], [Z, Z, Z, Z], [Z, Z, Z, Z], [half, Z, Z, half]]);
  assert(bell.isDensity && bell.purity);
  assert.equal(bell.purity.display.real.coordinate, 1);
  assert.equal(bell.bloch, null);
  for (const invalid of [
    [[O, Z], [Z, O]], // Trace-growing image: no hidden renormalization.
    [[real(-1), Z], [Z, real(2)]],
    [[real(1, 2), O], [Z, real(1, 2)]],
    [[Z, Z], [Z, Z]], // Impossible branch is not the maximally mixed state.
  ]) {
    const observed = observeQubitDensity(invalid);
    assert.equal(observed.isDensity, false);
    assert.equal(observed.purity, null); assert.equal(observed.bloch, null);
  }
  assert.throws(() => observeQubitDensity([[O]]), RangeError);
});

test('all independent two/four-dimensional report matrices agree with separate density and display checks', t => {
  let checked = 0, physical = 0;
  const scalar = (z: { real: string[]; imaginary: string[] }) => complex(
    z.real.map(decodeQubitFraction) as unknown as Q, z.imaginary.map(decodeQubitFraction) as unknown as Q);
  const visit = (value: unknown) => {
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value) && [2, 4].includes(value.length)
      && value.every(row => Array.isArray(row) && row.length === value.length
        && row.every(z => z && Array.isArray(z.real) && Array.isArray(z.imaginary)))) {
      const matrix = value.map(row => row.map(scalar));
      const observed = observeQubitDensity(matrix);
      assert(validateQubitDensityObservation(matrix, observed)); checked++;
      if (observed.isDensity) {
        physical++;
        const changed = structuredClone(observed);
        changed.purity!.display.real.coordinate = -3;
        assert.equal(validateQubitDensityObservation(matrix, changed), false);
      } else {
        assert.equal(validateQubitDensityObservation(matrix, { ...observed, isDensity: true } as unknown as ReturnType<typeof observeQubitDensity>), false);
      }
    } else for (const child of Object.values(value)) visit(child);
  };
  for (const c of qubitWorkerCases) visit(c.expected);
  assert(checked > 1000 && physical > 100);
  t.diagnostic(JSON.stringify({ matrices: checked, physical, nonphysical: checked - physical }));
});
