import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finiteStateResponseExperiment as run, isFiniteStateResponseExperimentParameters as valid, type FiniteStateResponseExperimentParameters as Parameters } from '../src/lib/learning-finite-state-response-experiment';
import type { PartitionObservation as Observation, PartitionRational } from '../src/lib/learning-partition-exact-core';

interface F { n: bigint; d: bigint }
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const sub = (a: F, b: F) => frac(a.n * b.d - b.n * a.d, a.d * b.d);
const add = (a: F, b: F) => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const abs = (a: F) => frac(a.n < BigInt(0) ? -a.n : a.n, a.d);
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
function number(value: string): F {
  const match = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value);
  assert(match, value);
  const exponent = Number(match[4] || 0) - (match[3] || '').length;
  const n = BigInt((match[1] || '') + match[2] + (match[3] || ''));
  return exponent >= 0 ? frac(n * BigInt(10) ** BigInt(exponent)) : frac(n, BigInt(10) ** BigInt(-exponent));
}
function endpoint(value: PartitionRational): F {
  const exponent = value.numerator.exponent - value.denominator.exponent;
  const n = BigInt(value.numerator.coefficient), d = BigInt(value.denominator.coefficient);
  return exponent >= 0 ? frac(n << BigInt(exponent), d) : frac(n, d << BigInt(-exponent));
}
function check(value: Observation, reference: F, label: string) {
  assert(value.bounds, label);
  const lower = endpoint(value.bounds.lower), upper = endpoint(value.bounds.upper);
  assert(le(lower, upper), label);
  const magnitude = le(abs(reference), frac(BigInt(1))) ? frac(BigInt(1)) : abs(reference);
  // Numerical reference uncertainty, far below the256bit model quantum.
  const uncertainty = frac(magnitude.n, magnitude.d * BigInt(10) ** BigInt(140));
  assert(le(lower, add(reference, uncertainty)) && le(sub(reference, uncertainty), upper), label);
  const width = sub(upper, lower);
  assert(le(width, frac(magnitude.n, magnitude.d * (BigInt(1) << BigInt(180)))), `${label}: enclosure too wide for these reference cases`);
  if (value.resolution === 'unresolved-at-precision-ceiling') assert.equal(value.value, null);
  if (value.resolution === 'structural-zero') assert(lower.n === BigInt(0) && upper.n === BigInt(0));
}
type Reference = { label: string; input: Parameters; time: string; kernel: string[]; step: string[]; actual: string[] | null; predicted: string[] };
const fixture = JSON.parse(readFileSync('scripts/fixtures/linear-response-temporal-preparation.json', 'utf8')) as { cases: Reference[] };
const groups = new Map<string, Reference[]>();
for (const row of fixture.cases) groups.set(row.label, [...(groups.get(row.label) || []), row]);

for (const [label, references] of groups) test(`independent time/finite-input reference: ${label}`, () => {
  const input = references[0].input, result = run(input);
  assert.deepEqual(result.parameters, input); assert(Object.isFrozen(result.parameters));
  assert.equal(result.history.length, input.sampleSteps + 1);
  assert.equal(result.classification.normalizationOfTruncation, false);
  assert.equal(result.classification.sampledPaths, false); assert.equal(result.classification.eulerTimeStepping, false);
  for (const reference of references) {
    const parts = reference.time.split('/'), t = frac(BigInt(parts[0]), BigInt(parts[1] || 1));
    const row = result.history.find(r => {
      const value = endpoint(r.time.bounds!.lower);
      return value.n * t.d === t.n * value.d;
    });
    assert(row, reference.time);
    check(row.baselineProof.basePoissonMean, endpoint(row.baselineProof.basePoissonMean.bounds!.lower), 'exact base mean');
    assert(le(endpoint(row.baselineProof.basePoissonMean.bounds!.upper), frac(BigInt(1), BigInt(2))));
    for (let state = 0; state < 3; state++) {
      check(row.kernel[state], number(reference.kernel[state]), `${reference.time}/kernel${state}`);
      check(row.stepSusceptibility[state], number(reference.step[state]), `${reference.time}/step${state}`);
      check(row.predictedOccupancy[state], number(reference.predicted[state]), `${reference.time}/predicted${state}`);
      if (reference.actual) {
        assert(row.actualOccupancy && row.actualMinusLinearPrediction);
        check(row.actualOccupancy[state], number(reference.actual[state]), `${reference.time}/actual${state}`);
        check(row.actualMinusLinearPrediction[state], sub(number(reference.actual[state]), number(reference.predicted[state])), `${reference.time}/difference${state}`);
        assert(le(frac(BigInt(0)), endpoint(row.actualOccupancy[state].bounds!.lower)));
        assert(le(endpoint(row.actualOccupancy[state].bounds!.upper), frac(BigInt(1))));
      } else assert.equal(row.actualOccupancy, null);
    }
  }
});

const baseline = fixture.cases[0].input;
test('time-zero, zero input and exact-zero linear source preserve different identities', () => {
  const initial = run({ ...baseline, kineticNumerator: 1000000, amplitudeNumerator: 1, amplitudeDenominator: 1, timeNumerator: 0 });
  assert.equal(initial.classification.finiteInputRatesEnclosed, false);
  for (const row of initial.history) {
    assert.deepEqual(row.actualOccupancy, initial.reference.stationaryLaw);
    assert(row.stepSusceptibility.every(x => x.resolution === 'structural-zero'));
    assert(row.actualMinusLinearPrediction?.every(x => x.resolution === 'structural-zero'));
  }
  const off = run({ ...baseline, amplitudeNumerator: 0 });
  assert(off.history.every(row => row.finiteInputStatus === 'exact-unchanged-law'));
  assert(off.history.every(row => JSON.stringify(row.actualOccupancy) === JSON.stringify(off.reference.stationaryLaw)));
  const nonlinear = run({ ...baseline, a: 3, b: 3, c: 3, d: 1, e: 1, f: 1, kineticNumerator: -1, amplitudeNumerator: 1, amplitudeDenominator: 2 });
  assert.equal(nonlinear.classification.sourceExactlyZero, true);
  assert(nonlinear.history.every(row => [...row.kernel, ...row.stepSusceptibility].every(x => x.resolution === 'structural-zero')));
  assert.deepEqual(nonlinear.current.predictedOccupancy, nonlinear.reference.stationaryLaw);
  assert(nonlinear.current.actualMinusLinearPrediction?.some(x => x.value !== null && Math.abs(x.value) > 1e-4));
  const equilibrium = run({ ...baseline, a: 1, b: 1, c: 1, d: 1, e: 1, f: 1 });
  assert.deepEqual(equilibrium.classification.identicallyZeroResponseStates, [false, false, true]);
  assert(equilibrium.history.every(row => row.kernel[2].resolution === 'structural-zero' && row.stepSusceptibility[2].resolution === 'structural-zero'));
});

test('unavailable rates and unresolved positive late kernels are not structural zeros', () => {
  const huge = run({ ...baseline, a: 2, b: 3, c: 4, d: 1, e: 1, f: 1, kineticNumerator: 1000000, amplitudeNumerator: 1, amplitudeDenominator: 1 });
  assert.equal(huge.current.finiteInputStatus, 'unresolved-positive-rate-reciprocal'); assert.equal(huge.current.actualOccupancy, null);
  assert(huge.current.predictedOccupancy.some(x => x.value !== null && (x.value > 1 || x.value < 0)), 'signed tangents must not be clipped');
  const stiff = run({ ...baseline, a: 1000000, b: 1000000, c: 1000000, d: 1000000, e: 1000000, f: 1000000, kineticNumerator: 0 });
  assert.equal(stiff.current.kernel[1].resolution, 'unresolved-at-precision-ceiling');
  assert.equal(stiff.current.kernel[1].value, null);
  assert(stiff.current.baselineProof.squarings > 20);
});

test('record density and amplitude do not alter the background or exact frequency response', () => {
  const sparse = run({ ...baseline, sampleSteps: 1 }), dense = run({ ...baseline, sampleSteps: 64 });
  assert.deepEqual(sparse.current, { ...dense.current, step: 1 });
  const flipped = run({ ...baseline, amplitudeNumerator: -baseline.amplitudeNumerator });
  assert.deepEqual(flipped.reference, sparse.reference);
  assert.deepEqual(flipped.current.kernel, sparse.current.kernel);
  assert.deepEqual(flipped.current.stepSusceptibility, sparse.current.stepSusceptibility);
});

test('strict sixteen-field physical preparation and caller ownership', () => {
  assert(valid(baseline));
  for (const p of [null, { ...baseline, extra: 1 }, { ...baseline, a: 0 }, { ...baseline, sampleSteps: 0 }, { ...baseline, sampleSteps: 65 }, { ...baseline, timeNumerator: 5, timeDenominator: 1 }, { ...baseline, amplitudeNumerator: 2, amplitudeDenominator: 1 }, { ...baseline, timeDenominator: 0 }, { ...baseline, amplitudeNumerator: NaN }]) {
    assert.equal(valid(p), false); assert.throws(() => run(p as Parameters), RangeError);
  }
  const p = { ...baseline }, before = JSON.stringify(p), r = run(p);
  assert.equal(JSON.stringify(p), before); p.a++;
  assert.notEqual(r.parameters.a, p.a);
});
