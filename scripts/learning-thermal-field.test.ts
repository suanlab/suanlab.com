import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { calculateThermalField, type ThermalFieldInput, type ThermalJetInput } from '../src/lib/learning-thermal-field';
import { calculateGaussianField } from '../src/lib/learning-gaussian-field';

const root = '.runtime/learning/thermal-field-foundation';
mkdirSync(root, { recursive: true });
const run = spawnSync('python3', ['scripts/learning/verify-thermal-field-foundation.py'], { encoding: 'utf8', timeout: 120000, maxBuffer: 8000000 });
assert.equal(run.status, 0, run.stderr);
writeFileSync(`${root}/independent-fraction-references.json`, run.stdout);
const references = JSON.parse(run.stdout) as { cases: { input: ThermalFieldInput; expected: Record<string, unknown> }[] };
const jet = (value: string, first = '0', second = '0'): ThermalJetInput => ({ value, first, second });
const base = (): ThermalFieldInput => ({ shape: [2], boundary: 'open', constraint: 'none', spacing: '1', temperature: '2', referenceScale: '1',
  mass: jet('1'), stiffness: jet('0'), backgroundDensity: jet('0'), sourceDensity: [jet('0'), jet('0')] });
const normal = (input: ThermalFieldInput) => {
  const result = calculateThermalField(input);
  assert.equal(result.kind, 'normalizable');
  if (result.kind !== 'normalizable') throw Error('Expected normalizable field');
  return result;
};
const rational = (x: string) => { const [n, d = '1'] = x.split('/'); return Number(n) / Number(d); };

test('independent Taylor-ring log-Z derivatives agree with exact Gaussian thermal moments on every output entry', () => {
  let normalizable = 0, rejected = 0, matrixCells = 0;
  for (const { input, expected } of references.cases) {
    const result = calculateThermalField(input);
    for (const [key, value] of Object.entries(expected)) assert.deepEqual(result[key as keyof typeof result], value, `${JSON.stringify(input)}: ${key}`);
    if (result.kind === 'normalizable') { normalizable++; matrixCells += 4 * result.sites ** 2; }
    else rejected++;
  }
  writeFileSync(`${root}/verification.json`, JSON.stringify({ status: 'independent-reference-comparison-passed', models: references.cases.length, normalizable, rejected,
    comparedMatrixEntries: matrixCells, arithmetic: 'exact rational', independentDerivativeMethod: 'Taylor-ring absolute log-Z differentiation',
    productionReaderVerified: false, topicComplete: false }, null, 2) + '\n');
});

test('temperature-dependent effective energy differs from internal energy and may have negative family heat capacity', () => {
  const fixed = normal(base());
  assert.equal(fixed.expectedEnergy, '2'); assert.equal(fixed.thermalEnergy, '2'); assert.equal(fixed.heatCapacity, '1');
  const linear = normal({ ...base(), mass: jet('2', '1') });
  assert.equal(linear.expectedEnergy, '2'); assert.equal(linear.thermalEnergy, '0'); assert.equal(linear.heatCapacity, '0');
  assert.deepEqual(linear.covariance, [['1', '0'], ['0', '1']]);
  const squared = normal({ ...base(), mass: jet('4', '4', '2') });
  assert.equal(squared.expectedEnergy, '2'); assert.equal(squared.thermalEnergy, '-2'); assert.equal(squared.heatCapacity, '-1');
  assert.equal(squared.varianceAdjustedEnergy, '4'); assert.equal(squared.expectedSecondThermalDerivative, '1');
});

test('reference measure changes absolute Z while a coordinated field rescaling preserves it', () => {
  for (const constraint of ['none', 'mean-zero'] as const) {
    const input = { ...base(), constraint, mass: jet('2', '1', '3'), sourceDensity: [jet('1', '2', '3'), jet('-2', '-1', '4')] };
    const first = normal(input), changedMeasure = normal({ ...input, referenceScale: '3' });
    assert.equal(changedMeasure.thermalEnergy, first.thermalEnergy); assert.equal(changedMeasure.heatCapacity, first.heatCapacity);
    assert.deepEqual(changedMeasure.mean, first.mean); assert.equal(changedMeasure.logPartition.referenceScale, '3');
    const rescaled = normal({ ...input, referenceScale: '1/2', mass: jet('8', '4', '12'), sourceDensity: [jet('2', '4', '6'), jet('-4', '-2', '8')] });
    assert.equal(rescaled.thermalEnergy, first.thermalEnergy); assert.equal(rescaled.heatCapacity, first.heatCapacity);
    assert.equal(rescaled.logPartition.constant, first.logPartition.constant);
    // det(K_z)=c^(2q)det(K), scale_z=scale/c: the two log terms cancel exactly.
    assert.equal(rational(rescaled.restrictedDeterminant) / rational(first.restrictedDeterminant), 4 ** first.dimension);
    first.mean.forEach((x, i) => assert.equal(rational(x), 2 * rational(rescaled.mean[i])));
    first.covariance.forEach((row, i) => row.forEach((x, j) => assert.equal(rational(x), 4 * rational(rescaled.covariance[i][j]))));
  }
});

test('a temperature-dependent constant background leaves the law fixed and changes thermodynamic derivatives', () => {
  const first = normal(base());
  // Per-site offset 3+5T+7T² at T=2 has jet (41,33,14); there are two sites.
  const shifted = normal({ ...base(), backgroundDensity: jet('41', '33', '14') });
  assert.deepEqual(shifted.mean, first.mean); assert.deepEqual(shifted.covariance, first.covariance);
  assert.equal(shifted.minimumEnergy, '82'); assert.equal(shifted.expectedEnergy, '84');
  assert.equal(shifted.thermalEnergy, '-48'); assert.equal(shifted.heatCapacity, '-55');
  assert.equal(shifted.entropyRationalAddend, '-24');
});

test('zero-mode removal changes the measure, and a zero pivot does not imply a singular full precision', () => {
  const massless = { ...base(), boundary: 'periodic' as const, mass: jet('0'), stiffness: jet('1') };
  assert.equal(calculateThermalField(massless).kind, 'not-normalizable');
  const constrained = normal({ ...massless, constraint: 'mean-zero' });
  assert.equal(constrained.dimension, 1); assert.equal(constrained.coordinateDeterminant, '8');
  assert.equal(constrained.restrictedDeterminant, '4'); assert.equal(constrained.gramDeterminant, '2');
  assert.deepEqual(constrained.covariance, [['1/4', '-1/4'], ['-1/4', '1/4']]);
  const indefinite = calculateThermalField({ ...base(), mass: jet('-1'), stiffness: jet('1') });
  assert.equal(indefinite.kind, 'not-normalizable');
  if (indefinite.kind === 'not-normalizable') { assert.equal(indefinite.failedPivot, '0'); assert.deepEqual(indefinite.precision, [['0', '-1'], ['-1', '0']]); }
  assert.equal(normal({ ...massless, constraint: 'mean-zero', mass: jet('-1') }).restrictedDeterminant, '3');
});

test('two-site periodic axes preserve both edges and two-dimensional fixed faces contribute separately', () => {
  const oneD = normal({ ...base(), boundary: 'periodic', stiffness: jet('1') });
  assert.deepEqual(oneD.laplacian, [['2', '-2'], ['-2', '2']]);
  const input = { ...base(), shape: [2, 2], boundary: 'dirichlet' as const, sourceDensity: Array.from({ length: 4 }, () => jet('0')) };
  const twoD = normal(input);
  assert.deepEqual(twoD.laplacian, [['4', '-1', '-1', '0'], ['-1', '4', '0', '-1'], ['-1', '0', '4', '-1'], ['0', '-1', '-1', '4']]);
  assert.equal(twoD.volume, '4'); assert.equal(twoD.heatCapacity, '2');
});

test('unit-spacing overlap agrees with the existing Gaussian field without counting it as a new topic', () => {
  for (const sites of [3, 4, 7]) for (const boundary of ['open', 'periodic', 'dirichlet'] as const) for (const constraint of ['none', 'mean-zero'] as const) {
    const source = Array.from({ length: sites }, (_, i) => String(i % 3 - 1));
    const old = calculateGaussianField({ sites, boundary, constraint, massSquared: '2', stiffness: '1', temperature: '2', source, probes: [0, 1, 1, 0] });
    const fresh = normal({ ...base(), shape: [sites], boundary, constraint, mass: jet('2'), stiffness: jet('1'), sourceDensity: source.map(x => jet(x)) });
    assert.deepEqual(fresh.mean, old.mean); assert.deepEqual(fresh.covariance, old.connectedTwo);
    assert.equal(fresh.restrictedDeterminant, old.normalization.restrictedPrecisionDeterminant);
    assert.equal(fresh.heatCapacity, String(fresh.dimension / 2).includes('.') ? `${fresh.dimension}/2` : String(fresh.dimension / 2));
  }
});

test('periodic determinants count every real Fourier degree once, including conjugate partners', () => {
  for (const n of [2, 3, 4, 7]) {
    const result = normal({ ...base(), shape: [n], boundary: 'periodic', mass: jet('2'), stiffness: jet('3'), sourceDensity: Array.from({ length: n }, () => jet('0')) });
    const eigenvalues = Array.from({ length: n }, (_, k) => 2 + 12 * Math.sin(Math.PI * k / n) ** 2);
    const determinant = eigenvalues.reduce((a, b) => a * b, 1);
    assert(Math.abs(determinant / rational(result.restrictedDeterminant) - 1) < 1e-12);
    const selfConjugate = n % 2 === 0 ? 2 : 1;
    assert.equal(selfConjugate + 2 * ((n - selfConjugate) / 2), n);
  }
});

test('invalid finite geometry, missing jets and oversized rational drafts are rejected before arithmetic', () => {
  for (const change of [{ shape: [] }, { shape: Array(1) }, { shape: [1] }, { shape: [2.5] }, { shape: [8, 8] }, { shape: [2, 2, 2] }, { temperature: '0' },
    { spacing: '-1' }, { referenceScale: '0' }, { mass: { value: '1' } }, { stiffness: jet('-1') }, { sourceDensity: [] }, { sourceDensity: Array(2) },
    { temperature: '1e-1000' }, { temperature: '1/0' }, { mass: jet('9'.repeat(19)) }, { constraint: 'fake' }, { boundary: 'fake' }]) {
    assert.throws(() => calculateThermalField({ ...base(), ...change }), RangeError);
  }
  assert.throws(() => calculateThermalField(null), RangeError);
});
