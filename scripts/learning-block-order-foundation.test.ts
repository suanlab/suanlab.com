import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { calculateBlockOrder, readBlockOrderInput, type BlockOrderInput } from '../src/lib/learning-block-order';

const root = '.runtime/learning/block-order-foundation';
mkdirSync(root, { recursive: true });
const run = spawnSync('python3', ['scripts/learning/verify-block-order-foundation.py'], { encoding: 'utf8', timeout: 120000, maxBuffer: 16000000 });
assert.equal(run.status, 0, run.stderr);
writeFileSync(`${root}/independent-references.json`, run.stdout);
const { cases } = JSON.parse(run.stdout) as { cases: { input: BlockOrderInput; expected: Record<string, unknown> }[] };
const base = (): BlockOrderInput => ({ shape: [4], blockShape: [2], boundary: 'open', disagreementActivity: '1/2', blockUpActivities: ['1', '1'], selectedUpCounts: [1, 1], witnessPair: [0, 1] });

test('Gray spin updates match independent row-transfer coefficients and every exact statistic on 24 finite spatial models', () => {
  let profiles = 0, histogramCells = 0;
  for (const { input, expected } of cases) {
    const result = calculateBlockOrder(input);
    for (const [key, value] of Object.entries(expected)) {
      if (key === 'selected') {
        for (const [field, v] of Object.entries(value as Record<string, unknown>))
          assert.deepEqual(result.selected[field as keyof typeof result.selected], v, `${JSON.stringify(input)}: selected.${field}`);
      } else assert.deepEqual(result[key as keyof typeof result], value, `${JSON.stringify(input)}: ${key}`);
    }
    profiles += result.profiles.length;
    histogramCells += result.profiles.reduce((s, p) => s + p.disagreementHistogram.length, 0);
  }
  writeFileSync(`${root}/verification.json`, JSON.stringify({ status: 'independent-foundation-verified', models: cases.length, profiles, histogramCells,
    methods: ['TypeScript Gray-code incident-edge updates', 'Python integer row-transfer polynomial contraction'],
    exactStatistics: true, productionWorkerVerified: false, readerVerified: false, unitRegistered: false, originalTopicComplete: false }, null, 2) + '\n');
});

test('fixed block magnetization does not make microscopic configurations uniform', () => {
  const r = calculateBlockOrder(base()), selected = r.profiles[r.selected.profileIndex];
  assert.equal(r.relativePartition, '27/4');
  assert.equal(selected.bondPolynomial, '3/4');
  assert.equal(selected.probability, '1/9');
  assert.equal(selected.conditionalMeanDisagreements, '7/3');
  assert.equal(selected.conditionalVarianceDisagreements, '2/9');
  assert.deepEqual(r.selected.examples.map(e => e.conditionalProbability), ['1/6','1/3','1/3','1/6']);
  assert.equal(r.globalUpProbabilities[2], '7/27');
  assert.equal(r.selected.meanProductResidualPower, '9/8');
  assert.equal(r.selected.meanProductResidualDivisor, '2');
});

test('marginalizing hidden boundary spins can destroy nearest-neighbor coarse pair factorization', () => {
  const inp: BlockOrderInput = { ...base(), shape: [6], blockUpActivities: ['1','1','1'], selectedUpCounts: [1,1,1], witnessPair: [0,2] };
  const r = calculateBlockOrder(inp);
  assert.equal(r.localityWitness?.crossRatio, '16/25');
  assert.equal(r.localityWitness?.pairIsNeighbor, false);
  assert.equal(r.localityWitness?.rulesOutNearestNeighborPairFactorization, true);
  const periodic = calculateBlockOrder({ ...inp, boundary: 'periodic' });
  assert.equal(periodic.localityWitness?.pairIsNeighbor, true);
  assert.equal(periodic.localityWitness?.rulesOutNearestNeighborPairFactorization, false);
  assert.equal(calculateBlockOrder({ ...inp, blockUpActivities: ['3','2/7','5'] }).localityWitness?.crossRatio, '16/25');
});

test('boundary conditional means differ from the block mean and the global symmetric mean', () => {
  const r = calculateBlockOrder({ ...base(), blockShape: [4], blockUpActivities: ['1'], selectedUpCounts: [3], witnessPair: [] });
  assert.deepEqual(r.selected.conditionalSpinMeans, ['1/3','2/3','2/3','1/3']);
  assert.deepEqual(r.profiles[r.selected.profileIndex].magnetizations, ['1/2']);
  assert.deepEqual(r.meanMagnetizations, ['0']);
  const p = calculateBlockOrder({ ...base(), shape: [6], blockUpActivities: ['1','1','1'], selectedUpCounts: [2,1,0], witnessPair: [0,2] });
  assert.deepEqual(p.selected.conditionalSpinMeans, ['1','1','3/5','-3/5','-1','-1']);
});

test('uncoupled, saturated and identity blocks are exact special cases; periodic wraps remain internal when appropriate', () => {
  for (const inp of [base(), { ...base(), selectedUpCounts: [0, 2] }, { ...base(), shape: [2,2], blockShape: [2,1], boundary: 'periodic' as const }]) {
    const r = calculateBlockOrder({ ...inp, disagreementActivity: '1' });
    assert.equal(r.selected.meanProductResidualPower, '1');
  }
  const saturated = calculateBlockOrder({ ...base(), selectedUpCounts: [0,2] });
  assert.equal(saturated.selected.meanProductResidualPower, '1');
  assert.deepEqual(saturated.selected.examples.map(e => e.conditionalProbability), ['1']);
  const whole = calculateBlockOrder({ ...base(), shape: [2], blockShape: [2], boundary: 'periodic', blockUpActivities: ['1'], selectedUpCounts: [1], witnessPair: [] });
  assert.equal(whole.edgeCount, 2);
  assert.deepEqual(whole.selected.isolatedBlockPolynomials, ['1/2']);
  assert.equal(whole.selected.meanProductResidualPower, '1');
});

test('invalid geometry, zero activities, fractions outside bounds and ambiguous witness pairs are rejected', () => {
  for (const change of [ { shape: [17] }, { shape: [3,6] }, { shape: [1] }, { blockShape: [3] }, { blockShape: [0] },
    { disagreementActivity: '0' }, { disagreementActivity: '-1/2' }, { disagreementActivity: '1e-3' },
    { disagreementActivity: '1/0' }, { disagreementActivity: '1000000' }, { blockUpActivities: ['1'] },
    { selectedUpCounts: [3,0] }, { selectedUpCounts: [1.5,0] }, { witnessPair: [1,0] }, { witnessPair: [0,0] }, { extra: true } ]) {
    const inp = { ...base(), ...change };
    assert.equal(readBlockOrderInput(inp), null, JSON.stringify(change));
    assert.throws(() => calculateBlockOrder(inp), RangeError);
  }
  assert.equal(readBlockOrderInput(null), null);
  assert.equal(readBlockOrderInput({ ...base(), get shape() { throw Error('not JSON'); } }), null);
  assert.deepEqual(readBlockOrderInput(JSON.parse(JSON.stringify(base()))), base());
});
