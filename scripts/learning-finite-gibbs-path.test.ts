import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finiteGibbsModel as model, type FiniteGibbsParameters, type GibbsKernelName } from '../src/lib/learning-finite-gibbs';
import { finiteGibbsPath as path, type FiniteGibbsPathOptions } from '../src/lib/learning-finite-gibbs-path';
import { evaluateBoltzmannGibbsRequest as evaluate } from '../src/lib/learning-boltzmann-gibbs-evaluator';
import { isBoltzmannGibbsSuccessForInput as valid, type BoltzmannGibbsInput, type BoltzmannGibbsSuccess } from '../src/lib/learning-boltzmann-gibbs-protocol';
interface Case { id: string; parameters: FiniteGibbsParameters; options: FiniteGibbsPathOptions; reference: Record<string, unknown> }
const fixture = JSON.parse(readFileSync('scripts/fixtures/finite-gibbs-path-oracle.json', 'utf8')) as { cases: Case[] };
const cache = new Map<string, ReturnType<typeof model>>();
function cached(p: FiniteGibbsParameters) { const key = JSON.stringify(p); let r = cache.get(key); if (!r) { r = model(p); cache.set(key, r); } return r; }
for (const c of fixture.cases) test(`${c.id}: exact finite-bit kernel and seeded trajectory match independent Python reference`, () => {
  const r = path(cached(c.parameters), c.options); assert.equal(r.status, 'ready');
  for (const [key, value] of Object.entries(c.reference)) assert.deepEqual((r as unknown as Record<string, unknown>)[key], value, `${c.id}.${key}`);
  assert.ok(r.status === 'ready'); assert.match(r.contract, /not an iid source/); assert.match(r.contract, /scientific target\/kernel are unchanged/);
});
const first = fixture.cases[0], basic = cached(first.parameters);
test('deterministic replays do not mutate the scientific target and different seeds change the path', () => {
  const before = JSON.stringify(basic), opts: FiniteGibbsPathOptions = { seed: 1, bits: 16, steps: 128, mode: 'random' };
  assert.deepEqual(path(basic, opts), path(basic, opts)); const a = path(basic, opts), b = path(basic, { ...opts, seed: 2 });
  assert.ok(a.status === 'ready' && b.status === 'ready'); assert.notDeepEqual(a.states, b.states); assert.equal(JSON.stringify(basic), before);
});
test('left-grid quantization can remove a rare scientific edge without changing its theoretical positivity', () => {
  const c = fixture.cases.find(c => c.id.startsWith('rare-conditional'))!, exact = cached(c.parameters), sampled = path(exact, c.options); assert.ok(sampled.status === 'ready');
  assert.equal(exact.kernels.first[0][2].value, null); assert.equal(exact.kernels.first[0][2].sign, 1);
  assert.equal(exact.diagnostics.random.irreducible, true); assert.equal(sampled.quantizedIrreducible, false);
  assert.equal(sampled.quantizedKernel.numerators[0][2], '0'); assert.deepEqual(sampled.observedCounts, [65, 0, 0, 0]);
});
test('zero-sum coordinate paths stay trapped while a joint block can visit both states', () => {
  const c = fixture.cases.find(c => c.id.startsWith('fixed-zero-sum-uniform-random'))!, m = cached(c.parameters);
  const trapped = path(m, c.options), blocked = path(m, { ...c.options, mode: 'blocked' }); assert.ok(trapped.status === 'ready' && blocked.status === 'ready');
  assert.ok(trapped.states.every(i => i === 0)); assert.equal(trapped.quantizedIrreducible, false);
  assert.ok(blocked.observedCounts.every(n => n > 0)); assert.equal(blocked.quantizedIrreducible, true);
});
test('random selection and systematic sweeps consume two draws but count different coordinate updates', () => {
  const opts = { seed: 0, bits: 8, steps: 32 }, random = path(basic, { ...opts, mode: 'random' }), systematic = path(basic, { ...opts, mode: 'systematic' });
  assert.ok(random.status === 'ready' && systematic.status === 'ready');
  assert.equal(random.randomDrawCount, 64); assert.equal(systematic.randomDrawCount, 64); assert.equal(random.coordinatesResampled, 32); assert.equal(systematic.coordinatesResampled, 64);
});
test('uncertain grid boundaries withhold the path instead of guessing a threshold from rounded probabilities', () => {
  const weak = model({ ...first.parameters, couplingEnergy: 1e-200 }, { maximumPrecision: 128 });
  const r = path(weak, { seed: 1, bits: 8, steps: 16, mode: 'random' });
  assert.deepEqual(r, { status: 'unavailable', reason: 'uncertain-grid-boundary', options: { seed: 1, bits: 8, steps: 16, mode: 'random' } });
});
test('maximum supported trace length and precision remain normalized and transport-safe', () => {
  const input: BoltzmannGibbsInput = { kind: 'finite-gibbs-conditionals-scan-support', parameters: first.parameters,
    path: { seed: 0xffffffff, bits: 32, steps: 2048, mode: 'systematic' } };
  const r = evaluate({ version: 1, requestId: 7, input }); assert.ok(r.ok && r.kind === input.kind && r.path?.status === 'ready');
  assert.ok(valid(r, input, 7)); assert.equal(r.path.states.length, 2049); assert.equal(r.path.quantizedKernel.denominatorPower, 64);
  const M = BigInt(1) << BigInt(64); r.path.quantizedKernel.numerators.forEach(row => assert.equal(row.reduce((s, v) => s + BigInt(v), BigInt(0)), M));
});
test('every seeded update mode round-trips its checked envelope and rejects corrupted trace metadata', () => {
  for (const mode of ['first', 'second', 'random', 'systematic', 'blocked'] as GibbsKernelName[]) {
    const input: BoltzmannGibbsInput = { kind: 'finite-gibbs-conditionals-scan-support', parameters: first.parameters, path: { mode, seed: 0, bits: 8, steps: 32 } };
    const result = evaluate({ version: 1, requestId: 1, input }); assert.ok(result.ok && result.kind === input.kind && result.path?.status === 'ready'); assert.ok(valid(result, input, 1));
    for (const corruption of ['seed', 'counts', 'edge', 'draws', 'row', 'graph']) {
      const r: Extract<BoltzmannGibbsSuccess, { kind: 'finite-gibbs-conditionals-scan-support' }> = structuredClone(result); assert.ok(r.path?.status === 'ready');
      if (corruption === 'seed') r.path.options.seed++;
      if (corruption === 'counts') r.path.observedCounts[0]++;
      if (corruption === 'edge') r.path.states[0] = 3;
      if (corruption === 'draws') r.path.randomDrawCount++;
      if (corruption === 'row') r.path.quantizedKernel.numerators[0][0] = '0';
      if (corruption === 'graph') r.path.quantizedIrreducible = !r.path.quantizedIrreducible;
      assert.equal(valid(r, input, 1), false, `${mode}.${corruption}`);
    }
  }
});
test('invalid seeds, bit counts, trace lengths and update modes fail explicitly', () => {
  const validOptions: FiniteGibbsPathOptions = { seed: 1, bits: 8, steps: 16, mode: 'random' };
  for (const opts of [{ ...validOptions, seed: -1 }, { ...validOptions, seed: 2 ** 32 }, { ...validOptions, bits: 0 }, { ...validOptions, bits: 33 },
    { ...validOptions, steps: 0 }, { ...validOptions, steps: 2049 }, { ...validOptions, mode: 'unknown' }]) assert.throws(() => path(basic, opts as FiniteGibbsPathOptions), RangeError);
});
