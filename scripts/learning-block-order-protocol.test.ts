import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { blockOrderDefault, blockOrderExamples, readBlockOrderInput, type BlockOrderInput } from '../src/lib/learning-block-order-input';
import { evaluateBlockOrderRequest as calculate } from '../src/lib/learning-block-order-evaluator';
import { evaluateBlockOrderValidationRequest as validate } from '../src/lib/learning-block-order-validation-evaluator';
import { blockOrderSuccessHeader, blockOrderInputKey, type BlockOrderSuccess } from '../src/lib/learning-block-order-protocol';
import { snapshotQubitWire, QubitWireLimit } from '../src/lib/learning-qubit-wire';

const root = '.runtime/learning/block-order-protocol';
mkdirSync(root,{ recursive: true });
const fresh = spawnSync('python3',['scripts/learning/verify-block-order-foundation.py'],{ encoding: 'utf8', timeout: 120000, maxBuffer: 16000000 });
assert.equal(fresh.status,0,fresh.stderr);
const references = JSON.parse(fresh.stdout) as { cases: { input: BlockOrderInput; expected: Record<string,unknown> }[] };
const success = (input = blockOrderDefault): BlockOrderSuccess => {
  const response = calculate({ version: 1, requestId: 1, input });
  assert(response.ok,JSON.stringify(response));
  return response;
};

test('every independently checked spatial model and named example passes the full exact and logarithm gate', () => {
  let models = 0;
  for (const { input, expected } of references.cases) {
    const response = success(input), { profileChunks, ...rest } = response.result.exact, raw = { ...rest, profiles: profileChunks.flat() };
    for (const [key,value] of Object.entries(expected)) {
      if (key === 'selected') for (const [field,v] of Object.entries(value as Record<string,unknown>))
        assert.deepEqual(raw.selected[field as keyof typeof raw.selected],v,`${JSON.stringify(input)}:${field}`);
      else assert.deepEqual(raw[key as keyof typeof raw],value,`${JSON.stringify(input)}:${key}`);
    }
    const receipt = validate({ version: 1, requestId: 1, input, response });
    assert.equal(receipt.phase,'validated',JSON.stringify(input));
    models++;
  }
  for (const { input } of blockOrderExamples) {
    assert.equal(validate({ version: 1, requestId: 1, input, response: success(input) }).phase,'validated');
    models++;
  }
  writeFileSync(`${root}/models.json`,JSON.stringify({ status: 'complete-response-gate-passed', models, independentReferenceModels: references.cases.length,
    namedExamples: blockOrderExamples.length, precisionBits: 512, checkerWorkBits: 2048, readerVerified: false },null,2)+'\n');
});

test('canonical inputs are immutable and equivalently written activities share an attribution key', () => {
  const a = { ...structuredClone(blockOrderDefault), disagreementActivity: '2/4' }, input = readBlockOrderInput(a)!;
  assert.equal(blockOrderInputKey(input),blockOrderInputKey(blockOrderDefault));
  a.shape[0] = 8; a.blockUpActivities[0] = '3';
  assert.deepEqual(input,blockOrderDefault);
  assert(Object.isFrozen(input) && Object.isFrozen(input.shape) && Object.isFrozen(input.blockUpActivities));
  let calls = 0;
  const getter = () => { calls++; return 4; };
  const values: unknown[] = [];
  values.push({ ...blockOrderDefault, get shape() { calls++; return [4]; } });
  const arr = [4]; Object.defineProperty(arr,'0',{ get: getter });
  values.push({ ...blockOrderDefault, shape: arr });
  const extra = [4]; Object.assign(extra,{ toJSON() { calls++; return [4]; } });
  values.push({ ...blockOrderDefault, shape: extra });
  const sparse = new Array(2); sparse[0] = '1';
  values.push({ ...blockOrderDefault, blockUpActivities: sparse });
  values.push(Object.assign(Object.create({}),blockOrderDefault));
  values.push({ ...blockOrderDefault, [Symbol.iterator]: getter });
  for (const value of values) assert.equal(readBlockOrderInput(value),null);
  assert.equal(calls,0);
});

test('request attribution and valid JSON transport never substitute for whole-response mathematical verification', () => {
  const response = success();
  assert(blockOrderSuccessHeader(response,blockOrderDefault,1));
  const bad = structuredClone(response); bad.result.exact.profileChunks[0][0].probability = '1/2';
  assert(blockOrderSuccessHeader(bad,blockOrderDefault,1));
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: bad }).phase,'rejected');
  assert.equal(validate({ version: 1, requestId: 2, input: blockOrderDefault, response }).phase,'rejected');
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderExamples[1].input, response }).phase,'rejected');
  for (const patch of [{ requestId: 0 },{ version: 2 },{ extra: 1 },{ input: null }]) assert.equal(calculate({ version: 1, requestId: 1, input: blockOrderDefault, ...patch }).ok,false);
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: JSON.parse(JSON.stringify(response)) }).phase,'validated');
});

test('every scalar leaf in the exact result is independently checked, including geometry, all profiles and interpretive labels', () => {
  const response = success(), leaves: (string | number)[][] = [];
  const collect = (value: unknown, path: (string | number)[]) => {
    if (value && typeof value === 'object') for (const key of Object.keys(value)) collect((value as Record<string,unknown>)[key],[...path,Array.isArray(value) ? Number(key) : key]);
    else leaves.push(path);
  };
  collect(response.result.exact,[]);
  for (const path of leaves) {
    const copy = structuredClone(response);
    let at = copy.result.exact as unknown as Record<string | number,unknown>;
    for (const key of path.slice(0,-1)) at = at[key] as Record<string | number,unknown>;
    const key = path.at(-1)!, value = at[key];
    at[key] = typeof value === 'number' ? value+1 : typeof value === 'boolean' ? !value : typeof value === 'string' ? `${value}0` : true;
    assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: copy }).phase,'rejected',path.join('.'));
  }
  writeFileSync(`${root}/exact-mutations.json`,JSON.stringify({ rejectedScalarMutations: leaves.length, includesLabels: true, fullMaximumProfileCaseCoveredSeparately: true },null,2)+'\n');
});

test('directed observation bounds must be canonical, narrow dyadic enclosures of independent truths', () => {
  const response = success(), keys = Object.keys(response.result.observations).filter(k => !['precisionBits','interpretation'].includes(k));
  let rejected = 0;
  for (const key of keys) {
    for (const replacement of [null,{ lower: '0', upper: '0' },{ lower: '-1000000', upper: '1000000' },{ lower: '0/2', upper: '1' },{ lower: '1/3', upper: '2/3' },{ lower: '1', upper: '0' }]) {
      const copy = structuredClone(response);
      (copy.result.observations as unknown as Record<string,unknown>)[key] = replacement;
      assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: copy }).phase,'rejected',key);
      rejected++;
    }
  }
  const noPair = success(blockOrderExamples[2].input);
  assert.equal(noPair.result.observations.localityLogCrossRatio,null);
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderExamples[2].input, response: noPair }).phase,'validated');
  writeFileSync(`${root}/observation-mutations.json`,JSON.stringify({ rejected, nullOnlyWhenNoPair: true },null,2)+'\n');
});

test('maximum 625 profiles, six-digit rational activities and selected-state outputs fit bounded transport without truncation', () => {
  const input: BlockOrderInput = { shape: [4,4], blockShape: [2,2], boundary: 'periodic', disagreementActivity: '999983/999979',
    blockUpActivities: ['999961/999959','999953/999931','999917/999907','999883/999863'], selectedUpCounts: [1,2,3,1], witnessPair: [0,3] };
  const response = success(input), chunks = response.result.exact.profileChunks;
  assert.deepEqual(chunks.map(c => c.length),[128,128,128,128,113]);
  assert.equal(chunks.flat().length,625);
  assert.equal(validate({ version: 1, requestId: 1, input, response }).phase,'validated');
  const bad = structuredClone(response); bad.result.exact.profileChunks.at(-1)!.at(-1)!.probability = '0';
  assert.equal(validate({ version: 1, requestId: 1, input, response: bad }).phase,'rejected');
  assert.throws(() => snapshotQubitWire({ ...response.result.exact, profiles: chunks.flat() }),QubitWireLimit);
  writeFileSync(`${root}/maximum.json`,JSON.stringify({ sites: 16, profiles: 625, chunks: chunks.map(c => c.length),
    responseJsonCharacters: JSON.stringify(response).length, allProfilesPreserved: true, lastProfileMutationRejected: true },null,2)+'\n');
});

test('cyclic, accessor, oversized and incorrectly shaped payloads fail with explicit resource classification', () => {
  let calls = 0;
  const response = success();
  const accessor = { ...response, get result() { calls++; return response.result; } };
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: accessor }).phase,'rejected');
  assert.equal(calls,0);
  const cycle: Record<string,unknown> = {}; cycle.self = cycle;
  const rejected = validate({ version: 1, requestId: 1, input: blockOrderDefault, response: cycle });
  assert.equal(rejected.phase,'rejected');
  const resource = validate({ version: 1, requestId: 1, input: blockOrderDefault, response: { ...response, result: 'x'.repeat(65537) } });
  assert(resource.phase === 'rejected' && resource.error === 'resource-limit');
  const extra = structuredClone(response); Object.assign(extra.result.exact,{ extra: true });
  assert.equal(validate({ version: 1, requestId: 1, input: blockOrderDefault, response: extra }).phase,'rejected');
});

test('checker graph excludes producer arithmetic, model, display logarithms and reference fixtures', async () => {
  const bundle = await build({ entryPoints: ['src/lib/learning-block-order-validation-evaluator.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  const inputs = Object.keys(bundle.metafile!.inputs);
  for (const file of ['learning-block-order.ts','learning-block-order-evaluator.ts','learning-block-order-observations.ts','learning-finite-markov-exact.ts',
    'learning-interval-transcendentals.ts','learning-real-time-interval.ts']) assert(!inputs.includes(`src/lib/${file}`),file);
  assert(!inputs.some(file => file.includes('fixtures') || file.startsWith('src/data/')));
  writeFileSync(`${root}/checker-dependencies.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
});
