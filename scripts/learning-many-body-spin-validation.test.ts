import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readSpinChainSuccessForInput as read } from '../src/lib/learning-many-body-spin-validation';
import { snapshotManyBodyWire } from '../src/lib/learning-many-body-owned-wire';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const runtime = '.runtime/learning/many-body-spin-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/spin-chain-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'spin-chain', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'spin-chain'); return { input, packet }; }
let rejectedMutations = 0;
function rejects(input: ManyBodyInput, packet: unknown, edit: (packet: any) => void) { const changed = structuredClone(packet); edit(changed); assert.notDeepEqual(changed, packet, 'mutation changes data'); assert.equal(read(changed, input, 7), null); rejectedMutations++; }
type F = { n: bigint; d: bigint };
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
const add = (a: F, b: F) => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a: F, b: F) => frac(a.n * b.n, a.d * b.d);
function parse(raw: string): F {
  if (raw.includes('/')) { const [n, d] = raw.split('/'); return frac(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(raw); assert(m, raw);
  const n = BigInt(m[1] + m[2] + (m[3] ?? '')), exponent = Number(m[4] ?? 0) - (m[3] ?? '').length;
  return exponent >= 0 ? frac(n * BigInt(10) ** BigInt(exponent)) : frac(n, BigInt(10) ** BigInt(-exponent));
}
function endpoint(b: NonNullable<Q['bounds']>['lower']): F {
  const n = BigInt(b.numerator.coefficient), d = BigInt(b.denominator.coefficient), e = b.numerator.exponent - b.denominator.exponent;
  return e >= 0 ? frac(n * BigInt(10) ** BigInt(e), d) : frac(n, d * BigInt(10) ** BigInt(-e));
}
const exact = (q: Q) => { assert(q.bounds); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(equal(lo, hi)); return lo; };
let exactChecks = 0, directedChecks = 0;
function observation(actual: unknown, expected: string, path: string) {
  assert(isMasterObservation(actual), path); const q = actual as Q; assert(q.bounds, path);
  const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(le(lo, hi), path);
  const directed = expected.startsWith('decimal:'), value = parse(directed ? expected.slice(8) : expected);
  const absolute = value.n < BigInt(0) ? -value.n : value.n, scale = absolute > value.d ? absolute : value.d;
  if (directed) {
    const eps = BigInt(10) ** BigInt(205);
    assert(le(lo, frac(value.n * eps + scale, value.d * eps)) && le(frac(value.n * eps - scale, value.d * eps), hi), path + ' reference overlap');
    directedChecks++;
  } else { assert(equal(lo, value) && equal(hi, value), path + ' exact equality'); exactChecks++; }
  const width = frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d);
  assert(le(width, frac(scale, value.d * BigInt(10) ** BigInt(125))), path + ' directed width');
}
function compare(actual: unknown, expected: unknown, path = 'result') {
  if (typeof expected === 'string' && (/^-?\d/.test(expected) || expected.startsWith('decimal:'))) return observation(actual, expected, path);
  if (Array.isArray(expected)) { assert(Array.isArray(actual), path); assert.equal(actual.length, expected.length, path); expected.forEach((v, i) => compare(actual[i], v, `${path}[${i}]`)); return; }
  if (expected && typeof expected === 'object') { assert(actual && typeof actual === 'object', path); for (const [key, value] of Object.entries(expected)) compare((actual as Record<string, unknown>)[key], value, `${path}.${key}`); return; }
  assert.equal(actual, expected, path);
}
let directedCertificateChecks = 0;
function directedZero(value: unknown) {
  if (value === null) return;
  if (isMasterObservation(value)) {
    const v = value as Q; assert(v.bounds);
    const lo = endpoint(v.bounds.lower), hi = endpoint(v.bounds.upper);
    assert(le(lo, frac(BigInt(0))) && le(frac(BigInt(0)), hi), 'directed certificate contains0');
    assert(le(frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d), frac(BigInt(1), BigInt(10) ** BigInt(125))), 'directed certificate width');
    directedCertificateChecks++; return;
  }
  assert(value && typeof value === 'object'); Object.values(value).forEach(directedZero);
}
for (const ref of refs.cases) test(`spin response attributed to independent preparation: ${ref.label}`, () => {
  const input = snapshot({ kind: 'spin-chain', parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok);
  const owned = read(structuredClone(packet), input, 7); assert(owned); assert.notEqual(owned, packet);
  assert(Object.isFrozen(owned)); assert(Object.isFrozen(owned.result.hamiltonian[0][0].real.bounds));
  compare(owned.result, ref.expected);
  assert.equal(read(packet, input, 8), null);
});

test('missing root fields, foreign nested fields and wrong primitive metadata are rejected', () => {
  const { input, packet } = preparation('general complex 2 sites graph2 case0');
  for (const name of Object.keys(packet.result)) rejects(input, packet, p => { delete p.result[name]; });
  for (const path of [[], ['result'], ['input'], ['input', 'parameters'], ['result', 'parameters'], ['result', 'terms', 0], ['result', 'localStates', 0], ['result', 'totalZ'], ['result', 'classification'], ['result', 'cost'], ['result', 'exactResiduals'], ['result', 'energy'], ['result', 'energy', 'bounds'], ['result', 'energy', 'bounds', 'lower']]) {
    const target = (p: any) => path.reduce((v: any, k) => v[k], p);
    rejects(input, packet, p => { target(p).foreign = 1; });
    const symbol = structuredClone(packet); target(symbol)[Symbol('foreign')] = 1; assert.equal(read(symbol, input, 7), null); rejectedMutations++;
  }
  for (const edit of [(p: any) => { p.version = 2; }, (p: any) => { p.requestId = 8; }, (p: any) => { p.kind = 'mps'; }, (p: any) => { p.inputKey = 'foreign'; }, (p: any) => { p.result.dimension = 16; }, (p: any) => { p.result.normalizationPrecisionBits = 128; }, (p: any) => { p.result.approximationContract += ' computed ground'; }, (p: any) => { p.result.classification.exactEigenstate = !p.result.classification.exactEigenstate; }, (p: any) => { p.result.terms[0].word = 'ZZ'; }]) rejects(input, packet, edit);
});

test('all numeric observations in noncommuting and diagonal packets reject valid but wrong rational replacements', () => {
  for (const label of ['general complex 2 sites graph2 case0', 'GHZ4 diagonal ferromagnet']) {
    const { input, packet } = preparation(label), paths: (string | number)[][] = [];
    function walk(v: unknown, path: (string | number)[]) {
      if (isMasterObservation(v)) { paths.push(path); return; }
      if (v && typeof v === 'object') for (const [key, value] of Object.entries(v)) walk(value, [...path, Array.isArray(v) ? Number(key) : key]);
    }
    walk(packet.result, ['result']); assert(paths.length > 100);
    for (const path of paths) rejects(input, packet, p => {
      const target = path.slice(0, -1).reduce((v: any, k) => v[k], p), name = path.at(-1)!;
      target[name] = exactObservation(BigInt(target[name].value === 7 ? 8 : 7), BigInt(1));
    });
  }
});

test('display value/log metadata, malformed fractions and normalized signs/enclosures cannot bypass exact accounts', () => {
  const { input, packet } = preparation('general complex 2 sites graph2 case0');
  for (const edit of [(p: any) => { p.result.energy.value = 1000000; }, (p: any) => { p.result.energy.logMagnitude = 1000; }, (p: any) => { p.result.energy.sign = null; }, (p: any) => { p.result.energy.bounds.lower.denominator.coefficient = '0'; }, (p: any) => { p.result.energy.bounds.lower.numerator.coefficient = '1'.repeat(8193); }, (p: any) => { p.result.energy.bounds.lower.numerator.exponent = 1; }, (p: any) => { p.result.normalizedState[0].real = exactObservation(BigInt(1), BigInt(1)); }, (p: any) => { p.result.variance.value = Infinity; }]) {
    const changed = structuredClone(packet); edit(changed); assert.equal(read(changed, input, 7), null); rejectedMutations++;
  }
});

test('diagonal level partitions, gaps, probabilities and false ground claims stay tied to the Hamiltonian', () => {
  const { input, packet } = preparation('GHZ4 diagonal ferromagnet');
  for (const edit of [(p: any) => { p.result.diagonalSpectrum.levels[0].basisIndices.reverse(); }, (p: any) => { p.result.diagonalSpectrum.levels[0].multiplicity++; }, (p: any) => { p.result.diagonalSpectrum.levels.pop(); }, (p: any) => { p.result.diagonalSpectrum.groundMultiplicity++; }, (p: any) => { p.result.diagonalSpectrum.selectedStateGround = false; }, (p: any) => { p.result.classification.selectedStateGround = false; }]) rejects(input, packet, edit);
  const noncommuting = preparation('GHZ4 transverse field');
  rejects(noncommuting.input, noncommuting.packet, p => { p.result.diagonalSpectrum = packet.result.diagonalSpectrum; });
  rejects(noncommuting.input, noncommuting.packet, p => { p.result.classification.selectedStateGround = true; });
});

test('same-shape response swaps and a spoofed input/key cannot change the physical result attribution', () => {
  const first = preparation('general complex 2 sites graph2 case0'), second = preparation('general complex 2 sites graph2 case1');
  rejects(first.input, first.packet, p => { p.result = second.packet.result; p.result.parameters = { ...first.input.parameters }; });
  assert.equal(read(second.packet, first.input, 7), null);
  const foreign = snapshot({ kind: 'mps', parameters: JSON.parse(readFileSync('scripts/fixtures/mps-reference.json', 'utf8')).cases[0].input });
  assert.equal(read(first.packet, foreign, 7), null);
});

test('all nested accessors, caller get traps and hostile reflection are handled without invoking getters', () => {
  const { input, packet } = preparation('general complex 2 sites graph2 case0'); let calls = 0;
  for (const path of [['result', 'energy'], ['result', 'hamiltonian', 0, 0, 'real'], ['result', 'terms', 0, 'coefficient']]) {
    const changed: any = structuredClone(packet), parent = path.slice(0, -1).reduce((v: any, k) => v[k], changed);
    Object.defineProperty(parent, path.at(-1)!, { get() { calls++; return packet.result.energy; } });
    assert.equal(read(changed, input, 7), null);
  }
  assert.equal(calls, 0);
  const proxy = new Proxy(packet, { get() { calls++; throw Error('Get trap'); } });
  assert(read(proxy, input, 7)); assert.equal(calls, 0);
  for (const value of [Object.create(packet), Object.assign(Object.create(null), packet), new Proxy({}, { ownKeys() { throw Error('Reflection'); } })]) assert.equal(read(value, input, 7), null);
});

test('sparse, foreign, accessor and wrong-length arrays are rejected', () => {
  const { input, packet } = preparation('general complex 2 sites graph2 case0');
  for (const edit of [(p: any) => { delete p.result.state[0]; }, (p: any) => { p.result.state.extra = 0; }, (p: any) => { p.result.state.push(p.result.state[0]); }, (p: any) => { p.result.correlations[0].values[0].pop(); }, (p: any) => { p.result.hamiltonian[0].length = 0; }]) rejects(input, packet, edit);
  let calls = 0; const changed = structuredClone(packet); Object.defineProperty(changed.result.state, '0', { get() { calls++; return packet.result.state[0]; } });
  assert.equal(read(changed, input, 7), null); assert.equal(calls, 0);
});

test('owned snapshot bounds depth, nodes and shapes while preserving aliases and ordinary __proto__ data', () => {
  const shared = { value: 1 }, source = { a: shared, b: shared }, saved = snapshotManyBodyWire(source) as typeof source;
  assert.equal(saved.a, saved.b); assert.notEqual(saved.a, source.a); assert(Object.isFrozen(saved.a));
  const cycle: any = {}; cycle.self = cycle; assert.throws(() => snapshotManyBodyWire(cycle));
  let deep: any = 0; for (let i = 0; i < 66; i++) deep = { child: deep }; assert.throws(() => snapshotManyBodyWire(deep));
  assert.throws(() => snapshotManyBodyWire(Array(65537).fill(0)));
  assert.throws(() => snapshotManyBodyWire(Array.from({ length: 1024 }, () => Array(512).fill(0))));
  assert.throws(() => snapshotManyBodyWire({ value: undefined })); assert.throws(() => snapshotManyBodyWire({ value: BigInt(1) }));
  const proto = JSON.parse('{"__proto__":{"bad":1}}'), owned = snapshotManyBodyWire(proto) as Record<string, unknown>;
  assert.equal(Object.getPrototypeOf(owned), Object.prototype); assert(Object.hasOwn(owned, '__proto__'));
});

test('response validation bundle excludes calculator, spectral and interval engines', async () => {
  const code = await build({ entryPoints: ['src/lib/learning-many-body-spin-validation.ts'], bundle: true, write: false, metafile: true, platform: 'browser', format: 'esm', target: 'es2020' });
  const inputs = Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p => /learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator|ritz|rayleigh)\.ts$|interval-transcendentals|quantum-evolution-interval/.test(p)), inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`, JSON.stringify({ inputs, bytes: code.outputFiles[0].contents.length, heavyNumericalEnginesExcluded: true }, null, 2) + '\n');
});
after(() => writeFileSync(`${runtime}/metrics.json`, JSON.stringify({ exactChecks, directedChecks, rejectedMutations }, null, 2) + '\n'));
