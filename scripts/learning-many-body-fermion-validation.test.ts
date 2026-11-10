import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readFermionFockSuccessForInput as read } from '../src/lib/learning-many-body-fermion-validation';
import { snapshotManyBodyWire } from '../src/lib/learning-many-body-owned-wire';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/many-body-fermion-response', { recursive: true });

const runtime = '.runtime/learning/many-body-fermion-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/fermion-fock-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'fermion-fock', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'fermion-fock'); return { input, packet }; }
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
for (const ref of refs.cases) test(`fermion response attributed to independent preparation: ${ref.label}`, () => {
  const input = snapshot({ kind: 'fermion-fock', parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok);
  const owned = read(structuredClone(packet), input, 7); assert(owned); assert.notEqual(owned, packet);
  assert(Object.isFrozen(owned)); assert(Object.isFrozen(owned.result.gamma[0][0].real.bounds));
  compare(owned.result, ref.expected); assert.equal(read(packet, input, 8), null);
});

test('root and nested fields, fixed descriptions, array dimensions and primitive metadata are exact', () => {
  const { input, packet } = preparation('general complex 2 modes sector0 case0');
  for (const key of Object.keys(packet.result)) rejects(input, packet, p => { delete p.result[key]; });
  for (const path of [[], ['input'], ['input', 'parameters'], ['result'], ['result', 'parameters'], ['result', 'terms', 0], ['result', 'probe'], ['result', 'operators', 0], ['result', 'operators', 0, 'annihilationColumns', 0], ['result', 'car', 0], ['result', 'parityAssumption'], ['result', 'parityDephased'], ['result', 'naiveComparison'], ['result', 'energy'], ['result', 'energy', 'bounds'], ['result', 'energy', 'bounds', 'lower']]) {
    const target = (p: any) => path.reduce((v: any, k) => v[k], p);
    rejects(input, packet, p => { target(p).foreign = 1; });
    const changed: any = structuredClone(packet); target(changed)[Symbol('extra')] = 0; assert.equal(read(changed, input, 7), null); rejectedMutations++;
  }
  for (const edit of [(p: any) => { p.version = 2; }, (p: any) => { p.requestId = 8; }, (p: any) => { p.inputKey += 'foreign'; }, (p: any) => { p.kind = 'spin-chain'; }, (p: any) => { p.result.dimension = 16; }, (p: any) => { p.result.occupationCounts[0] = 1; }, (p: any) => { p.result.normalizationPrecisionBits = 128; }, (p: any) => { p.result.terms[0].includesAdjoint = !p.result.terms[0].includesAdjoint; }, (p: any) => { p.result.approximationContract += ' ground certified'; }]) rejects(input, packet, edit);
});

test('every observation in complex formal and selected-parity diagonal responses rejects wrong valid rationals', () => {
  for (const label of ['general complex 2 modes sector0 case0', 'even sector minimum differs from full Fock minimum']) {
    const { input, packet } = preparation(label), paths: (string | number)[][] = [];
    function walk(value: unknown, path: (string | number)[]) {
      if (isMasterObservation(value)) { paths.push(path); return; }
      if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) walk(child, [...path, Array.isArray(value) ? Number(key) : key]);
    }
    walk(packet.result, ['result']); assert(paths.length > 100);
    for (const path of paths) rejects(input, packet, p => {
      const parent = path.slice(0, -1).reduce((v: any, k) => v[k], p), name = path.at(-1)!;
      parent[name] = exactObservation(BigInt(parent[name].value === 7 ? 8 : 7), BigInt(1));
    });
  }
});

test('annihilation columns, creation order, pairing adjoints and real/naive CAR comparisons cannot be swapped', () => {
  const { input, packet } = preparation('creation order on vacuum');
  rejects(input, packet, p => { p.result.probe.forward = structuredClone(p.result.probe.reverse); });
  for (const op of packet.result.operators) for (let column = 0; column < op.annihilationColumns.length; column++) {
    rejects(input, packet, p => { const a = p.result.operators[op.mode].annihilationColumns[column]; a.sign = a.sign === 1 ? -1 : 1; });
    rejects(input, packet, p => { const a = p.result.operators[op.mode].annihilationColumns[column]; a.row = a.row === null ? 0 : null; });
  }
  const mixed = preparation('same mode mixed CAR on vacuum');
  rejects(mixed.input, mixed.packet, p => { p.result.probe.expectedAnticommutatorScalar = 0; });
  const phase = preparation('complex pairing adjoint order');
  rejects(phase.input, phase.packet, p => { p.result.hamiltonian[0][3].imaginary = exactObservation(BigInt(7), BigInt(1)); });
  const long = preparation('long range hopping through occupied middle mode');
  rejects(long.input, long.packet, p => { p.result.energy = structuredClone(p.result.naiveComparison.energy); });
});

test('zero odd means do not erase formal parity coherence and twirl remains a different state', () => {
  const cat = preparation('formal cat has zero first odd moments'); assert(cat.packet.kind === 'fermion-fock');
  assert(cat.packet.result.oddMeans.every(v => v.real.sign === 0 && v.imaginary.sign === 0));
  assert.equal(cat.packet.result.parityAssumption.coherenceFrobeniusSquared.sign, 1);
  rejects(cat.input, cat.packet, p => { p.result.parityAssumption.pureStateCompatibleWithParitySuperselection = true; });
  rejects(cat.input, cat.packet, p => { p.result.parityAssumption.declaredInput = 'even-pure-sector'; });
  rejects(cat.input, cat.packet, p => { p.result.parityDephased.density = structuredClone(p.result.density); });
  const formal = preparation('formal even odd coherent one mode');
  rejects(formal.input, formal.packet, p => { p.result.parityDephased.oddMeans = structuredClone(p.result.oddMeans); });
});

test('full and selected-sector minima and global projector probabilities retain their different meanings', () => {
  const { input, packet } = preparation('even sector minimum differs from full Fock minimum'); assert(packet.kind === 'fermion-fock'); assert(packet.result.diagonalSpectrum);
  assert.equal(packet.result.classification.selectedStateGround, false); assert.equal(packet.result.classification.selectedStateParitySectorGround, true);
  for (const edit of [(p: any) => { p.result.classification.selectedStateGround = true; }, (p: any) => { p.result.classification.selectedStateParitySectorGround = false; }, (p: any) => { p.result.diagonalSpectrum.scope = 'selected-sector'; }, (p: any) => { p.result.diagonalSpectrum.parityMinima.reverse(); }, (p: any) => { p.result.diagonalSpectrum.numberMinima[0].number = 1; }, (p: any) => { p.result.diagonalSpectrum.levels[0].multiplicity++; }]) rejects(input, packet, edit);
  const noncommuting = preparation('even pair plus eigenstate');
  rejects(noncommuting.input, noncommuting.packet, p => { p.result.diagonalSpectrum = structuredClone(packet.result.diagonalSpectrum); });
  rejects(noncommuting.input, noncommuting.packet, p => { p.result.classification.selectedStateGround = true; });
});

test('input swaps, spoofed parameter attribution and conserved/sharp flags cannot hide another result', () => {
  const first = preparation('general complex 2 modes sector0 case0'), second = preparation('general complex 2 modes sector0 case1');
  rejects(first.input, first.packet, p => { p.result = structuredClone(second.packet.result); p.result.parameters = { ...first.input.parameters }; });
  assert.equal(read(second.packet, first.input, 7), null);
  for (const field of ['totalNumber', 'fermionParity']) for (const flag of ['conserved', 'selectedStateSharp']) rejects(first.input, first.packet, p => { p.result[field][flag] = !p.result[field][flag]; });
  const foreign = snapshot({ kind: 'spin-chain', parameters: JSON.parse(readFileSync('scripts/fixtures/spin-chain-reference.json', 'utf8')).cases[0].input });
  assert.equal(read(first.packet, foreign, 7), null);
});

test('display metadata, fractions, normalized coordinates and malformed arrays are rejected', () => {
  const { input, packet } = preparation('general complex 2 modes sector0 case0');
  for (const edit of [(p: any) => { p.result.energy.value = 1234567; }, (p: any) => { p.result.energy.logMagnitude = 1000; }, (p: any) => { p.result.energy.bounds.lower.numerator.exponent = 1; }, (p: any) => { p.result.energy.bounds.lower.denominator.coefficient = '0'; }, (p: any) => { p.result.energy.bounds.lower.numerator.coefficient = '1'.repeat(8193); }, (p: any) => { p.result.normalizedState[0].real = exactObservation(BigInt(7), BigInt(1)); }, (p: any) => { delete p.result.gamma[0][0]; }, (p: any) => { p.result.gamma.extra = 0; }, (p: any) => { p.result.state.push(p.result.state[0]); }, (p: any) => { p.result.probe.forward.length = 0; }]) rejects(input, packet, edit);
});

test('accessors, hostile reflection and caller get traps cannot execute while inspecting a response', () => {
  const { input, packet } = preparation('general complex 2 modes sector0 case0'); let calls = 0;
  for (const path of [['result', 'energy'], ['result', 'gamma', 0, 0, 'real'], ['result', 'parityDephased', 'density', 0, 0, 'real'], ['result', 'probe', 'forward', 0]]) {
    const changed: any = structuredClone(packet), parent = path.slice(0, -1).reduce((v: any, k) => v[k], changed);
    Object.defineProperty(parent, path.at(-1)!, { get() { calls++; return packet.result.energy; } });
    assert.equal(read(changed, input, 7), null);
  }
  assert.equal(calls, 0);
  const proxy = new Proxy(packet, { get() { calls++; throw Error('Get trap'); } }); assert(read(proxy, input, 7)); assert.equal(calls, 0);
  for (const value of [Object.create(packet), Object.assign(Object.create(null), packet), new Proxy({}, { ownKeys() { throw Error('Reflection'); } })]) assert.equal(read(value, input, 7), null);
  const saved = snapshotManyBodyWire(packet); assert(Object.isFrozen(saved));
});

test('fermion validation bundle excludes the product calculators, root isolation and interval/log engines', async () => {
  const code = await build({ entryPoints: ['src/lib/learning-many-body-fermion-validation.ts'], bundle: true, write: false, metafile: true, platform: 'browser', format: 'esm', target: 'es2020' });
  const inputs = Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p => /learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator|ritz|rayleigh)\.ts$|interval-transcendentals|quantum-evolution-interval/.test(p)), inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`, JSON.stringify({ inputs, bytes: code.outputFiles[0].contents.length, heavyNumericalEnginesExcluded: true }, null, 2) + '\n');
});
after(() => writeFileSync(`${runtime}/metrics.json`, JSON.stringify({ exactChecks, directedChecks, rejectedMutations }, null, 2) + '\n'));
