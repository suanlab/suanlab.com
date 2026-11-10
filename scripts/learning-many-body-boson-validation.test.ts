import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readBosonFockSuccessForInput as read } from '../src/lib/learning-many-body-boson-validation';
import { snapshotManyBodyWire } from '../src/lib/learning-many-body-owned-wire';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { readSpinChainSuccessForInput as readSpin } from '../src/lib/learning-many-body-spin-validation';
import { readFermionFockSuccessForInput as readFermion } from '../src/lib/learning-many-body-fermion-validation';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const runtime = '.runtime/learning/many-body-boson-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/boson-fock-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'boson-fock', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'boson-fock'); return { input, packet }; }
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
for (const ref of refs.cases) test(`raw boson response attributed to independent preparation: ${ref.label}`, () => {
  const input = snapshot({ kind: 'boson-fock', parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok);
  const owned = read(structuredClone(packet), input, 7); assert(owned, ref.label); assert.notEqual(owned, packet);
  assert(Object.isFrozen(owned.result.energy.coefficients)); compare(owned.result, ref.expected);
  assert.equal(read(packet, input, 8), null);
});

test('every field value in general and diagonal examples rejects a wrong exact coefficient', () => {
  for (const label of ['general complex 1 modes cutoff2-0', 'negative onsite finite ground has no infinite ground certificate']) {
    const { input, packet } = preparation(label), paths: (string | number)[][] = [];
    function walk(value: any, path: (string | number)[]) {
      if (value && typeof value === 'object' && Array.isArray(value.coefficients) && 'exactSign' in value) { paths.push(path); return; }
      if (value && typeof value === 'object') for (const [k, v] of Object.entries(value)) walk(v, [...path, Array.isArray(value) ? Number(k) : k]);
    }
    walk(packet.result, ['result']); assert(paths.length > 100);
    for (const path of paths) rejects(input, packet, p => {
      const field = path.reduce((v: any, k) => v[k], p);
      field.coefficients[0] = exactObservation(BigInt(field.coefficients[0].value === 7 ? 8 : 7), BigInt(1));
    });
  }
});

test('root/nested fields, dimensions, coefficients and exact signs are owned and complete', () => {
  const { input, packet } = preparation('general complex 1 modes cutoff2-0');
  for (const key of Object.keys(packet.result)) rejects(input, packet, p => { delete p.result[key]; });
  for (const path of [[], ['input'], ['input', 'parameters'], ['result'], ['result', 'parameters'], ['result', 'energy'], ['result', 'energy', 'observation'], ['result', 'localStates', 0], ['result', 'localStates', 0, 'quadratures'], ['result', 'commutation', 0], ['result', 'probe'], ['result', 'operators', 0]]) {
    const target = (p: any) => path.reduce((v: any, k) => v[k], p);
    rejects(input, packet, p => { target(p).foreign = 1; });
    const changed: any = structuredClone(packet); target(changed)[Symbol('foreign')] = true; assert.equal(read(changed, input, 7), null); rejectedMutations++;
  }
  for (const edit of [(p: any) => { p.version = 2; }, (p: any) => { p.kind = 'boson-coherent'; }, (p: any) => { p.inputKey += 'foreign'; }, (p: any) => { p.result.dimension++; }, (p: any) => { p.result.cutoffs[0] = 1; }, (p: any) => { p.result.basisOccupations[0][0] = 1; }, (p: any) => { p.result.energy.exactSign = p.result.energy.exactSign === 1 ? -1 : 1; }, (p: any) => { p.result.energy.coefficients.pop(); }, (p: any) => { p.result.energy.coefficients.extra = 0; }]) rejects(input, packet, edit);
});

test('finite commutator boundary, quadrature bound and cutoff0 cannot claim the infinite CCR', () => {
  const top = preparation('top occupation reverses finite commutator'); assert(top.packet.kind === 'boson-fock');
  assert.equal(top.packet.result.localStates[0].finiteCommutatorExpectation.observation.value, -3);
  for (const path of [['localStates', 0, 'finiteCommutatorExpectation'], ['localStates', 0, 'boundaryCorrection'], ['localStates', 0, 'quadratures', 'robertsonBound'], ['commutation', 0, 'infiniteCcrResidualFrobeniusSquared']]) rejects(top.input, top.packet, p => {
    const x = path.reduce((v: any, k) => v[k], p.result); x.coefficients[0] = exactObservation(BigInt(1), BigInt(1));
  });
  const zero = preparation('cutoff0 has vanishing quadratures');
  rejects(zero.input, zero.packet, p => { p.result.localStates[0].quadratures.xVariance.coefficients[0] = exactObservation(BigInt(1), BigInt(2)); });
  rejects(zero.input, zero.packet, p => { p.result.approximationContract += ' exact infinite CCR'; });
});

test('sqrt2 sqrt3 sqrt6 operator factors, complex phase and actual ordered probes are checked', () => {
  for (const label of ['number1 raises with sqrt2', 'number2 raises with sqrt3', 'boson hopping ladder enhancement sqrt6', 'complex drive and p orientation']) {
    const { input, packet } = preparation(label);
    rejects(input, packet, p => { p.result.probe.forward = structuredClone(p.result.probe.reverse); });
  }
  const p = preparation('general complex 1 modes cutoff2-0');
  rejects(p.input, p.packet, x => { x.result.operators[0].annihilation[1][2].real.coefficients[1] = exactObservation(BigInt(0), BigInt(1)); });
  const imaginary = preparation('imaginary hopping phase orientation');
  rejects(imaginary.input, imaginary.packet, p => { p.result.gamma[0][1].imaginary.exactSign *= -1; });
});

test('finite diagonal spectrum and conservation/sharpness never assert uncomputed infinite ground information', () => {
  const { input, packet } = preparation('negative onsite finite ground has no infinite ground certificate');
  for (const edit of [(p: any) => { p.result.diagonalSpectrum.scope = 'infinite-Fock-space'; }, (p: any) => { p.result.diagonalSpectrum.levels[0].multiplicity++; }, (p: any) => { p.result.classification.selectedStateFiniteCutoffGround = false; }, (p: any) => { p.result.totalNumber.conserved = !p.result.totalNumber.conserved; }, (p: any) => { p.result.totalNumber.selectedStateSharp = !p.result.totalNumber.selectedStateSharp; }]) rejects(input, packet, edit);
  const noncommuting = preparation('complex drive and p orientation');
  rejects(noncommuting.input, noncommuting.packet, p => { p.result.diagonalSpectrum = structuredClone(packet.result.diagonalSpectrum); });
});

test('observations cannot use rounded roots, forged unresolved metadata or another input result', () => {
  const first = preparation('general complex 1 modes cutoff2-0'), second = preparation('general complex 1 modes cutoff3-0');
  rejects(first.input, first.packet, p => { p.result.energy.observation = exactObservation(BigInt(7), BigInt(1)); });
  rejects(first.input, first.packet, p => { p.result.operators[0].annihilation[1][2].real.observation = exactObservation(BigInt(141421356), BigInt(100000000)); });
  rejects(first.input, first.packet, p => { const q = p.result.operators[0].annihilation[1][2].real.observation; q.resolution = 'unresolved-at-precision-ceiling'; q.value = null; q.logMagnitude = null; q.logResolution = 'unresolved'; });
  rejects(first.input, first.packet, p => { p.result = structuredClone(second.packet.result); p.result.parameters = { ...first.input.parameters }; });
  // Shared metadata hardening applies to the previously verified families too.
  for (const kind of ['spin-chain', 'fermion-fock'] as const) {
    const ref = JSON.parse(readFileSync(`scripts/fixtures/${kind}-reference.json`, 'utf8')).cases[0];
    const input = snapshot({ kind, parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && (packet.kind === 'spin-chain' || packet.kind === 'fermion-fock'));
    const readFamily = kind === 'spin-chain' ? readSpin : readFermion; assert(readFamily(packet, input, 7));
    const changed: any = structuredClone(packet), q = changed.result.normalizedState.flatMap((z: any) => [z.real, z.imaginary]).find((q: Q) => q.resolution === 'interval-resolved'); assert(q);
    q.resolution = 'unresolved-at-precision-ceiling'; q.value = null; q.logMagnitude = null; q.logResolution = 'unresolved';
    assert.equal(readFamily(changed, input, 7), null); rejectedMutations++;
  }
});

test('getters, sparse arrays, symbols and hostile reflection cannot bypass descriptor-only copying', () => {
  const { input, packet } = preparation('general complex 1 modes cutoff2-0'); let calls = 0;
  for (const path of [['result', 'energy'], ['result', 'energy', 'coefficients', 0], ['result', 'probe', 'forward', 0]]) {
    const changed: any = structuredClone(packet), parent = path.slice(0, -1).reduce((v: any, k) => v[k], changed);
    Object.defineProperty(parent, path.at(-1)!, { get() { calls++; return packet.result.energy; } }); assert.equal(read(changed, input, 7), null);
  }
  assert.equal(calls, 0); assert(read(new Proxy(packet, { get() { calls++; throw Error('get'); } }), input, 7)); assert.equal(calls, 0);
  for (const edit of [(p: any) => { delete p.result.state[0]; }, (p: any) => { p.result.state.extra = 1; }, (p: any) => { p.result.probe.forward.pop(); }]) rejects(input, packet, edit);
  const bad = new Proxy({}, { getPrototypeOf() { throw Error('reflection'); } }); assert.equal(read(bad, input, 7), null);
  assert(Object.isFrozen(snapshotManyBodyWire(packet)));
});

test('raw boson validator bundle excludes all production calculators, field observation and root/interval engines', async () => {
  const code = await build({ entryPoints: ['src/lib/learning-many-body-boson-validation.ts'], bundle: true, write: false, metafile: true, platform: 'browser', format: 'esm', target: 'es2020' });
  const inputs = Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p => /learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|boson-algebraic|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator|ritz|rayleigh)\.ts$|interval-transcendentals|quantum-evolution-interval/.test(p)), inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`, JSON.stringify({ inputs, bytes: code.outputFiles[0].contents.length, heavyNumericalEnginesExcluded: true }, null, 2) + '\n');
});
after(() => writeFileSync(`${runtime}/metrics.json`, JSON.stringify({ exactChecks, directedChecks, rejectedMutations }, null, 2) + '\n'));
