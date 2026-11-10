import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readBosonCoherentSuccessForInput as read } from '../src/lib/learning-many-body-coherent-validation';
import { snapshotManyBodyWire } from '../src/lib/learning-many-body-owned-wire';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { coherentExpCertificate } from '../src/lib/learning-many-body-coherent-wire';
import { f as wf } from '../src/lib/learning-tensor-wire-math';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/many-body-coherent-response', { recursive: true });

const runtime = '.runtime/learning/many-body-coherent-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/boson-coherent-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'boson-coherent', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'boson-coherent'); return { input, packet }; }
let rejectedMutations = 0, exponentialReferenceChecks = 0;
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
for (const ref of refs.cases) test(`coherent response attributed to independent preparation: ${ref.label}`, () => {
  const input = snapshot({ kind: 'boson-coherent', parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok);
  const owned = read(structuredClone(packet), input, 7); assert(owned, ref.label); assert.notEqual(owned, packet);
  assert(Object.isFrozen(owned.result.normalizedProjection.energy.coefficients)); compare(owned.result, ref.expected);
  assert.equal(read(packet, input, 8), null);
});

test('direct alternating exponential certificates match independent 245-digit vacuum amplitudes/probabilities', () => {
  let checked = 0;
  for (const ref of refs.cases) for (let mode = 0; mode < ref.input.modes; mode++) {
    const p = ref.input, d = BigInt(p[`alphaDenominator${mode}`]), a = BigInt(p[`alphaReal${mode}`]), b = BigInt(p[`alphaImag${mode}`]);
    const mu = wf(a*a+b*b,d*d), targets = [ref.expected as any];
    for (const [exponent, expected] of [[mu, targets[0].perMode[mode].unnormalizedProjection.probabilities[0]], [wf(mu.n,mu.d*BigInt(2)), targets[0].perMode[mode].unnormalizedProjection.amplitudes[0].real]] as const) {
      const r = coherentExpCertificate(exponent), value = parse(expected.startsWith('decimal:') ? expected.slice(8) : expected), eps = frac(BigInt(1), BigInt(10)**BigInt(240));
      assert(le(r.lower, add(value,eps)) && le(add(value,frac(-eps.n,eps.d)),r.upper));
      const width = frac(r.upper.n*r.lower.d-r.lower.n*r.upper.d,r.upper.d*r.lower.d); assert(le(width, frac(BigInt(1),BigInt(10)**BigInt(300)))); checked++;
      assert(Object.isFrozen(r) && Object.isFrozen(r.lower));
    }
  }
  exponentialReferenceChecks = checked; assert(checked >= 80); assert.throws(()=>coherentExpCertificate(wf(BigInt(-1)))); assert.throws(()=>coherentExpCertificate(wf(BigInt(33))));
});

test('every observation in a nonzero coherent response rejects a wrong valid rational', () => {
  const { input, packet } = preparation('general coherent 1 modes cutoff1-0'), paths: (string | number)[][] = [];
  function walk(value: unknown, path: (string | number)[]) {
    if (isMasterObservation(value)) { paths.push(path); return; }
    if (value && typeof value === 'object') for (const [k,v] of Object.entries(value)) walk(v,[...path,Array.isArray(value)?Number(k):k]);
  }
  walk(packet.result,['result']); assert(paths.length>100);
  for(const path of paths) rejects(input,packet,p=>{const parent=path.slice(0,-1).reduce((v:any,k)=>v[k],p),key=path.at(-1)!;parent[key]=exactObservation(BigInt(parent[key].value===7?8:7),BigInt(1));});
});

test('root/nested fields, preparation controls, certificate descriptions and normalized-state schema are exact',()=>{
  const {input,packet}=preparation('unit real alpha cutoff3');
  for(const key of Object.keys(packet.result)) rejects(input,packet,p=>{delete p.result[key];});
  for(const path of [[],['input'],['input','parameters'],['result'],['result','parameters'],['result','perMode',0],['result','perMode',0,'infiniteState'],['result','perMode',0,'unnormalizedProjection'],['result','perMode',0,'normalizedProjection'],['result','normalizedProjection'],['result','truncation'],['result','certificate']]){
    const target=(p:any)=>path.reduce((v:any,k)=>v[k],p); rejects(input,packet,p=>{target(p).extra=0;});
    const changed:any=structuredClone(packet);target(changed)[Symbol('extra')]=0;assert.equal(read(changed,input,7),null);rejectedMutations++;
  }
  for(const edit of [(p:any)=>{p.version=2;},(p:any)=>{p.kind='boson-fock';},(p:any)=>{p.inputKey+='foreign';},(p:any)=>{p.result.certificate.exponentialPrecisionBits=512;},(p:any)=>{p.result.certificate.exponentialMethod='finite-displacement';},(p:any)=>{p.result.normalizedProjection.parameters={};}]) rejects(input,packet,edit);
});

test('infinite state, unnormalized retained component and renormalized finite state cannot be interchanged',()=>{
  const {input,packet}=preparation('unit real alpha cutoff3');
  rejects(input,packet,p=>{p.result.unnormalizedProjection.amplitudes=structuredClone(p.result.normalizedProjection.normalizedState);});
  rejects(input,packet,p=>{p.result.unnormalizedProjection.normalizationApplied=true;});
  rejects(input,packet,p=>{p.result.normalizedProjection.normalizationApplied=false;});
  rejects(input,packet,p=>{p.result.infiniteState.meanTotalNumber=structuredClone(p.result.normalizedProjection.totalNumber.expectation.observation);});
  rejects(input,packet,p=>{p.result.infiniteState.noGroundCertificate=false;});
  const driven=preparation('unit imaginary alpha drive');
  rejects(driven.input,driven.packet,p=>{p.result.infiniteState.energyForSameNormallyOrderedHamiltonian=structuredClone(p.result.normalizedProjection.energy);});
});

test('consistent but false tail/fidelity/overlap/distance values fail independent exponential attribution',()=>{
  const {input,packet}=preparation('unit real alpha cutoff3');
  const point=(n:number)=>exactObservation(BigInt(n),BigInt(1));
  rejects(input,packet,p=>{
    p.result.truncation.retainedProbability=point(1);p.result.truncation.discardedProbability=point(0);p.result.truncation.productOfModeRetained=point(1);
    p.result.unnormalizedProjection.normSquared=point(1);p.result.unnormalizedProjection.distanceSquaredToInfinite=point(0);
    p.result.normalizedProjection.fidelitySquaredToInfinite=point(1);p.result.normalizedProjection.overlapWithInfinite=point(1);p.result.normalizedProjection.distanceSquaredToInfinite=point(0);
  });
  rejects(input,packet,p=>{p.result.normalizedProjection.distanceSquaredToInfinite=structuredClone(p.result.truncation.discardedProbability);});
  rejects(input,packet,p=>{p.result.normalizedProjection.fidelitySquaredToInfinite=structuredClone(p.result.normalizedProjection.overlapWithInfinite);});
  rejects(input,packet,p=>{p.result.perMode[0].normalizedProjection.annihilationEigenResidualSquared=structuredClone(p.result.normalizedProjection.exactResiduals.varianceAccount);});
});

test('tiny positive tails are not rounded to zero and zero-alpha inputs retain the exact vacuum',()=>{
  const tiny=preparation('tiny mu positive Poisson tail');assert(tiny.packet.kind==='boson-coherent');
  assert.equal(tiny.packet.result.truncation.discardedProbability.sign,1);
  rejects(tiny.input,tiny.packet,p=>{p.result.truncation.discardedProbability=exactObservation(BigInt(0),BigInt(1));});
  const vacuum=preparation('vacuum coherent cutoff3');
  rejects(vacuum.input,vacuum.packet,p=>{p.result.normalizedProjection.distanceSquaredToInfinite=exactObservation(BigInt(1),BigInt(1));});
  rejects(vacuum.input,vacuum.packet,p=>{p.result.perMode[0].normalizedProjection.finiteCommutatorExpectation=exactObservation(BigInt(0),BigInt(1));});
});

test('rectangular product retention and two-mode finite correlations remain tied to both alphas and cutoffs',()=>{
  const {input,packet}=preparation('general coherent 2 modes cutoff1-2');
  rejects(input,packet,p=>{p.result.perMode.reverse();});
  rejects(input,packet,p=>{p.result.truncation.productOfModeRetained=structuredClone(p.result.perMode[0].unnormalizedProjection.retainedProbability);});
  rejects(input,packet,p=>{p.result.normalizedProjection.gamma[0][1].real.coefficients[0]=exactObservation(BigInt(7),BigInt(1));});
  rejects(input,packet,p=>{p.result.normalizedProjection.localStates[0].boundaryCorrection.coefficients[0]=exactObservation(BigInt(0),BigInt(1));});
});

test('another preparation, spoofed controls and dense/sparse arrays cannot hide a different state',()=>{
  const first=preparation('general coherent 1 modes cutoff1-0'),second=preparation('general coherent 1 modes cutoff2-0');
  rejects(first.input,first.packet,p=>{p.result=structuredClone(second.packet.result);p.result.parameters={...first.input.parameters};});
  for(const edit of [(p:any)=>{delete p.result.exactRawCoefficients[0];},(p:any)=>{p.result.perMode.extra=0;},(p:any)=>{p.result.normalizedProjection.normalizedState.pop();},(p:any)=>{p.result.truncation.retainedProbability.value=1234567;}]) rejects(first.input,first.packet,edit);
});

test('caller getters, get traps, prototypes and hostile reflection cannot execute during validation',()=>{
  const {input,packet}=preparation('general coherent 1 modes cutoff1-0');let calls=0;
  for(const path of [['result','truncation','retainedProbability'],['result','normalizedProjection','state',0],['result','perMode',0,'rawCoefficients',0]]){
    const changed:any=structuredClone(packet),parent=path.slice(0,-1).reduce((v:any,k)=>v[k],changed);
    Object.defineProperty(parent,path.at(-1)!,{get(){calls++;return packet.result.rawSeriesNorm;}});assert.equal(read(changed,input,7),null);
  }
  assert.equal(calls,0);assert(read(new Proxy(packet,{get(){calls++;throw Error('get');}}),input,7));assert.equal(calls,0);
  for(const value of [Object.create(packet),Object.assign(Object.create(null),packet),new Proxy({},{ownKeys(){throw Error('reflection');}})])assert.equal(read(value,input,7),null);
});

test('coherent validator bundle excludes every production calculator and transcendental/field observation engine',async()=>{
  const code=await build({entryPoints:['src/lib/learning-many-body-coherent-validation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|boson-algebraic|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator|ritz|rayleigh)\.ts$|interval-transcendentals|quantum-evolution-interval/.test(p)),inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`,JSON.stringify({inputs,bytes:code.outputFiles[0].contents.length,heavyNumericalEnginesExcluded:true},null,2)+'\n');
});
after(() => writeFileSync(`${runtime}/metrics.json`, JSON.stringify({ exactChecks, directedChecks, rejectedMutations, exponentialReferenceChecks }, null, 2) + '\n'));
