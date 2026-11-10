import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readMpsSuccessForInput as read } from '../src/lib/learning-many-body-mps-validation';
import { createMpsParameters } from '../src/lib/learning-mps-input';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/many-body-mps-response', { recursive: true });

const runtime = '.runtime/learning/many-body-mps-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/mps-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'mps', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'mps'); return { input, packet }; }
let rejectedMutations = 0;
function rejects(input: ManyBodyInput, packet: unknown, edit: (packet: any) => void) { const changed = structuredClone(packet); edit(changed); assert.notDeepEqual(changed, packet, 'mutation changes data'); assert.equal(read(changed, input, 7), null); rejectedMutations++; }
type F = { n: bigint; d: bigint };
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
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
for (const ref of refs.cases) test(`MPS response matches independent reference: ${ref.label}`, () => {
  const input = snapshot({ kind: 'mps', parameters: ref.input }), packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'mps');
  const owned = read(structuredClone(packet), input, 7); assert(owned, ref.label); assert.notEqual(owned, packet);
  assert(Object.isFrozen(owned.result.original.tensors[0][0][0])); compare(owned.result, ref.expected);
  assert.equal(read(packet, input, 8), null);
});

test('every observation rejects a wrong valid rational across physical and virtual coordinates', () => {
  for (const label of ['GHZ3 scale shear parameter norm changes', 'product right Y positive']) {
    const { input, packet } = preparation(label), paths: (string | number)[][] = [];
    function walk(value: unknown, path: (string | number)[]) {
      if (isMasterObservation(value)) { paths.push(path); return; }
      if (value && typeof value === 'object') for (const [k,v] of Object.entries(value)) walk(v,[...path,Array.isArray(value)?Number(k):k]);
    }
    walk(packet.result,['result']); assert(paths.length > 100);
    for (const path of paths) rejects(input,packet,p=>{const parent=path.slice(0,-1).reduce((v:any,k)=>v[k],p),key=path.at(-1)!;parent[key]=exactObservation(BigInt(parent[key].value===7?8:7),BigInt(1));});
  }
});

test('root, contracts, input identity and array ownership reject forged or missing fields',()=>{
  const {input,packet}=preparation('complex Bell phase fixes environment orientation');
  const paths = [[],['input'],['result'],['result','parameters'],['result','cost'],['result','validity'],['result','original'],['result','gauged'],['result','gauge']];
  for(const path of paths){
    rejects(input,packet,p=>{path.reduce((v:any,k)=>v[k],p).extra=0;});
    const obj=path.reduce((v:any,k)=>v[k],packet); for(const key of Object.keys(obj)) rejects(input,packet,p=>{delete path.reduce((v:any,k)=>v[k],p)[key];});
  }
  for(const edit of [(p:any)=>{p.requestId++;},(p:any)=>{p.kind='schmidt';},(p:any)=>{p.inputKey+=' ';},(p:any)=>{p.result.original.state.extra=0;},(p:any)=>{delete p.result.original.state[0];},(p:any)=>{p.result.original.state.pop();},(p:any)=>{p.result.original.normSquared.value=1234567;},(p:any)=>{p.result.original.normSquared.resolution='unresolved-at-precision-ceiling';}]) rejects(input,packet,edit);
  rejects(input,packet,p=>{p.result.approximationContract+=' ';});
  const symbolic:any=structuredClone(packet);symbolic.result[Symbol('hidden')]=1;assert.equal(read(symbolic,input,7),null);
  assert.equal(read(packet,input,0),null);assert.equal(read(packet,input,NaN),null);
});

test('gauge changes representation coordinates while preserving the physical state',()=>{
  const {input,packet}=preparation('GHZ3 scale shear parameter norm changes');
  assert.notDeepEqual(packet.result.original.tensorParameterSquaredNorm,packet.result.gauged.tensorParameterSquaredNorm);
  assert.deepEqual(packet.result.original.state,packet.result.gauged.state);
  for(const field of ['tensors','prefixes','leftEnvironments','rightEnvironments','tensorParameterSquaredNorm','canonicality']) rejects(input,packet,p=>{p.result.gauged[field]=structuredClone(p.result.original[field]);});
  rejects(input,packet,p=>{p.result.gauge.inverse=structuredClone(p.result.gauge.matrix);});
  const complex=preparation('general complex MPS profile2 sites2 bonds[2]');
  rejects(complex.input,complex.packet,p=>{p.result.original.rightEnvironments[1]=structuredClone(p.result.original.leftEnvironments[1]);});
});

test('virtual bond dimension does not imply Schmidt rank or canonical tensors',()=>{
  const {input,packet}=preparation('redundant bond4 product rank1');
  assert.equal(packet.result.original.cuts[0].bondDimension,4);assert.equal(packet.result.original.cuts[0].schmidtRank,1);
  rejects(input,packet,p=>{p.result.original.cuts[0].schmidtRank=4;p.result.original.cuts[0].coefficientRank=4;});
  rejects(input,packet,p=>{p.result.original.canonicality[0].leftIsometry=!p.result.original.canonicality[0].leftIsometry;});
});

test('nonzero local tensors may contract to zero without a physical normalized state',()=>{
  const {input,packet}=preparation('nonzero tensors cancel to global zero');
  assert.equal(packet.result.original.physical,null);assert.equal(packet.result.original.cuts[0].schmidtRank,null);
  rejects(input,packet,p=>{p.result.original.cuts[0].schmidtRank=0;});
  rejects(input,packet,p=>{p.result.validity.physicalStateDefined=true;p.result.original.stateDefined=true;});
  rejects(input,packet,p=>{p.result.original.physical={normalizedState:[],localStates:[],correlations:[]};});
  const one=preparation('one site no virtual gauge');
  rejects(one.input,one.packet,p=>{p.result.gauge={};});
  rejects(one.input,one.packet,p=>{p.result.exactResiduals.leftEnvironmentCovariance=exactObservation(BigInt(0),BigInt(1));});
});

test('normalization width follows normalized amplitude even for huge raw coordinates',()=>{
  const p={...createMpsParameters(4,1)};for(let i=0;i<4;i++)p[`t${i}Real000`]=1000000;
  const input=snapshot({kind:'mps',parameters:p}),packet=evaluate({version:1,requestId:7,input});assert(packet.ok&&packet.kind==='mps');assert(read(packet,input,7));
  rejects(input,packet,p=>{
    const q=p.result.original.physical.normalizedState[0].real,d=BigInt(10)**BigInt(120);
    q.bounds.lower={numerator:{coefficient:String(d-BigInt(1)),exponent:0},denominator:{coefficient:String(d),exponent:0}};
    q.bounds.upper={numerator:{coefficient:String(d+BigInt(1)),exponent:0},denominator:{coefficient:String(d),exponent:0}};
    q.value=1;q.sign=1;q.logMagnitude=0;q.resolution='interval-resolved';q.logResolution='finite';
    assert(isMasterObservation(q),'counterexample is a structurally valid observation');
  });
});

test('responses from another preparation cannot be laundered through copied input controls',()=>{
  const first=preparation('product right Y positive'),second=preparation('complex Bell phase fixes environment orientation');
  rejects(first.input,first.packet,p=>{p.result=structuredClone(second.packet.result);p.result.parameters={...first.input.parameters};});
});

test('getters, get traps, inherited records and hostile reflection are never trusted',()=>{
  const {input,packet}=preparation('product right Y positive');let calls=0;
  for(const path of [['result','original','normSquared'],['result','original','state',0],['result','gauged','canonicality',0]]){
    const changed:any=structuredClone(packet),parent=path.slice(0,-1).reduce((v:any,k)=>v[k],changed);
    Object.defineProperty(parent,path.at(-1)!,{get(){calls++;return 0;}});assert.equal(read(changed,input,7),null);
  }
  assert.equal(calls,0);assert(read(new Proxy(packet,{get(){calls++;throw Error('get');}}),input,7));assert.equal(calls,0);
  for(const value of [Object.create(packet),Object.assign(Object.create(null),packet),new Proxy({},{ownKeys(){throw Error('reflection');}})])assert.equal(read(value,input,7),null);
});

test('MPS validator bundle excludes production calculators, contractions and spectral engines',async()=>{
  const code=await build({entryPoints:['src/lib/learning-many-body-mps-validation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|boson-algebraic|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|many-body-evaluator|ritz|rayleigh)\.ts$|interval-transcendentals|quantum-evolution-interval/.test(p)),inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`,JSON.stringify({inputs,bytes:code.outputFiles[0].contents.length,heavyNumericalEnginesExcluded:true},null,2)+'\n');
});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({exactChecks,directedChecks,rejectedMutations},null,2)+'\n'));
