import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { evaluateManyBodyRequest as evaluate } from '../src/lib/learning-many-body-evaluator';
import { readSchmidtSuccessForInput as read } from '../src/lib/learning-many-body-schmidt-validation';
import { snapshotManyBodyInput as snapshot, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import { exact as exactObservation } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
import { f as rational, zero, one, int, add, sub, mul, div, le as rationalLe, c, type Range } from '../src/lib/learning-tensor-wire-math';
import { wireRange, wireAdd, wireSub, wireMul, wireDiv, wireSquare, wireSqrt, wireSpectralProjector } from '../src/lib/learning-many-body-schmidt-interval-wire';
ensureTestArtifactDirectory('.runtime/learning/many-body-schmidt-response', { recursive: true });

const runtime = '.runtime/learning/many-body-schmidt-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/schmidt-reference.json', 'utf8')) as { cases: { label: string; input: Record<string, number>; expected: unknown }[] };
function preparation(label: string) { const r = refs.cases.find(r => r.label === label); assert(r); const input = snapshot({ kind: 'schmidt', parameters: r.input }); const packet = evaluate({ version: 1, requestId: 7, input }); assert(packet.ok && packet.kind === 'schmidt'); return { input, packet }; }
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
  const lo = endpoint(q.bounds!.lower), hi = endpoint(q.bounds!.upper); assert(le(lo, hi), path);
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
for(const ref of refs.cases)test(`Schmidt response matches independent preparation: ${ref.label}`,()=>{
  const input=snapshot({kind:'schmidt',parameters:ref.input}),packet=evaluate({version:1,requestId:7,input});assert(packet.ok&&packet.kind==='schmidt');
  const owned=read(structuredClone(packet),input,7);assert(owned,ref.label);assert.notEqual(owned,packet);assert(Object.isFrozen(owned.result.cuts[0].spectrum[0].projectorOnSmallerPhysicalSide[0]));compare(owned.result,ref.expected);assert.equal(read(packet,input,8),null);
});

test('every numerical observation rejects a wrong valid rational',()=>{
  const{input,packet}=preparation('general complex 2 sites case0'),paths:(string|number)[][]=[];
  function walk(v:unknown,path:(string|number)[]){if(isMasterObservation(v)){paths.push(path);return;}if(v&&typeof v==='object')for(const[k,x]of Object.entries(v))walk(x,[...path,Array.isArray(v)?Number(k):k]);}
  walk(packet.result,['result']);assert(paths.length>150);
  for(const path of paths)rejects(input,packet,p=>{const parent=path.slice(0,-1).reduce((v:any,k)=>v[k],p),key=path.at(-1)!;parent[key]=exactObservation(BigInt(parent[key].value===7?8:7),BigInt(1));});
});

test('owned fields, array lengths, contracts and metadata remain exact',()=>{
  const{input,packet}=preparation('general complex 2 sites case0');
  for(const path of [[],['input'],['result'],['result','parameters'],['result','cuts',0],['result','cuts',0,'spectrum',0],['result','cuts',0,'truncation'],['result','cuts',0,'truncation','chosen'],['result','cuts',0,'directedResiduals']]){
    const obj=path.reduce((v:any,k)=>v[k],packet);assert(obj);
    rejects(input,packet,p=>{path.reduce((v:any,k)=>v[k],p).extra=0;});for(const key of Object.keys(obj))rejects(input,packet,p=>{delete path.reduce((v:any,k)=>v[k],p)[key];});
  }
  for(const edit of[(p:any)=>{p.result.approximationContract+=' ';},(p:any)=>{p.result.cuts[0].spectrum.reverse();},(p:any)=>{p.result.cuts[0].spectrum.extra=0;},(p:any)=>{delete p.result.cuts[0].spectrum[0];},(p:any)=>{p.inputKey+=' ';},(p:any)=>{p.kind='mps';},(p:any)=>{p.result.cuts[0].entropyNats.value=1234567;},(p:any)=>{p.result.cuts[0].entropyNats.resolution='unresolved-at-precision-ceiling';}])rejects(input,packet,edit);
  const symbolic:any=structuredClone(packet);symbolic.result[Symbol('extra')]=0;assert.equal(read(symbolic,input,7),null);assert.equal(read(packet,input,0),null);assert.equal(read(packet,input,NaN),null);
});

test('exact positive degeneracies do not invent a chosen optimum or eigenvectors',()=>{
  for(const label of ['GHZ4 degenerate rank1 optimum','fourfold equal middle weights budget2','two positive degenerate groups budget1']){
    const{input,packet}=preparation(label),i=packet.result.cuts.findIndex(v=>v.truncation.boundaryDegenerate);assert(i>=0);
    assert.equal(packet.result.cuts[i].truncation.chosen,null);
    rejects(input,packet,p=>{p.result.cuts[i].truncation.chosen={};});
    rejects(input,packet,p=>{p.result.cuts[i].truncation.boundaryDegenerate=false;p.result.cuts[i].truncation.optimalUnnormalizedStateUnique=true;});
    rejects(input,packet,p=>{p.result.cuts[i].truncation.noArbitraryDegenerateEigenvectorsChosen=false;});
    rejects(input,packet,p=>{p.result.cuts[i].spectrum[0].multiplicity=1;});
  }
});

test('rank zero budget has a unique zero unnormalized state but no normalized optimum',()=>{
  const{input,packet}=preparation('rank0 zero truncation no normalized optimum');
  assert.equal(packet.result.cuts[0].truncation.normalizedStateDefinedForEveryOptimum,false);assert(packet.result.cuts[0].truncation.chosen);
  rejects(input,packet,p=>{p.result.cuts[0].truncation.chosen.normalizedAmplitudes=[];});
  rejects(input,packet,p=>{p.result.cuts[0].truncation.optimalNormalizedFidelitySquared=exactObservation(BigInt(0),BigInt(1));});
  rejects(input,packet,p=>{p.result.cuts[0].truncation.chosen=null;p.result.cuts[0].truncation.chosenStateAvailable=false;});
});

test('discarded error, normalized distance and fidelity are different input-derived quantities',()=>{
  const{input,packet}=preparation('diagonal weights9-4-1 rank budget1');
  const i=packet.result.cuts.findIndex(v=>v.smallerDimension===4);assert(i>=0);
  rejects(input,packet,p=>{p.result.cuts[i].truncation.optimalNormalizedDistanceSquared=structuredClone(p.result.cuts[i].truncation.discardedWeight);});
  rejects(input,packet,p=>{p.result.cuts[i].truncation.optimalNormalizedFidelitySquared=structuredClone(p.result.cuts[i].truncation.discardedWeight);});
  rejects(input,packet,p=>{p.result.cuts[i].truncation.retainedWeight=exactObservation(BigInt(1),BigInt(1));p.result.cuts[i].truncation.discardedWeight=exactObservation(BigInt(0),BigInt(1));});
});

test('physical right density and projector act with transpose on raw coefficient matrix',()=>{
  const{input,packet}=preparation('right physical spectral truncation complex phase');
  const i=packet.result.cuts.findIndex(v=>v.smallerSide==='right');assert(i>=0);
  rejects(input,packet,p=>{p.result.cuts[i].spectrum[0].projectorOnSmallerPhysicalSide.reverse();});
  rejects(input,packet,p=>{p.result.cuts[i].rhoRight[0][1]=structuredClone(p.result.cuts[i].rhoRight[1][0]);});
  rejects(input,packet,p=>{p.result.cuts[i].truncation.chosen.truncatedRawCoefficientMatrix=structuredClone(p.result.cuts[i].coefficientMatrix);});
});

test('small positive eigenvalues and close distinct weights cannot be rounded away or merged',()=>{
  const tiny=preparation('near rank deficient exact nonzero determinant');
  rejects(tiny.input,tiny.packet,p=>{p.result.cuts[0].schmidtRank=1;});
  rejects(tiny.input,tiny.packet,p=>{p.result.cuts[0].spectrum.at(-1).weight=exactObservation(BigInt(0),BigInt(1));});
  const close=preparation('close positive Schmidt weights not merged');
  rejects(close.input,close.packet,p=>{p.result.cuts[0].spectrum[1].weight=structuredClone(p.result.cuts[0].spectrum[0].weight);});
});

test('tightly bounded residuals and projectors cannot be replaced by broad enclosures',()=>{
  const{input,packet}=preparation('general complex 2 sites case0');
  rejects(input,packet,p=>{const q=p.result.cuts[0].directedResiduals.probabilityMass;q.bounds={lower:exactObservation(BigInt(-1),BigInt(1)).bounds!.lower,upper:exactObservation(BigInt(1),BigInt(1)).bounds!.upper};q.value=null;q.sign=null;q.logMagnitude=null;q.logResolution='unresolved';q.resolution='unresolved-at-precision-ceiling';});
  rejects(input,packet,p=>{p.result.cuts[0].spectrum[0].projectorOnSmallerPhysicalSide[0][0].real=exactObservation(BigInt(1),BigInt(1));});
});

test('another state cannot be laundered through copied controls and selected-cut metadata',()=>{
  const first=preparation('general complex 2 sites case0'),second=preparation('general complex 2 sites case1');
  rejects(first.input,first.packet,p=>{p.result=structuredClone(second.packet.result);p.result.parameters={...first.input.parameters};});
  rejects(first.input,first.packet,p=>{p.result.selectedCutIndex=1;});
});

test('descriptor ownership rejects accessors and hostile reflection without invoking getters',()=>{
  const{input,packet}=preparation('general complex 2 sites case0');let calls=0;
  for(const path of[['result','cuts',0,'spectrum',0],['result','cuts',0,'truncation','retainedWeight'],['result','cuts',0,'entropyNats']]){
    const changed:any=structuredClone(packet),parent=path.slice(0,-1).reduce((v:any,k)=>v[k],changed);Object.defineProperty(parent,path.at(-1)!,{get(){calls++;return null;}});assert.equal(read(changed,input,7),null);
  }
  assert.equal(calls,0);assert(read(new Proxy(packet,{get(){calls++;throw Error('get');}}),input,7));assert.equal(calls,0);
  for(const value of[Object.create(packet),Object.assign(Object.create(null),packet),new Proxy({},{ownKeys(){throw Error('reflection');}})])assert.equal(read(value,input,7),null);
});

test('Schmidt validator bundle excludes production calculators, root, matrix and logarithm engines',async()=>{
  const code=await build({entryPoints:['src/lib/learning-many-body-schmidt-validation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
  assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|ritz-interval|interval-transcendentals|many-body-evaluator)\.ts$/.test(p)),inputs.join('\n'));
  writeFileSync(`${runtime}/validation-dependency-graph.json`,JSON.stringify({inputs,bytes:code.outputFiles[0].contents.length,heavyNumericalEnginesExcluded:true},null,2)+'\n');
});
const includes=(box:Range,target:ReturnType<typeof rational>)=>rationalLe(box.lower,target)&&rationalLe(target,box.upper);
let intervalArithmeticChecks=0;
test('dyadic interval arithmetic encloses exact signed rational results without gcd rounding assumptions',()=>{
  for(let i=-18;i<=18;i++)for(const j of[-11,-1,1,7]){
    const a=rational(BigInt(17*i+3),BigInt(29)),b=rational(BigInt(j),BigInt(13)),x=wireRange(a),y=wireRange(b);
    for(const [box,value]of [[wireAdd(x,y),add(a,b)],[wireSub(x,y),sub(a,b)],[wireMul(x,y),mul(a,b)],[wireDiv(x,y),div(a,b)]] as const){assert(includes(box,value));intervalArithmeticChecks++;}
  }
});

test('interval endpoint products and divisions include all corner values in either divisor sign',()=>{
  for(const sign of[-1,1]){
    const x=wireRange(rational(BigInt(-3),BigInt(7)),rational(BigInt(11),BigInt(13))),y=sign>0?wireRange(rational(BigInt(2),BigInt(9)),rational(BigInt(5),BigInt(8))):wireRange(rational(BigInt(-5),BigInt(8)),rational(BigInt(-2),BigInt(9)));
    for(const a of[x.lower,x.upper])for(const b of[y.lower,y.upper]){
      assert(includes(wireMul(x,y),mul(a,b)));assert(includes(wireDiv(x,y),div(a,b)));intervalArithmeticChecks+=2;
    }
  }
  assert.throws(()=>wireDiv(wireRange(one),wireRange(int(-1),one)));assert.throws(()=>wireRange(one,zero));
});

test('Newton square-root enclosures satisfy exact squared inequalities from tiny to large rationals',()=>{
  for(const x of[zero,one,int(2),int(7),rational(BigInt(1),BigInt(3)),rational(BigInt(1),BigInt(10)**BigInt(200)),rational(BigInt(10)**BigInt(24)),rational(BigInt(999999),BigInt(1000000))]){
    const root=wireSqrt(wireRange(x));assert(rationalLe(mul(root.lower,root.lower),x)&&rationalLe(x,mul(root.upper,root.upper)));intervalArithmeticChecks++;
  }
  const square=wireSquare(wireRange(int(-2),int(3)));assert(includes(square,zero)&&includes(square,int(9)));
  assert.throws(()=>wireSqrt(wireRange(int(-2),int(-1))));
});

test('Horner projectors recover a complex Hermitian eigenspace with its physical orientation',()=>{
  const half=rational(BigInt(1),BigInt(2)),quarter=rational(BigInt(1),BigInt(4)),negativeQuarter=rational(BigInt(-1),BigInt(4));
  const density=[[c(half),c(zero,negativeQuarter)],[c(zero,quarter),c(half)]],roots=[wireRange(rational(BigInt(3),BigInt(4))),wireRange(quarter)],P=wireSpectralProjector(density,roots,0);
  const expected=[[c(half),c(zero,rational(BigInt(-1),BigInt(2)))],[c(zero,half),c(half)]];
  P.forEach((row,i)=>row.forEach((v,j)=>{assert(includes(v.re,expected[i][j].re)&&includes(v.im,expected[i][j].im));intervalArithmeticChecks+=2;}));
  assert.throws(()=>wireSpectralProjector(density,[roots[0],roots[0]],0));
});

after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({exactChecks,directedChecks,rejectedMutations,intervalArithmeticChecks},null,2)+'\n'));
