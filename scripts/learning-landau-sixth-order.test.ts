import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {landauSixthOrderModel as model,type LandauSixthOrderParameters} from '../src/lib/learning-landau-sixth-order';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const cases=JSON.parse(readFileSync('scripts/fixtures/landau-sixth-order-oracle.json','utf8')).cases as {id:string;parameters:LandauSixthOrderParameters;confirmationPrecisionDigits:number;reference:{rootCount:number;realRootMultiplicity:number;roots:Record<string,string|number|boolean|null>[];globalMinimumIndices:number[];minimumFreeEnergy:string;exactCoexistence:boolean;barriers:{minimumIndex:number;maximumIndex:number;barrier:string}[]}}[];
const saved=new Map<string,ReturnType<typeof model>>();
function evaluate(p:LandauSixthOrderParameters){const key=JSON.stringify(p);if(!saved.has(key))saved.set(key,model(p));return saved.get(key)!;}
const Z = BigInt(0), O = BigInt(1);
interface Q { n: bigint; d: bigint }
const add = (a: Q, b: Q): Q => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
function decimal(s: string): Q {
  const match = /^([+-]?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s); assert.ok(match, s);
  const n = BigInt(`${match[1] === '-' ? '-' : ''}${match[2]}${match[3] ?? ''}`), exponent = Number(match[4] ?? 0) - (match[3]?.length ?? 0);
  return exponent >= 0 ? { n: n * BigInt(10) ** BigInt(exponent), d: O } : { n, d: BigInt(10) ** BigInt(-exponent) };
}
function decode(q: PartitionRational): Q {
  let n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient);
  const shift = q.numerator.exponent - q.denominator.exponent;
  if (shift >= 0) n <<= BigInt(shift); else d <<= BigInt(-shift);
  assert.ok(d > Z); return { n, d };
}
function check(q: PartitionObservation, reference: string, precision: number, label: string) {
  if (reference.endsWith('infinity')) {
    assert.equal(q.resolution, 'mathematical-infinity', label); assert.equal(q.sign, reference[0] === '-' ? -1 : 1); return;
  }
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling', label);
  const truth = decimal(reference), truthSign = truth.n < Z ? -1 : truth.n > Z ? 1 : 0;
  assert.equal(q.sign, truthSign, label);
  if (!truth.n) { assert.equal(q.value, 0, label); assert.equal(q.resolution, 'structural-zero', label); }
  else {
    const expected = Number(reference);
    if (expected === 0 || !Number.isFinite(expected)) assert.equal(q.value, null, label);
    else {
      assert.notEqual(q.value, null, label);
      assert.ok(Math.abs(q.value! - expected) <= Math.max(2 * Number.MIN_VALUE, 1e-13 * Math.abs(expected)), `${label}: ${q.value} vs ${expected}`);
    }
  }
  assert.ok(q.bounds, label);
  const lower = decode(q.bounds.lower), upper = decode(q.bounds.upper);
  assert.ok(cmp(lower, upper) <= Z, label);
  // The oracle is independently recomputed at 400/900 and +80 Decimal digits.
  // Allow 150 relative decimal digits plus a much smaller absolute rounding budget.
  const error = add({ n: truth.n < Z ? -truth.n : truth.n, d: truth.d * BigInt(10) ** BigInt(150) }, { n: O, d: BigInt(10) ** BigInt(precision - 110) });
  assert.ok(cmp(lower, add(truth, error)) <= Z && cmp(upper, add(truth, { n: -error.n, d: error.d })) >= Z, `${label}: independent reference must lie in directed bounds within its stated precision`);
  if (q.resolution === 'interval-resolved') {
    assert.ok(lower.n > Z || upper.n < Z, label);
    const width = upper.n * lower.d - lower.n * upper.d;
    const near = lower.n > Z ? lower.n * upper.d : -upper.n * lower.d;
    assert.ok((width << BigInt(48)) <= near, `${label}: relative width`);
  }
}

const observations=['magnetization','curvature','constrainedFreeEnergy','leadingStationaryDerivative','branchFieldDerivative','freeEnergyExcess'] as const;
for(const c of cases)test(`${c.id}: independent Decimal roots, multiplicities, stability and global selection`,()=>{
 const r=evaluate(c.parameters),ref=c.reference;assert.equal(r.fullyResolved,true,`${c.id}: ${r.unresolvedReason}`);assert.equal(r.rootsComplete,true);assert.equal(r.globalSelectionResolved,true);
 assert.equal(r.rootCount,ref.rootCount);assert.equal(r.realRootMultiplicity,ref.realRootMultiplicity);assert.deepEqual(r.globalMinimumIndices,ref.globalMinimumIndices);assert.equal(r.exactCoexistence,ref.exactCoexistence);
 for(const [i,x] of r.roots.entries()){
  for(const key of observations){const truth=ref.roots[i][key];if(truth===null)assert.equal(x[key],null);else{assert.ok(x[key]);check(x[key]!,truth as string,c.confirmationPrecisionDigits,`${c.id}.roots[${i}].${key}`);}}
  for(const key of ['stationaryMultiplicity','localKind','globalMinimum','role'] as const)assert.equal(x[key],ref.roots[i][key],`${c.id}.${key}[${i}]`);
  assert.ok(decode(x.residualEnclosure.lower).n<=Z&&decode(x.residualEnclosure.upper).n>=Z);
  if(i)assert.ok(cmp(decode(r.roots[i-1].magnetization.bounds!.upper),decode(x.magnetization.bounds!.lower))<Z);
 }
 assert.ok(r.minimumFreeEnergy);check(r.minimumFreeEnergy,ref.minimumFreeEnergy,c.confirmationPrecisionDigits,`${c.id}.minimum`);
 assert.equal(r.barriers.length,ref.barriers.length);r.barriers.forEach((x,i)=>{assert.equal(x.minimumIndex,ref.barriers[i].minimumIndex);assert.equal(x.maximumIndex,ref.barriers[i].maximumIndex);check(x.barrier,ref.barriers[i].barrier,c.confirmationPrecisionDigits,`${c.id}.barrier[${i}]`);assert.equal(x.barrier.sign,1);});
 function finite(v:unknown){if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(finite);else if(v&&typeof v==='object')Object.values(v).forEach(finite);}finite(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
const p={quadraticA:3/16,quarticB:-1,sexticC:1,fieldH:0};
const near=(x:number|null,y:number,tol=2e-12)=>{assert.notEqual(x,null);assert.ok(Math.abs(x!-y)<=tol*Math.max(1,Math.abs(y)),`${x} vs ${y}`);};
test('exact coexistence retains three global minima and all four adjacent barriers',()=>{
 const r=evaluate(p);assert.equal(r.rootCount,5);assert.deepEqual(r.globalMinimumIndices,[0,2,4]);assert.equal(r.barriers.length,4);r.barriers.forEach(x=>near(x.barrier.value,1/96));
 r.globalMinimumIndices.forEach(i=>{assert.equal(r.roots[i].constrainedFreeEnergy.resolution,'structural-zero');assert.equal(r.roots[i].freeEnergyExcess!.resolution,'structural-zero');});
 near(r.roots[0].magnetization.value,-Math.sqrt(3)/2);near(r.roots[4].magnetization.value,Math.sqrt(3)/2);
});
test('spinodal roots are genuine double stationary inflections, distinct from triple coexistence',()=>{
 const r=evaluate({...p,quadraticA:1/4});assert.equal(r.rootCount,3);assert.equal(r.realRootMultiplicity,5);assert.deepEqual(r.globalMinimumIndices,[1]);assert.equal(r.barriers.length,0);
 for(const i of [0,2]){const x=r.roots[i];assert.equal(x.stationaryMultiplicity,2);assert.equal(x.role,'stationary-inflection');assert.equal(x.curvature.resolution,'structural-zero');assert.equal(x.branchFieldDerivative,null);assert.equal(x.derivativeStatus,'fold-no-single-valued-derivative');}
});
test('quartic, sextic and flat-maximum degeneracies have different multiplicities and limiting derivative signs',()=>{
 for(const [b,mult,local,sign] of [[1,3,'minimum',1],[0,5,'minimum',1],[-1,3,'maximum',-1]] as const){const r=evaluate({...p,quadraticA:0,quarticB:b}),x=r.roots.find(x=>x.magnetization.sign===0)!;assert.equal(x.stationaryMultiplicity,mult);assert.equal(x.localKind,local);assert.equal(x.branchFieldDerivative!.resolution,'mathematical-infinity');assert.equal(x.branchFieldDerivative!.sign,sign);}
});
test('an exact tilted quartic minimum remains a triple root at a nonzero order parameter',()=>{
 const r=evaluate({quadraticA:15/16,quarticB:-2.5,sexticC:3,fieldH:.25});assert.equal(r.rootCount,1);assert.equal(r.roots[0].stationaryMultiplicity,3);assert.equal(r.roots[0].magnetization.value,.5);assert.equal(r.roots[0].curvature.value,0);assert.equal(r.roots[0].constrainedFreeEnergy.value,-5/128);
});
test('adjacent actual coefficients distinguish coexistence and folded-root counts without tolerance merging',()=>{
 const byId=(id:string)=>evaluate(cases.find(x=>x.id===id)!.parameters);
 assert.deepEqual(byId('below-coexistence').globalMinimumIndices,[0,4]);assert.deepEqual(byId('above-coexistence').globalMinimumIndices,[2]);
 assert.deepEqual(['below-ordered-spinodal','ordered-spinodal','above-ordered-spinodal'].map(id=>byId(id).rootCount),[5,3,1]);
 assert.deepEqual(['below-tilted-fold','exact-tilted-fold','above-tilted-fold'].map(id=>byId(id).rootCount),[1,2,3]);
});
test('a tilted subnormal coexistence field selects one minimum and preserves positive energy differences',()=>{
 const r=evaluate({...p,fieldH:Number.MIN_VALUE});assert.equal(r.fullyResolved,true);assert.deepEqual(r.globalMinimumIndices,[4]);assert.equal(r.precisionBits,128);
 for(const i of [0,2]){assert.equal(r.roots[i].freeEnergyExcess!.sign,1);assert.equal(r.roots[i].freeEnergyExcess!.resolution,'interval-resolved');}
 const capped=model({...p,fieldH:Number.MIN_VALUE},{maximumPrecision:128});assert.equal(capped.fullyResolved,true);assert.deepEqual(capped.globalMinimumIndices,[4]);
});
test('the scalar order parameter can exceed the bounded spin-magnetization interval',()=>{
 const r=evaluate(cases.find(x=>x.id==='outside-spin-domain')!.parameters);assert.ok(r.roots.some(x=>Math.abs(x.magnetization.value!)>1));
});
test('invalid coefficients, confinement and precision requests fail; preparation is an owned frozen snapshot',()=>{
 for(const patch of [{quadraticA:NaN},{quadraticA:4.1},{quarticB:Infinity},{sexticC:0},{sexticC:.249},{sexticC:4.1},{fieldH:-4.1}])assert.throws(()=>model({...p,...patch}),RangeError);
 for(const options of [{maximumPrecision:129},{minimumPrecision:64},{minimumPrecision:256,maximumPrecision:128}])assert.throws(()=>model(p,options),RangeError);
 const owned={...p},r=model(owned);owned.fieldH=1;assert.equal(r.parameters.fieldH,0);assert.ok(Object.isFrozen(r.parameters));
});

test('exact rational tilted coexistence can certify multiple positive global minima',()=>{
 const r=model({quadraticA:235/256,quarticB:-23/8,sexticC:3,fieldH:3/16});assert.equal(r.fullyResolved,true);assert.equal(r.globalMinimumIndices.length,2);assert.deepEqual(r.globalMinimumIndices.map(i=>r.roots[i].magnetization.value),[.25,.75]);assert.equal(r.minimumFreeEnergy!.value,-171/8192);
});
test('an unresolved nonzero-field algebraic tie retains both positive candidates at its precision cap',()=>{
 // Phi+9/1024=(1/2)(m^2-m+1/8)^2(m^2+2m+9/8)>=0.
 const r=model({quadraticA:59/64,quarticB:-13/4,sexticC:3,fieldH:1/8},{maximumPrecision:128});assert.equal(r.rootsComplete,true);assert.equal(r.fullyResolved,false);assert.equal(r.globalSelectionResolved,false);assert.deepEqual(r.globalMinimumIndices,[]);assert.equal(r.globalMinimumCandidates.length,2);assert.ok(r.globalMinimumCandidates.every(i=>r.roots[i].magnetization.sign===1));
});
