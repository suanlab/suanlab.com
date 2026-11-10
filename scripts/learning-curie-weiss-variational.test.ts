import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { curieWeissVariationalModel as model, type CurieWeissVariationalParameters } from '../src/lib/learning-curie-weiss-variational';
import { finiteCurieWeissModel } from '../src/lib/learning-curie-weiss';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
const cases = JSON.parse(readFileSync('scripts/fixtures/curie-weiss-variational-oracle.json','utf8')).cases as {
  id:string; parameters:CurieWeissVariationalParameters; confirmationPrecisionDigits:number;
  reference:{rootCount:number; roots:Record<string,string|boolean|number>[]; spinodal:Record<string,string>|null;
  globalMinimumIndices:number[]; thermodynamicLimitPressure:string; barriers:{minimumIndex:number;barrier:string}[]; minimumFreeEnergyGap:string|null};
}[];
const saved=new Map<string,ReturnType<typeof model>>();
function evaluate(p:CurieWeissVariationalParameters){const key=JSON.stringify(p);if(!saved.has(key))saved.set(key,model(p));return saved.get(key)!;}
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

const rootObservations = ['magnetization','curvature','branchFieldDerivative','constrainedFreeEnergy','freeEnergyDifferenceToZero','entropyNats'] as const;
for (const c of cases) test(`${c.id}: independent direct-log roots, curvatures and free energies`,()=>{
  const r=evaluate(c.parameters), ref=c.reference;
  assert.equal(r.fullyResolved,true,`${c.id}: ${r.unresolvedReason}`);
  assert.equal(r.rootCount,ref.rootCount); assert.equal(r.roots.length,ref.roots.length);
  assert.deepEqual(r.globalMinimumIndices,ref.globalMinimumIndices);
  for(const [i,x] of r.roots.entries()){
    for(const key of rootObservations)check(x[key],ref.roots[i][key] as string,c.confirmationPrecisionDigits,`${c.id}.roots[${i}].${key}`);
    for(const key of ['role','globalMinimum','stationaryMultiplicity'] as const)assert.equal(x[key],ref.roots[i][key]);
    assert.ok(decode(x.residualEnclosure.lower).n<=Z);assert.ok(decode(x.residualEnclosure.upper).n>=Z);
    if(i)assert.ok(cmp(decode(r.roots[i-1].magnetization.bounds!.upper),decode(x.magnetization.bounds!.lower))<Z,'distinct ordered roots');
  }
  if(ref.spinodal){assert.ok(r.spinodal);for(const key of ['turningMagnetization','fieldMagnitude'] as const)check(r.spinodal[key],ref.spinodal[key],c.confirmationPrecisionDigits,`${c.id}.spinodal.${key}`);}
  else assert.equal(r.spinodal,null);
  assert.ok(r.thermodynamicLimitPressure);check(r.thermodynamicLimitPressure,ref.thermodynamicLimitPressure,c.confirmationPrecisionDigits,`${c.id}.pressure`);
  assert.equal(r.barriers.length,ref.barriers.length);
  r.barriers.forEach((x,i)=>{assert.equal(x.minimumIndex,ref.barriers[i].minimumIndex);check(x.barrier,ref.barriers[i].barrier,c.confirmationPrecisionDigits,`${c.id}.barrier[${i}]`);assert.equal(x.barrier.sign,1);});
  if(ref.minimumFreeEnergyGap!==null){assert.ok(r.minimumFreeEnergyGap);check(r.minimumFreeEnergyGap,ref.minimumFreeEnergyGap,c.confirmationPrecisionDigits,`${c.id}.gap`);}else assert.equal(r.minimumFreeEnergyGap,null);
  function finite(x:unknown){if(typeof x==='number')assert.ok(Number.isFinite(x));else if(Array.isArray(x))x.forEach(finite);else if(x&&typeof x==='object')Object.values(x).forEach(finite);}
  finite(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
const near=(a:number|null,b:number,tol=2e-12)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
test('independent spins give tanh magnetization, sech squared response and log(2 cosh b) pressure',()=>{
  for(const b of [-4,-.2,0,.2,4]){const r=evaluate({couplingK:0,fieldB:b});assert.equal(r.rootCount,1);near(r.roots[0].magnetization.value,Math.tanh(b));near(r.roots[0].branchFieldDerivative.value,1/Math.cosh(b)**2);near(r.thermodynamicLimitPressure!.value,Math.log(2*Math.cosh(b)));}
});
test('the exact critical root has zero quadratic curvature, cubic stationary multiplicity and infinite response',()=>{
  const r=evaluate({couplingK:1,fieldB:0}),x=r.roots[0];assert.equal(r.rootCount,1);assert.equal(x.role,'critical-minimum');assert.equal(x.stationaryMultiplicity,3);
  for(const key of ['magnetization','curvature','freeEnergyDifferenceToZero'] as const)assert.equal(x[key].resolution,'structural-zero');
  assert.equal(x.branchFieldDerivative.resolution,'mathematical-infinity');assert.equal(x.branchFieldDerivative.sign,1);
});
test('a subnormal critical field retains cubic-root scaling and an unrepresentable negative free-energy change',()=>{
  const x=evaluate({couplingK:1,fieldB:Number.MIN_VALUE}).roots[0];
  near(x.magnetization.value!/Math.cbrt(3*Number.MIN_VALUE),1);assert.equal(x.freeEnergyDifferenceToZero.sign,-1);assert.equal(x.freeEnergyDifferenceToZero.value,null);
  assert.equal(x.freeEnergyDifferenceToZero.resolution,'interval-resolved');assert.ok(Number.isFinite(x.freeEnergyDifferenceToZero.logMagnitude));
});
test('field sign selects a unique global minimum despite identical displayed minimum energies',()=>{
  const r=evaluate({couplingK:2,fieldB:Number.MIN_VALUE});assert.equal(r.roots[0].constrainedFreeEnergy.value,r.roots[2].constrainedFreeEnergy.value);
  assert.deepEqual(r.globalMinimumIndices,[2]);assert.equal(r.roots[0].role,'metastable-minimum');assert.equal(r.minimumFreeEnergyGap!.sign,1);
  const tiny=evaluate({couplingK:1.0000000000000002,fieldB:Number.MIN_VALUE});assert.equal(tiny.minimumFreeEnergyGap!.value,null);assert.equal(tiny.minimumFreeEnergyGap!.sign,1);assert.equal(tiny.minimumFreeEnergyGap!.resolution,'interval-resolved');
});
test('neighboring binary64 spinodal fields resolve separate roots without a distance cutoff',()=>{
  const rows=['below-rounded-spinodal','rounded-spinodal','above-rounded-spinodal'].map(id=>evaluate(cases.find(c=>c.id===id)!.parameters));
  assert.deepEqual(rows.map(r=>r.rootCount),[3,1,1]);
  assert.ok(rows[0].precisionBits>128);assert.equal(rows[0].roots[0].freeEnergyDifferenceToZero.value,rows[0].roots[1].freeEnergyDifferenceToZero.value);
  assert.ok(rows[0].roots[0].magnetization.value!<rows[0].roots[1].magnetization.value!);
  const capped=model(cases.find(c=>c.id==='below-rounded-spinodal')!.parameters,{maximumPrecision:128});assert.equal(capped.fullyResolved,false);assert.deepEqual(capped.roots,[]);assert.ok(capped.unresolvedReason);
});
test('field reversal reflects root ordering, swaps minima and preserves pressure and curvature',()=>{
 const p=evaluate({couplingK:2,fieldB:.2}),n=evaluate({couplingK:2,fieldB:-.2});near(p.thermodynamicLimitPressure!.value,n.thermodynamicLimitPressure!.value!);
 p.roots.forEach((x,i)=>{const y=n.roots[2-i];near(x.magnetization.value,-y.magnetization.value!);near(x.curvature.value,y.curvature.value!);near(x.constrainedFreeEnergy.value,y.constrainedFreeEnergy.value!);assert.equal(x.role,y.role);});assert.deepEqual(p.globalMinimumIndices,[2]);assert.deepEqual(n.globalMinimumIndices,[0]);
});
test('a finite symmetric mixture has increasing fluctuation response while each pure branch has finite local response',()=>{
 const r=evaluate({couplingK:2,fieldB:0});assert.deepEqual(r.globalMinimumIndices,[0,2]);assert.equal(r.roots[1].branchFieldDerivative.sign,-1);
 const pure=r.roots[2].branchFieldDerivative.value!;assert.ok(pure>0&&pure<1);
 const finite=[16,32,64].map(spinCount=>finiteCurieWeissModel({spinCount,couplingK:2,fieldB:0}));finite.forEach(x=>assert.equal(x.meanMagnetization.value,0));
 assert.ok(finite[0].fieldSusceptibility.value!<finite[1].fieldSusceptibility.value!);assert.ok(finite[1].fieldSusceptibility.value!<finite[2].fieldSusceptibility.value!);
 const pressure=finite.map(x=>x.logPartition.value!/x.parameters.spinCount);assert.ok(Math.abs(pressure[2]-r.thermodynamicLimitPressure!.value!)<Math.abs(pressure[0]-r.thermodynamicLimitPressure!.value!));
});
test('invalid parameters and precision ranges fail explicitly and preparation is frozen',()=>{
 const p={couplingK:2,fieldB:0};for(const patch of [{couplingK:-Number.MIN_VALUE},{couplingK:8.1},{couplingK:NaN},{fieldB:Infinity},{fieldB:4.1}])assert.throws(()=>model({...p,...patch}),RangeError);
 for(const options of [{maximumPrecision:129},{minimumPrecision:64},{minimumPrecision:256,maximumPrecision:128}])assert.throws(()=>model(p,options),RangeError);
 const r=model(p);assert.ok(Object.isFrozen(r.parameters));p.fieldB=1;assert.equal(r.parameters.fieldB,0);
});
