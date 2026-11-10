import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { periodicIsingRingModel as model, type PeriodicIsingRingParameters } from '../src/lib/learning-periodic-ising-ring';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
interface Fixture { id: string; parameters: PeriodicIsingRingParameters; confirmationPrecisionDigits: number; transferChecked: boolean; enumerationChecked: boolean; reference: Record<string, string | string[]> }
const cases = (JSON.parse(readFileSync('scripts/fixtures/periodic-ising-ring-oracle.json', 'utf8')) as { cases: Fixture[] }).cases;
const saved = new Map<string, ReturnType<typeof model>>();
function evaluate(p: PeriodicIsingRingParameters) { const key = JSON.stringify(p), old = saved.get(key); if (old) return old; const r = model(p); saved.set(key, r); return r; }
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
  // The oracle is independently recomputed at 400/900/1200 and +80 Decimal digits.
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
for (const c of cases) test(`${c.id}: independent actual-binary ring reference, signs and directed bounds`, () => {
  const r=evaluate(c.parameters);assert.equal(r.fullyResolved,true);assert.equal(r.bondCount,c.parameters.spinCount);
  for (const [key,v] of Object.entries(c.reference)) {
    if (key==='correlations'||key==='logAbsoluteCorrelations') (v as string[]).forEach((truth,i)=>check(r.correlations[i][key==='correlations'?'correlation':'logAbsoluteCorrelation'],truth,c.confirmationPrecisionDigits,`${c.id}.${key}[${i}]`));
    else check((r as unknown as Record<string,PartitionObservation>)[key],v as string,c.confirmationPrecisionDigits,`${c.id}.${key}`);
  }
  const visit=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(visit);else if(v&&typeof v==='object')Object.values(v).forEach(visit);};
  visit(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
const near=(a:number|null,b:number,tolerance=2e-12)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
test('independence, self correlation and assigned zero-range limit are exact structural cases',()=>{
  for(const N of [3,4,127,128]){
    const r=evaluate({spinCount:N,couplingK:0});assert.equal(r.microstateCount,(O<<BigInt(N)).toString());assert.equal(r.meanSpin.resolution,'structural-zero');assert.equal(r.correlations[0].correlation.value,1);
    for(const c of r.correlations.slice(1)){assert.equal(c.correlation.resolution,'structural-zero');assert.equal(c.logAbsoluteCorrelation.resolution,'mathematical-infinity');assert.equal(c.logAbsoluteCorrelation.sign,-1);}
    assert.equal(r.commonFieldResponsePerSpin.value,1);near(r.magnetizationMeanVariance.value,1/N);assert.equal(r.totalSpinVariance.value,N);near(r.logPartition.value,N*Math.log(2));
    assert.equal(r.infiniteChainCorrelationLength.resolution,'structural-zero');assert.equal(r.correlationLengthStatus,'independent-zero-range-limit');
  }
});
test('small named configurations independently reproduce correlations and magnetization response',()=>{
  for(const N of [3,4,5,6,7,8,9])for(const K of [-1.3,-.2,.4,1.1]){
    const states=Array.from({length:2**N},(_,i)=>Array.from({length:N},(_,j)=>(i>>j)&1?1:-1));
    const weights=states.map(s=>Math.exp(K*s.reduce((a,x,i)=>a+x*s[(i+1)%N],0))),W=weights.reduce((a,b)=>a+b,0),r=evaluate({spinCount:N,couplingK:K});
    for(let d=0;d<N;d++)near(r.correlations[d].correlation.value,states.reduce((a,s,i)=>a+weights[i]*s[0]*s[d],0)/W);
    near(r.commonFieldResponsePerSpin.value,states.reduce((a,s,i)=>a+weights[i]*s.reduce((a,b)=>a+b,0)**2,0)/W/N);
  }
});
test('reflection, translation and antiferromagnetic parity use nearest ring distance',()=>{
  for(const N of [3,4,7,8,127,128]){
    const r=evaluate({spinCount:N,couplingK:-.7});
    for(let offset=0;offset<N;offset++){
      const c=r.correlations[offset];assert.equal(c.distance,Math.min(offset,N-offset));assert.deepEqual(c.correlation,r.correlations[(N-offset)%N].correlation);assert.equal(c.correlation.sign,c.distance%2?-1:1);
    }
    assert.equal(r.oddAntiferromagnetic,N%2===1);near(r.commonFieldResponsePerSpin.value,r.correlations.reduce((a,c)=>a+c.correlation.value!,0));
    near(r.totalSpinVariance.value,N*r.commonFieldResponsePerSpin.value!);near(r.magnetizationMeanVariance.value,r.commonFieldResponsePerSpin.value!/N);
  }
});
test('covariance circulants have nonnegative directional curvatures, including negative entries',()=>{
  for(const N of [3,4,5,8,11])for(const K of [-2,-.6,.6,2]){
    const r=evaluate({spinCount:N,couplingK:K});
    for(let k=0;k<N;k++){
      const eigenvalue=r.correlations.reduce((a,c)=>a+c.correlation.value!*Math.cos(2*Math.PI*k*c.offset/N),0);assert.ok(eigenvalue>0,`${N},${K},${k}: ${eigenvalue}`);
    }
  }
});
test('odd frustrated and even antiferromagnetic cold preparations have distinct uniform responses',()=>{
  const odd=evaluate({spinCount:127,couplingK:-8}),even=evaluate({spinCount:128,couplingK:-8});
  near(odd.commonFieldResponsePerSpin.value,1/127,1e-10);assert.ok(even.commonFieldResponsePerSpin.value!>0&&even.commonFieldResponsePerSpin.value!<1e-10);
  near(odd.correlations[1].correlation.value,-125/127,1e-10);
  assert.equal(odd.infiniteChainCorrelationLength.sign,1);assert.ok(odd.infiniteChainCorrelationLength.value!>1e6);assert.ok(odd.infiniteChainCommonFieldResponse.value!<1e-6);
});
test('fixed-distance infinite-chain envelope and response differ from finite-ring corrections',()=>{
  const K=.4,t=Math.tanh(K),small=evaluate({spinCount:3,couplingK:K}),large=evaluate({spinCount:128,couplingK:K});
  near(large.correlations[3].correlation.value,t**3);near(large.infiniteChainCorrelationLength.value,-1/Math.log(t));near(large.infiniteChainCommonFieldResponse.value,Math.exp(2*K));
  assert.ok(Math.abs(small.correlations[1].correlation.value!-t)>.05);near(large.commonFieldResponsePerSpin.value,large.infiniteChainCommonFieldResponse.value!);
});
test('long subnormal tails retain positive enclosures and directed logarithms beyond ordinary underflow',()=>{
  const r=evaluate({spinCount:128,couplingK:Number.MIN_VALUE}),c=r.correlations[64];
  assert.equal(c.correlation.value,null);assert.equal(c.correlation.sign,1);assert.equal(c.correlation.resolution,'interval-resolved');assert.ok(c.correlation.bounds);assert.ok(BigInt(c.correlation.bounds.lower.numerator.coefficient)>Z);
  near(c.logAbsoluteCorrelation.value,64*Math.log(Number.MIN_VALUE)+Math.log(2),1e-12);assert.ok(r.precisionBits>=2048);
  const capped=model({spinCount:128,couplingK:Number.MIN_VALUE},{maximumPrecision:128});assert.equal(capped.fullyResolved,false);assert.equal(capped.correlations[64].correlation.resolution,'unresolved-at-precision-ceiling');assert.equal(capped.correlations[64].correlation.value,null);assert.equal(capped.correlations[64].correlation.sign,1);
  assert.equal(capped.infiniteChainCorrelationLength.resolution,'unresolved-at-precision-ceiling');
});
test('invalid sizes, nonfinite couplings and precision ranges are rejected',()=>{
  for(const N of [0,1,2,129,3.5,NaN,Infinity])assert.throws(()=>model({spinCount:N,couplingK:0}),RangeError);
  for(const K of [8.001,-8.001,NaN,Infinity,-Infinity])assert.throws(()=>model({spinCount:3,couplingK:K}),RangeError);
  assert.throws(()=>model({spinCount:3,couplingK:0},{maximumPrecision:64}),RangeError);assert.throws(()=>model({spinCount:3,couplingK:0},{minimumPrecision:256,maximumPrecision:128}),RangeError);
  assert.throws(()=>model(null as unknown as PeriodicIsingRingParameters),RangeError);
});
