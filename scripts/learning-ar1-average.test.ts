import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ar1FiniteAverageModel as model, type AR1AverageParameters } from '../src/lib/learning-ar1-average';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
interface Fixture { id:string;parameters:AR1AverageParameters;confirmationPrecisionDigits:number;reference:Record<string,string|null|{lag?:number;sampleCount?:number;reference:string}[]> }
const cases=(JSON.parse(readFileSync('scripts/fixtures/ar1-average-oracle.json','utf8')) as {cases:Fixture[]}).cases;
const saved=new Map<string,ReturnType<typeof model>>();
function evaluate(p:AR1AverageParameters){const key=JSON.stringify(p),old=saved.get(key);if(old)return old;const r=model(p);saved.set(key,r);return r;}
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
  // The oracle is independently recomputed at exact Fraction projections at 400 and +80 Decimal digits.
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
for(const c of cases)test(`${c.id}: exact actual-binary Fraction covariance references and directed square root`,()=>{
 const r=evaluate(c.parameters);assert.equal(r.fullyResolved,true);
 for(const [key,v] of Object.entries(c.reference)){
  if(v===null){assert.equal((r as unknown as Record<string,unknown>)[key],null);continue;}
  if(Array.isArray(v))for(const row of v){
   const q=key==='autocovariances'?r.autocovariances[row.lag!].covariance:r.varianceBySampleCount[row.sampleCount!-1].meanVariance;
   check(q,row.reference,c.confirmationPrecisionDigits,`${c.id}.${key}.${row.lag??row.sampleCount}`);
  }
  else check((r as unknown as Record<string,PartitionObservation>)[key],v,c.confirmationPrecisionDigits,`${c.id}.${key}`);
 }
 const visit=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(visit);else if(v&&typeof v==='object')Object.values(v).forEach(visit);};visit(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
const near=(a:number|null,b:number,tolerance=2e-12)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
test('one observation retains variance one independently of dependence and endpoint mode',()=>{
 for(const rho of [-1,-.9,0,.9,1]){const r=evaluate({sampleCount:1,correlationRho:rho});assert.equal(r.meanVariance.value,1);assert.equal(r.standardError.value,1);assert.equal(r.varianceEquivalentSampleCount.value,1);assert.equal(r.totalSumVariance.value,1);}
});
test('independent reference and every finite prefix have the expected exact rational normalization',()=>{
 const r=evaluate({sampleCount:512,correlationRho:0});assert.equal(r.meanVariance.value,1/512);assert.equal(r.varianceEquivalentSampleCount.value,512);assert.equal(r.varianceInflation.value,1);assert.equal(r.totalSumVariance.value,512);
 for(const row of r.varianceBySampleCount)near(row.meanVariance.value,1/row.sampleCount);
 assert.equal(r.autocovariances[0].covariance.value,1);for(const row of r.autocovariances.slice(1))assert.equal(row.covariance.resolution,'structural-zero');
});
test('positive and negative half correlation have different finite corrections from tau/n',()=>{
 const a=evaluate({sampleCount:4,correlationRho:.5}),b=evaluate({sampleCount:4,correlationRho:-.5});near(a.meanVariance.value,33/64);near(b.meanVariance.value,7/64);near(a.asymptoticIntegratedCorrelationTime!.value,3);near(b.asymptoticIntegratedCorrelationTime!.value,1/3);near(b.varianceEquivalentSampleCount.value,64/7);near(b.finiteToAsymptoticVarianceRatio!.value,21/16);
 const short=evaluate({sampleCount:2,correlationRho:.5});near(short.meanVariance.value,.75);near(short.asymptoticMeanVariance!.value,1.5);
});
test('stationary innovation normalization differs from fixing innovation variance to one',()=>{
 for(const rho of [-.8,-.5,0,.5,.8]){const r=evaluate({sampleCount:3,correlationRho:rho});near(r.innovationVariance.value,1-rho*rho);assert.equal(r.autocovariances[0].covariance.value,1);}
});
test('degenerate endpoints retain stationarity but do not report an ordinary mixing IAT',()=>{
 for(const rho of [-1,1])for(const n of [1,2,3,511,512]){
  const r=evaluate({sampleCount:n,correlationRho:rho});assert.equal(r.asymptoticStatus,'not-applicable-nonmixing-endpoint');assert.equal(r.asymptoticIntegratedCorrelationTime,null);assert.equal(r.asymptoticMeanVariance,null);assert.equal(r.finiteToAsymptoticVarianceRatio,null);assert.equal(r.innovationVariance.resolution,'structural-zero');
  if(rho===1){assert.equal(r.processMode,'constant-degenerate');assert.equal(r.meanVariance.value,1);assert.equal(r.varianceEquivalentSampleCount.value,1);}
  else if(n%2===0){assert.equal(r.deterministicAverage,true);assert.equal(r.meanVariance.resolution,'structural-zero');assert.equal(r.standardError.resolution,'structural-zero');assert.equal(r.varianceEquivalentSampleCount.resolution,'mathematical-infinity');}
  else {near(r.meanVariance.value,1/(n*n));near(r.standardError.value,1/n);assert.equal(r.varianceEquivalentSampleCount.value,n*n);}
 }
});
test('finite covariance quadratic forms agree with all curve prefixes for ordinary interior parameters',()=>{
 for(const rho of [-.9,-.3,.3,.9]){
  const r=evaluate({sampleCount:16,correlationRho:rho});for(let n=1;n<=16;n++){
   let total=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++)total+=rho**Math.abs(i-j);
   near(r.varianceBySampleCount[n-1].meanVariance.value,total/(n*n));
  }
 }
});
test('standard-error bounds square outward around the variance enclosure',()=>{
 const square=(v:Q):Q=>({n:v.n*v.n,d:v.d*v.d});
 for(const rho of [-1,-.9999999999999999,-.5,0,.5,.9999999999999999,1])for(const n of [1,2,7,512]){
  const r=evaluate({sampleCount:n,correlationRho:rho});assert.ok(r.standardError.bounds&&r.meanVariance.bounds);
  assert.ok(cmp(square(decode(r.standardError.bounds.lower)),decode(r.meanVariance.bounds.lower))<=Z);
  assert.ok(cmp(square(decode(r.standardError.bounds.upper)),decode(r.meanVariance.bounds.upper))>=Z);
 }
});
test('autocovariance powers retain exact dyadic bounds and signs far below linear range',()=>{
 for(const rho of [Number.MIN_VALUE,-Number.MIN_VALUE]){
  const r=evaluate({sampleCount:512,correlationRho:rho});for(const lag of [0,1,2,511]){
   const q=r.autocovariances[lag].covariance;assert.ok(q.bounds);assert.equal(cmp(decode(q.bounds.lower),decode(q.bounds.upper)),Z);assert.equal(q.sign,rho<0&&lag%2?-1:1);
  }
  const tiny=r.autocovariances[511].covariance;assert.equal(tiny.value,null);assert.equal(tiny.resolution,'interval-resolved');near(tiny.logMagnitude,511*Math.log(Number.MIN_VALUE),1e-12);
 }
});
test('nearly alternating even averages remain positive and distinct from the exact endpoint',()=>{
 const rho=-.9999999999999999,r=evaluate({sampleCount:512,correlationRho:rho});assert.equal(r.deterministicAverage,false);assert.equal(r.meanVariance.sign,1);assert.ok(r.meanVariance.value!>0&&r.meanVariance.value!<1e-18);assert.equal(r.asymptoticStatus,'mixing-interior');assert.ok(r.varianceEquivalentSampleCount.value!>1e18);
});
test('adding an observation need not reduce finite variance for a negatively correlated sequence',()=>{
 const r=evaluate({sampleCount:4,correlationRho:-1});assert.equal(r.varianceBySampleCount[1].meanVariance.value,0);near(r.varianceBySampleCount[2].meanVariance.value,1/9);assert.equal(r.varianceBySampleCount[3].meanVariance.value,0);
});
test('invalid counts, nonstationary coefficients and precision budgets are rejected',()=>{
 for(const n of [0,513,-1,1.5,NaN,Infinity])assert.throws(()=>model({sampleCount:n,correlationRho:0}),RangeError);
 for(const rho of [1.01,-1.01,NaN,Infinity,-Infinity])assert.throws(()=>model({sampleCount:2,correlationRho:rho}),RangeError);
 assert.throws(()=>model({sampleCount:2,correlationRho:0},{minimumPrecision:64}),RangeError);assert.throws(()=>model({sampleCount:2,correlationRho:0},{minimumPrecision:256,maximumPrecision:128}),RangeError);assert.throws(()=>model(null as unknown as AR1AverageParameters),RangeError);
});
