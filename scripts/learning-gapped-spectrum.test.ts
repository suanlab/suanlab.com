import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/gapped-spectrum-independent.json';
import {gappedSpectrum as evaluate,type GappedSpectrumParameters,type SpectrumQuantity}from '../src/lib/learning-gapped-spectrum';
const base:GappedSpectrumParameters={copies:1,groundDegeneracy:1,upperDegeneracy:1,gapTemperature:300,temperature:300};
interface Reference{value:string|null;sign:number;logMagnitude:string|null;logNegativeLogMagnitude:string|null;logMagnitudeStatus:string;logNegativeLogMagnitudeStatus:string}
function near(a:number,b:number,tolerance=5e-13){assert(Number.isFinite(a));assert(Math.abs(a-b)<=Math.max(2*Number.MIN_VALUE,tolerance*Math.abs(b)),`${a} != ${b}`);}
function logNear(a:number,b:number){assert(Number.isFinite(a));assert(Math.abs(a-b)<=Math.max(3e-11,5e-13*Math.abs(b)),`${a} != ${b}`);}
function observation(a:SpectrumQuantity|null,b:Reference|null){
 if(b===null){assert.equal(a,null);return;}assert.notEqual(a,null);assert.equal(a!.sign,b.sign);assert.equal(a!.logMagnitudeStatus,b.logMagnitudeStatus);assert.equal(a!.logNegativeLogMagnitudeStatus,b.logNegativeLogMagnitudeStatus);
 if(b.value===null)assert.equal(a!.value,null);else near(a!.value!,Number(b.value));
 if(b.sign!==0)assert.notEqual(a!.value,0);
 if(b.logMagnitude===null)assert.equal(a!.logMagnitude,null);else{assert.notEqual(a!.logMagnitude,null);logNear(a!.logMagnitude!,Number(b.logMagnitude));}
 if(b.logNegativeLogMagnitude===null)assert.equal(a!.logNegativeLogMagnitude,null);else{assert.notEqual(a!.logNegativeLogMagnitude,null);logNear(a!.logNegativeLogMagnitude!,Number(b.logNegativeLogMagnitude));}
}
for(const[family,cases]of Object.entries({ordinary:refs.ordinary,limits:refs.limits,extreme:refs.extreme}))for(const[i,row]of cases.entries())test(`${family} independent finite-spectrum reference ${i+1}`,()=>{
 const q=evaluate(row.parameters);for(const[k,expected]of Object.entries(row.expected))observation((q as unknown as Record<string,SpectrumQuantity|null>)[k],expected);
 near(q.partitionFunctionPerCopy,row.partitionFunctionPerCopy);assert.equal(q.effectiveGroundDegeneracy,row.effectiveGroundDegeneracy);assert.equal(q.groundStateCount,row.groundStateCount);assert.equal(q.totalStateCount,row.totalStateCount);
});
test('reference methods distinguish direct state sums from extreme asymptotic log bounds',()=>{
 assert.equal(refs.ordinary.length,24);assert.equal(refs.limits.length,16);assert.equal(refs.extreme.length,5);assert.equal(refs.ordinary.filter(x=>x.temperatureDerivativesChecked).length,4);
 assert(refs.extreme.every(x=>x.method.includes('logarithmic bounds')));
});
test('worked example has dimensionless statistics and restored physical units',()=>{
 const q=evaluate(base);near(q.upperGroupProbability.value!,.2689414213699951);near(q.meanEnergyPerCopyInKB.value!,80.68242641099853);near(q.entropyPerCopyInKB.value!,.582203108888218);near(q.heatCapacityPerCopyInKB.value!,.19661193324148185);assert(q.meanEnergy.value!>0);assert(q.entropy.value!>0);assert(q.heatCapacity.value!>0);
});
test('a nonzero gap does not disappear when its SI energy is below the number range',()=>{
 const q=evaluate({...base,gapTemperature:Number.MIN_VALUE,temperature:Number.MIN_VALUE});assert.equal(q.gapEnergy.value,null);assert.equal(q.gapEnergy.sign,1);assert.equal(q.spectrum,'positive-gap');assert.equal(q.gapToTemperatureRatio!.value,1);near(q.heatCapacity.value!,evaluate(base).heatCapacity.value!);assert.equal(q.energyVariance.value,null);assert.equal(q.energyVariance.sign,1);
});
test('underflowed ratio retains positive heat-capacity scale instead of becoming the exact zero-gap model',()=>{
 const small=evaluate({...base,gapTemperature:Number.MIN_VALUE,temperature:800}),zero=evaluate({...base,gapTemperature:0,temperature:800});assert.equal(small.gapToTemperatureRatio!.value,null);assert.equal(small.gapToTemperatureRatio!.sign,1);assert.equal(small.heatCapacity.value,null);assert.equal(small.heatCapacity.sign,1);assert.equal(zero.heatCapacity.value,0);assert.equal(zero.heatCapacity.sign,0);assert.equal(small.effectiveGroundDegeneracy,1);assert.equal(zero.effectiveGroundDegeneracy,2);
});
test('normalizing before projecting rescues heat capacity and mean energy from an unrepresentable probability',()=>{
 const q=evaluate({...base,gapTemperature:750,temperature:1});assert.equal(q.upperGroupProbability.value,null);assert.equal(q.upperGroupProbability.sign,1);assert(q.heatCapacityPerCopyInKB.value!>0);assert(q.meanEnergyPerCopyInKB.value!>0);assert.equal(q.heatCapacity.value,null);assert.equal(q.heatCapacity.sign,1);
});
test('first and second logarithm ranges distinguish finite-temperature tails from the zero-temperature limit',()=>{
 const q=evaluate({...base,temperature:Number.MIN_VALUE});assert.equal(q.mode,'positive-temperature');assert.equal(q.upperGroupProbability.value,null);assert.equal(q.upperGroupProbability.logMagnitude,null);assert.equal(q.upperGroupProbability.logMagnitudeStatus,'outside-number-range');assert(q.upperGroupProbability.logNegativeLogMagnitude!>700);assert.equal(q.upperGroupProbability.sign,1);
 const limit=evaluate({...base,temperature:0});assert.equal(limit.mode,'zero-temperature-limit');assert.equal(limit.upperGroupProbability.value,0);assert.equal(limit.upperGroupProbability.sign,0);assert.equal(limit.gapToTemperatureRatio,null);assert.equal(limit.entropyTemperatureDerivative,null);
});
test('a ground probability rounded to one retains logarithmic information about its nonzero deficit',()=>{
 const q=evaluate({...base,gapTemperature:1000,temperature:1});assert.equal(q.groundGroupProbability.value,1);assert.equal(q.groundGroupProbability.logMagnitude,null);assert.equal(q.groundGroupProbability.logMagnitudeStatus,'outside-number-range');near(q.groundGroupProbability.logNegativeLogMagnitude!,-1000);assert.equal(q.upperGroupProbability.sign,1);
 const extreme=evaluate({...base,temperature:Number.MIN_VALUE});assert.equal(extreme.groundGroupProbability.logNegativeLogMagnitude,null);assert.equal(extreme.groundGroupProbability.logNegativeLogMagnitudeStatus,'outside-number-range');
});
test('residual entropy and excitation entropy remain separate when their sum rounds to the residual value',()=>{
 const q=evaluate({...base,groundDegeneracy:2,upperDegeneracy:3,gapTemperature:1000,temperature:1});near(q.entropy.value!,q.residualEntropy.value!);assert.equal(q.excitationEntropy.value,null);assert.equal(q.excitationEntropy.sign,1);assert(Number.isFinite(q.excitationEntropy.logMagnitude));
});
test('zero-gap then low-temperature and positive-gap then low-temperature have distinct state counts',()=>{
 const gap=evaluate({...base,temperature:0}),zero=evaluate({...base,gapTemperature:0,temperature:0});assert.equal(gap.entropy.value,0);assert(zero.entropy.value!>0);assert.equal(gap.groundStateCount,'1');assert.equal(zero.groundStateCount,'2');assert(zero.upperGroupProbability.value!>0);assert.equal(zero.meanEnergy.value,0);assert.equal(zero.excitationEntropy.value,0);
});
test('independent copies scale extensive moments and preserve per-copy probabilities and normalized quantities',()=>{
 const a=evaluate({...base,groundDegeneracy:2}),b=evaluate({...base,groundDegeneracy:2,copies:64});for(const k of ['meanEnergy','entropy','heatCapacity','energyVariance','residualEntropy']as const)near(b[k].value!,64*a[k].value!);assert.deepEqual(a.upperGroupProbability,b.upperGroupProbability);assert.deepEqual(a.entropyPerCopyInKB,b.entropyPerCopyInKB);assert.equal(b.groundStateCount,(BigInt(2)**BigInt(64)).toString());
});
test('state counts are exact decimal strings even beyond integer resolution',()=>{
 const q=evaluate({...base,copies:64,groundDegeneracy:16,upperDegeneracy:16,gapTemperature:0});assert.equal(q.groundStateCount,(BigInt(32)**BigInt(64)).toString());assert.equal(q.totalStateCount,q.groundStateCount);assert.equal(typeof q.groundStateCount,'string');
});
test('positive-temperature derivative and fluctuation relations agree at ordinary scales',()=>{
 for(const T of [30,100,300,800]){const q=evaluate({...base,temperature:T});near(q.entropyTemperatureDerivative!.value!,q.heatCapacity.value!/T);const kb=refs.boltzmannConstantBinary;near(q.heatCapacity.value!,q.energyVariance.value!/(kb*T*T));}
});
test('entropy increases toward the finite-state bound without requiring a zero probability at positive temperature',()=>{
 for(const g0 of [1,2,16]){let prior=0;for(const T of [0,1e-300,.01,1,10,100,300,800]){const q=evaluate({...base,groundDegeneracy:g0,temperature:T});if(q.entropyPerCopyInKB.value!==null){assert(q.entropyPerCopyInKB.value>=prior-1e-12);assert(q.entropyPerCopyInKB.value<=Math.log(g0+1)+1e-12);prior=q.entropyPerCopyInKB.value;}if(T>0)assert.equal(q.upperGroupProbability.sign,1);}}
});
test('all exposed numeric values remain finite and results serialize at supported extremes',()=>{
 const walk=(v:unknown):void=>{if(typeof v==='number')assert(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};for(const gap of [0,Number.MIN_VALUE,1,1000])for(const T of [0,Number.MIN_VALUE,1e-306,1,800]){const q=evaluate({...base,gapTemperature:gap,temperature:T});walk(q);assert.doesNotThrow(()=>JSON.stringify(q));}
});
test('invalid counts and physical ranges are rejected',()=>{
 for(const k of Object.keys(base)as(keyof GappedSpectrumParameters)[])for(const v of [NaN,Infinity,-Infinity,-1])assert.throws(()=>evaluate({...base,[k]:v}),RangeError);
 for(const k of ['copies','groundDegeneracy','upperDegeneracy']as const)for(const v of [0,1.5,100])assert.throws(()=>evaluate({...base,[k]:v}),RangeError);
 assert.throws(()=>evaluate({...base,gapTemperature:1001}),RangeError);assert.throws(()=>evaluate({...base,temperature:801}),RangeError);
});

test('half-minimum energy projection is not rounded upward by a logarithmic round trip',()=>{
 const q=evaluate({...base,gapTemperature:Number.MIN_VALUE,temperature:800});
 assert.equal(q.upperGroupProbability.value,.5);assert.equal(q.meanEnergyPerCopyInKB.value,null);assert.equal(q.meanEnergyPerCopyInKB.sign,1);
});
