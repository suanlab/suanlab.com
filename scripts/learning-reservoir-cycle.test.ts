import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/reservoir-cycle-independent.json';
import {reservoirCycle as evaluate,type ReservoirCycleParameters}from '../src/lib/learning-reservoir-cycle';
import type {BinaryObservation}from '../src/lib/learning-binary-products';
const base:ReservoirCycleParameters={hotTemperature:600,coldTemperature:300,heatHot:1000,relativeEntropyGeneration:0};
interface Reference{value:string;sign:number;logMagnitude:string|null}
function near(a:number,b:number){assert(Number.isFinite(a));assert(Math.abs(a-b)<=Math.max(2*Number.MIN_VALUE,4e-13*Math.abs(b)),`${a} != ${b}`);}
function observation(a:BinaryObservation|null,b:Reference){assert.notEqual(a,null);assert.equal(a!.sign,b.sign);const target=Number(b.value);if(b.sign===0){assert.equal(a!.value,0);assert.equal(a!.logMagnitude,null);}else{assert.notEqual(a!.value,0);if(target===0||!Number.isFinite(target))assert.equal(a!.value,null);else near(a!.value!,target);assert(Math.abs(a!.logMagnitude!-Number(b.logMagnitude))<3e-11);}}
for(const[family,cases]of Object.entries({ordinary:refs.ordinary,limits:refs.limits}))for(const[i,row]of cases.entries())test(`${family} exact-rational reservoir budget ${i+1}`,()=>{
 const q=evaluate(row.parameters);assert.equal(q.regime,row.regime);assert.equal(q.secondLawAdmissible,row.secondLawAdmissible);
 for(const[k,expected]of Object.entries(row.expected))observation((q as unknown as Record<string,BinaryObservation|null>)[k],expected);
});
test('reference coverage includes24 ordinary and12 limit cases',()=>{assert.equal(refs.ordinary.length,24);assert.equal(refs.limits.length,12);});
test('worked four-port examples distinguish reversible, irreversible, driven and counterfactual budgets',()=>{
 const expected=[[0,500,0,'heat-engine'],[.5,250,5/6,'heat-engine'],[2,-500,10/3,'work-driven-dissipation'],[-.5,750,-5/6,'counterfactual']]as const;
 for(const[sigma,work,entropy,regime]of expected){const q=evaluate({...base,relativeEntropyGeneration:sigma});near(q.workByDevice.value!,work);near(q.entropyGeneration.value!,entropy);assert.equal(q.regime,regime);}
});
test('energy conservation alone does not make a complete heat-to-work port physically admissible',()=>{
 const q=evaluate({...base,relativeEntropyGeneration:-1});assert.equal(q.heatColdIntoDevice.value,0);assert.equal(q.workByDevice.value,1000);assert.equal(q.workFraction!.value,1);assert.equal(q.carnotTemperatureBound.value,.5);assert.equal(q.secondLawAdmissible,false);assert.equal(q.engineEfficiency,null);
});
test('the exact zero-work boundary and neighboring inputs have correct operation classifications',()=>{
 const q=evaluate({...base,relativeEntropyGeneration:1});assert.equal(q.workByDevice.value,0);assert.equal(q.regime,'heat-transfer');assert.equal(q.engineEfficiency,null);
 assert.equal(evaluate({...base,relativeEntropyGeneration:1-Number.EPSILON/2}).regime,'heat-engine');assert.equal(evaluate({...base,relativeEntropyGeneration:1+Number.EPSILON}).regime,'work-driven-dissipation');
});
test('tiny generation survives rounded reservoir sums and retains the work deficit',()=>{
 const q=evaluate({...base,relativeEntropyGeneration:Number.MIN_VALUE});assert.equal(q.hotReservoirEntropyChange.value!+q.coldReservoirEntropyChange.value!,0);assert.equal(q.entropyGeneration.sign,1);assert(q.entropyGeneration.value!>0);assert.equal(q.workDeficitFromReversible.sign,1);
});
test('equal-temperature ports with tiny dissipation retain negative work despite rounded heat balance',()=>{
 const q=evaluate({...base,hotTemperature:300,relativeEntropyGeneration:Number.MIN_VALUE});assert.equal(q.heatHotIntoDevice.value!+q.heatColdIntoDevice.value!,0);assert.equal(q.workByDevice.sign,-1);assert(q.workByDevice.value!<0);assert.equal(q.regime,'work-driven-dissipation');assert.equal(q.secondLawAdmissible,true);
});
test('underflowed negative entropy cannot become a physically admissible zero',()=>{
 const q=evaluate({...base,heatHot:Number.MIN_VALUE,relativeEntropyGeneration:-Number.MIN_VALUE});assert.equal(q.entropyGeneration.value,null);assert.equal(q.entropyGeneration.sign,-1);assert(Number.isFinite(q.entropyGeneration.logMagnitude));assert.equal(q.secondLawAdmissible,false);assert.equal(q.regime,'counterfactual');
});
test('zero hot input is no transfer even for a negative relative coefficient; ratios remain undefined',()=>{
 const q=evaluate({...base,heatHot:0,relativeEntropyGeneration:-.5});assert.equal(q.regime,'no-transfer');assert.equal(q.secondLawAdmissible,true);assert.equal(q.workFraction,null);assert.equal(q.engineEfficiency,null);assert.equal(q.workByDevice.value,0);assert.equal(q.entropyGeneration.value,0);
});
test('heat scales extensive ports without changing formal ratios or operating class',()=>{
 const a=evaluate({...base,relativeEntropyGeneration:.5}),b=evaluate({...base,heatHot:2000,relativeEntropyGeneration:.5});for(const k of ['workByDevice','heatColdIntoDevice','entropyGeneration','workDeficitFromReversible']as const)near(b[k].value!,2*a[k].value!);assert.deepEqual(a.engineEfficiency,b.engineEfficiency);assert.equal(a.regime,b.regime);
});
test('admissible heat engines stay below the Carnot bound and other work-input ports are not called refrigerators',()=>{
 for(const sigma of [0,.1,.5,1,2,4]){const q=evaluate({...base,relativeEntropyGeneration:sigma});assert(q.secondLawAdmissible);if(q.regime==='heat-engine')assert(q.engineEfficiency!.value!<=q.carnotTemperatureBound.value!);assert(!String(q.regime).includes('refrigerator'));}
});
test('outputs serialize without BigInt or nonfinite numbers',()=>{
 const walk=(v:unknown):void=>{if(typeof v==='number')assert(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};for(const sigma of [-1,-Number.MIN_VALUE,0,Number.MIN_VALUE,1,4]){const q=evaluate({...base,relativeEntropyGeneration:sigma});walk(q);assert.doesNotThrow(()=>JSON.stringify(q));}
});
test('invalid reservoir roles and bounds are rejected rather than silently reordered',()=>{
 for(const k of Object.keys(base)as(keyof ReservoirCycleParameters)[])for(const v of [NaN,Infinity,-Infinity])assert.throws(()=>evaluate({...base,[k]:v}),RangeError);
 assert.throws(()=>evaluate({...base,hotTemperature:250,coldTemperature:300}),RangeError);assert.throws(()=>evaluate({...base,heatHot:-1}),RangeError);assert.throws(()=>evaluate({...base,relativeEntropyGeneration:-1.1}),RangeError);assert.throws(()=>evaluate({...base,relativeEntropyGeneration:5}),RangeError);
});
