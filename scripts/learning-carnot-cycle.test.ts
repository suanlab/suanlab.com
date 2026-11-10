import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/carnot-cycle-independent.json';
import { carnotCycle as cycle, carnotCycleState as state, type CarnotParameters, type CarnotDirection } from '../src/lib/learning-carnot-cycle';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
const base: CarnotParameters = { moles:1, hotTemperature:600, coldTemperature:300, initialVolume:.01, expansionRatio:2 };
interface Reference { value:string; sign:number; logMagnitude:string|null }
function near(actual:number,expected:number,tolerance=4e-13){assert(Number.isFinite(actual));assert(Math.abs(actual-expected)<=Math.max(2*Number.MIN_VALUE,tolerance*Math.abs(expected)),`${actual} != ${expected}`);}
function observation(actual:BinaryObservation,expected:Reference){
 assert.equal(actual.sign,expected.sign);const target=Number(expected.value);
 if(expected.sign===0){assert.equal(actual.value,0);assert.equal(actual.logMagnitude,null);}
 else{assert.notEqual(actual.value,0);if(target===0||!Number.isFinite(target))assert.equal(actual.value,null);else{assert.notEqual(actual.value,null);near(actual.value!,target);}assert(Math.abs(actual.logMagnitude!-Number(expected.logMagnitude))<3e-11);}
}
for(const[family,cases]of Object.entries({ordinary:refs.ordinary,limits:refs.limits}))for(const[i,row]of cases.entries())test(`${family} independent Carnot reference ${i+1}`,()=>{
 const s=state(row.parameters,row.direction as CarnotDirection,row.progress);
 for(const[key,expected]of Object.entries(row.expected)){const value=(s as unknown as Record<string,number|BinaryObservation>)[key];if(typeof value==='number')near(value,Number(expected.value));else observation(value,expected);}
 assert.equal(s.entropyProduction,0);
 if('physicalFullLegs'in row){const c=cycle(row.parameters,row.direction as CarnotDirection);for(const[i,leg]of row.physicalFullLegs!.entries())for(const[key,expected]of Object.entries(leg)){const field=key==='work'?'work':key==='heat'?'heat':key==='energyChange'?'energyChange':'entropyChange';observation(c.legs[i][field],expected);}}
});
test('reference scope includes30 physical-volume quadratures and16 high-precision limits',()=>{
 assert.equal(refs.ordinary.length,30);assert(refs.ordinary.every(r=>r.physicalQuadrature));assert.equal(refs.limits.length,16);
 assert.deepEqual(new Set(refs.ordinary.map(r=>r.direction)),new Set(['forward','reverse']));
});
test('worked engine and reversed refrigerator have correct signed ports and performance ratios',()=>{
 const a=cycle(base,'forward'),b=cycle(base,'reverse');assert.equal(a.regime,'heat-engine');assert.equal(b.regime,'refrigerator');
 near(a.complete.heatHotIntoGas.value!,3457.8877929863875);near(a.complete.heatColdIntoGas.value!,-1728.9438964931937);near(a.complete.workByGas.value!,1728.9438964931937);
 assert.equal(a.engineEfficiency!.value,.5);assert.equal(a.refrigeratorCOP,null);assert.equal(b.engineEfficiency,null);assert.equal(b.refrigeratorCOP!.value,1);assert.equal(b.heatingCOP!.value,2);
 for(const k of ['heatHotIntoGas','heatColdIntoGas','netHeat','workByGas']as const)near(a.complete[k].value!,-b.complete[k].value!);
});
test('four physical corners close exactly, including binary temperatures with a nonrepresentable rounded difference',()=>{
 for(const p of [base,{...base,hotTemperature:790.718,coldTemperature:250+2**-45}])for(const d of ['forward','reverse']as const){
  const s=state(p,d,1);assert.equal(s.temperature,p.hotTemperature);assert.equal(s.volume,p.initialVolume);
  for(const k of ['temperatureChange','volumeChange','energyChange','entropyChangeGas']as const){assert.equal(s[k].value,0);assert.equal(s[k].sign,0);}
 }
});
test('whole-cycle reversal retraces ordinary states but accumulated reverse work can initially be positive',()=>{
 const full=cycle(base,'forward').complete;
 for(const s of [0,.125,.25,.375,.5,.625,.75,.875,1]){const a=state(base,'forward',1-s),b=state(base,'reverse',s);near(a.temperature,b.temperature);near(a.volume,b.volume);near(a.pressure,b.pressure);
  assert(Math.abs(b.workByGas.value!-(a.workByGas.value!-full.workByGas.value!))<1e-9);
 }
 assert(state(base,'reverse',.125).workByGas.value!>0);assert(cycle(base,'reverse').complete.workByGas.value!<0);
});
test('reverse work crossing is exact at the rational balance and changes sign on either side',()=>{
 const s=.875,low=s-2**-53,high=s+2**-53;
 assert.equal(state(base,'reverse',s).workByGas.value,0);assert.equal(state(base,'reverse',low).workByGas.sign,1);assert.equal(state(base,'reverse',high).workByGas.sign,-1);
});
test('tiny reverse progress is not lost by subtracting it from one',()=>{
 const tiny=Number.MIN_VALUE;assert.equal(1-tiny,1);const s=state(base,'reverse',tiny);
 assert.equal(s.temperature,600);assert.equal(s.temperatureChange.sign,-1);assert(s.workByGas.value!>0);
 assert.equal(s.volumeChange.value,null);assert.equal(s.volumeChange.sign,1);assert(Number.isFinite(s.volumeChange.logMagnitude));
});
test('combined tiny progress and tiny expansion retain nonzero signs outside linear range',()=>{
 const s=state({...base,expansionRatio:1+Number.EPSILON},'forward',Number.MIN_VALUE);
 assert.equal(s.workByGas.value,null);assert.equal(s.workByGas.sign,1);assert(s.workByGas.logMagnitude!<Math.log(Number.MIN_VALUE));
 assert.equal(s.heatHotIntoGas.sign,1);assert.equal(s.entropyChangeGas.sign,1);
});
test('zero temperature difference and zero expansion are degenerate ratios, not infinite performance',()=>{
 for(const p of [{...base,hotTemperature:300},{...base,expansionRatio:1}])for(const d of ['forward','reverse']as const){const c=cycle(p,d);assert.equal(c.regime,'degenerate');assert.equal(c.engineEfficiency,null);assert.equal(c.refrigeratorCOP,null);assert.equal(c.heatingCOP,null);assert.equal(c.complete.workByGas.value,0);}
 const equal=cycle({...base,hotTemperature:300},'forward');assert.equal(equal.workPerHotHeat!.value,0);assert.equal(equal.engineEfficiency,null);
 const c=cycle({...base,expansionRatio:1},'forward');assert.equal(c.workPerHotHeat,null);assert.equal(c.carnotTemperatureBound.value,.5);assert(c.legs[1].work.value!>0);assert(c.legs[3].work.value!<0);
});
test('reversible entropy ledger includes both reservoirs throughout the path',()=>{
 for(const d of ['forward','reverse']as const)for(let i=0;i<=64;i++){const s=state(base,d,i/64);assert(Math.abs(s.entropyChangeGas.value!+s.hotReservoirEntropyChange.value!+s.coldReservoirEntropyChange.value!)<1e-12);assert.equal(s.entropyProduction,0);}
});
test('amount scales energies and pressures while volume scales only geometry and inverse pressure',()=>{
 const a=state(base,'forward',.375),b=state({...base,moles:2},'forward',.375),c=state({...base,initialVolume:.02},'forward',.375);
 near(b.temperature,a.temperature);near(b.volume,a.volume);near(b.pressure,2*a.pressure);near(c.volume,2*a.volume);near(c.pressure,a.pressure/2);
 for(const k of ['energyChange','heatHotIntoGas','netHeat','workByGas','entropyChangeGas']as const){near(b[k].value!,2*a[k].value!);near(c[k].value!,a[k].value!);}
});
test('signed p-V polygon work converges to the independently integrated net work',()=>{
 for(const d of ['forward','reverse']as const){let area=0,last=state(base,d,0);for(let i=1;i<=512;i++){const next=state(base,d,i/512);area+=(last.pressure+next.pressure)/2*(next.volume-last.volume);last=next;}near(area,cycle(base,d).complete.workByGas.value!,2e-4);}
});
test('phase boundaries have the declared isothermal/adiabatic order and finite outputs serialize',()=>{
 for(const d of ['forward','reverse']as const){const stages=[0,.25,.5,.75].map(s=>state(base,d,s).stage);assert.deepEqual(stages,d==='forward'?['isothermal','adiabatic','isothermal','adiabatic']:['adiabatic','isothermal','adiabatic','isothermal']);
  const walk=(v:unknown):void=>{if(typeof v==='number')assert(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(walk);};const c=cycle(base,d);walk(c);assert.doesNotThrow(()=>JSON.stringify(c));
 }
});
test('invalid reservoir order, physical bounds, direction and progress are rejected',()=>{
 for(const key of Object.keys(base)as(keyof CarnotParameters)[])for(const v of [NaN,Infinity,-1,0])assert.throws(()=>state({...base,[key]:v},'forward',0),RangeError);
 assert.throws(()=>cycle({...base,hotTemperature:300,coldTemperature:400},'forward'),RangeError);assert.throws(()=>cycle(base,'other'as CarnotDirection),RangeError);
 for(const s of [-.1,1.1,NaN,Infinity])assert.throws(()=>state(base,'forward',s),RangeError);
});
