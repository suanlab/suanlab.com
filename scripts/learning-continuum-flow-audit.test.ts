import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/continuum-flow-audit-independent.json';
import {linearizedEulerAt,taylorGreenAt,taylorGreenDecay,taylorGreenCell,taylorGreenParticle} from '../src/lib/learning-continuum-euler-navier-stokes';
function lookup(tree:unknown,path:string):number|null {
 let value=tree;
 for(const key of path.split('.')){assert.ok(value!==null&&typeof value==='object');value=(value as Record<string,unknown>)[key];}
 assert.ok(value===null||typeof value==='number');return value;
}
function match(actual:number|null,expected:{value:number|null;sign:number},label:string){
 if(expected.value===null){assert.equal(actual,null,label+' nonzero unavailable');return;}
 assert.notEqual(actual,null,label);assert.equal(actual===0?0:Math.sign(actual!),expected.sign,label+' sign');
 assert.ok(Math.abs(actual!-expected.value)<=2*Number.MIN_VALUE+4e-11*Math.abs(expected.value),label+': '+actual+' vs '+expected.value);
}
for(const [i,c]of refs.acousticCases.entries())test(`independent1600-digit acoustic point and derivative ${i+1}`,()=>{
 const q=linearizedEulerAt(c.parameters,c.x,c.time);
 for(const [path,expected]of Object.entries(c.expected))match(lookup(q,path),expected,path);
 assert.equal(q.positiveBackgroundAndTotals,true);assert.doesNotThrow(()=>JSON.stringify(q));
});
for(const [i,c]of refs.vortexCases.entries())test(`independent1600-digit vortex field and physical cell ${i+1}`,()=>{
 const q={...taylorGreenAt(c.parameters,c.x,c.y,c.time),cell:taylorGreenCell(c.parameters,c.time)};
 for(const [path,expected]of Object.entries(c.expected))match(lookup(q,path),expected,path);
 assert.doesNotThrow(()=>JSON.stringify(q));
});
const gas={density:.25,soundSpeed:.25,meanVelocity:1,gamma:1.4,waveNumber:1,rightAmplitude:Number.MIN_VALUE,leftAmplitude:0,contactAmplitude:0,phase:0};
const vortex={density:1,amplitude:4,waveNumber:.5,viscosity:Number.MIN_VALUE,pressureOffset:1,depth:1};
test('unavailable perturbations do not erase finite positive total gas fields',()=>{
 const q=linearizedEulerAt(gas,0,0);assert.equal(q.perturbation.density,null);assert.equal(q.perturbation.velocity,null);assert.equal(q.density,.25);assert.equal(q.velocity,1);assert.ok(q.pressure!==null&&q.pressure>0);assert.equal(q.positiveBackgroundAndTotals,true);
});
test('sub-range decay ratio can coexist with available weighted amplitude and cell-energy changes',()=>{
 const q=taylorGreenDecay(vortex,1);assert.equal(q.exponent,null);assert.equal(q.stableDecayChange,null);assert.equal(q.amplitudeChange,-2*Number.MIN_VALUE);assert.equal(q.stableEnergyChange,-Number.MIN_VALUE);
 const c=taylorGreenCell(vortex,1);assert.ok(c.stableEnergyChange!==null&&c.stableEnergyChange<0);assert.equal(c.roundedEnergyChange,0);
});
test('quadratic correction resolves a subnormal exponential half-way rounding case',()=>{
 const q=taylorGreenDecay({...vortex,amplitude:1},3);assert.equal(q.exponent,2*Number.MIN_VALUE);assert.equal(q.stableDecayChange,-Number.MIN_VALUE);
});
test('physical cell volume can rescue energy even when mean energy density is unavailable',()=>{
 const q=taylorGreenCell({...vortex,amplitude:1e-162,viscosity:0},1);assert.equal(q.meanKineticEnergy,null);assert.ok(q.kineticEnergy!==null&&q.kineticEnergy>0);
 const p=taylorGreenAt({...vortex,amplitude:1e-200},.4,.8,1);assert.equal(p.pressureChange,null);assert.equal(p.pressure,1);
});
test('zero amplitude and zero elapsed time retain genuine zero absolute changes',()=>{
 assert.equal(taylorGreenDecay({...vortex,amplitude:0},1).amplitudeChange,0);assert.equal(taylorGreenCell({...vortex,amplitude:0},1).stableEnergyChange,0);assert.equal(taylorGreenCell(vortex,0).stableEnergyChange,0);
});
test('velocity-only RK stages still integrate the declared field rather than another clock',()=>{
 const p={...vortex,amplitude:2,viscosity:.1,waveNumber:1};
 const q=taylorGreenParticle(p,[.4,.8],.5,256),last=q.points.at(-1)!;assert.equal(q.status,'integrated');assert.equal(last.time,.5);
 assert.ok(Math.abs(last.x-.4)>.01&&Math.abs(last.y-.8)>.01);
});
test('large exponential decay remains between complete loss and zero change',()=>{
 const p={...vortex,amplitude:1,waveNumber:3.9924318869828457,viscosity:.34804413239719156};
 assert.equal(taylorGreenDecay(p,2.836215666327569).stableEnergyChange,-1);
 for(let i=0;i<401;i++){
  const q=taylorGreenDecay({...vortex,viscosity:(i%97)/97,waveNumber:.25+3.75*((i*31)%89)/89},4*((i*53)%83)/83);
  for(const value of [q.stableDecayChange,q.stableEnergyChange])assert.ok(value!==null&&value>=-1&&value<=0);
 }
});
