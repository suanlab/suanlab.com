import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/continuum-conservation-audit-independent.json';
import {movingMassWindowBalance,affineGasWindowBalance,affineGasWindowState,type AffineControlWindow} from '../src/lib/learning-continuum-conservation';
function match(value:number|null,expected:{rounded:number|null;sign:number},label:string){
  if(!expected.sign){assert.equal(value,0,label);return;}
  if(expected.rounded===null){assert.equal(value,null,label+' must be unavailable, not exact zero');return;}
  assert.notEqual(value,null,label);assert.notEqual(value,0,label+' nonzero');assert.equal(Math.sign(value!),expected.sign,label+' sign');
  assert.ok(Math.abs(value!-expected.rounded)<=2*Number.MIN_VALUE+3e-12*Math.abs(expected.rounded),label+': '+value+' vs '+expected.rounded);
}
for(const [i,c]of refs.transportCases.entries())test(`domain audit independent1600-digit moving mass difference ${i+1}`,()=>{
  match(movingMassWindowBalance(c.system,c.window,c.start,c.end).massChange,c.massChange,'mass '+i);
});
for(const [i,c]of refs.gasCases.entries())test(`domain audit independent1600-digit internal-energy difference ${i+1}`,()=>{
  match(affineGasWindowBalance(c.system,c.window as AffineControlWindow,c.start,c.end).internalChange,c.internalChange,'internal '+i);
});
test('a nonzero midpoint below ordinary range cannot become a structural zero',()=>{
  const s={meanDensity:1,contrast:.5,wavenumber:1,velocity:1,area:1},w={center:0,halfWidth:.5,speed:0};
  assert.equal(movingMassWindowBalance(s,w,0,Number.MIN_VALUE).massChange,null);
  assert.equal(movingMassWindowBalance(s,w,-Number.MIN_VALUE,Number.MIN_VALUE).massChange,0);
});
test('weighted pressure work survives a dimensionless power argument that rounds to zero',()=>{
  const s={initialDensity:1,initialPressure:1,expansionRate:Number.MIN_VALUE,drift:0,gamma:1.5,area:1},w:AffineControlWindow={center:0,halfWidth:.5,speed:0,mode:'material'};
  const q=affineGasWindowBalance(s,w,0,1);assert.equal(q.internalChange,-Number.MIN_VALUE);assert.equal(q.changes[2],-Number.MIN_VALUE);assert.equal(affineGasWindowState(s,w,0).pressureWorkNet,-Number.MIN_VALUE);
});
test('small physical volume may have unavailable mass while its weighted internal energy is representable',()=>{
  const c=refs.gasCases.at(-1)!,q=affineGasWindowState(c.system,c.window as AffineControlWindow,0);
  assert.equal(q.volume,null);assert.ok(q.internalEnergy!==null&&q.internalEnergy>0);
});
