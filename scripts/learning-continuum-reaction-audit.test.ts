import test from 'node:test';
import assert from 'node:assert/strict';
import {reversibleReactionState,reversibleReactionAt} from '../src/lib/learning-continuum-reaction-transport';
import refs from './fixtures/continuum-reaction-audit-independent.json';

function check(actual:number|null, expected:{value:number|null;sign:number}) {
  if(expected.value===null||expected.sign===0){assert.equal(actual,expected.value);return;}
  assert.notEqual(actual,null);
  assert.equal(Math.sign(actual!),expected.sign);
  if(Math.abs(expected.value)<1e-300)assert.equal(actual,expected.value);
  else assert.ok(Math.abs(actual!-expected.value)<=Math.abs(expected.value)*3e-12,`${actual} vs ${expected.value}`);
}
for(const [index,ref] of refs.cases.entries()) {
  test(`independent single-round kinetic observations ${index}`,()=>{
    const state=reversibleReactionState(ref.parameters,ref.time);
    const point=reversibleReactionAt({...ref.parameters,mass:3,area:1,initialSigma:1,
      initialMean:0,velocity:0,diffusivity:0},0,ref.time);
    for(const key of ['fractionA','fractionB','equilibriumA','equilibriumB','fractionRateA','stableFractionChangeA'] as const)
      check(state[key],ref.expected[key]);
    for(const key of ['globalMassA','globalMassB','globalSourceA','globalSourceB'] as const)
      check(point[key],ref.expected[key]);
    check(point.a.source,ref.expected.localSourceA);
    if(ref.logRateMagnitude===null)assert.equal(state.logFractionRateMagnitude,null);
    else assert.ok(Math.abs(state.logFractionRateMagnitude!-ref.logRateMagnitude)<2e-11);
    assert.doesNotThrow(()=>JSON.stringify(point));
  });
}
test('subnormal kinetic midpoint is unavailable while a complete mass-weighted source survives',()=>{
  const parameters={forwardRate:Number.MIN_VALUE,reverseRate:Number.MIN_VALUE,initialFractionA:.25};
  const q=reversibleReactionState(parameters,1);
  assert.equal(q.fractionRateA,null);assert.equal(q.stableFractionChangeA,null);
  assert.equal(q.fractionRateSign,1);assert.equal(q.equilibriumA,.5);
  const p=reversibleReactionAt({...parameters,mass:3,area:1,initialSigma:1,initialMean:0,velocity:0,diffusivity:0},0,1);
  assert.equal(p.globalSourceA,Number.MIN_VALUE);
  assert.equal(reversibleReactionAt({...parameters,mass:0,area:1,initialSigma:1,initialMean:0,velocity:0,diffusivity:0},0,1).globalSourceA,0);
});
test('log fraction retains a tiny change when the ordinary fraction rounds to one',()=>{
  const q=reversibleReactionState({forwardRate:1e-20,reverseRate:0,initialFractionA:1},1);
  assert.equal(q.fractionA,1);assert.equal(q.logFractionA,-1e-20);
  assert.ok(q.fractionB!>0);
});
