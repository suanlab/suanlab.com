import test from 'node:test';
import assert from 'node:assert/strict';
import {scalarRiccati} from '../src/lib/learning-control-optimization';
import {robustSafeControl} from '../src/lib/learning-control-robust';
import {sequentialOffPolicy} from '../src/lib/learning-control-off-policy';
const relative=(actual:number,expected:number)=>assert.ok(Math.abs(actual/expected-1)<2e-12,`${actual} != ${expected}`);

test('Riccati retains subnormal penalties and gains without an artificial floor',()=>{
 const m=Number.MIN_VALUE;
 for(const n of [1,2,4,16]){
  const r=scalarRiccati(1,0,.125,n*m,3);
  assert.deepEqual(r.P,[n*m,n*m,n*m,n*m]);
  assert.deepEqual(r.K,[8*n*m,8*n*m,8*n*m]);
 }
 const r=scalarRiccati(.5,0,.125,m,1);
 assert.equal(r.K[0],4*m);
 assert.equal(r.P[0],0); // True result is below half the smallest subnormal.
 assert.equal(r.expandedK[0],0);
 assert.deepEqual(scalarRiccati(1,0,1,0,3).P,[0,0,0,0]);
});

test('small state penalties retain signed robust optima and zero-penalty endpoints',()=>{
 for(const sign of [-1,1])for(const q of [1e-20,1e-100,1e-250]){
  const r=robustSafeControl(1,sign,.25,2,2,q);
  relative(r.robust.u!,-sign*1.25*q);
  relative(r.nominal.u!,-sign*q);
  assert.equal(r.robust.expandedUnconstrained,0);
 }
 assert.equal(robustSafeControl(1,1,.25,2,2,0).robust.u,0);
 assert.equal(robustSafeControl(1,0,.25,2,2,4).robust.u,0);
 const kink=robustSafeControl(1,1,.5,2,2,4);
 assert.equal(kink.robust.u,-1);
});

test('on-policy Bernoulli return variance survives rounded constant plus tiny reward',()=>{
 for(const gamma of [1e-10,1e-20,1e-80,1e-150])for(const alpha of [.5,.75,1]){
  const r=sequentialOffPolicy(0,0,0,0,alpha,gamma,1);
  const expected=gamma*gamma*alpha*(1-alpha);
  if(expected===0){assert.equal(r.fullVariance,0);assert.equal(r.pdisVariance,0);}
  else {relative(r.fullVariance!,expected);relative(r.pdisVariance!,expected);}
  if(gamma<1e-16)assert.equal(r.expandedFullVariance,0);
 }
});

test('pairwise moments retain zero-discount support and overflow distinctions',()=>{
 const r=sequentialOffPolicy(.8,.6,.5,0,.8,0,1);
 assert.equal(r.fullVariance,null);
 relative(r.pdisVariance!,1/25);
 assert.equal(sequentialOffPolicy(0,0,0,0,.5,0,1).fullVariance,0);
 const rare=sequentialOffPolicy(.8,.6,Number.MIN_VALUE,Number.MIN_VALUE,.8,1,1);
 assert.equal(rare.fullSupport,true);
 assert.equal(rare.fullVariance,null);
 const zero=sequentialOffPolicy(1,0,Number.MIN_VALUE,.5,1,1,1);
 assert.equal(zero.fullVariance,0);
});
