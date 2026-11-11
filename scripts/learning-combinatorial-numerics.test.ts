import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateCombinatorial} from '../src/lib/learning-combinatorial';
import {entropyDiscreteChoice as solve} from '../src/lib/learning-combinatorial-soft';
const relative=(actual:number,expected:number)=>assert.ok(Math.abs(actual/expected-1)<2e-12,`${actual} != ${expected}`);

test('rare binary entropy retains the normalization contribution even when the partition rounds to one',()=>{
 for(const tau of [.01,.02,.04]){
  const r=solve(0,-1,tau,0),gapOverTemperature=3/tau,w=Math.exp(-gapOverTemperature);
  relative(r.entropy,(1+gapOverTemperature)*w);
  relative(r.logProbabilities[0]!,-w);
  relative(r.softMinimumCorrection,-tau*w);
  relative(r.expectedExcess,3*w);
  assert.ok(r.entropy>r.expandedEntropy);
 }
});

test('cooling around either unique binary minimizer preserves reflection symmetry',()=>{
 for(const tau of [.01,.02,.1]){
  const left=solve(0,-1,tau,0),right=solve(0,2,tau,0);
  assert.equal(left.entropy,right.entropy);
  assert.equal(left.logProbabilities[0],right.logProbabilities[1]);
  assert.equal(left.expectedExcess,right.expectedExcess);
 }
});

test('rare third option retains its finite log probability while normalization stays nonnegative',()=>{
 const r=solve(1,-1,.01,0);
 assert.equal(r.probabilities[2],0);
 assert.equal(r.logProbabilities[2],-1500);
 assert.ok(r.logPartition>0);
 relative(r.entropy,301*Math.exp(-300));
 relative(r.logProbabilities[0]!,-Math.exp(-300));
});

test('analytical limits retain exact ties and ignore temperature in all new diagnostics',()=>{
 for(const target of [.5,1.25,2]){
  const a=solve(1,target,.01,1),b=solve(1,target,2,1);
  for(const key of ['expandedEntropy','logPartition','expectedExcess','softMinimumCorrection'] as const)assert.equal(a[key],b[key]);
  assert.equal(a.expectedExcess,0);
  assert.equal(a.softMinimumCorrection,0);
  assert.equal(a.entropy,Math.log(a.minimizers.length));
 }
});

test('cost-gap readout preserves rare variance while retaining rounded subtraction as a diagnostic',()=>{
 for(const [mode,target] of [[0,-1],[0,2],[1,-1]]){
  const r=solve(mode,target,.01,0),model=evaluateCombinatorial('entropy-softmin-discrete-choice',[mode,target,.01,0]);
  assert.equal(r.expandedCostGap,0);
  relative(r.costGap,Math.exp(-300));
  assert.equal(model.stats['discrete cost minus mean loss'],r.costGap);
  assert.ok(r.costGap>0);
 }
});

test('centered compensated residual retains mean loss at a unique minimum and an irregular tie',()=>{
 const unique=solve(1,1,.01,0);
 assert.equal(unique.expandedMeanLoss,0);
 relative(unique.meanResidual,-Math.exp(-100));
 relative(unique.meanLoss,Math.exp(-200));
 for(const tau of [.01,.1]){
  const tie=solve(1,2,tau,0),w=Math.exp(-3/tau);
  if(tau===.01)assert.equal(tie.expandedMeanLoss,0);
  relative(tie.meanResidual,-w);
  relative(tie.meanLoss,w*w);
  assert.equal(evaluateCombinatorial('entropy-softmin-discrete-choice',[1,2,tau,0]).stats['loss of weighted mean'],tie.meanLoss);
 }
 assert.equal(solve(0,.5,.01,0).meanLoss,0);
 assert.equal(solve(1,2,.01,1).meanLoss,0);
});
