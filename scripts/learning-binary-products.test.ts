import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/continuum-binary-audit-independent.json';
import {binaryProduct,binarySum,binaryRatioObservation} from '../src/lib/learning-binary-products';
import {compactMassKernelObservations,coarseGrainedParticleState} from '../src/lib/learning-continuum-assumptions';
function check(actual:{value:number|null;sign:number;logMagnitude:number|null}|null,expected:{value:number|null;sign:number;logMagnitude:number|null}|null,label:string){
  if(expected===null){assert.equal(actual,null,label);return;}
  assert.ok(actual!==null);assert.equal(actual.value,expected.value,label+' rounded value');assert.equal(actual.sign,expected.sign,label+' sign');
  if(expected.logMagnitude===null)assert.equal(actual.logMagnitude,null,label);else assert.ok(actual.logMagnitude!==null&&Math.abs(actual.logMagnitude-expected.logMagnitude)<1e-8,label+' log');
}
for(const [i,c]of refs.binaryCases.entries())test(`independent Fraction binary product-sum-ratio rounding ${i+1}`,()=>{
  check(binaryRatioObservation(binarySum(c.terms.map(xs=>binaryProduct(...xs))),binarySum(c.denominator.map(xs=>binaryProduct(...xs)))),c.expected,'binary '+i);
});
for(const [i,c]of refs.kernelCases.entries())test(`independent exact rational compact kernel ${i+1}`,()=>{
  const r=compactMassKernelObservations(c.x,c.h);for(const key of ['value','spatialDerivative','bandwidthDerivative']as const)check(r[key],c.expected[key],'kernel '+i+' '+key);
});
for(const [i,c]of refs.particleCases.entries())test(`independent exact rational particle moment and derivative ${i+1}`,()=>{
  const r=coarseGrainedParticleState(c.particles,c.x,c.time,c.h,c.bandwidthRate);
  for(const key of Object.keys(c.expected)as (keyof typeof r.observations)[])check(r.observations[key],c.expected[key],'particle '+i+' '+key);
  assert.doesNotThrow(()=>JSON.stringify(r));
});
test('tiny opposing streams retain positive dispersion while equal streams have exact zero dispersion',()=>{
  const p={mass:1,position:0,velocity:1e-200};
  const opposite=coarseGrainedParticleState([p,{...p,velocity:-p.velocity}],0,0,1);
  assert.equal(opposite.velocity,0);
  assert.equal(opposite.momentumFlux,null);
  assert.equal(opposite.dispersion,null);
  assert.equal(opposite.observations.dispersion?.sign,1);
  assert.ok(opposite.observations.dispersion!.logMagnitude!<-900);
  assert.equal(opposite.meanMomentumTransport,0);
  assert.equal(opposite.status,'observed');
  const equal=coarseGrainedParticleState([p,p],0,0,1);
  assert.equal(equal.dispersion,0);
  assert.equal(equal.observations.dispersion?.sign,0);
  assert.equal(equal.meanMomentumTransport,null);
  const empty=coarseGrainedParticleState([p,p],4,0,1);
  assert.equal(empty.status,'empty-support');
  assert.equal(empty.observations.velocity,null);
  assert.equal(empty.observations.dispersion,null);
});
