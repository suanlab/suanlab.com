import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {reversibleTwoStateMixingModel as model,type TwoStateMixingParameters} from '../src/lib/learning-two-state-mixing';
import type {PartitionObservation} from '../src/lib/learning-partition-exact-core';
interface Expected{numerator:string;denominator:string;value:number|null;sign:number;logMagnitude:string|null}
interface Fixture{id:string;parameters:TwoStateMixingParameters;reference:Record<string,unknown>}
const fixture=JSON.parse(readFileSync('scripts/fixtures/two-state-mixing-oracle.json','utf8')) as {cases:Fixture[]};
const cache=new Map<string,ReturnType<typeof model>>();
function evaluate(p:TwoStateMixingParameters){const k=JSON.stringify(p);if(!cache.has(k))cache.set(k,model(p));return cache.get(k)!;}
const base={forwardNumerator:1,backwardNumerator:2,transitionDenominator:4,initialOneNumerator:0,initialDenominator:1,stepCount:12};
function compare(actual:unknown,reference:unknown,label:string){
 if(reference&&typeof reference==='object'&&'numerator'in reference){const q=actual as PartitionObservation,e=reference as Expected;assert.ok(q.bounds,label);for(const b of [q.bounds.lower,q.bounds.upper]){assert.equal(b.numerator.coefficient,e.numerator,label);assert.equal(b.denominator.coefficient,e.denominator,label);assert.equal(b.numerator.exponent,0,label);assert.equal(b.denominator.exponent,0,label);}assert.equal(q.value,e.value,label);assert.equal(q.sign,e.sign,label);assert.equal(q.resolution,e.sign===0?'structural-zero':'exact-rational',label);if(e.logMagnitude===null){assert.equal(q.logMagnitude,null,label);assert.equal(q.logResolution,'not-applicable',label);}else{const truth=Number(e.logMagnitude);assert.notEqual(q.logMagnitude,null,label);assert.ok(Math.abs(q.logMagnitude!-truth)<1e-10*Math.max(1,Math.abs(truth)),label);assert.equal(q.logResolution,'finite',label);}return;}
 if(Array.isArray(reference)){assert.ok(Array.isArray(actual),label);assert.equal(actual.length,reference.length,label);reference.forEach((v,i)=>compare(actual[i],v,`${label}[${i}]`));return;}
 if(reference&&typeof reference==='object'){assert.ok(actual&&typeof actual==='object',label);Object.entries(reference).forEach(([k,v])=>compare((actual as Record<string,unknown>)[k],v,`${label}.${k}`));return;}
 assert.deepEqual(actual,reference,label);
}
for(const c of fixture.cases)test(`${c.id}: exact evolution matches independent Fraction powers, closed sums and support`,()=>{
 const r=evaluate(c.parameters);for(const [k,v] of Object.entries(c.reference)){if(k==='history')for(const row of v as {step:number}[])compare(r.history[row.step],row,`${c.id}.history[${row.step}]`);else compare((r as unknown as Record<string,unknown>)[k],v,`${c.id}.${k}`);}assert.equal(r.history.length,c.parameters.stepCount+1);assert.equal(r.arithmetic,'exact-bigint-rational');assert.equal(r.fullyResolved,true);
 const visit=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(visit);else if(v&&typeof v==='object')Object.values(v).forEach(visit);};visit(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
interface Q{n:bigint;d:bigint}
const decode=(q:PartitionObservation):Q=>{assert.ok(q.bounds);assert.deepEqual(q.bounds.lower,q.bounds.upper);return {n:BigInt(q.bounds.lower.numerator.coefficient),d:BigInt(q.bounds.lower.denominator.coefficient)};};
const add=(a:Q,b:Q):Q=>({n:a.n*b.d+b.n*a.d,d:a.d*b.d});
const subtract=(a:Q,b:Q):Q=>({n:a.n*b.d-b.n*a.d,d:a.d*b.d});
const cmp=(a:Q,b:Q)=>a.n*b.d-b.n*a.d;
const exact=(q:PartitionObservation,n:bigint|number,d:bigint|number=1)=>assert.equal(cmp(decode(q),{n:BigInt(n),d:BigInt(d)}),BigInt(0));
test('all 4033 runtime rows preserve mass, the closed recurrence, flux increments and TV contraction',()=>{
 let count=0;for(const c of fixture.cases){const r=evaluate(c.parameters);let prior:Q|null=null;for(const row of r.history){assert.ok(row.closedFormMatchesEvolution);assert.ok(row.probabilities.every(q=>q.sign===0||q.sign===1));const sum=add(decode(row.probabilities[0]),decode(row.probabilities[1]));assert.equal(sum.n,sum.d);assert.equal(cmp(decode(row.probabilities[1]),decode(row.closedFormProbabilityOne)),BigInt(0));const next=row.step===c.parameters.stepCount?r.nextDistribution:r.history[row.step+1].probabilities;assert.equal(cmp(subtract(decode(next[1]),decode(row.probabilities[1])),decode(row.netForwardFlux)),BigInt(0));if(row.totalVariationFromStationary){const tv=decode(row.totalVariationFromStationary);assert.ok(cmp(tv,decode(row.worstCaseTotalVariation!))<=BigInt(0));if(prior)assert.ok(cmp(tv,prior)<=BigInt(0));prior=tv;}count++;}}
 assert.equal(count,4033);
});
test('identity preparations have a stationary family and initial-dependent limits without a chosen reference',()=>{
 const r=evaluate({...base,forwardNumerator:0,backwardNumerator:0,initialOneNumerator:1,initialDenominator:3});assert.equal(r.stationaryDistribution,null);assert.equal(r.stationaryFamilyDimension,1);assert.equal(r.stationaryIndicatorVariance,null);assert.equal(r.stationaryLagOneCorrelation,null);assert.equal(r.stationaryTimeReversal,null);assert.ok(r.history.every(row=>row.totalVariationFromStationary===null&&row.deviationFromStationary===null));assert.deepEqual(r.limitDistribution,r.initial);assert.deepEqual(r.cesaroLimitDistribution,r.initial);assert.equal(r.classification.currentDistributionIsStationary,true);assert.equal(r.classification.hasPointwiseLimitForThisInitial,true);assert.equal(r.classification.convergesToUniqueStationaryForThisInitial,null);
});
test('alternation preserves a unique reversible stationary law but nonstationary preparations never converge',()=>{
 const r=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1,stepCount:3});assert.equal(r.classification.irreducible,true);assert.equal(r.classification.stationaryIsUnique,true);assert.deepEqual(r.classification.returnPeriods,[2,2]);assert.equal(r.limitDistribution,null);assert.equal(r.mixingAtQuarter.status,'never-periodic');exact(r.current.totalVariationFromStationary!,1,2);assert.deepEqual(r.cesaroAverage.map(q=>q.value),[.5,.5]);assert.deepEqual(r.cesaroLimitDistribution,r.stationaryDistribution);
 const stationary=evaluate({...r.parameters,initialOneNumerator:1,initialDenominator:2});assert.equal(stationary.classification.hasPointwiseLimitForThisInitial,true);assert.deepEqual(stationary.limitDistribution,stationary.stationaryDistribution);assert.equal(stationary.mixingAtQuarter.status,'never-periodic');assert.ok(stationary.history.every(row=>row.totalVariationFromStationary!.resolution==='structural-zero'));
});
test('absorbing chains can converge uniquely despite reducibility and leave zero-support reverse rows unspecified',()=>{
 const r=evaluate({...base,forwardNumerator:0,backwardNumerator:1,transitionDenominator:2,initialOneNumerator:1});assert.equal(r.classification.irreducible,false);assert.equal(r.classification.convergesToUniqueStationaryFromEveryInitial,true);assert.deepEqual(r.stationarySupport,[0]);assert.deepEqual(r.stationaryDistribution!.map(q=>q.value),[1,0]);assert.deepEqual(r.stationaryTimeReversal![0]!.map(q=>q.value),[1,0]);assert.equal(r.stationaryTimeReversal![1],null);exact(r.stationaryIndicatorVariance!,0);exact(r.stationaryLagOneCovariance!,0);assert.equal(r.stationaryLagOneCorrelation,null);
 const immediate=evaluate({...r.parameters,transitionDenominator:1});assert.deepEqual(immediate.classification.returnPeriods,[1,null]);assert.deepEqual(immediate.classification.closedCommunicatingClasses,[[0]]);
});
test('one-step reset occurs after rather than at the initial recorded step',()=>{
 const r=evaluate({...base,forwardNumerator:1,backwardNumerator:3,transitionDenominator:4,initialOneNumerator:2,initialDenominator:3,stepCount:3});exact(r.history[0].probabilities[1],2,3);for(const row of r.history.slice(1))exact(row.probabilities[1],1,4);assert.equal(r.history[0].closedFormMatchesEvolution,true);assert.equal(r.mixingAtQuarter.firstStep,1);
});
test('worst-initial quarter mixing differs from distance of an already stationary preparation',()=>{
 const r=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:3,initialOneNumerator:1,initialDenominator:2,stepCount:0});exact(r.current.totalVariationFromStationary!,0);exact(r.current.worstCaseTotalVariation!,1,2);assert.equal(r.mixingAtQuarter.status,'not-reached-within-recorded-range');assert.equal(r.mixingAtQuarter.firstStep,null);
});
test('quarter equality is inclusive and a finite recorded horizon does not imply permanent nonconvergence',()=>{
 const symmetric=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:4});assert.equal(symmetric.mixingAtQuarter.firstStep,1);exact(symmetric.history[1].worstCaseTotalVariation!,1,4);
 const short=evaluate({...base,forwardNumerator:0,backwardNumerator:1,transitionDenominator:2,initialOneNumerator:1,stepCount:1});assert.equal(short.mixingAtQuarter.status,'not-reached-within-recorded-range');assert.equal(short.classification.convergesToUniqueStationaryFromEveryInitial,true);assert.equal(evaluate({...short.parameters,stepCount:2}).mixingAtQuarter.firstStep,2);
 const slow=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1000000,stepCount:512});assert.equal(slow.mixingAtQuarter.status,'not-reached-within-recorded-range');assert.equal(slow.classification.convergesToUniqueStationaryFromEveryInitial,true);
});
test('Cesaro averaging includes the initial distribution and changes neither pointwise convergence nor sampling variance',()=>{
 const r=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1,stepCount:2});exact(r.current.probabilities[1],0);exact(r.current.cesaroProbabilityOne,1,3);exact(r.current.cesaroDistanceFromStationary!,1,6);assert.equal(r.classification.hasPointwiseLimitForThisInitial,false);assert.equal(r.history[0].cesaroProbabilityOne.value,0);
});
test('tiny signed deviations stay nonzero when projected probabilities equal the stationary display',()=>{
 for(const backwardNumerator of [499999,500001]){const r=evaluate({...base,forwardNumerator:500000,backwardNumerator,transitionDenominator:1000000,stepCount:64});assert.deepEqual(r.current.probabilities.map(q=>q.value),r.stationaryDistribution!.map(q=>q.value));assert.equal(r.current.totalVariationFromStationary!.value,null);assert.equal(r.current.totalVariationFromStationary!.sign,1);assert.equal(r.classification.currentDistributionIsStationary,false);assert.ok(r.current.deviationFromStationary!.every(q=>q.sign!==0&&q.value===null));const sign=backwardNumerator===499999?-1:1;assert.equal(r.history[63].deviationFromStationary![1].sign,sign);assert.equal(r.history[64].deviationFromStationary![1].sign,-1);}
});
test('eigenvalue, variance and normalized lag correlation have distinct meanings',()=>{
 const r=evaluate(base);exact(r.secondEigenvalue,1,4);exact(r.stationaryIndicatorVariance!,2,9);exact(r.stationaryLagOneCovariance!,1,18);exact(r.stationaryLagOneCorrelation!,1,4);
 const flip=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1});exact(flip.stationaryLagOneCovariance!,-1,4);exact(flip.stationaryIndicatorVariance!,1,4);exact(flip.stationaryLagOneCorrelation!,-1);
});
test('a 270-setting rational grid preserves exact recurrence and class/convergence distinctions',()=>{
 let count=0;for(let d=1;d<=5;d++)for(let a=0;a<=d;a++)for(let b=0;b<=d;b++)for(const u of [0,1,2]){
  const r=model({forwardNumerator:a,backwardNumerator:b,transitionDenominator:d,initialOneNumerator:u,initialDenominator:2,stepCount:12});assert.ok(r.history.every(row=>row.closedFormMatchesEvolution));assert.equal(r.classification.stationaryIsUnique,a+b>0);assert.equal(r.classification.irreducible,a>0&&b>0);assert.equal(r.classification.convergesToUniqueStationaryFromEveryInitial,a+b>0&&a+b<2*d);assert.equal(r.classification.allStationaryLawsSatisfyDetailedBalance,true);if(a+b>0){assert.equal(r.mixingAtQuarter.status,a+b===2*d?'never-periodic':'reached-within-recorded-range');}else assert.equal(r.mixingAtQuarter.status,'not-applicable-nonunique-stationary');count++;
 }assert.equal(count,270);
});
test('common numerator/denominator scales preserve all scientific results',()=>{
 const a=evaluate({...base,initialOneNumerator:1,initialDenominator:3}),b=evaluate({...a.parameters,forwardNumerator:3,backwardNumerator:6,transitionDenominator:12}),c=evaluate({...a.parameters,initialOneNumerator:2,initialDenominator:6});const without=({parameters,...r}:ReturnType<typeof model>)=>{assert.ok(parameters);return r;};assert.deepEqual(without(a),without(b));assert.deepEqual(without(a),without(c));
});
test('canonical input is immutable and unknown properties are excluded',()=>{
 const input={...base,unrelated:Infinity},before=JSON.stringify(input),r=model(input);assert.equal(JSON.stringify(input),before);assert.equal(Object.isFrozen(r.parameters),true);assert.ok(!('unrelated'in r.parameters));
});
test('invalid denominators, probability numerators and horizons are rejected',()=>{
 for(const field of ['transitionDenominator','initialDenominator'] as const)for(const value of [0,-1,.5,1000001,NaN,Infinity,'2',undefined])assert.throws(()=>model({...base,[field]:value} as TwoStateMixingParameters),RangeError);
 for(const field of ['forwardNumerator','backwardNumerator','initialOneNumerator'] as const)for(const value of [-1,.5,1000001,NaN,Infinity,'1',undefined])assert.throws(()=>model({...base,[field]:value} as TwoStateMixingParameters),RangeError);
 assert.throws(()=>model({...base,forwardNumerator:5}),RangeError);assert.throws(()=>model({...base,backwardNumerator:5}),RangeError);assert.throws(()=>model({...base,initialOneNumerator:2}),RangeError);for(const stepCount of [-1,.5,513,NaN,Infinity,'1'])assert.throws(()=>model({...base,stepCount} as TwoStateMixingParameters),RangeError);for(const input of [null,undefined,{},[]])assert.throws(()=>model(input as TwoStateMixingParameters),RangeError);
});

test('a large signed spectral gap does not imply contraction when the eigenvalue is near minus one',()=>{
 const flip=evaluate({...base,forwardNumerator:1,backwardNumerator:1,transitionDenominator:1});exact(flip.signedSpectralGap,2);exact(flip.absoluteSpectralGap,0);assert.equal(flip.mixingAtQuarter.status,'never-periodic');
 const near=evaluate({...base,forwardNumerator:999999,backwardNumerator:999999,transitionDenominator:1000000,stepCount:512});exact(near.signedSpectralGap,999999,500000);exact(near.absoluteSpectralGap,1,500000);assert.equal(near.mixingAtQuarter.status,'not-reached-within-recorded-range');assert.equal(near.classification.convergesToUniqueStationaryFromEveryInitial,true);
});
