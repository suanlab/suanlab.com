import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {stationaryThreeStateCycleModel as model,type StationaryCycleParameters} from '../src/lib/learning-stationary-cycle';
import {exactMarkovObservation,markovFraction} from '../src/lib/learning-finite-markov-exact';
import type {PartitionObservation} from '../src/lib/learning-partition-exact-core';

interface ReferenceObservation {numerator:string;denominator:string;value:number|null;sign:number;logMagnitude:string|null}
interface Fixture {id:string;parameters:StationaryCycleParameters;reference:Record<string,unknown>;classification:unknown;stationaryRank:number}
const fixture=JSON.parse(readFileSync('scripts/fixtures/stationary-cycle-oracle.json','utf8')) as {cases:Fixture[];caseCount:number;selectedHistoryRows:number};
const results=new Map<string,ReturnType<typeof model>>();
function evaluate(input:StationaryCycleParameters){const k=JSON.stringify(input);if(!results.has(k))results.set(k,model(input));return results.get(k)!;}
const defaultInput={forwardWeight:2,backwardWeight:1,stayWeight:1,initialWeights:[1,0,0],stepCount:12};
function quantity(actual:PartitionObservation,expected:ReferenceObservation,label:string){
 assert.ok(actual.bounds,label);
 for(const side of [actual.bounds.lower,actual.bounds.upper]){
  assert.equal(side.numerator.exponent,0,label);assert.equal(side.denominator.exponent,0,label);
  assert.equal(side.numerator.coefficient,expected.numerator,label);assert.equal(side.denominator.coefficient,expected.denominator,label);
 }
 assert.equal(actual.sign,expected.sign,label);assert.equal(actual.value,expected.value,label);
 assert.equal(actual.resolution,expected.sign===0?'structural-zero':'exact-rational',label);
 if(expected.logMagnitude===null){assert.equal(actual.logMagnitude,null,label);assert.equal(actual.logResolution,'not-applicable',label);}
 else{const truth=Number(expected.logMagnitude);assert.notEqual(actual.logMagnitude,null,label);assert.ok(Math.abs(actual.logMagnitude!-truth)<1e-10*Math.max(1,Math.abs(truth)),label);assert.equal(actual.logResolution,'finite',label);}
}
function compare(actual:unknown,reference:unknown,label:string){
 if(reference&&typeof reference==='object'&&'numerator'in reference){quantity(actual as PartitionObservation,reference as ReferenceObservation,label);return;}
 if(Array.isArray(reference)){assert.ok(Array.isArray(actual),label);assert.equal(actual.length,reference.length,label);reference.forEach((v,i)=>compare(actual[i],v,`${label}[${i}]`));return;}
 if(reference&&typeof reference==='object'){assert.ok(actual&&typeof actual==='object',label);for(const [k,v] of Object.entries(reference))compare((actual as Record<string,unknown>)[k],v,`${label}.${k}`);return;}
 assert.deepEqual(actual,reference,label);
}
for(const c of fixture.cases)test(`${c.id}: exact observations match independent Fraction matrices and graph classification`,()=>{
 const r=evaluate(c.parameters);assert.equal(r.arithmetic,'exact-bigint-rational');assert.equal(r.fullyResolved,true);
 for(const [k,v] of Object.entries(c.reference)){
  if(k==='history'){for(const row of v as {step:number}[])compare(r.history[row.step],row,`${c.id}.history[${row.step}]`);}
  else compare((r as unknown as Record<string,unknown>)[k],v,`${c.id}.${k}`);
 }
 assert.deepEqual(r.classification,c.classification);assert.equal(r.history.length,c.parameters.stepCount+1);
 const walk=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(walk);else if(v&&typeof v==='object')Object.values(v).forEach(walk);};walk(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
function exact(q:PartitionObservation,n:bigint|number,d:bigint|number=1){const expected=markovFraction(BigInt(n),BigInt(d));assert.ok(q.bounds);assert.equal(q.bounds.lower.numerator.coefficient,expected.n.toString());assert.equal(q.bounds.lower.denominator.coefficient,expected.d.toString());assert.deepEqual(q.bounds.lower,q.bounds.upper);}
interface Q{n:bigint;d:bigint}
const decode=(q:PartitionObservation):Q=>{assert.ok(q.bounds);return {n:BigInt(q.bounds.lower.numerator.coefficient),d:BigInt(q.bounds.lower.denominator.coefficient)};};
const add=(a:Q,b:Q):Q=>({n:a.n*b.d+b.n*a.d,d:a.d*b.d});
const cmp=(a:Q,b:Q)=>a.n*b.d-b.n*a.d;

test('all runtime history rows preserve normalization, TV contraction and exact upper bounds',()=>{
 let count=0;for(const c of fixture.cases){let previous:Q|null=null;for(const row of evaluate(c.parameters).history){
  const total=row.probabilities.map(decode).reduce(add);assert.equal(total.n,total.d);assert.ok(row.probabilities.every(p=>p.sign===0||p.sign===1));
  const tv=decode(row.totalVariationFromUniform),bound=decode(row.contractionUpperBound);assert.ok(cmp(tv,bound)<=BigInt(0));if(previous)assert.ok(cmp(tv,previous)<=BigInt(0));previous=tv;count++;
 }}assert.equal(count,3135);
});
test('a vertex can balance while its paired stationary currents are nonzero',()=>{
 const r=evaluate(defaultInput);exact(r.stationaryFlux[0][1],1,6);exact(r.stationaryFlux[1][0],1,12);exact(r.clockwiseCurrent,1,12);
 for(const row of r.stationaryBalance){exact(row.incoming,1,4);exact(row.outgoing,1,4);exact(row.holding,1,12);exact(row.netOutflow,0);}
 assert.equal(r.classification.detailedBalanceAtUniform,false);assert.equal(r.classification.currentDistributionIsStationary,false);
 const initial=evaluate({...defaultInput,stepCount:0});[3/4,-1/2,-1/4].forEach((v,i)=>assert.equal(initial.currentNetOutflow[i].value,v));
});
test('directed period three is distinct from the identity family and from an already uniform preparation',()=>{
 const directed=evaluate({...defaultInput,forwardWeight:1,backwardWeight:0,stayWeight:0,stepCount:8});
 assert.deepEqual(directed.current.probabilities.map(p=>p.value),[0,0,1]);assert.equal(directed.classification.stationaryReferenceIsUnique,true);assert.equal(directed.classification.uniqueClassPeriod,3);assert.equal(directed.classification.convergesToUniformForThisInitial,false);
 const uniform=evaluate({...directed.parameters,initialWeights:[1,1,1]});assert.equal(uniform.classification.convergesToUniformFromEveryInitial,false);assert.equal(uniform.classification.convergesToUniformForThisInitial,true);assert.ok(uniform.history.every(row=>row.totalVariationFromUniform.resolution==='structural-zero'));
 const identity=evaluate({...defaultInput,forwardWeight:0,backwardWeight:0,stayWeight:1,initialWeights:[1,2,7]});assert.equal(identity.classification.stationaryReferenceIsUnique,false);assert.equal(identity.classification.uniqueClassPeriod,null);assert.deepEqual(identity.classification.returnPeriods,[1,1,1]);assert.equal(identity.classification.currentDistributionIsStationary,true);exact(identity.current.totalVariationFromUniform,11,30);
});
test('two movement directions on a three-cycle are aperiodic even without holding',()=>{
 const r=evaluate({...defaultInput,forwardWeight:1,backwardWeight:1,stayWeight:0});assert.equal(r.classification.uniqueClassPeriod,1);assert.equal(r.classification.detailedBalanceAtUniform,true);assert.ok(r.transition.some((row,i)=>row.some((p,j)=>i!==j&&p.sign===1)));
});
test('time reversal swaps directional traffic while retaining the same stationary marginals',()=>{
 const a=evaluate(defaultInput),b=evaluate({...defaultInput,forwardWeight:1,backwardWeight:2});assert.deepEqual(a.timeReversedTransition,b.transition);assert.deepEqual(a.stationaryReference,b.stationaryReference);exact(a.cycleProductDifference,7,64);exact(b.cycleProductDifference,-7,64);
});
test('transition and initial common scale preserve the whole scientific result',()=>{
 const a=evaluate(defaultInput),b=evaluate({...defaultInput,forwardWeight:6,backwardWeight:3,stayWeight:3}),c=evaluate({...defaultInput,initialWeights:[1000000,0,0]});
 const withoutParameters=({parameters,...r}:ReturnType<typeof model>)=>{assert.ok(parameters);return r;};assert.deepEqual(withoutParameters(a),withoutParameters(b));assert.deepEqual(withoutParameters(a),withoutParameters(c));
});
test('the total-variation row coefficient bounds rather than equals every preparation decay',()=>{
 const r=evaluate({...defaultInput,forwardWeight:7,backwardWeight:2,stayWeight:5,stepCount:1});exact(r.contractionCoefficient,5,14);exact(r.current.totalVariationFromUniform,4,21);exact(r.current.contractionUpperBound,5,21);
});
test('an exact nonzero distance survives equal rounded marginals and linear-display underflow',()=>{
 const r=evaluate({forwardWeight:1000000,backwardWeight:999999,stayWeight:1000000,initialWeights:[1,0,0],stepCount:64});
 assert.deepEqual(r.current.probabilities.map(p=>p.value),[1/3,1/3,1/3]);assert.equal(r.classification.currentDistributionIsStationary,false);
 const distance=r.current.totalVariationFromUniform;assert.equal(distance.value,null);assert.equal(distance.sign,1);assert.equal(distance.resolution,'exact-rational');exact(distance,2,BigInt(3)*BigInt(2999999)**BigInt(64));assert.deepEqual(distance,r.current.contractionUpperBound);
 assert.ok(r.current.deviationFromUniform.every(q=>q.value===null&&q.sign!==0&&q.resolution==='exact-rational'));
});
test('canonical input snapshot is immutable and no unknown source property leaks into the result',()=>{
 const input={...defaultInput,initialWeights:[1,0,0],unrelated:Infinity},before=JSON.stringify(input),r=model(input);assert.equal(JSON.stringify(input),before);assert.notEqual(r.parameters.initialWeights,input.initialWeights);assert.equal(Object.isFrozen(r.parameters),true);assert.equal(Object.isFrozen(r.parameters.initialWeights),true);assert.ok(!('unrelated'in r.parameters));input.initialWeights[0]=0;assert.equal(r.parameters.initialWeights[0],1);
});
test('invalid weights, sparse preparations, zero sums and invalid transition counts are rejected',()=>{
 for(const field of ['forwardWeight','backwardWeight','stayWeight'] as const)for(const value of [-1,.5,Number.MIN_VALUE,1000001,NaN,Infinity,'1',null,undefined])assert.throws(()=>model({...defaultInput,[field]:value} as StationaryCycleParameters),RangeError);
 assert.throws(()=>model({...defaultInput,forwardWeight:0,backwardWeight:0,stayWeight:0}),RangeError);
 for(const initialWeights of [[0,0,0],[],[1,0],[1,0,0,0],[1,NaN,0],[1,.5,0],[1,-1,0],[1,1000001,0],new Array(3),[1,,0],null,'100'])assert.throws(()=>model({...defaultInput,initialWeights} as StationaryCycleParameters),RangeError);
 for(const stepCount of [-1,.5,513,NaN,Infinity,'1',undefined])assert.throws(()=>model({...defaultInput,stepCount} as StationaryCycleParameters),RangeError);
 for(const input of [null,undefined,{},[]])assert.throws(()=>model(input as StationaryCycleParameters),RangeError);
});
test('exact rational projection rounds ties to even and distinguishes nonzero underflow from zero',()=>{
 const p=BigInt(2)**BigInt(53);assert.equal(exactMarkovObservation(p+BigInt(1),p).value,1);assert.equal(exactMarkovObservation(p+BigInt(3),p).value,1+2**-51);
 const tiny=exactMarkovObservation(BigInt(1),BigInt(10)**BigInt(500));assert.equal(tiny.value,null);assert.equal(tiny.sign,1);exact(tiny,1,BigInt(10)**BigInt(500));
 assert.throws(()=>exactMarkovObservation(BigInt(1),BigInt(0)),RangeError);assert.deepEqual(markovFraction(BigInt(2),BigInt(-4)),{n:BigInt(-1),d:BigInt(2)});
});
