import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {metropolisHastingsProposalModel as model,type HastingsProposalParameters} from '../src/lib/learning-hastings-proposal';
import type {PartitionObservation} from '../src/lib/learning-partition-exact-core';
interface Expected{numerator:string;denominator:string;value:number|null;sign:number;logMagnitude:string|null}
interface Fixture{id:string;parameters:HastingsProposalParameters;reference:Record<string,unknown>}
const fixture=JSON.parse(readFileSync('scripts/fixtures/hastings-proposal-oracle.json','utf8')) as {cases:Fixture[]};
const cache=new Map<string,ReturnType<typeof model>>();
function evaluate(p:HastingsProposalParameters){const k=JSON.stringify(p);if(!cache.has(k))cache.set(k,model(p));return cache.get(k)!;}
const base={targetWeights:[1,2,4],forwardWeight:2,backwardWeight:1,stayWeight:1};
function compare(actual:unknown,reference:unknown,label:string){
 if(reference&&typeof reference==='object'&&'numerator'in reference){
  const q=actual as PartitionObservation,e=reference as Expected;assert.ok(q.bounds,label);
  for(const b of [q.bounds.lower,q.bounds.upper]){assert.equal(b.numerator.coefficient,e.numerator,label);assert.equal(b.denominator.coefficient,e.denominator,label);assert.equal(b.numerator.exponent,0,label);assert.equal(b.denominator.exponent,0,label);}
  assert.equal(q.value,e.value,label);assert.equal(q.sign,e.sign,label);assert.equal(q.resolution,e.sign===0?'structural-zero':'exact-rational',label);
  if(e.logMagnitude===null){assert.equal(q.logMagnitude,null,label);assert.equal(q.logResolution,'not-applicable',label);}
  else{const truth=Number(e.logMagnitude);assert.notEqual(q.logMagnitude,null,label);assert.ok(Math.abs(q.logMagnitude!-truth)<=1e-10*Math.max(1,Math.abs(truth)),label);assert.equal(q.logResolution,'finite',label);}return;
 }
 if(Array.isArray(reference)){assert.ok(Array.isArray(actual),label);assert.equal(actual.length,reference.length,label);reference.forEach((v,i)=>compare(actual[i],v,`${label}[${i}]`));return;}
 if(reference&&typeof reference==='object'){assert.ok(actual&&typeof actual==='object',label);Object.entries(reference).forEach(([k,v])=>compare((actual as Record<string,unknown>)[k],v,`${label}.${k}`));return;}
 assert.deepEqual(actual,reference,label);
}
for(const c of fixture.cases)test(`${c.id}: exact Hastings/uncorrected observations match independent pair flows, trees and graph cycles`,()=>{
 const r=evaluate(c.parameters);compare(r,c.reference,c.id);assert.equal(r.arithmetic,'exact-bigint-rational');assert.equal(r.fullyResolved,true);
 const visit=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(visit);else if(v&&typeof v==='object')Object.values(v).forEach(visit);};visit(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
interface Q{n:bigint;d:bigint}
const decode=(q:PartitionObservation):Q=>{assert.ok(q.bounds);assert.deepEqual(q.bounds.lower,q.bounds.upper);return {n:BigInt(q.bounds.lower.numerator.coefficient),d:BigInt(q.bounds.lower.denominator.coefficient)};};
const plus=(a:Q,b:Q):Q=>({n:a.n*b.d+b.n*a.d,d:a.d*b.d});
const times=(a:Q,b:Q):Q=>({n:a.n*b.n,d:a.d*b.d});
const equal=(a:Q,b:Q)=>a.n*b.d===b.n*a.d;
const exact=(q:PartitionObservation,n:bigint|number,d:bigint|number=1)=>assert.ok(equal(decode(q),{n:BigInt(n),d:BigInt(d)}),`${q.value} expected ${n}/${d}`);
const total=(qs:PartitionObservation[])=>qs.map(decode).reduce(plus,{n:BigInt(0),d:BigInt(1)});

test('the worked counterexample separates target drift from the actual stationary law',()=>{
 const r=evaluate(base),u=r.uncorrected;[0,-1/28,1/28].forEach((v,i)=>assert.equal(u.targetResidual[i].value,v));assert.equal(u.targetInvariant,false);assert.equal(u.classification.targetIsUniqueStationary,false);
 assert.ok(u.stationaryDistribution&&u.stationaryCurrent&&u.stationaryTargetDistance);[1/7,5/21,13/21].forEach((v,i)=>assert.equal(u.stationaryDistribution![i].value,v));exact(u.stationaryTargetDistance,1,21);
 for(const [i,j] of [[0,1],[1,2],[2,0]])exact(u.stationaryCurrent[i][j],1,24);
 assert.equal(r.corrected.targetInvariant,true);assert.equal(r.corrected.targetDetailedBalance,true);exact(r.corrected.acceptance[2][0]!,1,8);exact(r.corrected.transition[2][0],1,16);exact(r.corrected.totalHoldingByState[2],11,16);
});
test('absent proposals, rejected one-way proposals, and accepted self proposals remain distinct',()=>{
 const r=evaluate({...base,forwardWeight:1,backwardWeight:0,stayWeight:0}),p=r.corrected;
 assert.equal(p.acceptance[0][2],null);assert.equal(p.acceptance[0][0],null);assert.notEqual(p.acceptance[0][1],null);exact(p.acceptance[0][1]!,0);exact(p.rejectionByState[0],1);exact(p.totalHoldingByState[0],1);
 const self=evaluate({...base,forwardWeight:0,backwardWeight:0,stayWeight:1}).corrected;exact(self.acceptance[0][0]!,1);assert.equal(self.acceptance[0][1],null);exact(self.meanProposalAcceptanceAtTarget,1);exact(self.meanMovementAtTarget,0);
});
test('mean acceptance includes self proposals and is not movement or an efficiency certificate',()=>{
 const r=evaluate(base).corrected;exact(r.meanProposalAcceptanceAtTarget,3,4);exact(r.meanProposalRejectionAtTarget,1,4);exact(r.meanMovementAtTarget,1,2);
 const stuck=evaluate({...base,forwardWeight:1,backwardWeight:0,stayWeight:1000000}).corrected;exact(stuck.meanProposalAcceptanceAtTarget,1000000,1000001);exact(stuck.meanMovementAtTarget,0);assert.equal(stuck.classification.convergesToTargetFromEveryInitial,false);
});
test('symmetric proposals make both transition rules identical for nonuniform targets',()=>{
 for(const weights of [[1,2,4],[1,1000000,1]]){const r=evaluate({...base,targetWeights:weights,forwardWeight:3,backwardWeight:3});assert.equal(r.correctionChangesTransition,false);assert.deepEqual(r.corrected,r.uncorrected);}
});
test('uniform-target invariance under asymmetric proposals does not establish detailed balance',()=>{
 const r=evaluate({...base,targetWeights:[1,1,1]});assert.equal(r.uncorrected.targetInvariant,true);assert.equal(r.uncorrected.targetDetailedBalance,false);assert.equal(r.correctionChangesTransition,true);assert.equal(r.corrected.targetDetailedBalance,true);
});
test('a nonuniform target can accidentally remain invariant under the uncorrected rule',()=>{
 const r=evaluate({...base,targetWeights:[1,1,4]});assert.equal(r.uncorrected.targetInvariant,true);assert.equal(r.uncorrected.targetDetailedBalance,false);for(const [i,j] of [[0,1],[1,2],[2,0]])exact(r.uncorrected.targetCurrent[i][j],1,24);assert.equal(r.corrected.targetDetailedBalance,true);
});
test('correct rejection of one-way proposals preserves a stationary family without exploring it',()=>{
 const r=evaluate({...base,forwardWeight:1,backwardWeight:0,stayWeight:0});assert.equal(r.corrected.targetInvariant,true);assert.equal(r.corrected.stationaryDistribution,null);assert.equal(r.corrected.stationaryFlux,null);assert.equal(r.corrected.stationaryTargetDistance,null);assert.equal(r.corrected.stationaryFamilyDimension,2);assert.equal(r.corrected.classification.stationaryIsUnique,false);assert.equal(r.uncorrected.classification.irreducible,true);assert.equal(r.uncorrected.classification.uniqueClassPeriod,1);
 const uniform=evaluate({...base,targetWeights:[1,1,1],forwardWeight:1,backwardWeight:0,stayWeight:0});assert.equal(uniform.uncorrected.classification.uniqueClassPeriod,3);assert.equal(uniform.uncorrected.targetInvariant,true);assert.equal(uniform.uncorrected.classification.convergesToTargetFromEveryInitial,false);
});
test('elementary proposal/accept/reject ledgers retain normalized transition and target-flux contracts',()=>{
 for(const c of fixture.cases){const r=evaluate(c.parameters);for(const mode of [r.corrected,r.uncorrected])for(let i=0;i<3;i++){
  assert.ok(equal(total(mode.transition[i]),{n:BigInt(1),d:BigInt(1)}));assert.ok(mode.transition[i].every(q=>q.sign===0||q.sign===1));
  assert.ok(equal(plus(decode(mode.selfProposalByState[i]),decode(mode.rejectionByState[i])),decode(mode.totalHoldingByState[i])));
  for(let j=0;j<3;j++){assert.ok(equal(plus(decode(mode.acceptedProposal[i][j]),decode(mode.rejectedProposal[i][j])),decode(r.proposal[i][j])));assert.ok(equal(times(decode(r.target[i]),decode(mode.transition[i][j])),decode(mode.targetFlux[i][j])));}
 }
 assert.ok(equal(plus(decode(r.corrected.meanProposalAcceptanceAtTarget),decode(r.corrected.meanProposalRejectionAtTarget)),{n:BigInt(1),d:BigInt(1)}));}
});
test('a 702-setting integer grid distinguishes conservation, reversibility, classes and periods exactly',()=>{
 let count=0;
 for(let a=1;a<=3;a++)for(let b=1;b<=3;b++)for(let c=1;c<=3;c++)for(let f=0;f<=2;f++)for(let q=0;q<=2;q++)for(let s=0;s<=2;s++){
  if(f+q+s===0)continue;const weights=[a,b,c],r=model({targetWeights:weights,forwardWeight:f,backwardWeight:q,stayWeight:s});
  assert.equal(r.corrected.targetInvariant,true);assert.equal(r.corrected.targetDetailedBalance,true);assert.equal(r.corrected.classification.irreducible,f>0&&q>0);assert.equal(r.corrected.classification.convergesToTargetFromEveryInitial,f>0&&q>0);
  const repeatedMinimum=weights.filter(w=>w===Math.min(...weights)).length>=2;
  assert.equal(r.uncorrected.targetInvariant,f===q||repeatedMinimum);assert.equal(r.uncorrected.targetDetailedBalance,f===q);assert.equal(r.correctionChangesTransition,f!==q);
  const uniform=a===b&&b===c,period=f+q===0?null:s===0&&(f===0||q===0)&&uniform?3:1;assert.equal(r.uncorrected.classification.uniqueClassPeriod,period);
  const changed=r.corrected.transition.some((row,i)=>row.some((v,j)=>!equal(decode(v),decode(r.uncorrected.transition[i][j]))));assert.equal(r.correctionChangesTransition,changed);count++;
 }assert.equal(count,702);
});
test('common target and proposal scales do not alter scientific values or classifications',()=>{
 const a=evaluate(base),b=evaluate({...base,targetWeights:[2,4,8]}),c=evaluate({...base,forwardWeight:6,backwardWeight:3,stayWeight:3});
 const without=({parameters,...r}:ReturnType<typeof model>)=>{assert.ok(parameters);return r;};assert.deepEqual(without(a),without(b));assert.deepEqual(without(a),without(c));
});
test('limited-digit marginal displays never replace the exact target drift test',()=>{
 const r=evaluate({targetWeights:[1000000,999999,1000000],forwardWeight:1000000,backwardWeight:999999,stayWeight:1000000});assert.deepEqual(r.target.map(q=>q.value!.toFixed(6)),r.uncorrected.targetOneStep.map(q=>q.value!.toFixed(6)));assert.equal(r.uncorrected.targetInvariant,false);assert.ok(r.uncorrected.targetResidual.some(q=>q.sign!==0));assert.equal(r.corrected.targetInvariant,true);
});
test('the equal-weight independent reset has zero row contraction with a unique target',()=>{
 const r=evaluate({targetWeights:[1,1,1],forwardWeight:1,backwardWeight:1,stayWeight:1}).corrected;exact(r.contractionCoefficient,0);assert.equal(r.classification.targetIsUniqueStationary,true);r.transition.forEach(row=>row.forEach(q=>exact(q,1,3)));
});
test('canonical parameters are immutable copies and unknown properties do not leak',()=>{
 const input={...base,targetWeights:[1,2,4],unrelated:Infinity},before=JSON.stringify(input),r=model(input);assert.equal(JSON.stringify(input),before);assert.notEqual(r.parameters.targetWeights,input.targetWeights);assert.equal(Object.isFrozen(r.parameters),true);assert.equal(Object.isFrozen(r.parameters.targetWeights),true);assert.ok(!('unrelated'in r.parameters));input.targetWeights[0]=100;assert.equal(r.parameters.targetWeights[0],1);
});
test('full-support target and finite integer proposal input contracts reject malformed preparations',()=>{
 for(const targetWeights of [[],[1,2],[1,2,3,4],[0,1,2],[-1,1,2],[.5,1,2],[1000001,1,2],[NaN,1,2],[Infinity,1,2],new Array(3),[1,,2],null,'124'])assert.throws(()=>model({...base,targetWeights} as HastingsProposalParameters),RangeError);
 for(const field of ['forwardWeight','backwardWeight','stayWeight'] as const)for(const value of [-1,.5,Number.MIN_VALUE,1000001,NaN,Infinity,'1',undefined])assert.throws(()=>model({...base,[field]:value} as HastingsProposalParameters),RangeError);
 assert.throws(()=>model({...base,forwardWeight:0,backwardWeight:0,stayWeight:0}),RangeError);for(const p of [null,undefined,{},[]])assert.throws(()=>model(p as HastingsProposalParameters),RangeError);
});

test('deleting all holding changes the stationary sampling law even for a correct target kernel',()=>{
 const r=evaluate(base).corrected;assert.ok(r.jumpStationaryDistribution&&r.jumpTargetDistance&&r.meanMovementAtStationary);[3/14,3/7,5/14].forEach((v,i)=>assert.equal(r.jumpStationaryDistribution![i].value,v));exact(r.jumpTargetDistance,3,14);exact(r.meanMovementAtStationary,1,2);assert.ok(r.jumpStationaryResidual!.every(q=>q.resolution==='structural-zero'));
 const identity=evaluate({...base,forwardWeight:0,backwardWeight:0,stayWeight:1}).corrected;assert.equal(identity.jumpTransition,null);assert.equal(identity.jumpStationaryDistribution,null);assert.equal(identity.meanMovementAtStationary,null);assert.equal(identity.jumpTargetDistance,null);
});
