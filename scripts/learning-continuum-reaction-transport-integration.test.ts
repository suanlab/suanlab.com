import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumReactionKinds,reversibleTransportLab,fisherTransportLab,reactionControlBounds,fisherControlBounds}from '../src/lib/learning-continuum-reaction-transport-lab';
import{evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import{continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import{curriculumDomains}from '../src/data/learning-curriculum';
import refs from './fixtures/continuum-reaction-transport-independent.json';
const near=(a:number|null,b:number,tol=5e-10)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const o=v as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
for(const[index,id]of continuumReactionKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: bilingual registration connects real prerequisites, four concepts, three experiments and two quizzes`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));assert.equal(unit.questions.length,2);const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);
  bilingual({unit,guide,spec});assert.equal(spec.controls.length,index===0?13:11);assert.equal(spec.presets.length,index===0?16:18);assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?reactionControlBounds:fisherControlBounds);assert.ok(spec.controls.every(c=>c.step==='any'));
 });
 test(`${id}: all presets evaluate without input mutation, hidden infinities or invalid concentration bounds`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.reversibleTransport||r.fisherTransport);
   if(r.fisherTransport)for(const s of r.fisherTransport.spatial){const c=s.value.concentration;assert.ok(c===null||c>=0&&c<=r.fisherTransport.parameters.carryingConcentration);}
  }
 });
 test(`${id}: wrong-length, sparse, nonfinite and out-of-range controls are rejected`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
  for(let i=0;i<v.length;i++){for(const bad of[NaN,Infinity,spec.controls[i].max+1,spec.controls[i].min-1]){const a=[...v];a[i]=bad;assert.throws(()=>evaluateContinuum(id,a),RangeError);}const sparse=[...v];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);}
 });
}
test('reaction adapter preserves24 independent species derivatives and actual physical-window balances',()=>{
 for(const r of refs.reactionCases){const p=r.parameters,q=reversibleTransportLab([p.mass,p.area,p.initialSigma,p.initialMean,p.velocity,p.diffusivity,p.forwardRate,p.reverseRate,p.initialFractionA,r.time,r.x,r.left,r.right]);
  for(const[i,s]of[q.query.a,q.query.b].entries()){const e=r.expected.species[i];near(s.timeDerivative,e.dt);near(s.fluxGradient,e.fluxGradient);near(s.source,e.source);near((i===0?q.window!.a:q.window!.b).massRate,e.window.massRate);}
 }
});
test('Fisher adapter preserves24 independent derivatives, nonlinear sources and source integrals',()=>{
 for(const r of refs.frontCases){const p=r.parameters,q=fisherTransportLab([p.diffusivity,p.growthRate,p.carryingConcentration,p.bulkVelocity,p.initialFront,p.area,r.time,r.x,r.left,r.right,.5]);near(q.query.timeDerivative,r.expected.dt);near(q.query.secondDerivative,r.expected.dxx);near(q.query.source,r.expected.source);near(q.window!.integratedSource,r.expected.window.integratedSource);}
});
test('worked reaction, equilibrium turnovers and zero-rate nonunique states match the guide distinctions',()=>{
 const presets=continuumLabs[continuumReactionKinds[0]].presets,q=reversibleTransportLab(presets[0].values),eq=reversibleTransportLab(presets[1].values),none=reversibleTransportLab(presets[4].values);
 near(q.query.state.fractionA,1/3+(0.8-1/3)*Math.exp(-1.5));near(q.query.globalMassA!+q.query.globalMassB!,2);assert.ok(eq.query.forwardTurnover!>0);near(eq.query.forwardTurnover,eq.query.reverseTurnover!);assert.equal(eq.query.a.source,0);assert.equal(none.query.state.equilibriumA,null);assert.equal(none.query.state.fractionRateA,0);
});
test('reaction tiny production, underflowing transients and measured zero-mass fractions remain distinct',()=>{
 const presets=continuumLabs[continuumReactionKinds[0]].presets,tiny=reversibleTransportLab(presets[3].values),empty=reversibleTransportLab(presets[5].values),fast=reversibleTransportLab(presets[6].values),tail=reversibleTransportLab(presets[9].values);
 assert.equal(tiny.query.state.fractionA,1);assert.equal(tiny.query.state.roundedComplementB,0);assert.ok(tiny.query.state.fractionB!>0);assert.notEqual(tiny.query.state.stableFractionChangeA,0);assert.equal(empty.query.massNormalizedFractionA,null);assert.equal(empty.query.a.concentration,0);assert.equal(fast.query.state.fractionRateA,null);assert.ok(fast.query.state.logFractionRateMagnitude!<-1000);assert.equal(tail.query.a.concentration,null);assert.ok(tail.query.a.logConcentration!<-7000);
});
test('worked Fisher local balance and truly stationary laboratory preset match the fixed concepts',()=>{
 const presets=continuumLabs[continuumReactionKinds[1]].presets,q=fisherTransportLab(presets[0].values),stationary=fisherTransportLab(presets[1].values);near(q.query.concentration,1);near(q.query.timeDerivative,5/6);near(q.query.source,3/4);near(q.parameters.diffusivity*q.query.secondDerivative!,1/12);near(q.levelPosition,Math.log(Math.sqrt(2)-1));assert.equal(stationary.query.laboratorySpeed,0);assert.equal(stationary.query.timeDerivative,0);assert.ok(stationary.query.source!>0);assert.ok(stationary.query.relativeSpeed>0);assert.ok(stationary.temporal.every(s=>s.levelPosition===stationary.levelPosition));
});
test('Fisher plateau, tail and source-rescue presets retain stable information lost by rounded controls',()=>{
 const presets=continuumLabs[continuumReactionKinds[1]].presets,plateau=fisherTransportLab(presets[3].values),tail=fisherTransportLab(presets[5].values),rescued=fisherTransportLab(presets[6].values);
 assert.equal(plateau.query.roundedDeficit,0);assert.ok(plateau.query.stableDeficit!>0);assert.equal(plateau.query.roundedLogisticSource,0);assert.ok(plateau.query.source!>0);assert.equal(tail.query.concentration,null);assert.ok(Number.isFinite(tail.query.logConcentration));assert.equal(rescued.query.concentration,null);assert.equal(rescued.query.source,Number.MIN_VALUE);assert.equal(tail.query.globalMassStatus,'diverges-on-entire-line');assert.ok(tail.window!.logMassInside!<-700);
});
test('invalid window geometry remains visible without rewriting inputs or discarding local fields',()=>{
 for(const[id,indices]of[[continuumReactionKinds[0],[11,12]],[continuumReactionKinds[1],[13,14]]]as const)for(const index of indices){const v=continuumLabs[id].presets[index].values,q=evaluateContinuum(id,v),r=q.reversibleTransport??q.fisherTransport!;assert.equal(r.windowStatus,'invalid-boundary-order');assert.equal(r.window,null);assert.ok(r.query.x===r.x);assert.ok(r.spatial.length>240);}
});
test('adaptive spatial samples include narrow transitions, query and physical boundaries without truncating tails',()=>{
 const q=fisherTransportLab(continuumLabs[continuumReactionKinds[1]].presets[9].values);assert.ok(q.spatial.some(s=>s.x===q.query.center));assert.ok(q.spatial.some(s=>s.x===q.halfPosition));assert.ok(q.spatial.some(s=>s.x===q.x));assert.ok(q.spatial.some(s=>s.x===q.left));assert.ok(q.spatial.some(s=>s.x===q.right));assert.ok(q.spatial.some(s=>Math.abs(s.value.phase)<.3));for(let i=1;i<q.spatial.length;i++)assert.ok(q.spatial[i].x>q.spatial[i-1].x);assert.equal(q.spatial[0].x,-16);assert.equal(q.spatial.at(-1)!.x,16);
});
test('transport changes the cloud while declared composition stays fixed, and source omission exposes reaction',()=>{
 const v=continuumLabs[continuumReactionKinds[0]].presets[0].values,a=reversibleTransportLab(v),copy=[...v];copy[4]=-1;copy[5]=.8;const b=reversibleTransportLab(copy);near(a.query.state.fractionA,b.query.state.fractionA!);assert.notEqual(a.query.a.concentration,b.query.a.concentration);near(a.query.a.omittedReactionResidual,a.query.a.source!);near(a.window!.a.omittedReactionResidual,a.window!.a.integratedSource!);
});
test('reaction display resolves the cloud while retaining query, boundaries and the unchanged whole-line mass',()=>{
 const presets=continuumLabs[continuumReactionKinds[0]].presets,normal=reversibleTransportLab(presets[0].values),tail=reversibleTransportLab(presets[9].values);
 assert.ok(normal.maximum-normal.minimum<8);assert.equal(normal.spatial[0].x,normal.minimum);assert.equal(normal.spatial.at(-1)!.x,normal.maximum);assert.ok(normal.spatial.some(s=>s.x===normal.query.total.state.mean));assert.ok(normal.spatial.some(s=>s.x===normal.parameters.initialMean));
 for(const r of[normal,tail]){for(const x of[r.x,r.left,r.right])assert.ok(r.spatial.some(s=>s.x===x));assert.equal(r.query.globalMass,r.parameters.mass);assert.ok(r.spatial.every(s=>s.x>=r.minimum&&s.x<=r.maximum));}assert.equal(tail.maximum,15);assert.equal(tail.query.a.concentration,null);
});
