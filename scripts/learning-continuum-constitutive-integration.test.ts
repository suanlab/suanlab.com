import test from 'node:test';
import assert from 'node:assert/strict';
import{continuumConstitutiveKinds,viscoelasticLab,neoHookeanLab,viscoelasticControlBounds,neoHookeanControlBounds}from '../src/lib/learning-continuum-constitutive-lab';
import{evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import{viscoelasticCycle,viscoelasticModuli}from '../src/lib/learning-continuum-constitutive';
import{continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import{curriculumDomains}from '../src/data/learning-curriculum';
import refs from './fixtures/continuum-constitutive-independent.json';
const near=(a:number|null,b:number,tol=4e-10)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const o=v as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
for(const[index,id]of continuumConstitutiveKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: bilingual registration connects prerequisites, four fixed concepts, three guides and two quizzes`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));assert.equal(unit.questions.length,2);const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);bilingual({unit,guide,spec});assert.equal(spec.controls.length,8);assert.equal(spec.presets.length,index===0?20:18);assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?viscoelasticControlBounds:neoHookeanControlBounds);assert.ok(spec.controls.every(c=>c.step==='any'));
 });
 test(`${id}: all presets preserve inputs and expose finite or explicitly missing observations`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.viscoelastic||r.neoHookean);}
 });
 test(`${id}: wrong-length, sparse, nonfinite and out-of-range controls are rejected`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);for(let i=0;i<v.length;i++){for(const bad of[NaN,Infinity,spec.controls[i].max+1,spec.controls[i].min-1]){const a=[...v];a[i]=bad;assert.throws(()=>evaluateContinuum(id,a),RangeError);}const a=[...v];delete a[i];assert.throws(()=>evaluateContinuum(id,a),RangeError);}
 });
}
test('viscoelastic adapter retains24 independent convolution fields and physical-time budgets',()=>{
 for(const r of refs.viscoCases){const p=r.parameters,q=viscoelasticLab([p.elasticModulus,p.viscosity,p.strainAmplitude,p.angularFrequency,p.initialElasticStrain,r.time,r.start,r.end]);for(const name of['kelvin','maxwell']as const){near(q.query[name].stress,r.expected[name].stress);near(q.query[name].stressRate,r.expected[name].stressRate);near(q.window![name].work,r.window[name].work);near(q.window![name].dissipation,r.window[name].dissipation);}}
});
test('hyperelastic adapter retains24 independent matrix, area, traction and conjugate-work references',()=>{
 for(const r of refs.neoCases){const p=r.parameters,q=neoHookeanLab([p.shearModulus,p.logStretch,p.rotation,p.normalAngle,p.referenceArea,p.referenceLength,r.start,r.end]);near(q.query.force,r.expected.force);near(q.query.normalStress,r.expected.normalStress);near(q.query.signedShearStress,r.expected.signedShearStress);near(q.work.workDensity,r.path.workDensity);near(q.work.wrongCauchyWorkDensity,r.path.wrongCauchyWorkDensity);}
});
test('frequency-only evaluation matches the complete cycle coefficients without imposing a cycle horizon',()=>{
 const p={elasticModulus:2,viscosity:1,strainAmplitude:.002,angularFrequency:1,initialElasticStrain:0};for(const angularFrequency of[0,Number.MIN_VALUE,.01,1,4]){const a=viscoelasticModuli({...p,angularFrequency}),b=viscoelasticCycle({...p,angularFrequency});for(const key of Object.keys(a)as(keyof typeof a)[])assert.equal(a[key],b[key]);}const r=viscoelasticLab(continuumLabs[continuumConstitutiveKinds[0]].presets[0].values);assert.equal(r.spectrum.length,61);near(r.spectrum[0].value.kelvinLoss,.0001);near(r.spectrum.at(-1)!.value.maxwellLoss,.8);near(r.spectrum.at(-1)!.value.maxwellStorage,1.6);
});
test('worked startup, initial memory and first-cycle examples preserve their different energy meanings',()=>{
 const s=continuumLabs[continuumConstitutiveKinds[0]].presets,a=viscoelasticLab(s[0].values),memory=viscoelasticLab(s[1].values),first=viscoelasticLab(s[2].values),periodic=viscoelasticLab(s[4].values);near(a.cycle.maxwellStorage,.4);near(a.cycle.maxwellLoss,.8);near(memory.query.homogeneousMemory,.004*Math.exp(-2));assert.notEqual(first.cycle.actualFirstCycle!.maxwell.work,first.cycle.maxwellSteadyCycleWork);assert.ok(first.cycle.actualFirstCycle!.maxwell.energyChange!>0);near(periodic.query.maxwellTransientStress,0);
});
test('tiny cubic change keeps its negative sign while rounded stress change vanishes and linear values underflow',()=>{
 const s=continuumLabs[continuumConstitutiveKinds[0]].presets,tiny=viscoelasticLab(s[3].values),sub=viscoelasticLab(s[12].values),minimum=viscoelasticLab(s[13].values);assert.equal(tiny.query.roundedMaxwellStressChange,0);assert.ok(tiny.query.stableMaxwellStressChange!<0);assert.equal(sub.query.stableMaxwellStressChange,null);assert.equal(sub.query.stressChangeSign,-1);assert.ok(sub.query.logStressChangeMagnitude!<-1000);assert.ok(minimum.query.logStressChangeMagnitude!<-2000);assert.ok(minimum.smallChanges.some(s=>s.time===Number.MIN_VALUE));
});
test('zero-frequency, tiny-frequency, negative-power and reversed-interval presets retain explicit statuses',()=>{
 const s=continuumLabs[continuumConstitutiveKinds[0]].presets,zero=viscoelasticLab(s[5].values),fast=viscoelasticLab(s[6].values),far=viscoelasticLab(s[7].values),range=viscoelasticLab(s[8].values),negative=viscoelasticLab(s[11].values),empty=viscoelasticLab(s[14].values),reversed=viscoelasticLab(s[15].values);assert.equal(zero.cycle.periodStatus,'no-cycle');assert.equal(zero.query.maxwell.power,0);assert.equal(fast.query.homogeneousMemory,null);assert.equal(fast.query.memorySign,1);assert.equal(far.cycle.firstCycleStatus,'outside-teaching-time-horizon');assert.equal(range.cycle.periodStatus,'outside-arithmetic-range');assert.ok(negative.query.maxwell.power!<0);assert.ok(negative.query.maxwell.dissipation!>0);assert.equal(empty.window!.maxwell.work,0);assert.equal(reversed.windowStatus,'invalid-time-order');assert.equal(reversed.window,null);assert.ok(reversed.temporal.length>240);
});
test('time samples retain loading onset, rapid relaxation, query, interval boundaries and a complete first cycle when available',()=>{
 const r=viscoelasticLab(continuumLabs[continuumConstitutiveKinds[0]].presets[0].values);assert.equal(r.temporal[0].time,0);assert.equal(r.temporal.at(-1)!.time,r.horizon);assert.ok(r.horizon>=r.cycle.period!);for(const time of[r.time,r.start,r.end])assert.ok(r.temporal.some(s=>s.time===time));for(let i=1;i<r.temporal.length;i++)assert.ok(r.temporal[i].time>r.temporal[i-1].time);
});
test('worked neo-Hookean example and co-rotated observation preserve force, energy and physical face',()=>{
 const s=continuumLabs[continuumConstitutiveKinds[1]].presets,a=neoHookeanLab(s[0].values),b=neoHookeanLab(s[2].values),fixed=neoHookeanLab(s[5].values);near(a.query.energyDensity,2);near(a.query.cauchyStress,7);near(a.query.nominalStress,3.5);near(a.query.currentArea,1);near(a.query.force,7);near(a.query.storedEnergy,12);near(b.query.normalStress,a.query.normalStress!);near(b.query.signedShearStress,0);assert.notEqual(fixed.query.normalStress,b.query.normalStress);assert.notEqual(b.query.stress[0][0],a.query.stress[0][0]);
});
test('small-stretch, rescued-energy and symmetric-path presets retain stable magnitudes and physically signed work',()=>{
 const s=continuumLabs[continuumConstitutiveKinds[1]].presets,tiny=neoHookeanLab(s[3].values),rescued=neoHookeanLab(s[8].values),missing=neoHookeanLab(s[9].values),symmetric=neoHookeanLab(s[12].values),back=neoHookeanLab(s[13].values);assert.equal(tiny.query.stretch,1);assert.equal(tiny.query.roundedEnergyDensity,0);assert.ok(tiny.query.energyDensity!>0);assert.equal(rescued.query.energyDensity,null);assert.equal(rescued.query.storedEnergy,Number.MIN_VALUE);assert.equal(missing.query.energyDensity,null);assert.ok(missing.query.logEnergyDensity!<-900);assert.ok(symmetric.work.workDensity!>0);assert.equal(symmetric.work.workSign,1);near(symmetric.work.logWorkDensityMagnitude,symmetric.work.logEnergyChangeMagnitude!);assert.equal(back.work.workSign,-1);near(back.closedPathResidual,0);
});
test('neo-Hookean samples include selected stretches, both work endpoints and normal-angle coverage without changing parameters',()=>{
 const r=neoHookeanLab(continuumLabs[continuumConstitutiveKinds[1]].presets[11].values);for(const ell of[0,r.parameters.logStretch,r.start,r.end])assert.ok(r.response.some(s=>s.logStretch===ell));assert.equal(r.normals[0].normalAngle,-Math.PI);assert.equal(r.normals.at(-1)!.normalAngle,Math.PI);for(let i=1;i<r.response.length;i++)assert.ok(r.response[i].logStretch>r.response[i-1].logStretch);near(r.query.currentArea*r.query.currentLength,r.query.referenceVolume);
});
