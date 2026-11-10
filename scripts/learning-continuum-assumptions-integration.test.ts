import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumAssumptionKinds,particleAssumptionLab,scaleAssumptionLab} from '../src/lib/learning-continuum-assumptions-lab';
import {evaluateContinuum,isContinuumKind} from '../src/lib/learning-continuum';
import {continuumLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
function finiteTree(value:unknown):void{if(typeof value==='number')assert.ok(Number.isFinite(value));else if(value&&typeof value==='object')Object.values(value).forEach(finiteTree);}
function translated(value:unknown):void{if(!value||typeof value!=='object')return;const o=value as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(translated);}
for(const[index,id]of continuumAssumptionKinds.entries()){
  const spec=continuumLabs[id];
  test(`${id}: bilingual registered theory, prerequisites, four steps, three guides and two quizzes`,()=>{
    const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));
    for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));assert.equal(unit.questions.length,2);
    const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);
    assert.ok(spec.axis.ko&&spec.axis.en&&spec.note.ko&&spec.note.en);assert.ok(spec.legend.length>=2);
    for(const preset of spec.presets){assert.ok(preset.name.ko&&preset.name.en);}
    translated({unit,guide,spec});assert.equal(spec.controls.length,index===0?11:9);assert.equal(spec.presets.length,index===0?17:16);
  });
  test(`${id}: every preset preserves inputs and produces only finite or explicitly missing observations`,()=>{
    for(const preset of spec.presets){const values=[...preset.values],before=[...values],r=evaluateContinuum(id,values);assert.deepEqual(values,before);finiteTree(r);assert.ok(r.particleAveraging||r.scaleSeparation);}
  });
  test(`${id}: malformed, sparse, nonfinite, out-of-range and fractional selector controls fail`,()=>{
    const values=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,values.slice(1)),RangeError);
    for(let i=0;i<values.length;i++){for(const bad of [NaN,Infinity,spec.controls[i].max+1]){const v=[...values];v[i]=bad;assert.throws(()=>evaluateContinuum(id,v),RangeError);}const sparse=[...values];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);}
    const v=[...values];v[index===0?10:7]=1.5;assert.throws(()=>evaluateContinuum(id,v),RangeError);
  });
}
test('particle presets agree with opposing-stream, weighted-mass and single stationary-particle guide readouts',()=>{
  const spec=continuumLabs[continuumAssumptionKinds[0]],worked=particleAssumptionLab(spec.presets[0].values);assert.equal(worked.query.density,15/8);assert.equal(worked.query.velocity,0);assert.equal(worked.query.dispersion,15/8);assert.equal(worked.interval.mass,203/128);
  assert.equal(particleAssumptionLab(spec.presets[2].values).query.velocity,-.5);
  const stationary=particleAssumptionLab(spec.presets[6].values);assert.equal(stationary.totalMass,1);assert.equal(stationary.query.smoothingRate,-15/64);assert.equal(stationary.query.flux,0);
});
test('empty supports, zero windows and unsupported positive windows retain their different meanings',()=>{
  const spec=continuumLabs[continuumAssumptionKinds[0]];
  const empty=particleAssumptionLab(spec.presets[3].values);assert.equal(empty.query.velocity,null);assert.equal(empty.query.dispersion,null);
  const zero=particleAssumptionLab(spec.presets[7].values);assert.equal(zero.widthStatus,'zero-length');assert.equal(zero.interval.mass,0);assert.equal(zero.interval.meanDensity,null);
  const tiny=particleAssumptionLab(spec.presets[8].values);assert.equal(tiny.widthStatus,'unsupported-positive-width');assert.equal(tiny.interval.mass,null);assert.ok(tiny.halfWidth>0);assert.ok(tiny.temporal.every(p=>p.mass===null));
});
test('adaptive spatial coordinates retain each compact support edge and center at the narrowest width',()=>{
  const v=continuumLabs[continuumAssumptionKinds[0]].controls.map(c=>c.initial);v[4]=.02;v[1]=1.23456789;v[6]=.371234567;
  const r=particleAssumptionLab(v),xs=r.spatial.map(p=>p.x);assert.equal(xs[0],-6);assert.equal(xs.at(-1),6);assert.ok(xs.length>=210);
  for(const p of r.query.contributions)for(const target of [p.position-r.h,p.position,p.position+r.h])assert.ok(xs.some(x=>Math.abs(x-target)<1e-12));
  assert.equal(r.temporal[0].time,-1);assert.equal(r.temporal.at(-1)!.time,1);
});
test('scale presets distinguish interval existence, selected-width failure, empty and full reservoir observations',()=>{
  const spec=continuumLabs[continuumAssumptionKinds[1]],worked=scaleAssumptionLab(spec.presets[0].values);assert.ok(worked.query.windowExists&&worked.query.selectedSatisfies);assert.ok(Math.abs(worked.query.expectedCount-1e9)<1e-5);
  assert.equal(scaleAssumptionLab(spec.presets[1].values).query.windowExists,false);const small=scaleAssumptionLab(spec.presets[2].values);assert.equal(small.query.windowExists,true);assert.equal(small.query.selectedSatisfies,false);
  assert.equal(scaleAssumptionLab(spec.presets[6].values).count.relativeStd,null);assert.equal(scaleAssumptionLab(spec.presets[7].values).count.relativeStd,0);
  assert.equal(worked.lengths[0].logLength,-12);assert.equal(worked.lengths.at(-1)!.logLength,6);assert.equal(worked.fractions[0].value.relativeStd,null);
});
