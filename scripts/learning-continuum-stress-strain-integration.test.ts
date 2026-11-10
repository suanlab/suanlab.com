import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumStressStrainKinds,tractionStressLab,finiteStrainLab}from '../src/lib/learning-continuum-stress-strain-lab';
import {evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import {continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import {curriculumDomains}from '../src/data/learning-curriculum';
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const o=v as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
for(const[index,id]of continuumStressStrainKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: full bilingual registration, prerequisite links, fixed steps, guides and quizzes`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));
  assert.equal(unit.questions.length,2);const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);
  assert.ok(spec.axis.ko&&spec.axis.en&&spec.note.ko&&spec.note.en&&spec.legend.length>=2);for(const p of spec.presets)assert.ok(p.name.ko&&p.name.en);bilingual({unit,guide,spec});assert.equal(spec.controls.length,index===0?17:9);assert.equal(spec.presets.length,index===0?21:19);
 });
 test(`${id}: every preset produces finite or explicitly missing observations without mutating inputs`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.stressTraction||r.finiteDeformation);}
 });
 test(`${id}: missing, sparse, nonfinite and out-of-range controls fail rather than changing the experiment`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
  for(let i=0;i<v.length;i++){for(const bad of [NaN,Infinity,spec.controls[i].max+1]){const x=[...v];x[i]=bad;assert.throws(()=>evaluateContinuum(id,x),RangeError);}const sparse=[...v];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);}
 });
}
test('tilted-face and balanced/omitted-body-force guides preserve their worked physical values',()=>{
 const spec=continuumLabs[continuumStressStrainKinds[0]],q=tractionStressLab(spec.presets[0].values).query;assert.ok(Math.abs(q.normalStress!-2)<1e-14);assert.ok(Math.abs(q.signedShear!+2)<1e-14);assert.ok(Math.abs(q.force[0]!-4*Math.SQRT2)<1e-14);
 const balanced=tractionStressLab(spec.presets[1].values),omitted=tractionStressLab(spec.presets[2].values);assert.ok(Math.abs(balanced.balance.surfaceForce![0]!-12)<1e-13);assert.ok(Math.abs(balanced.balance.totalForce![0]!)<1e-13);assert.deepEqual(balanced.balance.surfaceForce,omitted.balance.surfaceForce);assert.ok(Math.abs(omitted.balance.totalForce![0]!-12)<1e-13);
});
test('isotropic directions, collapsed/unsupported geometry and tiny patch-force arithmetic remain distinct',()=>{
 const spec=continuumLabs[continuumStressStrainKinds[0]],iso=tractionStressLab(spec.presets[3].values);assert.equal(iso.query.principalDirection,null);assert.equal(iso.query.directionStatus,'nonunique-isotropic');
 assert.equal(tractionStressLab(spec.presets[8].values).balance.geometryStatus,'zero-area');const tiny=tractionStressLab(spec.presets[9].values);assert.equal(tiny.balance.geometryStatus,'below-supported-geometry');assert.equal(tiny.balance.surfaceForce,null);assert.ok(tiny.rectangle.halfWidth>0);
 const underflow=tractionStressLab(spec.presets[10].values);assert.equal(underflow.query.traction[0],Number.MIN_VALUE);assert.equal(underflow.query.force[0],null);
});
test('rotation, shear, skew and tiny-correction presets match the concept and guide narratives',()=>{
 const spec=continuumLabs[continuumStressStrainKinds[1]],rotation=finiteStrainLab(spec.presets[1].values),shear=finiteStrainLab(spec.presets[0].values),skew=finiteStrainLab(spec.presets[3].values),tiny=finiteStrainLab(spec.presets[6].values);
 assert.deepEqual(rotation.query.green,[[0,0],[0,0]]);assert.ok(Math.abs(rotation.query.infinitesimal[0][0]!+.5)<1e-14);assert.ok(Math.abs(shear.query.green[1][1]!-.08)<1e-14);assert.ok(Math.abs(shear.query.stableEngineeringStrain!-(Math.sqrt(1.16)-1))<1e-14);assert.equal(skew.control.greenScalar,.125);assert.ok(Math.abs(tiny.query.infinitesimal[0][0]!/(-5e-41)-1)<1e-14);assert.equal(tiny.query.F[0][0]-1,0);
});
test('grid observations and full direction/rotation/reference sweeps preserve physical maps and endpoints',()=>{
 const spec=continuumLabs[continuumStressStrainKinds[1]],r=finiteStrainLab(spec.presets[10].values);assert.equal(r.grid.length,12);assert.deepEqual(r.query.green,r.rotated.green);assert.ok(Math.abs(r.query.roundedStretch-r.rotated.roundedStretch)<1e-14);assert.equal(r.directions[0].angle,-Math.PI);assert.equal(r.directions.at(-1)!.angle,Math.PI);assert.equal(r.tinyRotations[0].logAngle,-30);assert.equal(r.tinyRotations.at(-1)!.logAngle,0);assert.equal(r.controls[0].omega,-1);assert.equal(r.controls.at(-1)!.omega,1);
});
