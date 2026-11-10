import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumFlowKinds,acousticFlowLab,taylorGreenFlowLab,acousticControlBounds,vortexControlBounds}from '../src/lib/learning-continuum-euler-navier-stokes-lab';
import {evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import {continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import {curriculumDomains}from '../src/data/learning-curriculum';
import references from './fixtures/continuum-euler-navier-stokes-independent.json';
const near=(a:number|null,b:number,tolerance=5e-10)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const o=v as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
for(const[index,id]of continuumFlowKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: bilingual registration, exact control bounds, fixed steps, guides and quizzes`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));
  for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));assert.equal(unit.questions.length,2);
  const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);
  bilingual({unit,guide,spec});assert.equal(spec.controls.length,index===0?13:11);assert.equal(spec.presets.length,16);
  assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?acousticControlBounds:vortexControlBounds);
  assert.ok(spec.controls.every(c=>c.step==='any'));assert.ok(spec.axis.ko&&spec.axis.en&&spec.legend.length>=2&&spec.note.ko&&spec.note.en);
 });
 test(`${id}: all presets yield finite or missing observations and preserve input arrays`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.acousticFlow||r.taylorGreenFlow);}
 });
 test(`${id}: missing, sparse, nonfinite and out-of-range controls are rejected`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
  for(let i=0;i<v.length;i++){for(const bad of [NaN,Infinity,spec.controls[i].max+1,spec.controls[i].min-1]){const x=[...v];x[i]=bad;assert.throws(()=>evaluateContinuum(id,x),RangeError);}const sparse=[...v];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);}
 });
}
test('acoustic fixed-window quadrature matches 24 independent differentiated physical-window integrals',()=>{
 for(const r of references.acousticCases){const p=r.parameters,q=acousticFlowLab([p.density,p.soundSpeed,p.meanVelocity,p.gamma,p.waveNumber,p.rightAmplitude,p.leftAmplitude,p.contactAmplitude,p.phase,r.time,r.x,.2,.9]);
  q.window.forEach((w,j)=>{near(w.rate,r.expected.windowRate[j],2e-8);near(w.netFlux,r.expected.windowFlux[j]);near(w.residual,0,2e-8);});near(q.energyWindow.residual,0,2e-8);
 }
});
test('sound, contact and stationary-characteristic guides preserve their physical narratives',()=>{
 const spec=continuumLabs[continuumFlowKinds[0]],right=acousticFlowLab(spec.presets[0].values),contact=acousticFlowLab(spec.presets[1].values),stationary=acousticFlowLab(spec.presets[2].values);
 near(right.query.characteristicSpeeds[0],2.5);near(right.query.velocity,.52);near(right.query.perturbation.pressure,.04);
 near(contact.query.perturbation.pressure,0);near(contact.query.perturbation.velocity,0);near(contact.query.acousticEnergy,0);assert.notEqual(contact.query.perturbation.density,0);
 assert.equal(stationary.characteristics[1].speed,0);assert.deepEqual(stationary.characteristics[1].points,[[0,0],[0,4]]);
});
test('half amplitude, tiny contact, underflowed energy and collapsed/unsupported windows stay distinct',()=>{
 const spec=continuumLabs[continuumFlowKinds[0]],mixed=acousticFlowLab(spec.presets[3].values),half=acousticFlowLab(spec.presets[4].values);
 near(mixed.query.nonlinearResidual.mass,4*half.query.nonlinearResidual.mass!);near(mixed.query.nonlinearResidual.pressure,4*half.query.nonlinearResidual.pressure!);
 assert.equal(acousticFlowLab(spec.presets[9].values).query.acousticEnergy,null);
 const tiny=acousticFlowLab(spec.presets[10].values);near(tiny.query.roundedContactVariable,0);assert.notEqual(tiny.query.contactVariable,0);
 const zero=acousticFlowLab(spec.presets[12].values);assert.equal(zero.windowStatus,'collapsed-zero-width');assert.equal(zero.window[0].rate,0);
 const small=acousticFlowLab(spec.presets[13].values);assert.equal(small.windowStatus,'below-supported-width');assert.equal(small.window[0].rate,null);assert.equal(small.window[0].netFlux,null);
 const v=[...spec.presets[0].values];v[5]=Number.MIN_VALUE;assert.equal(acousticFlowLab(v).halfAmplitude,null);
});
test('Taylor-Green worked acceleration, zero viscosity and pressure gauge use the same actual family',()=>{
 const spec=continuumLabs[continuumFlowKinds[1]],worked=taylorGreenFlowLab(spec.presets[0].values),euler=taylorGreenFlowLab(spec.presets[2].values),gauge=taylorGreenFlowLab(spec.presets[6].values);
 near(worked.query.velocity[0],-1);near(worked.query.velocity[1],1);near(worked.query.convection[0],-2);near(worked.query.timeDerivative[0],.2);
 for(let i=0;i<2;i++){near(worked.query.residual[i],0);near(worked.query.omittedPressureResidual[i],worked.query.convection[i]!);near(euler.query.viscousAcceleration[i],0);near(euler.query.timeDerivative[i],0);}
 assert.deepEqual(gauge.query.velocity,worked.query.velocity);assert.deepEqual(gauge.query.pressureGradient,worked.query.pressureGradient);near(gauge.query.pressure!-worked.query.pressure!,3);
});
test('vortex sweeps preserve physical geometry, streamline labels, time and original tiny changes',()=>{
 const spec=continuumLabs[continuumFlowKinds[1]],r=taylorGreenFlowLab(spec.presets[1].values),tiny=taylorGreenFlowLab(spec.presets[3].values),underflow=taylorGreenFlowLab(spec.presets[4].values),rounded=taylorGreenFlowLab(spec.presets[15].values);
 assert.equal(r.grid.length,81);assert.equal(r.profile.length,121);assert.equal(r.temporal[0].time,0);assert.equal(r.temporal.at(-1)!.time,4);near(r.grid[0].x,-r.cell.side/2);near(r.grid.at(-1)!.y,r.cell.side/2);
 assert.equal(r.particle.status,'integrated');assert.equal(r.particle.points.at(-1)!.time,r.time);near(r.initialLabel,r.endpointLabel,1e-10);assert.ok(r.endpointDifference!<1e-8);
 for(const contour of r.contours)for(const pt of contour.points)if(pt)near(Math.cos(r.parameters.waveNumber*pt[0])*Math.cos(r.parameters.waveNumber*pt[1]),contour.level);
 assert.equal(tiny.query.decay.roundedDecay,1);assert.notEqual(tiny.query.decay.stableDecayChange,0);assert.equal(underflow.query.decay.stableDecayChange,null);
 assert.ok(rounded.particle.roundedCoordinateUpdates>0);assert.equal(taylorGreenFlowLab(spec.presets[5].values).contours.length,0);
 assert.equal(r.tinyViscosities[0].logViscosity,-30);assert.equal(r.tinyViscosities.at(-1)!.logViscosity,0);
});
