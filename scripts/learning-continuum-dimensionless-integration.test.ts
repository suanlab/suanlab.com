import test from 'node:test';
import assert from 'node:assert/strict';
import { continuumDimensionlessKinds, pecletLab, damkohlerLab, pecletBounds, damkohlerBounds } from '../src/lib/learning-continuum-dimensionless-lab';
import { evaluateContinuum, isContinuumKind } from '../src/lib/learning-continuum';
import { continuumLabs, learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import refs from './fixtures/continuum-dimensionless-independent.json';
const near=(a:number|null,b:number,tolerance=3e-9)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(Math.abs(b),1e-290),`${a} != ${b}`);};
function finiteTree(v:unknown):void { if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree); }
function bilingual(v:unknown):void {
  if(!v||typeof v!=='object')return;const item=v as Record<string,unknown>;
  if('ko' in item&&'en' in item){assert.ok(typeof item.ko==='string'&&item.ko.length>0);assert.ok(typeof item.en==='string'&&item.en.length>0);assert.doesNotMatch(item.en as string,/[가-힣]/);}
  Object.values(item).forEach(bilingual);
}
for(const [index,id] of continuumDimensionlessKinds.entries()) {
  const spec=continuumLabs[id];
  test(`${id}: bilingual registration connects four fixed concepts, three experiments, two quizzes and actual prerequisites`,()=>{
    const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));
    for(const prerequisite of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===prerequisite));
    const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
    guide.experiments.forEach(e=>assert.ok(spec.presets[e.preset]));assert.equal(spec.controls.length,8);assert.equal(spec.presets.length,18);
    assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?pecletBounds:damkohlerBounds);assert.ok(spec.controls.every(c=>c.step==='any'));
    bilingual({unit,guide,spec});
  });
  test(`${id}: all18 presets keep caller inputs and finite or explicitly unavailable observations`,()=>{
    for(const p of spec.presets){const values=[...p.values],before=[...values],r=evaluateContinuum(id,values);assert.deepEqual(values,before);finiteTree(r);assert.ok(r.peclet||r.damkohler);}
  });
  test(`${id}: wrong-length, sparse, nonfinite and out-of-range arrays are rejected`,()=>{
    const values=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,values.slice(1)),RangeError);
    for(let i=0;i<values.length;i++){
      for(const bad of[NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const a=[...values];a[i]=bad;assert.throws(()=>evaluateContinuum(id,a),RangeError);}
      const sparse=[...values];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);
    }
  });
}
test('Peclet adapter retains20 independent physical-space fields, gradients, fluxes and clocks',()=>{
  for(const row of refs.pecletCases){const p=row.parameters,r=pecletLab([p.velocity,p.diffusivity,p.length,p.leftConcentration,p.rightConcentration,p.area,row.xi,1]);for(const key of['concentration','gradient','curvature','totalFlux','totalTransport','diffusionTime'] as const)near(r.query[key].value,Number(row.expected[key].value));}
});
test('Damkohler adapter retains20 independent fields, partial integrals, storage and uptake',()=>{
  for(const row of refs.damkohlerCases){const p=row.parameters,r=damkohlerLab([p.diffusivity,p.length,p.reactionRate,p.surfaceConcentration,p.area,row.xi,1,1]);for(const key of['concentration','gradient','storedMass','totalConsumption','partialStoredMass','partialConsumption'] as const)near(r.query[key].value,Number(row.expected[key].value));}
});
const pe=continuumLabs[continuumDimensionlessKinds[0]].controls.map(c=>c.initial),da=continuumLabs[continuumDimensionlessKinds[1]].controls.map(c=>c.initial);
function change(v:number[],index:number,value:number){const result=[...v];result[index]=value;return result;}
test('all eight Peclet inputs change their physical or similarity quantities with the declared dimensions',()=>{
  const b=pecletLab(pe);
  near(pecletLab(change(pe,0,1)).query.peclet.value,2*b.query.peclet.value!);
  near(pecletLab(change(pe,1,.2)).query.diffusionTime.value,b.query.diffusionTime.value!/2);
  near(pecletLab(change(pe,2,1)).query.position.value,.5);
  const left=pecletLab(change(pe,3,1.1)),right=pecletLab(change(pe,4,.3));
  near(left.query.concentration.value!-b.query.concentration.value!,.1*b.query.leftWeight.value!);
  near(right.query.concentration.value!-b.query.concentration.value!,.1*b.query.basisShape.value!);
  near(pecletLab(change(pe,5,3)).query.totalTransport.value,2*b.query.totalTransport.value!);
  near(pecletLab(change(pe,6,.25)).query.position.value,.5);
  near(pecletLab(change(pe,7,1)).comparison!.diffusionTime.value,b.query.diffusionTime.value!);
});
test('all eight Damkohler inputs retain half-thickness, actual area, window and similarity semantics',()=>{
  const b=damkohlerLab(da);
  near(damkohlerLab(change(da,0,.5)).query.diffusionTime.value,b.query.diffusionTime.value!/2);
  near(damkohlerLab(change(da,1,2)).query.damkohler.value,4*b.query.damkohler.value!);
  near(damkohlerLab(change(da,2,2)).query.damkohler.value,2*b.query.damkohler.value!);
  near(damkohlerLab(change(da,3,4)).query.totalConsumption.value,2*b.query.totalConsumption.value!);
  near(damkohlerLab(change(da,4,3)).query.storedMass.value,2*b.query.storedMass.value!);
  assert.ok(damkohlerLab(change(da,5,.75)).query.partialStoredMass.value!>b.query.partialStoredMass.value!);
  near(damkohlerLab(change(da,6,1)).comparison!.storedMass.value,b.query.storedMass.value!);
  near(damkohlerLab(change(da,7,1)).comparison!.totalConsumption.value,b.query.totalConsumption.value!*1.5);
});
test('adaptive coordinates retain exact endpoints, query and resolve both signed Pe80000 layers',()=>{
  for(const index of[10,11]){
    const r=pecletLab(continuumLabs[continuumDimensionlessKinds[0]].presets[index].values);
    assert.equal(r.spatial[0].xi,0);assert.equal(r.spatial.at(-1)!.xi,1);assert.ok(r.spatial.some(s=>s.xi===r.xi));
    const inner=r.parameters.velocity>0?r.spatial.find(s=>s.xi<1&&s.xi>1-.00001):r.spatial.find(s=>s.xi>0&&s.xi<.00001);assert.ok(inner);
    assert.ok(inner.value.concentration.value!>.1&&inner.value.concentration.value!<1);
  }
});
test('large-phi coordinates resolve the inlet layer and keep a nonzero sealed concentration log',()=>{
  const r=damkohlerLab(continuumLabs[continuumDimensionlessKinds[1]].presets[9].values);
  assert.ok(r.spatial.some(s=>s.xi>0&&s.xi<1/Math.sqrt(128*16/.0001)));
  assert.equal(r.sealed.diffusiveFlux.sign,0);assert.equal(r.sealed.concentration.sign,1);assert.equal(r.sealed.concentration.value,null);assert.ok(r.sealed.concentration.logMagnitude!<-4000);
});
test('similarity preserves groups and basis while clocks, area transport, inventory and rates scale',()=>{
  const p=pecletLab(pe),d=damkohlerLab(da);
  near(p.comparison!.peclet.value,p.query.peclet.value!);near(p.comparison!.basisShape.value,p.query.basisShape.value!);
  near(p.comparison!.totalTransport.value,p.scale*p.query.totalTransport.value!);near(p.comparison!.diffusionTime.value,p.scale**2*p.query.diffusionTime.value!);
  near(d.comparison!.damkohler.value,d.query.damkohler.value!);near(d.comparison!.storedMass.value,d.lengthScale**3*d.query.storedMass.value!);
  near(d.comparison!.totalConsumption.value,d.lengthScale*d.diffusivityScale*d.query.totalConsumption.value!);
});
test('unavailable comparisons never clamp coefficients or turn nonzero rates into zero reactions',()=>{
  const p=pecletLab(change(pe,2,4));assert.equal(p.comparison,null);assert.equal(p.comparisonParameters.length,6);assert.equal(p.parameters.length,4);
  const tiny=pecletLab([Number.MIN_VALUE,4,.25,0,1,1,.5,1.5]);assert.equal(tiny.comparison,null);assert.equal(tiny.comparisonStatus,'nonzero-coefficient-outside-arithmetic-range');
  const d=damkohlerLab([4,.25,Number.MIN_VALUE,2,1,.5,2,1]);assert.equal(d.comparison,null);assert.equal(d.comparisonStatus,'nonzero-coefficient-outside-arithmetic-range');assert.equal(d.query.totalConsumption.sign,1);
});
test('partial integrals preserve zero windows, whole-slab limits and a tiny nonzero observation window',()=>{
  const a=damkohlerLab(change(da,5,0)),b=damkohlerLab(change(da,5,1));
  assert.equal(a.query.partialStoredMass.sign,0);assert.equal(a.query.partialConsumption.sign,0);
  near(b.query.partialStoredMass.value,b.query.storedMass.value!);near(b.query.partialConsumption.value,b.query.totalConsumption.value!);
  const tiny=damkohlerLab([4,.25,Number.MIN_VALUE,2,1,Number.MIN_VALUE,1,1]);assert.equal(tiny.query.partialStoredMass.sign,1);assert.equal(tiny.query.partialConsumption.sign,1);assert.equal(tiny.query.partialConsumption.value,null);assert.ok(tiny.query.partialConsumption.logMagnitude!<-1400);
});
test('zero mechanisms and zero normalization denominators remain distinct from positive unavailable tails',()=>{
  const p=pecletLab(change(pe,0,0));assert.equal(p.query.advectionTime,null);assert.equal(p.query.peclet.sign,0);
  const contrast=pecletLab([.5,.1,2,1,1,1,.5,1]);assert.equal(contrast.query.observedNormalizedShape,null);assert.notEqual(contrast.query.basisShape.value,null);
  const d=damkohlerLab(change(da,2,0));assert.equal(d.query.reactionTime,null);assert.equal(d.query.observedEffectiveness,null);assert.equal(d.query.analyticEffectiveness.value,1);assert.ok(d.query.storedMass.value!>0);
});
