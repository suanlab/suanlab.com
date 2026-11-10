import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-turbulence-registry-hashes.json';
import { continuumTurbulenceKinds, reynoldsLab, gaussianFilterLab, reynoldsBounds, gaussianFilterBounds } from '../src/lib/learning-continuum-turbulence-lab';
import { isContinuumKind, evaluateContinuum } from '../src/lib/learning-continuum';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { reynoldsEnsembleAt, reynoldsInstantaneousAt } from '../src/lib/learning-continuum-turbulence';
import { gaussianBurgersAt } from '../src/lib/learning-continuum-gaussian-filter';
import reynoldsRefs from './fixtures/continuum-reynolds-independent.json';
import gaussianRefs from './fixtures/continuum-gaussian-independent.json';
const { learningUnits, learningGuides, continuumLabs } = data;
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const near=(a:number|null,b:number,tol=5e-9)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(Math.abs(b),1e-280)+Number.MIN_VALUE,`${a} versus ${b}`);};
function finite(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finite);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const r=v as Record<string,unknown>;if('ko'in r&&'en'in r){assert.ok(typeof r.ko==='string'&&r.ko.length>0);assert.ok(typeof r.en==='string'&&r.en.length>0);assert.doesNotMatch(r.en as string,/[가-힣]/);}Object.values(r).forEach(bilingual);}
test('498 prior units/guides and43 lab groups489 entries retain canonical content and order',()=>{
  const ids=new Set(prior.ids),units=learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,498);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);
  assert.equal(hash(Object.fromEntries(Object.entries(learningGuides).filter(([id])=>ids.has(id as typeof units[number]['id'])))),prior.guides);
  const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(key=>[key,(data as Record<string,any>)[name][key]]))]));assert.equal(hash(labs),prior.labs);assert.equal(Object.keys(labs).length,43);assert.equal(Object.values(prior.labEntries).reduce((sum,v)=>sum+v.length,0),489);
});
for(const [index,id]of continuumTurbulenceKinds.entries()){
  const spec=continuumLabs[id];
  test(`${id}: bilingual registration connects concepts, guides, quizzes and actual prerequisites`,()=>{
    const u=learningUnits.find(u=>u.id===id)!;assert.ok(u&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));u.prerequisiteIds.forEach(id=>assert.ok(learningUnits.some(u=>u.id===id)));
    assert.equal(u.questions.length,2);assert.equal(learningGuides[id].steps.length,4);assert.equal(learningGuides[id].experiments.length,3);learningGuides[id].experiments.forEach(e=>assert.ok(spec.presets[e.preset]));
    assert.equal(spec.presets.length,20);assert.equal(spec.controls.length,index===0?13:11);assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?reynoldsBounds:gaussianFilterBounds);assert.ok(spec.controls.every(c=>c.step==='any'));bilingual({unit:u,guide:learningGuides[id],spec});
  });
  test(`${id}: every preset preserves caller data and returns finite or explicitly unavailable observations`,()=>{
    for(const p of spec.presets){const v=[...p.values],before=[...v],result=evaluateContinuum(id,v);assert.deepEqual(v,before);finite(result);assert.ok(result.reynolds||result.gaussianFilter);}
  });
  test(`${id}: missing, sparse, nonfinite and out-of-range controls throw`,()=>{
    const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
    for(let i=0;i<v.length;i++){for(const bad of[NaN,Infinity,spec.controls[i].min-1,spec.controls[i].max+1]){const a=[...v];a[i]=bad;assert.throws(()=>evaluateContinuum(id,a),RangeError);}const a=[...v];delete a[i];assert.throws(()=>evaluateContinuum(id,a),RangeError);}
  });
}
test('Reynolds adapter agrees with24 independently averaged physical-force and spatial-integral references',()=>{
  for(const row of reynoldsRefs.ordinaryCases){const p=row.parameters,r=reynoldsLab([p.height,p.wavenumber,p.spanWidth,p.density,p.dynamicViscosity,p.bulkVelocity,p.meanAsymmetry,p.firstStreamfunctionAmplitude,p.secondStreamfunctionAmplitude,p.candidateEddyViscosity,row.xi,row.xCycles,row.phaseCycles]);
    for(const key of['covarianceUU','covarianceVV','covarianceUV','meanForceX','meanForceY']as const)near(r.query[key].value,Number(row.expected[key].value));for(const key of['u','v','forceX','forceY']as const){const expected=Number(row.instantaneous[key].value);if(expected!==0||row.instantaneous[key].sign===0)near(r.point[key].value,expected);}
    for(const key of['meanKineticEnergy','fluctuationKineticEnergy','molecularMeanLoss','meanToFluctuationTransfer','meanForcingWork']as const)near(r.budget[key].value,Number(row.budgets[key].value));
  }
});
test('Gaussian adapter agrees with24 independent filtering, force, time and energy-budget references',()=>{
  for(const row of gaussianRefs.ordinaryCases){const p=row.parameters,r=gaussianFilterLab([p.wavenumber,p.filterWidth,p.kinematicViscosity,p.initialFirstAmplitude,p.initialSecondAmplitude,p.density,p.area,p.frameVelocity,row.time,row.xCycles,row.phaseCycles]);
    for(const key of['filteredVelocity','variance','varianceGradient','filteredForce','filteredTimeRate','filteredResidualVelocity']as const)near(r.query[key].value,Number(row.expected[key].value));for(const key of['totalKineticEnergy','resolvedKineticEnergy','residualKineticEnergy','resolvedForcingWork','resolvedToResidualTransfer','resolvedEnergyRate']as const)near(r.budget[key].value,Number(row.budgets[key].value));
  }
});
const rv=continuumLabs[continuumTurbulenceKinds[0]].controls.map(c=>c.initial),gv=continuumLabs[continuumTurbulenceKinds[1]].controls.map(c=>c.initial);
const edit=(v:number[],i:number,x:number)=>{const a=[...v];a[i]=x;return a;};
test('all13 Reynolds controls bind to actual geometry, physics, covariance candidate or phase observation',()=>{
  const base=reynoldsLab(rv);near(reynoldsLab(edit(rv,0,4)).budget.cellVolume,2*base.budget.cellVolume);near(reynoldsLab(edit(rv,1,3)).budget.cellLength,base.budget.cellLength/2);near(reynoldsLab(edit(rv,2,2)).budget.meanForcingWork.value,2*base.budget.meanForcingWork.value!);near(reynoldsLab(edit(rv,3,2)).query.reynoldsShearStress.value,2*base.query.reynoldsShearStress.value!);
  assert.equal(reynoldsLab(edit(rv,4,0)).query.molecularMeanLossDensity.sign,0);assert.equal(reynoldsLab(edit(rv,5,0)).query.meanU.sign,0);assert.notEqual(reynoldsLab(edit(rv,6,0)).query.meanU.value,base.query.meanU.value);assert.equal(reynoldsLab(edit(rv,7,0)).query.covarianceUV.sign,0);assert.equal(reynoldsLab(edit(rv,8,0)).query.covarianceUV.sign,0);assert.equal(reynoldsLab(edit(rv,9,0)).query.modeledCovarianceUV.sign,0);assert.equal(reynoldsLab(edit(rv,10,0)).point.u.sign,0);
  assert.notEqual(reynoldsLab(edit(rv,11,.25)).point.uPrime.value,base.point.uPrime.value);assert.notEqual(reynoldsLab(edit(rv,12,.25)).point.uPrime.value,base.point.uPrime.value);
});
test('all11 Gaussian controls bind to filter, diffusion, modes, reference weights or moving observations',()=>{
  const base=gaussianFilterLab(gv);near(gaussianFilterLab(edit(gv,0,2)).budget.cellLength,base.budget.cellLength/2);assert.equal(gaussianFilterLab(edit(gv,1,0)).query.variance.sign,0);assert.equal(gaussianFilterLab(edit(gv,2,0)).budget.resolvedMolecularLoss.sign,0);assert.equal(gaussianFilterLab(edit(gv,3,0)).budget.resolvedToResidualTransfer.sign,0);assert.equal(gaussianFilterLab(edit(gv,4,0)).budget.resolvedToResidualTransfer.sign,0);near(gaussianFilterLab(edit(gv,5,2)).budget.resolvedKineticEnergy.value,2*base.budget.resolvedKineticEnergy.value!);near(gaussianFilterLab(edit(gv,6,2)).budget.resolvedKineticEnergy.value,2*base.budget.resolvedKineticEnergy.value!);
  assert.notEqual(gaussianFilterLab(edit(gv,7,1)).query.filteredTimeRate.value,base.query.filteredTimeRate.value);assert.notEqual(gaussianFilterLab(edit(gv,8,1)).query.firstAmplitude.value,base.query.firstAmplitude.value);assert.notEqual(gaussianFilterLab(edit(gv,9,.25)).query.variance.value,base.query.variance.value);assert.notEqual(gaussianFilterLab(edit(gv,10,.25)).query.variance.value,base.query.variance.value);
});
test('all plot sweeps retain exact endpoints, actual queries, phase samples and independent quiet comparison',()=>{
  const r=reynoldsLab(edit(rv,10,Number.MIN_VALUE));assert.equal(r.spatial[0].xi,0);assert.equal(r.spatial.at(-1)!.xi,1);assert.ok(r.spatial.some(s=>s.xi===Number.MIN_VALUE));assert.ok(r.small.some(s=>s.xi===Number.MIN_VALUE));assert.deepEqual(r.ensemble.map(s=>s.phase),Array.from({length:8},(_,j)=>j/8));assert.equal(r.phases[0].phase,0);assert.equal(r.phases.at(-1)!.phase,1);
  const point=r.spatial.find(s=>s.xi===Number.MIN_VALUE)!;assert.deepEqual(point.value,reynoldsEnsembleAt(r.parameters,point.xi));assert.deepEqual(point.point,reynoldsInstantaneousAt(r.parameters,point.xi,r.xCycles,r.phaseCycles));assert.deepEqual(point.quiet.meanU,point.value.meanU);assert.equal(point.quiet.covarianceUV.sign,0);
  const g=gaussianFilterLab(edit(edit(edit(gv,1,Number.MIN_VALUE),8,Number.MIN_VALUE),9,Number.MIN_VALUE));assert.ok(g.spatial.some(s=>s.xCycles===Number.MIN_VALUE));assert.ok(g.temporal.some(s=>s.time===Number.MIN_VALUE));assert.ok(g.widths.some(s=>s.width===Number.MIN_VALUE));assert.ok(g.small.some(s=>s.width===Number.MIN_VALUE));assert.equal(g.widths.at(-1)!.width,4);assert.deepEqual(g.query,gaussianBurgersAt(g.parameters,g.time,g.xCycles,g.phaseCycles));
});
test('singular, indeterminate and defined-but-unavailable viscosities remain distinct in actual presets',()=>{
  const spec=continuumLabs[continuumTurbulenceKinds[0]],singular=reynoldsLab(spec.presets[2].values),indeterminate=reynoldsLab(spec.presets[4].values),large=reynoldsLab(spec.presets[16].values);
  assert.equal(singular.query.inferredEddyViscosityStatus,'nonzero-stress-at-zero-shear');assert.equal(indeterminate.query.inferredEddyViscosityStatus,'indeterminate');assert.equal(large.query.inferredEddyViscosityStatus,'defined');assert.equal(large.query.inferredEddyViscosity!.value,null);assert.ok(large.query.inferredEddyViscosity!.logMagnitude!>700);
  const wall=reynoldsLab(spec.presets[15].values);assert.equal(wall.point.uPrime.sign,1);assert.equal(wall.point.uPrime.value,null);
});
test('actual Gaussian presets retain fourth/sixth-order minima, forcing tails and forced decay balance',()=>{
  const spec=continuumLabs[continuumTurbulenceKinds[1]],six=gaussianFilterLab(spec.presets[11].values),four=gaussianFilterLab(spec.presets[12].values),tail=gaussianFilterLab(spec.presets[16].values),base=gaussianFilterLab(gv);
  assert.equal(six.query.variance.sign,1);assert.equal(six.query.variance.value,null);assert.ok(six.query.variance.logMagnitude! < -4400);assert.ok(four.query.variance.logMagnitude! > six.query.variance.logMagnitude!);assert.notEqual(tail.query.filteredForce.sign,0);near(base.budget.resolvedForcingWork.value,base.budget.resolvedToResidualTransfer.value!);near(base.budget.resolvedEnergyRate.value,-base.budget.resolvedMolecularLoss.value!);
});

test('viscosity curves never bridge a zero-shear pole between samples, even when its coordinate is not in the grid',()=>{
  const regular=reynoldsLab(rv);assert.equal(regular.viscositySegments.length,2);
  for(const c of[-1,-.25,0,.25,1,Number.MIN_VALUE]){const r=reynoldsLab(edit(rv,6,c));for(const branch of r.viscositySegments){assert.ok(branch.every(s=>s.value.inferredEddyViscosity?.value!=null));assert.equal(new Set(branch.map(s=>s.value.meanShear.sign)).size,1);}}
  const quiet=reynoldsLab(edit(rv,5,0));assert.equal(quiet.viscositySegments.length,0);
});
