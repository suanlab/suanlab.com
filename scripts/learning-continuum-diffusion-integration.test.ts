import test from 'node:test';
import assert from 'node:assert/strict';
import {continuumDiffusionKinds,heatDiffusionLab,gaussianDiffusionLab,heatControlBounds,gaussianControlBounds,roundedStandardNormalCdf}from '../src/lib/learning-continuum-diffusion-lab';
import {evaluateContinuum,isContinuumKind}from '../src/lib/learning-continuum';
import {continuumLabs,learningUnits,learningGuides}from '../src/data/interactive-book';
import {curriculumDomains}from '../src/data/learning-curriculum';
import refs from './fixtures/continuum-diffusion-independent.json';
const near=(a:number|null,b:number,tolerance=5e-10)=>{assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);};
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
function bilingual(v:unknown):void{if(!v||typeof v!=='object')return;const o=v as Record<string,unknown>;if('ko'in o&&'en'in o){assert.ok(typeof o.ko==='string'&&o.ko.length>0);assert.ok(typeof o.en==='string'&&o.en.length>0);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
for(const[index,id]of continuumDiffusionKinds.entries()){
 const spec=continuumLabs[id];
 test(`${id}: bilingual registration, exact bounds, fixed steps, guides and quizzes`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit&&isContinuumKind(id));assert.ok(curriculumDomains.some(d=>d.unitIds.includes(id)));for(const p of unit.prerequisiteIds)assert.ok(learningUnits.some(u=>u.id===p));
  assert.equal(unit.questions.length,2);const guide=learningGuides[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const e of guide.experiments)assert.ok(spec.presets[e.preset]);
  bilingual({unit,guide,spec});assert.equal(spec.controls.length,index===0?11:10);assert.equal(spec.presets.length,index===0?18:20);assert.deepEqual(spec.controls.map(c=>[c.min,c.max]),index===0?heatControlBounds:gaussianControlBounds);assert.ok(spec.controls.every(c=>c.step==='any'));assert.ok(spec.axis.ko&&spec.axis.en&&spec.legend.length>=2&&spec.note.ko&&spec.note.en);
 });
 test(`${id}: all presets yield finite or explicitly missing observations without mutation`,()=>{
  for(const p of spec.presets){const v=[...p.values],before=[...v],r=evaluateContinuum(id,v);assert.deepEqual(v,before);finiteTree(r);assert.ok(r.heatDiffusion||r.gaussianDiffusion);}
 });
 test(`${id}: sparse, missing, nonfinite and individually out-of-range inputs are rejected`,()=>{
  const v=spec.controls.map(c=>c.initial);assert.throws(()=>evaluateContinuum(id,v.slice(1)),RangeError);
  for(let i=0;i<v.length;i++){for(const bad of [NaN,Infinity,spec.controls[i].max+1,spec.controls[i].min-1]){const x=[...v];x[i]=bad;assert.throws(()=>evaluateContinuum(id,x),RangeError);}const sparse=[...v];delete sparse[i];assert.throws(()=>evaluateContinuum(id,sparse),RangeError);}
 });
}
test('worked heat and norm guides retain 120 W budget and distinct physical quantities',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[0]],worked=heatDiffusionLab(spec.presets[0].values),later=heatDiffusionLab(spec.presets[2].values),maximum=heatDiffusionLab(spec.presets[17].values);
 near(worked.query.heatFlux,30);near(worked.window.modeHeat,240);near(worked.window.energyRate,-120);near(worked.window.stableNetPower,120);near(worked.cell.quadraticNorm,900*Math.PI);
 near(worked.cell.sensibleEnergy,later.cell.sensibleEnergy);assert.ok(later.cell.quadraticNorm!<worked.cell.quadraticNorm!);near(maximum.query.temperature,330);near(maximum.query.heatFlux,0);
 const negative=heatDiffusionLab(spec.presets[16].values);assert.ok(negative.cell.sensibleEnergy<0);assert.ok(negative.query.temperatureLowerBound>0);
});
test('heat presets preserve tiny changes, rescued amplitude, zero width and coincident rounded endpoints',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[0]],tiny=heatDiffusionLab(spec.presets[3].values),subrange=heatDiffusionLab(spec.presets[9].values),rescued=heatDiffusionLab(spec.presets[11].values);
 assert.equal(tiny.query.decayFactor,1);assert.notEqual(tiny.query.amplitudeChange,0);assert.equal(subrange.query.stableRelativeDecayChange,null);
 assert.equal(rescued.query.decayFactor,0);assert.equal(rescued.query.amplitude,Number.MIN_VALUE);
 assert.equal(heatDiffusionLab(spec.presets[12].values).window.geometryStatus,'zero-width');const endpoints=heatDiffusionLab(spec.presets[13].values);assert.equal(endpoints.window.left,endpoints.window.right);assert.notEqual(endpoints.window.modeHeat,0);assert.equal(endpoints.window.roundedFaceNetPower,0);
});
test('heat rate and original time sweeps retain supported domains and quadratic wave-number scaling',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[0]],r=heatDiffusionLab(spec.presets[1].values);assert.equal(r.temporal[0].time,0);assert.equal(r.temporal.at(-1)!.time,4);assert.equal(r.waveNumbers[0].waveNumber,.25);assert.equal(r.waveNumbers.at(-1)!.waveNumber,4);
 for(const point of r.waveNumbers)near(point.rate,r.parameters.conductivity/r.parameters.heatCapacity*point.waveNumber**2);assert.equal(r.smallConductivities[0].logConductivity,-30);assert.equal(r.smallConductivities.at(-1)!.logConductivity,0);
 for(const s of r.spatial)assert.ok(Math.abs(s.x)<=16);assert.equal(r.spatial[0].x,-r.radius);assert.equal(r.spatial.at(-1)!.x,r.radius);
});
test('Gaussian hand calculations, actual window exchange and rounded CDF loss match guide claims',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[1]],symmetric=gaussianDiffusionLab(spec.presets[0].values),moving=gaussianDiffusionLab(spec.presets[1].values),incoming=gaussianDiffusionLab(spec.presets[2].values),tail=gaussianDiffusionLab(spec.presets[3].values);
 near(symmetric.state.variance,.5);near(symmetric.peak,2/Math.sqrt(Math.PI));near(moving.state.mean,.5);assert.equal(symmetric.parameters.mass,2);
 assert.ok(incoming.window!.massRate!>0);assert.ok(incoming.window!.netOutflow!<0);near(incoming.window!.balanceResidual,0);
 assert.equal(tail.leftCdf,1);assert.equal(tail.rightCdf,1);assert.equal(tail.cdfDifference,0);assert.ok(tail.window!.fraction!>0);assert.ok(tail.window!.massInside!>0);
});
test('Gaussian tail presets match six independent physical-width log quadratures and rescued mass',()=>{
 for(const r of refs.tailCases){const p=r.parameters,q=gaussianDiffusionLab([p.mass,p.area,p.initialSigma,p.initialMean,p.velocity,p.diffusivity,0,r.left,r.left,r.right]);near(q.window!.logFraction,Number(r.expected.logFraction),2e-12);}
 const spec=continuumLabs[continuumDiffusionKinds[1]],rescued=gaussianDiffusionLab(spec.presets[9].values),derivative=gaussianDiffusionLab(spec.presets[15].values);
 assert.equal(rescued.window!.fraction,null);assert.equal(rescued.window!.massInside,2*Number.MIN_VALUE);assert.equal(derivative.query.concentration,null);assert.ok(derivative.query.spatialDerivative!<0);
});
test('invalid window order preserves fields and inputs, while zero mass keeps normalized moments missing',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[1]];
 for(const index of [11,12]){const v=[...spec.presets[index].values],r=gaussianDiffusionLab(v);assert.equal(r.windowStatus,'invalid-boundary-order');assert.equal(r.window,null);assert.equal(r.cdfDifference,null);assert.ok(r.query.concentration!>0);assert.equal(r.left,v[8]);assert.equal(r.right,v[9]);}
 const empty=gaussianDiffusionLab(spec.presets[5].values);assert.equal(empty.query.concentration,0);assert.equal(empty.state.massWeightedMean,null);assert.equal(empty.state.massWeightedVariance,null);assert.equal(empty.window!.massInside,0);
 const tiny=gaussianDiffusionLab(spec.presets[6].values);assert.equal(tiny.state.roundedVarianceChange,0);assert.notEqual(tiny.state.varianceChange,0);
});
test('Gaussian plot samples preserve peaks and initial center within a finite display without renormalization',()=>{
 const spec=continuumLabs[continuumDiffusionKinds[1]],r=gaussianDiffusionLab(spec.presets[16].values);assert.ok(r.spatial.some(s=>s.x===r.state.mean));assert.ok(r.spatial.some(s=>s.x===r.parameters.initialMean));
 for(let i=0;i<r.spatial.length;i++){assert.ok(r.spatial[i].x>=-16&&r.spatial[i].x<=16);if(i>0)assert.ok(r.spatial[i].x>r.spatial[i-1].x);}assert.equal(r.parameters.mass,2);assert.equal(r.maximum,16);
});
test('rounded CDF control matches independent mpmath70 erfc values including negative tails',()=>{
 const table=[[-12,1.776482112077679e-33],[-9,1.1285884059538405e-19],[-8,6.220960574271784e-16],[-2,.02275013194817921],[-1,.15865525393145705],[-.5,.3085375387259869],[0,.5],[.5,.6914624612740131],[1,.8413447460685429],[2,.9772498680518208],[8,.9999999999999993],[9,1],[12,1]];
 for(const[z,expected]of table){const value=roundedStandardNormalCdf(z);if(expected<1e-10)assert.ok(Math.abs(value/expected-1)<2e-12);else near(value,expected,2e-15);}assert.throws(()=>roundedStandardNormalCdf(Infinity),RangeError);
});
