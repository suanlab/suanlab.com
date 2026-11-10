import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-conservative-registry-hashes.json';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {continuumConservativeKinds,volumeDiscretizationLab,godunovDiscretizationLab,conservativeScaledObservation,volumeControlBounds,godunovControlBounds} from '../src/lib/learning-continuum-conservative-lab';
import {evaluateContinuum,isContinuumKind} from '../src/lib/learning-continuum';
import {nonuniformVolumeSimulation} from '../src/lib/learning-continuum-volume-discretization';
import {burgersRiemannSimulation} from '../src/lib/learning-continuum-godunov';
const hash=(x:unknown)=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const {learningUnits,learningGuides,continuumLabs}=data;
const volume=continuumLabs[continuumConservativeKinds[0]],godunov=continuumLabs[continuumConservativeKinds[1]];
const v=volume.controls.map(c=>c.initial),g=godunov.controls.map(c=>c.initial);
const change=(v:number[],i:number,x:number)=>v.map((a,j)=>i===j?x:a);
const near=(a:number|null,b:number)=>assert.ok(a!==null&&Math.abs(a-b)<1e-10*Math.max(Math.abs(b),1e-280));
function finite(x:unknown):void{if(typeof x==='number')assert.ok(Number.isFinite(x));else if(x&&typeof x==='object')Object.values(x).forEach(finite);}
function bilingual(x:unknown):void{if(!x||typeof x!=='object')return;const o=x as Record<string,unknown>;if('ko'in o&&'en'in o){assert.equal(typeof o.ko,'string');assert.ok((o.ko as string).length);assert.equal(typeof o.en,'string');assert.ok((o.en as string).length);assert.doesNotMatch(o.en as string,/[가-힣]/);}Object.values(o).forEach(bilingual);}
test('500 prior units, guides and43 lab groups491 entries retain content and order',()=>{
  const ids=new Set(prior.ids),units=learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,500);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);
  assert.equal(hash(Object.fromEntries(Object.entries(learningGuides).filter(([id])=>ids.has(id as typeof units[number]['id'])))),prior.guides);
  const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(key=>[key,(data as unknown as Record<string,Record<string,unknown>>)[name][key]]))]));assert.equal(hash(labs),prior.labs);assert.equal(Object.keys(labs).length,43);assert.equal(Object.values(prior.labEntries).flat().length,491);
});
for(const id of continuumConservativeKinds)test(`${id}: complete bilingual registration, concepts, quizzes and three reproducible guides`,()=>{
  const unit=learningUnits.find(u=>u.id===id)!;assert.ok(unit);bilingual(unit);bilingual(learningGuides[id]);bilingual(continuumLabs[id]);
  assert.equal(learningGuides[id].steps.length,4);assert.equal(learningGuides[id].experiments.length,3);assert.equal(unit.questions.length,2);
  for(const e of learningGuides[id].experiments)assert.ok(continuumLabs[id].presets[e.preset]);
  for(const question of unit.questions)assert.ok(question.answer>=0&&question.answer<question.options.length);
  assert.ok(isContinuumKind(id));assert.ok(curriculumDomains.find(d=>d.id==='continuum-physics')!.unitIds.includes(id));
});
for(const [id,evaluate,bounds]of [[continuumConservativeKinds[0],volumeDiscretizationLab,volumeControlBounds],[continuumConservativeKinds[1],godunovDiscretizationLab,godunovControlBounds]]as const){
  const spec=continuumLabs[id];
  test(`${id}: controls match declared numerical domains and all presets return finite/explicit unavailable trees`,()=>{
    assert.equal(spec.controls.length,bounds.length);assert.equal(spec.presets.length,20);
    spec.controls.forEach((c,i)=>assert.deepEqual([c.min,c.max],bounds[i]));
    for(const preset of spec.presets){assert.equal(preset.values.length,bounds.length);const result=evaluate(preset.values);finite(result);assert.doesNotThrow(()=>JSON.stringify(result));finite(evaluateContinuum(id,preset.values));}
  });
  test(`${id}: sparse, wrong-size, fractional and nonfinite control vectors are rejected`,()=>{
    const values=spec.controls.map(c=>c.initial);
    assert.throws(()=>evaluate(values.slice(1)),RangeError);
    for(let i=0;i<values.length;i++){const sparse=values.slice();delete sparse[i];assert.throws(()=>evaluate(sparse),RangeError);assert.throws(()=>evaluate(change(values,i,NaN)),RangeError);if(spec.controls[i].step!=='any')assert.throws(()=>evaluate(change(values,i,values[i]+.1)),RangeError);}
  });
}
test('all12 volume controls bind directly to the actual model parameters and outputs',()=>{
  const alt=[4,3,2,-.5,-1,.2,2,-.5,2,.25,.005,12];
  for(let i=0;i<v.length;i++){
    const values=change(v,i,alt[i]),r=volumeDiscretizationLab(values);
    assert.deepEqual(r,nonuniformVolumeSimulation({cellCountExponent:values[0],length:values[1],area:values[2],meshWarp:values[3],velocity:values[4],diffusivity:values[5],meanConcentration:values[6],relativeContrast:values[7],mode:values[8],phaseCycles:values[9],timeStep:values[10],steps:values[11]}));assert.notEqual(hash(r),hash(volumeDiscretizationLab(v)));
  }
  near(volumeDiscretizationLab(change(v,2,2)).correct.massFromConservationLedger.value,2*volumeDiscretizationLab(v).correct.massFromConservationLedger.value!);
});
test('all7 Burgers controls map to actual trajectories and declared quarter-position encoding',()=>{
  const alt=[4,3,-1,1,1,.005,8];
  for(let i=0;i<g.length;i++){
    const values=change(g,i,alt[i]),r=godunovDiscretizationLab(values),{exactSegments,...model}=r;
    assert.deepEqual(model,burgersRiemannSimulation({cellCountExponent:values[0],length:values[1],leftState:values[2],rightState:values[3],interfaceFraction:values[4]/4,timeStep:values[5],steps:values[6]}));assert.notEqual(hash(r),hash(godunovDiscretizationLab(g)));assert.ok(exactSegments.length>0);
  }
});
test('exact profile paths split shocks, clip only the physical domain and remain absent as wave boundaries for constants',()=>{
  for(const preset of godunov.presets){
    const r=godunovDiscretizationLab(preset.values);
    for(const segment of r.exactSegments){assert.equal(segment.length,2);assert.ok(segment[0][0]>=0&&segment[1][0]<=r.parameters.length);assert.ok(segment[1][0]>segment[0][0]);if(r.waveType==='shock')assert.equal(segment[0][1],segment[1][1]);}
    if(r.waveType==='constant'){assert.equal(r.exactSegments.length,1);assert.equal(r.exactShockPosition,null);assert.equal(r.rarefactionLeftPosition,null);}
    if(r.waveType==='shock'&&r.shockInsideDomain)assert.equal(r.exactSegments.length,2);
  }
});
test('small UI readouts preserve nonzero products and nonlinear errors while exact zeros stay distinct',()=>{
  const tiny=volumeDiscretizationLab(volume.presets[13].values);assert.ok(tiny.cells.some(c=>c.change.sign!==0&&c.change.value===null));
  const q=conservativeScaledObservation({value:Number.MIN_VALUE,sign:1,logMagnitude:Math.log(Number.MIN_VALUE)},Number.MIN_VALUE);assert.equal(q.value,null);assert.equal(q.sign,1);assert.ok(q.logMagnitude! < -1400);
  assert.equal(conservativeScaledObservation(q,0).sign,0);
  const b=godunovDiscretizationLab(godunov.presets[11].values);assert.ok(b.godunov.cells.some(c=>c.error.sign!==0&&c.error.value===null));
  assert.equal(volumeDiscretizationLab(volume.presets[8].values).correct.normChangeUsingMassLedger.sign,0);
});
