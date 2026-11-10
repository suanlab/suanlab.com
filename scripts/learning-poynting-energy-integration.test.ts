import test from 'node:test';import assert from 'node:assert/strict';
import {learningUnits,learningGuides,electromagnetismLabs} from '../src/data/interactive-book';
import {poyntingEnergyKinds as ids,evaluatePoyntingEnergy} from '../src/lib/learning-poynting-energy-lab';
import {electromagnetismKinds,evaluateElectromagnetism} from '../src/lib/learning-electromagnetism';
import {curriculumDomains} from '../src/data/learning-curriculum';
const evaluate=(i:number,p=0)=>evaluatePoyntingEnergy(ids[i],electromagnetismLabs[ids[i]].presets[p].values);
const close=(a:number|null,b:number,tol=5e-13)=>assert.ok(a!==null&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const finite=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finite);};
test('energy pair follows induction with prior prerequisites and curriculum links',()=>{
 const order=learningUnits.map(u=>u.id),index=order.indexOf(ids[0]);assert.equal(order[index-1],'electromagnetic-wave-equation-transverse-initial-data');assert.equal(order[index+1],ids[1]);assert.deepEqual(electromagnetismKinds.filter(x=>(ids as readonly string[]).includes(x)),ids);
 for(const id of ids){assert.equal(order.filter(x=>x===id).length,1);for(const p of learningUnits.find(u=>u.id===id)!.prerequisiteIds)assert.ok(order.indexOf(p)>=0&&order.indexOf(p)<order.indexOf(id));assert.ok(curriculumDomains.find(d=>d.id==='electromagnetism')!.unitIds.includes(id));}
});
test('paired data supply eight stages,six guided experiments,four quizzes and21 controls',()=>{
 let controls=0,continuous=0;
 for(const id of ids){const u=learningUnits.find(u=>u.id===id)!,g=learningGuides[id],s=electromagnetismLabs[id];assert.equal(g.steps.length,4);assert.equal(g.experiments.length,3);assert.equal(u.questions.length,2);controls+=s.controls.length;continuous+=s.controls.filter(c=>c.step==='any').length;for(const e of g.experiments)assert.ok(e.preset>=0&&e.preset<s.presets.length);
  const visit=(v:unknown):void=>{if(v&&typeof v==='object'){if('ko'in v&&'en'in v){assert.ok(String(v.ko).trim());assert.ok(String(v.en).trim());assert.doesNotMatch(String(v.en),/[가-힣]/);}Object.values(v).forEach(visit);}};visit([u,g,s]);
 }assert.equal(controls,21);assert.equal(continuous,19);
});
test('all29 presets retain raw controls and return finite or explicitly unavailable observations',()=>{
 let n=0;for(const id of ids)for(const p of electromagnetismLabs[id].presets){const before=[...p.values],r=evaluateElectromagnetism(id,p.values);assert.ok(r.sourceEnergy||r.pulseEnergy);finite(r);assert.deepEqual(p.values,before);n++;}assert.equal(n,29);
});
test('SI omitted-current preset retains the tiny defect despite ordinary zero',()=>{
 const r=evaluate(0,2).sourceEnergy!,q=r.query;assert.equal(q.naiveTotalResidual,0);assert.ok(q.groupedResidual!>0);assert.ok(Math.abs(q.groupedResidual!/3.6833421321300816e-26-1)<5e-15);assert.equal(q.maxwellFamilyCompatible,false);assert.deepEqual(q.totalCurrent,q.displacementCurrent);
 assert.equal(evaluate(0).sourceEnergy!.query.groupedResidual,0);assert.equal(evaluate(0,3).sourceEnergy!.query.groupedResidual,0);assert.equal(evaluate(0,3).sourceEnergy!.query.maxwellFamilyCompatible,false);
});
test('zero acceleration,negative source work and outward-normal power retain distinct meanings',()=>{
 const a=evaluate(0).sourceEnergy!,b=evaluate(0,1).sourceEnergy!,c=evaluate(0,4).sourceEnergy!;
 assert.ok(a.query.fieldToMatterPowerDensity!<0);assert.ok(a.volume.outwardPower!<0);assert.equal(b.query.fieldToMatterPowerDensity,0);assert.equal(b.query.electricStorageRate,0);assert.ok(b.volume.magneticStorageRate!>0);assert.ok(c.query.fieldToMatterPowerDensity!>0);
});
test('radius scaling and degenerate or underflow volumes are not conflated',()=>{
 const a=evaluate(0).sourceEnergy!.volume,b=evaluate(0,12).sourceEnergy!.volume;close(b.electricEnergy!/a.electricEnergy!,16);close(b.magneticEnergy!/a.magneticEnergy!,4);
 for(const p of[7,8])assert.equal(evaluate(0,p).sourceEnergy!.volume.totalEnergy,0);assert.equal(evaluate(0,9).sourceEnergy!.volume.totalEnergy,null);
});
test('one-way and split initial fields carry different whole-axis energy budgets',()=>{
 const a=evaluate(1,1).pulseEnergy!,b=evaluate(1,2).pulseEnergy!;a.spatial.forEach((s,i)=>assert.equal(s.initial.electricValue,b.spatial[i].initial.electricValue));close(a.query.wholeLineActualEnergyPerArea!/b.query.wholeLineActualEnergyPerArea!,2);assert.notEqual(a.spatial[80].initial.magneticValue,b.spatial[80].initial.magneticValue);
});
test('actual zero-B energy and flow do not borrow compatible-reference magnetic storage',()=>{
 const q=evaluate(1,4).pulseEnergy!.query;assert.equal(q.magneticValue,0);assert.equal(q.magneticEnergyDensity,0);assert.equal(q.poyntingZ,0);assert.ok(q.energyResidual!>0);assert.ok(q.componentEnergyReference!>q.energyDensity!);assert.equal(q.admissibleSourceFreeSolution,false);
 const peak=evaluate(1,5).pulseEnergy!.query;assert.equal(peak.energyResidual,0);assert.equal(peak.admissibleSourceFreeSolution,false);
});
test('tiny flux survives square subtraction while absolute energy loss retains analytic-only ratios',()=>{
 const q=evaluate(1,7).pulseEnergy!.query;assert.equal(q.naiveComponentFluxReference,0);assert.ok(q.poyntingZ!>0);assert.ok(q.analyticCompatibleVelocity!>0);
 const underflow=evaluate(1,8).pulseEnergy!.query;assert.equal(underflow.energyDensity,null);assert.equal(underflow.observedEnergyVelocity,null);assert.equal(underflow.analyticCompatibleVelocity,0);
 const zero=evaluate(1,6).pulseEnergy!.query;assert.equal(zero.energyDensity,0);assert.equal(zero.analyticCompatibleVelocity,null);assert.equal(zero.observedVelocityStatus,'zero-energy-undefined');
});
test('thin positive slabs,unrepresentable endpoints and exact empty windows are distinct',()=>{
 const thin=evaluate(1,10).pulseEnergy!,lost=evaluate(1,11).pulseEnergy!,empty=evaluate(1,12).pulseEnergy!;assert.ok(thin.window!.width>0);assert.ok(thin.window!.energyPerArea!>0);assert.equal(lost.halfWidth,Number.MIN_VALUE);assert.equal(lost.geometryStatus,'endpoint-representation');assert.equal(lost.window,null);assert.ok(lost.query.energyDensity!>0);assert.equal(empty.window!.energyPerArea,0);assert.equal(empty.window!.balanceResidual,0);
});
test('all control endpoints stay within joint geometry and pulse displacement bounds',()=>{
 for(const id of ids){const s=electromagnetismLabs[id];for(let i=0;i<s.controls.length;i++)for(const x of[s.controls[i].min,s.controls[i].max]){const v=s.controls.map(c=>c.initial);v[i]=x;finite(evaluatePoyntingEnergy(id,v));}
  const v=s.controls.map(c=>c.initial);v[0]=.731234567891;const q=evaluatePoyntingEnergy(id,v);assert.equal(q.sourceEnergy?.system.magneticOffset??q.pulseEnergy!.system.amplitude,id===ids[0]?v[0]*1e-6:v[0]);}
});
test('unit-conversion loss,sparse inputs and fractional selectors fail explicitly',()=>{
 for(let i=0;i<2;i++){const v=[...electromagnetismLabs[ids[i]].presets[0].values];v[i===0?0:3]=Number.MIN_VALUE;assert.equal(evaluatePoyntingEnergy(ids[i],v).unavailable,'input-representation');assert.throws(()=>evaluatePoyntingEnergy(ids[i],v.slice(1)),RangeError);delete v[0];assert.throws(()=>evaluatePoyntingEnergy(ids[i],v),RangeError);const s=[...electromagnetismLabs[ids[i]].presets[0].values];s[i===0?8:5]=.5;assert.throws(()=>evaluatePoyntingEnergy(ids[i],s),RangeError);}
});
