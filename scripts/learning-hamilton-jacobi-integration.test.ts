import test from 'node:test';
import assert from 'node:assert/strict';
import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {analyticalMechanicsKinds,evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
const ids=['uniform-force-hamilton-jacobi-family','oscillator-hamilton-jacobi-caustic-atlas']as const;
for(const id of ids)test(`${id}: live guides distinguish physical paths,query data and chart availability`,()=>{
 const unit=learningUnits.find(u=>u.id===id)!,lab=analyticalMechanicsLabs[id],guide=learningGuides[id];
 assert.equal(unit.questions.length,2);assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);
 for(const[i,p]of lab.presets.entries()){
  assert.deepEqual(p.name,guide.experiments[i].title);assert.equal(guide.experiments[i].preset,i);
  p.values.forEach((v,j)=>assert.ok(Number.isInteger(v)&&v>=lab.controls[j].min&&v<=lab.controls[j].max));
  const r=evaluateAnalyticalMechanics(id,p.values);assert.doesNotThrow(()=>JSON.stringify(r));
  for(const v of Object.values(r.stats))assert.ok(v===null||typeof v==='number'&&Number.isFinite(v));
  assert.ok(r.table.rows.every(row=>row.every(v=>typeof v==='number'||typeof v==='string')));
  if(r.uniformHJ){const m=r.uniformHJ,q=m.query;assert.equal(m.globalHamiltonJacobiSolution,i!==2);assert.equal(m.wholeFrozenPathPhysical,i!==2);assert.equal(q.canonicalMapDeterminant.label,'1');if(i<2){assert.equal(q.mechanicalAction.label,'11/6');assert.equal(q.referencePosition.label,'3/2');assert.equal(q.pathActionDefect.label,'0');}if(i===1){assert.equal(m.initialAction.label,'1/2');assert.equal(q.action.label,'7/3');}if(i===2){assert.equal(r.stats['query position m'],0);assert.equal(r.stats['selected position m'],1);assert.equal(r.stats['query gradient momentum kg m per s'],1);assert.equal(r.stats['selected frozen momentum kg m per s'],2);assert.equal(r.stats['query Hamilton Jacobi residual J'],0);assert.equal(r.stats['selected velocity residual m per s'],-1);assert.equal(r.stats['selected force residual N'],2);}}
  else{const q=r.oscillatorHJ!.query;assert.equal(q.flowDeterminant.label,'1');assert.equal(q.energyResidual.label,'0');assert.equal(q.initialGraphMember,true);if(i===0){assert.equal(q.endpointChart!.action.label,'-1');assert.equal(r.stats['mixed action Js'],null);assert.equal(q.graphChart!.action.label,'-1/2');}if(i===1){assert.equal(r.stats['endpoint action Js'],null);assert.equal(q.boundaryInitialMomentumClass,'undetermined');assert.equal(q.physicalAction.label,'0');assert.ok(q.mixedChart);}if(i===2){assert.equal(r.stats['graph action Js'],null);assert.equal(r.stats['graph PDE residual J'],null);assert.ok(q.endpointChart&&q.mixedChart);assert.equal(q.graphProjection.label,'0');assert.equal(q.energy.label,'25/8');assert.equal(q.momentum.label,'-5/2');}}
 }
});
test('HJ kinds extend the sequence and every prerequisite precedes its lesson',()=>{
 assert.equal(analyticalMechanicsKinds[0],'polar-generalized-velocity-force-metric');const start=analyticalMechanicsKinds.indexOf(ids[0]);assert.deepEqual(analyticalMechanicsKinds.slice(start,start+2),ids);
 for(const id of ids){const i=learningUnits.findIndex(u=>u.id===id);assert.ok(i>=0);for(const prior of learningUnits[i].prerequisiteIds)assert.ok(learningUnits.findIndex(u=>u.id===prior)<i);}
});
