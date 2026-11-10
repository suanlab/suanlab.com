import test from 'node:test';
import assert from 'node:assert/strict';
import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {analyticalMechanicsKinds,evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
const ids=['routh-cyclic-momentum-reduction','circle-dirac-constraint-geometry']as const;
for(const id of ids)test(`${id}: live guides preserve physical versus singular query semantics`,()=>{
 const u=learningUnits.find(u=>u.id===id)!,lab=analyticalMechanicsLabs[id],guide=learningGuides[id];
 assert.equal(u.questions.length,2);assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);
 for(const[i,p]of lab.presets.entries()){
  assert.deepEqual(p.name,guide.experiments[i].title);assert.equal(guide.experiments[i].preset,i);
  p.values.forEach((v,j)=>assert.ok(Number.isInteger(v)&&v>=lab.controls[j].min&&v<=lab.controls[j].max));
  const r=evaluateAnalyticalMechanics(id,p.values);assert.doesNotThrow(()=>JSON.stringify(r));
  for(const v of Object.values(r.stats))assert.ok(v===null||typeof v==='number'&&Number.isFinite(v));
  assert.ok(r.table.rows.every(row=>row.every(v=>typeof v==='number'||typeof v==='string')));
  if(r.routhReduction){const m=r.routhReduction;assert.equal(m.query===null,i===2);if(i===0){assert.equal(m.query!.trueRadialForce.value,0);assert.equal(m.query!.naiveRadialForce.value,-2);assert.equal(m.query!.motorPower.value,2);assert.equal(m.query!.routhForceSlope.value,-4);assert.equal(m.query!.drivenForceSlope.value,0);}if(i===1)assert.equal(m.query!.angularVelocity.value,-.5);if(i===2){assert.equal(r.stats['query physical force N'],null);assert.equal(r.stats['reduced phase dimension'],null);}}
  else{const m=r.diracConstraint!;assert.equal(m.geometry===null,i===2);assert.equal(m.onPhysicalConstraintSurface,i===0);if(i===0){assert.deepEqual(m.geometry!.momentumRate.map(v=>v.value),[-1,0]);assert.equal(m.geometry!.physicalReactionPower!.value,0);}if(i===1){assert.equal(m.constraints[1].value,1);assert.equal(m.geometry!.physicalReactionForce,null);assert.equal(r.stats['physical reaction power W'],null);}if(i===2){assert.equal(r.stats['Dirac tensor rank'],null);assert.equal(r.stats['leaf dimension'],null);}}
 }
});
test('constraint lessons follow prerequisites without changing earlier analytical indices',()=>{
 assert.equal(analyticalMechanicsKinds[0],'polar-generalized-velocity-force-metric');
 const start=analyticalMechanicsKinds.indexOf(ids[0]);assert.deepEqual(analyticalMechanicsKinds.slice(start,start+2),ids);
 for(const id of ids){const i=learningUnits.findIndex(u=>u.id===id);assert.ok(i>=0);for(const p of learningUnits[i].prerequisiteIds)assert.ok(learningUnits.findIndex(u=>u.id===p)<i);}
});
