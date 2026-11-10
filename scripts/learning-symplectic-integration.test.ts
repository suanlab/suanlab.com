import test from 'node:test';
import assert from 'node:assert/strict';
import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {evaluateAnalyticalMechanics,analyticalMechanicsKinds} from '../src/lib/learning-analytical-mechanics';
const ids=['four-dimensional-symplectic-maps','lagrangian-momentum-graphs']as const;
for(const id of ids)test(`${id}: executable guides match lesson claims and publish finite scalar readouts`,()=>{
 const lab=analyticalMechanicsLabs[id],unit=learningUnits.find(u=>u.id===id)!;
 assert.equal(learningGuides[id].steps.length,4);assert.equal(unit.questions.length,2);assert.equal(lab.presets.length,3);
 for(const [i,preset]of lab.presets.entries()){
  assert.equal(learningGuides[id].experiments[i].preset,i);assert.deepEqual(learningGuides[id].experiments[i].title,preset.name);
  preset.values.forEach((v,j)=>{const c=lab.controls[j];assert.ok(v>=c.min&&v<=c.max&&Number.isInteger(v));});
  const r=evaluateAnalyticalMechanics(id,preset.values);for(const v of Object.values(r.stats))assert.ok(v===null||typeof v==='number'&&Number.isFinite(v));assert.doesNotThrow(()=>JSON.stringify(r));
  if(r.symplecticMap){const m=r.symplecticMap;assert.equal(m.isSymplectic,i===1);assert.equal(m.isVolumePreserving,true);assert.equal(m.originalClosedAction.label,['-1','-1','0'][i]);assert.equal(m.transformedClosedAction.label,['-2','-1','0'][i]);if(i===1){assert.deepEqual(m.transformedPairAreas.map(v=>v.label),['2','-1']);assert.equal(m.query.transformedCumulativeAction.label,'1');}if(i===2)assert.equal(r.stats['area ratio'],null);}
  else{const m=r.lagrangianGraph!;assert.equal(m.isLagrangianGraph,i!==1);assert.equal(m.graphDimension,2);assert.equal(m.momentumMapRank,i===2?0:2);assert.equal(m.closedLoopAction.label,i===1?'-2':'0');assert.equal(m.lineIsLagrangian,false);}
 }
});
test('analytical kind indices remain stable while new units follow their prerequisites',()=>{
 assert.equal(analyticalMechanicsKinds[0],'polar-generalized-velocity-force-metric');assert.equal(analyticalMechanicsKinds[1],'double-pendulum-generalized-mass-metric');const start=analyticalMechanicsKinds.indexOf(ids[0]);assert.ok(start>=0);assert.deepEqual(analyticalMechanicsKinds.slice(start,start+2),ids);
 for(const id of ids){const i=learningUnits.findIndex(u=>u.id===id);for(const prior of learningUnits[i].prerequisiteIds)assert.ok(learningUnits.findIndex(u=>u.id===prior)<i);}
});
