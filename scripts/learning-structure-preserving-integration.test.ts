import test from 'node:test';
import assert from 'node:assert/strict';
import {learningUnits,learningGuides,analyticalMechanicsLabs} from '../src/data/interactive-book';
import {analyticalMechanicsKinds,evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
const ids=['quartic-discrete-action-splitting','circle-rattle-projection-phase']as const;
for(const id of ids)test(`${id}: live guides,presets,finite readouts and physical prerequisites`,()=>{
 const unit=learningUnits.find(u=>u.id===id)!,lab=analyticalMechanicsLabs[id],guide=learningGuides[id];
 assert.equal(unit.questions.length,2);assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);
 const index=learningUnits.indexOf(unit);for(const prior of unit.prerequisiteIds)assert.ok(learningUnits.findIndex(u=>u.id===prior)<index);
 for(const[i,p]of lab.presets.entries()){
  assert.deepEqual(p.name,guide.experiments[i].title);assert.equal(guide.experiments[i].preset,i);p.values.forEach((v,j)=>assert.ok(Number.isInteger(v)&&v>=lab.controls[j].min&&v<=lab.controls[j].max));
  const r=evaluateAnalyticalMechanics(id,p.values);for(const v of Object.values(r.stats))assert.ok(v===null||typeof v==='number'&&Number.isFinite(v));assert.ok(r.table.rows.every(row=>row.every(v=>typeof v==='number'||typeof v==='string')));
  assert.ok(!/[가-힣]/.test(JSON.stringify({unit:unit.title.en,explanation:guide.experiments[i].explain.en,action:guide.experiments[i].action.en}))); 
  if(r.quarticSplitting){const m=r.quarticSplitting.methods;if(i===0){assert.equal(m[2].reversePositionError,null);assert.equal(m[2].reverseStopReason,'energy-target-unreachable');assert.ok(Math.abs(m[2].firstStep.determinant!-1)>.007);}if(i===1){assert.ok(m[2].reversePositionError!>.003);assert.ok(m[2].reverseMomentumError!<-.006);}if(i===2)for(const method of m){assert.equal(method.completed,false);assert.equal(method.query,null);assert.equal(method.normalizedTerminalError,null);}}
  else{const m=r.circleRattle!;if(i===0){assert.ok(Math.abs(m.rattle.query!.phaseError-.1887902047863923)<1e-12);assert.ok(Math.abs(m.projected.query!.energy-1/6)<1e-12);}if(i===1){assert.equal(m.rattleRoot,0);assert.ok(m.rattle.query);assert.equal(m.rattle.query!.jacobian,null);}if(i===2){assert.equal(m.projected.query!.determinant,1);assert.equal(m.projectedSymplecticOnWholeReducedDomain,false);assert.equal(m.projected.query!.jacobian![0][1],4);}}
 }
});
test('structure kinds preserve original indices and extend the analytical sequence',()=>{assert.equal(analyticalMechanicsKinds[0],'polar-generalized-velocity-force-metric');assert.equal(analyticalMechanicsKinds[1],'double-pendulum-generalized-mass-metric');const start=analyticalMechanicsKinds.indexOf(ids[0]);assert.deepEqual(analyticalMechanicsKinds.slice(start,start+2),ids);});
test('quartic readouts preserve zero-duration action and undefined out-of-run queries',()=>{const r=evaluateAnalyticalMechanics(ids[0],[1,1,1,2,0,0,0,64]);for(const m of r.quarticSplitting!.methods){assert.equal(m.query,null);assert.equal(m.firstStep.discreteAction,null);assert.equal(m.observedOrders[0],null);assert.equal(r.stats[m.method+' query energy drift J'],null);assert.equal(r.stats[m.method+' reverse position error m'],0);}});
test('unreachable RATTLE query leaves projection available and preserves its derivative',()=>{const r=evaluateAnalyticalMechanics(ids[1],[1,2,2,22,4,0,0,1]);assert.equal(r.circleRattle!.rattle.query,null);assert.ok(r.circleRattle!.projected.query);assert.equal(r.stats['rattle query determinant'],null);assert.ok(typeof r.stats['projected query determinant']==='number'&&r.stats['projected query determinant']<1);const[x0,x1,y0,y1]=r.bounds!;assert.ok(r.curves.flat().every(([x,y])=>x>=x0&&x<=x1&&y>=y0&&y<=y1));});
