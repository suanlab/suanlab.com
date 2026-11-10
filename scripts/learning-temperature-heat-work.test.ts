import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import prior from './fixtures/pre-temperature-heat-work-registry-hashes.json';
import { learningUnits, learningGuides, temperatureHeatWorkLabs as labs } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { evaluateThermalContact as thermal, evaluateGasPaths as gas, temperatureHeatWorkKinds as ids, validThermalContactValues, validGasPathValues } from '../src/lib/learning-temperature-heat-work';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('registration preserves existing unit order, metadata and guides',()=>{
 assert.equal(prior.ids.length,502);assert.ok(learningUnits.length>=504);
 assert.deepEqual(learningUnits.slice(0,502).map(u=>u.id),prior.ids);
 for(const u of learningUnits.slice(0,502)){assert.equal(hash(u),(prior.units as Record<string,string>)[u.id]);assert.equal(hash(learningGuides[u.id]),(prior.guides as Record<string,string>)[u.id]);}
});
test('two new units have six guides, eight concepts, four questions and valid earlier prerequisites',()=>{
 const order=new Map(learningUnits.map((u,i)=>[u.id,i]));
 for(const id of ids){const unit=learningUnits.find(u=>u.id===id)!;assert(unit);for(const p of unit.prerequisiteIds)assert(order.get(p)!<order.get(id)!);assert.equal(unit.questions.length,2);const g=learningGuides[id];assert.equal(g.steps.length,4);assert.equal(g.experiments.length,3);for(const e of g.experiments)assert(labs[id].presets[e.preset]);}
 assert(ids.every(id=>curriculumDomains.find(d=>d.id==='thermo-statmech')!.unitIds.includes(id)));
});
for(const id of ids)for(const [i,p]of labs[id].presets.entries())test(`${id} complete preset vector ${i}`,()=>{
 const spec=labs[id];assert.equal(p.values.length,spec.controls.length);
 p.values.forEach((v,j)=>assert(v>=spec.controls[j].min&&v<=spec.controls[j].max));
 const r=id===ids[0]?thermal(p.values):gas(p.values);assert.equal(r.history.length,65);
 const walk=(x:unknown):void=>{if(typeof x==='number')assert(Number.isFinite(x));else if(x&&typeof x==='object')Object.values(x).forEach(walk);};walk(r);assert.doesNotThrow(()=>JSON.stringify(r));
});
test('all eight thermal controls reach their physical parameter, query or interval bound',()=>{
 const v=[2,3,330,270,.4,3,0,3];const cases=[1.25,2.5,320,280,.8,4,1,5];
 for(let i=0;i<v.length;i++){const a=[...v];a[i]=cases[i];const r=thermal(a);assert.deepEqual([r.parameters.capacity1,r.parameters.capacity2,r.parameters.initialTemperature1,r.parameters.initialTemperature2,r.parameters.conductance,r.query.time,r.windowStart,r.windowEnd],a);}
});
test('all six gas controls reach amount, endpoints or both current path positions',()=>{
 const v=[1,300,330,.01,.02,.75],cases=[1.5,280,340,.015,.025,.25];
 for(let i=0;i<v.length;i++){const a=[...v];a[i]=cases[i];const r=gas(a);assert.deepEqual([r.parameters.moles,r.parameters.initialTemperature,r.parameters.finalTemperature,r.parameters.initialVolume,r.parameters.finalVolume,r.queryA.progress],a);assert.equal(r.queryB.progress,a[5]);}
});
test('a reversed thermal window remains invalid instead of silently exchanging inputs',()=>{
 const r=thermal([2,3,330,270,.4,3,4,3]);assert.equal(r.window,null);assert.equal(r.windowStart,4);assert.equal(r.windowEnd,3);assert(r.query.heat.value!>0);
});
test('time and progress sweeps retain the full physical domain including zero states',()=>{
 const t=thermal([2,3,330,270,0,0,0,0]);assert.equal(t.history[0].time,0);assert.equal(t.history.at(-1)!.time,8);assert(t.history.every(q=>q.heat.sign===0));
 const g=gas([1,300,300,.01,.01,0]);assert.equal(g.history[0].progress,0);assert.equal(g.history.at(-1)!.progress,1);assert(g.history.every(q=>q.a.work.sign===0&&q.b.work.sign===0));
});
test('saved control validation rejects malformed, nonfinite and out-of-range vectors',()=>{
 assert(!validThermalContactValues([]));assert(!validGasPathValues({}));
 assert(!validThermalContactValues([2,3,330,270,NaN,3,0,3]));assert(!validGasPathValues([1,300,330,.01,.02,2]));
 assert.throws(()=>thermal([2,3,330,270,.4]),RangeError);assert.throws(()=>gas([0,300,330,.01,.02,1]),RangeError);
});
