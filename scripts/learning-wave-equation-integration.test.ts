import test from 'node:test';import assert from 'node:assert/strict';import{createHash}from'node:crypto';
import*as data from '../src/data/interactive-book';import prior from './fixtures/pre-wave-equation-registry-hashes.json';
import{evaluateWaves,wavesKinds}from'../src/lib/learning-waves';
import{wavePulsePoint,periodicWaveChain}from'../src/lib/learning-wave-equation';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const close=(a:number,b:number,tol=2e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const [chain,pulse]=[wavesKinds[4],wavesKinds[5]];
test('all446 prior units,guides,lab entries and relative order are preserved',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('chain wrapper preserves the same physical material and independent initial fields under refinement',()=>{
 const a=evaluateWaves(chain,data.wavesLabs[chain].presets[0].values).chain!,b=evaluateWaves(chain,data.wavesLabs[chain].presets[1].values).chain!;
 assert.equal(a.query.time,4);assert.deepEqual(a.harmonics,b.harmonics);assert.equal(a.query.c,b.query.c);close(a.query.mass,2*b.query.mass);close(b.query.stiffness,2*a.query.stiffness);assert.ok(b.query.rmsDisplacementError<a.query.rmsDisplacementError);
 const direct=periodicWaveChain(a.system,a.harmonics,4);assert.deepEqual(a.query,direct);assert.ok(a.harmonics.some(h=>h.positionCos===0&&h.positionSin===0&&h.velocityCos!==0));
});
test('guide chain presets reproduce affine drift and a resolved non-long-wave discrepancy',()=>{
 const a=evaluateWaves(chain,data.wavesLabs[chain].presets[2].values).chain!;a.query.position.forEach(q=>close(q,.4));close(a.query.energy,.01);close(a.query.momentum,.4);
 const b=evaluateWaves(chain,data.wavesLabs[chain].presets[3].values).chain!;close(b.query.spectrum[3].frequencyRatio!,.7842133035765372);close(b.query.rmsDisplacementError,.030068519533523935);
});
test('pulse guides reproduce both reflection signs,the velocity plateau and actual signed window loss',()=>{
 const get=(i:number)=>evaluateWaves(pulse,data.wavesLabs[pulse].presets[i].values).pulse!;
 const a=get(0),b=get(1);assert.equal(a.time,4);close(a.query.displacement,-.1);close(b.query.displacement,.1);
 const c=get(2);close(c.query.displacement,.09142857142857143);close(c.query.energyDensity,0);assert.ok(c.budget.totalEnergyNow>0);
 const d=get(3);close(d.budget.energyNow,0);close(d.budget.integratedOutflow,d.budget.energy0);close(d.budget.integratedInflow,0);close(d.budget.totalEnergyNow,d.budget.totalEnergy0);close(d.budget.residual,0);
 for(const i of[4,5]){const r=get(i),q=wavePulsePoint(r.system,0,r.time);close(r.system.boundary==='fixed'?q.displacement:q.gradient,0);close(q.flux,0);}
});
test('continuous query fractions select real times without changing full-window histories',()=>{
 for(const id of[chain,pulse]){const p=data.wavesLabs[id].controls.map(c=>c.initial),a=evaluateWaves(id,[...p.slice(0,-1),0]),b=evaluateWaves(id,[...p.slice(0,-1),.73123456789]);const x=a.chain??a.pulse!,y=b.chain??b.pulse!;assert.deepEqual(x.history,y.history);close(y.query.time,y.duration*.73123456789);assert.equal(x.query.time,0);}
});
test('zero duration and zero input preserve the model and reject unsupported or sparse control values',()=>{
 for(const id of[chain,pulse]){const spec=data.wavesLabs[id],p=spec.controls.map(c=>c.initial);for(const bad of[-.1,1.1,NaN,Infinity])assert.throws(()=>evaluateWaves(id,[...p.slice(0,-1),bad]));assert.throws(()=>evaluateWaves(id,new Array(p.length)));assert.throws(()=>evaluateWaves(id,[...p.slice(0,2),.5,...p.slice(3)]));}
 const zero=evaluateWaves(chain,data.wavesLabs[chain].presets[5].values).chain!;close(zero.query.energy,0);close(zero.query.rmsDisplacementError,0);
 const p=evaluateWaves(pulse,data.wavesLabs[pulse].presets[6].values).pulse!;assert.equal(p.time,0);assert.equal(p.history.length,1);assert.equal(p.history[0].netBoundaryWork,0);
});
