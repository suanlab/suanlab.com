import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-coupled-registry-hashes.json';
import {evaluateWaves,wavesKinds} from '../src/lib/learning-waves';
import {coupledModeState,coupledBeatingState} from '../src/lib/learning-coupled-modes';
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const close=(a:number,b:number,tol=2e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const normal=wavesKinds[2],beating=wavesKinds[3];
test('all 444 previous units, guides, configurations and relative order are preserved',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(key=>[key,(data as Record<string,any>)[name][key]]))]));assert.equal(hash(labs),prior.labs);
});
test('coupled wrappers preserve physical initial data, units and actual query time',()=>{
 const r=evaluateWaves(normal,data.wavesLabs[normal].presets[0].values).coupledNormal!;
 assert.deepEqual(r.spectrum.system,{mass1:1,mass2:2,stiffness1:1,stiffness2:1,coupling:1});assert.equal(r.query.time,6);
 const direct=coupledModeState(r.spectrum.system,{position:[1,0],velocity:[0,0]},6);close(r.query.position[0],direct.position[0]);close(r.query.velocity[1],direct.velocity[1]);close(r.energy0,1);
});
test('continuous query fractions reach the actual weak-coupling landmark without clamping',()=>{
 const r=evaluateWaves(beating,data.wavesLabs[beating].presets[0].values).coupledBeating!;
 close(r.query.time,r.firstEnvelopeSwap!.time,1e-13);assert.ok(r.query.bareEnergy[0]>0);close(r.query.bareEnergy[0],.000458974307408621,1e-12);
 const direct=coupledBeatingState(r.system,r.query.time);close(direct.netTransfer,r.query.netTransfer,1e-13);
});
test('guide presets produce translation, repeated modes, local zero flux and out-of-window landmarks',()=>{
 const translation=evaluateWaves(normal,data.wavesLabs[normal].presets[1].values).coupledNormal!;close(translation.query.position[0],7);close(translation.query.velocity[1],.5);close(translation.energy0,.375);
 const repeated=evaluateWaves(normal,data.wavesLabs[normal].presets[2].values).coupledNormal!;assert.equal(repeated.spectrum.degenerate,true);close(repeated.query.modalEnergies[0],.625);close(repeated.query.modalEnergies[1],1.25);
 const free=evaluateWaves(beating,data.wavesLabs[beating].presets[2].values).coupledBeating!;close(free.query.flux12,0);close(free.query.assignedEnergy[0],.25);assert.equal(free.query.approximation,null);
 const outside=evaluateWaves(beating,data.wavesLabs[beating].presets[5].values).coupledBeating!;assert.equal(outside.query.time,4);assert.equal(outside.duration,8);assert.equal(outside.firstEnvelopeSwap!.withinWindow,false);assert.ok(outside.firstEnvelopeSwap!.time>128);
});
test('changing query fraction preserves the whole physical trajectory and energy diagnostics',()=>{
 for(const id of [normal,beating]){const p=data.wavesLabs[id].controls.map(c=>c.initial),a=evaluateWaves(id,[...p.slice(0,-1),0]),b=evaluateWaves(id,[...p.slice(0,-1),1]);const x=a.coupledNormal??a.coupledBeating!,y=b.coupledNormal??b.coupledBeating!;assert.deepEqual(x.states,y.states);assert.equal(x.query.time,0);assert.equal(y.query.time,y.duration);}
});
test('zero input energy, zero duration and invalid fractional controls remain distinct',()=>{
 const a=evaluateWaves(normal,data.wavesLabs[normal].presets[4].values).coupledNormal!;assert.equal(a.energy0,0);assert.equal(a.modalEnergyFractions,null);
 const b=evaluateWaves(beating,data.wavesLabs[beating].presets[4].values).coupledBeating!;assert.equal(b.assignedEnergyFractions,null);assert.equal(b.observedEnvelopePeriod,null);assert.ok(b.spectralEnvelopePeriod!>0);
 for(const id of [normal,beating]){const p=data.wavesLabs[id].controls.map(c=>c.initial);for(const bad of [-.1,1.1,NaN,Infinity])assert.throws(()=>evaluateWaves(id,[...p.slice(0,-1),bad]));assert.throws(()=>evaluateWaves(id,new Array(p.length)));}
});
