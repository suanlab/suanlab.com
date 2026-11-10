import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as data from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import baseline from './fixtures/pre-waves-registry-hashes.json';
import {evaluateWaves, isWavesKind, wavesKinds} from '../src/lib/learning-waves';
import {oscillatorFrequencyResponse} from '../src/lib/learning-waves-oscillations';
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const close = (a: number, b: number, tolerance = 2e-9) => assert.ok(Math.abs(a-b) <= tolerance*(1+Math.abs(b)), `${a} != ${b}`);
test('all 442 prior units, guides, lab configurations and relative order remain byte-equivalent as canonical data', () => {
  const ids = new Set(baseline.ids), old = data.learningUnits.filter(u => ids.has(u.id));
  assert.deepEqual(old.map(u=>u.id),baseline.ids);
  assert.equal(hash(old),baseline.units);
  assert.equal(hash(Object.fromEntries(old.map(u=>[u.id,data.learningGuides[u.id]]))),baseline.guides);
  assert.equal(hash(Object.fromEntries(baseline.labExports.map(k=>[k,(data as Record<string,unknown>)[k]]))),baseline.labs);
});
test('both actual wave units have earlier prerequisites, four concepts, three guides, two explained quizzes and domain links', () => {
  const ids=data.learningUnits.map(u=>u.id), domain=curriculumDomains.find(d=>d.id==='waves-optics')!;
  assert.deepEqual(domain.unitIds,wavesKinds);
  for(const id of wavesKinds){
    assert.equal(isWavesKind(id),true);
    const unit=data.learningUnits.find(u=>u.id===id)!,guide=data.learningGuides[id];
    assert.ok(unit.prerequisiteIds.every(p=>ids.indexOf(p)>=0&&ids.indexOf(p)<ids.indexOf(id)));
    assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
    for(const q of unit.questions){assert.ok(q.answer>=0&&q.answer<q.options.length);assert.ok(q.explanation.ko&&q.explanation.en);}
    for(const g of guide.experiments)assert.ok(data.wavesLabs[id].presets[g.preset]);
    assert.doesNotMatch(JSON.stringify([unit.title.en,unit.subtitle.en,guide.intuition.en,...guide.steps.map(s=>[s.title.en,s.explanation.en,s.diagram.en]),...guide.experiments.map(g=>[g.title.en,g.predict.en,g.action.en,g.explain.en])]),/[가-힣]/);
  }
});
test('every registered preset and every individual control endpoint produces finite or explicit unavailable values', () => {
  const finite=(value:unknown):void=>{if(typeof value==='number')assert.ok(Number.isFinite(value));else if(Array.isArray(value))value.forEach(finite);else if(value&&typeof value==='object')Object.values(value).forEach(finite);};
  for(const id of wavesKinds){const spec=data.wavesLabs[id],initial=spec.controls.map(c=>c.initial);
    for(const p of spec.presets){assert.equal(p.values.length,spec.controls.length);finite(evaluateWaves(id,p.values));}
    spec.controls.forEach((control,i)=>{for(const value of [control.min,control.max]){const values=[...initial];values[i]=value;finite(evaluateWaves(id,values));}});
  }
});
test('guide examples compute the claimed periodic, resonant, negative-work and response landmarks',()=>{
  const values=(kind:number,preset:number)=>data.wavesLabs[wavesKinds[kind]].presets[preset].values;
  const periodic=evaluateWaves(wavesKinds[0],values(0,0));close(periodic.query!.position,Math.sin(3));close(periodic.query!.energy,.5);
  const resonant=evaluateWaves(wavesKinds[0],values(0,1));close(resonant.query!.position,2*Math.sin(4));assert.equal(resonant.transient!.heat,0);
  const negative=evaluateWaves(wavesKinds[0],values(0,2));assert.ok(negative.transient!.work<0&&negative.transient!.heat>0);
  const base=evaluateWaves(wavesKinds[1],values(1,0));close(base.response.meanPower!,.5);close(base.response.displacementPeakFrequency!,Math.SQRT1_2);
  const boundary=evaluateWaves(wavesKinds[1],values(1,1));assert.equal(boundary.response.dampingRegime,'underdamped');assert.equal(boundary.response.displacementPeakFrequency,null);
});
test('irrational natural-frequency markers are structurally singular at zero damping without clamping neighbors',()=>{
  const s={mass:1,stiffness:2,damping:0,forceAmplitude:1,frequency:Math.sqrt(2)};
  assert.equal(oscillatorFrequencyResponse(s).gain,null);
  assert.notEqual(oscillatorFrequencyResponse({...s,frequency:s.frequency+1e-12}).gain,null);
  const r=evaluateWaves(wavesKinds[1],[1,2,0,1,4]);
  assert.equal(r.sweep!.find(p=>p.frequency===Math.sqrt(2))!.response.gain,null);
  assert.ok(r.sweep!.some(p=>p.frequency<Math.sqrt(2))&&r.sweep!.some(p=>p.frequency>Math.sqrt(2)));
});
test('DC, zero forcing and singular selected frequencies do not invent periodic traces or measured phases',()=>{
  const dc=evaluateWaves(wavesKinds[1],[1,1,4,1,0]);assert.equal(dc.periodic,undefined);assert.equal(dc.response.phaseLag,null);assert.equal(dc.response.intrinsicLag,0);
  const zero=evaluateWaves(wavesKinds[1],[1,1,4,0,4]);assert.equal(zero.response.phaseLag,null);assert.ok(zero.sweep!.every(p=>p.response.phaseLag===null));
  const singular=evaluateWaves(wavesKinds[1],[1,1,0,1,4]);assert.equal(singular.periodic,undefined);assert.deepEqual(singular.curves,[]);assert.equal(singular.response.meanPower,null);
});
test('the zero-input truth table separates a zero particular response from singular transfer and undefined power bandwidth',()=>{
  for(const dampingCode of [0,4])for(const frequencyCode of [0,2,4,8]){
    const r=evaluateWaves(wavesKinds[1],[1,1,dampingCode,0,frequencyCode]);
    assert.equal(r.response.amplitude,0);assert.equal(r.response.phaseLag,null);assert.equal(r.response.meanPower,0);
    assert.equal(r.response.powerPeakFrequency,null);assert.equal(r.response.halfPower,null);
    if(frequencyCode===0)assert.equal(r.periodic,undefined);
    else{assert.ok(r.periodic);assert.ok(r.periodic.every(p=>p.position===0&&p.velocity===0&&p.power===0&&p.heat===0&&p.energy===0));}
    if(dampingCode===0&&frequencyCode===4)assert.equal(r.response.gain,null);
  }
});
test('query control selects actual time without changing the full-window trajectory or energy accounting',()=>{
  const a=[1,1,4,1,4,0,0,12,0],first=evaluateWaves(wavesKinds[0],a),last=evaluateWaves(wavesKinds[0],[...a.slice(0,8),128]);
  assert.equal(first.query!.time,0);assert.equal(last.query!.time,6);
  assert.deepEqual(first.transient!.states,last.transient!.states);assert.equal(first.transient!.work,last.transient!.work);
});
test('malformed kind, sparse arrays and fractional integer controls are rejected',()=>{
  assert.throws(()=>evaluateWaves('unknown' as typeof wavesKinds[number],[1,1,4,1,4]));
  assert.throws(()=>evaluateWaves(wavesKinds[1],new Array(5)));
  assert.throws(()=>evaluateWaves(wavesKinds[1],[1,1,4,1,4.5]));
});
