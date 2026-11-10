import test from 'node:test';import assert from 'node:assert/strict';import {createHash} from 'node:crypto';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-electromagnetism-registry-hashes.json';
import {getLearningContent} from '../src/lib/learning-content';
import {electromagnetismKinds,evaluateElectromagnetism,isElectromagnetismKind}from '../src/lib/learning-electromagnetism';
import {wavesKinds}from '../src/lib/learning-waves';
import {curriculumDomains}from '../src/data/learning-curriculum';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all 460 pre-electromagnetism units,guides and 41 lab groups retain their canonical content and relative order',()=>{
  const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,460);assert.deepEqual(units.map(u=>u.id),prior.ids);
  assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
  const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
const [f,c]=electromagnetismKinds,F=(i:number)=>evaluateElectromagnetism(f,data.electromagnetismLabs[f].presets[i].values).charges!,C=(i:number)=>evaluateElectromagnetism(c,data.electromagnetismLabs[c].presets[i].values).transport!;
const known=(value:number|null):number=>{assert.notEqual(value,null);return value!;};
const close=(a:number|null,b:number|null,tol=3e-11)=>assert.ok(a!==null&&b!==null&&Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
test('electromagnetism follows the optical sensor lessons and belongs to its own curriculum family',()=>{
  const ids=data.learningUnits.map(u=>u.id),index=ids.indexOf(f);assert.equal(ids[index-1],'photon-counting-exposure-sensor-noise');assert.equal(ids[index+1],c);
  for(const id of[f,c]){assert.equal(ids.filter(x=>x===id).length,1);assert.ok(isElectromagnetismKind(id));assert.ok(!(wavesKinds as readonly string[]).includes(id));assert.ok(curriculumDomains.find(d=>d.id==='electromagnetism')!.unitIds.includes(id));for(const p of data.learningUnits.find(u=>u.id===id)!.prerequisiteIds)assert.ok(ids.indexOf(p)>=0&&ids.indexOf(p)<ids.indexOf(id));}
});
test('every registered preset preserves continuous input values and produces only finite available numeric output',()=>{
  const check=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(check);};
  for(const id of[f,c]){const spec=data.electromagnetismLabs[id];assert.ok(spec.controls.every(x=>x.step==='any'));for(const preset of spec.presets){assert.equal(preset.values.length,spec.controls.length);const before=[...preset.values];check(evaluateElectromagnetism(id,preset.values));assert.deepEqual(preset.values,before);}}
});
test('off-axis source units and independent test-charge units yield the manuscript field,potential and force',()=>{
  const q=F(0).query;close(q.field![0],0);close(q.field![1],9/Math.sqrt(2));close(q.potential!,18/Math.sqrt(2));close(q.force![1]*1e9,-18/Math.sqrt(2));
});
test('midpoint,zero-test and source-collision guides load the physics they describe',()=>{
  const guides=data.learningGuides[f].experiments;assert.equal(F(guides[0].preset).query.fieldMagnitude,0);close(F(2).query.field![0],18);assert.equal(F(2).query.potential,0);assert.deepEqual(F(guides[1].preset).query.force,[0,0]);assert.equal(F(guides[2].preset).query.status,'source-singularity');
});
test('near-source display gaps leave small nonzero query distances and large finite forces intact',()=>{
  const r=F(6);assert.equal(r.query.status,'finite');assert.ok(r.query.contributions[0].distance>0&&r.query.contributions[0].distance<2e-13);assert.ok(r.query.fieldMagnitude!>1e26);assert.ok(r.slice.some(p=>p.displayExcluded&&p.x===-1));
});
test('changing only arrow compression cannot change the physical field,force or potential',()=>{
  const a=F(0),b=F(10);assert.notEqual(a.arrowReference,b.arrowReference);assert.deepEqual(a.query,b.query);
});
test('common geometric dilation keeps source charge while changing field and potential by their correct powers',()=>{
  const a=F(0).query,b=F(9).query;close(b.field![1],a.field![1]/4);close(b.potential!,a.potential!/2);
});
test('physical stream area and independent patch area are distinct controls',()=>{
  const a=C(0),b=C(5);close(b.query.density,2*known(a.query.density));close(b.interval.chargeInside,a.interval.chargeInside);close(b.interval.netInflow,a.interval.netInflow);close(b.surface.current,2*a.surface.current!);assert.equal(a.patchArea,b.patchArea);
});
test('the default transport retains the initial positive rate and later negative integrated change',()=>{
  const initial=C(data.learningGuides[c].experiments[1].preset),later=C(0);close(known(initial.interval.chargeInside)*1e6,.4796070626906165);close(known(initial.interval.netInflow)*1e6,.062373774002816795);close(known(later.interval.chargeInside)*1e6,.35393308929203887);close(known(later.balance.chargeChange)*1e6,-.1256739733985776);close(later.balance.balanceResidual,0);close(later.balance.integratedNetInflow,known(later.balance.chargeChange));
});
test('negative-carrier,normal-reversal and zero-patch examples preserve separate sign and area effects',()=>{
  const a=C(0),negative=C(data.learningGuides[c].experiments[0].preset),reverse=C(6),emptyPatch=C(13);assert.ok(known(negative.query.currentDensity)<0&&negative.system.velocity>0);close(negative.balance.chargeChange,-known(a.balance.chargeChange));close(reverse.surface.current,-a.surface.current!);close(reverse.interval.chargeInside,a.interval.chargeInside);assert.equal(emptyPatch.surface.current,0);close(emptyPatch.interval.netInflow,a.interval.netInflow);
});
test('positive narrow-tail mass remains nonzero and numerical underflow is separately labeled',()=>{
  const small=C(10),under=C(11);assert.ok(known(small.interval.fraction)>5e-25&&known(small.interval.fraction)<5.1e-25);assert.equal(small.interval.fractionStatus,'finite');assert.equal(under.interval.fraction,null);assert.equal(under.interval.fractionStatus,'numerical-underflow');
});
test('unavailable query tails propagate to lab readouts and patch curves without replacing resolved interval charge',()=>{
 const values=[...data.electromagnetismLabs[c].presets[0].values];values[1]=.1;values[2]=0;values[3]=1;values[7]=0;values[8]=4;
 const result=evaluateElectromagnetism(c,values),q=result.transport!;
 assert.equal(q.query.density,null);assert.equal(q.query.currentDensity,null);assert.equal(q.surface.current,null);
 assert.ok(q.interval.chargeInside!==null&&q.interval.chargeInside>0);assert.ok(q.angular.every(x=>x.current===null));
 assert.equal(result.stats['rho query · µC/m³'],null);assert.equal(result.stats['J query · µA/m²'],null);assert.equal(result.stats['patch current · µA'],null);
 values[10]=0;assert.equal(evaluateElectromagnetism(c,values).transport!.surface.current,0);
});
test('representable lab density and lost nonzero current differ from an exact zero-velocity preparation',()=>{
 const values=[...data.electromagnetismLabs[c].presets[0].values];values[1]=1;values[2]=0;values[3]=1e-320;values[7]=0;values[8]=0;
 const result=evaluateElectromagnetism(c,values);assert.ok(result.transport!.query.density!==null);assert.equal(result.transport!.query.currentDensity,null);
 assert.ok(result.stats['rho query · µC/m³']!==null);assert.equal(result.stats['J query · µA/m²'],null);
 values[3]=0;assert.equal(evaluateElectromagnetism(c,values).transport!.query.currentDensity,0);
});
test('continuous controls retain small charges and reject unit-conversion underflow as an explicit unavailable observation',()=>{
  const v=[...data.electromagnetismLabs[f].presets[0].values];v[0]=1e-13;const a=evaluateElectromagnetism(f,v);assert.equal(a.charges!.sources[0].charge,1e-22);v[0]=Number.MIN_VALUE;const b=evaluateElectromagnetism(f,v);assert.equal(b.unavailable,'input-representation');assert.ok(Object.values(b.stats).every(x=>x===null));v[0]=1;v[2]=Number.MIN_VALUE;assert.equal(evaluateElectromagnetism(f,v).unavailable,'input-representation');const w=[...data.electromagnetismLabs[c].presets[0].values];w[0]=Number.MIN_VALUE;assert.equal(evaluateElectromagnetism(c,w).unavailable,'input-representation');
});
test('model adapters reject missing,out-of-range and nonfinite controls without silent clamping',()=>{
  assert.throws(()=>evaluateElectromagnetism(f,[]));const a=[...data.electromagnetismLabs[f].presets[0].values];a[3]=5;assert.throws(()=>evaluateElectromagnetism(f,a));a[3]=NaN;assert.throws(()=>evaluateElectromagnetism(f,a));const b=[...data.electromagnetismLabs[c].presets[0].values];b[6]=0;assert.throws(()=>evaluateElectromagnetism(c,b));b[6]=1;b[4]=0;assert.throws(()=>evaluateElectromagnetism(c,b));
});
test('both new bilingual manuscripts render through the reader with strict math and valid unique outline targets',async()=>{
  for(const id of['charge-superposition-test-field','charge-continuity-surface-current']){
    const content=await getLearningContent(id);
    for(const locale of['ko','en'] as const){const html=content[locale],outline=content.outline[locale];assert.doesNotMatch(html,/katex-error/);assert.match(html,/katex-display/);assert.ok(outline.length>0);assert.equal(new Set(outline.map(h=>h.id)).size,outline.length);for(const heading of outline)assert.ok(html.includes(`id="${heading.id}"`));}
    assert.doesNotMatch(content.en,/[가-힣]/);
  }
});

test('force representation loss survives SI conversion,stats and plotted slices without a false zero',()=>{
 const id=electromagnetismKinds[0],values=data.electromagnetismLabs[id].controls.map(c=>c.initial);values[0]=1e-21;values[1]=0;values[2]=0;values[3]=1;values[4]=0;values[5]=1e-301;
 const result=evaluateElectromagnetism(id,values),q=result.charges!.query;assert.equal(result.unavailable,undefined);assert.notEqual(q.field,null);assert.notEqual(q.potential,null);assert.equal(q.force,null);assert.equal(q.forceStatus,'numerical-range');assert.equal(result.stats['F x · nN'],null);assert.equal(result.stats['F y · nN'],null);const sample=result.charges!.slice.find(p=>p.x===1)!;assert.deepEqual(sample.forceNano,[null,null]);
});
test('sparse arrow-reference and transport-control vectors fail before producing malformed graphics',()=>{
 for(const [id,index]of[[electromagnetismKinds[0],6],[electromagnetismKinds[1],10]]as const){const values=data.electromagnetismLabs[id].controls.map(c=>c.initial);delete values[index];assert.throws(()=>evaluateElectromagnetism(id,values),RangeError);}
});

test('tiny nonzero patch area preserves source density while current stats and angular samples are unavailable',()=>{
 const id=electromagnetismKinds[1],values=data.electromagnetismLabs[id].controls.map(c=>c.initial);values[9]=0;values[10]=1e-320;const r=evaluateElectromagnetism(id,values);assert.equal(r.unavailable,undefined);assert.notEqual(r.transport!.query.currentDensity,0);assert.equal(r.transport!.surface.current,null);assert.equal(r.transport!.surface.status,'numerical-range');assert.equal(r.stats['patch current · µA'],null);assert.ok(r.transport!.angular.every(p=>p.current===null));values[10]=0;assert.equal(evaluateElectromagnetism(id,values).transport!.surface.current,0);
});
