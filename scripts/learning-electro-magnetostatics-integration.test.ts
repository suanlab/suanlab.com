import test from 'node:test';import assert from 'node:assert/strict';import {createHash}from'node:crypto';
import * as data from '../src/data/interactive-book';import prior from './fixtures/pre-electro-magnetostatics-registry-hashes.json';
import {getLearningContent} from '../src/lib/learning-content';
import {readFile} from 'node:fs/promises';
import {electroMagnetostaticsKinds as ids,evaluateElectroMagnetostatics as evaluate} from '../src/lib/learning-electro-magnetostatics-lab';
import {electromagnetismKinds,isElectromagnetismKind,evaluateElectromagnetism} from '../src/lib/learning-electromagnetism';
import {curriculumDomains} from '../src/data/learning-curriculum';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<=3e-12*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const run=(i:number,preset=0)=>evaluate(ids[i],data.electromagnetismLabs[ids[i]].presets[preset].values).statics!;
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all462 prior units,guides,42lab groups and their relative order are preserved',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,462);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('static lessons follow continuity,retain prerequisites and join the electromagnetic curriculum',()=>{
 const order=data.learningUnits.map(u=>u.id),start=order.indexOf(ids[0]);assert.equal(order[start-1],'charge-continuity-surface-current');assert.equal(order[start+1],ids[1]);
 assert.deepEqual(electromagnetismKinds.filter(id=>(ids as readonly string[]).includes(id)),ids);
 for(const id of ids){assert.equal(order.filter(x=>x===id).length,1);assert.ok(isElectromagnetismKind(id));assert.ok(curriculumDomains.find(d=>d.id==='electromagnetism')!.unitIds.includes(id));for(const p of data.learningUnits.find(u=>u.id===id)!.prerequisiteIds)assert.ok(order.indexOf(p)>=0&&order.indexOf(p)<order.indexOf(id));}
});
test('new units expose complete bilingual concepts,guides,questions and continuous controls',()=>{
 for(const id of ids){const u=data.learningUnits.find(u=>u.id===id)!,g=data.learningGuides[id],s=data.electromagnetismLabs[id];assert.equal(g.steps.length,4);assert.equal(g.experiments.length,3);assert.equal(u.questions.length,2);assert.equal(s.controls.length,8);assert.equal(s.presets.length,12);assert.ok(s.controls.every(c=>c.step==='any'));
  const walk=(x:unknown):void=>{if(x&&typeof x==='object'){if('ko'in x&&'en'in x){assert.ok(String(x.ko).trim());assert.ok(String(x.en).trim());assert.ok(!/[가-힣]/.test(String(x.en)));}Object.values(x).forEach(walk);}};walk([u,g,s]);for(const ex of g.experiments)assert.ok(ex.preset>=0&&ex.preset<s.presets.length);
 }
});
test('all24 presets preserve inputs,reach their family and return only finite available numbers',()=>{
 const finite=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finite);};
 for(const id of ids)for(const p of data.electromagnetismLabs[id].presets){const before=[...p.values],r=evaluateElectromagnetism(id,p.values);assert.ok(r.statics);finite(r);assert.deepEqual(p.values,before);}
});
test('sphere default and background guides reproduce SI worked examples with separate potentials',()=>{
 const a=run(0),b=run(0,3);assert.ok('sourcePotential'in a.query&&'sourcePotential'in b.query);close(a.query.sourceField![0],4.5);close(a.query.sourcePotential!,12.375);close(b.query.totalField![0],7.5);close(b.query.closedElectricFlux!,4.5*Math.PI);close(b.query.meanRadialFromFlux!,4.5);close(b.query.backgroundPotential,-1.5);close(b.query.totalPotential!,10.875);
});
test('cylinder default and background guides retain current,circulation and distinct point field',()=>{
 const a=run(1),b=run(1,3);assert.ok('circulation'in a.query&&'circulation'in b.query);close(a.query.sourceField![1]*1e6,.1);close(b.query.totalField![1]*1e6,.4);close(b.query.meanAzimuthalFromCirculation!*1e6,.1);close(a.query.enclosedCurrent!,.25);close(b.query.circulation!,Math.PI*1e-7);
});
test('registered center and interface examples keep zero geometry,one-sided limits and zero-source exceptions',()=>{
 for(let i=0;i<2;i++){const center=run(i,2).query,boundary=run(i,1).query,zero=run(i,6).query;assert.deepEqual(center.sourceField,[0,0,0]);assert.equal(center.sourceFieldDirection,null);assert.notEqual(center.normalizationStatus,'finite');assert.equal(boundary.derivativeStatus,'boundary-one-sided-only');assert.equal(boundary.sourceRadialDerivative,null);assert.equal(zero.derivativeStatus,'finite');assert.equal(zero.sourceRadialDerivative,0);assert.ok(run(i,10).query.radialDistance>1);assert.equal(run(i,10).query.derivativeStatus,'finite');}
});
test('adapter angular measurements independently retain zero background mean and correct source mean',()=>{
 for(let i=0;i<2;i++){const r=run(i,3);let source=0,bg=0,total=0;for(let j=0;j<128;j++){source+=(r.angular[j].source!+r.angular[j+1].source!)/256;bg+=(r.angular[j].background+r.angular[j+1].background)/256;total+=(r.angular[j].total!+r.angular[j+1].total!)/256;}close(bg,0);close(source,total);const expected='meanRadialFromFlux'in r.query?r.query.meanRadialFromFlux:r.query.meanAzimuthalFromCirculation;close(total,expected!);assert.ok(Math.max(...r.angular.map(p=>p.total!))-Math.min(...r.angular.map(p=>p.total!))>0);}
});
test('3D cylinder queries retain transverse field,axial background and z translation invariance',()=>{
 const values=[...data.electromagnetismLabs[ids[1]].presets[7].values],a=evaluate(ids[1],values).statics!.query;values[4]=-3.123456789;const b=evaluate(ids[1],values).statics!.query;assert.deepEqual(a.sourceField,b.sourceField);assert.deepEqual(a.totalField,b.totalField);close(a.totalField![2],-.1e-6);close(a.radialDistance,.5);
});
test('small nonzero measurements and conversion underflow remain distinct from physical zero',()=>{
 for(let i=0;i<2;i++){const small=run(i,8).query;assert.ok(small.radialDistance>0);assert.ok(small.sourceFieldMagnitude!>0);const values=[...data.electromagnetismLabs[ids[i]].presets[0].values];if(i===0)values[1]=Number.MIN_VALUE;else values[5]=Number.MIN_VALUE;const r=evaluate(ids[i],values);assert.equal(r.unavailable,'input-representation');assert.equal(r.statics,undefined);assert.deepEqual(r.curves,[]);}
});
test('continuous inputs are retained and malformed static controls are rejected',()=>{
 for(const id of ids){const v=[...data.electromagnetismLabs[id].presets[0].values];v[0]=.731234567891;v[2]=.234567891234;const r=evaluate(id,v).statics!;assert.equal(r.query.radius,v[0]);assert.equal(r.observation[0],v[2]);assert.throws(()=>evaluate(id,v.slice(1)),RangeError);for(const [index,value]of [[0,0],[1,5],[2,NaN],[5,Infinity]]){const bad=[...v];bad[index]=value;assert.throws(()=>evaluate(id,bad),RangeError);}}
});
test('both manuscript pairs render with valid math, matching equations and accessible unique outlines',async()=>{
 for(const id of ['gauss-symmetry-electrostatic-sphere','ampere-symmetry-steady-current-cylinder']){
  const rendered=await getLearningContent(id);
  const equations:string[][]=[];
  for(const locale of ['ko','en'] as const){
   const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8');
   equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));
   const html=rendered[locale],outline=rendered.outline[locale];
   assert.ok(outline.length>=15);assert.equal(new Set(outline.map(h=>h.id)).size,outline.length);
   for(const heading of outline)assert.ok(html.includes(`id="${heading.id}"`));
   assert.ok(!html.includes('katex-error'));assert.ok(html.includes('katex-display'));
   assert.ok(html.includes('tabindex="0"'));assert.ok(html.includes('aria-label='));
   assert.equal((source.match(/^### (?:Exercise|문제|연습) \d/gm)||[]).length,6);
   if(locale==='en')assert.ok(!/[가-힣]/.test(html));
  }
  assert.ok(equations[0].length>0);assert.deepEqual(equations[0],equations[1]);
 }
});
