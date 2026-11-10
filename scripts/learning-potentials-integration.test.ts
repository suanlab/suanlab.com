import test from 'node:test';import assert from 'node:assert/strict';import {createHash}from'node:crypto';import {readFile}from'node:fs/promises';
import * as data from '../src/data/interactive-book';import prior from './fixtures/pre-potentials-registry-hashes.json';import {getLearningContent}from'../src/lib/learning-content';
import {potentialsKinds as ids,evaluatePotentials as evaluate,potentialRectangleFlux}from '../src/lib/learning-potentials-lab';
import {electromagnetismKinds,evaluateElectromagnetism}from '../src/lib/learning-electromagnetism';
import {curriculumDomains}from '../src/data/learning-curriculum';
import refs from './fixtures/potentials-independent.json';
const close=(a:number,b:number,tol=4e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const scalar=(i=0)=>evaluate(ids[0],data.electromagnetismLabs[ids[0]].presets[i].values).scalarPotential!;
const vector=(i=0)=>evaluate(ids[1],data.electromagnetismLabs[ids[1]].presets[i].values).vectorPotential!;
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all464 prior units,guides,42lab groups and relative order retain their frozen canonical values',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,464);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('potential units extend the prior static lessons and all prerequisites precede their units',()=>{
 const order=data.learningUnits.map(u=>u.id),start=order.indexOf(ids[0]);assert.equal(order[start-1],'ampere-symmetry-steady-current-cylinder');assert.equal(order[start+1],ids[1]);assert.deepEqual(electromagnetismKinds.filter(id=>(ids as readonly string[]).includes(id)),ids);
 for(const id of ids){assert.equal(order.filter(x=>x===id).length,1);assert.ok(curriculumDomains.find(d=>d.id==='electromagnetism')!.unitIds.includes(id));for(const p of data.learningUnits.find(u=>u.id===id)!.prerequisiteIds)assert.ok(order.indexOf(p)>=0&&order.indexOf(p)<order.indexOf(id));}
});
test('bilingual data provide eight concepts,six guides,four quizzes and33continuous controls',()=>{
 let count=0;for(const id of ids){const unit=data.learningUnits.find(u=>u.id===id)!,guide=data.learningGuides[id],spec=data.electromagnetismLabs[id];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);assert.ok(spec.controls.every(c=>c.step==='any'));count+=spec.controls.length;for(const ex of guide.experiments)assert.ok(ex.preset>=0&&ex.preset<spec.presets.length);
 const visit=(v:unknown):void=>{if(v&&typeof v==='object'){if('ko'in v&&'en'in v){assert.ok(String(v.ko).trim());assert.ok(String(v.en).trim());assert.ok(!/[가-힣]/.test(String(v.en)));}Object.values(v).forEach(visit);}};visit([unit,guide,spec]);}assert.equal(count,33);
});
test('all26 presets preserve raw controls and expose only finite available output',()=>{
 const visit=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(visit);};let count=0;
 for(const id of ids)for(const preset of data.electromagnetismLabs[id].presets){const before=[...preset.values],r=evaluateElectromagnetism(id,preset.values);assert.ok(r.scalarPotential||r.vectorPotential);assert.deepEqual(preset.values,before);visit(r);count++;}assert.equal(count,26);
});
test('registered scalar guides retain reference invariance,test-charge signs and a tiny nonzero difference',()=>{
 const a=scalar(),b=scalar(1),negative=scalar(2),tiny=scalar(3);close(a.query.potentialDifference!,1.5);close(b.query.final.potential!-a.query.final.potential!,95);assert.equal(a.query.potentialDifference,b.query.potentialDifference);close(a.query.fieldWorkBezier!*1e9,-3);close(negative.query.fieldWorkBezier!*1e9,3);assert.deepEqual(negative.query.final.field,[-3,1,0]);assert.equal(tiny.query.naiveEndpointDifference,0);assert.equal(tiny.query.potentialDifference,1e-16);assert.ok(tiny.samples.at(-1)!.curveWork.potentialDifference!>0);
});
test('partial Bezier integrals follow the actual 3D path and finish at the full-work observation',()=>{
 for(const index of[0,5,6,7,11]){const r=scalar(index),last=r.samples.at(-1)!;last.curve.forEach((x,j)=>close(x,r.end[j]));close(last.curveWork.bezierFieldIntegral!,r.query.bezierFieldIntegral!);close(last.straightWork.straightFieldIntegral!,r.query.straightFieldIntegral!);assert.equal(r.samples[0].curveWork.potentialDifference,0);}assert.equal(scalar(6).query.potentialDifference,0);
});
test('gauged edge measurements and independently integrated magnetic surface flux match all24 references',()=>{
 for(const r of refs.rectangles){const f=potentialRectangleFlux({radius:r.radius,current:r.current,permeability:1},r.left,r.right,r.bottom,r.top,r.y);close(f.flux!,r.independentMagneticFluxOverMu,2e-11);}
});
test('orientation reverses flux and normal mean while retaining Cartesian mean B y',()=>{
 const a=vector(),b=vector(7);close(a.loop.circulation!*1e6,-.17484302162163287);close(a.flux.flux!,a.loop.circulation!);close(a.flux.flux!,-b.flux.flux!);close(a.flux.meanYField!,b.flux.meanYField!);assert.ok(a.flux.meanNormalField!<0&&b.flux.meanNormalField!>0);const z=vector(8);assert.equal(z.flux.normalizationStatus,'zero-area');assert.equal(z.flux.meanNormalField,null);assert.equal(z.flux.meanYField,null);
});
test('gauge changes potential and individual edges while preserving B and the complete loop',()=>{
 const a=vector(),none=vector(1),constant=vector(2),harmonic=vector(3);assert.notDeepEqual(a.query.vectorPotential,none.query.vectorPotential);assert.deepEqual(a.query.magneticField,none.query.magneticField);close(a.query.divergenceA*1e6,.6);assert.equal(harmonic.query.divergenceA,0);assert.notEqual(constant.loop.edgeIntegrals[1],none.loop.edgeIntegrals[1]);close(a.loop.circulation!,constant.loop.circulation!);assert.equal(vector(5).query.secondDerivativeStatus,'boundary-one-sided-only');assert.ok(vector(5).query.magneticField);
});
test('arithmetic states and SI conversion loss remain distinct from exact zero charge or current',()=>{
 const r=scalar(8);assert.equal(r.query.final.potential,null);assert.ok(r.query.final.field);assert.equal(scalar(4).charge,0);assert.deepEqual(vector(10).query.magneticField,[0,0,0]);
 for(let i=0;i<2;i++){const values=[...data.electromagnetismLabs[ids[i]].presets[0].values];values[i===0?19:5]=Number.MIN_VALUE;const result=evaluate(ids[i],values);assert.equal(result.unavailable,'input-representation');assert.equal(result.scalarPotential,undefined);assert.equal(result.vectorPotential,undefined);}
});
test('continuous values are preserved and malformed,sparse or out-of-domain controls fail explicitly',()=>{
 for(let i=0;i<2;i++){const id=ids[i],values=[...data.electromagnetismLabs[id].presets[0].values];values[0]=.731234567891;const r=evaluate(id,values);assert.equal(i===0?r.scalarPotential!.system.hessian[0][0]:r.vectorPotential!.system.radius,values[0]);assert.throws(()=>evaluate(id,values.slice(1)),RangeError);const sparse=values.slice();delete sparse[2];assert.throws(()=>evaluate(id,sparse),RangeError);values[1]=Infinity;assert.throws(()=>evaluate(id,values),RangeError);}
});
test('every individual control endpoint remains evaluable without leaving the specified domain',()=>{
 const finite=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finite);};
 for(const id of ids){const spec=data.electromagnetismLabs[id];for(let i=0;i<spec.controls.length;i++)for(const x of[spec.controls[i].min,spec.controls[i].max]){const values=spec.controls.map(c=>c.initial);values[i]=x;finite(evaluate(id,values));}}
});
test('thin loops keep a nonzero stable source flux even when a large gauge hides it in edge subtraction',()=>{
 const left=.5,right=.5+1e-15,r=evaluate(ids[1],[1,1,.5,0,0,.2,0,4,left,right,-.5,.5,0]).vectorPotential!;
 const expected=-(right-left)*(right+left)*1e-7;assert.ok(r.loop.sourceCirculation!<0&&r.flux.flux!<0);assert.ok(Math.abs(r.loop.sourceCirculation!/expected-1)<1e-12);assert.ok(Math.abs(r.flux.flux!/expected-1)<1e-12);
});
test('four potential manuscripts render valid paired equations,unique accessible outlines and six explained exercises each',async()=>{
 for(const id of ['electrostatic-potential-reference-path-work','magnetic-vector-potential-curl-gauge']){
  const rendered=await getLearningContent(id),pairs:string[][]=[];
  for(const locale of ['ko','en']as const){
   const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),html=rendered[locale],outline=rendered.outline[locale];
   assert.ok(outline.length>=12);assert.equal(new Set(outline.map(h=>h.id)).size,outline.length);for(const h of outline)assert.ok(html.includes(`id="${h.id}"`));
   assert.ok(!/<h1/.test(html));assert.ok(!html.includes('katex-error'));assert.ok(html.includes('katex-display'));assert.ok(html.includes('tabindex="0"'));assert.ok(html.includes('aria-label='));
   assert.equal((source.match(/^### (?:Exercise|연습) \d/gm)||[]).length,6);if(locale==='en')assert.ok(!/[가-힣]/.test(html));
   pairs.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));
  }
  assert.ok(pairs[0].length>0);assert.deepEqual(pairs[0],pairs[1]);
 }
});
