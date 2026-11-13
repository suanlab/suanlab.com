import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {thermalKinds,thermalPresets,thermalPresetInput,thermalInputKey,thermalFields,editThermalDraft} from '../src/lib/learning-thermal';
import {thermalConceptSteps,thermalConceptInputs,thermalGuides} from '../src/lib/learning-thermal-concepts';
import {evaluateThermalRequest} from '../src/lib/learning-thermal-evaluator';
import {isThermalSuccessForInput} from '../src/lib/learning-thermal-protocol';
import {ThermalResultView} from '../src/components/learning/thermal-view';
import {getLearningContent} from '../src/lib/learning-content';
const baseline=JSON.parse(readFileSync('scripts/fixtures/pre-thermal-registry-hashes.json','utf8'));
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('thermal units follow the543 preserved prior units and allow later curriculum additions',()=>{
 assert.deepEqual(Object.keys(learningGuides).slice(0,543),Object.keys(baseline.guides));assert(learningUnits.length>=545);assert.deepEqual(learningUnits.slice(543,545).map(u=>u.id),[...thermalKinds]);
 for(const unit of learningUnits.slice(0,543)){assert.equal(hash(unit),baseline.units[unit.id],unit.id);assert.equal(hash(learningGuides[unit.id]),baseline.guides[unit.id],unit.id);}
 assert.deepEqual(curriculumDomains.map(d=>({...d,unitIds:d.unitIds.filter(id=>Object.hasOwn(baseline.units,id))})),baseline.domains);
 for(const kind of thermalKinds){assert(curriculumDomains.find(d=>d.id==='nonequilibrium')!.unitIds.includes(kind));assert.equal(learningGuides[kind].steps.length,6);assert.equal(learningGuides[kind].experiments.length,3);assert.equal(learningUnits.find(u=>u.id===kind)!.questions.length,3);}
});
for(const kind of thermalKinds){
 for(let i=0;i<thermalPresets[kind].length;i++)test(`rendered thermal preparation ${kind} ${i}`,()=>{
  const input=thermalPresetInput(kind,i),response=evaluateThermalRequest({version:1,requestId:1,input});assert(response.ok);assert(isThermalSuccessForInput(response,input,1));
  for(const language of ['ko','en'] as const){
   const html=renderToStaticMarkup(React.createElement(ThermalResultView,{response,language}));assert(!html.includes('NaN'));assert(!html.includes('Infinity'));assert(!/\b(?:cx|cy|x|y)="(?:NaN|Infinity)/.test(html));assert(html.includes('data-thermal-diagram'));
   if(language==='en')assert(!/[가-힣]/.test(html));
   assert(html.includes(kind===thermalKinds[0]?'data-thermal-levels':'data-thermal-channels'));
  }
 });
 test(`${kind} concepts and guides keep fixed samples independent of editable state`,()=>{
  assert.equal(thermalConceptSteps[kind].length,6);for(let i=0;i<6;i++){
   const samples=thermalConceptInputs(kind,i);assert(samples.length>=2);assert(samples.every(s=>s.kind===kind));assert(samples.every(s=>Object.isFrozen(s.parameters)));assert.deepEqual(samples.map(thermalInputKey),thermalConceptSteps[kind][i].presets.map(n=>thermalInputKey(thermalPresetInput(kind,n))));
  }
  for(const g of thermalGuides[kind])for(const i of g.presetIndices)assert(thermalPresetInput(kind,i));
 });
 test(`${kind} editing rejects lossy numbers and preserves previous valid preparation`,()=>{
  const input=thermalPresetInput(kind,0),before=thermalInputKey(input);
  for(const text of ['1.000000000000000001','1e-9999','NaN','Infinity','1e10','-1'])assert.equal(editThermalDraft(input,'temperatureNumerator',text),null);
  assert.equal(editThermalDraft(input,'unknown','1'),null);assert.equal(thermalInputKey(input),before);assert(editThermalDraft(input,'temperatureNumerator','301'));assert.equal(thermalFields(input).length,kind===thermalKinds[0]?18:16);
 });
 for(const lang of ['ko','en'] as const)test(`${kind}.${lang} reader prose includes the verified research extension without a draft heading`,async()=>{
  const html=await getLearningContent(kind),source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8');assert.equal(html.outline[lang].length,15);assert.equal((source.match(/\$\$/g)||[]).length,kind===thermalKinds[0]?38:40);assert(!/^# /m.test(source));assert(!html[lang].includes('katex-error'));if(lang==='en')assert(!/[가-힣]/.test(source));
 });
}
test('thermal lab and fixed concept bundles keep both numerical engines out of the reader',async()=>{
 const {build}=await import('esbuild');
 for(const entry of ['thermal-lab.tsx','thermal-concept.tsx']){
  const result=await build({entryPoints:[`src/components/learning/${entry}`],bundle:true,write:false,metafile:true,platform:'browser',external:['next/*']});
  assert(!Object.keys(result.metafile!.inputs).some(p=>/learning-(?:driven|reservoir)-two-state-thermodynamics|learning-entropy-production-math|learning-interval-transcendentals/.test(p)));
 }
});
