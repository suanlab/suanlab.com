import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {qubitTeaching} from '../src/data/qubit-teaching';
import {learningUnits} from '../src/data/interactive-book';
import {qubitTeachingIds as ids,qubitTeachingPresets as presets,qubitTeachingInput as input,qubitExperimentPresets as groups,parseQubitTeachingDraft,parseQubitTeachingStoredInput} from '../src/lib/learning-qubit-teaching-input';
import {qubitInputKey} from '../src/lib/learning-qubit-input';
import {qubitWorkerCases as references} from './learning/qubit-worker-cases';
import {evaluateQubitRequest} from '../src/lib/learning-qubit-evaluator';
import {evaluateQubitValidationRequest} from '../src/lib/learning-qubit-validation-evaluator';
import {getLearningContent} from '../src/lib/learning-content';
import QubitConceptDiagram from '../src/components/learning/qubit-concept-diagram';
import QubitConcept from '../src/components/learning/qubit-concept';
(globalThis as unknown as {React:typeof React}).React=React;
const byInput=new Map(references.map(c=>[qubitInputKey(c.input),c]));
for(const id of ids)for(const [index,preset]of presets[id].entries())test(`prepared teaching input matches independent reference: ${id}/${preset.id}`,()=>{
 const prepared=input(id,index),reference=byInput.get(qubitInputKey(prepared));assert(reference,`No independent reference for ${preset.id}`);
 assert(Object.isFrozen(prepared));assert.deepEqual(prepared,preset.input);
 const response=evaluateQubitRequest({version:1,requestId:1,input:prepared});assert(response.ok);assert.deepEqual(response.result,reference.expected);
 assert.equal(evaluateQubitValidationRequest({version:1,requestId:1,input:prepared,response}).phase,'validated');
 assert.deepEqual(parseQubitTeachingDraft(id,JSON.stringify(prepared)),prepared);
 assert.deepEqual(parseQubitTeachingStoredInput(id,JSON.stringify({version:1,value:prepared})),prepared);
});
for(const item of qubitTeaching){
 test(`complete bilingual draft and linked experiments: ${item.unit.id}`,async()=>{
  assert.equal(item.guide.steps.length,item.guide.experiments.length);assert.equal(item.unit.questions.length,item.guide.steps.length);
  assert.equal(groups[item.unit.id].length,item.guide.experiments.length);
  item.guide.experiments.forEach((g,i)=>{assert.equal(g.preset,groups[item.unit.id][i][0]);for(const n of groups[item.unit.id][i])assert(presets[item.unit.id][n]);});
  for(const locale of ['ko','en'] as const){
   for(const s of item.guide.steps){assert(s.title[locale].length>5);assert(s.explanation[locale].length>80);assert(s.diagram[locale].length>30);}
   for(const g of item.guide.experiments)for(const text of [g.predict,g.action,g.explain])assert(text[locale].length>20);
   for(const q of item.unit.questions){assert.equal(q.options.length,3);assert(q.answer>=0&&q.answer<3);assert(q.explanation[locale].length>80);}
  }
  const content=await getLearningContent(item.unit.id);for(const locale of ['ko','en'] as const){assert(content.outline[locale].length>=8);assert(!content[locale].includes('katex-error'));}
 });
 for(let step=0;step<item.guide.steps.length;step++)test(`schematic is labeled and localized: ${item.unit.id}/${step}`,()=>{
  for(const language of ['ko','en'] as const){
   const svg=renderToStaticMarkup(React.createElement(QubitConceptDiagram,{kind:item.unit.id,step,language}));assert(svg.includes('role="img"'));assert(svg.includes('role="region"'));assert(svg.includes('tabindex="0"'));assert(svg.includes(`data-qubit-diagram-step="${step}"`));assert(svg.includes('var(--learning-'));assert(/<desc[^>]*>[^<]+<\/desc>/.test(svg));
   if(language==='en')assert(!/[가-힣]/.test(svg));
  }
 });
}
test('teaching totals and prerequisites preserve registered units and form an acyclic graph',()=>{
 assert.equal(qubitTeaching.length,4);assert.equal(ids.reduce((n,id)=>n+presets[id].length,0),33);
 assert.equal(qubitTeaching.reduce((n,x)=>n+x.guide.experiments.length,0),13);assert.equal(qubitTeaching.reduce((n,x)=>n+x.unit.questions.length,0),13);
 assert(learningUnits.length>=575);assert.deepEqual(learningUnits.slice(571,575).map(u=>u.id),[...ids]);for(const item of qubitTeaching)assert.deepEqual(learningUnits.find(u=>u.id===item.unit.id),item.unit);
 const all=new Set<string>([...learningUnits.map(u=>u.id),...ids]),questions=qubitTeaching.flatMap(x=>x.unit.questions.map(q=>q.id));assert.equal(new Set(questions).size,13);
 const visit=(id:string,ancestors:Set<string>)=>{assert(!ancestors.has(id),'Cyclic prerequisite');const item=qubitTeaching.find(x=>x.unit.id===id);if(!item)return;for(const prerequisite of item.unit.prerequisiteIds){assert(all.has(prerequisite),prerequisite);visit(prerequisite,new Set([...ancestors,id]));}};
 ids.forEach(id=>visit(id,new Set()));
});
test('new concept component never reads Worker during server rendering',()=>{
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;
 Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker read');}});
 try{for(const kind of ids)assert(renderToStaticMarkup(React.createElement(QubitConcept,{kind,language:'en'})).includes('data-qubit-concept'));assert.equal(reads,0);}
 finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('a lesson rejects JSON belonging to another calculation family',()=>{
 assert.equal(parseQubitTeachingDraft(ids[0],JSON.stringify(input(ids[1]))),null);
 assert.equal(parseQubitTeachingDraft(ids[2],JSON.stringify(input(ids[3]))),null);
 assert(parseQubitTeachingDraft(ids[3],JSON.stringify(input(ids[3],7))));assert.throws(()=>input(ids[0],-1),RangeError);
});
