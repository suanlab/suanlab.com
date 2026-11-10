import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import React from 'react';import {renderToStaticMarkup} from 'react-dom/server';
import {tensorKinds as kinds,tensorPresets as presets,tensorPresetInput as preset,tensorInputKey as key,tensorFields as fields,editTensorField as edit,tensorProductInput,tensorWernerInput} from '../src/lib/learning-tensor';
import {tensorSteps as steps} from '../src/lib/learning-tensor-concepts';
import {tensorTeaching as teaching} from '../src/data/tensor-teaching';
import {tensorAmplitudeCoordinates as amplitudes,tensorCoordinate,TensorResultView} from '../src/components/learning/tensor-view';
import Lab from '../src/components/learning/tensor-lab';import Concept from '../src/components/learning/tensor-concept';import Cover from '../src/components/learning/tensor-cover';import Quiz from '../src/components/learning/learning-quiz';
import {LanguageProvider} from '../src/components/language-provider';
import {evaluateTensorRequest as evaluate} from '../src/lib/learning-tensor-evaluator';import {isTensorSuccessForInput} from '../src/lib/learning-tensor-protocol';
for(const[family,kind]of kinds.entries()){
 test(`tensor authored examples and bilingual fields:${kind}`,()=>{
  const refs=JSON.parse(readFileSync(`scripts/fixtures/${family===0?'tensor-pure':'mixed-entanglement'}-reference.json`,'utf8')).cases;assert.equal(presets[kind].length,16);assert.equal(new Set(presets[kind].map(p=>key(p.input))).size,16);
  for(const[i,p]of presets[kind].entries()){assert(refs.find((r:{label:string})=>r.label===p.id));assert.equal(key(preset(kind,i)),key(p.input));assert(Object.isFrozen(p.input.parameters));assert(!/[가-힣]/.test(p.label.en));}
  const p=preset(kind,0);assert.equal(fields(p).length,family===0?50:32);assert(fields(p).every(f=>f.label.en!==f.id&&f.label.ko!==f.id));
  const field=family===0?'stateReal00':'hReal00';for(const raw of ['1e-9999','1.00000000000000001','1000001','NaN'])assert.equal(edit(p,field,raw),null);assert.equal(edit(p,'unknown','1'),null);assert.throws(()=>preset(kind,-1),RangeError);
  assert(edit(p,field,'2'));if(family===0){assert.equal(edit(p,'dimensionA','0'),null);assert.equal(edit(p,'stateReal33','1'),null);}else assert.equal(edit(p,'hReal00','-1'),null);
 });
 test(`tensor concept predictions and draft teaching metadata:${kind}`,()=>{
  assert.equal(steps[kind].length,3);const entry=teaching.find(v=>v.unit.id===kind)!;assert.equal(entry.unit.questions.length,3);assert.equal(entry.guide.experiments.length,3);
  for(const s of steps[kind]){assert(s.presets.every(i=>presets[kind][i]));assert(s.options[s.answer]);assert(s.explanation.en.length>180&&s.explanation.ko.length>80);assert(s.solution.en.length>150);assert(!/[가-힣]/.test(JSON.stringify(s,(k,v)=>k==='ko'?undefined:v)));}
 });
 for(const p of presets[kind])test(`tensor authored preparation executes through Worker schema:${p.id}`,()=>{const r=evaluate({version:1,requestId:7,input:p.input});assert(r.ok);assert(isTensorSuccessForInput(r,p.input,7));});
}
test('new-dimension builder starts a declared product state for all16 dimension pairs',()=>{
 for(let a=1;a<=4;a++)for(let b=1;b<=4;b++){const input=tensorProductInput(a,b),r=evaluate({version:1,requestId:1,input});assert(r.ok&&r.kind===kinds[0]);assert.equal(r.result.schmidtRank,1);assert.equal(r.result.dimensionA,a);assert.equal(r.result.dimensionB,b);assert.equal(r.result.normalizedAmplitudes[0].real.value,1);assert(isTensorSuccessForInput(r,input,1));}
 assert.throws(()=>tensorProductInput(0,2),RangeError);assert.throws(()=>tensorProductInput(2,5),RangeError);
});
test('Werner builder preserves local settings while explicitly replacing a general joint state',()=>{
 const base=preset(kinds[1],13);
 for(const percent of [0,33,50,71,100]){const input=tensorWernerInput(base,percent),r=evaluate({version:1,requestId:1,input});assert(r.ok&&r.kind===kinds[1]);for(const[k,v]of Object.entries(base.parameters))if(!k.startsWith('h'))assert.equal(input.parameters[k],v);assert(isTensorSuccessForInput(r,input,1));assert.equal(r.result.classification.entangledTwoQubit,percent>100/3);assert.equal(r.result.chsh.optimizedViolates,percent>100/Math.sqrt(2));}
 assert.throws(()=>tensorWernerInput(preset(kinds[0],0),50),RangeError);assert.throws(()=>tensorWernerInput(base,101),RangeError);
});
test('amplitude geometry uses normalized complex components and omits unavailable coordinates',()=>{
 for(const p of presets[kinds[0]]){const r=evaluate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind===kinds[0]);const points=amplitudes(r.result);let mass=0;for(const row of points)for(const v of row){assert(v);assert(Math.abs(v.real*v.real+v.imaginary*v.imaginary-v.probability)<1e-12);mass+=v.probability;}assert(Math.abs(mass-1)<1e-12);}
 const r=evaluate({version:1,requestId:1,input:preset(kinds[0],0)});assert(r.ok&&r.kind===kinds[0]);const x=structuredClone(r.result);x.normalizedAmplitudes[0].real.value=null;assert.equal(amplitudes(x)[0][0],null);assert.equal(tensorCoordinate(null),null);x.normalizedAmplitudes[1].real.value=NaN;assert.equal(tensorCoordinate(x.normalizedAmplitudes[1].real),null);assert.equal(tensorCoordinate(r.result.normalizedAmplitudes[1].real),0);
});
test('server-rendered calculated teaching is bilingual,unresolved-safe and backed by exact classifications',()=>{
 for(const kind of kinds)for(const index of [0,2,8,11])for(const language of ['ko','en']as const){const response=evaluate({version:1,requestId:1,input:preset(kind,index)});assert(response.ok);const html=renderToStaticMarkup(React.createElement(TensorResultView,{response,language,contextLabel:language==='ko'?'검산':'Reference check'}));assert(html.includes('data-tensor-result'));assert(html.includes('role="region"'));assert(html.includes('tabindex="0"'));assert(!html.includes('NaN'));assert(!html.includes('Infinity'));if(language==='en')assert(!/[가-힣]/.test(html));}
});
test('draft lab,concepts,quiz and cover render without reading Worker during SSR',()=>{
 const before=Object.getOwnPropertyDescriptor(globalThis,'Worker'),reactBefore=Object.getOwnPropertyDescriptor(globalThis,'React');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker during SSR');}});
 try{for(const kind of kinds){const unit=teaching.find(v=>v.unit.id===kind)!.unit;const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(React.Fragment,null,React.createElement(Cover,{kind}),React.createElement(Concept,{kind,language:'en'}),React.createElement(Lab,{kind}),React.createElement(Quiz,{unit}))));assert(html.includes('data-tensor-lab'));assert(html.includes('data-tensor-cover'));assert(!html.includes('data-tensor-result='));}assert.equal(reads,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');if(reactBefore)Object.defineProperty(globalThis,'React',reactBefore);else Reflect.deleteProperty(globalThis,'React');}
});
test('registered teaching graph excludes numerical engines and retains both unit IDs',async()=>{
 const built=await build({entryPoints:['src/components/learning/tensor-lab.tsx','src/components/learning/tensor-concept.tsx'],bundle:true,write:false,outdir:'.runtime/learning/quantum-tensor-teaching/reader-check',metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(built.metafile!.inputs);assert(!inputs.some(p=>/learning-(tensor-pure|tensor-roots|mixed-entanglement(?:-input|-algebra)?|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|interval-transcendentals/.test(p)),inputs.join('\n'));
 const source=readFileSync('src/data/interactive-book.ts','utf8');for(const kind of kinds)assert(source.includes(kind));
});
