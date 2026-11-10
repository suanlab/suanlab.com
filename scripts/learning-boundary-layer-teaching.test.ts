import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
import {boundaryLayerSteps,boundaryLayerExperiments,boundaryLayerQuestions,boundaryLayerLabels} from '../src/data/boundary-layer-teaching';
import {boundaryLayerExamples,boundaryLayerDefault} from '../src/lib/learning-boundary-layer-input';
import {calculateBoundaryLayer} from '../src/lib/learning-boundary-layer';
import {boundaryLayerView} from '../src/lib/learning-boundary-layer-view';
import {parseStoredBoundaryLayer} from '../src/components/learning/use-saved-boundary-layer';
import Lab from '../src/components/learning/boundary-layer-lab';
import Result from '../src/components/learning/boundary-layer-result';
const results=boundaryLayerExamples.map(c=>({id:c.id,result:calculateBoundaryLayer(c.input)}));
test('eight concepts,six guides,eight explained questions and seventeen examples cover the bilingual teaching scope',()=>{
 assert.equal(boundaryLayerSteps.length,8);assert.equal(boundaryLayerExperiments.length,6);assert.equal(boundaryLayerQuestions.length,8);assert.equal(Object.keys(boundaryLayerLabels).length,17);
 assert.deepEqual(new Set(boundaryLayerExperiments.flatMap(g=>g.examples)),new Set(boundaryLayerExamples.map(c=>c.id)));
 for(const language of ['ko','en']as const){for(const s of boundaryLayerSteps){assert(s.title[language].length>4);assert(s.body[language].length>40);}for(const q of boundaryLayerQuestions){assert(q.explanation[language].length>30);assert(q.options[q.correct][language]);}if(language==='en')assert(!/[가-힣]/.test(JSON.stringify([...boundaryLayerSteps.map(s=>s.body.en),...boundaryLayerExperiments.map(g=>g.explanation.en),...boundaryLayerQuestions.map(q=>q.explanation.en)])));}
});
test('all examples remain finite in physical/stretched,linear/signed-log and optional-inner views',()=>{
 for(const {id,result}of results)for(const coordinate of ['physical','inner']as const)for(const scale of ['linear','signed-log']as const)for(const showInner of [false,true]){const m=boundaryLayerView(result,coordinate,scale,showInner);assert.equal(m.points.length,65);assert(m.xMax>0&&Number.isFinite(m.xMax));assert(m.yMin<m.yMax&&Number.isFinite(m.yMin)&&Number.isFinite(m.yMax));for(const p of m.points){assert(Number.isFinite(p.coordinate));for(const kind of m.kinds)assert(Number.isFinite(m.transform(p[kind]!)),id+kind);}if(result.layer===null){assert.equal(m.coordinate,'physical');assert.deepEqual(m.kinds,['exact']);}}
});
test('right layers reverse physical position in stretched coordinates; zero drift never acquires fake layers',()=>{
 const r=results.find(c=>c.id==='right-layer')!.result,m=boundaryLayerView(r,'inner','linear',false);assert.equal(m.points[0].physicalX,1);assert(m.points[64].physicalX<m.points[0].physicalX);assert(m.points[64].coordinate>m.points[0].coordinate);
 for(const id of ['zero-drift-forced','zero-drift-unforced']){const m=boundaryLayerView(results.find(c=>c.id===id)!.result,'inner','signed-log',true);assert(m.zeroDrift);assert.equal(m.kinds.length,1);assert.equal(m.coordinate,'physical');}
});
test('storage is bounded,versioned and returns an owned validated physical input',()=>{
 const raw=JSON.stringify({version:1,value:boundaryLayerDefault}),restored=parseStoredBoundaryLayer(raw);assert.deepEqual(restored,boundaryLayerDefault);assert(Object.isFrozen(restored));for(const value of ['{',' '.repeat(4097),JSON.stringify({version:2,value:boundaryLayerDefault}),JSON.stringify({version:1,value:{...boundaryLayerDefault,epsilon:'0'}})])assert.equal(parseStoredBoundaryLayer(value),null);
});
test('full and embedded labs server-render without Worker creation or duplicate lesson questions',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),saved=Object.getOwnPropertyDescriptor(globalThis,'React'),worker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{for(const language of ['ko','en']as const){const full=renderToStaticMarkup(React.createElement(Lab,{language})),embedded=renderToStaticMarkup(React.createElement(Lab,{language,embedded:true}));assert(full.includes('information-question'));assert(!embedded.includes('information-question'));assert(!embedded.includes('information-step'));if(language==='en')assert(!/[가-힣]/.test(full));}assert.equal(calls,0);}finally{if(saved)Object.defineProperty(globalThis,'React',saved);else Reflect.deleteProperty(globalThis,'React');if(worker)Object.defineProperty(globalThis,'Worker',worker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('result renders full fractions,zero-drift distinctions and explicit mesh/precision provenance',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),saved=Object.getOwnPropertyDescriptor(globalThis,'React');Object.defineProperty(globalThis,'React',{configurable:true,value:React});
 try{for(const c of results){const html=renderToStaticMarkup(React.createElement(Result,{result:c.result,language:'en'}));assert(html.includes(c.result.probe.exact.lower));assert(html.includes('not solver-error proofs'));assert(html.includes('working grid'));assert(!/[가-힣]/.test(html));if(c.result.layer===null)assert(html.includes('data-boundary-zero-drift'));}}finally{if(saved)Object.defineProperty(globalThis,'React',saved);else Reflect.deleteProperty(globalThis,'React');}
});
test('teaching bundle excludes producer,independent validator and transcendental arithmetic',async()=>{
 const b=await build({entryPoints:['src/components/learning/boundary-layer-lab.tsx'],bundle:true,write:false,format:'esm',platform:'browser',metafile:true,jsx:'automatic'});for(const p of Object.keys(b.metafile!.inputs))assert(!/learning-(?:boundary-layer|quantum-hilbert-core|interval-transcendentals)(?:\.ts|-validation\.ts|-validation-exp\.ts|-evaluator\.ts|-validation-evaluator\.ts)$/.test(p));
});
