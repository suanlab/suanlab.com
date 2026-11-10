import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {calculateDuffingSecular} from '../src/lib/learning-duffing-secular';
import {duffingSecularExamples,duffingSecularDefault} from '../src/lib/learning-duffing-secular-input';
import {duffingSecularView} from '../src/lib/learning-duffing-secular-view';
import {duffingSecularSteps,duffingSecularExperiments,duffingSecularQuestions,duffingSecularLabels} from '../src/data/duffing-secular-teaching';
import {parseStoredDuffingSecular} from '../src/components/learning/use-saved-duffing-secular';
import {duffingSecularOrbitView} from '../src/lib/learning-duffing-secular-orbit-view';
import Result from '../src/components/learning/duffing-secular-result';
import Lab from '../src/components/learning/duffing-secular-lab';
const results=duffingSecularExamples.map(c=>({...c,result:calculateDuffingSecular(c.input)}));
test('all seventeen cases have finite projections and original enclosures across both windows and scales',()=>{
 for(const {id,result}of results)for(const window of ['full','local']as const)for(const scale of ['linear','signed-log']as const)for(const show of [false,true]){
  const m=duffingSecularView(result,window,scale,show);assert.equal(m.points.length,65);assert(Number.isFinite(m.yMin)&&Number.isFinite(m.yMax)&&m.yMin<m.yMax,id);for(const [i,p]of m.points.entries()){assert(Number.isFinite(p.time));assert(Number.isFinite(m.x(i)));for(const kind of m.methods)assert(Number.isFinite(m.y(p.values[kind])),id);assert.equal(p.source,(window==='full'?result.fullSamples:result.localSamples)[i]);}
 }
});
test('long-horizon aliasing never acquires connecting segments; invalid formal frequencies are opt-in',()=>{
 for(const {result}of results){const m=duffingSecularView(result,'full','linear',false);assert.equal(m.connect,result.mode==='equilibrium'||result.horizonDiagnostics.fullGridResolvesEightPointsPerCycle===true);assert.equal(m.methods.includes('secondPosition'),result.approximations[1].positiveFrequency);assert(duffingSecularView(result,'local','linear',true).methods.includes('secondPosition'));}
 assert.equal(duffingSecularView(results.find(c=>c.id==='naive-breakdown')!.result,'full','linear',false).connect,false);
 const eq=duffingSecularView(results.find(c=>c.id==='equilibrium')!.result,'local','linear',false);assert.equal(eq.phase,null);assert(eq.points.every(p=>Object.values(p.values).every(v=>v===0)));
 const initial=duffingSecularView(results.find(c=>c.id==='initial-time')!.result,'local','linear',false);assert.equal(initial.start,0);assert.equal(initial.end,0);
});
test('guides cover every example and concepts/questions explain failures in both languages',()=>{
 assert.equal(duffingSecularSteps.length,8);assert.equal(duffingSecularExperiments.length,6);assert.equal(duffingSecularQuestions.length,8);
 assert.deepEqual(new Set(duffingSecularExperiments.flatMap(e=>e.examples)),new Set(duffingSecularExamples.map(e=>e.id)));
 assert.deepEqual(new Set(Object.keys(duffingSecularLabels)),new Set(duffingSecularExamples.map(e=>e.id)));
 for(const lang of ['ko','en']as const){for(const s of duffingSecularSteps)assert(s.body[lang].length>80);for(const q of duffingSecularQuestions){assert(q.options[q.correct][lang]);assert(q.explanation[lang].length>60);}if(lang==='en')assert(!/[가-힣]/.test(JSON.stringify([...duffingSecularSteps.map(s=>s.body.en),...duffingSecularExperiments.map(e=>e.explanation.en),...duffingSecularQuestions.map(q=>q.explanation.en)])));}
});
test('saved fraction spelling is owned and bounded; corrupt and out-of-range input is rejected',()=>{
 const input={...duffingSecularDefault,lambda:'2/200'},raw=JSON.stringify({version:1,value:input});const parsed=parseStoredDuffingSecular(raw);assert.deepEqual(parsed,input);assert(Object.isFrozen(parsed));for(const bad of ['{',' '.repeat(4097),JSON.stringify({version:2,value:input}),JSON.stringify({version:1,value:{...input,probeFraction:'2'}})])assert.equal(parseStoredDuffingSecular(bad),null);
});
test('server rendering preserves all diagnostic fractions and never creates Workers',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),prior=Object.getOwnPropertyDescriptor(globalThis,'React'),worker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{for(const language of ['ko','en']as const){for(const {result}of results){const html=renderToStaticMarkup(React.createElement(Result,{result,language}));assert(html.includes(result.probe.exactPosition.lower));for(const p of result.probe.methods)assert(html.includes(p.energyDefect.lower));if(language==='en')assert(!/[가-힣]/.test(html));if(result.mode==='equilibrium')assert(html.includes('data-duffing-equilibrium'));if(!result.approximations[1].positiveFrequency)assert(html.includes('data-duffing-invalid-frequency'));}
 const full=renderToStaticMarkup(React.createElement(Lab,{language})),embedded=renderToStaticMarkup(React.createElement(Lab,{language,embedded:true}));assert(full.includes('information-question'));assert(!embedded.includes('information-question'));if(language==='en')assert(!/[가-힣]/.test(full));}assert.equal(calls,0);}finally{if(prior)Object.defineProperty(globalThis,'React',prior);else Reflect.deleteProperty(globalThis,'React');if(worker)Object.defineProperty(globalThis,'Worker',worker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('reader display bundle excludes numerical producer and independent validator',async()=>{
 const result=await build({entryPoints:['src/components/learning/duffing-secular-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,loader:{'.css':'empty'}});const inputs=Object.keys(result.metafile!.inputs);assert(!inputs.some(p=>/learning-duffing-secular(?:\.ts|-validation\.ts|-validation-math\.ts|-evaluator\.ts|-validation-evaluator\.ts)$/.test(p)));assert(inputs.some(p=>p.includes('use-duffing-secular-calculation')));
});

test('space-time reference uses actual local velocities; equilibrium and zero horizon do not acquire fake motion',()=>{
 for(const {id,result}of results){const m=duffingSecularOrbitView(result,37);assert.equal(m.selected,37);for(const p of m.points)for(const v of Object.values(p))assert(Number.isFinite(v),id);if(result.mode==='equilibrium')assert(m.points.every(p=>p.x===0&&p.y===0));if(id==='initial-time')assert(m.points.every(p=>p.z===0));}
 const r=results.find(c=>c.id==='weak-short')!.result,m=duffingSecularOrbitView(r,0);assert(m.points.some(p=>Math.abs(p.y)>1));assert.equal(m.points[0].z,-3);assert.equal(m.points[64].z,3);
});
