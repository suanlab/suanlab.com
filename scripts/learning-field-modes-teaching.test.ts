import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {fieldModesSteps,fieldModesExperiments,fieldModesQuestions} from '../src/data/field-modes-teaching';
import {fieldModesExamples} from '../src/lib/learning-field-modes-examples';
import {calculateFieldModes} from '../src/lib/learning-field-modes';
import {fieldModeSummary,fieldModeMidpoint,fieldModesView,fieldModesSeries} from '../src/lib/learning-field-modes-view';
import {parseStoredFieldModes} from '../src/components/learning/use-saved-field-modes';
const example=(id:string)=>fieldModesExamples.find(x=>x.id===id)!;
test('worked modal energy and nonlinear decomposition agree with the teaching comparisons',()=>{
 const r=calculateFieldModes(example('four-site-cosine').input);for(const[m,energy]of [[0,1/8],[1,9/4],[2,0],[3,1/8]])assert(Math.abs(fieldModeMidpoint(r.modes[m].quadraticEnergy)!-energy)<1e-12);assert.equal(r.exact.fullEnergy,'5/2');
 const well=calculateFieldModes(example('double-well-reference').input);assert.equal(well.exact.referencePotential,'-1');assert.equal(well.exact.quadraticEnergy,'9/2');assert.equal(well.exact.nonlinearRemainder,'1/2');assert.equal(well.exact.fullEnergy,'4');
 const drift=calculateFieldModes(example('zero-curvature-drift').input);assert.equal(drift.modes[0].classification,'zero-curvature');assert.notEqual(fieldModeMidpoint(drift.modes[0].velocity),0);
});
test('plot projections distinguish exact zero from a rounded interval and preserve boundary coordinates',()=>{
 assert.equal(fieldModeSummary({lower:'0',upper:'0'}),'0');assert(fieldModeSummary({lower:'-1/1000000',upper:'1/1000000'})!.startsWith('≈'));assert.equal(fieldModeMidpoint({lower:'1/'+'1'+'0'.repeat(400),upper:'1'}),null);
 const periodic=fieldModesView(calculateFieldModes(example('four-site-cosine').input),'shape')!;assert.equal(periodic.curve.length,5);assert.equal(periodic.points.length,4);
 const fixed=fieldModesView(calculateFieldModes(example('fixed-length-dirichlet').input),'shape')!;assert.equal(fixed.points[0].x,1/5);assert.equal(fixed.curve[0].y,0);assert.equal(fixed.curve.length,6);
 const neumann=fieldModesView(calculateFieldModes(example('cell-centered-neumann').input),'shape')!;assert.equal(neumann.points[0].x,1/8);assert.equal(neumann.curve.length,4);
 const r=calculateFieldModes(example('four-site-cosine').input),series=fieldModesSeries(r,'quadraticEnergy')!;assert.equal(series.points[1].y,1);assert.equal(series.maximum,9/4);
});
test('guided cutoff comparison holds separate Nyquist selections distinct from fixed-m refinement',()=>{
 const coarse=calculateFieldModes(example('periodic-nyquist').input),fine=calculateFieldModes(example('thirty-two-nyquist').input);assert.equal(coarse.refinements[0].physicalWave,2);assert.equal(fine.refinements[0].physicalWave,16);assert(Math.abs(fieldModeMidpoint(coarse.refinements[0].relativeWaveOperatorGap)!-fieldModeMidpoint(fine.refinements[0].relativeWaveOperatorGap)!)<1e-14);
 assert(coarse.refinements.every(row=>row.physicalWave===2));assert(fieldModeMidpoint(coarse.refinements[1].relativeWaveOperatorGap)!<fieldModeMidpoint(coarse.refinements[0].relativeWaveOperatorGap)!);
});
test('stored mode preparations own vectors and reject corruption,wrong versions and invalid indices',()=>{
 const input=example('thirty-two-components').input,raw=JSON.stringify({version:1,value:input}),saved=parseStoredFieldModes(raw);assert.deepEqual(saved,input);assert(Object.isFrozen(saved?.position));
 for(const bad of ['{','x'.repeat(16001),JSON.stringify({version:2,value:input}),JSON.stringify({version:1,value:{...input,selectedBasis:32}}),JSON.stringify({version:1,value:input,extra:true})])assert.equal(parseStoredFieldModes(bad),null);
});
test('all localized guides refer to actual presets and provide explained answers',()=>{
 assert.equal(fieldModesSteps.length,8);assert.equal(fieldModesExperiments.length,6);assert.equal(fieldModesQuestions.length,7);
 for(const g of fieldModesExperiments){for(const id of g.examples)assert(example(id));assert(g.explanation.en.length>140);}
 for(const q of fieldModesQuestions)assert(q.correct>=0&&q.correct<q.options.length);
 for(const collection of [fieldModesSteps,fieldModesExperiments,fieldModesQuestions])for(const item of collection)for(const v of Object.values(item))if(v&&typeof v==='object'&&'en'in v)assert(!/[가-힣]/.test((v as{en:string}).en));
});
test('server rendering uses no Worker or WebGL and keeps all seven explained questions',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),previous=Object.getOwnPropertyDescriptor(globalThis,'React'),worker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{const{default:Lab}=await import('../src/components/learning/field-modes-lab'),{default:Result}=await import('../src/components/learning/field-modes-result');for(const language of ['ko','en']as const){const lab=renderToStaticMarkup(React.createElement(Lab,{language})),result=renderToStaticMarkup(React.createElement(Result,{language,result:calculateFieldModes(example('thirty-two-components').input)}));assert(lab.includes('data-modes-phase="idle"'));assert.equal((lab.match(/class="information-question"/g)||[]).length,7);assert(result.includes('data-modes-refinement'));assert(!result.includes('<canvas'));assert(!result.includes('NaN'));if(language==='en')assert(!/[가-힣]/.test(lab+result));}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'React',previous);else Reflect.deleteProperty(globalThis,'React');if(worker)Object.defineProperty(globalThis,'Worker',worker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('teaching bundle excludes numerical engines and lazy-loads the existing Three renderer',async()=>{
 const r=await build({entryPoints:['src/components/learning/field-modes-lab.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:'.runtime/learning/field-modes-teaching/graph',metafile:true,jsx:'automatic',logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);assert(!inputs.some(p=>/learning-field-modes(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)|learning-spin-angle\.ts|learning-quantum-evolution-interval\.ts/.test(p)));assert(inputs.includes('src/components/learning/scalar-field-plot-engine.ts'));assert(!r.outputFiles.find(f=>f.path.endsWith('field-modes-lab.js'))!.text.includes('WebGLRenderer'));
});
