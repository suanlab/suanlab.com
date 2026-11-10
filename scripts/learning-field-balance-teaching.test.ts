import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {fieldBalanceSteps,fieldBalanceExperiments,fieldBalanceQuestions} from '../src/data/field-balance-teaching';
import {fieldBalanceExamples} from '../src/lib/learning-field-balance-examples';
import {calculateFieldBalance} from '../src/lib/learning-field-balance';
import {fieldBalanceView} from '../src/lib/learning-field-balance-view';
import {parseStoredFieldBalance} from '../src/components/learning/use-saved-field-balance';
const example=(id:string)=>fieldBalanceExamples.find(e=>e.id===id)!;
test('worked energy partitions and oriented currents match the independently explained three-site example',()=>{
 for(const[id,energies,currents]of [['symmetric-flow',['7/4','1/2','5/4'],['1/2','0','-1']],['left-share-zero',['5/2','1/2','1/2'],['1','0','0']],['left-share-one',['1','1/2','2'],['0','0','-2']]]as const){const r=calculateFieldBalance(example(id).input);assert.equal(r.totalEnergy,'7/2');assert.deepEqual(r.cells.map(c=>c.energy),energies);assert.deepEqual(r.edges.map(e=>e.energyCurrent),currents);}
 const r=calculateFieldBalance(example('symmetric-flow').input);assert.equal(r.prefixes[1].energy,'9/4');assert.equal(r.prefixes[1].energyDerivative,'-1');assert.equal(r.prefixes[1].energyInflow,'-1');
});
test('teaching counterexamples distinguish symmetry,instantaneous zeros and storage without current',()=>{
 const fixed=calculateFieldBalance(example('fixed-instant-zero').input);assert.equal(fixed.shiftDerivative,'0');assert.equal(fixed.shiftSecondDerivative,'-1');assert.equal(fixed.uniformFieldShiftSymmetry,false);assert.equal(fixed.spatialGenerator,null);
 const quartic=calculateFieldBalance(example('quartic-cyclic').input);assert.equal(quartic.totalEnergy,'29/4');assert.equal(quartic.spatialGenerator?.derivative,'3');assert.equal(quartic.spatialGenerator?.cyclicPermutationSymmetry,true);
 const uniform=calculateFieldBalance(example('uniform-velocity').input);assert(uniform.edges.every(e=>e.energyDerivative==='0'));assert(uniform.edges.some(e=>e.energyCurrent!=='0'));
});
test('energy drawings preserve boundary coordinates without inventing zero-energy wall sites',()=>{
 for(const id of ['symmetric-flow','fixed-instant-zero','neumann-shift']){const r=calculateFieldBalance(example(id).input),view=fieldBalanceView(r,'energy')!;assert.equal(view.points.length,3);assert.equal(view.curve.length,id==='symmetric-flow'?4:3);assert.equal(view.points[0].x,id==='fixed-instant-zero'?1/4:id==='neumann-shift'?1/6:0);}
 const r=calculateFieldBalance(example('symmetric-flow').input);r.cells[0].energy='1/'+'1'+'0'.repeat(400);assert.equal(fieldBalanceView(r,'energy'),null);
});
test('saved nested inputs own both vectors,match version and reject corruption',()=>{
 const input=example('thirty-two-asymmetric').input,raw=JSON.stringify({version:1,value:input}),parsed=parseStoredFieldBalance(raw);assert.deepEqual(parsed,input);assert(Object.isFrozen(parsed?.field.position));
 for(const bad of ['{','x'.repeat(16001),JSON.stringify({version:2,value:input}),JSON.stringify({version:1,value:{...input,share:'2'}}),JSON.stringify({version:1,value:input,extra:true})])assert.equal(parseStoredFieldBalance(bad),null);
});
test('concepts,guides and explained questions are bilingual and use existing examples',()=>{
 assert.equal(fieldBalanceSteps.length,7);assert.equal(fieldBalanceExperiments.length,5);assert.equal(fieldBalanceQuestions.length,6);
 for(const guide of fieldBalanceExperiments){for(const id of guide.examples)assert(example(id));assert(guide.explanation.en.length>140);}
 for(const q of fieldBalanceQuestions)assert(q.correct>=0&&q.correct<q.options.length);
 for(const collection of [fieldBalanceSteps,fieldBalanceExperiments,fieldBalanceQuestions])for(const item of collection)for(const v of Object.values(item))if(v&&typeof v==='object'&&'en'in v)assert(!/[가-힣]/.test((v as {en:string}).en));
});
test('lab and exact result render on the server without accessing Worker or requiring WebGL',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),before=Object.getOwnPropertyDescriptor(globalThis,'React'),worker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{const{default:Lab}=await import('../src/components/learning/field-balance-lab'),{default:Result}=await import('../src/components/learning/field-balance-result');for(const language of ['ko','en']as const){const html=renderToStaticMarkup(React.createElement(Lab,{language})),result=renderToStaticMarkup(React.createElement(Result,{language,result:calculateFieldBalance(example('thirty-two-asymmetric').input)}));assert(html.includes('data-balance-phase="idle"'));assert.equal((html.match(/class="information-question"/g)||[]).length,6);assert(result.includes('data-balance-prefix'));assert(!result.includes('<canvas'));if(language==='en')assert(!/[가-힣]/.test(html+result));}assert.equal(reads,0);}finally{if(before)Object.defineProperty(globalThis,'React',before);else Reflect.deleteProperty(globalThis,'React');if(worker)Object.defineProperty(globalThis,'Worker',worker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('teaching main thread excludes both mathematical engines and loads the existing Three renderer lazily',async()=>{
 const r=await build({entryPoints:['src/components/learning/field-balance-lab.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:'.runtime/learning/field-balance-teaching/graph',metafile:true,jsx:'automatic',logLevel:'silent'});const inputs=Object.keys(r.metafile!.inputs);assert(!inputs.some(p=>/learning-(?:field-balance|scalar-field)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));assert(inputs.includes('src/components/learning/scalar-field-plot-engine.ts'));assert(!r.outputFiles.find(f=>f.path.endsWith('field-balance-lab.js'))!.text.includes('WebGLRenderer'));
});
