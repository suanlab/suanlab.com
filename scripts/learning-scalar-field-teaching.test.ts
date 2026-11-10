import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
import {scalarFieldSteps,scalarFieldExperiments,scalarFieldQuestions} from '../src/data/scalar-field-teaching';
import {scalarFieldExamples} from '../src/lib/learning-scalar-field-examples';
import {calculateScalarField} from '../src/lib/learning-scalar-field';
import {scalarFieldProjection,scalarFieldView} from '../src/lib/learning-scalar-field-view';
import {parseStoredScalarField} from '../src/components/learning/use-saved-scalar-field';
const example=(id:string)=>scalarFieldExamples.find(x=>x.id===id)!;
test('worked guided comparisons preserve independently stated energies,masses and forces',()=>{
 const expected=[['three-site-power','7/2','1',['-3','0','3']],['same-shape-rest','3','1',['-3','0','3']],['uniform-shift','7/2','1',['-3','0','3']],['half-spacing','25/4','1/2',['-12','0','12']],['periodic-uniform','0','1',['0','0','0']],['neumann-uniform','0','1',['0','0','0']],['fixed-uniform','1','1',['-1','0','-1']],['double-well-minimum','-3/4','1',['0','0','0']]]as const;
 for(const[id,energy,mass,acc]of expected){const r=calculateScalarField(example(id).input);assert.equal(r.energy,energy);assert.equal(r.massPerSite,mass);assert.deepEqual(r.acceleration,acc);}
 assert.equal(calculateScalarField(example('boundary-stabilized').input).localPotentialLowerBound,null);
});
test('drawing projections avoid numerator/denominator overflow and preserve zeros honestly',()=>{
 assert.equal(scalarFieldProjection('3'+'0'.repeat(600)+'/'+'1'+'0'.repeat(600)),3);assert.equal(scalarFieldProjection('-3/2'),-1.5);assert.equal(scalarFieldProjection('0/2'),0);assert.equal(scalarFieldProjection('1/'+'1'+'0'.repeat(400)),null);assert.equal(scalarFieldProjection('1/0'),null);
 const r=calculateScalarField(example('three-site-power').input),view=scalarFieldView(r,'acceleration')!;assert.equal(view.maximum,3);assert.deepEqual(view.points.map(p=>p.y),[-1,0,1]);assert.equal(view.curve.length,4);assert.equal(view.curve[3].site,-1);
 const fixed=scalarFieldView(calculateScalarField(example('fixed-uniform').input),'position')!;assert.deepEqual(fixed.points.map(p=>p.x),[.25,.5,.75]);assert.equal(fixed.curve.length,5);assert.equal(fixed.curve[0].y,0);
 const neumann=scalarFieldView(calculateScalarField(example('neumann-uniform').input),'position')!;assert.equal(neumann.curve.length,3);assert.equal(neumann.points[0].x,1/6);
});
test('stored input is versioned,owned,bounded and rejects malformed or mismatched vectors',()=>{
 const input=example('thirty-two-sites').input,raw=JSON.stringify({version:1,value:input}),saved=parseStoredScalarField(raw);assert.deepEqual(saved,input);assert(Object.isFrozen(saved?.position));
 for(const bad of ['{','x'.repeat(16001),JSON.stringify({version:2,value:input}),JSON.stringify({version:1,value:{...input,position:['1']}}),JSON.stringify({version:1,value:input,extra:true})])assert.equal(parseStoredScalarField(bad),null);
});
test('teaching material has localized concepts,valid guides and explained answers',()=>{
 assert.equal(scalarFieldSteps.length,7);assert.equal(scalarFieldExperiments.length,4);assert.equal(scalarFieldQuestions.length,5);
 for(const guide of scalarFieldExperiments){assert(guide.examples.length>=2);for(const id of guide.examples)assert(example(id),id);assert(guide.explanation.en.length>100);}
 for(const q of scalarFieldQuestions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
 for(const collection of [scalarFieldSteps,scalarFieldExperiments,scalarFieldQuestions])assert(!/[가-힣]/.test(JSON.stringify(collection.map(item=>Object.values(item).filter(value=>value&&typeof value==='object'&&'en'in value).map(value=>(value as {en:string}).en)))));
});
test('lab and full result render without WebGL or Worker access during SSR',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server');const before=Object.getOwnPropertyDescriptor(globalThis,'React'),worker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{const{default:Lab}=await import('../src/components/learning/scalar-field-lab'),{default:Result}=await import('../src/components/learning/scalar-field-result');for(const language of ['ko','en']as const){const html=renderToStaticMarkup(React.createElement(Lab,{language}));assert(html.includes('data-scalar-phase="idle"'));assert.equal((html.match(/class="information-question"/g)||[]).length,5);if(language==='en')assert(!/[가-힣]/.test(html));const result=renderToStaticMarkup(React.createElement(Result,{language,result:calculateScalarField(example('thirty-two-sites').input)}));assert(result.includes('data-scalar-hessian'));assert(!result.includes('<canvas'));assert(!result.includes('NaN'));if(language==='en'){assert(!/[가-힣]/.test(result));assert(result.includes('Canonical momentum sum'));}}assert.equal(reads,0);}finally{if(before)Object.defineProperty(globalThis,'React',before);else Reflect.deleteProperty(globalThis,'React');if(worker)Object.defineProperty(globalThis,'Worker',worker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('teaching bundle excludes mathematical calculation/validation and uses lazy Three drawing',async()=>{
 const r=await build({entryPoints:['src/components/learning/scalar-field-lab.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:'.runtime/learning/scalar-field-teaching/graph',metafile:true,jsx:'automatic',logLevel:'silent'});const inputs=Object.keys(r.metafile!.inputs);assert(!inputs.some(p=>/learning-scalar-field(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));assert(inputs.includes('src/components/learning/scalar-field-plot-engine.ts'));assert(!r.outputFiles.find(f=>f.path.endsWith('scalar-field-lab.js'))!.text.includes('WebGLRenderer'));
});
