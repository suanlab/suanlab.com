import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {validateGaugeOrbitResult} from '../src/lib/learning-gauge-orbits-validation';
import {evaluateGaugeOrbitRequest as calculate} from '../src/lib/learning-gauge-orbits-evaluator';
import {evaluateGaugeOrbitValidationRequest as validate} from '../src/lib/learning-gauge-orbits-validation-evaluator';
import {gaugeOrbitExamples} from '../src/lib/learning-gauge-orbits-examples';
import type {GaugeOrbitSuccess} from '../src/lib/learning-gauge-orbits-protocol';
import type {GaugeOrbitInput} from '../src/lib/learning-gauge-orbits-input';
const cases=JSON.parse(readFileSync('scripts/fixtures/gauge-orbits-independent.json','utf8')).cases as {label:string;input:GaugeOrbitInput;expected:{comparison:{full:{jointEquivalent:boolean;connectionsEquivalent:boolean};anchored:{jointEquivalent:boolean;connectionsEquivalent:boolean}}}}[];
const preparation=(id:string)=>{const x=gaugeOrbitExamples.find(x=>x.id===id);assert(x);return x.input;};
for(const x of cases)test(`independent all-edge response verification: ${x.label}`,()=>{
 const started:unknown[]=[],response=calculate({version:1,requestId:1,input:x.input},s=>started.push(s));assert(response.ok);assert.equal(started.length,1);
 for(const group of ['full','anchored']as const){assert.equal(response.result.comparison[group].connectionsEquivalent,x.expected.comparison[group].connectionsEquivalent);assert.equal(response.result.comparison[group].jointEquivalent,x.expected.comparison[group].jointEquivalent);}
 const receipt=validate({version:1,requestId:1,input:x.input,response});assert.equal(receipt.phase,'validated');if(receipt.phase==='validated'){assert.deepEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result.left.full.nodeTree[0]));}
});
test('every individual numerical leaf,phase,path,index,claim and contract rejects mutation',()=>{
 const input=preparation('same-orbit-triangle'),response=calculate({version:1,requestId:2,input});assert(response.ok);const paths:(string|number)[][]=[];
 const visit=(value:unknown,path:(string|number)[])=>{if(value&&typeof value==='object'){for(const[k,v]of Object.entries(value))if(k!=='input')visit(v,[...path,Array.isArray(value)?Number(k):k]);}else paths.push(path);};visit(response.result,[]);
 for(const path of paths){
  // JSON removes intentional full/anchored aliases so exactly one leaf changes.
  const altered=JSON.parse(JSON.stringify(response))as GaugeOrbitSuccess;let parent=altered.result as unknown as Record<string|number,unknown>;for(const k of path.slice(0,-1))parent=parent[k]as Record<string|number,unknown>;const k=path.at(-1)!,old=parent[k];parent[k]=typeof old==='boolean'?!old:typeof old==='number'?old+1:old===null?0:old==='0'?'1':'0';assert.equal(validate({version:1,requestId:2,input,response:altered}).phase,'rejected',path.join('.'));
 }assert.equal(paths.length,462);
});
test('full-group witnesses cannot be substituted for anchored equivalence',()=>{
 for(const id of ['two-incompatible-anchors','global-phase-becomes-observable']){
  const input=preparation(id),response=calculate({version:1,requestId:3,input});assert(response.ok);assert(response.result.comparison.full.jointWitness);assert(!response.result.comparison.anchored.jointEquivalent);
  const altered:GaugeOrbitSuccess=structuredClone(response);altered.result.comparison.anchored.jointEquivalent=true;altered.result.comparison.anchored.jointWitness=altered.result.comparison.full.jointWitness;assert.equal(validate({version:1,requestId:3,input,response:altered}).phase,'rejected');
 }
});
test('partial forests,open anchor paths,zero stabilizers and Gram entries are checked separately',()=>{
 const ids=['anchored-open-chain','two-incompatible-anchors','zero-matter-stabilizer','different-relative-phases','anchor-away-from-root','isolated-sites'];
 for(const id of ids){const input=preparation(id),response=calculate({version:1,requestId:4,input});assert(response.ok);const altered:GaugeOrbitSuccess=structuredClone(response);
  if(id==='anchored-open-chain')altered.result.counts.anchoredConnectionInvariantCount=0;
  if(id==='two-incompatible-anchors')altered.result.anchoredForest.edges=[0,1];
  if(id==='zero-matter-stabilizer'){altered.result.counts.leftJointStabilizerDimension=0;altered.result.counts.leftJointOrbitDimension=3;}
  if(id==='different-relative-phases')altered.result.left.treeGram[0][1]={real:'0',imaginary:'1'};
  if(id==='anchor-away-from-root')altered.result.right.fullTreeFixPossibleWithAnchors=false;
  if(id==='isolated-sites')altered.result.left.treeGram[0][1]={real:'0',imaginary:'0'};
  assert.equal(validate({version:1,requestId:4,input,response:altered}).phase,'rejected',id);
 }
});
test('tree order,roots,anchors,chart spelling and request IDs retain exact attribution',()=>{
 const input=preparation('two-incompatible-anchors'),response=calculate({version:1,requestId:5,input});assert(response.ok);
 for(const other of [{...input,forestEdges:[1,0]},{...input,roots:[1]},{...input,anchors:[1,0]},{...input,leftLinks:input.leftLinks.map(p=>({...p,cayley:'0/3'}))}])assert.equal(validate({version:1,requestId:5,input:other,response}).phase,'rejected');
 assert.equal(validate({version:1,requestId:6,input,response}).phase,'rejected');const altered:GaugeOrbitSuccess=structuredClone(response);altered.kind='graph-gauge' as 'gauge-orbits';assert.equal(validate({version:1,requestId:5,input,response:altered}).phase,'rejected');
});
test('request and result ownership reject accessors and enforce unchanged wire budgets',()=>{
 const input=preparation('same-orbit-triangle');for(const frame of [{version:2,requestId:1,input},{version:1,requestId:0,input},{version:1,requestId:1.5,input},{version:1,requestId:1,input,extra:true},{version:1,requestId:1,input:{...input,forestEdges:[]}}])assert.equal(calculate(frame).ok,false);
 let calls=0;assert.equal(calculate({get version(){calls++;return 1;},requestId:1,input}).ok,false);const response=calculate({version:1,requestId:7,input});assert(response.ok);const accessor=structuredClone(response.result);Object.defineProperty(accessor.left,'treeGram',{get(){calls++;throw Error('Getter');}});assert.equal(validateGaugeOrbitResult(input,accessor),null);assert.equal(calls,0);
 const huge:GaugeOrbitSuccess=structuredClone(response);huge.result.left.originalLinks[0].real='1'.repeat(65537);const receipt=validate({version:1,requestId:7,input,response:huge});assert.equal(receipt.phase,'rejected');if(receipt.phase==='rejected')assert.equal(receipt.error,'resource-limit');
});
const budget=JSON.parse(readFileSync('scripts/fixtures/gauge-orbits-budget-preparations.json','utf8'))as{label:string;input:GaugeOrbitInput}[];
for(const x of budget)test(`maximum wire and independent arithmetic budget: ${x.label}`,()=>{const response=calculate({version:1,requestId:8,input:x.input});assert(response.ok);assert.equal(validate({version:1,requestId:8,input:x.input,response}).phase,'validated');});
test('reader hook excludes exact arithmetic; validator excludes every producer engine',async()=>{
 for(const entry of ['src/components/learning/use-gauge-orbits-calculation.ts','src/lib/learning-gauge-orbits-validation.worker.ts']){const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);for(const file of ['learning-gauge-orbits.ts','learning-graph-gauge.ts','learning-quantum-hilbert-core.ts','learning-gauge-orbits-evaluator.ts'])assert(!inputs.includes(`src/lib/${file}`));if(!entry.includes('validation.worker')){assert(!inputs.includes('src/lib/learning-gauge-orbits-validation.ts'));assert(!r.outputFiles[0].text.includes('BigInt('));}}
});
test('the hook server-renders idle without constructing a Worker',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useGaugeOrbitCalculation}=await import('../src/components/learning/use-gauge-orbits-calculation');const before=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return createElement('span',null,useGaugeOrbitCalculation(preparation('same-orbit-triangle'),true).status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');}
});
