import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { euclideanKernelSteps, euclideanKernelExperiments, euclideanKernelQuestions, euclideanKernelLabels } from '../src/data/euclidean-kernel-teaching';
import { euclideanKernelExamples, euclideanKernelDefault, type EuclideanKernelInput } from '../src/lib/learning-euclidean-kernel-input';
import { calculateEuclideanKernel, type EuclideanKernelResult } from '../src/lib/learning-euclidean-kernel';
import { euclideanKernelDraft, parseEuclideanKernelDraft, euclideanKernelAlignDisplay, euclideanKernelChangeMatrix, euclideanKernelMatrixView, euclideanKernelPathView, euclideanKernelComparison, euclideanKernelCovariance, euclideanKernelDefaultDisplay, readEuclideanKernelDisplay, euclideanKernelDefaultLearning, readEuclideanKernelLearning } from '../src/lib/learning-euclidean-kernel-view';
import { parseStoredEuclideanKernel } from '../src/components/learning/use-saved-euclidean-kernel';
const references = JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json','utf8')) as {cases:{id:string;expected:EuclideanKernelResult}[]};
const result = (id: string) => references.cases.find(c=>c.id===id)!.expected;
test('eight bilingual concepts,six comparisons,eight explained questions and20 named inputs agree', () => {
  assert.equal(euclideanKernelSteps.length,8); assert.equal(euclideanKernelExperiments.length,6); assert.equal(euclideanKernelQuestions.length,8); assert.equal(euclideanKernelExamples.length,20);
  for(const e of euclideanKernelExamples) assert(euclideanKernelLabels[e.id].ko&&euclideanKernelLabels[e.id].en);
  for(const s of euclideanKernelSteps) assert(s.body.en.length>100&&s.body.ko.length>50&&s.diagram.ko&&s.diagram.en);
  for(const g of euclideanKernelExperiments){assert(g.predict.en&&g.explanation.en.length>100);for(const id of g.examples)assert(euclideanKernelExamples.some(e=>e.id===id));}
  for(const q of euclideanKernelQuestions)assert(q.correct>=0&&q.correct<q.options.length&&q.explanation.ko&&q.explanation.en);
  assert(!/[가-힣]/.test(JSON.stringify({steps:euclideanKernelSteps.map(s=>[s.title.en,s.body.en,s.diagram.en]),guides:euclideanKernelExperiments.map(g=>[g.title.en,g.predict.en,g.explanation.en]),questions:euclideanKernelQuestions.map(q=>[q.prompt.en,q.options.map(o=>o.en),q.explanation.en]),labels:Object.values(euclideanKernelLabels).map(l=>l.en)})));
});
test('all20 inputs decode complete matrices and finite path projections without recomputing exact arithmetic', () => {
  for(const e of euclideanKernelExamples){const r=result(e.id),path=euclideanKernelPathView(r);assert(path);assert.equal(path.times.length,e.input.intervals+1);assert(path.sigma.every(v=>Number.isFinite(v)&&v>=0));assert(path.continuous.every(Number.isFinite));
    const covariance=euclideanKernelCovariance(r);assert.equal(covariance.length,e.input.intervals+1);assert(covariance.every(row=>row.length===covariance.length));
    for(const kind of ['position','precision']as const)for(const scale of ['linear','signed-log']as const){const m=euclideanKernelMatrixView(r,{...euclideanKernelDefaultDisplay,kind,scale});assert(m.projectionAvailable);assert.equal(m.points.length,m.n*m.n);assert(m.points.every(p=>[p.x,p.y,p.z].every(Number.isFinite)));assert.equal(m.n,e.input.intervals+(kind==='position'?1:-1));if(m.n===1)assert.deepEqual(m.points.map(p=>[p.x,p.z]),[[0,0]]);}
  }
});
test('underflow is unavailable rather than zero,and actual fixed endpoints retain exact zero variance', () => {
  const r=result('one-variable-hand-check');assert.equal(euclideanKernelPathView(r)!.sigma[0],0);
  const changed=structuredClone(r),index=changed.covariance.indices[1][1];changed.covariance.values[Math.floor(index/128)][index%128]='1/1'+'0'.repeat(400);
  assert.equal(euclideanKernelPathView(changed),null);assert.equal(euclideanKernelMatrixView(changed,euclideanKernelDefaultDisplay).projectionAvailable,false);
});
test('draft parsing preserves26-character signed fractions and rejects rounding and silent clamping', () => {
  for(const e of euclideanKernelExamples)assert.deepEqual(parseEuclideanKernelDraft(euclideanKernelDraft(e.input)),e.input);
  const signed={...euclideanKernelDefault,start:'-999999999983/999999999989',end:'-999999999983/999999999989'};assert.deepEqual(parseEuclideanKernelDraft(euclideanKernelDraft(signed)),signed);
  for(const patch of [{intervals:'2.0'},{intervals:'2.0000000000000001'},{intervals:'33'},{intervals:'02'},{probe:'1e0'},{probe:'3'},{time:'0.5'},{hbar:'0'},{omegaSquared:'-1'}])assert.equal(parseEuclideanKernelDraft({...euclideanKernelDraft(euclideanKernelDefault),...patch}),null);
});
test('owned display and versioned storage reject stale ranges,extra fields and getters', () => {
  const display={kind:'precision'as const,scale:'signed-log'as const,selectedTime:2,selectedCell:0},value={version:1,input:euclideanKernelDefault,display,learning:euclideanKernelDefaultLearning};
  assert.deepEqual(parseStoredEuclideanKernel(JSON.stringify(value)),{input:euclideanKernelDefault,display,learning:euclideanKernelDefaultLearning});
  for(const patch of [{selectedTime:3},{selectedCell:1},{selectedCell:-1},{selectedTime:1.5},{kind:'increment'},{scale:'log'},{unknown:true}])assert.equal(parseStoredEuclideanKernel(JSON.stringify({...value,display:{...display,...patch}})),null);
  for(const raw of ['{','{}',JSON.stringify({...value,version:2}),JSON.stringify({...value,input:{...euclideanKernelDefault,hbar:'0'}}),' '.repeat(4097)])assert.equal(parseStoredEuclideanKernel(raw),null);
  let calls=0;const getter={...display};Object.defineProperty(getter,'kind',{get(){calls++;return'position';}});assert.equal(readEuclideanKernelDisplay(getter,euclideanKernelDefault),null);assert.equal(calls,0);
});
test('reference figures,concept and cover render both languages without Worker access', async () => {
  const require=createRequire(`${process.cwd()}/package.json`);require.extensions['.css']=()=>{};
  const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),savedReact=Object.getOwnPropertyDescriptor(globalThis,'React'),savedWorker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
  try{const{default:Graphic}=await import('../src/components/learning/euclidean-kernel-graphic'),{default:Concept}=await import('../src/components/learning/euclidean-kernel-concept'),{default:Cover}=await import('../src/components/learning/euclidean-kernel-cover');const{LanguageProvider}=require('./src/components/language-provider')as typeof import('../src/components/language-provider');
    for(const language of ['ko','en']as const){for(let step=0;step<8;step++){const html=renderToStaticMarkup(React.createElement(Graphic,{step,language}));assert(html.includes('<title')&&html.includes('<desc')&&html.includes(`data-euclidean-reference="${step}"`));assert(!/NaN|Infinity/.test(html));if(language==='en')assert(!/[가-힣]/.test(html));}assert(renderToStaticMarkup(React.createElement(Concept,{language})).includes('href="#experiment"'));}
    assert(renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Cover))).includes('min-width:0'));assert.equal(reads,0);
  }finally{if(savedReact)Object.defineProperty(globalThis,'React',savedReact);else Reflect.deleteProperty(globalThis,'React');if(savedWorker)Object.defineProperty(globalThis,'Worker',savedWorker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('the complete lab excludes exact production and independent mathematical engines', async () => {
  const bundle=await build({entryPoints:['src/components/learning/euclidean-kernel-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,jsx:'automatic'}),inputs=Object.keys(bundle.metafile!.inputs);
  for(const p of ['learning-euclidean-kernel.ts','learning-euclidean-kernel-validation.ts','learning-euclidean-kernel-evaluator.ts','learning-euclidean-kernel-validation-evaluator.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+p));
});
test('learning progress rejects out-of-range steps,answers and sparse/accessor arrays', () => {
  assert.deepEqual(readEuclideanKernelLearning(euclideanKernelDefaultLearning),euclideanKernelDefaultLearning);
  for(const patch of [{step:8},{experiment:6},{explanationOpen:1},{answers:Array(8).fill(3)},{answers:Array(7).fill(-1)},{answers:new Array(8)}])assert.equal(readEuclideanKernelLearning({...euclideanKernelDefaultLearning,...patch}),null);
  const answers=Array(8).fill(-1);let calls=0;Object.defineProperty(answers,'0',{get(){calls++;return 0;}});assert.equal(readEuclideanKernelLearning({...euclideanKernelDefaultLearning,answers}),null);assert.equal(calls,0);
});
test('Mehler display diagnostics agree with independent80-digit references or explicitly hide unresolved differences', () => {
  const refs=JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-display-reference.json','utf8'))as{cases:{id:string;input:EuclideanKernelInput;continuousAction:string;continuousLogTrace:string|null;rows:{symmetricLogRatio:string;rightLogRatio:string}[]}[]};
  for(const c of refs.cases){const r=references.cases.find(v=>v.id===c.id)?.expected??calculateEuclideanKernel(c.input),view=euclideanKernelComparison(r);assert(view);
    const close=(a:number,b:number)=>assert(Math.abs(a-b)<=1e-11*Math.max(1,Math.abs(b)),`${c.id}:${a} != ${b}`);close(view.action,Number(c.continuousAction));if(c.continuousLogTrace!==null)close(view.logTrace!,Number(c.continuousLogTrace));
    view.rows.forEach((row,i)=>{assert(row);if(row.symmetric!==null)close(row.symmetric,Number(c.rows[i].symmetricLogRatio));if(row.right!==null)close(row.right,Number(c.rows[i].rightLogRatio));});
    if(c.id==='large-log-unresolved')assert(view.rows.every(row=>row?.symmetric===null&&row.right===null));
    if(c.id==='free-particle')assert(view.free&&view.logTrace===null&&view.rows.every(row=>row?.symmetric===0&&row.right===0&&row.exactFreeIdentity));
  }
});

test('new inputs align to the declared probe and matrix changes preserve physical index pairs', () => {
  for(const e of euclideanKernelExamples){for(const kind of ['position','precision']as const){const display=euclideanKernelAlignDisplay(e.input,{...euclideanKernelDefaultDisplay,kind});assert.equal(display.selectedTime,e.input.probe);assert(readEuclideanKernelDisplay(display,e.input));}}
  const input={...euclideanKernelDefault,intervals:32,probe:17},position=euclideanKernelAlignDisplay(input,euclideanKernelDefaultDisplay);assert.equal(position.selectedCell,17*33+17);
  const precision=euclideanKernelChangeMatrix(input,position,'precision');assert.equal(precision.selectedCell,16*31+16);assert.deepEqual(euclideanKernelChangeMatrix(input,precision,'position'),position);
  const endpoint=euclideanKernelChangeMatrix(input,{...position,selectedCell:32},'precision');assert.equal(endpoint.selectedCell,30);assert.equal(endpoint.selectedTime,17);
});
