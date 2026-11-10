import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {saddleFluctuationSteps,saddleFluctuationExperiments,saddleFluctuationQuestions,saddleFluctuationLabels} from '../src/data/saddle-fluctuation-teaching';
import {saddleFluctuationDefault,saddleFluctuationExamples} from '../src/lib/learning-saddle-fluctuation-input';
import {calculateSaddleFluctuation} from '../src/lib/learning-saddle-fluctuation';
import {saddleFluctuationDraft,parseSaddleFluctuationDraft,saddleFluctuationDefaultDisplay,saddleFluctuationDefaultLearning,readSaddleFluctuationDisplay,readSaddleFluctuationLearning,saddleFluctuationAlignDisplay,saddleFluctuationChangeMatrix,saddleFluctuationMatrixView,saddleFluctuationPathView,saddleFluctuationSlice,saddleFluctuationWellView,saddleFluctuationCorrectionView} from '../src/lib/learning-saddle-fluctuation-view';
import {parseStoredSaddleFluctuation} from '../src/components/learning/use-saved-saddle-fluctuation';
import Graphic from '../src/components/learning/saddle-fluctuation-graphic';
const example=(id:string)=>saddleFluctuationExamples.find(e=>e.id===id)!.input;
test('complete bilingual teaching material covers every owned example with valid prediction links',()=>{
 assert.equal(saddleFluctuationSteps.length,10);assert.equal(saddleFluctuationExperiments.length,7);assert.equal(saddleFluctuationQuestions.length,8);assert.equal(Object.keys(saddleFluctuationLabels).length,32);
 for(const e of saddleFluctuationExamples)assert(saddleFluctuationLabels[e.id]?.en&&saddleFluctuationLabels[e.id].ko);
 for(const g of saddleFluctuationExperiments){assert(g.predict.en.length>40&&g.explanation.en.length>100);for(const id of g.examples)assert(saddleFluctuationExamples.some(e=>e.id===id),id);}
 for(const q of saddleFluctuationQuestions){assert.equal(q.options.length,3);assert(q.correct>=0&&q.correct<3);assert(q.explanation.en.length>80);}
 const inspect=(x:unknown)=>{if(x&&typeof x==='object'){const d=x as Record<string,unknown>;if(typeof d.en==='string')assert(!/[가-힣]/u.test(d.en),d.en);Object.values(d).forEach(inspect);}};
 [saddleFluctuationSteps,saddleFluctuationExperiments,saddleFluctuationQuestions,saddleFluctuationLabels].forEach(inspect);
});
test('both draft modes preserve all32 raw inputs, including long signed fractions',()=>{
 for(const e of saddleFluctuationExamples)assert.deepEqual(parseSaddleFluctuationDraft(saddleFluctuationDraft(e.input)),e.input);
 const draft=saddleFluctuationDraft(saddleFluctuationDefault);assert(draft.mode==='chain');
 for(const patch of [{intervals:'2.0'},{intervals:'02'},{probe:'1e0'},{quartic:'-1'},{mass:'0'},{start:'-0'},{hbar:'1.0'}])assert.equal(parseSaddleFluctuationDraft({...draft,...patch}),null);
 assert.equal(parseSaddleFluctuationDraft({mode:'well',hbar:'1',kappa:'17'}),null);
});
test('physical matrix selection survives interior/full switches and new inputs align to the declared probe',()=>{
 const input=example('fixed-endpoint-probe');assert(input.mode==='chain');
 const d={...saddleFluctuationDefaultDisplay,selectedCell:13,selectedTime:2};
 const interior=saddleFluctuationChangeMatrix(input,d,'hessian');assert.equal(interior.selectedCell,5);
 assert.equal(saddleFluctuationChangeMatrix(input,interior,'covariance').selectedCell,13);
 const aligned=saddleFluctuationAlignDisplay(input,d);assert.equal(aligned.selectedTime,0);assert.equal(aligned.selectedCell,0);
 const well=saddleFluctuationAlignDisplay(example('double-well'),d);assert.equal(well.selectedTime,0);assert.equal(well.selectedCell,0);
});
test('saved state validates model, dimensions, learning boundaries and strict schemas before restoration',()=>{
 const input=saddleFluctuationDefault,display=saddleFluctuationDefaultDisplay,learning=saddleFluctuationDefaultLearning,value={version:1,input,display,learning};
 assert.deepEqual(parseStoredSaddleFluctuation(JSON.stringify(value)),{input,display,learning});
 for(const patch of [{version:2},{display:{...display,selectedCell:9}},{learning:{...learning,step:10}},{learning:{...learning,experiment:7}},{learning:{...learning,answers:[0]}},{input:{...input,mode:'well'}},{extra:true}])assert.equal(parseStoredSaddleFluctuation(JSON.stringify({...value,...patch})),null);
 assert.equal(parseStoredSaddleFluctuation('x'.repeat(4097)),null);assert.equal(parseStoredSaddleFluctuation('{'),null);
 assert.equal(readSaddleFluctuationDisplay({...display,extra:1},input),null);assert.equal(readSaddleFluctuationLearning({...learning,answers:Array(8).fill(3)}),null);
 let reads=0;const getter={...learning};Object.defineProperty(getter,'step',{get(){reads++;return 0;}});assert.equal(readSaddleFluctuationLearning(getter),null);assert.equal(readSaddleFluctuationDisplay({...display,kind:{toString(){reads++;return 'covariance';}}},input),null);assert.equal(reads,0);
});
test('hand example path uses the certified stationary point and Gaussian covariance',()=>{
 const r=calculateSaddleFluctuation(saddleFluctuationDefault);assert(r.kind==='chain');
 const path=saddleFluctuationPathView(r)!;assert(Math.abs(path.q[1]-.5)<1e-14);assert(Math.abs(path.sigma[1]**2-8/39)<1e-14);
 const slice=saddleFluctuationSlice(r,1)!;assert(Math.abs(slice.gaussianWidth**2-8/39)<1e-14);assert(slice.rows.some(p=>Math.abs(p.weight-p.gaussian)>1e-4));
 assert.equal(saddleFluctuationSlice(r,0),null);assert.equal(saddleFluctuationSlice(r,2),null);assert.equal(saddleFluctuationSlice(r,1.5),null);
 const correction=saddleFluctuationCorrectionView(r)!;assert(correction.rows[0].firstContribution===0);assert(correction.rows.every(p=>p.lowerDifference===null&&p.upperDifference===null));
});
test('coupled coordinate slice and marginal widths differ and endpoint zeros stay visible',()=>{
 const r=calculateSaddleFluctuation(example('coupled-known-saddle'));assert(r.kind==='chain');
 const path=saddleFluctuationPathView(r)!,slice=saddleFluctuationSlice(r,1)!;assert(path.sigma[1]>slice.gaussianWidth);assert.equal(path.sigma[0],0);assert.equal(path.sigma.at(-1),0);
 for(const kind of ['covariance','inverse','hessian'] as const){const view=saddleFluctuationMatrixView(r,{...saddleFluctuationDefaultDisplay,kind});assert.equal(view.n,kind==='hessian'?2:4);assert(view.projectionAvailable);assert.equal(view.points.length,view.n**2);}
});
test('flat Gaussian curvature does not hide a narrow quartic slice behind undersampling',()=>{
 const r=calculateSaddleFluctuation({...saddleFluctuationDefault,mass:'1/999999999989',omegaSquared:'0',end:'0'});assert(r.kind==='chain');
 const s=saddleFluctuationSlice(r,1)!;assert(s.gaussianWidth>s.width*1000);assert(s.rows.filter(p=>p.weight>.1&&p.weight<.9).length>10);
});
test('scalar well curves are relative weights and use a finite critical scale at zero/weak curvature',()=>{
 for(const id of ['single-well','double-well','degenerate-well','weak-well-curvature','narrow-double-well']){
  const r=calculateSaddleFluctuation(example(id));assert(r.kind==='well');const v=saddleFluctuationWellView(r,'right')!;
  assert(v.width>0&&Number.isFinite(v.width));assert(v.local.every(p=>p.weight>=0&&p.weight<=1+1e-12));assert(v.potential.every(p=>p.cost>=0));
  if(id==='degenerate-well'){assert(v.degenerate);assert.equal(v.gaussianWidth,null);assert(v.local.every(p=>p.gaussian===null));}
  if(id==='weak-well-curvature'){assert(v.gaussianWidth!>v.width*1000);assert(v.local.filter(p=>p.weight>.1&&p.weight<.9).length>10);}
  if(id==='double-well'){const left=saddleFluctuationWellView(r,'left')!;assert.equal(left.center,-v.center);assert(Math.abs(v.local[80].weight-1)<1e-14);}
 }
});
test('zero-saddle Taylor drawing is available only under the declared condition',()=>{
 for(const id of ['zero-endpoints-certified','opposite-endpoints','quadratic-reference']){const r=calculateSaddleFluctuation(example(id));assert(r.kind==='chain');const c=saddleFluctuationCorrectionView(r)!;assert(c.rows.every(p=>p.lowerDifference!==null&&p.upperDifference!==null));}
});
test('all20 bilingual reference diagrams render distinct accessible content without a browser',()=>{
 for(const language of ['ko','en'] as const)for(let step=0;step<10;step++){
  const html=renderToStaticMarkup(createElement(Graphic,{step,language}));assert(html.includes('role="img"'));assert(html.includes('<title'));assert(html.includes('<desc'));assert(html.includes(`data-saddle-reference="${step}"`));assert(!html.includes('NaN'));
  if(language==='en')assert(!/[가-힣]/u.test(html));
 }
});
