import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {evaluateRealTimePhaseValidationRequest} from '../src/lib/learning-real-time-phase-validation-evaluator';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey,type RealTimePhaseValidatedSuccess} from '../src/lib/learning-real-time-phase-protocol';
import {realTimePhaseMatrixView as view,realTimeMatrixKinds,defaultRealTimeMatrixDisplay as display} from '../src/lib/learning-real-time-phase-matrix-view';
import Matrix from '../src/components/learning/real-time-phase-matrix';
import {mkdirSync,writeFileSync} from 'node:fs';
const root='.runtime/learning/real-time-phase-matrix';mkdirSync(root,{recursive:true});
let cells=0,models=0;const reports:unknown[]=[];
for(const example of realTimePhaseExamples.filter(e=>e.input.mode!=='fold'))test(`validated matrix projections: ${example.id}`,()=>{
  const input=example.input,result=calculateRealTimePhase(input),response={version:1 as const,requestId:1,ok:true as const,kind:'real-time-phase' as const,input,inputKey:realTimePhaseInputKey(input),result};
  const validation=evaluateRealTimePhaseValidationRequest({version:1,requestId:1,input,response});assert.equal(validation.phase,'validated');
  const owned=(validation as {phase:'validated';response:RealTimePhaseValidatedSuccess}).response;
  for(const kind of realTimeMatrixKinds(owned))for(const component of ['real','imaginary'] as const)for(const scale of ['linear','signed-log'] as const){
    const m=view(owned,{...display,kind,component,scale,selectedCell:9999})!;assert(m);assert.equal(m.selected,m.n*m.n-1);assert.equal(m.points.length,m.n*m.n);assert(m.projectionAvailable);
    assert(m.points.every(p=>Number.isFinite(p.y)&&Math.abs(p.y)<=2.5+1e-12));
    for(let i=0;i<m.n;i++)for(let j=0;j<m.n;j++){
      const cell=m.source[i][j];assert.equal(cell.projectionAvailable,true);assert.equal(cell.row,i);assert.equal(cell.column,j);cells++;
      if(m.offset===1){assert.deepEqual(cell.real,m.source[j][i].real);assert.deepEqual(cell.imaginary,m.source[j][i].imaginary);}
      if(cell.exactZero)assert.equal(cell.midpoint,0);
      if(kind==='formal-connected'&&component==='real')assert(cell.exactZero);
      if(kind==='formal-connected')assert.equal(cell.multiplier,input.hbar);
      if(kind==='inverse'||kind==='regulated-connected'||kind==='formal-connected')assert.equal(cell.coefficientBits,512);
    }
    models++;
  }
  if(result.kind==='chain'&&result.exact.inertia.zero){assert(!realTimeMatrixKinds(owned).includes('inverse'));assert.equal(view(owned,{...display,kind:'inverse'}),null);}
  if(example.id==='negative-mode-hand'){const m=view(owned,{...display,kind:'hessian'})!;assert.equal(m.values[0][0],-1);assert.equal(m.points[0].marker,'bar');}
  if(example.id==='zero-leading-pivot-invertible'){const m=view(owned,{...display,kind:'hessian'})!;assert.deepEqual(m.values,[[0,-1],[-1,0]]);assert.equal(m.points[0].marker,'zero');}
  if(example.id==='regulated-free-branch'){
    const m=view(owned,{...display,kind:'regulated-connected',component:'imaginary'})!;
    assert(m.values.some(row=>row.some(x=>x!==0)));assert.deepEqual(m.source[0][1].imaginary,m.source[1][0].imaginary);
  }
  if(example.id==='minimum-width'){const tiny=view(owned,{...display,kind:'initial-covariance'})!;assert(tiny.points.some(p=>p.marker==='tiny'));assert.equal(tiny.source[0][0].exactZero,false);}
  if(example.id==='fixed-width-free')for(const language of ['ko','en'] as const){const html=renderToStaticMarkup(createElement(Matrix,{response:owned,language}));assert(html.includes('data-real-time-selected'));if(language==='en')assert(!/[가-힣]/u.test(html));}
  reports.push({id:example.id,kinds:realTimeMatrixKinds(owned),models,cells});writeFileSync(root+'/projection-progress.json',JSON.stringify({reports,models,cells},null,2)+'\n');
});
test('matrix reader bundle keeps producer, validator and exact arithmetic in separate Workers',async()=>{
  const output=await build({entryPoints:['src/components/learning/real-time-phase-matrix.tsx'],bundle:true,write:false,splitting:true,outdir:'unused',platform:'browser',format:'esm',jsx:'automatic',metafile:true});
  const inputs=Object.keys(output.metafile!.inputs);assert(inputs.some(p=>p.includes('three/')));
  assert(!inputs.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation|-evaluator)|learning-quantum-hilbert-core/.test(p)),inputs.join('\n'));
  assert(Object.values(output.metafile!.outputs).some(x=>Object.keys(x.inputs).some(p=>p.includes('three/'))));
});
