import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {graphGaugeDraft,readGraphGaugeDraft} from '../src/lib/learning-graph-gauge-draft';
import {graphGaugeView,graphGaugeEnergyView,graphGaugeLoopView} from '../src/lib/learning-graph-gauge-view';
import {graphGaugeExamples} from '../src/lib/learning-graph-gauge-examples';
import {calculateGraphGauge} from '../src/lib/learning-graph-gauge';
import {parseStoredGraphGauge} from '../src/components/learning/use-saved-graph-gauge';
import {graphGaugeSteps,graphGaugeExperiments,graphGaugeQuestions} from '../src/data/graph-gauge-teaching';
import type {GraphGaugeInput} from '../src/lib/learning-graph-gauge-input';
const example=(id:string)=>{const x=graphGaugeExamples.find(x=>x.id===id);assert(x);return x.input;};
test('all thirteen exact preparations round-trip through the human editor',()=>{
 for(const x of graphGaugeExamples){assert.deepEqual(readGraphGaugeDraft(graphGaugeDraft(x.input)),x.input);assert.deepEqual(parseStoredGraphGauge(JSON.stringify({version:1,value:x.input})),x.input);}
 const reverse=readGraphGaugeDraft({...graphGaugeDraft(example('flat-triangle')),walks:'0: -2 -1 -0'});assert(reverse);assert.deepEqual(reverse.walks[0].steps,[{edge:2,forward:false},{edge:1,forward:false},{edge:0,forward:false}]);
});
test('the editor rejects wrong columns,ambiguous booleans,indices and open or disconnected walks',()=>{
 const d=graphGaugeDraft(example('flat-triangle'));for(const change of [{matter:'1 0\n1 0'},{matter:'1 0 extra\n1 0\n1 0'},{frames:'0 yes\n1 false\n0 true'},{edges:'0 1e0 0 false 1'},{edges:'0 1 0 false 0'},{walks:'0: +0'},{walks:'0: +1 +0 +2'},{walks:'0: 0 1 2'},{walks:'0: +0 +1 +2:'},{walks:'0: +0 +1 +02'},{quartic:'-1'},{quadratic:'1e-3'}])assert.equal(readGraphGaugeDraft({...d,...change}),null);
});
test('storage rejects invalid schema and supports every maximal graph preparation',()=>{
 for(const raw of ['{',JSON.stringify({version:2,value:example('flat-triangle')}),JSON.stringify({version:1,value:{}}),JSON.stringify({version:1,value:example('flat-triangle'),extra:1}),' '.repeat(65537)])assert.equal(parseStoredGraphGauge(raw),null);
 const xs=JSON.parse(readFileSync('scripts/fixtures/graph-gauge-budget-preparations.json','utf8'))as{input:GraphGaugeInput}[];for(const x of xs){assert.deepEqual(parseStoredGraphGauge(JSON.stringify({version:1,value:x.input})),x.input);assert.deepEqual(readGraphGaugeDraft(graphGaugeDraft(x.input)),x.input);}
});
test('three graph views use one amplitude scale and energies retain one signed scale',()=>{
 const r=calculateGraphGauge(example('flat-triangle')),views=(['original','coordinated','matter-only']as const).map(mode=>graphGaugeView(r,mode,0)!);assert(views.every(v=>v.maximum===1));assert(views.every(v=>v.extent===views[0].extent));assert.deepEqual(views[1].nodes.map(n=>[n.re,n.im]),[[1,0],[0,1],[-1,0]]);assert.deepEqual(views[1].nodes,views[2].nodes);assert.deepEqual(views[1].edges.map(e=>[e.re,e.im]),[[0,-1],[0,-1],[-1,0]]);assert.deepEqual(views[2].edges.map(e=>[e.re,e.im]),[[1,0],[1,0],[1,0]]);assert.deepEqual(graphGaugeEnergyView(r)?.normalized,[0,0,1]);
 const negative=calculateGraphGauge({...example('no-edges'),quadratic:'-1',quartic:'0'});assert.deepEqual(graphGaugeEnergyView(negative)?.normalized,[-1,-1,-1]);
});
test('complex loop drawings preserve sign,conjugation and missing selected measurements',()=>{
 const positive=graphGaugeLoopView(calculateGraphGauge(example('positive-flux')),0),negative=graphGaugeLoopView(calculateGraphGauge(example('negative-flux')),0);assert.deepEqual(positive,[[0,1],[0,1],[0,-1]]);assert.deepEqual(negative,[[0,-1],[0,-1],[0,1]]);assert.equal(graphGaugeLoopView(calculateGraphGauge(example('open-chain')),0),null);
});
test('parallel and self-loop curves remain distinct and inside the declared drawing extent',()=>{
 for(const self of [false,true]){const r=calculateGraphGauge({...example('no-edges'),edges:Array.from({length:32},(_,i)=>({from:self?0:i%2,to:self?0:1-i%2,cayley:'1',minusOne:false,weight:'1'}))}),v=graphGaugeView(r,'original',0)!;assert(v.edges.every(e=>e.points.every(p=>Math.abs(p.x)<v.extent&&Math.abs(p.z)<v.extent)));assert(v.extent<5);assert(new Set(v.edges.map(e=>JSON.stringify(e.points[12]))).size===32);}
});
test('guides reference actual preparations and explanations cover invariance limits',()=>{
 assert.equal(graphGaugeSteps.length,8);assert.equal(graphGaugeExperiments.length,6);assert.equal(graphGaugeQuestions.length,7);
 for(const g of graphGaugeExperiments)for(const id of g.examples)assert(graphGaugeExamples.some(x=>x.id===id));
 for(const q of graphGaugeQuestions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko.length>20&&q.explanation.en.length>20);}
 assert(graphGaugeSteps.some(s=>s.body.en.includes('Gauss')));assert(graphGaugeSteps.some(s=>s.body.en.includes('redundant')));
});
test('complete teaching bundle excludes producer and validator arithmetic',async()=>{
 const r=await build({entryPoints:['src/components/learning/graph-gauge-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);for(const name of ['learning-graph-gauge.ts','learning-graph-gauge-validation.ts','learning-graph-gauge-validation-math.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes(`src/lib/${name}`));assert(!r.outputFiles[0].text.includes('BigInt('));
});
