import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateGraphGauge,type GraphGaugeResult} from '../src/lib/learning-graph-gauge';
import {readGraphGaugeInput,graphGaugeInputKey,type GraphGaugeInput} from '../src/lib/learning-graph-gauge-input';
import {graphGaugeExamples} from '../src/lib/learning-graph-gauge-examples';
const zero={real:'0',imaginary:'0'};
const example=(id:string)=>{const found=graphGaugeExamples.find(x=>x.id===id);assert(found);return found.input;};
type Reference={label:string;input:GraphGaugeInput;expected:{componentCount:number;cycleRank:number;energy:Record<string,string>;operator:{original:unknown;transformed:unknown};gradients:unknown[];transformedGradients:unknown[];edges:Record<string,unknown>[];walks:Record<string,unknown>[]}};
const references=JSON.parse(readFileSync('scripts/fixtures/graph-gauge-independent.json','utf8')).cases as Reference[];
for(const x of references)test(`independent quadratic polarization and polynomial gradient: ${x.label}`,()=>{
 const r=calculateGraphGauge(x.input),e=x.expected;assert.equal(r.componentCount,e.componentCount);assert.equal(r.cycleRank,e.cycleRank);
 for(const [k,v] of Object.entries(e.energy))assert.equal(r.energy[k as keyof typeof r.energy],v,k);
 assert.deepEqual(r.operator.original,e.operator.original);assert.deepEqual(r.operator.transformed,e.operator.transformed);
 assert.deepEqual(r.nodes.map(z=>z.gradient),e.gradients);assert.deepEqual(r.nodes.map(z=>z.transformedGradient),e.transformedGradients);
 e.edges.forEach((row,i)=>{for(const [k,v]of Object.entries(row))assert.deepEqual(r.edges[i][k as keyof GraphGaugeResult['edges'][number]],v,`edge ${i}/${k}`);});
 e.walks.forEach((row,i)=>{for(const [k,v]of Object.entries(row))assert.deepEqual(r.walks[i][k as keyof GraphGaugeResult['walks'][number]],v,`walk ${i}/${k}`);});
 for(const row of r.operator.covarianceResidual)for(const z of row)assert.deepEqual(z,zero);
 for(const row of r.operator.hermiticityResidual)for(const z of row)assert.deepEqual(z,zero);
 for(const node of r.nodes){assert.equal(node.densityResidual,'0');assert.deepEqual(node.gradientCovarianceResidual,zero);}
 for(const edge of r.edges){assert.equal(edge.unitNormResidual,'0');assert.equal(edge.transformedUnitNormResidual,'0');assert.deepEqual(edge.covarianceResidual,zero);assert.deepEqual(edge.dressedResidual,zero);}
 for(const walk of r.walks){assert.equal(walk.unitNormResidual,'0');assert.deepEqual(walk.invarianceResidual,zero);}
 assert.equal(r.energy.invarianceResidual,'0');assert.deepEqual(r.energy.quadraticOperatorResidual,zero);
 assert.equal(r.operator.trace,r.operator.transformedTrace);assert.equal(r.operator.frobeniusSquared,r.operator.transformedFrobeniusSquared);
 assert.equal(r.classification.componentwiseConstantFrame,r.classification.linksUnchanged);
});
test('all current teaching preparations are represented by the independently generated fixture',()=>{
 const exported=JSON.parse(readFileSync('scripts/fixtures/graph-gauge-preparations.json','utf8'));assert.deepEqual(exported,JSON.parse(JSON.stringify(graphGaugeExamples)));
 for(const x of graphGaugeExamples){const reference=references.find(r=>r.label===x.id);assert(reference,x.id);assert.deepEqual(reference.input,x.input);}
});
test('worked triangle distinguishes a coordinate transformation from rotating matter alone',()=>{
 const r=calculateGraphGauge(example('flat-triangle'));assert.equal(r.energy.total,'0');assert.equal(r.energy.transformedTotal,'0');assert.equal(r.energy.fieldOnlyTotal,'8');assert.deepEqual(r.edges.map(e=>e.fieldOnlyEnergy),['2','2','4']);assert.deepEqual(r.edges.map(e=>e.transformedLink),[{real:'0',imaginary:'-1'},{real:'0',imaginary:'-1'},{real:'-1',imaginary:'0'}]);
});
test('real loop penalties,selected walks and matter observables have different information',()=>{
 const positive=calculateGraphGauge(example('positive-flux')),negative=calculateGraphGauge(example('negative-flux'));
 assert.equal(positive.energy.total,'3');assert.equal(negative.energy.total,'3');assert.equal(positive.walks[0].holonomy.real,negative.walks[0].holonomy.real);assert.notDeepEqual(positive.walks[0].holonomy,negative.walks[0].holonomy);
 const hidden=calculateGraphGauge(example('hidden-flux'));assert.equal(hidden.cycleRank,1);assert.deepEqual(hidden.walks[0].holonomy,{real:'1',imaginary:'0'});assert.equal(hidden.energy.total,'0');assert.equal(hidden.classification.completeLoopBasisClaimed,false);
 const reversed=calculateGraphGauge(example('reverse-walk'));assert.deepEqual(reversed.walks[0].reverseHolonomy,reversed.walks[1].holonomy);
});
test('self-loops,parallel edges,isolated sites and componentwise frames are retained',()=>{
 const self=calculateGraphGauge(example('self-loop'));assert.equal(self.operator.original[0][0].real,'2');assert.equal(self.energy.total,'3');assert.equal(self.componentCount,3);assert.equal(self.cycleRank,1);
 const parallel=calculateGraphGauge(example('parallel-edges'));assert.equal(parallel.componentCount,2);assert.equal(parallel.cycleRank,1);
 const disconnected=calculateGraphGauge(example('disconnected'));assert.equal(disconnected.classification.globalUniformFrame,false);assert.equal(disconnected.classification.componentwiseConstantFrame,true);assert.equal(disconnected.classification.linksUnchanged,true);
 const empty=calculateGraphGauge(example('no-edges'));assert.equal(empty.componentCount,3);assert.equal(empty.cycleRank,0);assert.equal(empty.energy.total,'8/5');
});
test('input ownership rejects accessors,holes,wrong graph walks and coercion',()=>{
 const p=example('flat-triangle'),owned=readGraphGaugeInput(p);assert(owned);assert.notEqual(owned,p);assert(Object.isFrozen(owned.walks[0].steps));assert(Object.isFrozen(owned.matter[0]));assert.equal(graphGaugeInputKey(owned),graphGaugeInputKey(p));assert.notEqual(graphGaugeInputKey({...p,quadratic:'0/3'}),graphGaugeInputKey(p));
 for(const value of [{...p,extra:1},{...p,frames:p.frames.slice(1)},{...p,quadratic:'0.1'},{...p,quartic:'-1'},{...p,loopCoupling:'-1'},{...p,quadratic:'1/0'},{...p,quadratic:'1000000000000000000'}, {...p,edges:[{...p.edges[0],weight:'0'}]},{...p,edges:[{...p.edges[0],from:'0'}]},{...p,frames:[{cayley:'1',minusOne:true},...p.frames.slice(1)]},{...p,walks:[{start:0,steps:[{edge:0,forward:true}]}]},{...p,walks:[{start:0,steps:[{edge:1,forward:true}]}]},{...p,walks:[{start:0,steps:[{edge:0,forward:1}]}]},{...p,walks:[{start:0,steps:[]}]},{...p,edges:[]}])assert.equal(readGraphGaugeInput(value),null);
 let calls=0;const getter={...p};Object.defineProperty(getter,'quadratic',{get(){calls++;throw Error('accessor');}});assert.equal(readGraphGaugeInput(getter),null);assert.equal(calls,0);
 const hole=[...p.matter];delete hole[1];assert.equal(readGraphGaugeInput({...p,matter:hole}),null);const symbol={...p,[Symbol('extra')]:1};assert.equal(readGraphGaugeInput(symbol),null);
 assert.throws(()=>calculateGraphGauge(getter),RangeError);
});
test('reader preparation bundle excludes exact arithmetic and calculator implementations',async()=>{
 const bundle=await build({entryPoints:['src/lib/learning-graph-gauge-examples.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});const names=Object.keys(bundle.metafile!.inputs);assert(names.every(n=>!n.includes('hilbert-core')));assert(!bundle.outputFiles[0].text.includes('BigInt('));
});
test('largest declared graph and long rational inputs finish with exact invariant identities',()=>{
 const p=example('twelve-site-ring');const edges=[...p.edges,...Array.from({length:20},(_,i)=>({...p.edges[i%12],cayley:`${i+1}/999999999999999989`,weight:'1/999999999999999989'}))];
 const input={...p,edges,frames:p.frames.map((g,i)=>({...g,cayley:`${i+1}/999999999999999989`})),walks:Array.from({length:8},(_,i)=>({start:i,steps:Array.from({length:32},(_,j)=>({edge:i,forward:j%2===0}))}))};
 const r=calculateGraphGauge(input);assert.equal(r.edgeCount,32);assert.equal(r.sites,12);assert.equal(r.energy.invarianceResidual,'0');assert(r.operator.covarianceResidual.flat().every(z=>z.real==='0'&&z.imaginary==='0'));assert(r.nodes.every(z=>z.gradientCovarianceResidual.real==='0'&&z.gradientCovarianceResidual.imaginary==='0'));
});
