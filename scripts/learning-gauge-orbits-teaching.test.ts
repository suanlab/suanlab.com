import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {gaugeOrbitDraft,readGaugeOrbitDraft} from '../src/lib/learning-gauge-orbits-draft';
import {gaugeOrbitView,gaugeOrbitCoordinateView} from '../src/lib/learning-gauge-orbits-view';
import {gaugeOrbitExamples} from '../src/lib/learning-gauge-orbits-examples';
import {calculateGaugeOrbits} from '../src/lib/learning-gauge-orbits';
import {parseStoredGaugeOrbit} from '../src/components/learning/use-saved-gauge-orbits';
import {gaugeOrbitSteps,gaugeOrbitExperiments,gaugeOrbitQuestions} from '../src/data/gauge-orbits-teaching';
import type {GaugeOrbitInput} from '../src/lib/learning-gauge-orbits-input';
const example=(id:string)=>{const x=gaugeOrbitExamples.find(x=>x.id===id);assert(x);return x.input;};
test('all 17 preparations and 5 maximum graphs round-trip through editor and storage',()=>{
 const xs=JSON.parse(readFileSync('scripts/fixtures/gauge-orbits-budget-preparations.json','utf8'))as{input:GaugeOrbitInput}[];
 for(const {input} of [...gaugeOrbitExamples,...xs]){assert.deepEqual(readGaugeOrbitDraft(gaugeOrbitDraft(input)),input);assert.deepEqual(parseStoredGaugeOrbit(JSON.stringify({version:1,value:input})),input);}
});
test('draft rejects incomplete/cyclic forests,duplicate anchors,wrong roots,columns and coerced indices',()=>{
 const draft=gaugeOrbitDraft(example('same-orbit-triangle'));
 for(const change of [{forestEdges:'0'},{forestEdges:'0 1 2'},{roots:'0 1'},{anchors:'0 0'},{anchors:'3'},{sites:'3e0'},{forestEdges:'00 1'},{leftLinks:'0 yes\n0 false\n0 false'},{leftMatter:'1 0 extra\n1 0\n1 0'},{leftMatter:'1 0\n1 0'},{edges:'0 1e0\n1 2\n2 0'}])assert.equal(readGaugeOrbitDraft({...draft,...change}),null,JSON.stringify(change));
});
test('storage rejects schema/version/size and malformed scientific inputs',()=>{
 for(const raw of ['{',JSON.stringify({version:2,value:example('same-orbit-triangle')}),JSON.stringify({version:1,value:{}}),JSON.stringify({version:1,value:example('same-orbit-triangle'),extra:1}),' '.repeat(65537)])assert.equal(parseStoredGaugeOrbit(raw),null);
});
test('both states and every representation share amplitude scale,not independently normalized',()=>{
 const r=calculateGaugeOrbits(example('different-amplitudes'));
 for(const side of ['left','right']as const)for(const scheme of ['full','anchored']as const)for(const rep of ['original','fixed']as const){const view=gaugeOrbitView(r,side,scheme,rep,0)!;assert.equal(view.maximum,2);assert.equal(view.nodes[0].re,side==='left'?.5:1);assert.equal(view.nodes[0].height,side==='left'?.3:1.2);}
 const x=calculateGaugeOrbits(example('same-orbit-triangle'));assert.deepEqual(gaugeOrbitView(x,'right','full','fixed',0)!.nodes.map(n=>[n.re,n.im]),[[1,0],[1,0],[1,0]]);
});
test('forest/root/anchor glyphs and signed open paths match the selected scheme',()=>{
 const r=calculateGaugeOrbits(example('two-incompatible-anchors')),v=gaugeOrbitView(r,'right','anchored','fixed',0)!;
 assert.deepEqual(v.edges.filter(e=>e.forest).map(e=>e.index),[1]);assert.deepEqual(v.nodes.filter(n=>n.anchor).map(n=>n.site),[0,1]);assert.deepEqual(v.nodes.filter(n=>n.root).map(n=>n.site),[0,1]);
 assert(r.right.anchored.coordinates.some(c=>c.kind==='anchored-path'));assert.deepEqual(gaugeOrbitCoordinateView(r,'full',0),[[1,0],[1,0]]);assert.deepEqual(gaugeOrbitCoordinateView(r,'anchored',0),[[1,0],[0,-1]]);
 assert.equal(gaugeOrbitCoordinateView(calculateGaugeOrbits(example('isolated-sites')),'full',0),null);
});
test('32 parallel/reversed edges and self loops have distinct curves within declared bounds',()=>{
 for(const self of [false,true]){const input=example('isolated-sites'),edges=Array.from({length:32},(_,i)=>({from:self?0:i%2,to:self?0:1-i%2})),links=edges.map(()=>({cayley:'0',minusOne:false}));
 const r=calculateGaugeOrbits({...input,edges,leftLinks:links,rightLinks:links,forestEdges:self?[]:[0],roots:self?[0,1,2]:[0,2]}),v=gaugeOrbitView(r,'left','full','original',0)!;assert(v.extent<5);assert(v.edges.every(e=>e.points.every(p=>Math.abs(p.x)<v.extent&&Math.abs(p.z)<v.extent)));assert.equal(new Set(v.edges.map(e=>JSON.stringify(e.points[12]))).size,32);}
});
test('guides cover every example; nine concepts and eight explained questions retain scope',()=>{
 assert.equal(gaugeOrbitSteps.length,9);assert.equal(gaugeOrbitExperiments.length,7);assert.equal(gaugeOrbitQuestions.length,8);
 const covered=new Set(gaugeOrbitExperiments.flatMap(g=>g.examples));for(const x of gaugeOrbitExamples)assert(covered.has(x.id));for(const id of covered)assert(gaugeOrbitExamples.some(x=>x.id===id));
 for(const q of gaugeOrbitQuestions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.en.length>20&&q.explanation.ko.length>20);}
 assert(gaugeOrbitSteps.some(s=>s.body.en.includes('Gauss')));assert(gaugeOrbitSteps.some(s=>s.body.en.includes('zero')));
});
test('teaching reader excludes all producer and independent validator arithmetic',async()=>{
 const r=await build({entryPoints:['src/components/learning/gauge-orbits-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);
 for(const name of ['learning-gauge-orbits.ts','learning-gauge-orbits-validation.ts','learning-graph-gauge-validation-math.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes(`src/lib/${name}`));assert(!r.outputFiles[0].text.includes('BigInt('));
});
