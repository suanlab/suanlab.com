import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateGaugeOrbits,type GaugeOrbitResult} from '../src/lib/learning-gauge-orbits';
import {readGaugeOrbitInput,gaugeOrbitInputKey,type GaugeOrbitInput} from '../src/lib/learning-gauge-orbits-input';
import {gaugeOrbitExamples} from '../src/lib/learning-gauge-orbits-examples';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {Rational as R,complex,unitPhase,times,conjugate,minus,norm,wire} from '../src/lib/learning-graph-gauge-validation-math';
const zero={real:'0',imaginary:'0'},one={real:'1',imaginary:'0'};
const example=(id:string)=>{const x=gaugeOrbitExamples.find(x=>x.id===id);assert(x);return x.input;};
type Reference={label:string;input:GaugeOrbitInput;expected:{componentCount:number;cycleRank:number;counts:unknown;anchoredForest:unknown;left:Pick<GaugeOrbitResult['left'],'full'|'anchored'|'treeGram'>;right:Pick<GaugeOrbitResult['right'],'full'|'anchored'|'treeGram'>;comparison:{full:{connectionsEquivalent:boolean;jointEquivalent:boolean};anchored:{connectionsEquivalent:boolean;jointEquivalent:boolean}}}};
const cases=JSON.parse(readFileSync('scripts/fixtures/gauge-orbits-independent.json','utf8')).cases as Reference[];
for(const x of cases)test(`independent all-edge orbit constraints and exhaustive anchor forest: ${x.label}`,()=>{
 const r=calculateGaugeOrbits(x.input),e=x.expected;assert.equal(r.componentCount,e.componentCount);assert.equal(r.cycleRank,e.cycleRank);assert.deepEqual(r.counts,e.counts);assert.deepEqual(r.anchoredForest,e.anchoredForest);
 for(const side of ['left','right']as const){assert.deepEqual(r[side].full,e[side].full);assert.deepEqual(r[side].anchored,e[side].anchored);assert.deepEqual(r[side].treeGram,e[side].treeGram);assert(r[side].full.treeResiduals.every(x=>x.value.real==='0'&&x.value.imaginary==='0'));assert(r[side].anchored.anchorResiduals.every(x=>x.value.real==='0'&&x.value.imaginary==='0'));}
 for(const group of ['full','anchored']as const){const actual=r.comparison[group],expected=e.comparison[group];assert.equal(actual.connectionsEquivalent,expected.connectionsEquivalent);assert.equal(actual.jointEquivalent,expected.jointEquivalent);assert.equal(actual.completeJointInvariantCriterion,expected.jointEquivalent);assert(actual.criterionAgreesWithWitness);
  for(const kind of ['connectionWitness','jointWitness']as const){const witness=actual[kind];assert.equal(witness!==null,kind==='connectionWitness'?expected.connectionsEquivalent:expected.jointEquivalent);if(!witness)continue;
   // Reapply the witness with arithmetic independent of the producer core.
   const phases=witness.phases.map(z=>complex(R.read(z.real),R.read(z.imaginary)));assert(phases.every(g=>norm(g).text()==='1'));
   x.input.edges.forEach((edge,k)=>assert.deepEqual(wire(minus(times(times(phases[edge.from],unitPhase(x.input.leftLinks[k])),conjugate(phases[edge.to])),unitPhase(x.input.rightLinks[k]))),zero));
   if(kind==='jointWitness')x.input.leftMatter.forEach((z,i)=>assert.deepEqual(wire(minus(times(phases[i],complex(R.read(z.real),R.read(z.imaginary))),complex(R.read(x.input.rightMatter[i].real),R.read(x.input.rightMatter[i].imaginary)))),zero));
   if(group==='anchored')for(const i of x.input.anchors)assert.deepEqual(witness.phases[i],one);
  }
 }
});
test('all current prepared inputs are covered by the independent reference export',()=>{
 assert.deepEqual(JSON.parse(readFileSync('scripts/fixtures/gauge-orbits-preparations.json','utf8')),JSON.parse(JSON.stringify(gaugeOrbitExamples)));
 for(const x of gaugeOrbitExamples){const reference=cases.find(r=>r.label===x.id);assert(reference);assert.deepEqual(reference.input,x.input);}
});
test('two anchors expose a difference invisible to closed-loop products',()=>{
 const r=calculateGaugeOrbits(example('two-incompatible-anchors'));assert(r.comparison.full.jointEquivalent);assert(!r.comparison.anchored.connectionsEquivalent);assert.equal(r.counts.freeConnectionInvariantCount,1);assert.equal(r.counts.anchoredConnectionInvariantCount,2);assert.deepEqual(r.anchoredForest.edges,[1]);assert(r.right.anchored.coordinates.some(c=>c.kind==='anchored-path'));assert(!r.right.fullTreeFixPossibleWithAnchors);
 const chain=calculateGaugeOrbits(example('anchored-open-chain'));assert.equal(chain.cycleRank,0);assert(chain.comparison.full.connectionsEquivalent);assert(!chain.comparison.anchored.connectionsEquivalent);
});
test('root-normalized fixing and existence of an allowed full-tree fixing are distinct',()=>{
 const away=calculateGaugeOrbits(example('anchor-away-from-root'));assert(!away.right.fullRootNormalizedFixAllowedAtAnchors);assert(away.right.fullTreeFixPossibleWithAnchors);assert(away.comparison.anchored.connectionsEquivalent);assert(!away.comparison.anchored.jointEquivalent);
 const compatible=calculateGaugeOrbits(example('compatible-anchor-path'));assert(compatible.left.fullTreeFixPossibleWithAnchors);assert(compatible.comparison.anchored.jointEquivalent);
});
test('Gram information,anchored matter and zero-field stabilizers are retained',()=>{
 const phase=calculateGaugeOrbits(example('different-relative-phases'));assert(phase.comparison.full.connectionsEquivalent);assert(!phase.comparison.full.jointEquivalent);assert(phase.comparison.full.gramMismatchPairs.length>0);
 const zeroState=calculateGaugeOrbits(example('zero-matter-stabilizer'));assert.equal(zeroState.counts.leftJointStabilizerDimension,1);assert(zeroState.comparison.full.matterComponents[0].phaseFree);assert.equal(zeroState.counts.leftJointOrbitDimension,2);
 const pinned=calculateGaugeOrbits(example('global-phase-becomes-observable'));assert(pinned.comparison.full.jointEquivalent);assert(pinned.comparison.anchored.connectionsEquivalent);assert(!pinned.comparison.anchored.jointEquivalent);
});
test('tree and root changes preserve the orbit decision and its valid joint witness',()=>{
 const a=calculateGaugeOrbits(example('same-orbit-triangle')),b=calculateGaugeOrbits(example('alternative-tree-root'));assert.deepEqual(a.comparison.full.jointWitness?.phases,b.comparison.full.jointWitness?.phases);assert.notDeepEqual(a.left.full.nodeTree,b.left.full.nodeTree);assert.equal(a.comparison.full.jointEquivalent,b.comparison.full.jointEquivalent);
});
test('forest,root and anchor validation owns records and rejects incompatible combinatorics',()=>{
 const p=example('same-orbit-triangle'),owned=readGaugeOrbitInput(p);assert(owned);assert.notEqual(owned,p);assert(Object.isFrozen(owned.edges[0])&&Object.isFrozen(owned.forestEdges));assert.equal(gaugeOrbitInputKey(owned),gaugeOrbitInputKey(p));assert.notEqual(gaugeOrbitInputKey({...p,leftMatter:p.leftMatter.map(z=>({...z,real:'1/1'}))}),gaugeOrbitInputKey(p));
 for(const change of [{sites:2},{sites:13},{sites:'3'},{forestEdges:[0,1,2]},{forestEdges:[0,0]},{forestEdges:[0]},{forestEdges:[9,1]},{roots:[]},{roots:[0,1]},{roots:[3]},{anchors:[0,0]},{anchors:[3]},{leftLinks:p.leftLinks.slice(1)},{leftMatter:p.leftMatter.slice(1)},{leftLinks:[{cayley:'1',minusOne:true},...p.leftLinks.slice(1)]},{leftMatter:p.leftMatter.map(z=>({...z,real:'1e-3'}))},{extra:true}])assert.equal(readGaugeOrbitInput({...p,...change}),null);
 let calls=0;const getter={...p};Object.defineProperty(getter,'roots',{get(){calls++;throw Error('Getter');}});assert.equal(readGaugeOrbitInput(getter),null);assert.equal(calls,0);assert.throws(()=>calculateGaugeOrbits(getter),RangeError);
 const hole=[...p.leftLinks];delete hole[1];assert.equal(readGaugeOrbitInput({...p,leftLinks:hole}),null);
 const self=example('parallel-and-self-loops');assert.equal(readGaugeOrbitInput({...self,forestEdges:[2]}),null);
});
test('12 sites,32 edges and distinct18-digit chart fractions fit the existing wire budget',()=>{
 const n=12,edges=Array.from({length:32},(_,i)=>({from:i%12,to:(i+1)%12})),links=edges.map((_,i)=>({cayley:`${BigInt('999999999999999989')-BigInt(2*i)}/${BigInt('999999999999998987')-BigInt(6*i)}`,minusOne:false})),matter=Array.from({length:n},(_,i)=>({real:`${i+1}/999999999999997981`,imaginary:'1/999999999999996977'}));
 for(const anchors of [[],[0],[0,4,8],Array.from({length:n},(_,i)=>i)]){const r=calculateGaugeOrbits({sites:n,edges,leftLinks:links,rightLinks:links,leftMatter:matter,rightMatter:matter,forestEdges:Array.from({length:11},(_,i)=>i),roots:[0],anchors});assert(r.comparison.anchored.jointEquivalent);assert.deepEqual(snapshotQubitWire(r),r);}
});
test('the reader input and preparation bundle contains no exact arithmetic engine',async()=>{
 const bundle=await build({entryPoints:['src/lib/learning-gauge-orbits-examples.ts'],bundle:true,write:false,format:'esm',platform:'browser',metafile:true});assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/learning-quantum-hilbert-core.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));
});
