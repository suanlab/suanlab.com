import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {getLearningContent} from '../src/lib/learning-content';
import {mkdirSync,writeFileSync} from 'node:fs';
import {readGraphGaussInput,graphGaussInputKey,type GraphGaussInput} from '../src/lib/learning-graph-gauss-input';
import {calculateGraphGauss} from '../src/lib/learning-graph-gauss';
import {graphGaussExamples} from '../src/lib/learning-graph-gauss-examples';
const reference=JSON.parse(readFileSync('scripts/fixtures/graph-gauss-independent.json','utf8'))as{cases:{label:string;input:GraphGaussInput;expected:Record<string,unknown>}[]};
const allZero=(values:unknown)=>{if(Array.isArray(values))values.forEach(allZero);else assert(values==='0'||values===0||values===null,JSON.stringify(values));};
for(const fixture of reference.cases)test(`independent canonical graph: ${fixture.label}`,()=>{
 const r=calculateGraphGauss(fixture.input),x=fixture.expected,k=r.coordinates.length;
 for(const name of ['incidence','interiorSites','interiorIncidence','chargesCompatible','inputAdmissible','divergence','gaussResiduals']as const)assert.deepEqual(r[name],x[name],name);
 assert.deepEqual(r.fixing.forestEdges,x.forestEdges);assert.deepEqual(r.canonical.nonForestEdges,x.nonForestEdges);assert.equal(r.counts.independentConstraintCount,x.rank);assert.equal(r.counts.effectiveGaugeDimension,x.rank);assert.equal(r.counts.rankFormula,x.rank);assert.equal(r.counts.configurationQuotientDimension,k);assert.equal(k,r.edgeCount-r.counts.independentConstraintCount);assert.equal(r.counts.physicalPhaseSpaceDimension,x.physicalPhaseSpaceDimension);assert.equal(r.counts.constraintSurfaceDimension,r.chargesCompatible?2*r.edgeCount-r.counts.independentConstraintCount:null);
 for(const name of ['pathMatrix','electricKernel','coordinateMomentumBrackets','pathConstraintBrackets','gaussBrackets','affineElectric','electricMetric','electricLinear','electricConstant','liftHamiltonian']as const)assert.deepEqual(r.canonical[name],x[name],name);
 assert.deepEqual(r.canonical.liftFromCurrentNonForestMomenta,x.liftElectric);assert.deepEqual(r.coordinates.map(c=>c.product),x.coordinateProducts);assert.deepEqual(r.coordinates.map(c=>c.transformedProduct),x.transformedCoordinateProducts);assert.deepEqual(r.terms.map(t=>({product:t.product,transformedProduct:t.transformedProduct,potential:t.potential,force:t.force})),x.terms);assert.deepEqual({electric:r.energy.electric,potential:r.energy.potential,total:r.energy.total},x.energy);
 for(const name of ['angleRates','electricRates','gaussRates','coordinateRates']as const)assert.deepEqual(r.flow[name],x[name],name);assert.deepEqual(r.flow.nonForestMomentumRates,x.momentumRates);
 allZero(r.flow.forceKernelResiduals);allZero(r.flow.gaussRates);allZero(r.canonical.pathConstraintBrackets);allZero(r.canonical.gaussBrackets);allZero(r.energy.gaugeResidual);allZero(r.canonical.affineGaussResiduals);allZero(r.canonical.liftGaussResiduals);
 for(const c of r.coordinates)allZero(Object.values(c.invarianceResidual));for(const t of r.terms){allZero(Object.values(t.invarianceResidual));allZero(Object.values(t.reductionResidual));allZero(t.incidenceResidual);}
 if(r.inputAdmissible){allZero(r.canonical.liftMinusInput);allZero(r.canonical.currentHamiltonianResidual);allZero(r.flow.coordinateRateResiduals);}else{assert.equal(r.canonical.currentStateReduced,false);assert.equal(r.flow.reducedCoordinateRates,null);assert.equal(r.canonical.currentHamiltonianResidual,null);}
 assert.doesNotThrow(()=>JSON.stringify(r));
});
const preset=(id:string)=>{const x=graphGaussExamples.find(x=>x.id===id);assert(x);return x.input;};
test('all seventeen prepared examples are owned,deeply frozen and match the independent preparation list',()=>{
 const prepared=JSON.parse(readFileSync('scripts/fixtures/graph-gauss-preparations.json','utf8'));assert.equal(graphGaussExamples.length,17);for(const [i,x]of graphGaussExamples.entries()){assert.deepEqual(x.input,prepared[i].input);assert(Object.isFrozen(x.input)&&Object.isFrozen(x.input.edges)&&Object.isFrozen(x.input.magneticTerms));assert.deepEqual(readGraphGaussInput(x.input),x.input);}
});
test('default canonical pair and exact instantaneous Hamiltonian flow have hand-checkable values',()=>{
 const r=calculateGraphGauss(preset('physical-triangle'));assert.equal(r.energy.total,'5/2');assert.equal(r.counts.physicalPhaseSpaceDimension,2);assert.deepEqual(r.canonical.pathMatrix,[[1,1,1]]);assert.deepEqual(r.canonical.electricMetric,[['3']]);assert.deepEqual(r.flow.coordinateRates,['3']);assert.deepEqual(r.flow.electricRates,['-1','-1','-1']);assert.deepEqual(r.flow.nonForestMomentumRates,['-1']);
});
test('inconsistent levels,violated inputs and fixed-charge lifts are distinguished',()=>{
 const empty=calculateGraphGauss(preset('incompatible-total-charge'));assert.equal(empty.chargesCompatible,false);assert.equal(empty.counts.physicalPhaseSpaceDimension,null);assert.equal(empty.canonical.affineElectric,null);
 const violated=calculateGraphGauss(preset('constraint-violating-triangle'));assert.equal(violated.chargesCompatible,true);assert.equal(violated.inputAdmissible,false);assert.deepEqual(violated.canonical.liftFromCurrentNonForestMomenta,['0','0','0']);assert.deepEqual(violated.canonical.liftMinusInput,['-1','0','0']);assert.equal(violated.canonical.currentHamiltonianResidual,null);
 const disconnected=calculateGraphGauss(preset('global-neutrality-insufficient'));assert.equal(disconnected.chargesCompatible,false);assert(disconnected.components.every(c=>!c.compatible));
});
test('boundary sources and empty/full anchor groups have the declared canonical dimensions',()=>{
 const r=calculateGraphGauss(preset('boundary-charge-flux'));assert.equal(r.inputAdmissible,true);assert.equal(r.energy.total,'9/2');assert.deepEqual(r.flow.coordinateRates,['3']);assert.deepEqual(r.flow.nonForestMomentumRates,['-2']);assert.equal(r.counts.physicalPhaseSpaceDimension,2);assert.deepEqual(r.gaussResiduals,[null,'0',null]);assert.equal(r.components[0].prescribedInteriorCharge,'1');assert.equal(r.components[0].anchorDivergence,'-1');assert.deepEqual(r.canonical.affineElectric,['-1','0']);assert.deepEqual(r.canonical.liftFromCurrentNonForestMomenta,['1','2']);assert.deepEqual(r.flow.gaussRates,['0']);assert(r.flow.anchorChargeRates.some(x=>x.value!=='0'));
 assert.equal(calculateGraphGauss(preset('every-site-anchored')).counts.physicalPhaseSpaceDimension,6);assert.equal(calculateGraphGauss(preset('charged-open-chain')).counts.physicalPhaseSpaceDimension,0);assert.equal(calculateGraphGauss(preset('isolated-sites')).counts.physicalPhaseSpaceDimension,0);
});
test('changing the tree retains the explicitly fixed Hamiltonian and physical field',()=>{
 const a=calculateGraphGauss(preset('charged-triangle')),b=calculateGraphGauss(preset('alternative-tree-root'));assert.deepEqual(a.energy,b.energy);assert.deepEqual(a.flow.electricRates,b.flow.electricRates);assert.deepEqual(a.divergence,b.divergence);assert.notDeepEqual(a.canonical.affineElectric,b.canonical.affineElectric);assert.deepEqual(a.canonical.liftFromCurrentNonForestMomenta,b.canonical.liftFromCurrentNonForestMomenta);
});
test('reader rejects accessors,extra keys,bad ranks,unfixed anchors,wrong charge placeholders and noninvariant potentials',()=>{
 const p=preset('physical-triangle'),anchored=preset('anchored-open-chain');let called=false;const getter={...p};Object.defineProperty(getter,'electric',{enumerable:true,get(){called=true;return p.electric;}});assert.equal(readGraphGaussInput(getter),null);assert.equal(called,false);
 const sparse=Array(3);sparse[0]='1';for(const x of [{...p,extra:0},{...p,electric:sparse},{...p,forestEdges:[0]},{...p,electricWeights:['0','1','1']},{...p,electric:['1e0','1','1']},{...p,magneticTerms:[{start:0,steps:[{edge:0,forward:true}],weight:'1'}]},{...anchored,charges:['0','0',null]},{...anchored,frames:[{cayley:'1',minusOne:false},...anchored.frames.slice(1)]},{...p,magneticTerms:[{start:0,steps:[{edge:1,forward:true}],weight:'1'}]}])assert.equal(readGraphGaussInput(x),null);
 const raw={...p,electric:['2/2','1','1']};assert.notEqual(graphGaussInputKey(readGraphGaussInput(raw)!),graphGaussInputKey(p));
});
test('reader and preparation bundle excludes canonical/orbit calculation engines and exact arithmetic',async()=>{
 const bundle=await build({entryPoints:['src/lib/learning-graph-gauss-examples.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'});for(const name of ['learning-graph-gauss.ts','learning-gauge-orbits.ts','learning-quantum-hilbert-core.ts'])assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/'+name));assert(!bundle.outputFiles[0].text.includes('BigInt('));
});

test('both substantive unregistered manuscripts render strict equations,18 sections and7 explained problems',async()=>{
 const id='graph-gauss-constraints-physical-phase-space',html=await getLearningContent(id as Parameters<typeof getLearningContent>[0]),root='.runtime/learning/graph-gauss-foundation';mkdirSync(root,{recursive:true});
 for(const language of ['ko','en']as const){const source=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal(html.outline[language].length,18);assert.equal((source.match(/^### [A-G]\./gm)??[]).length,7);assert(source.length>9000);assert(!html[language].includes('katex-error'));assert(!/<h1(?:\s|>)/.test(html[language]));assert(html[language].includes('<math'));writeFileSync(`${root}/manuscript-${language}.html`,html[language]);}assert(!/[가-힣]/.test(html.en));
});

test('maximum12-site32-edge18-digit preparations and8 paths of32 steps fit owned wire copying',()=>{
 const rows=JSON.parse(readFileSync('scripts/fixtures/graph-gauss-budget-preparations.json','utf8'))as {label:string;input:GraphGaussInput}[];assert.equal(rows.length,5);assert.equal(rows[4].input.magneticTerms.length,8);assert(rows[4].input.magneticTerms.every(t=>t.steps.length===32));
 for(const x of rows){const result=calculateGraphGauss(x.input);assert.deepEqual(snapshotQubitWire(result),result);assert.equal(result.inputAdmissible,true);}
});
