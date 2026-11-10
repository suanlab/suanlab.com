import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateBoundaryLayer,type BoundaryLayerInterval} from '../src/lib/learning-boundary-layer';
import {boundaryLayerDefault,boundaryLayerExamples,readBoundaryLayerInput,type BoundaryLayerInput} from '../src/lib/learning-boundary-layer-input';
import {Rational as Q} from '../src/lib/learning-graph-gauge-validation-math';
import {snapshotQubitWire} from '../src/lib/learning-qubit-wire';
import {getLearningContent} from '../src/lib/learning-content';
const B=BigInt,S=B(1)<<B(256),zero=Q.read('0'),le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
const tolerance=new Q(B(1),B(1)<<B(80));
interface ReferencePoint {x:string;exact:string;outer?:string;inner?:string;composite?:string;derivative?:string;curvature?:string;}
interface ReferenceCase {id:string;input:BoundaryLayerInput;probe:ReferencePoint;physicalSamples:ReferencePoint[];layerSamples:ReferencePoint[];mismatch?:string;delta?:string;uniformError?:string;derivativeUniformError?:string;}
const fixture=JSON.parse(readFileSync('scripts/fixtures/boundary-layer-reference.json','utf8')) as {cases:ReferenceCase[];referenceQuantumBits:number;symbolicExactResidual:string;symbolicCompositeInteriorResidual:string};
/** Tight 256-bit reference cells bracket numerical reference uncertainty;
 * they cross-check the producer, rather than supplying its analytic proof. */
function agrees(range:BoundaryLayerInterval,cell:string,label:string){const lo=Q.read(range.lower),hi=Q.read(range.upper),n=B(cell),refLo=new Q(n-B(1),S),refHi=new Q(n+B(2),S);assert(le(lo,hi),label+' ordered');assert(le(lo,refHi)&&le(refLo,hi),label+' independent agreement');assert(le(hi.sub(lo),tolerance),label+' useful width');}
for(const c of fixture.cases)test(`independent signed-coordinate solution,derivatives and all plotted samples: ${c.id}`,()=>{
 const r=calculateBoundaryLayer(c.input);assert.deepEqual(r.input,c.input);assert.equal(r.analyticResidual,'0');
 for(const[key,cases]of [['physicalSamples',c.physicalSamples],['layerSamples',c.layerSamples]] as const){assert.equal(r[key].length,cases.length);for(const[i,ref]of cases.entries()){const point=r[key][i];assert.equal(point.x,ref.x);agrees(point.exact,ref.exact,`${key}.${i}.exact`);if(point.outer!==null){agrees({lower:point.outer,upper:point.outer},ref.outer!,`${key}.${i}.outer`);agrees(point.inner!,ref.inner!,`${key}.${i}.inner`);agrees(point.composite!,ref.composite!,`${key}.${i}.composite`);}}}
 agrees(r.probe.exact,c.probe.exact,'probe');agrees(r.probe.derivative,c.probe.derivative!,'derivative');agrees(r.probe.curvature,c.probe.curvature!,'curvature');
 if(r.layer){assert.equal(r.layer.mismatch,c.mismatch);assert.equal(r.layer.delta,c.delta);agrees(r.layer.compositeUniformError,c.uniformError!,'uniform error');agrees(r.layer.compositeDerivativeUniformError,c.derivativeUniformError!,'derivative uniform error');}
 assert.deepEqual(r.physicalSamples[0].exact,{lower:Q.read(c.input.left).text(),upper:Q.read(c.input.left).text()});assert.deepEqual(r.physicalSamples[64].exact,{lower:Q.read(c.input.right).text(),upper:Q.read(c.input.right).text()});
});
test('example ledger and independent symbolic identities retain their declared scope',()=>{
 assert.equal(fixture.referenceQuantumBits,256);assert.equal(fixture.symbolicExactResidual,'0');assert.equal(fixture.symbolicCompositeInteriorResidual,'0');assert.equal(boundaryLayerExamples.length,17);assert.deepEqual(fixture.cases.slice(0,17).map(({id,input})=>({id,input})),boundaryLayerExamples);assert.deepEqual(JSON.parse(readFileSync('scripts/fixtures/boundary-layer-examples.json','utf8')),boundaryLayerExamples);
});
test('reflection moves the layer and reverses only the physical first derivative',()=>{
 const left=calculateBoundaryLayer({...boundaryLayerDefault,forcing:'1/3',left:'-2/3',right:'3/4',probe:'1/5'}),right=calculateBoundaryLayer({...boundaryLayerDefault,drift:'-1',forcing:'1/3',left:'3/4',right:'-2/3',probe:'4/5'});
 assert.equal(left.layer?.side,'left');assert.equal(right.layer?.side,'right');assert.deepEqual(left.probe.exact,right.probe.exact);assert.deepEqual(left.probe.curvature,right.probe.curvature);assert.equal(Q.read(left.probe.derivative.lower).neg().text(),right.probe.derivative.upper);assert.equal(Q.read(left.probe.derivative.upper).neg().text(),right.probe.derivative.lower);
});
test('compatible data suppress the layer; zero drift uses an exact polynomial',()=>{
 const compatible=calculateBoundaryLayer({...boundaryLayerDefault,forcing:'1'});assert.equal(compatible.layer?.hasLayerAmplitude,false);assert.equal(compatible.layer?.outerUniformError,'0');for(const p of compatible.physicalSamples){assert.deepEqual(p.exact,{lower:p.x,upper:p.x});assert.deepEqual(p.composite,p.exact);}
 const polynomial=calculateBoundaryLayer({...boundaryLayerDefault,drift:'0',forcing:'1',probe:'1/2'});assert.equal(polynomial.branch,'zero-drift');assert.equal(polynomial.layer,null);assert.equal(polynomial.probe.exact.lower,'-3/4');assert.equal(polynomial.probe.derivative.lower,'1');assert.equal(polynomial.probe.curvature.lower,'10');assert.equal(polynomial.probe.composite,null);assert.equal(polynomial.grid.cellOverLayer,null);
});
test('zero equation residual does not erase boundary defects or outer uniform error',()=>{
 const r=calculateBoundaryLayer(boundaryLayerDefault);assert(r.layer);assert.equal(r.analyticResidual,'0');assert.equal(r.layer.outerUniformError,'1');assert.equal(r.layer.compositeNearBoundaryResidual,'0');assert(Q.read(r.layer.compositeFarBoundaryResidual.upper).n<B(0));
 const t=Q.read(r.layer.inverseDelta),e=Q.read(r.input.epsilon),b=Q.read(r.input.drift),D=Q.read(r.layer.mismatch);assert.equal(e.mul(D).mul(t).mul(t).neg().add(b.mul(D).mul(t)).text(),'0');assert(le(zero,Q.read(r.layer.compositeUniformError.lower)));assert(le(Q.read(r.layer.compositeUniformError.upper),Q.read('1/20000')));
});
test('mesh criteria are distinct from thinness and do not change the analytic solution',()=>{
 const coarse=calculateBoundaryLayer({...boundaryLayerDefault,panels:4}),fine=calculateBoundaryLayer({...boundaryLayerDefault,panels:256});assert.deepEqual(coarse.physicalSamples,fine.physicalSamples);assert.deepEqual(coarse.layerSamples,fine.layerSamples);assert.equal(coarse.grid.fourPointsPerEfold,false);assert.equal(fine.grid.fourPointsPerEfold,true);assert.equal(coarse.grid.cellOverLayer,'5/2');assert(coarse.layer?.asymptoticallyThinAtCurrentInput);
 const weak=calculateBoundaryLayer({...boundaryLayerDefault,drift:'1/1000'});assert.equal(weak.layer?.delta,'100');assert.equal(weak.layer?.asymptoticallyThinAtCurrentInput,false);
 const thin=calculateBoundaryLayer({...boundaryLayerDefault,epsilon:'1/999999999989'});assert(thin.layer?.hasLayerAmplitude);assert.equal(thin.layer?.farDecay.lower,'0');assert(Q.read(thin.layer!.farDecay.upper).n>B(0));assert.equal(thin.layerSamples[64].stretchedCoordinate,'8');
});
test('bounded input owns raw spellings without executing getters or coercions',()=>{
 const v={...boundaryLayerDefault,epsilon:'2/20'},owned=readBoundaryLayerInput(v);assert(owned);v.epsilon='1';assert.equal(owned.epsilon,'2/20');assert(Object.isFrozen(owned));assert.equal(readBoundaryLayerInput({...boundaryLayerDefault,epsilon:'0'}),null);
 for(const field of ['epsilon','drift','forcing','left','right','probe']as const)for(const value of ['1e-3','NaN','1/0','1234567890123','-0',' 1','1/01'])assert.equal(readBoundaryLayerInput({...boundaryLayerDefault,[field]:value}),null);
 for(const input of [{...boundaryLayerDefault,panels:4.5},{...boundaryLayerDefault,panels:257},{...boundaryLayerDefault,probe:'2'},{...boundaryLayerDefault,drift:'17'},{...boundaryLayerDefault,extra:true},{...boundaryLayerDefault,[Symbol('s')]:1}])assert.equal(readBoundaryLayerInput(input),null);
 let reads=0;const accessor={...boundaryLayerDefault};Object.defineProperty(accessor,'epsilon',{get(){reads++;throw Error('getter');}});assert.equal(readBoundaryLayerInput(accessor),null);assert.equal(reads,0);assert.throws(()=>calculateBoundaryLayer({...boundaryLayerDefault,epsilon:'0'}),RangeError);
});
test('maximum outputs fit existing wire ownership and do not share mutable result objects',()=>{
 for(const c of fixture.cases.filter(c=>c.id==='maximum-coprime'||c.id.startsWith('extreme-small-drift'))){const r=calculateBoundaryLayer(c.input);assert.deepEqual(snapshotQubitWire(r),r);assert(JSON.stringify(r).length<2000000);const next=calculateBoundaryLayer(c.input);r.physicalSamples[1].exact.lower='forged';assert.notEqual(next.physicalSamples[1].exact.lower,'forged');}
});
test('reader-safe inputs exclude the rational and exponential numerical engines',async()=>{
 const result=await build({entryPoints:['src/lib/learning-boundary-layer-input.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert.equal(Object.keys(result.metafile!.inputs).length,1);assert(!result.outputFiles[0].text.includes('BigInt('));
});
test('bilingual manuscripts render eighteen substantive sections and eight explained exercises',async()=>{
 const id='singular-perturbation-boundary-layer-matching',rendered=await getLearningContent(id),root='.runtime/learning/boundary-layer-foundation';mkdirSync(root,{recursive:true});const rows=[];
 for(const language of ['ko','en']as const){const source=readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`,'utf8');assert.equal(rendered.outline[language].length,18);assert.equal((source.match(/^\d\. \*\*/gm)??[]).length,8);assert(source.length>8000);assert(!rendered[language].includes('katex-error'));assert(!/<h1(?:\s|>)/.test(rendered[language]));assert(rendered[language].includes('<math'));if(language==='en')assert(!/[가-힣]/.test(rendered.en));writeFileSync(`${root}/manuscript-${language}.html`,rendered[language]);rows.push({language,sections:18,displayEquations:(source.match(/^\$\$$/gm)??[]).length/2,explainedExercises:8,sourceCharacters:source.length});}
 writeFileSync(root+'/manuscripts.json',JSON.stringify(rows,null,2)+'\n');
});
