import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test,{after} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
import {mpsNetworkCameraDistance} from '../src/components/learning/many-body-mps-network-engine';
import {mpsTeachingPresets as presets,mpsTeachingInput as preset,editMpsTeachingField as edit,mpsTeachingFields as fields,mpsPrefixRows} from '../src/lib/learning-many-body-mps-teaching';
import {mpsModel as model,createMpsParameters} from '../src/lib/learning-mps';
import {manyBodyMpsUnit as unit,manyBodyMpsGuide as guide} from '../src/data/many-body-mps-teaching';
import {manyBodyInputKey,parseManyBodyStoredInput,snapshotManyBodyInput} from '../src/lib/learning-many-body-input';
import {getLearningContent} from '../src/lib/learning-content';
const originalManuscript=(s:string)=>{const start=s.search(/^## (Research extension:|연구 확장:)/m);if(start<0)return s;const end=s.indexOf('\n## ',start+4)+1;assert(end>start);return s.slice(0,start)+s.slice(end);};
ensureTestArtifactDirectory('.runtime/learning/many-body-mps-teaching', { recursive: true });

const runtime='.runtime/learning/many-body-mps-teaching',results=presets.map(p=>model(p.input.parameters));let prefixCoordinates=0;
test('eight teaching preparations preserve hand-calculated norms,ranks and complex phase',()=>{
 const norms=[2,2,0,0,2,1,2,2],ranks=[[2,2],[2,2],[0],[0],[1],[1,1],[],[2,2,2]];
 results.forEach((r,i)=>{assert.equal(r.original.normSquared.value,norms[i]);assert.deepEqual(r.original.cuts.map(c=>c.coefficientRank),ranks[i]);assert.equal(r.validity.physicalStateDefined,norms[i]!==0);assert.deepEqual(r.original.state,r.gauged.state);assert.deepEqual(r.original.physical,r.gauged.physical);assert(Object.isFrozen(presets[i].input.parameters));});
 assert.equal(results[4].original.physical!.localStates[1].bloch[1].value,1);assert.equal(results[6].original.physical!.localStates[0].bloch[1].value,1);assert.deepEqual(results[7].bondDimensions,[2,3,2]);
});
test('GHZ real gauge changes coordinate norm6→19/2 while full vector and every norm closure stay unchanged',()=>{
 const r=results[0];assert.equal(r.original.tensorParameterSquaredNorm.value,6);assert.equal(r.gauged.tensorParameterSquaredNorm.value,9.5);for(const result of results)for(const v of Object.values(result.exactResiduals))if(v)assert.equal(v.sign,0);for(const representation of[r.original,r.gauged]){assert(representation.cutClosureNorms.every(q=>q.value===2));assert(representation.physical!.correlations.every(pair=>pair.values[2][2].connected.value===1));}assert.equal(r.original.canonicality[0].leftIsometry,true);assert.equal(r.gauged.canonicality[0].leftIsometry,false);
});
test('complex closure uses matching E_ab R_ab and cancellation has no normalized physical state',()=>{
 const r=results[2];assert(r.original.leftEnvironments[1][0][1].imaginary.value===1);assert.deepEqual(r.original.leftEnvironments[1],r.original.rightEnvironments[1]);assert.equal(r.original.normSquared.value,0);assert.equal(r.original.physical,null);assert(r.original.cuts.every(c=>c.schmidtRank===null));
 const E=r.original.leftEnvironments[1],R=r.original.rightEnvironments[1];let right=0,wrong=0;for(let a=0;a<2;a++)for(let b=0;b<2;b++){const e=E[a][b];right+=e.real.value!*R[a][b].real.value!-e.imaginary.value!*R[a][b].imaginary.value!;wrong+=e.real.value!*R[b][a].real.value!-e.imaginary.value!*R[b][a].imaginary.value!;}assert.equal(right,0);assert.equal(wrong,4);
});
test('redundant bond4 is rank1 with48 active entries versus8 dense amplitudes;one site has no cut',()=>{assert.equal(results[5].cost.activeComplexTensorEntries,48);assert.equal(results[5].cost.denseComplexStateAmplitudes,8);assert(results[5].original.cuts.every(c=>c.bondDimension===4&&c.schmidtRank===1));assert.equal(results[6].gauge,null);assert.equal(results[6].original.cuts.length,0);});
test('every original/gauged prefix row agrees with an independent explicit virtual-path sum',()=>{
 type C=[number,number];const mul=(a:C,b:C):C=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]],add=(a:C,b:C):C=>[a[0]+b[0],a[1]+b[1]];
 for(const r of results)for(const gauged of[false,true])for(let step=0;step<=r.sites;step++){
  const rows=mpsPrefixRows(r,step,gauged),tensors=(gauged?r.gauged:r.original).tensors;assert.equal(rows.length,2**step);
  for(const[index,entry]of rows.entries()){
   const bits=step===0?'':index.toString(2).padStart(step,'0');assert.equal(entry.physicalPrefix,bits||'∅');
   for(let target=0;target<entry.row.length;target++){
    let expected:C=[0,0];function paths(site:number,left:number,weight:C){if(site===step){if(left===target)expected=add(expected,weight);return;}const matrix=tensors[site][Number(bits[site])];for(let right=0;right<matrix[left].length;right++){const z=matrix[left][right];paths(site+1,right,mul(weight,[z.real.value!,z.imaginary.value!]));}}paths(0,0,[1,0]);
    assert(Math.abs(entry.row[target].real.value!-expected[0])<1e-10);assert(Math.abs(entry.row[target].imaginary.value!-expected[1])<1e-10);prefixCoordinates++;
   }
  }
 }assert(prefixCoordinates>200);assert.throws(()=>mpsPrefixRows(results[0],4));assert.throws(()=>mpsPrefixRows(results[0],-.5));
});
test('integer editors preserve structural constraints while allowing zero full contraction',()=>{
 const p=preset();for(const raw of['','-0','1.0','1e0','+1','01',' 1','1000001'])assert.equal(edit(p,'t0Real000',raw),null);for(const[key,value]of[['tensorDenominator','0'],['gaugeScaleNumerator','0'],['gaugeBond','2'],['t3Real000','1'],['unknown','1']])assert.equal(edit(p,key,value),null);
 const single=snapshotManyBodyInput({kind:'mps',parameters:createMpsParameters(1,1)}),zero=edit(single,'t0Real000','0');assert(zero);assert.equal(model(zero.parameters).validity.physicalStateDefined,false);assert.throws(()=>preset(8));
});
test('all269 fields survive versioned storage and active tensor labels respect nonuniform bonds',()=>{
 for(const p of presets){assert.equal(Object.keys(p.input.parameters).length,269);const saved=parseManyBodyStoredInput(JSON.stringify({version:1,value:p.input}));assert(saved);assert.equal(manyBodyInputKey(saved),manyBodyInputKey(p.input));}
 assert.equal(parseManyBodyStoredInput(JSON.stringify({version:2,value:preset()})),null);assert(!fields(preset(6)).some(f=>f.key.startsWith('gauge')));assert(fields(preset(7)).find(f=>f.key==='t2Imag210')!.label.en.includes('A2[s=1][2,0]'));assert(!fields(preset(7)).some(f=>f.key==='t2Real300'));
});
test('three concepts,guides and explained quizzes preserve complete audited bilingual chapters',async()=>{
 assert.equal(guide.steps.length,3);assert.equal(guide.experiments.length,3);assert.deepEqual(unit.questions.map(q=>q.answer),[0,1,2]);const content=await getLearningContent(unit.id);for(const lang of['ko','en'] as const){assert.equal(originalManuscript(readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8')),readFileSync(`docs/online-learning/physics-math-for-ai/${unit.id}-chapter.${lang}.md`,'utf8'));assert.equal(content.outline[lang].length,17);assert.equal((content[lang].match(/<table/g)||[]).length,5);assert(!content[lang].includes('katex-error'));for(const q of unit.questions)assert(q.explanation[lang]);for(const g of guide.experiments)assert(presets[g.preset]&&g.predict[lang]&&g.action[lang]&&g.explain[lang]);}
 assert(!/[가-힣]/.test(JSON.stringify({title:unit.title.en,steps:guide.steps.map(s=>s.explanation.en),questions:unit.questions.map(q=>q.question.en)})));
});
test('main-thread lesson bundle excludes every calculator and mathematical validator',async()=>{const b=await build({entryPoints:['src/components/learning/many-body-mps-lesson.tsx'],jsx:'automatic',bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(b.metafile!.inputs);assert(inputs.some(p=>p.includes('many-body-mps-network-engine')));assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|boson-algebraic|tensor-roots|quantum-hilbert-core|observable-matrix|ritz-interval|interval-transcendentals|many-body-(?:evaluator|validation-evaluator|.*-validation))\.ts$/.test(p)),inputs.join('\n'));writeFileSync(`${runtime}/main-thread-dependency-graph.json`,JSON.stringify({inputs,heavyNumericalEnginesExcluded:true,mathValidatorsExcluded:true,lazyThreeGeometryIncluded:true},null,2)+'\n');});
test('network camera encloses tensor boxes,physical legs and gauge labels',()=>{for(let sites=1;sites<=4;sites++)for(const aspect of[.3,.7,1,2,4]){const halfHeight=mpsNetworkCameraDistance(sites,aspect)*Math.tan(42*Math.PI/360);assert(halfHeight>=1.8);assert(halfHeight*aspect>=(sites-1)*1.05+1.05);}});
test('MPS result renders on server without WebGL and omits normalized physical outputs for a zero contraction',async()=>{const bundle=await build({stdin:{contents:"import {renderToString} from 'react-dom/server';import {createElement} from 'react';import View from './src/components/learning/many-body-mps-view';export const render=(result)=>renderToString(createElement(View,{result,language:'en'}));",resolveDir:process.cwd(),loader:'tsx'},jsx:'automatic',bundle:true,write:false,platform:'node',format:'cjs',packages:'external'}),mod={exports:{}as{render:(r:unknown)=>string}};new Function('require','module','exports',bundle.outputFiles[0].text)(createRequire(`${process.cwd()}/package.json`),mod,mod.exports);const nonzero=mod.exports.render(results[0]),zero=mod.exports.render(results[2]);assert(nonzero.includes('data-mps-physical'));assert(!zero.includes('data-mps-physical'));assert(zero.includes('No vacuum or uniform probability'));assert(!nonzero.includes('<canvas'));assert(!zero.includes('<canvas'));});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({prefixCoordinates,fixedPreparations:8,conceptSteps:3,guidedExperiments:3,explainedQuestions:3,bilingualManuscripts:2},null,2)+'\n'));
