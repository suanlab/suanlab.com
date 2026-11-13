import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test,{after}from'node:test';
import assert from'node:assert/strict';
import {readFileSync,writeFileSync}from'node:fs';
import {build}from'esbuild';
import {spinChainTeachingPresets as presets,spinChainTeachingInput as preset,editSpinTeachingField as edit,spinTeachingFields as fields}from'../src/lib/learning-many-body-spin-teaching';
import {manyBodySpinUnit as unit,manyBodySpinGuide as guide}from'../src/data/many-body-spin-teaching';
import {spinChainModel as model}from'../src/lib/learning-spin-chain';
import {snapshotManyBodyInput,manyBodyInputKey,parseManyBodyInputDraft,parseManyBodyStoredInput}from'../src/lib/learning-many-body-input';
import {getLearningContent}from'../src/lib/learning-content';
import {spinGraphCameraDistance}from'../src/components/learning/many-body-spin-graph-engine';
const originalManuscript=(s:string)=>{const start=s.search(/^## (Research extension:|연구 확장:)/m);if(start<0)return s;const end=s.indexOf('\n## ',start+4)+1;assert(end>start);return s.slice(0,start)+s.slice(end);};
ensureTestArtifactDirectory('.runtime/learning/many-body-spin-teaching', { recursive: true });

const runtime='.runtime/learning/many-body-spin-teaching';
const results=presets.map(p=>model(p.input.parameters));
let presetChecks=0;
test('six reader-safe presets match hand-computed energies,variances,correlations and Y phase',()=>{
 const expected=[[-3,1,0],[-3,1,1],[-3,0,1],[3,0,0],[-1,0,0],[-1,0,null]];
 results.forEach((r,i)=>{assert.equal(r.energy.value,expected[i][0]);assert.equal(r.variance.value,expected[i][1]);if(expected[i][2]!==null)assert.equal(r.correlations[0].values[2][2].connected.value,expected[i][2]);assert(Object.isFrozen(presets[i].input.parameters));assert.equal(manyBodyInputKey(preset(i)),manyBodyInputKey(presets[i].input));presetChecks++;});
 assert.equal(results[0].localStates[0].bloch[2].value,1);assert.equal(results[1].localStates[0].bloch[2].value,0);assert.equal(results[5].localStates[0].bloch[1].value,1);
 for(const i of[0,1,5])assert.equal(results[i].diagonalSpectrum,null);
 assert.equal(results[3].classification.selectedStateGround,false);assert.equal(results[4].classification.selectedStateGround,true);assert.deepEqual(results[3].diagonalSpectrum!.levels.map(v=>[v.energy.value,v.multiplicity]),[[-1,6],[3,2]]);
});

test('diagonal GHZ and transverse GHZ share energy but have different residuals and certificates',()=>{assert.equal(results[1].energy.value,results[2].energy.value);assert.equal(results[1].classification.exactEigenstate,false);assert.equal(results[2].classification.exactEigenstate,true);assert.equal(results[2].diagonalSpectrum!.groundMultiplicity,2);assert.equal(results[2].diagonalSpectrum!.groundProbability.value,1);});

test('integer field edits are atomic and reject unsafe lexical forms and invalid preparations',()=>{
 const p=preset();for(const raw of['','-0','1.0','1e0','+1',' 1','01','1.5','1000001','Infinity'])assert.equal(edit(p,'fieldX0',raw),null);
 assert.equal(edit(p,'unknown','1'),null);assert.equal(edit(p,'hDenominator','0'),null);assert.equal(edit(preset(5),'stateReal0','0')!==null,true); // i|1> is a valid nonzero state
 const q=edit(p,'fieldY0','-2');assert(q);assert.equal(q.parameters.fieldY0,-2);assert.equal(p.parameters.fieldY0,0);assert(Object.isFrozen(q.parameters));assert.throws(()=>preset(-1));assert.throws(()=>preset(6));
 const y=edit(preset(5),'stateImag1','0');assert(y);assert.equal(edit(y,'stateReal0','0'),null);
});

test('complete JSON and versioned storage retain all102 fields and reject a wrong kind',()=>{
 for(const p of presets){assert.equal(Object.keys(p.input.parameters).length,102);const stored=parseManyBodyStoredInput(JSON.stringify({version:1,value:p.input}));assert(stored);assert.equal(manyBodyInputKey(stored),manyBodyInputKey(p.input));assert(parseManyBodyInputDraft(JSON.stringify(p.input)));}
 assert.equal(parseManyBodyStoredInput(JSON.stringify({version:2,value:preset()})),null);assert.equal(parseManyBodyInputDraft(JSON.stringify({...preset(),kind:'unknown'})),null);
 const raw=JSON.parse(JSON.stringify(preset()));raw.parameters.stateReal0=0;assert.equal(parseManyBodyInputDraft(JSON.stringify(raw)),null);
});

test('active field labels follow real graph edges,all axes,and basis-bit order',()=>{
 assert.equal(fields(preset(5)).length,9);assert(!fields(preset(5)).some(f=>f.key==='stateReal2'));
 const p=snapshotManyBodyInput({kind:'spin-chain',parameters:{...preset(0).parameters,graph:2}});const f=fields(p);assert(f.some(f=>f.key==='couplingYZ03'));assert(f.find(f=>f.key==='stateImag5')!.label.en.includes('|0101〉'));assert(!f.some(f=>f.key==='sites'||f.key==='graph'));
 for(const p of presets)for(const field of fields(p.input))for(const lang of['ko','en'] as const)assert(field.label[lang].length>4);
});

test('three guided experiments and three explained questions use independent fixed examples',()=>{
 assert.equal(guide.steps.length,3);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,3);assert.equal(new Set(unit.questions.map(q=>q.id)).size,3);assert.equal(unit.questions[0].options[unit.questions[0].answer].en,String(results[0].dimension));assert.equal(unit.questions[1].options[unit.questions[1].answer].en,String(results[1].correlations[0].values[2][2].connected.value));assert(unit.questions[2].options[unit.questions[2].answer].en.includes(String(results[3].diagonalSpectrum!.groundEnergy.value).replace('-','−')));
 for(const lang of['ko','en'] as const){for(const g of guide.experiments){assert(g.predict[lang].length>0&&g.action[lang].length>0&&g.explain[lang].length>0);assert(presets[g.preset]);}for(const q of unit.questions){assert(q.options.length===3&&q.answer>=0&&q.answer<3&&q.explanation[lang].length>0);}}
 assert(!/[가-힣]/.test(JSON.stringify({title:unit.title.en,subtitle:unit.subtitle.en,steps:guide.steps.map(s=>[s.title.en,s.explanation.en,s.diagram.en]),quiz:unit.questions.map(q=>[q.question.en,q.options.map(o=>o.en),q.explanation.en])})));
});

test('the staged bilingual loader uses the full audited manuscript bytes with strict KaTeX',async()=>{
 for(const locale of['ko','en'])assert.equal(originalManuscript(readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${locale}.md`,'utf8')),readFileSync(`docs/online-learning/physics-math-for-ai/${unit.id}-chapter.${locale}.md`,'utf8'));
 const content=await getLearningContent(unit.id);for(const lang of['ko','en'] as const){assert.equal(content.outline[lang].length,15);assert(content[lang].includes('katex-display'));assert(!content[lang].includes('katex-error'));assert.equal((content[lang].match(/<table/g)||[]).length,4);}
});

test('camera fit contains every site sphere in portrait and wide viewports',()=>{for(const radius of[1.6,3,5.3])for(const aspect of[.45,1,2.5]){const d=spinGraphCameraDistance(radius,aspect),half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));assert(d*Math.sin(half)>=radius*1.1);}});

test('lesson main bundle excludes all calculation and mathematical validation engines',async()=>{
 const code=await build({entryPoints:['src/components/learning/many-body-spin-lesson.tsx'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
 assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|tensor-roots|quantum-hilbert-core|observable-matrix|ritz-interval|interval-transcendentals|many-body-(?:evaluator|validation-evaluator|.*-validation))\.ts$/.test(p)),inputs.join('\n'));
 writeFileSync(`${runtime}/main-thread-dependency-graph.json`,JSON.stringify({inputs,heavyNumericalEnginesExcluded:true,mathValidatorsExcluded:true,threeJsGeometryIncluded:inputs.some(p=>p.includes('many-body-spin-graph-engine'))},null,2)+'\n');
});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({presetChecks,conceptSteps:guide.steps.length,guidedExperiments:guide.experiments.length,explainedQuestions:unit.questions.length,bilingualManuscripts:2},null,2)+'\n'));
