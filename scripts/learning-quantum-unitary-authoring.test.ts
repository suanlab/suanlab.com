import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {unified} from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import type {Root,Element} from 'hast';
import {visit} from 'unist-util-visit';
import {spectralEvolutionModel as spectral,type SpectralEvolutionParameters} from '../src/lib/learning-spectral-evolution';
import {pulseEvolutionModel as pulse,type PulseEvolutionParameters} from '../src/lib/learning-pulse-evolution';
const ids=['two-level-spectral-schrodinger-evolution','piecewise-hamiltonian-pulse-time-ordering'];
for(const[index,id]of ids.entries()){
 const sources=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/${id}-chapter.${lang}.md`,'utf8')]));
 for(const lang of ['ko','en'])test(`unitary manuscript ${id}.${lang}: complete explanatory text, valid mathematics and rectangular tables`,async()=>{
  const s=sources[lang];assert.equal((s.match(/^## /gm)||[]).length,12);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\*\*(?:Explanation|해설)\./gm)||[]).length,6);
  assert(s.includes('https://www.preskill.caltech.edu/ph219/chap2_15.pdf'));assert(s.includes('https://dlmf.nist.gov/4.19'));assert(s.includes('Interactive Book'));assert(s.includes('null'));assert(s.includes('768'));assert(s.includes('512'));
  if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.split(/\s+/).length>1500);}else assert(s.length>6000);
  let tables=0,equations=0;
  const html=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(()=> (tree:Root)=>{visit(tree,'element',node=>{
   if(node.tagName==='table'){tables++;const rows:Element[]=[];visit(node,'element',n=>{if(n.tagName==='tr')rows.push(n);});const columns=(r:Element)=>r.children.filter(n=>n.type==='element'&&(n.tagName==='th'||n.tagName==='td')).length;assert(rows.length>=4);assert(rows.every(r=>columns(r)===(tables===2?4:3)),'table columns');}
   if(Array.isArray(node.properties.className)&&node.properties.className.includes('katex-display'))equations++;
  });}).use(rehypeStringify).process(s);
  assert.equal(html.messages.length,0);assert(!String(html).includes('katex-error'));assert.equal(tables,index+1);assert.equal(equations,7+index);
 });
 test(`unitary manuscript ${id}: translated equations and exercise ordering preserve conventions`,()=>{assert.deepEqual(sources.ko.match(/\$\$[\s\S]*?\$\$/g),sources.en.match(/\$\$[\s\S]*?\$\$/g));assert.deepEqual(sources.ko.match(/^\*\*\d\./gm),sources.en.match(/^\*\*\d\./gm));for(const s of Object.values(sources)){if(index===0){assert(s.includes('25/16'));assert(s.includes('(17/16)'));assert(s.includes('cos²t'));}else{assert(s.includes('0.913411'));assert(s.includes('1−sin²2 sin²1'));assert(s.includes('0.414549'));assert(!s.includes('1−sin⁴2'));assert(s.includes('2/15'));}}});
}
const spectralRefs=JSON.parse(readFileSync('scripts/fixtures/spectral-evolution-reference.json','utf8'))as{cases:{label:string;input:SpectralEvolutionParameters}[]};
const pulseRefs=JSON.parse(readFileSync('scripts/fixtures/pulse-evolution-reference.json','utf8'))as{cases:{label:string;input:PulseEvolutionParameters}[]};
const close=(actual:number|null,expected:number,tolerance=1e-14)=>{assert.notEqual(actual,null);assert(Math.abs(actual!-expected)<=tolerance,`${actual} versus ${expected}`);};
const initial=spectralRefs.cases[0].input;
const zInput:SpectralEvolutionParameters={...initial,stateReal0:1,stateImag0:0,stateReal1:1,stateImag1:0,axisReal0:1,axisImag0:0,axisReal1:0,axisImag1:0,centerNumerator:0,centerDenominator:1,gapNumerator:1,gapDenominator:1,timeNumerator:1,timeDenominator:1,hbarNumerator:1,hbarDenominator:1,sampleSteps:1,eulerSteps:2};
const P=(label:string)=>pulse(pulseRefs.cases.find(c=>c.label===label)!.input);
test('spectral lecture example: fixed energy distribution can conceal relative phase motion',()=>{const x=spectral(zInput),r=x.records.at(-1)!;assert.equal(x.initialEnergyMean.sign,0);close(x.initialEnergyVariance.value,1);close(r.standardProbabilities[0].value,.5);close(r.standardProbabilities[1].value,.5);close(r.fidelityToInitial.value,Math.cos(1)**2);close(r.amplitudes[0].real.value,Math.cos(1)/Math.sqrt(2));close(r.amplitudes[0].imaginary.value,-Math.sin(1)/Math.sqrt(2));close(r.amplitudes[1].imaginary.value,Math.sin(1)/Math.sqrt(2));assert(!x.classification.stationaryRayProved);});
test('Euler lecture example: exact raw norms and amplitudes on fixed-duration grids',()=>{for(const[n,norm]of [[1,2],[2,25/16],[4,(17/16)**4]]){const x=spectral({...zInput,eulerSteps:n}),r=x.records.at(-1)!;close(r.eulerNormSquared.value,norm);assert.equal(r.eulerNormIdentityResidual.sign,0);if(n===2){close(r.eulerAmplitudes[0].real.value,.75/Math.sqrt(2));close(r.eulerAmplitudes[0].imaginary.value,-1/Math.sqrt(2));close(r.eulerAmplitudes[1].imaginary.value,1/Math.sqrt(2));}}});
test('spectral lecture examples: eigenstate, scalar Hamiltonian and common energy shift',()=>{const eigen=spectral({...zInput,stateReal1:0}),scalar=spectral({...zInput,centerNumerator:2,gapNumerator:0}),shift=spectral({...zInput,centerNumerator:3}),plain=spectral(zInput);assert(eigen.classification.stationaryRayProved);assert.equal(eigen.initialEnergyVariance.sign,0);assert(scalar.classification.stationaryRayProved);assert.equal(scalar.energySpectrum.length,1);close(scalar.energySpectrum[0].eigenvalue.value,2);close(shift.initialEnergyMean.value,3);close(shift.initialEnergyVariance.value,1);for(let i=0;i<2;i++)close(shift.records.at(-1)!.standardProbabilities[i].value,plain.records.at(-1)!.standardProbabilities[i].value!);assert.notDeepEqual(shift.records.at(-1)!.amplitudes,plain.records.at(-1)!.amplitudes);});
test('pulse lecture example: equal Z histograms but unequal X probabilities and fidelity',()=>{const x=P('X then Z'),r=x.records.at(-1)!;for(const path of [r.forward,r.reverse]){close(path.standardProbabilities[0].value,Math.cos(1)**2);close(path.standardProbabilities[1].value,Math.sin(1)**2);}close(r.forward.plusXProbability.value,(1+Math.sin(2)**2)/2);close(r.reverse.plusXProbability.value,.5);close(r.comparisons.order.stateFidelity.value,1-Math.sin(2)**2*Math.sin(1)**2);close(r.comparisons.order.stateFidelity.value,.4145494543844084);close(r.comparisons.order.propagatorSquaredDifference.value,8*Math.sin(1)**4);assert(!x.classification.hamiltoniansCommute);});
test('pulse lecture example: rational switch exposures at t=4/5 sum to the same clock time',()=>{const x=P('off-grid switch times'),r=x.records.find(r=>r.time.value===4/5)!;assert(r);close(x.totalDuration.value,22/15);close(r.forwardDurations.a.value,2/3);close(r.forwardDurations.b.value,2/15);assert.equal(r.reverseDurations.a.sign,0);close(r.reverseDurations.b.value,4/5);});
test('pulse lecture example: commuting Z and 2Z disagree in the middle and coincide at the endpoint',()=>{const x=P('commuting unequal schedule'),r=x.records.find(r=>r.time.value===1)!;assert(x.classification.endpointEqualityProved);assert(!x.classification.allThreePrefixesEqualProved);close(r.accumulatedActionForward[0][0].real.value,1);close(r.accumulatedActionReverse[0][0].real.value,2);close(x.averageHamiltonian![0][0].real.value,1.5);assert(r.comparisons.order.propagatorSquaredDifference.value!>1);assert.equal(x.records.at(-1)!.comparisons.order.propagatorSquaredDifference.resolution,'unresolved-at-precision-ceiling');});
test('pulse lecture examples: cancelled action and empty schedule do not erase intermediate evolution or define an average at T=0',()=>{const x=P('opposite axes cancellation'),r=x.records.find(r=>r.time.value===1)!;assert.equal(x.totalIntegratedAction[0][0].real.sign,0);assert(r.comparisons.forwardAverage.stateFidelityDefect.value!>.1);assert.equal(x.records.at(-1)!.average.propagator[0][0].real.value,1);const empty=P('zero total duration');assert.equal(empty.averageHamiltonian,null);assert.equal(empty.records.length,1);assert.equal(empty.records[0].average.propagator[0][0].real.value,1);});
