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
import {hermitianSpectrumModel as finite,type HermitianParameters} from '../src/lib/learning-hermitian-spectrum';
import {sequentialObservationModel as sequential,type SequentialObservationParameters} from '../src/lib/learning-sequential-observation';
const ids=['hermitian-spectrum-born-expectation','degenerate-projectors-sequential-observation'];
for(const[index,id]of ids.entries()){
 const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/${id}-chapter.${lang}.md`,'utf8')]));
 for(const lang of ['ko','en'])test(`observable manuscript ${id}.${lang}: assumptions, derivation, examples and six explained exercises render`,async()=>{
  const s=source[lang];assert.equal((s.match(/^## /gm)||[]).length,12);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\*\*(?:Explanation|해설)\./gm)||[]).length,6);assert.equal((s.match(/^\| ---/gm)||[]).length,index+1);assert(s.includes('https://www.preskill.caltech.edu/ph219/chap2_15.pdf'));assert(s.includes('https://interactivetextbooks.tudelft.nl/mqp/hilbert-spaces-and-operators.html'));
  if(lang==='en'){assert(!/[가-힣]/.test(s));assert(s.split(/\s+/).length>1200);assert(s.includes('Interactive Book'));}else{assert(s.length>4500);assert(s.includes('Interactive Book'));}
  const html=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(s);assert.equal(html.messages.length,0);assert(!String(html).includes('katex-error'));assert.equal((String(html).match(/class="katex-display"/g)||[]).length,7+index);
 });
 test(`observable manuscript ${id}: bilingual equations preserve identical conventions`,()=>{assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));for(const s of Object.values(source)){assert(s.includes('null'));assert(s.includes('512'));assert(s.includes('−37/33')||s.includes('(1,0)'));assert(s.includes('POVM'));}});
}
test('observable manuscript hand examples agree with independently checked numerical preparations',()=>{
 const refs=JSON.parse(readFileSync('scripts/fixtures/quantum-observable-reference.json','utf8')) as {finiteCases:{label:string;input:HermitianParameters}[];sequentialCases:{label:string;input:SequentialObservationParameters}[]};
 const base=refs.finiteCases[0].input,z=finite(base),shifted=finite({...base,centerNumerator:-2,centerDenominator:3,gapNumerator:5,gapDenominator:11});assert.equal(z.mean.sign,0);assert.equal(z.variance.value,1);assert.equal(z.nonHermitianCounterexample.expectation.imaginary.value,.5);assert.equal(shifted.outcomes[0].eigenvalue.value,-37/33);assert.equal(shifted.outcomes[1].eigenvalue.value,-7/33);assert.equal(shifted.mean.value,-2/3);assert.equal(shifted.variance.value,25/121);
 const passive=finite({...base,basisReal0:1,basisImag0:0,basisReal1:1,basisImag1:0});assert.equal(passive.passiveBasis.matrix[0][1].real.value,-1);assert.equal(passive.passiveBasis.matrix[1][0].real.value,-1);assert.equal(passive.passiveBasis.coordinates[0].real.value,.5);assert.equal(passive.passiveBasis.coordinates[0].imaginary.value,.5);assert.equal(passive.passiveBasis.coordinates[1].real.value,-.5);assert.equal(passive.passiveBasis.coordinates[1].imaginary.value,.5);
 const S=(label:string)=>sequential(refs.sequentialCases.find(c=>c.label===label)!.input),coarse=S('degenerate coherence preserved');assert.equal(coarse.a.nonselectivePurity.value,1);assert.equal(coarse.refinedA.nonselectivePurity.value,.5);assert.equal(coarse.coarseRefinementSquaredDistance.value,.5);assert.deepEqual(coarse.aThenB.secondMarginal.map(q=>q.value),[0,1]);assert.deepEqual(coarse.refinedAThenB.secondMarginal.map(q=>q.value),[.5,.5]);assert.equal(coarse.a.nonselectiveDensity[1][2].real.value,.5);assert.equal(coarse.refinedA.nonselectiveDensity[1][2].real.sign,0);
 const order=S('order-dependent first plane');assert.deepEqual(order.aThenB.jointProbabilities.map(row=>row.map(q=>q.value)),[[.5,.5],[0,0]]);assert.deepEqual(order.bThenA.jointProbabilities.map(row=>row.map(q=>q.value)),[[.25,.25],[.25,.25]]);const invisible=S('same state statistics do not imply commuting');assert(!invisible.classification.observablesCommute);assert.deepEqual(invisible.aThenB.jointProbabilities.map(row=>row.map(q=>q.value)),[[0,0],[1,0]]);
});
