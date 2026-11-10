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
import {finiteKuramotoModel as run,type FiniteKuramotoParameters} from '../src/lib/learning-finite-kuramoto';
const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/finite-kuramoto-collective-order-chapter.${lang}.md`,'utf8')]));
for(const lang of ['ko','en'])test(`finite Kuramoto ${lang}: detailed bilingual manuscript renders mathematics and explained exercises`,async()=>{
 const s=source[lang];assert.equal((s.match(/^## /gm)||[]).length,14);assert.equal((s.match(/\$\$[\s\S]*?\$\$/g)||[]).length,10);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\| ---/gm)||[]).length,2);if(lang==='en')assert(!/[가-힣]/.test(s));
 const html=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(s);assert.equal(html.messages.length,0);assert(!String(html).includes('katex-error'));assert.equal((String(html).match(/class="katex-display"/g)||[]).length,10);
});
test('finite Kuramoto worked tables match independent preparations and preserve the finite/infinite distinction',()=>{
 assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));
 const refs=JSON.parse(readFileSync('scripts/fixtures/finite-kuramoto-reference.json','utf8'))as{cases:{label:string;input:FiniteKuramotoParameters}[]};const model=(label:string)=>run(refs.cases.find(c=>c.label===label)!.input);
 const aligned=model('identical aligned'),anti=model('identical antiphase clusters'),quarter=model('identical quarter roots');
 for(const r of [aligned,anti,quarter])assert(r.current.velocities.every(v=>v===1));
 assert.equal(aligned.current.firstHarmonic.magnitude,1);assert.equal(anti.current.firstHarmonic.magnitude,0);assert.equal(anti.current.secondHarmonic.magnitude,1);assert.equal(quarter.current.firstHarmonic.magnitude,0);assert.equal(quarter.current.secondHarmonic.magnitude,0);assert.equal(quarter.current.balancedGroupHarmonic!.magnitude,1);
 assert.equal(quarter.continuumReference.nonzeroOrderSquared!.value,1/3);assert.equal(quarter.continuumReference.radialRelaxationTimeAtNonzeroOrder!.value,1);assert.equal(model('below continuum reference').continuumReference.orderRelaxationTimeBelow!.value,2);assert.equal(model('critical continuum reference').continuumReference.radialRelaxationTimeAtNonzeroOrder,null);
 for(const s of Object.values(source)){assert(s.includes('| 0, 0, 0, 0 | 1 | 1 | R₄=1 | 1 |'));assert(s.includes('| 0, 0, π, π | 0 | 1 | R₄=1 | 1 |'));assert(s.includes('| 0, π/2, π, 3π/2 | 0 | 0 | R₄=1 | 1 |'));assert(s.includes('| 3 | 1/2 | 1/3 |'));assert(s.includes('https://arxiv.org/pdf/0806.0004'));}
});
