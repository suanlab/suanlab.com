import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from'node:fs';
import {unified}from'unified';import remarkParse from'remark-parse';import remarkGfm from'remark-gfm';import remarkMath from'remark-math';import remarkRehype from'remark-rehype';import rehypeKatex from'rehype-katex';import rehypeStringify from'rehype-stringify';
import {markovCoarseModel}from'../src/lib/learning-markov-coarse';
const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/markov-partition-lumpability-memory-chapter.${lang}.md`,'utf8')]));
for(const lang of ['ko','en'])test(`coarse Markov ${lang}: graduate explanations, equations, tables and worked exercises render`,async()=>{
 const s=source[lang];assert.equal((s.match(/^## /gm)||[]).length,14);assert.equal((s.match(/\$\$[\s\S]*?\$\$/g)||[]).length,10);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\| ---/gm)||[]).length,2);if(lang==='en')assert(!/[가-힣]/.test(s));
 const h=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(s);assert.equal(h.messages.length,0);assert(!String(h).includes('katex-error'));assert.equal((String(h).match(/class="katex-display"/g)||[]).length,10);
});
test('bilingual formulas and worked values match independent exact preparations',()=>{
 assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));
 const refs=JSON.parse(readFileSync('scripts/fixtures/markov-coarse-reference.json','utf8')),run=(label:string)=>markovCoarseModel(refs.cases.find((c:{label:string})=>c.label===label).input);
 const first=run('stationary-matched nonlumpable'),second=run('same macro different hidden'),stationary=run('stationary law'),lifted=run('lifted initial law'),negative=run('negative truncated pseudo-law');assert.equal(first.history[1].macro[1].value,.25);assert.equal(second.history[1].macro[1].value,.75);assert.equal(stationary.history[1].macro[1].value,1/3);assert.equal(lifted.history[2].macro[0].value,13/16);assert.equal(lifted.history[2].naiveErrorTV.value,1/16);assert.equal(negative.history[2].initialAwareTruncated[1].value,-.25);
 for(const s of Object.values(source)){assert(s.includes('| (1,0,0) | (1,0) | 1/4 | 1/2 |'));assert(s.includes('| (0,1,0) | (1,0) | 3/4 | 1/2 |'));assert(s.includes('| (1/3,1/3,1/3) | (2/3,1/3) | 1/3 | 1/3 |'));assert(s.includes('(13/16,3/16)'));assert(s.includes('(5/4,−1/4)'));}
});
