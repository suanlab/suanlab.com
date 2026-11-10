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
import {hopfAmplitudeModel} from '../src/lib/learning-hopf-amplitude';
const sources=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/hopf-amplitude-critical-relaxation-chapter.${lang}.md`,'utf8')]));
for(const language of ['ko','en'])test(`Hopf ${language}: graduate explanation,14sections,10equations,tables and6solutions`,async()=>{
 const s=sources[language];assert.equal((s.match(/^## /gm)||[]).length,14);assert.equal((s.match(/\$\$[\s\S]*?\$\$/g)||[]).length,10);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\| ---/gm)||[]).length,2);assert(!/\]\((?!https?:)[^)]+\.md\)/.test(s));if(language==='en')assert(!/[가-힣]/.test(s));
 const rendered=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(s);assert.equal(rendered.messages.length,0);assert(!String(rendered).includes('katex-error'));assert.equal((String(rendered).match(/class="katex-display"/g)||[]).length,10);
});
test('bilingual manuscripts share equations and the worked table is anchored to independent critical-flow references',()=>{
 assert.deepEqual(sources.ko.match(/\$\$[\s\S]*?\$\$/g),sources.en.match(/\$\$[\s\S]*?\$\$/g));
 const fixtures=JSON.parse(readFileSync('scripts/fixtures/hopf-amplitude-reference.json','utf8')),p=fixtures.cases.find((c:{label:string})=>c.label==='worked critical shear').input,r=hopfAmplitudeModel(p);
 assert.equal(r.history[1].time.value,.5);assert.equal(r.history[1].squaredAmplitude.value,.5);assert.equal(r.history[2].squaredAmplitude.value,1/3);assert.equal(r.history[3].amplitude.value,.5);assert(Math.abs(r.history[2].phaseShift!.value!-(2-Math.log(3)/2))<1e-14);
 for(const s of Object.values(sources)){assert(s.includes('| 1/2 | 1/2 | 1/√2 | 1−ln2/2 |'));assert(s.includes('| 1 | 1/3 | 1/√3 | 2−ln3/2 |'));assert(s.includes('| 3/2 | 1/4 | 1/2 | 3−ln2 |'));}
});
