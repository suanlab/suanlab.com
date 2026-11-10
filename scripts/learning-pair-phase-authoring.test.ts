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
import {pairPhaseLockingModel} from '../src/lib/learning-pair-phase-locking';
const source=Object.fromEntries(['ko','en'].map(lang=>[lang,readFileSync(`docs/online-learning/physics-math-for-ai/two-oscillator-locking-phase-slips-chapter.${lang}.md`,'utf8')]));
for(const lang of ['ko','en'])test(`pair phase ${lang}:14sections,10equations,2tables and6explained exercises render`,async()=>{
 const s=source[lang];assert.equal((s.match(/^## /gm)||[]).length,14);assert.equal((s.match(/\$\$[\s\S]*?\$\$/g)||[]).length,10);assert.equal((s.match(/^\*\*\d\./gm)||[]).length,6);assert.equal((s.match(/^\| ---/gm)||[]).length,2);if(lang==='en')assert(!/[가-힣]/.test(s));assert(!/\]\((?!https?:)[^)]+\.md\)/.test(s));
 const html=await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(s);assert.equal(html.messages.length,0);assert(!String(html).includes('katex-error'));assert.equal((String(html).match(/class="katex-display"/g)||[]).length,10);
});
test('bilingual phase equations match and worked tables bind to independent fixed/critical preparations',()=>{
 assert.deepEqual(source.ko.match(/\$\$[\s\S]*?\$\$/g),source.en.match(/\$\$[\s\S]*?\$\$/g));
 const refs=JSON.parse(readFileSync('scripts/fixtures/pair-phase-locking-reference.json','utf8')),find=(label:string)=>refs.cases.find((c:{label:string})=>c.label===label).input;
 const stable=pairPhaseLockingModel(find('stable exact fixed phase')),unstable=pairPhaseLockingModel(find('unstable exact fixed phase')),critical=pairPhaseLockingModel(find('critical from beyond the bottleneck'));
 assert.equal(stable.current.relativeSin.value,3/5);assert.equal(stable.current.relativeCos.value,4/5);assert.equal(unstable.current.relativeCos.value,-4/5);assert.equal(stable.current.firstHarmonicSquared.value,.9);assert.equal(unstable.current.firstHarmonicSquared.value,.1);assert.equal(stable.current.angularFrequency1.value,1.5);
 assert.equal(critical.history[4].time.value,2);assert.equal(critical.history[4].phaseBranch.signedCount,0);assert.equal(critical.current.phaseBranch.signedCount,1);assert.equal(critical.current.completedRelativeTurns.signedCount,0);assert.equal(critical.current.wrappedRelativePhase!.value,0);assert.equal(critical.current.relativeAngularVelocity.value,1);
 for(const s of Object.values(source)){assert(s.includes('| 2 | π | π | 0 | 0 |'));assert(s.includes('| 4 | 2π | 0 | 1 | 0 |'));assert(s.includes('| 3/5 | 4/5 | 9/10 | −4 |'));assert(s.includes('| 3/5 | −4/5 | 1/10 | +4 |'));}
});
