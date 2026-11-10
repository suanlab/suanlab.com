import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-multiphysics-registry-hashes.json';
import {getLearningContent} from '../src/lib/learning-content';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('496 prior units/guides and43 lab groups487 entries preserve canonical content and order',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));
 assert.equal(units.length,496);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);
 assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(key=>[key,(data as Record<string,any>)[name][key]]))]));assert.equal(hash(labs),prior.labs);
});
for(const[id,headings,equations]of[['thermoelastic-mode-reciprocal-energy-exchange',12,10],['thermoelectric-bar-seebeck-peltier-joule-balance',15,11]]as const){
 test(`${id}: complete bilingual explanations, matched equations and six answered exercises render accessibly`,async()=>{
  const result=await getLearningContent(id),math:string[][]=[];
  for(const locale of['ko','en']as const){
   const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),html=result[locale],outline=result.outline[locale];
   assert.equal(outline.length,headings);assert.equal(new Set(outline.map(h=>h.id)).size,headings);for(const h of outline)assert.ok(html.includes(`id="${h.id}"`));
   assert.doesNotMatch(html,/katex-error|<h1/);assert.match(html,/katex-display/);assert.match(html,/tabindex="0"/);assert.match(html,/aria-label=/);
   assert.equal((source.match(/^### (?:Exercise|문제) \d/gm)||[]).length,6);assert.equal((source.match(/\*\*(?:Answer|풀이):\*\*/g)||[]).length,6);
   if(locale==='en'){assert.doesNotMatch(html,/[가-힣]/);assert.ok(source.split(/\s+/).length>1600);}else assert.ok(source.length>6500);
   math.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));
  }
  assert.equal(math[0].length,equations);assert.deepEqual(math[0],math[1]);
 });
}
