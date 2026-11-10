import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-stress-strain-registry-hashes.json';
import {getLearningContent} from '../src/lib/learning-content';
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('all484 prior lessons and guides,43 lab groups and475 entries preserve their canonical order and content',()=>{
  const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,484);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);
  const guides=Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]));assert.equal(hash(guides),prior.guides);
  const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(key=>[key,(data as Record<string,any>)[name][key]]))]));assert.equal(Object.keys(labs).length,43);assert.equal(Object.values(prior.labEntries).reduce((n,v)=>n+v.length,0),475);assert.equal(hash(labs),prior.labs);
});
for(const id of ['cauchy-traction-stress-balance','finite-deformation-small-strain-rotation']){
  test(`${id}: detailed bilingual theory, matched math, accessible outlines and six answered exercises render`,async()=>{
    const result=await getLearningContent(id),equations:string[][]=[];
    for(const locale of ['ko','en'] as const){
      const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),html=result[locale],outline=result.outline[locale];
      assert.equal(outline.length,17);assert.equal(new Set(outline.map(h=>h.id)).size,outline.length);for(const h of outline)assert.ok(html.includes(`id="${h.id}"`));
      assert.doesNotMatch(html,/katex-error|<h1/);assert.match(html,/katex-display/);assert.match(html,/tabindex="0"/);assert.match(html,/aria-label=/);
      assert.equal((source.match(/^### (?:Exercise|연습) \d/gm)||[]).length,6);assert.equal((source.match(/\*\*(?:Answer|해설):\*\*/g)||[]).length,6);
      if(locale==='en'){assert.doesNotMatch(html,/[가-힣]/);assert.ok(source.split(/\s+/).length>1400);}else assert.ok(source.length>5500);
      equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));
    }
    assert.ok(equations[0].length>=11);assert.deepEqual(equations[0],equations[1]);
  });
}
