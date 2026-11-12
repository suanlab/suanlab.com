import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-poynting-energy-registry-hashes.json';
import {getLearningContent} from '../src/lib/learning-content';
const ids=['poynting-source-work-local-balance','transverse-pulse-energy-flux-balance'];
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('470 prior units,guides,42 lab groups and461 entries preserve their canonical baseline',()=>{
 const set=new Set(prior.ids),units=data.learningUnits.filter(u=>set.has(u.id));assert.equal(units.length,470);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);
 const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(Object.keys(labs).length,42);assert.equal(Object.values(prior.labEntries).reduce((n,keys)=>n+keys.length,0),461);assert.equal(hash(labs),prior.labs);
});
test('four detailed manuscripts render paired equations,unique navigable outlines and12 answered exercises',async()=>{
 for(const id of ids){const result=await getLearningContent(id),maths:string[][]=[];
  for(const locale of['ko','en']as const){const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),html=result[locale],outline=result.outline[locale];assert.ok(outline.length>=15);assert.equal(new Set(outline.map(h=>h.id)).size,outline.length);for(const h of outline)assert.ok(html.includes(`id="${h.id}"`));assert.ok(!/<h1/.test(html));assert.ok(!html.includes('katex-error'));assert.ok(html.includes('katex-display'));assert.ok(html.includes('tabindex="0"'));assert.ok(html.includes('aria-label='));assert.equal((source.match(/^### (?:Exercise|연습) \d/gm)||[]).length,6);const exercises=source.match(/^## (?:Exercises with explanations|풀이가 있는 연습문제)\n([\s\S]*?)(?=^## (?:References and next connections?|참고 자료와 다음 연결))/m);assert.ok(exercises);assert.equal((exercises[1].match(/\*\*(?:Answer|해설):\*\*/g)||[]).length,6);if(locale==='en'){assert.ok(!/[가-힣]/.test(html));}maths.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));}
  assert.ok(maths[0].length>=8);assert.deepEqual(maths[0],maths[1]);
 }
});
