import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';
test('actual Godunov/Roe chapter renders detailed bilingual theory, matched equations and six worked answers',async()=>{
  const id='godunov-burgers-conservative-shock-entropy',result=await getLearningContent(id),equations:string[][]=[];
  for(const locale of ['ko','en']as const){
    const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),html=result[locale];
    assert.equal(result.outline[locale].length,14);assert.equal(new Set(result.outline[locale].map(h=>h.id)).size,14);
    assert.doesNotMatch(html,/katex-error|<h1/);assert.match(html,/katex-display/);assert.match(html,/<table/);assert.match(html,/<pre/);assert.match(html,/tabindex="0"/);
    assert.equal((source.match(/^\d\. \*\*/gm)||[]).length,6);assert.match(source,/Burgers_approximate.html/);assert.match(source,/Tadmor_Acta2003.pdf/);
    if(locale==='en'){assert.doesNotMatch(html,/[가-힣]/);assert.ok(source.split(/\s+/).length>1800);}else assert.ok(source.length>8000);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m=>m[1].replace(/\s+/g,' ').trim()));
  }
  assert.deepEqual(equations[0],equations[1]);assert.ok(equations[0].length>=10);
});
