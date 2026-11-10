import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';

const units = ['finite-qubit-circuit-order-interference','qubit-channel-kraus-choi-verification','system-environment-stinespring-reference','quantum-instruments-records-feedforward'];
for (const id of units) test(`draft bilingual content loader: ${id}`, async () => {
  const content=await getLearningContent(id);
  for(const lang of ['ko','en'] as const){
    assert.ok(content.outline[lang].length>=7);
    assert.ok(content[lang].includes('katex'));
    assert.ok(!content[lang].includes('katex-error'));
    assert.ok(content[lang].includes('https://learning.quantum.ibm.com/'));
    const source=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');
    assert.ok(source.includes(lang==='ko'?'해설이 있는 연습':'Exercises with explanations'));
    if(lang==='en')assert.ok(!/[가-힣]/.test(content.en));
  }
});

test('registered chapters retain their four explicit IDs',async()=>{
  const registry=await readFile('src/data/interactive-book.ts','utf8');
  for(const id of units)assert.ok(registry.includes(id));
});
