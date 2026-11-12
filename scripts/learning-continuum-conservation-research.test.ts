import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';

const units=[
  {
    "id": "moving-control-volume-mass-transport",
    "columns": 5,
    "equations": 10,
    "image": "contact-weak-window",
    "original": [
      "7742336d351c3b615e5ee7cc71eae2a5938719209642ba8c06026d9acfe11b75",
      "5228213ee3cc5b989aedf6ba2c532d2ddad4564945335b1d849c87d222b658a9"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "2.00000000",
        "0.00000000",
        "0.00000000",
        "0.00000000"
      ],
      [
        "0.25000000",
        "1.50000000",
        "-0.50000000",
        "-0.50000000",
        "0.00000000"
      ],
      [
        "0.50000000",
        "1.00000000",
        "-1.00000000",
        "-1.00000000",
        "0.00000000"
      ],
      [
        "0.75000000",
        "1.00000000",
        "-1.00000000",
        "-1.00000000",
        "0.00000000"
      ]
    ]
  },
  {
    "id": "compressible-flow-momentum-energy-balance",
    "columns": 5,
    "equations": 10,
    "image": "shock-entropy-selection",
    "original": [
      "8eae86b3f949f2c7c52666bb80b581e144fc1ddab0c985894285e3e4db703ba6",
      "51d63d23612f4b697772b51d586f35e45910f606ad959f788383fad25c042946"
    ],
    "outline": 18,
    "rows": [
      [
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "0.00000000",
        "0.00000000"
      ],
      [
        "1.25000000",
        "1.42857143",
        "1.65625000",
        "0.00521109",
        "0.00000000"
      ],
      [
        "2.00000000",
        "2.66666667",
        "4.50000000",
        "0.13091644",
        "0.00000000"
      ],
      [
        "3.00000000",
        "3.85714286",
        "10.33333333",
        "0.44547751",
        "0.00000000"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-conservation research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
  const pairs=['en','ko'].map((lang,index)=>{
    const source=readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8');
    assert.doesNotMatch(source,/[\u0000-\u0008\u000b-\u001f]/);if(lang==='en')assert.doesNotMatch(source,/[가-힣]/);
    const start=source.search(/^## (Research extension:|연구 확장:)/m);assert.ok(start>=0);
    const headerEnd=source.indexOf('\n',start),next=source.slice(headerEnd).search(/^## /m);assert.ok(next>=0);
    const end=headerEnd+next,block=source.slice(start,end);
    assert.equal(createHash('sha256').update(source.slice(0,start)+source.slice(end)).digest('hex'),unit.original[index]);
    assert.equal((block.match(lang==='en'?/\*\*Answer:\*\*/g:/\*\*답:\*\*/g)||[]).length,3);
    const tree=unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(block);
    assert.doesNotMatch(JSON.stringify(tree),/"type":"emphasis"/,"Research symbols must not become Markdown emphasis");
    const math=tree.children.flatMap(n=>n.type==='math'?[n.value]:[]),code=tree.children.flatMap(n=>n.type==='code'&&n.lang==='python'?[n.value]:[]);
    assert.equal(math.length,unit.equations);assert.equal(code.length,1);
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    assert.deepEqual(tables[0].children.slice(1).map(row=>row.children.map(cell=>cell.children.map(n=>'value' in n?n.value:'').join(''))),unit.rows);
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,unit.outline);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`continuum-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
test('exact crossing budgets, missing jump sources, boosted shock conservation and entropy signs',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-conservation-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{coincidentMaterialFaceCases:3,exactBoostedShockJumpCases:60,exactEnthalpyConversionCases:15,exactEntropyDerivativeCases:15,exactMovingWindowBudgets:96,exactStoredQuantityBoostCases:60,missedJumpSourceNegativeControls:3,reversedConservationPassesEntropyFails:60,zeroWidthRejections:1});
});
