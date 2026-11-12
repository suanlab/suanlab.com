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
    "id": "reynolds-ensemble-stress-closure-defect",
    "columns": 5,
    "equations": 10,
    "image": "reynolds-fluctuation-forcing-budget",
    "original": [
      "c0c3449132c4f468cc14ac421b5fef971f82e06cb0fe2f3339878837f53894d9",
      "390b93c7a2bea1bfa2a4ae3a2aef0731b708c4b9fcd14e4c12b47dedb88045ef"
    ],
    "outline": 13,
    "rows": [
      [
        "0.00000000",
        "0.00000000",
        "0.00000000",
        "0.00152320",
        "-0.00152320"
      ],
      [
        "0.00100000",
        "0.01668814",
        "0.00042232",
        "0.00152320",
        "-0.00110087"
      ],
      [
        "0.01000000",
        "0.16688140",
        "0.00422323",
        "0.00152320",
        "0.00270004"
      ],
      [
        "0.10000000",
        "1.66881402",
        "0.04223232",
        "0.00152320",
        "0.04070912"
      ]
    ]
  },
  {
    "id": "gaussian-filtered-burgers-subgrid-transfer",
    "columns": 5,
    "equations": 10,
    "image": "gaussian-nested-filter-stress-transfer",
    "original": [
      "72ec4d84f083d74fb94eb849b157a8f9671b22dfb2495cfc0e6dd5e9459c5968",
      "9f8c5ffa1f658aed4c06cd37d80bfa07a3e735e0d56f23e7e8a8aa5d78ba8cc1"
    ],
    "outline": 15,
    "rows": [
      [
        "1.00000000",
        "-0.00000000",
        "0.75852762",
        "0.05779839",
        "0.05779839"
      ],
      [
        "0.66666667",
        "0.90051664",
        "1.17741002",
        "0.05779839",
        "0.03076172"
      ],
      [
        "0.50000000",
        "1.17741002",
        "1.40059220",
        "0.05779839",
        "0.01752924"
      ],
      [
        "0.25000000",
        "1.66510922",
        "1.82974120",
        "0.05779839",
        "0.00439434"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-turbulence research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent actual force budgets, Gaussian convolutions and exact transfer counterexample',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-turbulence-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"actualEightPhaseForceBudgetCases": 36, "negativeFluctuationForcingCases": 6, "exactInviscidDissipationControls": 12, "actualNestedGaussianCases": 60, "zeroWidthFamilies": 6, "exactTransferCounterexamples": 1});
});
