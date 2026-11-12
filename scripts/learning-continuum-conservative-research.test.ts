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
    "id": "nonuniform-volume-shared-flux-mass-norm",
    "columns": 5,
    "equations": 10,
    "image": "volume-markov-convex-entropy",
    "original": [
      "c334da6e6917b27b58ca74191fa0ec79d5cd2f3f7eb5db4cd9b54b1cc43c20e6",
      "e9d859c80aae4c0779936d34cbab77fd72704bfd88e3e6a1b0521413c0637d86"
    ],
    "outline": 15,
    "rows": [
      [
        "0.10000000",
        "0.10000000",
        "1.00000000",
        "0.33000000",
        "1.00000000"
      ],
      [
        "0.50000000",
        "0.00000000",
        "1.00000000",
        "0.25000000",
        "1.00000000"
      ],
      [
        "0.60000000",
        "-0.20000000",
        "1.00000000",
        "0.38000000",
        "1.40000000"
      ],
      [
        "0.66666667",
        "-0.33333333",
        "1.00000000",
        "0.50000000",
        "1.66666667"
      ]
    ]
  },
  {
    "id": "godunov-burgers-conservative-shock-entropy",
    "columns": 5,
    "equations": 10,
    "image": "burgers-entropy-conservative-time-step",
    "original": [
      "f8a4e5f01891dfed65391b84eb17588647599da002eced7a0f2d67100b26f6b8",
      "c7928a5135fe229710c839f5e8681b4722bf353126484f62445853167dae5e3a"
    ],
    "outline": 15,
    "rows": [
      [
        "0.00000000",
        "0.01000000",
        "-0.00500000",
        "0.00007500",
        "3.00000000"
      ],
      [
        "0.00000000",
        "0.10000000",
        "-0.05000000",
        "0.00750000",
        "3.00000000"
      ],
      [
        "2.00000000",
        "0.10000000",
        "0.25000000",
        "-0.50250000",
        "3.00000000"
      ],
      [
        "2.00000000",
        "0.70000000",
        "-0.45000000",
        "0.57750000",
        "3.00000000"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-conservative research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent weighted stochastic evolutions and exact entropy flux/time budgets',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-conservative-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"weightedSemigroupAndConvexEntropyCases": 15, "includesDisconnectedAndZeroGenerator": true, "independentTaylorAndUniformization": true, "exactFaceEntropyCases": 75, "exactNonuniformPeriodicEulerBudgets": 36, "exactNormPositivityCounterexamples": 1, "exactArithmeticFluxEntropyCounterexamples": 1});
});
