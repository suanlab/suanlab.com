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
    "id": "reversible-reaction-advection-diffusion-budget",
    "columns": 5,
    "equations": 10,
    "image": "reaction-spatial-composition-information",
    "original": [
      "36842b94c1d646f3aed105afb11546b9fc65c087ab79d41e29c262edf9126368",
      "ecfb6ce2fea46c9d2e0ccfa3d6bac2a652f872e573ae185d58f5d17136ed95c3"
    ],
    "outline": 14,
    "rows": [
      [
        "0.00000000",
        "0.50000000",
        "0.99966465",
        "0.50000000",
        "0.00033535"
      ],
      [
        "0.10000000",
        "0.50000000",
        "0.90853826",
        "0.50000000",
        "0.09146174"
      ],
      [
        "0.40000000",
        "0.50000000",
        "0.72127009",
        "0.50000000",
        "0.27872991"
      ],
      [
        "1.00000000",
        "0.50000000",
        "0.56170352",
        "0.50000000",
        "0.43829648"
      ]
    ]
  },
  {
    "id": "fisher-kpp-reaction-diffusion-front",
    "columns": 5,
    "equations": 10,
    "image": "kpp-spreading-comparison-bump",
    "original": [
      "3df986d63eb1a44ce46d8ebbfe030b726db5dee506fe04e8327bdd932cb6e154",
      "2ffefd39341a0d05eac36f3f45df50726f976a255e86fd288a3eefb22a9b23d0"
    ],
    "outline": 15,
    "rows": [
      [
        "0.00000000",
        "2.22144147",
        "0.50000000",
        "8.48049725e-03",
        "1.69609945e-02"
      ],
      [
        "0.50000000",
        "2.29429488",
        "0.53125000",
        "4.05399153e-03",
        "1.43884077e-02"
      ],
      [
        "1.00000000",
        "2.56509966",
        "0.62500000",
        "1.03329448e-03",
        "7.45175245e-03"
      ],
      [
        "1.80000000",
        "5.09633625",
        "0.90500000",
        "2.33417040e-09",
        "4.58291912e-07"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-reaction research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent reaction mixtures, information balance and upper/lower comparisons',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-reaction-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactConditionalDiffusionHessianCases": 81, "exactReactionLogRatioAndSignCases": 81, "exactReactionSemigroupCases": 144, "independentCompactMovingSubsolutionCases": 45, "independentConditionalInformationBudgets": 4, "independentLinearGaussianRayAndTailCases": 45, "independentMixturePdeDifferenceCases": 18});
});
