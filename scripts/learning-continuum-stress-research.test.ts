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
    "id": "cauchy-traction-stress-balance",
    "columns": 5,
    "equations": 10,
    "image": "stress-directional-identification",
    "original": [
      "0ca27fcfc66a495f6cfccc440494746b1e2686f485ee2894db159a1a8d8b93bd",
      "ce087dcedfb78f4688b6e98c42aa2835061e344e451004d3283c962a370aeed3"
    ],
    "outline": 18,
    "rows": [
      [
        "5.00000000",
        "0.00527621",
        "0.00881572",
        "276.45298154",
        "12867.20641698"
      ],
      [
        "15.00000000",
        "0.13397460",
        "0.08083776",
        "28.99405294",
        "153.02820107"
      ],
      [
        "30.00000000",
        "0.86602540",
        "0.33763747",
        "6.20274201",
        "8.77200187"
      ],
      [
        "60.00000000",
        "2.59807621",
        "1.22474487",
        "1.41421356",
        "0.66666667"
      ]
    ]
  },
  {
    "id": "finite-deformation-small-strain-rotation",
    "columns": 5,
    "equations": 10,
    "image": "metric-compatibility-loop",
    "original": [
      "1aa62fa9210f46cb80d4a3fbfd4464e4fba50943efdb287e646df5f02e346fea",
      "e109ef3f8d0f878fa0941367dc729f6c583b3891a780fa870705fcef7f06ee3b"
    ],
    "outline": 18,
    "rows": [
      [
        "-0.20000000",
        "0.00000000",
        "0.67032005",
        "-0.40000000",
        "-1.60000000"
      ],
      [
        "0.00000000",
        "0.00000000",
        "1.00000000",
        "0.00000000",
        "0.00000000"
      ],
      [
        "0.10000000",
        "0.00000000",
        "1.22140276",
        "0.20000000",
        "0.80000000"
      ],
      [
        "0.30000000",
        "0.00000000",
        "1.82211880",
        "0.60000000",
        "2.40000000"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-stress research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent exact stress observations, estimator covariance, metric loops and conformal gradients',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-stress-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{balancedDesignAttainsLowerBound:1,exactAdditiveNoisePropagations:81,exactNonDiagonalEstimatorCovariance:1,exactOneTractionNullCases:243,exactRadialAngleLoopCases:36,exactRepeatedAxisRankRejection:1,exactScalarStressRecoveries:27,independentConformalGradientMetricCases:24,independentJacobiSpectrumCases:6,locallyValidGloballyRepeatingMapCases:18});
});
