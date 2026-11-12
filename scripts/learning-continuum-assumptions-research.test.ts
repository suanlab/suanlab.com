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
    "id": "particle-kernel-continuum-fields",
    "columns": 5,
    "equations": 10,
    "image": "ballistic-energy-moment-closure",
    "original": [
      "271f32cd081cfbfe68070c8d9ba3c2e9cf0e8550d9cbbb9dbbb1e39c330ef6af",
      "0096267b4847457d2278065c5c40cecc3603b0bee257a7cd3f6467e21a23426b"
    ],
    "outline": 18,
    "rows": [
      [
        "0.50000000",
        "0.80000000",
        "0.20000000",
        "1.00000000",
        "0.75000000"
      ],
      [
        "1.00000000",
        "0.50000000",
        "0.50000000",
        "1.00000000",
        "0.00000000"
      ],
      [
        "2.00000000",
        "0.20000000",
        "0.80000000",
        "1.00000000",
        "-0.75000000"
      ],
      [
        "4.00000000",
        "0.05882353",
        "0.94117647",
        "1.00000000",
        "-1.87500000"
      ]
    ]
  },
  {
    "id": "continuum-scale-separation-counting",
    "columns": 5,
    "equations": 10,
    "image": "paired-count-window-covariance",
    "original": [
      "0c6142f1a0a9c94a77d295af4a1d58280bd2cdcdde66c3394214fa50c9053f5f",
      "11da460300386a77cbde72dc00bf974b56113f3c0669813377736ae35768b7ab"
    ],
    "outline": 18,
    "rows": [
      [
        "0.50000000",
        "0.50000000",
        "1.00000000",
        "1.41421356",
        "0.00000000"
      ],
      [
        "1.00000000",
        "4.00000000",
        "1.00000000",
        "0.50000000",
        "0.50000000"
      ],
      [
        "2.00000000",
        "32.00000000",
        "1.50000000",
        "0.21650635",
        "0.25000000"
      ],
      [
        "4.00000000",
        "256.00000000",
        "1.75000000",
        "0.08267973",
        "0.12500000"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-assumptions research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent exact energy transfers, frame transformations, parent overlaps and count thresholds',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-assumptions-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{dependentPoissonMarginalCases:1,emptySupportCases:1,exactEnergyBoosts:42,exactLocalEnergyBudgets:14,exactParentGeometryCases:144,exactThresholdBisections:80,positiveSameLowerMomentPreparations:5});
});
