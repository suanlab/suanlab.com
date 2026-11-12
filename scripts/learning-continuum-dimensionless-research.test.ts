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
    "id": "advection-diffusion-peclet-boundary-layer",
    "columns": 5,
    "equations": 10,
    "image": "peclet-grid-positive-flux",
    "original": [
      "52d588a8807f9678b1e96137d1016d4bfae5011480aa83f264d261fd8ed04e01",
      "46501865809923810ea9b88f2d33207eb9de2389310e795009681867f72f8107"
    ],
    "outline": 14,
    "rows": [
      [
        "0.25000000",
        "-0.00152439",
        "0.01677149",
        "0.00050771",
        "0.00050771"
      ],
      [
        "0.50000000",
        "0.01219512",
        "0.07547170",
        "0.00669285",
        "0.00669285"
      ],
      [
        "0.75000000",
        "-0.11128049",
        "0.28092243",
        "0.08204332",
        "0.08204332"
      ],
      [
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000"
      ]
    ]
  },
  {
    "id": "reaction-diffusion-damkohler-penetration",
    "columns": 5,
    "equations": 10,
    "image": "damkohler-robin-supply-identification",
    "original": [
      "0e29f82d88c4d22713897946f61bfb5e257e939ce49e187847c963ed28ff6f1d",
      "4b3c4332951316d9d2498cc1f9051a52070300c172d415e12d55a2abf2aede30"
    ],
    "outline": 14,
    "rows": [
      [
        "0.10000000",
        "0.04930832",
        "0.09506917",
        "0.02376729",
        "0.02826783"
      ],
      [
        "1.00000000",
        "0.34152362",
        "0.65847638",
        "0.16461909",
        "0.19579110"
      ],
      [
        "10.00000000",
        "0.83835964",
        "1.61640363",
        "0.40410091",
        "0.48062080"
      ],
      [
        "100.00000000",
        "0.98108416",
        "1.89158437",
        "0.47289609",
        "0.56244293"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-dimensionless research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent centered and fitted fluxes, Robin supply and similarity',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-dimensionless-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactUniformGridPairs": 28, "exactZeroNeighborCoefficientControls": 2, "nonuniformFittedFluxCases": 6, "positiveSameRateInversePairs": 4, "robinConsumptionAndSensitivityCases": 27, "robinSimilarityCases": 27, "zeroReactionSupplyControls": 9});
});
