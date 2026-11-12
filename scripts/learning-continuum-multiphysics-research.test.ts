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
    "id": "thermoelastic-mode-reciprocal-energy-exchange",
    "columns": 5,
    "equations": 10,
    "image": "thermoelastic-relaxation-spectrum",
    "original": [
      "9cc966c7f409e84441ddc676897994d248561b8d550b7d0baec51767601f0221",
      "3e0ff9dd8998d6f504359051d415899d4847021294d7dd69e1d428b352b26064"
    ],
    "outline": 13,
    "rows": [
      [
        "0.00000000",
        "0.00000000",
        "0.00000000",
        "1.00000000",
        "1.00000000"
      ],
      [
        "0.50000000",
        "0.11987770",
        "0.45936268",
        "0.88012230",
        "1.00000000"
      ],
      [
        "1.00000000",
        "0.42202815",
        "0.69845600",
        "0.57797185",
        "1.00000000"
      ],
      [
        "2.00000000",
        "0.97568156",
        "0.21783962",
        "0.02431844",
        "1.00000000"
      ]
    ]
  },
  {
    "id": "thermoelectric-bar-seebeck-peltier-joule-balance",
    "columns": 5,
    "equations": 10,
    "image": "thermoelectric-load-power-efficiency",
    "original": [
      "7b1a0b1df9ec4c5c38a0698ad5f2713d363b225c0e2ac5b27e90476e56f02bf5",
      "97b337eeddfbcdca9afaaa26fa4f7c247c51d9b6ef12595f15bc562a7a0ddf71"
    ],
    "outline": 16,
    "rows": [
      [
        "0.50000000",
        "20.00000000",
        "20.00000000",
        "240.00000000",
        "0.08333333"
      ],
      [
        "1.00000000",
        "15.00000000",
        "22.50000000",
        "188.75000000",
        "0.11920530"
      ],
      [
        "4.09267639",
        "5.89081216",
        "14.20226970",
        "88.95466257",
        "0.15965739"
      ],
      [
        "10.00000000",
        "2.72727273",
        "7.43801653",
        "52.35537190",
        "0.14206788"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-multiphysics research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent characteristic coefficients, time quadratures and loaded-leg entropy',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-multiphysics-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactCubicAndHurwitzCases": 45, "adiabaticPropagatorCases": 27, "independentTimeQuadratureCases": 27, "exactPersistentZeroControls": 6, "exactLoadedLegPortCases": 45, "localEntropyQuadratureCases": 45, "distinctObjectiveOptimumCases": 9});
});
