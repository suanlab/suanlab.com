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
    "id": "fourier-heat-diffusion-mode-budget",
    "columns": 5,
    "equations": 10,
    "image": "heat-entropy-availability",
    "original": [
      "bffd636868b9dd9a69a20997e3cc82f5cd3723eb5d0d759993d6df016edc0434",
      "8206a426d88f32a869d85486abb163e687efb592b6b4f580fd565917b616434a"
    ],
    "outline": 16,
    "rows": [
      [
        "0.00000000",
        "0.20000000",
        "0.01015342",
        "0.01000000",
        "0.02062073"
      ],
      [
        "0.50000000",
        "0.12130613",
        "0.00369926",
        "0.00367879",
        "0.00743980"
      ],
      [
        "1.00000000",
        "0.07357589",
        "0.00135611",
        "0.00135335",
        "0.00271774"
      ],
      [
        "2.00000000",
        "0.02706706",
        "0.00018321",
        "0.00018316",
        "0.00036651"
      ]
    ]
  },
  {
    "id": "gaussian-fick-diffusion-spreading",
    "columns": 5,
    "equations": 10,
    "image": "gaussian-relative-information-distance",
    "original": [
      "5abec547e1bd7d06979eb19be36584dfaec6ecc4127cb54c55b1acf032808b7e",
      "2009a0a394646375e8cb6190b389ce7c9c7cf06392a64bdfe9001347a84e64ab"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "1.00000000",
        "0.50000000",
        "1.00000000",
        "1.00000000"
      ],
      [
        "1.00000000",
        "2.00000000",
        "0.25000000",
        "0.25000000",
        "1.00000000"
      ],
      [
        "4.00000000",
        "5.00000000",
        "0.10000000",
        "0.04000000",
        "1.00000000"
      ],
      [
        "9.00000000",
        "10.00000000",
        "0.05000000",
        "0.01000000",
        "1.00000000"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-diffusion research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent physical entropy, Gaussian score integrals and exact transport costs',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-diffusion-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactGaussianScoreAndDissipationCases": 81, "exactNormalizedShapeMassBlindnessCases": 9, "exactPersistentCenterErrorCases": 15, "exactQuadraticCouplingCostCases": 108, "independentGaussianLogAndScoreIntegrals": 27, "independentLocalEntropyBalanceDifferences": 12, "independentPhysicalHeatPeriodIntegrals": 20});
});
