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
    "id": "linearized-euler-acoustic-characteristics",
    "columns": 5,
    "equations": 10,
    "image": "subsonic-incoming-norm",
    "original": [
      "955c432ce8effd27254fba0f73dbfabc506afd4f6f1d2be10a1e3defbcb790d5",
      "483cdbf8a2a669fbf548d72f1be674cf9eafdfd42319ee0ef120023a949ee058"
    ],
    "outline": 15,
    "rows": [
      [
        "0.10000000",
        "0.90453403",
        "1.10554160",
        "2.02020202",
        "1.22222222"
      ],
      [
        "0.30000000",
        "0.73379939",
        "1.36277029",
        "2.19780220",
        "1.85714286"
      ],
      [
        "0.60000000",
        "0.50000000",
        "2.00000000",
        "3.12500000",
        "4.00000000"
      ],
      [
        "0.90000000",
        "0.22941573",
        "4.35889894",
        "10.52631579",
        "19.00000000"
      ]
    ]
  },
  {
    "id": "taylor-green-navier-stokes-viscous-decay",
    "columns": 5,
    "equations": 10,
    "image": "perturbation-energy-exchange",
    "original": [
      "ec6a8836467622b3f048cae8c148ea264b35c73649496eab75d940c42b4a5990",
      "5ceaca9de43fcd8a9cbecd49ada362518c7b8ad65514c0e696a788a8c5d47db4"
    ],
    "outline": 17,
    "rows": [
      [
        "0.50000000",
        "-1.00000000",
        "-0.95000000",
        "0.00000000",
        "1.00000000"
      ],
      [
        "1.00000000",
        "0.00000000",
        "-0.90000000",
        "0.00000000",
        "1.00000000"
      ],
      [
        "5.00000000",
        "8.00000000",
        "-0.50000000",
        "1.60943791",
        "10.91963001"
      ],
      [
        "20.00000000",
        "38.00000000",
        "1.00000000",
        "2.99573227",
        "8924115.04815937"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-flow research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent norm fluxes, Fourier budgets, contact exit and shifted solution momentum',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-flow-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{exactInitialGrowthSignCases:5,exactNormBoundaryCases:64,exactPersistentMeanErrorCases:9,exactTwoModeFourierBudgets:27,independentContactStorageAndExitCases:6,independentEnvelopeIntegrations:16,independentShiftedSolutionMomentumCases:12});
});
