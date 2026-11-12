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
    "id": "kelvin-voigt-maxwell-memory-dissipation",
    "columns": 5,
    "equations": 10,
    "image": "viscoelastic-spectrum-identifiability",
    "original": [
      "c5d360b102bb8d97767c5b9ccd101e526e52fab7c55a922c3fea7b7e36d2333e",
      "aa39e5ed8476d91012dcca37fb65a2e1047db11ff8ddc480113eb998b8ae2c72"
    ],
    "outline": 14,
    "rows": [
      [
        "0.25000000",
        "1.00000000",
        "0.25000000",
        "0.11764706",
        "0.47058824"
      ],
      [
        "0.50000000",
        "1.00000000",
        "0.50000000",
        "0.40000000",
        "0.80000000"
      ],
      [
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000"
      ],
      [
        "2.00000000",
        "1.00000000",
        "2.00000000",
        "1.60000000",
        "0.80000000"
      ]
    ]
  },
  {
    "id": "neo-hookean-uniaxial-constitutive-objectivity",
    "columns": 5,
    "equations": 10,
    "image": "hyperelastic-uniaxial-invisible-shear",
    "original": [
      "3e0f5d8fe5aca2ae2b926a146a68052d579314122cd3e268a945128c9acfc3a3",
      "498bd21332b7cc0919fb40afe8de245f86f2c7a123fed0c596cc75c1c8b4d120"
    ],
    "outline": 16,
    "rows": [
      [
        "0.25000000",
        "0.03125000",
        "0.00099182",
        "0.25000000",
        "0.25023926"
      ],
      [
        "0.50000000",
        "0.12500000",
        "0.06640625",
        "0.50000000",
        "0.50812500"
      ],
      [
        "1.00000000",
        "0.50000000",
        "5.00000000",
        "1.00000000",
        "1.32000000"
      ],
      [
        "1.50000000",
        "1.12500000",
        "71.19140625",
        "1.50000000",
        "4.68937500"
      ]
    ]
  }
];
for(const unit of units)test(`Continuum-constitutive research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent constitutive spectra, full matrix gradients and periodic power',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-continuum-constitutive-research.py'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactDistinctPeriodicInitialStorage": 1, "exactOtherFrequencyComparisons": 108, "exactPositiveInverseSpectrumFits": 27, "exactRotatedUniaxialInvisibleGradients": 16, "exactShearObjectiveGradientCases": 21, "independentCycleWorkAndLossIntegrals": 6, "independentMatrixEnergyGradientDifferences": 27, "independentMemoryConvolutions": 10, "independentPeriodicPowerBalances": 15, "reducedConstraintGradientNegativeControl": 1});
});
