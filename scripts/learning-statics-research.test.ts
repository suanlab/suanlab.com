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
import {electrostaticSphere,magnetostaticCylinder} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "gauss-symmetry-electrostatic-sphere",
    "columns": 4,
    "equations": 8,
    "image": "radial-density-inverse",
    "original": [
      "de43de2611b174d0337a699d0d0c2481f3fd2bdc9059c13b99896cad373f904e",
      "7303eb29693eb973c8bdf0254f19fdf9f044adfbaddd7512bc79619d73b00715"
    ],
    "outline": 17,
    "rows": [
      [
        -1.0,
        0.725,
        1.65,
        0.25
      ],
      [
        0.0,
        0.5,
        1.5,
        0.25
      ],
      [
        0.5,
        0.3875,
        1.425,
        0.25
      ],
      [
        1.0,
        0.275,
        1.35,
        0.25
      ]
    ]
  },
  {
    "id": "ampere-symmetry-steady-current-cylinder",
    "columns": 4,
    "equations": 11,
    "image": "skin-diffusion",
    "original": [
      "2669ed0e5256e27feeee061562e128bd6ca61c4e7cf56137524269b56961bc5f",
      "8c9ca3d433c761aba920dfff9e474cd43009bae55d7300246afbd20324687bad"
    ],
    "outline": 16,
    "rows": [
      [
        0.1,
        0.999973959,
        1.00005208,
        0.0124996745
      ],
      [
        1.0,
        0.997404616,
        1.00518673,
        0.124675942
      ],
      [
        10.0,
        0.803660705,
        1.3720872,
        1.02495076
      ],
      [
        32.0,
        0.323501888,
        2.27380107,
        1.97099225
      ]
    ]
  }
];
for(const unit of units)test(`Statics research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
  const pairs=['en','ko'].map((lang,index)=>{
    const source=readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8');
    assert.doesNotMatch(source,/[\u0000-\u0008\u000b-\u001f]/);if(lang==='en')assert.doesNotMatch(source,/[가-힣]/);
    const start=source.search(/^## (Research extension:|연구 확장:)/m);assert.ok(start>=0);
    const headerEnd=source.indexOf('\n',start),next=source.slice(headerEnd).search(/^## /m);assert.ok(next>=0);
    const end=headerEnd+next,block=source.slice(start,end);
    assert.equal(createHash('sha256').update(source.slice(0,start)+source.slice(end)).digest('hex'),unit.original[index]);
    assert.equal((block.match(lang==='en'?/\*\*Answer:\*\*/g:/\*\*답:\*\*/g)||[]).length,3);
    const tree=unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(block);
    const math=tree.children.flatMap(n=>n.type==='math'?[n.value]:[]),code=tree.children.flatMap(n=>n.type==='code'&&n.lang==='python'?[n.value]:[]);
    assert.equal(math.length,unit.equations);assert.equal(code.length,1);
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,unit.outline);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`electromagnetism-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('shell quadrature, inverse sensitivity, independent Bessel integrals and MQS power checks',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-statics-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"sphereMassIntegrals": 5, "shellPotentials": 40, "shellFieldGradients": 30, "interiorRecoveries": 25, "exteriorAmbiguities": 20, "probeSensitivityChecks": 101, "besselAngularChecks": 60, "radialDifferentialChecks": 54, "currentNormalizations": 6, "complexPowerChecks": 12, "dcLimitSequences": 1});
});
test('native uniform sphere agrees with the supplemental alpha-zero case and common exterior',()=>{
 const refs=JSON.parse(execFileSync('python3',['scripts/learning/verify-statics-research.py','--references'],{encoding:'utf8',timeout:20000})) as {sphere:{R:number;x:number;Q:number;k:number;field:number;potential:number}[]};
 assert.equal(refs.sphere.length,18);
 for(const r of refs.sphere){
  const result=electrostaticSphere({radius:r.R,totalCharge:r.Q,coulombConstant:r.k},[r.R*r.x,0,0]);
  assert.equal(result.sourceFieldStatus,'finite');close(result.sourceField![0],r.field);close(result.sourcePotential!,r.potential);
  assert.deepEqual(result.sourceField!.slice(1),[0,0]);
  if(r.x>=1){
   for(const alpha of [-1,-.5,.5,1]){
    // Exact shell mass cancellation leaves the complete exterior independent of alpha.
    const Q=r.Q*(1+3*alpha*(1/5-(3/5)*(1/3)));
    close(result.sourceField![0],r.k*Q/(r.R*r.x)**2);
    close(result.sourcePotential!,r.k*Q/(r.R*r.x));
   }
  }
 }
});
test('native steady cylinder remains the zero-frequency limiting field',()=>{
 const {cylinder}=JSON.parse(execFileSync('python3',['scripts/learning/verify-statics-research.py','--references'],{encoding:'utf8',timeout:20000})) as {cylinder:{R:number;x:number;current:number;mu:number;field:number}[]};
 assert.equal(cylinder.length,30);
 for(const r of cylinder){
  const result=magnetostaticCylinder({radius:r.R,totalCurrent:r.current,permeability:r.mu},[r.R*r.x,0,0]);
  assert.equal(result.sourceFieldStatus,'finite');close(result.sourceField![1],r.field);
  close(result.sourceField![0],0);close(result.sourceField![2],0);
  close(result.sourceCurrentDensity!,r.current/(Math.PI*r.R*r.R));
  close(result.enclosedCurrent!,r.current*Math.min(r.x*r.x,1));
 }
});
