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
import {maxwellPlaneWaveState,maxwellRadialState} from '../src/lib/learning-maxwell';
import type {StaticVector} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "maxwell-plane-wave-local-integral-laws",
    "columns": 4,
    "equations": 9,
    "image": "staggered-wave-stability",
    "original": [
      "bab7e5fddfacf3a8038bd5a98dd8eb8b20910cdccb9cfb21477ccb372083ee0a",
      "278ea278ec26b930136bc1ef291083b9e114ded3a76c03a5b11712b96ac13582"
    ],
    "outline": 16,
    "rows": [
      [
        0.5,
        -0.4181366,
        -0.80984497,
        1.0
      ],
      [
        1.0,
        -1.0,
        1.0,
        1.0
      ],
      [
        1.5,
        -0.41764832,
        1.27432251,
        1.0
      ],
      [
        2.0,
        17.0,
        -16.0,
        1.0
      ]
    ]
  },
  {
    "id": "maxwell-charge-displacement-continuity",
    "columns": 4,
    "equations": 9,
    "image": "periodic-gauss-repair",
    "original": [
      "4b3dee5557326269f75517c971d8fb23d9ab5b52599576a74994dd5b84fb3007",
      "cc4829b57a0e623c44873b3ed2e5dd1036fde0f92ab70d6c6b813c2e11462623"
    ],
    "outline": 17,
    "rows": [
      [
        0.0,
        2.0,
        1.0,
        0.0
      ],
      [
        0.5,
        1.0,
        1.0,
        0.25
      ],
      [
        1.0,
        0.0,
        1.0,
        1.0
      ],
      [
        2.0,
        2.0,
        1.0,
        4.0
      ]
    ]
  }
];
for(const unit of units)test(`Maxwell research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,unit.outline);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`electromagnetism-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent full-grid updates, exact endpoint invariants, constrained solves and energy balances',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-maxwell-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactModalInvariants": 984, "stableModeBounds": 738, "criticalClosedForms": 41, "supercriticalSpectra": 4, "fullGridTrajectories": 36, "dispersionPairs": 24, "gaussProjections": 144, "independentPrefixSolutions": 144, "energyIdentities": 144, "continuityDefects": 108, "effectiveCurrentChecks": 72, "incompatibleChargeRejections": 36});
});
test('native counterpropagating waves initialize the bounded critical standing-wave eigenline',()=>{
 const {initializations}=JSON.parse(execFileSync('python3',['scripts/learning/verify-maxwell-research.py','--references'],{encoding:'utf8',timeout:20000})) as {initializations:{eps:number;mu:number;N:number;dx:number;k:number}[]};
 assert.equal(initializations.length,2);
 let evaluations=0;
 for(const r of initializations){
  const c=1/Math.sqrt(r.eps*r.mu),dt=r.dx/c;
  for(let j=0;j<r.N;j++){
   const values=(point:StaticVector,time:number)=>{
    const waves=([1,-1] as const).map(direction=>{evaluations++;return maxwellPlaneWaveState({amplitude:1,wavenumber:r.k,direction,permittivity:r.eps,permeability:r.mu},point,time);});
    return {E:(waves[0].electricField![0]+waves[1].electricField![0])/2,B:(waves[0].magneticField![1]+waves[1].magneticField![1])/2};
   };
   const E=values([0,0,j*r.dx],0).E;
   const B=values([0,0,(j+.5)*r.dx],-dt/2).B;
   close(E,(-1)**j);close(c*B,-((-1)**j));
  }
  let a=1,b=-1;
  for(let n=0;n<20;n++){close(a+b,0);close(a*a+b*b,2);b+=2*a;a-=2*b;}
 }
 assert.equal(evaluations,32);
});
test('native bounded radial source preserves a fixed Gauss defect while continuity and currents cancel',()=>{
 const {radial}=JSON.parse(execFileSync('python3',['scripts/learning/verify-maxwell-research.py','--references'],{encoding:'utf8',timeout:20000})) as {radial:{initialGradient:number;gradientRate:number;chargeOffset:number;permittivity:number;permeability:number;time:number;point:StaticVector;rho:number;gauss:number}[]};
 assert.equal(radial.length,6);
 for(const r of radial){
  const actual=maxwellRadialState(r,r.point,r.time);
  close(actual.chargeDensity!,r.rho);close(actual.gaussConstraintResidual!,r.gauss);
  close(actual.continuityResidual!,0);
  for(const value of actual.currentDensitySum!)close(value!,0);
 }
});
