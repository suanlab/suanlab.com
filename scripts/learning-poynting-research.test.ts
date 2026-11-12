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
import {inductionEnergyState,transversePulseEnergy} from '../src/lib/learning-poynting-energy';
import {teachingMaxwellPermittivity,teachingMaxwellPermeability} from '../src/lib/learning-maxwell';
import type {StaticVector} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "poynting-source-work-local-balance",
    "columns": 4,
    "equations": 9,
    "image": "power-invisible-current",
    "original": [
      "7ad81389e8c9133267ced269713b73062032208bf2289570bd657b7a1d01b12c",
      "ca6e6f82db07bd6a4cc9c7997c2c73e2b65968649e18f753d0beffab1aa32c7b"
    ],
    "outline": 16,
    "rows": [
      [
        "-2.00000000",
        "2.00000000",
        "-1.00000000",
        "2.00000000"
      ],
      [
        "0.00000000",
        "2.00000000",
        "1.00000000",
        "2.00000000"
      ],
      [
        "1.00000000",
        "2.00000000",
        "2.00000000",
        "2.00000000"
      ],
      [
        "3.00000000",
        "2.00000000",
        "4.00000000",
        "2.00000000"
      ]
    ]
  },
  {
    "id": "transverse-pulse-energy-flux-balance",
    "columns": 4,
    "equations": 9,
    "image": "vacuum-zero-flux-frame",
    "original": [
      "b10dc5f937f1e86f259061ba354f29d8433960d5136b4b92b9557dc594e7a081",
      "68170408d8f6bcf8f684c067471557d3adca2a6808e75d11772bd5f0cf0d01ee"
    ],
    "outline": 17,
    "rows": [
      [
        "0",
        "1",
        "unavailable",
        "unavailable"
      ],
      [
        "1e-06",
        "0.999999999998",
        "0.999998000002",
        "2e-06"
      ],
      [
        "0.1",
        "0.980198019802",
        "0.818181818182",
        "0.2"
      ],
      [
        "1",
        "0",
        "-0",
        "2"
      ]
    ]
  }
];
for(const unit of units)test(`Poynting research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`electromagnetism-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent power tensor, deficient designs, passivity, vacuum boosts and null references',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-poynting-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"powerBlindCurrents": 756, "symmetricTensorReconstructions": 60, "rankDeficientProbeRejections": 12, "powerNoiseVariances": 60, "passivityClassifications": 80, "negativePowerWitnesses": 61, "axialCurrentCounterexamples": 72, "exactStressTransforms": 245, "zeroFluxFrames": 36, "electricMagneticRestClasses": 36, "nullFrameRejections": 12, "zeroFieldRejections": 1, "rapidityFieldChecks": 100, "boostCompositionChecks": 9, "nearNullChecks": 3, "nearNullDecimalChecks": 3});
});
test('native prescribed energy field makes axial current hypotheses power-invisible without certifying Ampere',()=>{
 const {induction}=JSON.parse(execFileSync('python3',['scripts/learning/verify-poynting-research.py','--references'],{encoding:'utf8',timeout:20000})) as {induction:{position:StaticVector;time:number;E:StaticVector;J:StaticVector;power:number}[]};
 assert.equal(induction.length,12);let hypotheses=0,failures=0;
 for(const r of induction){
  const actual=inductionEnergyState({magneticOffset:1,magneticRate:.5,magneticAcceleration:.25,permittivity:1,permeability:1},r.position,r.time);
  for(let i=0;i<3;i++){close(actual.electricField![i],r.E[i]);close(actual.conductionCurrent![i],r.J[i]);close(actual.ampereResidual![i],0);}
  close(actual.fieldToMatterPowerDensity!,r.power);close(actual.groupedResidual!,0);
  for(const q of [-.5,0,.25]){
   const candidate=[...actual.conductionCurrent!] as StaticVector;candidate[2]+=q;
   close(candidate.reduce((sum,j,i)=>sum+j*actual.electricField![i],0),r.power);
   // Added current is a supplementary incompatible hypothesis, not a native control.
   const changedAmpere=actual.ampereResidual![2]-actual.permeability*q;
   close(changedAmpere,-q);if(q!==0){assert.ok(changedAmpere!==0);failures++;}hypotheses++;
  }
 }
 assert.equal(hypotheses,36);assert.equal(failures,24);
});
test('native default-vacuum event energies agree with normalized references and local boost classification',()=>{
 const {pulses}=JSON.parse(execFileSync('python3',['scripts/learning/verify-poynting-research.py','--references'],{encoding:'utf8',timeout:20000})) as {pulses:{beta:number;z:number;shift:number;amplitude:number;f:number;g:number;u:number;s:number;rest:number|null}[]};
 assert.equal(pulses.length,48);const eps=teachingMaxwellPermittivity,c0=1/Math.sqrt(eps*teachingMaxwellPermeability);let finite=0,nullCases=0,zero=0;
 for(const r of pulses){
  const actual=transversePulseEnergy({amplitude:r.amplitude,width:1,rightFraction:r.beta},r.z,r.shift/c0);
  assert.notEqual(actual.energyDensity,null);assert.notEqual(actual.poyntingZ,null);
  // Normalize SI energies before comparison so an absolute tolerance cannot hide tiny values.
  close(actual.energyDensity!/eps,r.u);close(actual.poyntingZ!/(eps*c0),r.s);
  close(actual.rightElectric!,r.f);close(actual.leftElectric!,r.g);
  if(r.amplitude===0){close(actual.energyDensity!,0);close(actual.poyntingZ!,0);assert.equal(actual.observedEnergyVelocity,null);zero++;continue;}
  close(actual.observedEnergyVelocity!/c0,r.s/r.u);
  if(r.rest===null){close(Math.abs(r.s/r.u),1);nullCases++;continue;}
  const alpha=r.rest;assert.ok(Math.abs(alpha)<1);const gamma=1/Math.sqrt(1-alpha*alpha);
  const Ep=gamma*(actual.electricValue!-alpha*c0*actual.magneticValue!);
  const Dp=gamma*(c0*actual.magneticValue!-alpha*actual.electricValue!);
  close(Ep*Dp,0);close((Ep*Ep+Dp*Dp)/2,2*Math.abs(r.f*r.g));finite++;
 }
 assert.deepEqual({finite,nullCases,zero},{finite:27,nullCases:18,zero:3});
});
