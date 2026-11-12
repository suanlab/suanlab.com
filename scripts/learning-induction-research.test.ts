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
import {inductionMovingLoop,transverseGaussianState} from '../src/lib/learning-induction-waves';

const units=[
  {
    "id": "faraday-induction-fixed-moving-loop",
    "columns": 4,
    "equations": 9,
    "image": "induction-mechanical-backreaction",
    "original": [
      "e9b397cefb97b0341dd534c5f27c79bc9f148ac6ddc05d64b290b38ad71fb1da",
      "9863cd9e1d3ab8d952efe6e995274e17c15b8027c77b1b0140b4ccf5cc538e85"
    ],
    "outline": 17,
    "rows": [
      [
        "0.00000000",
        "0.54030231",
        "-0.84147098",
        "1.00000000"
      ],
      [
        "0.50000000",
        "0.65970015",
        "-0.53350720",
        "0.71983422"
      ],
      [
        "1.00000000",
        "0.73575888",
        "-0.36787944",
        "0.67667642"
      ],
      [
        "2.00000000",
        "0.82226342",
        "-0.21390913",
        "0.72187425"
      ]
    ]
  },
  {
    "id": "electromagnetic-wave-equation-transverse-initial-data",
    "columns": 4,
    "equations": 9,
    "image": "electric-snapshot-observability",
    "original": [
      "220405bb70e8769747062c5b85e9bc3e23895147db6d3790062c6d26b350500c",
      "0d99033bb971d7b820a452088fb58eb204b66cc24ba853e6e5fba9e407e47f0b"
    ],
    "outline": 17,
    "rows": [
      [
        "0.00000000",
        "0.00000000",
        "unavailable",
        "unavailable"
      ],
      [
        "0.16666667",
        "0.50000000",
        "2.00000000",
        "0.78000000"
      ],
      [
        "0.50000000",
        "1.00000000",
        "1.00000000",
        "0.79000000"
      ],
      [
        "1.00000000",
        "0.00000000",
        "unavailable",
        "unavailable"
      ]
    ]
  }
];
for(const unit of units)test(`Induction research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent electromechanical power, damping, inverse ranks and noise references retain failure cases',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-induction-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"exactPowerBalances": 972, "variableInductanceBalances": 1944, "closedDampingTrajectories": 72, "independentRkTrajectories": 24, "dampingDerivativeChecks": 60, "integratedEnergyBalances": 48, "prescribedSpeedResponses": 288, "uncoupledCases": 27, "orientationCases": 12, "exactModeRotations": 24, "singleTimeInversions": 18, "blindInversionRejections": 6, "noiseCovariances": 48, "multiTimeFits": 180, "blindMultiTimeRejections": 12, "temporalModeSchedules": 56, "finiteBandBounds": 20, "spatialAliases": 9, "nearBlindAmplifications": 9});
});
test('native prescribed loop supplies the oriented instantaneous coupling before closing added circuit dynamics',()=>{
 const {loops}=JSON.parse(execFileSync('python3',['scripts/learning/verify-induction-research.py','--references'],{encoding:'utf8',timeout:20000})) as {loops:{B:number;height:number;u:number;emf:number;current:number;force:number}[]};
 assert.equal(loops.length,12);
 let evaluations=0;
 for(const r of loops){
  for(const orientation of [1,-1] as const){
   const actual=inductionMovingLoop({magneticOffset:r.B,magneticRate:0,magneticAcceleration:0,initialWidth:2,widthRate:r.u,height:r.height,orientation},0);evaluations++;
   close(actual.electricCirculation!,0);close(actual.motionalCirculation!,orientation*r.emf);close(actual.totalEmf!,orientation*r.emf);
   const orientedCurrent=orientation*r.current;
   // Added circuit force is an oriented wire cross product, not a native force control.
   close(orientedCurrent*orientation*r.height*r.B,r.force);
   close(actual.totalEmf!*orientedCurrent,-r.force*r.u);
  }
 }
 assert.equal(evaluations,24);
});
test('native Gaussian center is electric-only blind to direction fraction at all three observed times',()=>{
 const {pulses}=JSON.parse(execFileSync('python3',['scripts/learning/verify-induction-research.py','--references'],{encoding:'utf8',timeout:20000})) as {pulses:{beta:number;z:number;time:number;E:number;cB:number}[]};
 assert.equal(pulses.length,16);assert.equal(pulses.filter(r=>r.z===0).length,12);
 const observations=new Map<number,number[]>();
 for(const r of pulses){
  const actual=transverseGaussianState({amplitude:1,width:1,rightFraction:r.beta,permittivity:1,permeability:1},r.z,r.time);
  close(actual.electricValue!,r.E);close(actual.magneticValue!,r.cB);close(actual.stableCB!,r.cB);
  if(r.z===0){const group=observations.get(r.time)||[];group.push(actual.electricValue!);observations.set(r.time,group);}
 }
 for(const values of observations.values()){assert.equal(values.length,4);for(const value of values)close(value,values[0]);}
 const offCenter=pulses.filter(r=>r.z===1);assert.equal(offCenter.length,2);assert.ok(Math.abs(offCenter[0].E-offCenter[1].E)>.1);
});
