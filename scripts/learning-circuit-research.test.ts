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
import {rcPlateState,finiteAreaCoilState} from '../src/lib/learning-circuit-sensors';

const units=[
  {
    "id": "rc-circuit-field-closure-approximation",
    "columns": 5,
    "equations": 10,
    "image": "relaxing-capacitor-energy",
    "original": [
      "bc24e801abaa4403b743c277ab53d0e732938b1f277796f16a8b39893c8391af",
      "82f9f9ea92b1c78ae7c4d4ce1aebbcb191825a1d85f51d891d897cd3d114c805"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "1.00000000",
        "0.00000000",
        "0.50000000",
        "1.00000000"
      ],
      [
        "0.25000000",
        "0.62727759",
        "0.17406835",
        "0.21188848",
        "0.20539862"
      ],
      [
        "1.00000000",
        "0.24142772",
        "0.27260894",
        "0.06630149",
        "0.00097227"
      ],
      [
        "4.00000000",
        "0.05999745",
        "0.09703209",
        "0.00650746",
        "0.00137157"
      ]
    ]
  },
  {
    "id": "finite-area-coil-sensor-observability",
    "columns": 5,
    "equations": 10,
    "image": "coil-aperture-blind-modes",
    "original": [
      "2e4e0891a0a907de2ff87bdbaf9a1406d260dd593caa857bc2f36af5f5a1c803",
      "6aa5d812c6564a9ca6f5e7a81440ad314c6753ed30550e8aac4dbaae660ec20c"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.03077641"
      ],
      [
        "1.00000000",
        "0.88010117",
        "0.87500000",
        "0.88010117",
        "0.91283738"
      ],
      [
        "2.00000000",
        "0.57672481",
        "0.50000000",
        "0.57672481",
        "0.61727031"
      ],
      [
        "4.00000000",
        "-0.03302166",
        "-1.00000000",
        "0.03302166",
        "0.14791433"
      ]
    ]
  }
];
for(const unit of units)test(`Circuit research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
const close=(a:number|null,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(a!==null&&Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent passive relaxation, initial preparation and Fourier aperture/observation references',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-circuit-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"admittancePassivityCases": 108, "certifiedRootBrackets": 1, "conditionalInverseNoiseBounds": 27, "exactHalfRadiusLowerBounds": 1, "exactInitialRedistributions": 108, "exactInstantaneousPowerBalances": 108, "fastLayerOuterComparisons": 12, "fractionSeriesTailChecks": 7, "idealNullRadiusDiversitySolves": 9, "independentDiskQuadratures": 9, "independentMovingLineIntegrals": 48, "independentStateIntegrations": 24, "integratedEnergyBudgets": 24, "invalidRelaxationRejections": 4, "kinematicBlindCases": 6, "modalRateProducts": 24, "positiveDistinctRateProofCases": 108, "rootBracketDiversityChecks": 2, "roundedRootIsNotExactZero": 1, "smallApertureErrorBounds": 5, "unpreparedInitialCounterexamples": 3});
});
test('native no-branch RC agrees with the ideal limit while finite relaxation converges and field defects remain separate',()=>{
 const {rc}=JSON.parse(execFileSync('python3',['scripts/learning/verify-circuit-research.py','--references'],{encoding:'utf8',timeout:20000})) as {rc:{R:number;vi:number;theta:number;voltage:number;relaxingVoltages:number[]}[]};
 assert.equal(rc.length,24);let staticCases=0,nonzeroFieldDefects=0;
 for(const r of rc){
  const q=rcPlateState({resistance:r.R,plateRadius:.5,gap:.25,supplyVoltage:1,initialVoltage:r.vi,permittivity:1,permeability:1},r.theta,.5);
  close(q.capacitance,Math.PI);close(q.voltage,r.voltage);close(q.current,(1-r.vi)/r.R*Math.exp(-r.theta));
  close(q.powerResidual,0);close(q.integratedEnergyResidual,0);close(q.field.displacementCurrent,q.current!);
  const errors=r.relaxingVoltages.map(v=>Math.abs(v-q.voltage!));assert.equal(errors.length,3);
  errors.forEach(e=>assert.ok(e<=.02*Math.abs(1-r.vi)+1e-11));
  assert.ok(errors[2]<=errors[1]+1e-11&&errors[1]<=errors[0]+1e-11);
  if(r.vi===1){assert.equal(q.reconstructedFieldFamilyExactlyMaxwell,true);close(q.field.faradayPhi,0);staticCases++;}
  else {assert.equal(q.reconstructedFieldFamilyExactlyMaxwell,false);assert.ok(q.field.faradayPhi!<0);nonzeroFieldDefects++;}
 }
 assert.deepEqual({staticCases,nonzeroFieldDefects},{staticCases:8,nonzeroFieldDefects:16});
});
test('native exact quadratic disk average supplies the Fourier small-aperture correction without becoming a Fourier-field control',()=>{
 const {coils}=JSON.parse(execFileSync('python3',['scripts/learning/verify-circuit-research.py','--references'],{encoding:'utf8',timeout:20000})) as {coils:{k:number;radius:number;b:number;phase:number;B0:number;B0t:number;g:number;gt:number;quadratic:number;transfer:number;bound:number}[]};
 assert.equal(coils.length,18);
 for(const r of coils){
  const q=finiteAreaCoilState({uniformField:r.B0,uniformRate:r.B0t,uniformAcceleration:0,curvature:r.g,curvatureRate:r.gt,curvatureAcceleration:0},{radius:r.radius,turns:3,center:[0,0],velocity:[0,0]},0);
  close(q.analyticDiskAverage,r.B0*r.quadratic);close(q.observedDiskAverage,q.analyticDiskAverage!);
  const voltageScale=-3*Math.PI*r.radius**2*r.B0t;
  close(q.transformerEmf,voltageScale*r.quadratic);close(q.emf,q.transformerEmf!);close(q.motionalEmf,0);
  close(q.finiteAreaCorrection,voltageScale*(-r.k*r.k*r.radius*r.radius/8));
  const normalized=q.analyticDiskAverage!/r.B0;
  assert.ok(r.transfer-normalized>=-1e-14&&r.transfer-normalized<=r.bound+1e-14);
 }
});
test('native uniform finite-area limit remains distinct from zero area and zero turns',()=>{
 const field={uniformField:1,uniformRate:1,uniformAcceleration:0,curvature:0,curvatureRate:0,curvatureAcceleration:0};
 const actual=finiteAreaCoilState(field,{radius:.5,turns:1,center:[0,0],velocity:[0,0]},0);
 close(actual.observedDiskAverage,1);close(actual.emf,-Math.PI/4);close(actual.normalizedEmfObservation,1);
 const zero=finiteAreaCoilState(field,{radius:0,turns:1,center:[0,0],velocity:[0,0]},0);
 assert.equal(zero.observedDiskAverage,null);assert.equal(zero.normalizedEmfObservation,null);close(zero.emf,0);close(zero.analyticDiskAverage,1);
 const noTurns=finiteAreaCoilState(field,{radius:.5,turns:0,center:[0,0],velocity:[0,0]},0);
 close(noTurns.observedDiskAverage,1);assert.equal(noTurns.normalizedEmfObservation,null);close(noTurns.emf,0);
});
