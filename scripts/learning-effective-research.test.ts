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
import {openTransmissionLineState,dielectricBilayerState,dielectricBilayerField} from '../src/lib/learning-electromagnetic-effective-models';
import type {EffectiveComplex} from '../src/lib/learning-electromagnetic-effective-models';

const units=[
  {
    "id": "transmission-line-lumped-limit-energy",
    "columns": 5,
    "equations": 10,
    "image": "positive-modal-line-reduction",
    "original": [
      "6a696ee417b6f4cce36583c5a7e63798f27a32a1ada5e7182c46a75e04890d54",
      "4c78346dde1dbc759ef2c02b4f31a7abc8154da705b548bf3232cde18593211b"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "0.55740772",
        "0.68147693",
        "2.42551882",
        "2.97325241"
      ],
      [
        "1.00000000",
        "0.00502333",
        "0.00893262",
        "0.01549119",
        "0.02764029"
      ],
      [
        "2.00000000",
        "0.00077639",
        "0.00163743",
        "0.00234983",
        "0.00496624"
      ],
      [
        "4.00000000",
        "0.00010414",
        "0.00025346",
        "0.00031319",
        "0.00076292"
      ]
    ]
  },
  {
    "id": "layered-medium-bloch-effective-limit",
    "columns": 5,
    "equations": 10,
    "image": "bilayer-origin-boundary-phase",
    "original": [
      "42e14563b072317e4ac8285807fe4c5345c622a15caa4ee80a6df8ad6a983466",
      "52e49bcba890af2c4f6352a0d121e677a8094aa5f22aa125583a91d7a6aabc74"
    ],
    "outline": 18,
    "rows": [
      [
        "0.05000000",
        "0.99250891",
        "0.00330248",
        "0.00000000",
        "0.00000235"
      ],
      [
        "0.10000000",
        "0.97014233",
        "0.01285954",
        "0.00000000",
        "0.00003400"
      ],
      [
        "0.20000000",
        "0.88226449",
        "0.04678289",
        "0.00000000",
        "0.00038409"
      ],
      [
        "0.40000000",
        "0.55542921",
        "0.14411470",
        "0.00000000",
        "0.00244776"
      ]
    ]
  }
];
for(const unit of units)test(`Effective-model research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent positive LC realization, tail-energy bounds and boundary-order/logarithm calculations',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-effective-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"bchRemainderComparisons": 12, "bulkTracePreservation": 80, "cellSimilarityChecks": 80, "complexReflectionDifferences": 240, "energyTailBounds": 108, "homogeneousOrderControls": 132, "independentBoundaryPowerBudgets": 240, "independentBranchEnergySums": 108, "independentBranchImpedanceSolves": 360, "independentLayerExponentials": 80, "invalidModalComparisonRejections": 5, "logDomainRejections": 1, "logSeriesTailBounds": 12, "nonzeroReflectionPhaseContrasts": 54, "originDependentCommutatorChecks": 12, "physicalTerminationPowerContrasts": 9, "portTailBounds": 108, "positiveFiniteRealizations": 108, "reversedDiagonalSwaps": 240, "reversedStackPowerInvariance": 240, "staticVersusCycleLimitCases": 3, "symmetricCellSimilarityChecks": 3, "thirdOrderRefinements": 9});
});
test('native SI line response and energy remain above the passive two-mode approximation with separate normalized bounds',()=>{
 const {lines}=JSON.parse(execFileSync('python3',['scripts/learning/verify-effective-research.py','--references'],{encoding:'utf8',timeout:20000})) as {lines:{length:number;voltage:number;theta:number;capacitance:number;travel:number;exactSusceptance:number;exactSlopeRatio:number;approximateSusceptance:number;approximateSlope:number;portBound:number;slopeBound:number}[]};
 assert.equal(lines.length,36);let zeroAmplitude=0,negativeFits=0;
 for(const r of lines){
  const q=openTransmissionLineState({inductancePerLength:250e-9,capacitancePerLength:100e-12,length:r.length,inputVoltage:r.voltage},{kind:'finite-phase',theta:r.theta},.4);
  close(q.totalCapacitance/r.capacitance,1);close(q.inputSusceptance!*r.travel/r.capacitance,r.exactSusceptance*r.travel/r.capacitance);
  close(q.susceptanceDerivative!/r.capacitance,r.exactSlopeRatio);
  const portError=(q.inputSusceptance!-r.approximateSusceptance)*r.travel/r.capacitance;
  const slopeError=(q.susceptanceDerivative!-r.approximateSlope)/r.capacitance;
  assert.ok(portError>=-1e-12&&portError<=r.portBound*r.travel/r.capacitance+1e-12);
  assert.ok(slopeError>=-1e-12&&slopeError<=r.slopeBound/r.capacitance+1e-12);
  if(r.voltage===0){close(q.cycleEnergy,0);zeroAmplitude++;}
  else close(q.cycleEnergy!/(r.capacitance*r.voltage*r.voltage/4),r.exactSlopeRatio);
  if(r.theta===2){assert.ok(q.fittedCapacitance!<0);negativeFits++;}
 }
 assert.deepEqual({zeroAmplitude,negativeFits},{zeroAmplitude:12,negativeFits:9});
});
const complexClose=(a:EffectiveComplex|null,b:readonly number[])=>{assert.ok(a);close(a[0],b[0]);close(a[1],b[1]);};
test('native reversed stacks preserve power and common Bloch trace while retaining complex reflection contrast and internal flux',()=>{
 const {stacks}=JSON.parse(execFileSync('python3',['scripts/learning/verify-effective-research.py','--references'],{encoding:'utf8',timeout:20000})) as {stacks:{e1:number;e2:number;f:number;frequency:number;periods:number;halfTrace:number;r:number[];t:number[];points:{x:number;electric:number[];magnetic:number[]}[]}[]};
 assert.equal(stacks.length,16);let fieldCalls=0;const actuals=[];
 for(const r of stacks){
  const source={epsilon1:r.e1,epsilon2:r.e2,fraction:r.f,period:1};
  const q=dielectricBilayerState(source,r.frequency,r.periods);actuals.push(q);
  close(q.halfTrace,r.halfTrace);complexClose(q.scattering.rLeft,r.r);complexClose(q.scattering.tForward,r.t);close(q.reflectance!+q.transmittance!,1);
  for(const p of r.points){
   const field=dielectricBilayerField(source,r.frequency,r.periods,p.x);fieldCalls++;
   complexClose(field.electric,p.electric);complexClose(field.normalizedMagnetic,p.magnetic);close(field.normalizedFlux,q.transmittance!);
  }
 }
 for(let i=0;i<actuals.length;i+=2){
  const a=actuals[i],b=actuals[i+1];close(a.halfTrace,b.halfTrace!);close(a.reflectance,b.reflectance!);close(a.transmittance,b.transmittance!);
  complexClose(a.scattering.tForward,b.scattering.tForward!);
  assert.ok(Math.hypot(a.scattering.rLeft![0]-b.scattering.rLeft![0],a.scattering.rLeft![1]-b.scattering.rLeft![1])>1e-6);
 }
 assert.equal(fieldCalls,48);
});
test('native exact DC and symbolic poles preserve different static, average and uniqueness meanings',()=>{
 const source={inductancePerLength:250e-9,capacitancePerLength:100e-12,length:1,inputVoltage:1};
 const dc=openTransmissionLineState(source,{kind:'finite-phase',theta:0},.4);close(dc.staticEnergy!/dc.totalCapacitance,.5);close(dc.cycleEnergyLimitAtZero!/dc.totalCapacitance,.25);assert.equal(dc.cycleEnergy,null);
 const driven=openTransmissionLineState(source,{kind:'ideal-quarter-wave-pole',order:0},.4),free=openTransmissionLineState({...source,inputVoltage:0},{kind:'ideal-quarter-wave-pole',order:0},.4);
 assert.equal(driven.boundaryStatus,'incompatible-nonzero-drive');assert.equal(free.boundaryStatus,'nonunique-homogeneous-resonance');assert.equal(driven.cycleEnergy,null);assert.equal(free.cycleEnergy,null);
});
test('native homogeneous and absent-layer controls remove the order contrast without erasing vacuum-port reflection',()=>{
 let count=0;
 for(const source of [{epsilon1:4,epsilon2:4,fraction:.25,period:1},{epsilon1:2,epsilon2:8,fraction:0,period:1},{epsilon1:2,epsilon2:8,fraction:1,period:1}])for(const frequency of [.2,1]){
  const q=dielectricBilayerState(source,frequency,2);count++;
  assert.equal(q.structurallyHomogeneous,true);close(q.leadingLongWaveTraceContrast,0);close(q.matrix.a,q.matrix.d!);close(q.reflectance!+q.transmittance!,1);assert.ok(q.reflectance!>0);
 }
 assert.equal(count,6);
});
