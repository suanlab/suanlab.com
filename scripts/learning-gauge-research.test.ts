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
import {timeGaugeState,lorenzInitialConstraintState,constantLorenzOffsetState} from '../src/lib/learning-gauge-freedom';
import type {TimeGauge,GaugeTransformMode} from '../src/lib/learning-gauge-freedom';
import type {StaticVector} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "time-dependent-gauge-field-invariance",
    "columns": 5,
    "equations": 9,
    "image": "spacetime-gauge-projection",
    "original": [
      "4edeab1c0273eaa1216317a9dfefc89e1f2012960febee9a74677d28fa570e1f",
      "ce517b670cb31dd187c6effe232225d948104f5e0a5745b4400192e502344014"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "8.00000000",
        "0.53333333",
        "-1.00000000",
        "0.00000000"
      ],
      [
        "1.00000000",
        "73.00000000",
        "0.53333333",
        "-1.00000000",
        "0.00000000"
      ],
      [
        "10.00000000",
        "7948.00000000",
        "0.53333333",
        "-1.00000000",
        "0.00000000"
      ],
      [
        "100.00000000",
        "808408.00000000",
        "0.53333333",
        "-1.00000000",
        "0.00000000"
      ]
    ]
  },
  {
    "id": "lorenz-gauge-source-initial-compatibility",
    "columns": 5,
    "equations": 9,
    "image": "lorenz-constraint-energy",
    "original": [
      "925b4202bec8166190f0ddb4edc59b5aa51314c110f9a01d9752164c7f30b719",
      "4f760dc3f6cd2ac165f0a497ccbf9a0d8047f658744d73afd9e0fc69532b4842"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "0.00000000",
        "0.00000000",
        "0.00000000",
        "0.00000000"
      ],
      [
        "0.25000000",
        "0.00031250",
        "0.02500000",
        "0.00175781",
        "0.03515625"
      ],
      [
        "0.50000000",
        "0.00125000",
        "0.05000000",
        "0.05625000",
        "0.56250000"
      ],
      [
        "1.00000000",
        "0.00500000",
        "0.10000000",
        "1.80000000",
        "9.00000000"
      ]
    ]
  }
];
for(const unit of units)test(`Gauge research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent exact graph quotient/topology and forced/boundary constraint balances',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-gauge-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"boundaryEnergyInfluxIdentities": 12, "constantMeanBlindCases": 12, "exactnessDimensionChecks": 9, "forcedModeEnergyIdentities": 60, "incidenceComplexes": 9, "incomingPotentialResidualChecks": 48, "invalidGeometryRejections": 4, "invalidWeightRejections": 2, "periodicWrappingObstructions": 27, "puncturedDomainObstructions": 1, "pureGaugeReconstructions": 36, "quotientOrbitChecks": 144, "restrictedBoundaryCounterexamples": 3, "singleFaceClosedForms": 12, "standingModeQuadratures": 36, "uniformSourceHistoryEqualities": 48, "weightedNormalAndCurlChecks": 36});
});
test('native time-dependent potentials give oriented space-time electric integrals with coordinated and partial controls',()=>{
 const {loops}=JSON.parse(execFileSync('python3',['scripts/learning/verify-gauge-research.py','--references'],{encoding:'utf8',timeout:20000})) as {loops:{alpha:number;g2:number;mode:GaugeTransformMode;baseCirculation:number;circulation:number}[]};
 assert.equal(loops.length,36);let calls=0,nonzeroPartial=0;
 for(const r of loops){
  const source:TimeGauge={magneticConstant:1,magneticRate:1,magneticAcceleration:1,gaugeAmplitude:r.alpha,gaugeConstant:1,gaugeRate:1,gaugeAcceleration:r.g2,mode:r.mode};
  const state=(x:number,t:number)=>{calls++;return timeGaugeState(source,[x,1,1],t).normalized;};
  const bottom=state(0,0),top=state(0,.5);
  const temporal=(x:number)=>{const values=[state(x,0),state(x,.25),state(x,.5)].map(p=>-p.scalarPotential!);return .5/6*(values[0]+4*values[1]+values[2]);};
  const right=temporal(.5),left=temporal(-.5),face=bottom.vectorPotential![0]+right-top.vectorPotential![0]-left;
  close(face,r.circulation);close(face,.5*(bottom.electric![0]+top.electric![0])*.5);
  if(r.mode==='coordinated'||r.mode==='original'||r.alpha===0)close(face,r.baseCirculation);
  else {assert.ok(Math.abs(face-r.baseCirculation)>0);nonzeroPartial++;}
 }
 assert.equal(calls,288);assert.equal(nonzeroPartial,12);
});
test('native standing constraint derivatives integrate to persistent energy and expose both Maxwell residuals',()=>{
 const {standing}=JSON.parse(execFileSync('python3',['scripts/learning/verify-gauge-research.py','--references'],{encoding:'utf8',timeout:20000})) as {standing:{amplitude:number;k:number;tau:number;energy:number}[]};
 assert.equal(standing.length,27);let calls=0,zeroEnergy=0;
 for(const r of standing){
  const n=32,h=2*Math.PI/n;let energy=0,mean=0;
  for(let i=0;i<n;i++){
   const z=-Math.PI+(i+.5)*h,q=lorenzInitialConstraintState({amplitude:r.amplitude,wavenumber:r.k},z,r.tau).normalized;calls++;
   close(q.gaussResidual!,-q.lorenzTimeDerivative!);close(q.ampereResidual!,q.lorenzGradient![2]);
   close(q.initialLorenz!,0);close(q.initialLorenzTimeDerivative!,r.amplitude*r.k*r.k*Math.cos(r.k*z));
   close(q.lorenz!,r.amplitude*r.k*Math.cos(r.k*z)*Math.sin(r.k*r.tau));
   energy+=.5*(q.lorenzTimeDerivative!**2+q.lorenzGradient![2]**2)*h;mean+=q.lorenz!/n;
  }
  close(energy,r.energy);close(mean,0);if(r.amplitude===0){close(energy,0);zeroEnergy++;}else assert.ok(energy>0);
 }
 assert.equal(calls,864);assert.equal(zeroEnergy,9);
});
test('native constant non-Lorenz offsets keep physical vacuum compatibility and zero derivative energy',()=>{
 let cases=0;
 for(const gamma of [-2,0,3])for(const tau of [-1,0,1]){
  const r=constantLorenzOffsetState({},gamma,tau);cases++;
  close(r.normalized.scalarPotential,gamma*tau);close(r.normalized.lorenz,gamma);
  assert.deepEqual(r.normalized.electric,[0,0,0]);assert.deepEqual(r.normalized.magnetic,[0,0,0]);
  close(r.normalized.gaussResidual,0);close(r.normalized.ampereResidual,0);
  assert.equal(r.lorenzFamilySatisfied,gamma===0);assert.equal(r.assignedSourceMaxwellFamilyCompatible,true);
 }
 assert.equal(cases,9);
});
