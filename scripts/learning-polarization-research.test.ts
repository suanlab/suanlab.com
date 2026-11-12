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
import {jonesPolarizationObservation as jones,stokesPolarizationObservation as stokes,polarizationRetarder,coherencyFromStokes,polarizationCoherencyState,type PolarizationPopulation} from '../src/lib/learning-polarization';

const units=[
  {
    "id": "jones-polarization-waveplates-analyzers",
    "columns": 4,
    "equations": 7,
    "image": "spectral-delay",
    "original": [
      "7bf774a177503f12d6e0112b88f196ffc4349ad81f12162d8b36a7a5fe588ce7",
      "70a5147f651a4bcb281871630c00473458ea5e9d24f5859d8e08f77978300517"
    ],
    "rows": [
      [
        0.25,
        0.969233234,
        0.984616617,
        0.0153833828
      ],
      [
        0.5,
        0.882496903,
        0.941248451,
        0.0587515487
      ],
      [
        1,
        0.60653066,
        0.80326533,
        0.19673467
      ],
      [
        2,
        0.135335283,
        0.567667642,
        0.432332358
      ]
    ]
  },
  {
    "id": "stokes-coherency-polarization-tomography",
    "columns": 4,
    "equations": 8,
    "image": "instrument-conditioning",
    "original": [
      "6ce155602b2c1c2c16a3fe398070e57976b9eb9ba78e689715d93dfd43c85a50",
      "1d10a9691cd2810e357c9471de0a2bfea1f94688dca93269c410fb0d53e15d87"
    ],
    "rows": [
      [
        90,
        1.73205081,
        1.41421356,
        -0.00262557604
      ],
      [
        30,
        4.73205081,
        3.74165739,
        0.000377599179
      ],
      [
        10,
        14.0523715,
        11.4303871,
        0.00720776888
      ],
      [
        1,
        140.347206,
        114.58865,
        0.0967085729
      ]
    ]
  }
];
for(const unit of units)test(`polarization research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-polarization-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,tol=1e-11)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<tol*(1+Math.abs(b)),`${a} != ${b}`);
const optics={retarderAngle:0,retardance:0,analyzerAngle:0,order:'retarder-only' as const};
const wrap=(x:number)=>Math.atan2(Math.sin(x),Math.cos(x));
test('independent spectral averaging, instrument spectra, error bounds, exact noise covariance and calibration',()=>{
  const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-polarization-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(actual,{blindPairs:2,calibrationDerivatives:16,calibrationStates:32,finiteSpectrumBounds:18,fixedRetarderStates:54,frequencyCompensations:45,gaussianSpectra:24,instrumentSpectra:7,noiseCovariances:7,noiseEnumerations:448,positiveReadingCounterexamples:3,pureFrequencyStates:45,reconstructionBounds:336,spectralRefinements:4});
});
test('native resolved Jones fields and finite Stokes ensembles preserve spectral mixing and compensation order',()=>{
  const spectra=[{nodes:[-1,1],weights:[.5,.5]},{nodes:[-Math.sqrt(3),0,Math.sqrt(3)],weights:[1/6,2/3,1/6]},{nodes:[-Math.sqrt(2),1/Math.sqrt(2)],weights:[1/3,2/3]}];
  for(const {nodes,weights} of spectra)for(const s of [0,.2,1,Math.PI,2*Math.PI])for(const delta0 of [0,.7,Math.PI/2]){
    const population=nodes.map((x,i)=>({amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2,relativePhase:wrap(delta0+s*x),globalPhase:0,weight:weights[i]}));
    const a=stokes({...optics,population,retainedPolarization:1});let re=0,im=0;const averaged=[0,0,0,0];
    population.forEach((p,i)=>{
      re+=p.weight*Math.cos(delta0+s*nodes[i]);im+=p.weight*Math.sin(delta0+s*nodes[i]);
      const resolved=jones({...optics,amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2,relativePhase:0,globalPhase:0,retardance:p.relativePhase,observationPhase:0});
      close(resolved.output.degree!,1);resolved.output.stokes.forEach((x,j)=>averaged[j]+=p.weight*x);
      const inverse=polarizationRetarder(0,-p.relativePhase),e=resolved.output.jones;
      for(let row=0;row<2;row++){
        const out=inverse[row].reduce((z,v,k)=>[z[0]+v[0]*e[k][0]-v[1]*e[k][1],z[1]+v[0]*e[k][1]+v[1]*e[k][0]],[0,0]);
        close(out[0],Math.SQRT1_2);close(out[1],0);
      }
    });
    [1,0,re,im].forEach((x,i)=>close(a.output.stokes[i],x));averaged.forEach((x,i)=>close(x,a.output.stokes[i]));close(a.output.degree!,Math.hypot(re,im));close(a.output.intensity,1);
    close(a.measurements.values[4],(1+im)/2);close(a.measurements.values[5],(1-im)/2);
    const fixed=stokes({...optics,population,retainedPolarization:1,retarderAngle:.4,retardance:.9});close(fixed.output.degree!,a.output.degree!);close(fixed.output.intensity,1);
  }
});
test('native retarder-analysis order realizes the weak circular pair and rejects invalid reconstructed states',()=>{
  const h={amplitudeX:1,amplitudeY:0,relativePhase:0,globalPhase:0,weight:1},v={...h,amplitudeX:0,amplitudeY:1},d={...h,amplitudeX:Math.SQRT1_2,amplitudeY:Math.SQRT1_2};
  const populations:PolarizationPopulation[][]=[[h],[v],[d],[{...d,relativePhase:Math.PI}],[{...d,relativePhase:Math.PI/2}],[{...d,relativePhase:-Math.PI/2}],[{...h,weight:3},v]];
  for(const population of populations)for(const degrees of [1,10,30,90]){
    const delta=degrees*Math.PI/180,base=stokes({...optics,population,retainedPolarization:1}),input=base.input.stokes;
    const observed=[0,Math.PI/2].map(analyzerAngle=>stokes({...optics,population,retainedPolarization:1,retarderAngle:Math.PI/4,retardance:delta,analyzerAngle,order:'retarder-then-analyzer'}).output.intensity);
    const preference=Math.cos(delta)*input[1]+Math.sin(delta)*input[3];close(observed[0],(input[0]+preference)/2);close(observed[1],(input[0]-preference)/2);
    const y=[...base.measurements.values.slice(0,4),...observed],estimate=[y.reduce((a,b)=>a+b,0)/3,y[0]-y[1],y[2]-y[3],(y[4]-y[5]-Math.cos(delta)*(y[0]-y[1]))/Math.sin(delta)];
    estimate.forEach((x,i)=>close(x,input[i]));
  }
  for(const degrees of [.5,1,5]){
    const estimated:[number,number,number,number]=[1,0,0,.02/Math.sin(degrees*Math.PI/180)];
    if(degrees<5)assert.throws(()=>coherencyFromStokes(estimated),/outside the physical cone/);else close(polarizationCoherencyState(coherencyFromStokes(estimated)).degree!,estimated[3]);
  }
});
