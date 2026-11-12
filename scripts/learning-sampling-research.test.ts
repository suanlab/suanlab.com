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
import {pixelApertureSampling as pixel,photonCountingSensor as sensor} from '../src/lib/learning-sampling-sensors';

const units=[
  {
    "id": "pixel-aperture-sampling-aliasing",
    "columns": 4,
    "equations": 8,
    "image": "shifted-grids",
    "original": [
      "a7d2265443e15378153c0964390a9e6fd46bc76ce68f311f67eff380e4a06276",
      "7bd8e2d8051c7559fd58a44328e725d1990ed13d29bea08d8b40da63bdde32aa"
    ],
    "rows": [
      [
        0.05,
        12.7062047,
        0.333061166,
        0.148184947
      ],
      [
        0.1,
        6.31375151,
        0.282756336,
        0.16942725
      ],
      [
        0.25,
        2.41421356,
        0.128412171,
        0.13349023
      ],
      [
        0.5,
        1,
        0.107359211,
        -0.11243727
      ]
    ]
  },
  {
    "id": "photon-counting-exposure-sensor-noise",
    "columns": 4,
    "equations": 8,
    "image": "censored-information",
    "original": [
      "f31ff474ba507c4d86d01673afb602fd4ccf4b6c8ac38bdae6312162ceb9c781",
      "bf97c79d148332fd0733450731472bb73860e96827bf947b8544a2762d259228"
    ],
    "rows": [
      [
        0.5,
        1.99165959,
        0.995829794,
        0.014387678
      ],
      [
        2,
        0.429594981,
        0.859189961,
        0.323323584
      ],
      [
        5,
        0.0667240463,
        0.333620232,
        0.875347981
      ],
      [
        10,
        0.00187110431,
        0.0187110431,
        0.997230604
      ]
    ]
  }
];
for(const unit of units)test(`Sampling research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,17);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-sampling-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
const mul=(a:number[],b:number[])=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const sub=(a:number[],b:number[])=>a.map((x,j)=>x-b[j]);
const div=(a:number[],b:number[])=>mul(a,[b[0],-b[1]]).map(x=>x/(b[0]**2+b[1]**2));
const phasor=(x:number)=>[Math.cos(x),Math.sin(x)];
test('independent two-grid integration, censored information, score identities and noisy-channel contraction',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-sampling-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{apertureZeroChecks:12,censoredInformationStates:18,codeDerivativeChecks:72,conditionalScoreChecks:72,ditherErrorBounds:108,pixelIntegralChecks:216,railLikelihoodSequences:3,readChannelContractions:18,realPixelDemodulations:72,tailDerivativeChecks:18,tailScoreIdentities:18,twoGridSystems:36});
});
test('native single-stripe integrals combine into nonnegative two-tone frames and recover both calibrated coefficients',()=>{
 const pitch=10e-6,irradiance=.001,c=[[.25,0],phasor(.7).map(x=>x*.4)];let systems=0,calls=0;
 for(const f of [.1,.3,.45])for(const width of [.25,.5,1])for(const shift of [.05,.1,.25,.5]){
  const b:number[][]=[],h:number[]=[];
  for(const origin of [0,shift*pitch]){
   const input={pitch,apertureWidth:width*pitch,pixelHeight:5e-6,wavelength:550e-9,irradiance,velocity:0,exposure:.01,centerTime:0,origin,radius:3,queryCoordinate:origin};
   const components=[0,1].map(l=>pixel({...input,frequency:(f+l)/pitch,modulation:l===0?.25:.4,phase:l===0?0:.7}));calls+=2;
   h[0]=components[0].spatialTransfer;h[1]=components[1].spatialTransfer;
   const y=components[0].samples.map((first,j)=>{
    const second=components[1].samples[j],average=first.averageIrradiance!+second.averageIrradiance!-irradiance;
    const energy=first.energy+second.energy-irradiance*input.apertureWidth*input.pixelHeight*input.exposure;
    assert.ok(average>0&&energy>0);close(energy/(input.apertureWidth*input.pixelHeight*input.exposure),average,1e-16);
    return {index:first.index,value:average/irradiance-1};
   });
   const y0=y.find(r=>r.index===0)!.value,y1=y.find(r=>r.index===1)!.value,omega=2*Math.PI*f;
   const beta=[y0,(Math.cos(omega)*y0-y1)/Math.sin(omega)];b.push(mul(beta,phasor(-2*Math.PI*f*origin/pitch)));
  }
  const w=phasor(2*Math.PI*shift),difference=sub(w,[1,0]);
  const a0=div(sub(mul(w,b[0]),b[1]),difference),a1=div(sub(b[1],b[0]),difference);
  [a0,a1].forEach((a,l)=>a.forEach((x,j)=>close(x/h[l],c[l][j])));systems++;
 }
 assert.equal(systems,36);assert.equal(calls,144);
 for(const origin of [0,1.3e-6,5e-6,9e-6]){
  const result=pixel({pitch,apertureWidth:8e-6,pixelHeight:5e-6,wavelength:550e-9,irradiance,modulation:.4,frequency:125000,phase:.7,velocity:0,exposure:.01,centerTime:0,origin,radius:3,queryCoordinate:origin});assert.equal(result.spatialTransfer,0);result.samples.forEach(r=>close(r.averageIrradiance!,irradiance,1e-16));
 }
});
test('native capped/quantized likelihoods match independent code probabilities, derivatives and information contraction',()=>{
 const references=JSON.parse(execFileSync('python3',['scripts/learning/verify-sampling-research.py','--references'],{encoding:'utf8',timeout:20000})) as {mean:number;readSigma:number;probabilities:number[];derivatives:number[];information:number;chargeInformation:number;storedVariance:number}[];assert.equal(references.length,12);let cases=0;
 for(const ref of references)for(const quantumEfficiency of [1,.5]){
  const input={incidentPhotonMean:ref.mean/quantumEfficiency,quantumEfficiency,exposure:.01,darkRate:0,readSigma:ref.readSigma,gain:1,offset:0,bits:2,fullWell:3},a=sensor(input),step=2e-5*Math.max(1,input.incidentPhotonMean);
  const plus=sensor({...input,incidentPhotonMean:input.incidentPhotonMean+step}),minus=sensor({...input,incidentPhotonMean:input.incidentPhotonMean-step});
  const slopes=a.bins.map((bin,j)=>{close(bin.probability,ref.probabilities[j]);const slope=(plus.bins[j].probability-minus.bins[j].probability)/(2*step);close(slope,quantumEfficiency*ref.derivatives[j],2e-10,2e-6);return slope;});
  const information=a.bins.reduce((sum,bin,j)=>sum+slopes[j]**2/bin.probability,0);close(information,quantumEfficiency**2*ref.information,1e-10,3e-6);assert.ok(information<=quantumEfficiency**2*ref.chargeInformation+1e-8);close(a.storedVariance,ref.storedVariance);
  if(ref.readSigma===0){const p3=Math.exp(-ref.mean)*ref.mean**3/6;close(a.upperRailOccupancy,ref.probabilities[3]);close(a.fullWellOverflowProbability,ref.probabilities[3]-p3);assert.equal(a.adcUpperClippingProbability,0);}
  cases++;
 }
 assert.equal(cases,24);
 const base={quantumEfficiency:0,exposure:.01,darkRate:20,readSigma:.5,gain:1,offset:0,bits:2,fullWell:3};const a=sensor({...base,incidentPhotonMean:2}),b=sensor({...base,incidentPhotonMean:10});assert.deepEqual(a.bins,b.bins);assert.ok(a.darkMean>0&&a.adcVariance>0);
});
