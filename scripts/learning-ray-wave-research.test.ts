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
import {paraxialRayExperiment as rays,gaussianBeamExperiment as beam} from '../src/lib/learning-ray-wave-optics';

const units=[
  {
    "id": "paraxial-ray-matrices-thin-lens-imaging",
    "columns": 3,
    "equations": 8,
    "image": "image-uncertainty",
    "original": [
      "53d902d59c07672d52295c5cf3beebe7a549850b07aba9c8870d3cbf836f11ec",
      "b2f9d7ad48b768d454d6f62787e275978e2d7ad1eb1e4c6d76feedb34fc612ca"
    ],
    "rows": [
      [
        20,
        0.5,
        66.6666667
      ],
      [
        40,
        0.25,
        266.666667
      ],
      [
        80,
        0.125,
        666.666667
      ],
      [
        160,
        0.0625,
        1466.66667
      ]
    ]
  },
  {
    "id": "gaussian-beam-focusing-ray-wave-limit",
    "columns": 4,
    "equations": 8,
    "image": "aperture-moments",
    "original": [
      "cc51448b378f3e11d34c728f5fa48fafc35fa9714de0b5c1fa1d2f2c7b85b963",
      "356487c38a11d41ed787482df7d7c1590c5ee48d5d81ea020a4dfc8981199012"
    ],
    "rows": [
      [
        0.5,
        0.40824829,
        1.71428571,
        1.47720779
      ],
      [
        0.2,
        0.25819889,
        4.6875,
        2.25144398
      ],
      [
        0.1,
        0.182574186,
        9.67741935,
        3.1676337
      ],
      [
        0.05,
        0.129099445,
        19.6721311,
        4.47401468
      ]
    ]
  }
];
for(const unit of units)test(`ray/wave research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-ray-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-13,relative=2e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent change of variables, tail moments, sensor observations and regularized Fourier moments',()=>{
  const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-ray-wave-research.py','--json'],{encoding:'utf8',timeout:30000}));
  assert.deepEqual(actual,{apertureMoments:12,fixedSensorMoments:162,fourierDerivativeChecks:60,fourierFieldChecks:60,hardEdgeCutoffs:12,imageDensityChecks:108,localRemainderBounds:45,propagatedMomentBounds:4,spectralMomentBounds:4,spectralRefinements:4,tailProbabilities:72,truncatedImageMoments:72});
});
test('native ray snapshots reproduce finite sensor uncertainty across real,virtual and exact focal inputs',()=>{
  const a=.001,n=16;let snapshots=0,ensembles=0,singular=0;
  for(const f of [.05,.1,.2])for(const ratio of [.25,.5,.75])for(const z of [0,.1,.2])for(const theta of [.01,-.02]){
    const mu=ratio*a,heights:number[]=[],weights:number[]=[];
    for(let j=0;j<=n;j++){
      const d=mu-a+2*a*j/n,u=f+d,result=rays({objectDistance:u,screenDistance:z,focalLength:f,objectHeight:.001,centralAngle:theta,fanHalfAngle:0});
      const expected=(1-z/f)*.001+(u*(1-z/f)+z)*theta;
      close(result.chiefRay[0],expected);result.rays.forEach(r=>close(r.screenHeight,expected));close(result.screen.rmsSpread,0);
      if(u===f){assert.equal(result.imageDistance,null);assert.equal(result.imageKind,'at-infinity');singular++;}
      else{close(result.imageDistance!,f+f*f/(u-f));assert.equal(result.imageKind,u>f?'real':'virtual');}
      heights.push(result.chiefRay[0]);weights.push(j===0||j===n?1:j%2?4:2);snapshots++;
    }
    const norm=weights.reduce((x,y)=>x+y,0),mean=heights.reduce((x,y,j)=>x+y*weights[j],0)/norm;
    const variance=heights.reduce((x,y,j)=>x+(y-mean)**2*weights[j],0)/norm;
    close(mean,(1-z/f)*.001+((f+mu)*(1-z/f)+z)*theta);close(variance,theta*theta*(1-z/f)**2*a*a/3,1e-20);
    if(f===.1&&ratio===.5&&z===.2&&theta===.01){close(mean,-5e-6);close(Math.sqrt(variance),.01*.001/Math.sqrt(3));}
    ensembles++;
  }
  assert.equal(ensembles,54);assert.equal(snapshots,918);assert.ok(singular>0);
});
test('native finite Gaussian moments retain their smooth-field assumptions and physical units',()=>{
  let cases=0;
  for(const wavelength of [.3e-6,.5e-6,1e-6])for(const waistRadius of [20e-6,80e-6])for(const postDistance of [0,.01,.1]){
    const amplitude=.7,result=beam({wavelength,waistRadius,amplitude,preDistance:0,postDistance,focalLength:null}),k=2*Math.PI/wavelength;
    const position=waistRadius**2/4+(postDistance/k)**2/waistRadius**2;
    close(result.covariance.xx,position,1e-20);close(result.covariance.thetaTheta,1/(k*k*waistRadius**2),1e-20);close(result.covariance.xTheta,postDistance/(k*k*waistRadius**2),1e-20);
    close(result.output.width**2,4*position,1e-20);close(result.intensityIntegral,amplitude**2*Math.PI*waistRadius**2/2,1e-20);assert.ok(result.output.width>0&&Number.isFinite(result.output.width));cases++;
  }
  assert.equal(cases,18);
});
