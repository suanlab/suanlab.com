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
import {coherentFourierImaging as coherent,incoherentFourierImaging as incoherent,fourierPointSpread as psf,fourierPupilOTF as otf,type FourierPupil} from '../src/lib/learning-fourier-optics';

const units=[
  {
    "id": "coherent-fourier-pupil-spatial-filtering",
    "columns": 2,
    "equations": 8,
    "image": "reference-probes",
    "original": [
      "1f90c556432fb6080989bcb2d00995c214f7835cb2978e64f031c67b6f8bce7f",
      "525b3735ebaab611972812fcf1a97065850562c4a36a708d9b487c8945313e39"
    ],
    "rows": [
      [
        0,
        1.1798862
      ],
      [
        1.57079633,
        1.63563127
      ],
      [
        3.14159265,
        0.600113796
      ],
      [
        4.71238898,
        0.144368731
      ]
    ]
  },
  {
    "id": "incoherent-psf-otf-image-formation",
    "columns": 4,
    "equations": 8,
    "image": "pupil-ambiguity",
    "original": [
      "48d409f61d721b820f64302dcac86934f499de15fb8a4261e93aa42733b8aa08",
      "1aa75e34c6ca153c23d43ab05b9c5a8e43e6a7d64f871c20cda7b86df50f0427"
    ],
    "rows": [
      [
        0.25,
        0.767543502,
        0.644551038,
        0.847352514
      ],
      [
        0.5,
        0.498747493,
        0.259357732,
        0.68163876
      ],
      [
        0.75,
        0.318028594,
        0.0718187793,
        0.537387406
      ],
      [
        1,
        0.227324357,
        0.0235200013,
        0.420735492
      ]
    ]
  }
];
for(const unit of units)test(`Fourier research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-fourier-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=1e-11+5e-10*Math.abs(b),`${a} != ${b}`);
const p:FourierPupil={wavelength:500e-9,focalLength1:.1,focalLength2:.1,referenceHalfWidth:.001,apertureFraction:1,obstructionFraction:0,amplitudeTransmission:1,quadraticPhase:2,linearPhase:0};
const multiply=(a:number[],b:number[])=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const phasor=(x:number)=>[Math.cos(x),Math.sin(x)];
test('independent phase-cycle noise, reference calibration, conjugate-reflected pupil and finite diversity bounds',()=>{
  const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-fourier-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(actual,{certifiedNonzeroDiversityCases:36,clearPupilIntegrals:72,diversitySeriesBounds:36,intensityReadings:288,noiseEnumerations:16,otfAmbiguities:60,phaseCalibrationStates:216,phaseCycles:72,pointSpreadAmbiguities:60,pupilConjugacyStates:60,ratioErrorBounds:432,readErrorBounds:576,referenceAmbiguities:3});
});
test('native finite harmonics recover phase-cycled cross terms and the same ratio under physical relay scaling',()=>{
  let systems=0,exposures=0;
  for(const quadraticPhase of [-2,2])for(const linearPhase of [-.3,.4])for(const focalLength2 of [.1,.2])for(const amplitudeTransmission of [.4,1])for(const obstructionFraction of [0,.2])for(const queryCoordinate of [0,12.5e-6]){
    const system={...p,quadraticPhase,linearPhase,focalLength2,amplitudeTransmission,obstructionFraction,period:200e-6,globalPhase:.3,queryCoordinate},r=obstructionFraction===0?0:1;
    const reference={order:r,amplitude:.8,phase:.1},signal={order:2,amplitude:.5,phase:.7};
    const ref=coherent({...system,modes:[reference]}).query,sig=coherent({...system,modes:[signal]}).query;
    const snapshots=[0,Math.PI/2,Math.PI,3*Math.PI/2].map(phi=>coherent({...system,modes:[{...reference,phase:reference.phase+phi},signal]}).query);
    const recover=(values:number[])=>[(values[0]-values[2])/4,(values[1]-values[3])/4];
    const cross=recover(snapshots.map(q=>q.outputIntensity)),actualCross=recover(snapshots.map(q=>q.actualOutputIntensity));
    const expected=multiply([ref.outputField[0],-ref.outputField[1]],sig.outputField);cross.forEach((x,j)=>close(x,expected[j]));
    const ratio=cross.map(x=>x/ref.outputIntensity),actualRatio=actualCross.map(x=>x/ref.actualOutputIntensity);ratio.forEach((x,j)=>close(x,actualRatio[j]));
    const knownInputFactor=phasor(reference.phase-signal.phase-2*Math.PI*(2-r)*queryCoordinate/system.period).map(x=>x*.8/.5);
    const transferRatio=multiply(knownInputFactor,ratio),tr=r/4,ts=.5,phase=quadraticPhase*(ts*ts-tr*tr)+linearPhase*(ts-tr);
    transferRatio.forEach((x,j)=>close(x,phasor(phase)[j]));systems++;exposures+=6;
  }
  assert.equal(systems,64);assert.equal(exposures,384);
  const system={...p,obstructionFraction:.2,period:200e-6,globalPhase:0,queryCoordinate:0},signal={order:2,amplitude:.5,phase:.7};
  const values=[0,Math.PI/2,Math.PI,3*Math.PI/2].map(phase=>coherent({...system,modes:[{order:0,amplitude:.8,phase},signal]}).query.outputIntensity);
  values.forEach(x=>close(x,values[0]));assert.equal(coherent({...system,modes:[{order:0,amplitude:.8,phase:0}]}).query.outputIntensity,0);
});
test('native PSFs and full incoherent images preserve exact ambiguity while calibrated diversity changes the transfer',()=>{
  let systems=0,psfPairs=0,otfPairs=0;
  for(const quadraticPhase of [.5,2])for(const linearPhase of [-.3,.4])for(const apertureFraction of [.6,1])for(const obstructionFraction of [0,.4]){
    const a={...p,quadraticPhase,linearPhase,apertureFraction,obstructionFraction,amplitudeTransmission:.7},b={...a,quadraticPhase:-quadraticPhase},fc=20000*apertureFraction;
    for(const x of [-1,.25,0,.5]){const left=psf(a,x/fc),right=psf(b,x/fc);close(left.field[0],right.field[0]);close(left.field[1],-right.field[1]);close(left.rawIntensity,right.rawIntensity);close(left.conditionalPSF!,right.conditionalPSF!);psfPairs++;}
    for(const t of [-1.5,-.25,0,.75,1.5]){const left=otf(a,t*fc),right=otf(b,t*fc);left.conditional!.forEach((x,j)=>close(x,right.conditional![j]));left.raw.forEach((x,j)=>close(x,right.raw[j]));otfPairs++;}
    systems++;
  }
  assert.equal(systems,16);assert.equal(psfPairs,64);assert.equal(otfPairs,80);
  for(const quadraticPhase of [.5,2])for(const obstructionFraction of [0,.4]){
    const system={...p,quadraticPhase,linearPhase:.3,obstructionFraction,amplitudeTransmission:.7,sources:[{position:0,weight:1},{position:25e-6,weight:.4}],queryCoordinate:10e-6,gratingMean:1,gratingModulation:.8,gratingFrequency:15000,gratingPhase:.3};
    const a=incoherent(system),b=incoherent({...system,quadraticPhase:-quadraticPhase});close(a.pointImageIntegral,b.pointImageIntegral);
    a.pointImageProfile.forEach((q,j)=>{close(q.pointIntensity,b.pointImageProfile[j].pointIntensity);close(q.actualPointIntensity,b.pointImageProfile[j].actualPointIntensity);});a.gratingProfile.forEach((q,j)=>close(q.gratingOutput,b.gratingProfile[j].gratingOutput));
  }
  const sinc=(x:number)=>x===0?1:Math.sin(x)/x;
  for(const q of [.5,2,4])for(const d of [-.5,1])for(const t of [.1,.25,.5,.75]){
    const plus=otf({...p,quadraticPhase:q+d},t*20000).conditional![0],minus=otf({...p,quadraticPhase:-q+d},t*20000).conditional![0],length=2-t,v=t*length;
    close(plus-minus,length/2*(sinc((q+d)*v)-sinc((d-q)*v)));
    const leading=-2*q*d*v*v/3,bound=((q+d)**4+(d-q)**4)*v**4/120;assert.ok(Math.abs((plus-minus)/(length/2)-leading)<=bound+1e-11);
    if(Math.abs(leading)>bound)assert.ok(Math.abs(plus-minus)>=length/2*(Math.abs(leading)-bound)-1e-11);
  }
  const base={...p,period:200e-6,modes:[{order:0,amplitude:1,phase:0},{order:2,amplitude:1,phase:0}],queryCoordinate:25e-6,globalPhase:0};
  close(coherent({...base,quadraticPhase:2}).query.outputIntensity-coherent({...base,quadraticPhase:-2}).query.outputIntensity,-4*Math.sin(.5));
});
