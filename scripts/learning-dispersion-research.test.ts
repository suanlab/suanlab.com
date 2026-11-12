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
import { gaussianDispersivePacket, latticeDispersivePacket, periodicPacketCenter } from '../src/lib/learning-dispersion';
import { periodicWaveChain } from '../src/lib/learning-wave-equation';

const units=[
  {
    "id": "dispersive-wave-packets-phase-group-velocity",
    "columns": 4,
    "equations": 7,
    "image": "cubic-remainder",
    "original": [
      "63eb94a452ce2a6559480d8aa9a931ff6b8507ef001b9f4b4ea146dd239c1ebe",
      "5e1a1db508044bbb9e91e35c47dd053f5d157f0e1d9b3b7df7549042625b466d"
    ],
    "rows": [
      [
        0.25,
        0.0567041208,
        0.0570544331,
        0.0625
      ],
      [
        0.5,
        0.111459278,
        0.114108866,
        0.125
      ],
      [
        1,
        0.210457805,
        0.228217732,
        0.25
      ],
      [
        2,
        0.362955786,
        0.456435465,
        0.5
      ]
    ]
  },
  {
    "id": "lattice-wave-packet-transport-dispersion",
    "columns": 5,
    "equations": 8,
    "image": "energy-allocation",
    "original": [
      "0f650ea4cb7162fced696192744b6ec24cde5d51a16d277c8f73250b30a820d2",
      "6776a2d636c6e1203fe481e56d22598d0487f772e02310c5a00efe1d585b9779"
    ],
    "rows": [
      [
        0,
        1,
        0,
        0,
        0
      ],
      [
        0.392699,
        0.853553,
        0.270598,
        0,
        -0.270598
      ],
      [
        0.785398,
        0.5,
        0.382683,
        0,
        -0.382683
      ],
      [
        1.178097,
        0.146447,
        0.270598,
        0,
        -0.270598
      ]
    ]
  }
];
for(const unit of units)test(`dispersion research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-dispersion-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,tol=1e-11)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<tol*(1+Math.abs(b)),`${a} != ${b}`);
const sum=(a:number[])=>a.reduce((x,y)=>x+y,0);
test('independent cubic propagation, local continuity, circular moments and time averages',()=>{
  const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-dispersion-research.py','--json'],{encoding:'utf8',timeout:30000}));
  assert.deepEqual(actual,{allocationStates:512,bandTailBounds:9,circularMomentStates:512,illConditionedCenters:4,localBalanceSites:6144,pointwiseBounds:2564,realSpaceMomentIntegrals:4,spectralErrorStates:8,timeAverageBounds:12,travelingTransportStates:128});
});
test('native quadratic moments and traveling packet transport satisfy the research special cases',()=>{
  for(const sigma of [.5,1,2])for(const dispersion of [-1,0,1])for(const t of [-2,0,2]){
    const s={amplitude:.7,sigma,center:.3,carrierWaveNumber:3.2,carrierFrequency:5.3,groupVelocity:1.1,dispersion,phase:.2},actual=gaussianDispersivePacket(s,0,t);
    close(actual.intensityCenter!,s.center+s.groupVelocity*t);close(actual.intensityVariance,sigma*sigma/2+dispersion*dispersion*t*t/(2*sigma*sigma));close(actual.integratedNorm,.49*sigma*Math.sqrt(Math.PI));
  }
  for(const t of [0,.7,2,4]){
    const s={particles:32 as const,length:8,density:.8,tension:1.7,amplitude:.08,modeCenter:4,bandwidth:2,center:2.96,phase:.4},a=latticeDispersivePacket(s,t),n=s.particles;
    close(sum(a.linkEnergy),a.energy/2);
    for(const alpha of [0,.5,1]){
      const delta=alpha-.5,linkRate=a.position.map((q,j)=>a.stiffness*(a.position[(j+1)%n]-q)*(a.velocity[(j+1)%n]-a.velocity[j]));
      const local=a.kinetic.map((k,j)=>k+alpha*a.linkEnergy[(j+n-1)%n]+(1-alpha)*a.linkEnergy[j]);
      const currents=a.position.map((q,j)=>-a.stiffness*(a.position[(j+1)%n]-q)*((1-alpha)*a.velocity[(j+1)%n]+alpha*a.velocity[j]));
      close(sum(local),a.energy);close(a.spacing*sum(currents)/a.energy,a.energyTransportVelocity!);close(a.spacing*sum(currents)/a.energy,a.energyWeightedGroupVelocity!);
      currents.forEach((current,j)=>close(current-a.linkFlux[j],delta*linkRate[j]));
      const cos=a.x.map(x=>Math.cos(2*Math.PI*x/s.length)),sin=a.x.map(x=>Math.sin(2*Math.PI*x/s.length));
      const re=sum(local.map((e,j)=>e*cos[j]))/a.energy,im=sum(local.map((e,j)=>e*sin[j]))/a.energy;
      const re0=sum(a.assignedEnergy.map((e,j)=>e*cos[j]))/a.energy,im0=sum(a.assignedEnergy.map((e,j)=>e*sin[j]))/a.energy;
      const vr=sum(a.linkEnergy.map((e,j)=>e*cos[j]))/a.energy,vi=sum(a.linkEnergy.map((e,j)=>e*sin[j]))/a.energy,theta=2*Math.PI/n;
      close(re-re0,delta*((Math.cos(theta)-1)*vr-Math.sin(theta)*vi));close(im-im0,delta*(Math.sin(theta)*vr+(Math.cos(theta)-1)*vi));
      assert.ok(Math.hypot(re-re0,im-im0)<=Math.abs(delta)*Math.sin(Math.PI/n)+1e-12);
    }
  }
});
test('registered standing-wave preparation and weak-resultant centers expose the stated interpretation limits',()=>{
  const omega=2*Math.sin(Math.PI/8);
  for(const phase of [0,Math.PI/8,Math.PI/4,3*Math.PI/8]){
    const a=periodicWaveChain({particles:8,length:8,density:1,tension:1},[{mode:1,positionCos:.1,positionSin:0,velocityCos:0,velocitySin:0}],phase/omega);
    close(sum(a.linkEnergy)/a.energy,Math.cos(phase)**2);close(sum(a.linkFlux),0);
    for(const alpha of [0,.5,1]){
      const currents=a.position.map((q,j)=>-(a.position[(j+1)%8]-q)*((1-alpha)*a.velocity[(j+1)%8]+alpha*a.velocity[j]));
      close(sum(currents)/a.energy,-(alpha-.5)*omega*Math.sin(2*phase));
    }
  }
  for(const epsilon of [-.1,-1e-4,-1e-8,0,1e-8,1e-4,.1]){
    const a=periodicPacketCenter([1+epsilon,0,0,0,1-epsilon,0,0,0],[0,1,2,3,4,5,6,7],8);close(a.resultant,Math.abs(epsilon));assert.equal(a.undefinedThreshold,1e-10);
    if(epsilon===0)assert.equal(a.position,null);else{const expected=epsilon>0?0:4,d=Math.abs(a.position!-expected);assert.ok(Math.min(d,8-d)<1e-6);}
  }
});
