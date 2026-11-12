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
import { coherentSuperpositionDetector, finiteApertureDiffraction } from '../src/lib/learning-interference-diffraction';

const units=[
  {id:'coherent-superposition-detector-averages',columns:4,image:'moment-variance',original:['7b000e4b5225776714465943ae47d07a894a7cd9153eba17ed6a3395b94a4b35','f92f213a04e7f5693d92d0a98203e825ee6b6b3d9def3e36faa97a8df6281b03'],rows:[[0,2,4,0],[.785398,2,2,2],[1.570796,2,0,4],[2.356194,2,2,2]]},
  {id:'finite-aperture-fresnel-fraunhofer',columns:5,image:'aperture-bounds',original:['0278900bc601da357e5a5136b426052d42d391bc1400e52960efab95b3c4b786','c82fba2078fe783fc38e7d61818223271e2bf6e95167c7184c58dc675dfc7840'],rows:[[.25,4,3.27165135,.730223230,2.38153573],[.0625,8,3.80034784,.201548222,.758367305],[.015625,16,3.94896996,.0516312734,.201485253],[.00390625,32,3.98717236,.0129864866,.0511439685]]},
];
for(const unit of units)test(`interference research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
    assert.equal(math.length,8);assert.equal(code.length,1);
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-interference-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
test('independent moment positivity, fluctuation statistics, model bounds and high-precision aperture paths',()=>{
  const result=JSON.parse(execFileSync('python3',['scripts/learning/verify-interference-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(result,{apertureBoundStates:108,centeredMomentCases:6,correlatedMeanCases:18,iidMeanEnumerations:18,independentSeriesPaths:4,invalidMomentPairs:4,momentPopulations:36,positiveMomentForms:108,varianceStates:576});
});
test('native detector averages preserve the same-mean/different-variance example including signed exposure factors',()=>{
  for(const windowDuration of [0,1.5])for(const psi of [0,Math.PI/4,Math.PI/2,3*Math.PI/4])for(const phases of [[0,Math.PI],[Math.PI/2,3*Math.PI/2]]){
    const system={amplitude1:1,amplitude2:1,omega1:2*Math.PI,omega2:4*Math.PI,phaseDifference:psi,timeCenter:0,windowDuration,population:phases.map(phase=>({phase,weight:1}))};
    const actual=coherentSuperpositionDetector(system);assert.ok(Math.abs(actual.meanIntensity-2)<1e-12);
    const values=phases.map(phase=>coherentSuperpositionDetector({...system,population:[{phase,weight:1}]}).meanIntensity),mean=(values[0]+values[1])/2;
    const variance=((values[0]-mean)**2+(values[1]-mean)**2)/2;
    const expected=(2*actual.windowFactor)**2*(phases[0]===0?Math.cos(psi)**2:Math.sin(psi)**2);assert.ok(Math.abs(variance-expected)<1e-12);
  }
});
test('native unmodified aperture fields satisfy both absolute integral bounds across the supported geometry',()=>{
  for(const fresnelNumber of [.05,.25,1])for(const separationRatio of [1,2,4])for(const eta of [0,1,-.5]){
    const actual=finiteApertureDiffraction({wavelength:500e-9,slitWidth:50e-6,fresnelNumber,separationRatio,amplitude1:1,amplitude2:.7,relativePhase:.3,slits:2},eta);
    const m0=1.7,m2=m0*(separationRatio**2/4+1/12),bound=Math.min(2*m0,Math.PI*fresnelNumber*m2),phase=Math.PI*fresnelNumber*separationRatio**2/4;
    assert.ok(actual.complexFieldDifference/Math.sqrt(fresnelNumber)<=bound+1e-10);
    const corrected=[Math.cos(phase)*actual.fraunhofer[0]-Math.sin(phase)*actual.fraunhofer[1],Math.sin(phase)*actual.fraunhofer[0]+Math.cos(phase)*actual.fraunhofer[1]];
    assert.ok(Math.hypot(actual.fresnel[0]-corrected[0],actual.fresnel[1]-corrected[1])/Math.sqrt(fresnelNumber)<=Math.PI*fresnelNumber*m0*separationRatio/4+1e-10);
    assert.ok(Math.abs(actual.intensityDifference)/fresnelNumber<=2*m0*bound+1e-10);
  }
});
