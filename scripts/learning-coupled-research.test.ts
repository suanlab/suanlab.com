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
import { coupledModeSpectrum, coupledModeState, coupledBeatingState } from '../src/lib/learning-coupled-modes';

const units = [
  { id:'mass-weighted-coupled-normal-modes', equations:7, columns:4, image:'mode-gap', original:['f32a291bf2873a75f2a2cf8dfb07c12e2c5915e3bd2395aa8108424f92ee4154','e42ad4cf7ae8939efb70eda389d3947400d59f3234c6643fd60d444e63d3b147'], rows:[[.25,.353553,.707107,.807759],[.125,.176777,.707107,.41891],[.0625,.088388,.707107,.211288],[.03125,.044194,.707107,.105871]] },
  { id:'coupled-mode-beating-energy-transfer', equations:8, columns:5, image:'nearby-model', original:['a92e39bd44c5a9a60149460d6e298b9f12cbcc5e9002f5f40b0cfa75688a45db','5e0e8c11d5c7b667504c00732a85271c09884712f620acf96803c217200641a2'], rows:[[.25,32,.808164,.55265,.015625],[.125,128,.891649,.608572,.00390625],[.0625,512,.941992,.642326,.000976562],[.03125,2048,.96992,.659496,.000244141]] },
];
for(const unit of units) test(`coupled research pairs derivation, executable tables and preserved originals: ${unit.id}`,async()=>{
  const pair=['en','ko'].map((lang,index)=>{
    const source=readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8');
    assert.doesNotMatch(source,/[\u0000-\u0008\u000b-\u001f]/);
    if(lang==='en')assert.doesNotMatch(source,/[가-힣]/);
    const start=source.search(/^## (Research extension:|연구 확장:)/m);assert.ok(start>=0);
    const headerEnd=source.indexOf('\n',start),next=source.slice(headerEnd).search(/^## /m);assert.ok(next>=0);
    const end=headerEnd+next,block=source.slice(start,end);
    assert.equal(createHash('sha256').update(source.slice(0,start)+source.slice(end)).digest('hex'),unit.original[index]);
    assert.equal((block.match(lang==='en'?/\*\*Answer:\*\*/g:/\*\*답:\*\*/g)||[]).length,3);
    const tree=unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(block);
    const math=tree.children.flatMap(n=>n.type==='math'?[n.value]:[]),code=tree.children.flatMap(n=>n.type==='code'&&n.lang==='python'?[n.value]:[]);
    assert.equal(math.length,unit.equations);assert.equal(code.length,1);
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);
    assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});
    assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pair[0],pair[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-coupled-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
test('independent physical spring, gap, finite-time, nearby-energy and long-horizon checks',()=>{
  const result=JSON.parse(execFileSync('python3',['scripts/learning/verify-coupled-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(result,{degenerateBasisStates:81,finiteTimeBounds:540,longHorizonStates:54,modalOdeStates:540,nearbyModelStates:360,physicalSpringFamilies:135,projectorGapPairs:135,rk4Refinements:15});
});
test('native grid resolves the two equal-spectrum families and their distinct high-mode projectors',()=>{
  const epsilon=1/16;
  for(const angle of [0,-Math.PI/4]){
    const s=angle===0?{mass1:1,mass2:1,stiffness1:1+epsilon,stiffness2:1-epsilon,coupling:0}:{mass1:1,mass2:1,stiffness1:1-epsilon,stiffness2:1-epsilon,coupling:epsilon};
    const spec=coupledModeSpectrum(s),co=Math.cos(angle),sn=Math.sin(angle),expected=[[co*co,co*sn],[co*sn,sn*sn]];
    assert.ok(Math.abs(spec.squaredFrequencies[0]-(1-epsilon))<1e-12);assert.ok(Math.abs(spec.squaredFrequencies[1]-(1+epsilon))<1e-12);
    for(let i=0;i<2;i++)for(let j=0;j<2;j++)assert.ok(Math.abs(spec.projectors[1][i][j]-expected[i][j])<1e-12);
    for(const time of [0,1,5,12]){
      const high=Math.sqrt(1+epsilon),low=Math.sqrt(1-epsilon);
      const actual=coupledModeState(s,{position:[1,0],velocity:[0,0]},time);
      assert.ok(Math.abs(actual.position[0]-co*co*Math.cos(high*time)-sn*sn*Math.cos(low*time))<1e-11);
      assert.ok(Math.abs(actual.position[1]-co*sn*(Math.cos(high*time)-Math.cos(low*time)))<1e-11);
    }
  }
});
test('native weak-coupling approximation has the proved physical-energy defect at supported queries',()=>{
  for(const epsilon of [.25,.125,.0625])for(const time of [1,2/epsilon**2]){
    const actual=coupledBeatingState({mass:1,stiffness:1,coupling:epsilon,amplitude:1},time),approx=actual.approximation!;
    const w=1+epsilon,q=[(Math.cos(time)+Math.cos(w*time))/2,(Math.cos(time)-Math.cos(w*time))/2],v=[-(Math.sin(time)+w*Math.sin(w*time))/2,-(Math.sin(time)-w*Math.sin(w*time))/2];
    for(let i=0;i<2;i++){assert.ok(Math.abs(approx.position[i]-q[i])<1e-10);assert.ok(Math.abs(approx.velocity[i]-v[i])<1e-10);}
    const energy=(v[0]**2+v[1]**2+q[0]**2+q[1]**2+epsilon*(q[0]-q[1])**2)/2;
    assert.ok(Math.abs(energy-(1+epsilon)/2-epsilon**2/4*Math.sin(w*time)**2)<1e-11);
  }
});
