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
import { periodicWaveChain, wavePulsePoint } from '../src/lib/learning-wave-equation';

const units = [
  {id:'coupled-chain-continuum-wave-limit',equations:7,columns:4,image:'lattice-tail',original:['54d4a22c4aae1e502ca2d0c340559bc422815067824bcae6e0b85365cea346f0','c7776f6df273ca0e92b661814a3cb7ad2b398f737c06881f89d9063ae2d5a341'],rows:[[2,.0339957198,.0333333333,.0416666667],[3,.00120242897,.00119047619,.00138888889],[4,2.21795523e-5,2.20458554e-5,2.48015873e-5],[6,1.93269515e-9,1.92708526e-9,2.08767570e-9]]},
  {id:'wave-initial-data-reflection-energy-flux',equations:8,columns:5,image:'boundary-memory',original:['0d5b0eb41304f243385d5ce3d9b8a7b824077271e2253030c6e391aaa69d2d09','ed71355450678097db9e2087bf7eb57f0ae42d490fa9a0bbacfe4d4ecfcfb0c4'],rows:[[0,-.0710926056,.00252707929,.00252707929,0],[.5,-.0427660489,.000914467470,.000609644980,.000304822490],[1,-.0281144098,.000395210019,.000197605010,.000197605010],[2,-.0146481885,.000107284713,3.57615711e-5,7.15231423e-5]]},
];
for(const unit of units)test(`wave-front research preserves the originals and pairs derivation and executable tables: ${unit.id}`,async()=>{
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
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);
    assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});
    assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`waves-front-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
test('independent exact tail intervals, Fourier/ODE checks and passive boundary budgets',()=>{
  const result=JSON.parse(execFileSync('python3',['scripts/learning/verify-wave-front-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(result,{initialGreenSites:25,alternatingIntervals:47,boundaryRefinements:36,fourierCoefficients:52,latticeTailBounds:16,materialScalings:18,passiveFrequencyStates:192,periodicCoordinates:72,periodicRefinements:6,tailEnergySplits:36,transientBudgets:36});
});
test('native smooth periodic modes retain the Fourier symbol used in the distinct infinite-chain derivation',()=>{
  for(const particles of [8,16,32])for(const time of [0,1]){
    const actual=periodicWaveChain({density:1,tension:1,length:8,particles},[{mode:1,positionCos:.1,positionSin:0,velocityCos:0,velocitySin:0}],time);
    const a=8/particles,frequency=2/a*Math.sin(Math.PI/particles);
    assert.ok(Math.abs(actual.spectrum[0].frequency-frequency)<1e-12);
    for(let j=0;j<particles;j++)assert.ok(Math.abs(actual.position[j]-.1*Math.cos(2*Math.PI*j/particles)*Math.cos(frequency*time))<1e-12);
  }
});
test('native fixed/free endpoint limits match zero-storage reflection without adding a spring-damper control',()=>{
  for(const boundary of ['fixed','free'] as const)for(const time of [1,1.5,2,2.5,3,4]){
    const actual=wavePulsePoint({density:1,tension:1,amplitude:.1,velocityAmplitude:0,width:1,center:2,direction:'left',boundary},0,time);
    const z=time-2,f=Math.abs(z)<1?.1*(1-z*z)**3:0,fp=Math.abs(z)<1?-.6*z*(1-z*z)**2:0;
    assert.ok(Math.abs(actual.displacement-(boundary==='free'?2*f:0))<1e-12);
    if(boundary==='free')assert.ok(Math.abs(actual.velocity-2*fp)<1e-12);else assert.ok(Math.abs(actual.velocity)<1e-12);
    assert.ok(Math.abs(actual.flux)<1e-12);
  }
});
