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
import {pointChargeField,directedSurfaceCurrent} from '../src/lib/learning-charge-current-fields';

const units=[
  {
    "id": "charge-superposition-test-field",
    "columns": 4,
    "equations": 8,
    "image": "probe-backreaction",
    "original": [
      "02fc983e13391ae10e4f30ecdcb12b1f89f4410b847e3f37e9189875bd8ecad9",
      "6592c5301ce097255bc1a8f2df9b313e6119b06bc2ef4a8cf8a0f43d9ce12c62"
    ],
    "outline": 14,
    "rows": [
      [
        0.005,
        0.00505089421,
        1.01017884,
        0.989846929
      ],
      [
        0.025,
        0.0263726954,
        1.05490782,
        0.945825892
      ],
      [
        0.075,
        0.0907103305,
        1.20947107,
        0.800480895
      ],
      [
        0.14,
        0.246737798,
        1.76241284,
        0.344882041
      ]
    ]
  },
  {
    "id": "charge-continuity-surface-current",
    "columns": 4,
    "equations": 8,
    "image": "current-nullspace",
    "original": [
      "4bba5f9aa3247bbe4f6ee6bcc31b9898fc194b10f8c1b4c61268ee3c972c88f5",
      "c571c479c9b0c439b7320d263c094eee443ecfa4528c7320699defa9eed63e27"
    ],
    "outline": 16,
    "rows": [
      [
        -0.75,
        0.25,
        -0.75,
        0.75
      ],
      [
        0.0,
        1.0,
        0.0,
        3.0
      ],
      [
        0.5,
        1.5,
        0.5,
        7.0
      ],
      [
        1.0,
        2.0,
        1.0,
        13.0
      ]
    ]
  }
];
for(const unit of units)test(`Charge research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,unit.outline);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`electromagnetism-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent equilibrium, derivative, tolerance, exact graph rank and observation checks',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-charge-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"equilibriumBranches": 60, "potentialDerivativeChecks": 120, "localExtremumChecks": 60, "fieldToleranceChecks": 23, "smallProbeSequences": 1, "softTrapSequences": 3, "graphFamilies": 17, "allNodeSubsetBalances": 272, "augmentedRanks": 6, "noiseGains": 12, "regularizationIdentities": 17});
});
test('native fixed-geometry Coulomb fields agree with independent displaced-equilibrium references',()=>{
 const refs=JSON.parse(execFileSync('python3',['scripts/learning/verify-charge-research.py','--references'],{encoding:'utf8',timeout:20000})) as {probe:{s:number;eta:number;R:number;Q:number;q:number;spring:number;field:number}[]};
 assert.equal(refs.probe.length,15);
 for(const r of refs.probe){
  const prepared=pointChargeField([{position:[0,0],charge:r.Q}],[r.R,0],-r.q);
  const disturbed=pointChargeField([{position:[r.R*r.s,0],charge:r.Q}],[r.R,0],-r.q);
  assert.equal(prepared.status,'finite');assert.equal(disturbed.status,'finite');
  close(disturbed.field![0],r.field);close(disturbed.field![1],0);
  close(disturbed.force![0]/(-r.q),r.field);
  close(disturbed.field![0]/prepared.field![0],1/(1-r.s)**2);
  close(r.spring*r.R*r.s,-disturbed.force![0],1e-20,1e-11);
  const zeroProbe=pointChargeField([{position:[r.R*r.s,0],charge:r.Q}],[r.R,0],0);
  close(zeroProbe.field![0],disturbed.field![0]);assert.deepEqual(zeroProbe.force,[0,0]);
 }
});
test('native local surface readings distinguish loop currents that have identical accumulation',()=>{
 const {graph}=JSON.parse(execFileSync('python3',['scripts/learning/verify-charge-research.py','--references'],{encoding:'utf8',timeout:20000})) as {graph:{alpha:number;edges:number[];derivative:number[]}[]};
 assert.equal(graph.length,4);
 for(const row of graph){
  const readings=row.edges.map(j=>directedSurfaceCurrent([j/.5,0],.5,0).current!);
  assert.deepEqual(readings,row.edges);
  const b=readings.map((j,i)=>readings[(i+3)%4]-j);
  assert.deepEqual(b,row.derivative);
  close(readings[0]-1,row.alpha);close(readings[0]-readings[1],0);
  for(const j of row.edges)close(directedSurfaceCurrent([j/.5,0],.5,Math.PI).current!,-j);
 }
});
