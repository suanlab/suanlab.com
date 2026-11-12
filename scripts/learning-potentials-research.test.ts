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
import {quadraticPotentialPathWork,cylinderVectorPotential,cylinderPotentialRectangle,type PotentialMatrix} from '../src/lib/learning-potentials';
import type {StaticVector} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "electrostatic-potential-reference-path-work",
    "columns": 4,
    "equations": 8,
    "image": "weighted-potential-loops",
    "original": [
      "647a83f537004cdfbda7c9701f4ebc40a4b1606071db9e4dbe54c46c7b78b4cd",
      "a06f43d72702385856efe68b2591227af63c82593f30d67b414c78d9a78cbc18"
    ],
    "outline": 14,
    "rows": [
      [
        0.25,
        1.05555556,
        1.75555556,
        2.81111111
      ],
      [
        1.0,
        1.06666667,
        1.76666667,
        2.83333333
      ],
      [
        4.0,
        1.08333333,
        1.78333333,
        2.86666667
      ],
      [
        16.0,
        1.09444444,
        1.79444444,
        2.88888889
      ]
    ]
  },
  {
    "id": "magnetic-vector-potential-curl-gauge",
    "columns": 4,
    "equations": 8,
    "image": "periodic-flux-obstruction",
    "original": [
      "239a1ae9f38319881ca12bf4cd7c776f5c0df35dd7c4ed22c754f95662373f0f",
      "bb8b15cc86c807bcb12d81c3571d43004d629eb9c9976f65eba8e5831ace8633"
    ],
    "outline": 16,
    "rows": [
      [
        -1.0,
        -1.0,
        1.0,
        -0.5
      ],
      [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      [
        0.5,
        0.5,
        0.5,
        0.25
      ],
      [
        1.0,
        1.0,
        1.0,
        0.5
      ]
    ]
  }
];
for(const unit of units)test(`Potentials research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
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
test('independent exact weighted systems, covariance, periodic Fourier algebra and oriented flux',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-potentials-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"weightedNormalSystems": 36, "covarianceChecks": 6, "orientationChecks": 288, "gaugeShifts": 36, "correlationModels": 9, "fourierModes": 104, "fourierGaugeChecks": 104, "faceFluxChecks": 36, "cellMeanResiduals": 36, "finiteCurlChecks": 36, "boundaryJumps": 72, "constantPotentialLoops": 9});
});
test('native quadratic line integrals supply the exact oriented triangle before added measurement noise',()=>{
 const {paths}=JSON.parse(execFileSync('python3',['scripts/learning/verify-potentials-research.py','--references'],{encoding:'utf8',timeout:20000})) as {paths:{start:StaticVector;end:StaticVector;control:StaticVector;difference:number;reference:number;hessian:PotentialMatrix;linear:StaticVector}[]};
 assert.equal(paths.length,9);
 for(const r of paths){
  const result=quadraticPotentialPathWork({hessian:r.hessian,linear:r.linear,reference:r.reference},r.start,r.end,r.control,1);
  close(result.potentialDifference!,r.difference);
  close(result.straightFieldIntegral!,-r.difference);close(result.bezierFieldIntegral!,-r.difference);
  close(result.quasistaticExternalWork!,r.difference);
 }
});
test('native ordinary-space constant gauge demonstrates why its scalar is not periodic',()=>{
 const {constants}=JSON.parse(execFileSync('python3',['scripts/learning/verify-potentials-research.py','--references'],{encoding:'utf8',timeout:20000})) as {constants:{axial:number;L:number;point:StaticVector}[]};
 assert.equal(constants.length,3);
 for(const r of constants){
  const source={radius:1,current:0,gaugeAxialConstant:r.axial};
  const original=cylinderVectorPotential({radius:1,current:0},r.point);
  const transformed=cylinderVectorPotential(source,r.point);
  const shifted=cylinderVectorPotential(source,[r.point[0],r.point[1],r.point[2]+r.L]);
  assert.deepEqual(transformed.magneticField,original.magneticField);assert.deepEqual(transformed.magneticField,[0,0,0]);
  close(transformed.vectorPotential![2],r.axial);close(transformed.divergenceA,0);
  close(shifted.gaugeFunction!-transformed.gaugeFunction!,r.axial*r.L,1e-18);
  const loop=cylinderPotentialRectangle(source,.2,.5,-.4,.7,.3);close(loop.circulation!,0,1e-18);
 }
});
