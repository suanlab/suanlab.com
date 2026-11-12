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
import {dielectricInterfaceState,normalInterfaceCoefficients,normalInterfaceState} from '../src/lib/learning-material-interfaces';
import {teachingMaxwellPermittivity} from '../src/lib/learning-maxwell';
import type {StaticVector} from '../src/lib/learning-electro-magnetostatics';

const units=[
  {
    "id": "dielectric-interface-free-bound-charge",
    "columns": 5,
    "equations": 9,
    "image": "dipole-layer-limit",
    "original": [
      "f7c6acc34e99fc1b9d8d29d3dfd7b48db8251067d052640668249786efefc37b",
      "d8ee7ffb0e8f9490cc096d17014325dde018ade39ec222f342b40cc0dd6b5c92"
    ],
    "outline": 17,
    "rows": [
      [
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000",
        "1.00000000"
      ],
      [
        "0.25000000",
        "4.00000000",
        "1.00000000",
        "4.00000000",
        "0.25000000"
      ],
      [
        "0.06250000",
        "16.00000000",
        "1.00000000",
        "16.00000000",
        "0.06250000"
      ],
      [
        "0.01562500",
        "64.00000000",
        "1.00000000",
        "64.00000000",
        "0.01562500"
      ]
    ]
  },
  {
    "id": "normal-incidence-impedance-interface",
    "columns": 5,
    "equations": 10,
    "image": "coherent-sheet-absorption",
    "original": [
      "c1e6df3c4fa36961b4ba6dca77e4bc7f4203f5fe3fb84bd4390ef07c8bafb41f",
      "cd15891223aec33b6f0a96e56406c4729c60a0b3516b984ccbdfbc7bcab0d53d"
    ],
    "outline": 18,
    "rows": [
      [
        "0.00000000",
        "0.00000000",
        "0.00000000",
        "1.00000000",
        "0.00000000"
      ],
      [
        "0.50000000",
        "0.25000000",
        "0.25000000",
        "0.50000000",
        "-0.00000000"
      ],
      [
        "1.00000000",
        "0.50000000",
        "0.50000000",
        "0.00000000",
        "0.00000000"
      ],
      [
        "1.50000000",
        "0.25000000",
        "0.25000000",
        "0.50000000",
        "-0.00000000"
      ]
    ]
  }
];
for(const unit of units)test(`Materials research preserves original bodies and pairs proofs and executable tables: ${unit.id}`,async()=>{
  const pairs=['en','ko'].map((lang,index)=>{
    const source=readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8');
    assert.doesNotMatch(source,/[\u0000-\u0008\u000b-\u001f]/);if(lang==='en')assert.doesNotMatch(source,/[가-힣]/);
    const start=source.search(/^## (Research extension:|연구 확장:)/m);assert.ok(start>=0);
    const headerEnd=source.indexOf('\n',start),next=source.slice(headerEnd).search(/^## /m);assert.ok(next>=0);
    const end=headerEnd+next,block=source.slice(start,end);
    assert.equal(createHash('sha256').update(source.slice(0,start)+source.slice(end)).digest('hex'),unit.original[index]);
    assert.equal((block.match(lang==='en'?/\*\*Answer:\*\*/g:/\*\*답:\*\*/g)||[]).length,3);
    const tree=unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(block);
    assert.doesNotMatch(JSON.stringify(tree),/"type":"emphasis"/,"Research symbols must not become Markdown emphasis");
    const math=tree.children.flatMap(n=>n.type==='math'?[n.value]:[]),code=tree.children.flatMap(n=>n.type==='code'&&n.lang==='python'?[n.value]:[]);
    assert.equal(math.length,unit.equations);assert.equal(code.length,1);
    const tables=tree.children.filter(n=>n.type==='table');assert.equal(tables.length,1);assert.equal(tables[0].children.length,5);assert.ok(tables[0].children.every(row=>row.children.length===unit.columns));
    assert.deepEqual(tables[0].children.slice(1).map(row=>row.children.map(cell=>cell.children.map(n=>'value' in n?n.value:'').join(''))),unit.rows);
    const stdout=execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});assert.deepEqual(stdout.trim().split('\n').map(row=>row.split(/\s+/)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content=await getLearningContent(unit.id);
  for(const lang of ['en','ko'] as const)assert.equal(content.outline[lang].length,unit.outline);
  for(const lang of ['en','ko'] as const){assert.ok(!content[lang].includes('katex-error'));assert.ok(content[lang].includes(`electromagnetism-${unit.image}.${lang}.svg`));assert.equal(content.outline[lang].filter(h=>/Research extension:|연구 확장:/.test(h.title)).length,1);}
});
const close=(a:number,b:number,absolute=1e-11,relative=5e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${a} != ${b}`);
test('independent finite-layer weak/energy limits and full passive/coherent sheet references',()=>{
 const actual=JSON.parse(execFileSync('python3',['scripts/learning/verify-materials-research.py','--json'],{encoding:'utf8',timeout:20000}));
 assert.deepEqual(actual,{"finiteLayerIdentities": 48, "polynomialWeakMoments": 336, "weakTaylorBounds": 48, "potentialGradientChecks": 48, "energyScalingLimits": 48, "shapeEnergyBounds": 4, "invalidLayerRejections": 4, "exactOneBeamBudgets": 54, "absorptionBoundChecks": 54, "complexBoundaryBudgets": 675, "independentCycleIntegrals": 135, "coherentPerfectInputs": 9, "darkInputs": 9, "powerNormalizedReciprocity": 135, "symmetricPhaseExamples": 4, "coherenceEnsembles": 4, "invalidSheetRejections": 3, "zeroInputCases": 3, "absorptionOperators": 135, "brightDarkEigenPairs": 135, "fluctuatingAmplitudeEnsembles": 1});
});
test('native regular interface retains potential continuity without introducing a dipole source or gauge control',()=>{
 const {static:references}=JSON.parse(execFileSync('python3',['scripts/learning/verify-materials-research.py','--references'],{encoding:'utf8',timeout:20000})) as {static:{factor:number;x:number;potential:number}[]};
 assert.equal(references.length,9);
 for(const r of references){
  const point:StaticVector=[r.x,0,0];const actual=dielectricInterfaceState({permittivity1:teachingMaxwellPermittivity,permittivity2:r.factor*teachingMaxwellPermittivity,normalDisplacement1:0,freeSurfaceCharge:0,tangentialElectric:.5},point);
  close(actual.interfacePotential1!,r.potential);close(actual.interfacePotential2!,r.potential);close(actual.stablePotentialJump!,0);
  close(actual.medium1.electricField![2],0);close(actual.medium2.electricField![2],0);
  // Common reference addition is outside the API and cannot create a dipole jump.
  for(const C of [-2,0,3])close((actual.interfacePotential2!+C)-(actual.interfacePotential1!+C),0);
 }
});
test('native Fresnel and forced-transparent branches agree with the zero-sheet limit and signed required source',()=>{
 const {waves}=JSON.parse(execFileSync('python3',['scripts/learning/verify-materials-research.py','--references'],{encoding:'utf8',timeout:20000})) as {waves:{eps2:number;mu2:number;y1:number;y2:number;r:number;t:number;impliedTransparentG:number}[]};
 assert.equal(waves.length,9);let passive=0,active=0,matched=0,evaluations=0;
 for(const r of waves){
  const source={permittivity1:1,permeability1:1,permittivity2:r.eps2,permeability2:r.mu2,amplitude:1,wavenumber1:1};
  const coefficients=normalInterfaceCoefficients(source),state=normalInterfaceState(source,0,.25);evaluations+=2;
  close(coefficients.reflection,r.r);close(coefficients.transmission,r.t);
  close(coefficients.reflectance!,r.r*r.r);close(coefficients.transmittance!,r.y2/r.y1*r.t*r.t);
  close(state.boundary.electricLeft!,r.t*Math.cos(.25));close(state.boundary.magneticIntensityLeft!,state.boundary.magneticIntensityRight!);close(state.boundary.requiredSheetCurrentX!,0);
  const forced={...source,reflectionMode:'forced-zero-reflection' as const};
  const fc=normalInterfaceCoefficients(forced),fs=normalInterfaceState(forced,0,.25);evaluations+=2;
  close(fc.requiredSheetCurrentAmplitude!,r.impliedTransparentG);
  close(fs.boundary.requiredSheetCurrentX!,r.impliedTransparentG*Math.cos(.25));
  close(fs.boundary.requiredSheetWork!,r.impliedTransparentG*Math.cos(.25)**2);
  if(r.impliedTransparentG>0)passive++;else if(r.impliedTransparentG<0)active++;else matched++;
 }
 assert.deepEqual({evaluations,passive,active,matched},{evaluations:36,passive:3,active:3,matched:3});
});
