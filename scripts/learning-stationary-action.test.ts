import test from 'node:test';
import assert from 'node:assert/strict';
import {freeParticleActionBoundaryGauge as free,harmonicActionStationarySaddle as harmonic} from '../src/lib/learning-stationary-action';
import refs from './fixtures/stationary-action-independent.json';
const close=(a:number,b:number,tol=4e-11)=>assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);
const rational=(label:string)=>{const [n,d='1']=label.split('/');return Number(n)/Number(d);};
const fp=[1,10,0,10,0,0,0,0,0,10,0,50,50];
const hp=[1,2,20,10,0,10,0,50,50];

test('free action agrees with exact independent Boole integration at 83 boundary/random cases',()=>{
  for(const r of refs.free){const x=free(r.p);for(const key of Object.keys(r.exact) as (keyof typeof r.exact)[])assert.equal(x[key].label,r.exact[key],`${key}: ${r.p}`);for(const key of Object.keys(r.query)as(keyof typeof r.query)[])assert.equal(x.query[key].label,r.query[key],`${key}: ${r.p}`);}
});
test('harmonic action agrees with independent 60-digit quadrature and differentiation at 44 cases',()=>{
  for(const r of refs.harmonic){const x=harmonic(r.p);for(const key of Object.keys(r.numeric)as(keyof typeof r.numeric)[])close(x[key].value,Number(r.numeric[key]));for(const key of Object.keys(r.query)as(keyof typeof r.query)[])close(x.query[key],Number(r.query[key]));}
});
test('fixed-endpoint free motion is the minimum and has quadratic variation cost',()=>{
  const x=free(fp);assert.equal(x.action.label,'1/2');assert.equal(x.firstVariation.label,'0');assert.equal(x.secondVariation.label,'1/3');assert.equal(x.variedAction.label,'13/24');assert.equal(x.stationaryForFixedEndpoints,true);assert.equal(x.residualMeanSquare.label,'0');assert.equal(x.directionFixesEndpoints,true);
});
test('total derivative changes action and canonical momentum, not Newton residual or kinetic energy',()=>{
  const p=[...fp];p[6]=-20;const x=free(p),base=free(fp);assert.equal(x.action.label,'-1/2');assert.equal(x.gaugeShift.label,'-1');assert.equal(x.query.kinetic.label,'1/2');assert.equal(x.query.canonicalMomentum.label,'0');assert.equal(x.query.mechanicalMomentum.label,'1');assert.equal(x.query.hamiltonian.label,'1/2');assert.equal(x.firstVariation.label,base.firstVariation.label);assert.equal(x.stationaryForFixedEndpoints,true);
});
test('relaxed endpoint directions detect a different variational boundary problem',()=>{
  const p=[...fp];p[8]=10;p[9]=0;assert.equal(free(p).firstVariation.label,'1');p[6]=-20;const x=free(p);assert.equal(x.firstVariation.label,'-1');assert.equal(x.stationaryForFixedEndpoints,true);assert.equal(x.directionFixesEndpoints,false);assert.equal(x.stationaryForBothFreeEndpoints,false);
});
test('zero directional derivative is not whole-path stationarity',()=>{
  const p=[...fp];p[4]=10;p[9]=0;p[10]=10;const x=free(p);assert.equal(x.firstVariationAlongDirectionZero,true);assert.equal(x.stationaryForFixedEndpoints,false);assert.ok(x.residualMeanSquare.value>0);
});
test('a local residual node is not a whole-path solution',()=>{
  const p=[...fp];p[5]=10;const x=free(p);assert.equal(x.query.eulerLagrangeResidual.label,'0');assert.ok(x.residualMeanSquare.value>0);assert.equal(x.stationaryForFixedEndpoints,false);
  const q=[...hp];q[2]=25;q[8]=0;const y=harmonic(q);assert.equal(y.query.eulerLagrangeResidual,0);assert.ok(y.residualRms>0);assert.equal(y.selectedPathStationary,false);
});
test('free-particle exact integration by parts and full cumulative action hold over the control range',()=>{
  for(let j=0;j<350;j++){const p=[1+j%4,5+j%36,-20+j%41,20-j%41,-20+(j*3)%41,-20+(j*7)%41,-30+j%61,-10+j%21,10-(j*3)%21,-10+(j*5)%21,-10+(j*7)%21,-100+j%201,j%101],x=free(p);assert.equal(x.integrationByPartsResidual.label,'0');assert.equal(x.end.cumulativeAction.label,x.action.label);assert.equal(x.query.hamiltonian.label,x.query.kinetic.label);assert.ok(x.excessAction.value>=0);assert.equal(x.residualMeanSquare.numerator==='0',x.stationaryForFixedEndpoints);for(const series of x.curves)assert.equal(series.length,101);}
});
test('harmonic exact horizon classification distinguishes minimum, flat, saddle',()=>{
  for(const [tc,nature]of [[10,'strict-minimum'],[20,'flat-minimum'],[25,'saddle']]as const){const p=[...hp];p[2]=tc;const x=harmonic(p);assert.equal(x.zeroReferenceNature,nature);assert.ok(x.hessianBB.value>0);assert.equal(x.selectedPathStationary,tc===20);assert.equal(x.d1,1600-(2*tc)**2);}
});
test('negative-action trial is not itself a stationary saddle',()=>{
  const p=[...hp];p[2]=25;const x=harmonic(p);assert.ok(x.action.value<0);assert.equal(x.zeroReferenceNature,'saddle');assert.equal(x.selectedPathStationary,false);assert.ok(x.residualRms>0);p[3]=0;assert.equal(harmonic(p).selectedPathStationary,true);
});
test('critical family has zero action and positive constant energy; differing IVPs',()=>{
  const x=harmonic(hp);assert.equal(x.action.coefficient.label,'0');assert.equal(x.selectedPathStationary,true);assert.equal(x.residualMeanSquare.coefficient.label,'0');assert.ok(x.start.energy>0);for(const [,e]of x.curves[7])close(e,x.start.energy);const p=[...hp];p[3]=20;const y=harmonic(p);close(y.start.velocity,2*x.start.velocity);assert.equal(y.selectedPathStationary,true);
});
test('third mode prevents stationarity at the first critical horizon',()=>{
  const p=[...hp];p[4]=10;const x=harmonic(p);assert.equal(x.selectedPathStationary,false);assert.ok(x.residualRms>0);assert.ok(x.action.value>0);
});
test('free frequency limit has positive action; mass scales k, action and residual at fixed frequency',()=>{
  const p=[...hp];p[1]=0;const x=harmonic(p);assert.equal(x.stiffness,0);assert.equal(x.zeroReferenceNature,'strict-minimum');assert.ok(x.action.value>0);p[1]=3;const a=harmonic(p);p[0]=4;const b=harmonic(p);close(b.stiffness,4*a.stiffness);close(b.action.value,4*a.action.value);close(b.residualRms,4*a.residualRms);
});
test('nonstationary oscillator trial can have time-varying mechanical energy',()=>{
  const p=[...hp];p[2]=10;const x=harmonic(p);assert.equal(x.selectedPathStationary,false);assert.ok(Math.abs(x.query.energyChangeFromInitial)>0.1);assert.ok(x.curves[7].some(([,e])=>Math.abs(e-x.start.energy)>0.1));
});
test('oscillator cumulative action, endpoint identities, and exact quadratic variation agree',()=>{
  for(let j=0;j<450;j++){const x=harmonic([1+j%4,j%5,5+j%21,-20+(j*3)%41,-20+(j*7)%41,-10+(j*5)%21,-10+(j*7)%21,-100+j%201,j%101]);assert.equal(x.start.position,0);assert.equal(x.end.position,0);close(x.fullIntegralRepresentationResidual,0);close(x.query.representationResidual,0);assert.ok(x.d3>0);assert.equal(x.residualMeanSquare.coefficient.numerator==='0',x.selectedPathStationary);close(x.variedAction.value,x.action.value+x.epsilon*x.firstVariation.value+x.epsilon*x.epsilon*x.secondVariation.value/2);for(let i=0;i<201;i++)close(x.variationCurves[0][i][1],x.variationCurves[2][i][1]);}
});
for(const [name,fn,base]of [['free',free,fp],['harmonic',harmonic,hp]]as const){
  test(`${name}: invalid, missing, sparse and non-finite control codes are rejected`,()=>{for(const p of [[],base.slice(1),[...base,0],base.map((x,i)=>i===0?NaN:x),base.map((x,i)=>i===1?Infinity:x),base.map((x,i)=>i===0?0:x),base.map((x,i)=>i===2?0.5:x)])assert.throws(()=>fn(p));const sparse=[...base];delete sparse[2];assert.throws(()=>fn(sparse));});
  test(`${name}: results are fresh, JSON-safe and finite`,()=>{const a=fn([...base]),b=fn([...base]);assert.deepEqual(a,b);a.curves[0][0][1]=123456;assert.notDeepEqual(a,b);const walk=(x:unknown):void=>{if(typeof x==='number')assert.ok(Number.isFinite(x));else if(x&&typeof x==='object')Object.values(x).forEach(walk);};walk(b);assert.doesNotThrow(()=>JSON.stringify(b));});
}
test('rational exposed coefficients retain their exact numeric meaning',()=>{for(const r of refs.free){const x=free(r.p);close(x.action.value,rational(x.action.label));}for(const r of refs.harmonic){const x=harmonic(r.p);close(x.action.value,rational(x.action.coefficient.label)*Math.PI**2);}});

import {readFile} from 'node:fs/promises';
import {getLearningContent} from '../src/lib/learning-content';
for(const id of ['free-particle-action-boundary-gauge','harmonic-action-stationary-saddle'])test(`${id}: bilingual theory renders strict math, preserves explained exercises and English locale`,async()=>{
  const content=await getLearningContent(id);
  for(const locale of ['ko','en']as const){const source=await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8');assert.ok(content.outline[locale].length>=10);assert.equal(new Set(content.outline[locale].map(h=>h.id)).size,content.outline[locale].length);assert.ok(content[locale].includes('katex'));assert.ok(!content[locale].includes('katex-error'));for(let n=1;n<=6;n++)assert.match(source,new RegExp(`^${n}\\. `,'m'));if(locale==='en'){assert.doesNotMatch(content.en,/[가-힣]/);assert.ok(source.split(/\s+/).length>1000);}else assert.ok((source.match(/[가-힣]/g)||[]).length>1000);}
});
test('every stated control rejects out-of-range integer codes and accepts both range boundaries',()=>{
  const cases:[(p:number[])=>unknown,number[],[number,number][]][]=[
    [free,fp,[[1,4],[5,40],[-20,20],[-20,20],[-20,20],[-20,20],[-30,30],[-10,10],[-10,10],[-10,10],[-10,10],[-100,100],[0,100]]],
    [harmonic,hp,[[1,4],[0,4],[5,25],[-20,20],[-20,20],[-10,10],[-10,10],[-100,100],[0,100]]]
  ];
  for(const [fn,base,ranges]of cases)ranges.forEach(([low,high],i)=>{for(const value of [low,high]){const p=[...base];p[i]=value;assert.doesNotThrow(()=>fn(p));}for(const value of [low-1,high+1]){const p=[...base];p[i]=value;assert.throws(()=>fn(p));}});
});

import {analyticalMechanicsLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
import {evaluateAnalyticalMechanics} from '../src/lib/learning-analytical-mechanics';
for(const id of ['free-particle-action-boundary-gauge','harmonic-action-stationary-saddle']as const)test(`${id}: registered examples agree with their stated physical cases`,()=>{
  const spec=analyticalMechanicsLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;
  assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);
  const rows=spec.presets.map(p=>evaluateAnalyticalMechanics(id,p.values));
  if(id==='free-particle-action-boundary-gauge'){
    assert.deepEqual(rows.map(r=>r.stats['action Js']),[.5,-.5,-.5]);assert.deepEqual(rows.map(r=>r.stats['first variation Js']),[0,0,-1]);assert.deepEqual(rows.map(r=>r.stats['1{direction fixes endpoints}']),[1,1,0]);for(const r of rows)assert.equal(r.stats['1{fixed-endpoint stationary}'],1);
  }else{
    assert.deepEqual(rows.map(r=>r.harmonicAction!.zeroReferenceNature),['strict-minimum','flat-minimum','saddle']);assert.deepEqual(rows.map(r=>r.stats['1{selected path stationary}']),[0,1,0]);close(rows[0].stats['action Js']as number,3*Math.PI**2/16);assert.equal(rows[1].stats['action Js'],0);close(rows[2].stats['action Js']as number,-9*Math.PI**2/160);
  }
  for(let j=0;j<3;j++)assert.equal(guide.experiments[j].preset,j);
});
