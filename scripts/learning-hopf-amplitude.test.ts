import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {hopfAmplitudeModel as run,isHopfAmplitudeParameters as valid,type HopfAmplitudeParameters} from '../src/lib/learning-hopf-amplitude';
import {directedPhaseRotation} from '../src/lib/learning-directed-phase-rotation';
import {observeRange,fraction} from '../src/lib/learning-entropy-production-math';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q,PartitionRational} from '../src/lib/learning-partition-exact-core';
const fixtures=JSON.parse(readFileSync('scripts/fixtures/hopf-amplitude-reference.json','utf8')) as {cases:{label:string;input:HopfAmplitudeParameters;expected:unknown}[];rotationReferences:{n:string;d:string;sin:string;cos:string}[]};
type F={n:bigint;d:bigint};const f=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
const bound=(q:PartitionRational):F=>{assert.equal(q.numerator.exponent,0);assert.equal(q.denominator.exponent,0);return f(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));};
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return f(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),power=Number(m[4]||0)-(m[3]||'').length;return power>=0?f(n*BigInt(10)**BigInt(power)):f(n,BigInt(10)**BigInt(-power));}
let certificates=0;
function compare(actual:unknown,expected:unknown,path:string,count=true){
 if(typeof expected==='string'){
 const q=actual as Q;assert(isMasterObservation(q),path);assert(q.bounds,path);const x=parse(expected),a=x.n<BigInt(0)?-x.n:x.n,scale=a>x.d?a:x.d,epsilonDen=BigInt(10)**BigInt(160),lo=f(x.n*epsilonDen-scale,x.d*epsilonDen),hi=f(x.n*epsilonDen+scale,x.d*epsilonDen);
 assert(le(bound(q.bounds.lower),hi)&&le(lo,bound(q.bounds.upper)),path);if(count)certificates++;return;
 }
 if(expected===null){assert.equal(actual,null,path);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,`${path}.${i}`,count));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,`${path}.${k}`,count);return;}
 assert.deepEqual(actual,expected,path);
}
const find=(label:string)=>fixtures.cases.find(c=>c.label===label)!.input;
for(const c of fixtures.cases)test(`independent220-digit Hopf flow: ${c.label}`,()=>{
 const result=run(c.input);compare(result,c.expected,c.label);assert.doesNotThrow(()=>JSON.stringify(result));
 for(const row of result.history){
  assert(row.sensitivityToInitialSquaredAmplitude.sign===1);assert(isMasterObservation(row.coordinateRadiusResidual));assert(row.coordinateRadiusResidual.bounds);assert(le(bound(row.coordinateRadiusResidual.bounds.lower),f(BigInt(0)))&&le(f(BigInt(0)),bound(row.coordinateRadiusResidual.bounds.upper)));
  if(row.logAmplitudeBalanceResidual){assert(row.logAmplitudeBalanceResidual.bounds);assert(le(bound(row.logAmplitudeBalanceResidual.bounds.lower),f(BigInt(0)))&&le(f(BigInt(0)),bound(row.logAmplitudeBalanceResidual.bounds.upper)));assert.equal(row.algebraicLogAmplitudeBalanceResidual!.resolution,'structural-zero');}
 }
});
test('criticality is exact parameter evidence, not a slow finite-time trajectory classification',()=>{
 const critical=run(find('critical')),below=run(find('near onset below')),above=run(find('near onset above'));
 assert(critical.classification.criticalParameterExactlyZero);assert(!below.classification.criticalParameterExactlyZero);assert(!above.classification.criticalParameterExactlyZero);
 assert.equal(critical.current.squaredAmplitude.value,1/9);assert.equal(critical.current.squaredAmplitude.resolution,'exact-rational');assert.equal(critical.criticalSquaredAmplitudeHalfTime!.value,.5);assert.equal(critical.criticalAmplitudeHalfTime!.value,1.5);
 assert.equal(run(find('critical squared amplitude half time')).current.squaredAmplitude.value,.5);assert.equal(run(find('critical amplitude half time')).current.amplitude.value,.5);
 assert.equal(below.originLinearAmplitudeRelaxationTime!.value,1e9);assert.equal(above.attractingCircleRadialRelaxationTime!.value,5e8);assert.equal(critical.originLinearAmplitudeRelaxationTime,null);assert.equal(critical.attractingCircleRadialRelaxationTime,null);
});
test('amplitude, squared-amplitude and cycle relaxation refer to different linear observables',()=>{
 const r=run(find('below onset')),circle=run(find('on periodic circle'));
 assert.equal(r.originLinearAmplitudeGrowthRate.value,-1);assert.equal(r.originLinearSquaredAmplitudeGrowthRate.value,-2);assert.equal(r.originLinearAmplitudeRelaxationTime!.value,1);
 assert.equal(circle.attractingCircleRadialDecayRate!.value,2);assert.equal(circle.attractingCircleRadialRelaxationTime!.value,.5);
});
test('an attracting invariant circle need not be a rotating periodic orbit',()=>{
 const fixed=run(find('circle of equilibria')),reverse=run(find('reverse rotation on circle')),degenerate=run(find('zero frequency at onset'));
 assert(fixed.classification.positiveMuCircleIsEquilibria);assert(!fixed.classification.positiveMuCircleIsPeriodicOrbit);assert.equal(fixed.periodDividedByTwoPi,null);assert.equal(fixed.current.phaseShift!.resolution,'structural-zero');assert.equal(fixed.current.x.value,1);
 assert(reverse.classification.positiveMuCircleIsPeriodicOrbit);assert.equal(reverse.attractingCircleAngularFrequency!.value,-1);assert.equal(reverse.periodDividedByTwoPi!.value,1);assert(!degenerate.classification.ordinaryHopfCrossingAtMuZero);
});
test('the unstable exact origin stays at the origin and has no phase without a perturbation',()=>{
 const r=run(find('origin above onset'));assert(r.classification.positiveMuCircleExists);assert(r.classification.originInitial);
 for(const h of r.history){assert.equal(h.squaredAmplitude.resolution,'structural-zero');assert.equal(h.x.resolution,'structural-zero');assert.equal(h.y.resolution,'structural-zero');assert.equal(h.phaseShift,null);assert.equal(h.instantaneousAngularFrequency,null);assert.equal(h.logAmplitudeBalanceResidual,null);}
 assert(r.current.sensitivityToInitialSquaredAmplitude.value!>1);
});
test('zero time and equal rational inputs preserve coordinates and the same analytic flow',()=>{
 const p=find('zero time'),r=run(p);assert.equal(r.current.x.value,1);assert.equal(r.current.y.value,0);assert.equal(r.current.phaseShift!.resolution,'structural-zero');assert.equal(r.current.sensitivityToInitialSquaredAmplitude.value,1);
 const a=run(find('below onset')),b=run(find('nonreduced equivalent values'));for(const key of ['history','initialSquaredAmplitude','regime']as const)assert.deepEqual(a[key],b[key]);
});
test('record density changes neither physical endpoint nor the initial preparation',()=>{
 const p=find('below onset'),before=structuredClone(p),a=run(p),b=run({...p,sampleSteps:64});assert(Object.isFrozen(a.parameters));assert.deepEqual(p,before);const{step:sa,...aa}=a.current,{step:sb,...bb}=b.current;assert.equal(sa,8);assert.equal(sb,64);assert.deepEqual(aa,bb);
});
test('extreme phase accumulation stays enclosed without wrapping or normalizing the Cartesian state',()=>{
 const r=run(find('weak saturation huge accumulated phase'));assert(r.current.phaseShift!.value! < -1e10);assert(r.current.amplitude.value!>10000);assert(r.current.coordinateRadiusResidual.value===null||Math.abs(r.current.coordinateRadiusResidual.value)<1e-40);
 const tiny=run(find('critical tiny initial weak saturation'));assert.equal(tiny.current.squaredAmplitude.sign,1);assert.equal(tiny.current.integratedSquaredAmplitude.sign,1);
});
test('owned integer and bounded physical-ratio contracts reject invalid data',()=>{
 const p=find('below onset');assert(valid(p));
 for(const patch of [{gNumerator:0},{muNumerator:9},{omegaNumerator:9},{xNumerator:5},{timeNumerator:5},{sampleSteps:0},{sampleSteps:65},{muDenominator:0},{xNumerator:NaN},{gNumerator:Infinity},{muNumerator:.5}]){assert(!valid({...p,...patch}));assert.throws(()=>run({...p,...patch}));}
 assert(!valid({...p,extra:1}));assert(!valid({...p,[Symbol('extra')]:1}));assert(!valid(Object.assign(Object.create({extra:1}),p)));assert(!valid({...p,muNumerator:'1'}));
});
test('independent reference inventory is explicit',()=>{assert.equal(fixtures.cases.length,35);assert.equal(certificates,3920);console.log(JSON.stringify({independentPreparations:fixtures.cases.length,scalarCertificates:certificates}));});

test('expanded phase rotation has independent boundary and quadrant references at256/512bits',()=>{
 assert.equal(fixtures.rotationReferences.length,7);
 for(const r of fixtures.rotationReferences)for(const bits of [256,512]){const v=directedPhaseRotation(BigInt(r.n),BigInt(r.d),bits);for(const k of ['sin','cos']as const)compare(observeRange(fraction(v[k].lower,v.scale),fraction(v[k].upper,v.scale)),r[k],`rotation.${r.n}/${r.d}.${k}.${bits}`,false);}
});
test('phase rotation rejects unsupported arguments before approximating',()=>{
 assert.throws(()=>directedPhaseRotation((BigInt(1)<<BigInt(48))+BigInt(1),BigInt(1)));assert.throws(()=>directedPhaseRotation(BigInt(1),BigInt(0)));assert.throws(()=>directedPhaseRotation(BigInt(1),BigInt(1),63));assert.throws(()=>directedPhaseRotation(BigInt(1),BigInt(1),513));
});
