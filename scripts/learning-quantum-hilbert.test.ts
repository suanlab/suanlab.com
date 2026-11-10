import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,writeFileSync,mkdirSync}from'node:fs';
import {quantumStateModel as finite,isQuantumStateParameters,type QuantumStateParameters}from'../src/lib/learning-quantum-state';
import {wavefunctionL2Model as continuous,isWavefunctionL2Parameters,type WavefunctionL2Parameters}from'../src/lib/learning-wavefunction-l2';
import {isMasterObservation}from'../src/lib/learning-master-equation-protocol';import type {PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
const frefs=JSON.parse(readFileSync('scripts/fixtures/quantum-state-reference.json','utf8'))as{cases:{label:string;input:QuantumStateParameters;expected:unknown}[]},lrefs=JSON.parse(readFileSync('scripts/fixtures/wavefunction-l2-reference.json','utf8'))as{cases:{label:string;input:WavefunctionL2Parameters;expected:unknown}[]};
type R={n:bigint;d:bigint};const r=(n:bigint,d=BigInt(1)):R=>({n,d}),le=(a:R,b:R)=>a.n*b.d<=b.n*a.d,eq=(a:R,b:R)=>a.n*b.d===b.n*a.d;
function parse(s:string):R{if(s.includes('/')){const[n,d]=s.split('/');return r(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),k=Number(m[4]||0)-(m[3]||'').length;return k>=0?r(n*BigInt(10)**BigInt(k)):r(n,BigInt(10)**BigInt(-k));}
const bound=(x:any)=>r(BigInt(x.numerator.coefficient),BigInt(x.denominator.coefficient));let exactChecks=0,directedChecks=0;
function compare(actual:unknown,expected:unknown,path='root'){
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){const q=actual as Q;assert(isMasterObservation(q),path);assert(q.bounds,path);const a=bound(q.bounds.lower),b=bound(q.bounds.upper);assert(le(a,b),path);
  if(!expected.startsWith('decimal:')){const x=parse(expected);assert(eq(a,x)&&eq(b,x),path+' exact '+expected);exactChecks++;}
  else{const x=parse(expected.slice(8)),scale=x.n<BigInt(0)?-x.n:x.n,unit=scale>x.d?scale:x.d,eps=BigInt(10)**BigInt(150),lower=r(x.n*eps-unit,x.d*eps),upper=r(x.n*eps+unit,x.d*eps);assert(le(a,upper)&&le(lower,b),path+' interval '+expected);assert(le(r(b.n*a.d-a.n*b.d,b.d*a.d),r(unit,x.d*BigInt(10)**BigInt(100))),path+' width');directedChecks++;}return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,`${path}.${i}`));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,`${path}.${k}`);return;}assert.equal(actual,expected,path);
}
for(const c of frefs.cases)test(`symbolic complex-matrix reference: ${c.label}`,()=>{const x=finite(c.input);compare(x,c.expected,c.label);assert.doesNotThrow(()=>JSON.stringify(x));});
for(const c of lrefs.cases)test(`closed-binomial L2 reference: ${c.label}`,()=>{const x=continuous(c.input);compare(x,c.expected,c.label);assert.doesNotThrow(()=>JSON.stringify(x));});
const F=(label:string)=>finite(frefs.cases.find(c=>c.label===label)!.input),L=(label:string)=>continuous(lrefs.cases.find(c=>c.label===label)!.input);
test('global phase preserves a ray while relative phase changes complementary statistics',()=>{
 const a=F('equal Z statistics orthogonal rays');assert(a.classification.sameStandardProbabilities);assert(a.classification.orthogonalToReference);assert(!a.classification.sameRayAsReference);assert.deepEqual(a.standardProbabilities.map(q=>q.value),[.5,.5]);assert.deepEqual(a.basisProbabilities.map(q=>q.value),[1,0]);assert.deepEqual(a.referenceBasisProbabilities.map(q=>q.value),[0,1]);
 assert.equal(F('positive imaginary relative phase').plusYProbability.value,1);assert.equal(F('negative imaginary relative phase').plusYProbability.value,0);assert(F('same ray imaginary scaling').classification.sameRayAsReference);assert(!F('zero first component').classification.relativePhaseDefined);
 for(const c of frefs.cases){const r=finite(c.input);assert(r.residuals.globalBasisDifferences.every(q=>q.resolution==='structural-zero'));assert.equal(r.residuals.blochUnitNorm.resolution,'structural-zero');assert(r.residuals.coordinateReconstruction.every(q=>q.real.sign===0&&q.imaginary.sign===0));assert(r.residuals.basisGram.flat().every(q=>q.real.sign===0&&q.imaginary.sign===0));}
});
test('bra conjugation and Gram geometry are not ordinary probability-vector arithmetic',()=>{
 const x=F('complex reference overlap');assert.equal(x.overlap.real.value,x.reverseOverlap.real.value);assert.equal(x.overlap.imaginary.value,-x.reverseOverlap.imaginary.value!);
 const n=F('skew basis coefficient mass exceeds one');assert.equal(n.nonorthogonal.naiveCoefficientSquaredSum.value,2);assert.equal(n.nonorthogonal.gramNormSquared.value,1);assert.equal(n.nonorthogonal.normResidual.sign,0);
});
test('step projection has exact captured norm and zero projection stays unavailable for renormalization',()=>{
 for(const[label,error]of [['constant projection of step',1/2],['linear projection of step',1/8],['half interval step',9/128]]as const){const r=L(label);assert.equal(r.selectedSquaredL2Error.value,error);assert.equal(r.directSquaredL2Error.value,error);assert.equal(r.residuals.pythagoras.sign,0);}
 const zero=L('zero projection of sign-changing state');assert.equal(zero.selectedCapturedMass.value,0);assert.equal(zero.normalizedProjectionFidelity,null);assert(zero.probabilityBins.every(b=>b.renormalizedProjectedProbability===null));assert(L('global affine complex state').classification.projectionExactlyRepresentsState);
});
test('physical length and sample density change displays without changing exact integrated state probabilities',()=>{
 const p=lrefs.cases[0].input,a=continuous(p),b=continuous({...p,lengthNumerator:3,sampleSteps:64}),c=continuous({...p,bins:32});assert.deepEqual(a.coefficients,b.coefficients);assert.deepEqual(a.cutoffHistory,b.cutoffHistory);assert.deepEqual(a.cutoffHistory,c.cutoffHistory);assert.equal(b.points[0].density.value,a.points[0].density.value!/3);
 const tiny=L('small support missed by coarse samples');assert.equal(tiny.probabilityBins[0].targetProbability.value,1);assert.equal(tiny.points[1].density.value,0);assert.equal(tiny.points[0].density.value,1000000);
});
test('endpoint representatives can change without changing the L2 state or its integrated predictions',()=>{
 const p=lrefs.cases.find(c=>c.label==='split at one')!.input,a=continuous(p),b=continuous({...p,rightRealConstant:-55,rightImagConstant:17});assert.deepEqual(a.coefficients,b.coefficients);assert.deepEqual(a.probabilityBins,b.probabilityBins);assert.notDeepEqual(a.points.at(-1)?.wavefunction,b.points.at(-1)?.wavefunction);assert.equal(a.classification.pointValuesAreL2Invariants,false);
});
test('exact projection errors are nonnegative and decrease with degree; no residual is hidden by clipping',()=>{
 for(const item of lrefs.cases){const r=continuous(item.input);assert.equal(r.residuals.targetBinMass.sign,0);assert.equal(r.residuals.projectedMass.sign,0);assert(r.residuals.gram.flat().every(q=>q.sign===0));r.cutoffHistory.forEach((h,i)=>{assert(h.squaredL2Error.sign===0||h.squaredL2Error.sign===1);if(i)assert(le(bound(h.squaredL2Error.bounds!.upper),bound(r.cutoffHistory[i-1].squaredL2Error.bounds!.lower)));});}
});
test('complete owned inputs reject zero states and zero-norm functions rather than inventing a preparation',()=>{
 const a=frefs.cases[0].input,b=lrefs.cases[0].input;
 for(const patch of [{stateReal0:0,stateReal1:0},{basisReal0:0,basisReal1:0},{referenceReal0:0,referenceReal1:0},{phaseA:0,phaseB:0},{scaleNumerator:0},{scaleDenominator:0},{stateReal0:.5},{stateReal0:1000001},{phaseA:NaN}]){assert(!isQuantumStateParameters({...a,...patch}));assert.throws(()=>finite({...a,...patch}));}
 for(const patch of [{leftRealConstant:0},{breakNumerator:0},{lengthNumerator:0},{breakNumerator:3},{cutoffDegree:13},{bins:0},{sampleSteps:65},{lengthDenominator:0},{leftImagSlope:Infinity}]){assert(!isWavefunctionL2Parameters({...b,...patch}));assert.throws(()=>continuous({...b,...patch}));}
 for(const [p,valid]of [[a,isQuantumStateParameters],[b,isWavefunctionL2Parameters]]as const){assert(!valid({...p,extra:1}));assert(!valid({...p,[Symbol('foreign')]:1}));assert(!valid(Object.create(p)));}
 const before=structuredClone(a);assert(Object.isFrozen(finite(a).parameters));assert.deepEqual(a,before);assert(Object.isFrozen(continuous(b).parameters));
});
test('independent comparison inventory distinguishes exact rational and directed normalized coordinates',()=>{assert.equal(frefs.cases.length,24);assert.equal(lrefs.cases.length,22);assert(exactChecks>1000&&directedChecks>100);mkdirSync('.runtime/learning/quantum-state-foundation',{recursive:true});writeFileSync('.runtime/learning/quantum-state-foundation/reference-checks.json',JSON.stringify({finitePreparations:24,l2Preparations:22,l2SamplePoints:lrefs.cases.reduce((n,c)=>n+c.input.sampleSteps+1,0),exactRationalComparisons:exactChecks,directedCoordinateComparisons:directedChecks,maximumRelativeWidthAgainstMaxOneReference:1e-100},null,2)+'\n');});
