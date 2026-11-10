import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {hermitianSpectrumModel as finite,isHermitianParameters,type HermitianParameters} from '../src/lib/learning-hermitian-spectrum';
import {sequentialObservationModel as sequential,isSequentialObservationParameters,type SequentialObservationParameters} from '../src/lib/learning-sequential-observation';
import {hermitianKeys,sequentialKeys} from '../src/lib/learning-quantum-observable-input';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
type Case<P>={label:string;input:P;expected:unknown};
const refs=JSON.parse(readFileSync('scripts/fixtures/quantum-observable-reference.json','utf8')) as {finiteCases:Case<HermitianParameters>[];sequentialCases:Case<SequentialObservationParameters>[]};
type Fraction={n:bigint;d:bigint};
const r=(n:bigint,d=BigInt(1)):Fraction=>({n,d}),le=(a:Fraction,b:Fraction)=>a.n*b.d<=b.n*a.d,eq=(a:Fraction,b:Fraction)=>a.n*b.d===b.n*a.d;
function parse(s:string):Fraction{if(s.includes('/')){const[n,d]=s.split('/');return r(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),k=Number(m[4]||0)-(m[3]||'').length;return k>=0?r(n*BigInt(10)**BigInt(k)):r(n,BigInt(10)**BigInt(-k));}
const bound=(x:NonNullable<Q['bounds']>['lower'])=>{const n=x.numerator,d=x.denominator;const power=n.exponent-d.exponent;return power>=0?r(BigInt(n.coefficient)*BigInt(2)**BigInt(power),BigInt(d.coefficient)):r(BigInt(n.coefficient),BigInt(d.coefficient)*BigInt(2)**BigInt(-power));};
let exactComparisons=0,directedComparisons=0;
function compare(actual:unknown,expected:unknown,path='result'){
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){
  // Matrix ranks are ordinary discrete metadata, not observation objects.
  if(typeof actual==='number'){assert.equal(actual,Number(expected),path);return;}
  assert(isMasterObservation(actual),path);const q=actual as Q;assert(q.bounds,path);const lo=bound(q.bounds.lower),hi=bound(q.bounds.upper);assert(le(lo,hi),path);
  if(expected.startsWith('decimal:')){const x=parse(expected.slice(8)),unit=x.n<0?-x.n:x.n,max=unit>x.d?unit:x.d,eps=BigInt(10)**BigInt(165),lower=r(x.n*eps-max,x.d*eps),upper=r(x.n*eps+max,x.d*eps);assert(le(lo,upper)&&le(lower,hi),path+' directed reference');assert(le(r(hi.n*lo.d-lo.n*hi.d,hi.d*lo.d),r(max,x.d*BigInt(10)**BigInt(145))),path+' interval width');directedComparisons++;}
  else{const x=parse(expected);assert(eq(lo,x)&&eq(hi,x),path+' exact '+expected);exactComparisons++;}return;
 }
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}assert.equal(actual,expected,path);
}
function zeros(value:unknown,path='residual'){
 if(isMasterObservation(value)){assert.equal((value as Q).sign,0,path);return;}
 if(Array.isArray(value)){value.forEach((v,i)=>zeros(v,path+'.'+i));return;}
 if(value&&typeof value==='object'){for(const[k,v]of Object.entries(value))zeros(v,path+'.'+k);return;}
 assert.equal(value,null,path);
}
for(const c of refs.finiteCases)test('independent Hermitian eigenspace reference: '+c.label,()=>{const x=finite(c.input);compare(x,c.expected,c.label);zeros(x.residuals);zeros(x.passiveBasis.meanResidual);zeros(x.passiveBasis.matrixReconstructionResidual);assert(x.variance.sign!==null&&x.variance.sign>=0);assert.doesNotThrow(()=>JSON.stringify(x));});
for(const c of refs.sequentialCases)test('independent degenerate measurement trace reference: '+c.label,()=>{const x=sequential(c.input);compare(x,c.expected,c.label);for(const m of [x.a,x.b,x.refinedA]){zeros(m.residuals);assert(m.variance.sign!==null&&m.variance.sign>=0);}for(const s of [x.aThenB,x.bThenA,x.refinedAThenB])zeros(s.residuals);assert.doesNotThrow(()=>JSON.stringify(x));});
const F=(label:string)=>finite(refs.finiteCases.find(c=>c.label===label)!.input),S=(label:string)=>sequential(refs.sequentialCases.find(c=>c.label===label)!.input);
test('coincident eigenvalues merge before Born measurement and preserve the whole input',()=>{
 const f=F('fully degenerate nonzero');assert.equal(f.outcomes.length,1);assert.equal(f.outcomes[0].rank,2);assert.equal(f.outcomes[0].probability.value,1);assert.equal(f.variance.sign,0);assert.equal(f.nonselectivePurity.value,1);
 const x=S('fully degenerate A');assert.equal(x.a.outcomes.length,1);assert.equal(x.a.outcomes[0].rank,3);assert.equal(x.a.outcomes[0].probability.value,1);assert(x.classification.coarsePreservesInitialState);assert(!x.classification.coarseAndRefinedNonselectiveStateEqual);assert.equal(x.refinedA.outcomes.length,3);
 const same=S('equal rational eigenvalues');assert.equal(same.a.outcomes.length,1);assert.equal(same.a.outcomes[0].rank,3);
});
test('negative gap sorts eigenvalue and projector together; the correct eigenstate has probability one',()=>{const base=refs.finiteCases[0].input,x=finite({...base,stateReal1:0,stateImag1:0,gapNumerator:-1});assert.equal(x.outcomes[0].eigenvalue.value,-1);assert.equal(x.outcomes[0].probability.value,1);assert.equal(x.mean.value,-1);assert.equal(x.outcomes[1].probability.sign,0);assert.equal(x.outcomes[1].conditionalAmplitudes,null);});
test('same mean is insufficient to specify a probability law or its variance',()=>{const z=F('Z with imaginary relative phase'),flat=F('same mean zero spectrum');assert.equal(z.mean.sign,0);assert.equal(flat.mean.sign,0);assert.equal(z.variance.value,1);assert.equal(flat.variance.sign,0);assert.equal(z.outcomes.length,2);assert.equal(flat.outcomes.length,1);});
test('passive basis transforms both matrix and coordinates without changing the mean',()=>{const a=F('Z with imaginary relative phase'),b=F('passive complex basis');assert.deepEqual(a.matrix,b.matrix);assert.deepEqual(a.mean,b.mean);assert.notDeepEqual(a.passiveBasis.matrix,b.passiveBasis.matrix);zeros(b.passiveBasis.meanResidual);});
test('global state phase and nonzero raw scaling preserve all observable probability laws',()=>{
 const f=F('Z with imaginary relative phase');for(const label of ['global imaginary phase','global negative scale']){const x=F(label);assert.deepEqual(x.outcomes.map(v=>v.probability),f.outcomes.map(v=>v.probability));assert.deepEqual(x.nonselectiveDensity,f.nonselectiveDensity);}
 const s=S('degenerate coherence preserved'),t=S('global imaginary scale');assert.deepEqual(s.aThenB,t.aThenB);assert.deepEqual(s.a.nonselectiveDensity,t.a.nonselectiveDensity);
});
test('non-Hermitian expectation can be imaginary while every Hermitian expectation is real',()=>{assert.equal(F('Z with imaginary relative phase').nonHermitianCounterexample.expectation.imaginary.value,.5);for(const c of refs.finiteCases)zeros(finite(c.input).residuals.directMean);});
test('degenerate Lüders projection preserves coherence that rank-one refinement removes',()=>{const x=S('degenerate coherence preserved');assert.equal(x.a.nonselectivePurity.value,1);assert.equal(x.refinedA.nonselectivePurity.value,.5);assert.equal(x.coarseRefinementSquaredDistance.value,.5);assert.equal(x.aThenB.secondMarginal[1].value,1);assert.equal(x.refinedAThenB.secondMarginal[1].value,.5);assert.equal(x.secondMarginalCoarseMinusRefined[1].value,.5);assert(x.classification.observablesCommute);assert(!x.classification.coarseAndRefinedNonselectiveStateEqual);});
test('zero branches have no conditional state or conditional second probability',()=>{const x=S('rank-one zero complementary branch');const i=x.a.outcomes.findIndex(b=>b.probability.sign===0);assert(i>=0);assert.equal(x.a.outcomes[i].conditionalAmplitudes,null);assert.equal(x.aThenB.conditionalSecondProbabilities[i],null);assert(x.aThenB.jointProbabilities[i].every(q=>q.sign===0));});
test('A-then-B and B-then-A are separate joint experiments for noncommuting observables',()=>{const x=S('order-dependent first plane');assert(!x.classification.observablesCommute);assert.notDeepEqual(x.aThenB.jointProbabilities,x.bThenA.jointProbabilities);assert.equal(x.aThenB.jointProbabilities[0][1].value,.5);assert.equal(x.bThenA.firstMarginal[1].value,.5);zeros(x.aThenB.residuals);zeros(x.bThenA.residuals);});
test('exact equality of one preparation law does not turn a nonzero commutator into zero',()=>{const x=S('same state statistics do not imply commuting');assert.equal(x.classification.observablesCommute,false);assert.deepEqual(x.aThenB.jointProbabilities,x.bThenA.jointProbabilities[0].map((_,i)=>x.bThenA.jointProbabilities.map(row=>row[i])));const y=S('order-dependent superposition');assert(!y.classification.observablesCommute);assert(y.commutatorSquaredNorm.sign!==null&&y.commutatorSquaredNorm.sign>0);});
test('all integer fields are owned, finite, bounded, denominators positive and vectors nonzero',()=>{
 for(const[valid,model,base,keys,prefixes,n]of [[isHermitianParameters,finite,refs.finiteCases[0].input,hermitianKeys,['state','axis','basis'],2],[isSequentialObservationParameters,sequential,refs.sequentialCases[0].input,sequentialKeys,['state','axisA','axisB'],3]]as const){
  assert(valid(base));for(const k of keys){for(const bad of [NaN,Infinity,1.5,1000001,-1000001]){assert(!valid({...base,[k]:bad}),k);assert.throws(()=>model({...base,[k]:bad} as never),RangeError);}const missing={...base};delete (missing as Record<string,number>)[k];assert(!valid(missing),k);if(k.endsWith('Denominator'))for(const bad of [0,-1])assert(!valid({...base,[k]:bad}));}
  for(const prefix of prefixes){const zero:Record<string,number>={...base};for(let i=0;i<n;i++){zero[`${prefix}Real${i}`]=0;zero[`${prefix}Imag${i}`]=0;}assert(!valid(zero));}
  assert(!valid({...base,foreign:1}));assert(!valid({...base,[Symbol('foreign')]:1}));assert(!valid(Object.create(base)));assert(!valid(null));assert(!valid([]));const before=structuredClone(base);assert(Object.isFrozen(model(base as never).parameters));assert.deepEqual(base,before);
 }
});
test('independent reference inventory covers both spectral families and directed coordinate enclosures',()=>{assert.equal(refs.finiteCases.length,24);assert.equal(refs.sequentialCases.length,26);assert(exactComparisons>5000);assert(directedComparisons>100);mkdirSync('.runtime/learning/quantum-observables-foundation',{recursive:true});writeFileSync('.runtime/learning/quantum-observables-foundation/reference-checks.json',JSON.stringify({finitePreparations:24,sequentialPreparations:26,exactRationalComparisons:exactComparisons,directedCoordinateComparisons:directedComparisons,maximumRelativeWidthAgainstMaxOneReference:1e-145},null,2)+'\n');});
