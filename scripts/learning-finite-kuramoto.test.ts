import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {finiteKuramotoModel as run,isFiniteKuramotoParameters as valid,type FiniteKuramotoParameters} from '../src/lib/learning-finite-kuramoto';
import {pairPhaseLockingModel} from '../src/lib/learning-pair-phase-locking';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type{PartitionObservation as Q,PartitionRational}from'../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/finite-kuramoto-reference.json','utf8')) as {cases:{label:string;input:FiniteKuramotoParameters;analytic:boolean;exactRootGroups:number|null;expected:any}[]};
const find=(s:string)=>refs.cases.find(c=>c.label===s)!.input,maxdiff=(a:readonly number[],b:readonly number[])=>Math.max(...a.map((v,i)=>Math.abs(v-b[i]))),evidence:{label:string;maximumReferencePhaseError:number;formalEstimate:number;ratio:number|null;rawDeparture:number|null}[]=[];
let floatComparisons=0,directedComparisons=0;
function numeric(a:number,b:number,tolerance:number,label:string){assert(Number.isFinite(a)&&Number.isFinite(b),label);assert(Math.abs(a-b)<=tolerance,`${label}: ${a} vs ${b}, budget${tolerance}`);floatComparisons++;}
type Fraction={n:bigint;d:bigint};
const fraction=(n:bigint,d=BigInt(1)):Fraction=>({n,d}),le=(a:Fraction,b:Fraction)=>a.n*b.d<=b.n*a.d;
function parseReference(s:string):Fraction{
 if(s.includes('/')){const[n,d]=s.split('/');return fraction(BigInt(n),BigInt(d));}
 const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);
 const n=BigInt(m[1]+m[2]+(m[3]||'')),power=Number(m[4]||0)-(m[3]||'').length;
 return power>=0?fraction(n*BigInt(10)**BigInt(power)):fraction(n,BigInt(10)**BigInt(-power));
}
function rationalBound(q:PartitionRational):Fraction{assert.equal(q.numerator.exponent,0);assert.equal(q.denominator.exponent,0);return fraction(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));}
function directed(q:Q,expected:string){
 assert(isMasterObservation(q));assert(q.bounds);const x=parseReference(expected),a=x.n<BigInt(0)?-x.n:x.n,scale=a>x.d?a:x.d,precision=BigInt(10)**BigInt(65);
 const lo=fraction(x.n*precision-scale,x.d*precision),hi=fraction(x.n*precision+scale,x.d*precision);
 // The independent decimal reference is rounded; compare rational intervals,
 // without rounding narrow near-critical observations to binary64 first.
 assert(le(lo,rationalBound(q.bounds.lower))&&le(rationalBound(q.bounds.upper),hi),`directed reference ${expected}`);
 assert(q.value!==null);const value=Number(x.n)/Number(x.d);numeric(q.value,value,Math.max(1,Math.abs(value))*5e-15,'directed reference rounded display');directedComparisons++;
}
function finiteTree(value:unknown){if(typeof value==='number')assert(Number.isFinite(value));else if(value&&typeof value==='object')for(const child of Object.values(value))finiteTree(child);}
for(const c of refs.cases)test(`independent finite population: ${c.label}`,()=>{
 const r=run(c.input);finiteTree(r);assert.doesNotThrow(()=>JSON.stringify(r));assert.equal(r.populationSize,c.expected.populationSize);assert.equal(r.classification.initialBalancedRootGroups,c.exactRootGroups);assert.equal(r.numerics.errorEstimateIsCertified,false);assert.equal(r.classification.finitePopulationCriticalValueInferred,false);assert.equal(r.history.length,c.expected.history.length);
 let error=0;for(let i=0;i<r.history.length;i++)error=Math.max(error,maxdiff(r.history[i].phases,c.expected.history[i].phases));
 if(c.label==='coarse insufficient grid'){assert(error>1);assert(error>r.numerics.formalRichardsonErrorEstimate*10);}else{
 const budget=c.analytic?1e-13:1e-6;assert(error<=budget,`${c.label} phase accuracy ${error}`);const K=c.input.couplingNumerator/c.input.couplingDenominator;
 for(let i=0;i<r.history.length;i++){const a=r.history[i],b=c.expected.history[i];assert.equal(a.gridIndex,b.gridIndex);numeric(a.time,b.time,1e-14,'record time');for(let j=0;j<r.populationSize;j++){numeric(a.phases[j],b.phases[j],budget,'phase');numeric(a.velocities[j],b.velocities[j],2*K*budget+1e-11,'frequency');}
 for(const key of ['firstHarmonic','secondHarmonic']as const)for(const field of ['real','imaginary','magnitude','squaredMagnitude']as const)numeric(a[key][field],b[key][field],4*budget+1e-12,`${key}.${field}`);
 assert(Math.abs(a.meanPhaseBalanceResidual)<1e-11);assert(Math.abs(a.meanFrequencyResidual)<1e-11);
 }
 }
 for(const[k,v]of Object.entries(c.expected.continuumReference))if(v===null)assert.equal((r.continuumReference as any)[k],null);else directed((r.continuumReference as any)[k],v as string);
 evidence.push({label:c.label,maximumReferencePhaseError:error,formalEstimate:r.numerics.formalRichardsonErrorEstimate,ratio:r.numerics.observedRefinementRatio,rawDeparture:r.numerics.maxUnmodifiedRkDepartureFromAnalyticControl});
});
test('root-profile controls preserve exact order identities while exposing unmodified RK artifacts',()=>{
 const anti=run(find('antiphase roundoff-sensitive'));assert.equal(anti.trajectoryMethod,'analytic-invariant-profile-rotation');assert.equal(anti.classification.invariantRootGroups,2);assert.equal(anti.current.firstHarmonic.status,'analytic-zero');assert.equal(anti.current.firstHarmonic.magnitude,0);assert.equal(anti.current.firstHarmonic.meanDirection,null);assert.equal(anti.current.secondHarmonic.magnitude,1);assert.deepEqual(anti.current.phases,anti.initialPhases);assert(anti.numerics.maxUnmodifiedRkDepartureFromAnalyticControl!>.5);
 const quarter=run(find('identical quarter roots'));assert.equal(quarter.current.firstHarmonic.magnitude,0);assert.equal(quarter.current.secondHarmonic.magnitude,0);assert.equal(quarter.current.balancedGroupHarmonic!.harmonic,4);assert.equal(quarter.current.balancedGroupHarmonic!.magnitude,1);
 const roots=run(find('regular32roots'));assert.equal(roots.classification.invariantRootGroups,32);assert.equal(roots.current.balancedGroupHarmonic!.magnitude,1);
});
test('alignment does not certify frequency locking, and near-balanced profiles are not snapped onto exact controls',()=>{
 const aligned=run(find('aligned but different frequencies'));assert.equal(aligned.history[0].firstHarmonic.magnitude,1);assert(aligned.history[0].instantaneousFrequencySpread>0);assert.equal(aligned.classification.frequencyEqualityIdentityApplies,false);
 const perturbed=run(find('perturbed balanced clusters'));assert.equal(perturbed.classification.invariantRootGroups,null);assert.equal(perturbed.trajectoryMethod,'binary64-rk4-finest-grid');assert(perturbed.current.firstHarmonic.magnitude>.1);
});
test('finite samples are separate from Lorentzian branch assumptions, including rounded critical equality',()=>{
 const anti=run(find('antiphase roundoff-sensitive'));assert.equal(anti.continuumReference.regime,'above-onset');assert(anti.continuumReference.nonzeroOrder!.value!>.9);assert.equal(anti.current.firstHarmonic.magnitude,0);assert.equal(anti.continuumReference.fittedFromFiniteFrequencies,false);
 const a=run(find('continuum rounded equality above')),b=run(find('continuum rounded equality below'));
 assert.equal(a.parameters.couplingNumerator/a.parameters.couplingDenominator,2*a.parameters.referenceWidthNumerator/a.parameters.referenceWidthDenominator);assert.equal(a.continuumReference.regime,'above-onset');assert.equal(a.continuumReference.linearOrderGrowthRateAtZero.sign,1);assert.equal(b.continuumReference.regime,'below-onset');assert.equal(b.continuumReference.linearOrderGrowthRateAtZero.sign,-1);
 const critical=run(find('critical continuum reference'));assert.equal(critical.continuumReference.linearOrderGrowthRateAtZero.resolution,'structural-zero');assert.equal(critical.continuumReference.radialRelaxationTimeAtNonzeroOrder,null);
});
test('N2 matches the independent analytic pair family for K/N normalization and relative flow',()=>{
 for(const label of ['pair locked from alignment','pair drifting','pair critical from antiphase','pair quarter phase']){
 const p=find(label),r=run(p),quarter=p.phasePiDenominator===2,a=quarter?-1:p.phasePiNumerators[1]===1?1:0,b=quarter?1:p.phasePiNumerators[1]===1?0:1;
 const reference=pairPhaseLockingModel({omega1Numerator:p.omegaNumerators[0],omega1Denominator:p.omegaDenominator,omega2Numerator:p.omegaNumerators[1],omega2Denominator:p.omegaDenominator,couplingNumerator:p.couplingNumerator,couplingDenominator:p.couplingDenominator,initialHalfA:a,initialHalfB:b,meanPhaseNumerator:0,meanPhaseDenominator:1,timeNumerator:p.timeNumerator,timeDenominator:p.timeDenominator,sampleSteps:p.sampleSteps});
 numeric(Math.cos(r.current.phases[0]-r.current.phases[1]),reference.current.relativeCos.value!,1e-6,'pair relative cosine');numeric(r.current.phaseChanges[0]-r.current.phaseChanges[1],reference.current.phaseChange.value!,1e-6,'pair phase lift change');numeric(r.current.firstHarmonic.squaredMagnitude,reference.current.firstHarmonicSquared.value!,1e-6,'pair order');
 }
});
test('record density only selects existing master-grid points and does not alter integration or refinement diagnostics',()=>{
 const p=find('heterogeneous four'),a=run(p),b=run({...p,sampleSteps:64});const{step:sa,...aa}=a.current,{step:sb,...bb}=b.current;assert.equal(sa,8);assert.equal(sb,64);assert.deepEqual(aa,bb);assert.deepEqual(a.numerics,b.numerics);
 const nondividing=run(find('nondividing record grid'));assert.deepEqual(nondividing.history.map(r=>r.gridIndex),Array.from({length:14},(_,i)=>Math.floor(i*127/13)));assert.equal(nondividing.current.time,4);
});
test('mean-frame shifts and2pi lifts preserve observable relative dynamics to numerical error',()=>{
 const a=run(find('heterogeneous four')),b=run(find('mean frame shifted')),c=run(find('equivalent phase lifts'));
 for(let i=0;i<a.history.length;i++){numeric(a.history[i].firstHarmonic.magnitude,b.history[i].firstHarmonic.magnitude,1e-10,'frame-invariant order');numeric(a.history[i].instantaneousFrequencySpread,b.history[i].instantaneousFrequencySpread,1e-10,'frame-invariant spread');numeric(a.history[i].firstHarmonic.magnitude,c.history[i].firstHarmonic.magnitude,1e-10,'lift-invariant order');}
});
test('identical frequencies supply a gradient identity without assigning physical heat or entropy',()=>{
 const r=run(find('same frequency generic phases'));assert(r.current.potentialDescentIdentityApplies);for(const row of r.history){assert(row.negativeSquaredCouplingRate!<=0);numeric(row.instantaneousPotentialDerivative,row.negativeSquaredCouplingRate!,1e-11,'potential identity');}for(let i=1;i<r.history.length;i++)assert(r.history[i].couplingPotential<=r.history[i-1].couplingPotential+1e-10);
 assert.equal(run(find('heterogeneous four')).current.potentialDescentIdentityApplies,false);
});
test('step refinement is diagnostic and the formal estimate does not certify a coarse trajectory',()=>{
 const p=find('heterogeneous four'),a=run({...p,integrationSteps:32}),b=run({...p,integrationSteps:64}),c=run({...p,integrationSteps:128});assert(maxdiff(a.current.phases,b.current.phases)>maxdiff(b.current.phases,c.current.phases)*8);assert.equal(c.numerics.errorEstimateIsCertified,false);
 const coarse=run(find('coarse insufficient grid'));assert(coarse.numerics.formalRichardsonErrorEstimate>0);assert.equal(coarse.numerics.errorEstimateIsCertified,false);
});
test('zero time, free rotation, tiny windows and single oscillators retain defining distinctions',()=>{
 const zero=run(find('zero time'));assert.equal(zero.current.finiteWindowFrequencies,null);assert.equal(zero.current.finiteWindowFrequencySpread,null);assert.deepEqual(zero.numerics.stepCounts,[0,0,0]);assert.deepEqual(zero.current.phases,zero.initialPhases);
 const free=run(find('uncoupled four'));assert.equal(free.trajectoryMethod,'analytic-uncoupled-rotation');assert.deepEqual(free.current.finiteWindowFrequencies,free.omega);
 const single=run(find('single oscillator'));assert(single.classification.singleOscillator);assert.equal(single.current.firstHarmonic.magnitude,1);assert.equal(single.current.velocities[0],3);assert.equal(single.classification.finitePopulationCriticalValueInferred,false);
});
test('parameter snapshots own both arrays and reject sparse, mismatched or unsupported preparations',()=>{
 const p=structuredClone(find('heterogeneous four')),before=structuredClone(p),r=run(p);assert.deepEqual(p,before);assert(Object.isFrozen(r.parameters)&&Object.isFrozen(r.parameters.omegaNumerators)&&Object.isFrozen(r.parameters.phasePiNumerators));assert.notEqual(r.parameters.omegaNumerators,p.omegaNumerators);
 for(const patch of [{omegaNumerators:[]},{omegaNumerators:[1]},{omegaNumerators:[9,0,0,0]},{phasePiNumerators:[9,0,0,0]},{integrationSteps:0},{sampleSteps:65},{integrationSteps:4},{couplingNumerator:17},{omegaDenominator:0},{referenceWidthNumerator:0},{timeNumerator:5}]){assert(!valid({...p,...patch}));assert.throws(()=>run({...p,...patch}));}
 const sparse=Array(4);sparse[0]=0;assert(!valid({...p,omegaNumerators:sparse}));assert(!valid({...p,omegaNumerators:Object.assign([...p.omegaNumerators],{extra:1})}));assert(!valid({...p,[Symbol('extra')]:1}));assert(!valid(Object.assign(Object.create({inherited:1}),p)));
});
test('independent finite-number observations and analytic comparisons are recorded with distinct provenance',()=>{
 assert.equal(refs.cases.length,38);mkdirSync('.runtime/learning/kuramoto-foundation',{recursive:true});writeFileSync('.runtime/learning/kuramoto-foundation/reference-error-evidence.json',JSON.stringify({floatComparisons,directedComparisons,phaseAccuracyBudget:1e-6,coarseFailureIsIntentional:true,cases:evidence},null,2)+'\n');console.log(JSON.stringify({independentPreparations:refs.cases.length,floatComparisons,directedComparisons}));
});
