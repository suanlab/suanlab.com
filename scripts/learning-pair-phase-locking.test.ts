import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pairPhaseLockingModel as run,isPairPhaseParameters as valid,type PairPhaseParameters} from '../src/lib/learning-pair-phase-locking';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type{PartitionObservation as Q,PartitionRational}from'../src/lib/learning-partition-exact-core';
const fixture=JSON.parse(readFileSync('scripts/fixtures/pair-phase-locking-reference.json','utf8')) as {cases:{label:string;input:PairPhaseParameters;expected:unknown}[]};
type F={n:bigint;d:bigint};const f=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
function rational(q:PartitionRational):F{assert.equal(q.numerator.exponent,0);assert.equal(q.denominator.exponent,0);return f(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));}
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return f(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),power=Number(m[4]||0)-(m[3]||'').length;return power>=0?f(n*BigInt(10)**BigInt(power)):f(n,BigInt(10)**BigInt(-power));}
let certificates=0;
function compare(actual:unknown,expected:unknown,label:string){
 if(typeof expected==='string'){const q=actual as Q;assert(isMasterObservation(q),label);assert(q.bounds,label);const v=parse(expected),m=v.n<BigInt(0)?-v.n:v.n,scale=m>v.d?m:v.d,e=BigInt(10)**BigInt(160),lo=f(v.n*e-scale,v.d*e),hi=f(v.n*e+scale,v.d*e);assert(le(rational(q.bounds.lower),hi)&&le(lo,rational(q.bounds.upper)),`${label}: ${q.value} vs ${expected}`);certificates++;return;}
 if(expected===null){assert.equal(actual,null,label);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),label);assert.equal(actual.length,expected.length,label);expected.forEach((v,i)=>compare(actual[i],v,`${label}.${i}`));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',label);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,`${label}.${k}`);return;}
 assert.deepEqual(actual,expected,label);
}
const find=(label:string)=>fixture.cases.find(c=>c.label===label)!.input;
const enclosesZero=(q:Q)=>{assert(q.bounds);assert(le(rational(q.bounds.lower),f(BigInt(0)))&&le(f(BigInt(0)),rational(q.bounds.upper)));};
for(const c of fixture.cases)test(`independent pair phase reference: ${c.label}`,()=>{
 const r=run(c.input);compare(r,c.expected,c.label);assert.doesNotThrow(()=>JSON.stringify(r));assert(r.liftPartition.subintervals<=256);assert(r.liftPartition.maxRelativeIncrementBound.value!<=.5);
 enclosesZero(r.flowDeterminantResidual);for(const row of r.history){enclosesZero(row.relativeCircleResidual);assert.equal(row.phaseBranch.status,'exact-integer-count');assert.equal(row.completedRelativeTurns.status,'exact-integer-count');assert.equal(row.meanFrequencyConservationResidual.resolution,'structural-zero');}
});
test('total coupling differs from edge coupling and exact stable/unstable phases remain stationary',()=>{
 const stable=run(find('stable exact fixed phase')),unstable=run(find('unstable exact fixed phase'));
 assert.equal(stable.totalCoupling.value,5);assert.equal(stable.edgeCoupling.value,2.5);assert.equal(stable.localLockingDecayRate!.value,4);assert.equal(stable.localLockingRelaxationTime!.value,.25);
 assert.equal(stable.classification.initialFixedType,'stable');assert.equal(unstable.classification.initialFixedType,'unstable');assert.equal(stable.current.phaseChange.resolution,'structural-zero');assert.equal(unstable.current.phaseChange.resolution,'structural-zero');
 assert.equal(stable.current.angularFrequency1.value,1.5);assert.equal(stable.current.angularFrequency2.value,1.5);assert.equal(stable.fixedPhases[0].linearGrowthRate.value,-4);assert.equal(stable.fixedPhases[1].linearGrowthRate.value,4);
 assert.equal(stable.current.firstHarmonicSquared.value,.9);assert.equal(unstable.current.firstHarmonicSquared.value,.1);
});
test('critical cut crossings and completed relative turns are distinct with an explicit endpoint convention',()=>{
 const at=run(find('critical exact cut at two seconds')),beyond=run(find('critical from beyond the bottleneck')),reverse=run(find('reverse critical exact cut'));
 assert.equal(at.current.relativeCos.value,-1);assert.equal(at.current.relativeSin.value,0);assert.equal(at.current.phaseBranch.signedCount,0);assert.equal(at.current.completedRelativeTurns.signedCount,0);
 assert.equal(beyond.current.phaseBranch.signedCount,1);assert.equal(beyond.current.completedRelativeTurns.signedCount,0);assert.equal(beyond.current.wrappedRelativePhase!.resolution,'structural-zero');assert.equal(beyond.current.relativeAngularVelocity.value,1);assert.equal(beyond.current.firstHarmonicSquared.value,1);
 assert.equal(reverse.current.phaseBranch.signedCount,-1);assert(reverse.current.wrappedRelativePhase!.value!>3);assert.equal(reverse.current.completedRelativeTurns.signedCount,0);
});
test('drift turns have direction, finite-window frequency is not its asymptotic value, and period boundaries are not rounded',()=>{
 const positive=run(find('drift positive')),negative=run(find('drift negative'));assert.equal(positive.current.completedRelativeTurns.signedCount,3);assert.equal(negative.current.completedRelativeTurns.signedCount,-3);assert(positive.current.phaseChange.value!>0);assert(negative.current.phaseChange.value!<0);
 assert(Math.abs(positive.current.finiteWindowRelativeFrequency!.value!-positive.asymptoticRelativeFrequency.value!)>.01);
 assert.equal(run(find('drift near period below')).current.completedRelativeTurns.signedCount,0);assert.equal(run(find('drift near period above')).current.completedRelativeTurns.signedCount,1);
});
test('small nonzero discriminants stay on their exact side of the locking threshold',()=>{
 const locked=run(find('near threshold locked')),drift=run(find('near threshold drifting')),critical=run(find('critical from alignment'));
 assert.equal(locked.regime,'locking-fixed-phases');assert(locked.discriminant.sign===1);assert.equal(drift.regime,'drifting-relative-phase');assert(drift.discriminant.sign===-1);assert(drift.asymptoticRelativeFrequency.sign===1);assert.equal(drift.current.completedRelativeTurns.signedCount,0);assert.equal(critical.regime,'critical-one-sided-bottleneck');assert.equal(critical.localLockingRelaxationTime,null);
});
test('neutral and unstable antiphase states can have equal frequencies and zero first harmonic',()=>{
 const neutral=run(find('neutral antiphase')),unstable=run(find('identical unstable antiphase'));
 assert.equal(neutral.regime,'neutral-all-relative-phases');assert.equal(neutral.classification.initialFixedType,'neutral');assert.equal(unstable.classification.initialFixedType,'unstable');
 for(const r of [neutral,unstable]){assert.equal(r.current.firstHarmonicSquared.resolution,'structural-zero');assert.equal(r.current.secondHarmonicSquared.value,1);assert.equal(r.current.relativeAngularVelocity.resolution,'structural-zero');assert.equal(r.current.phaseChange.resolution,'structural-zero');}
});
test('record density and homogeneous half-angle gauge do not change the physical flow',()=>{
 const p=find('drift positive'),before=structuredClone(p),a=run(p),b=run({...p,sampleSteps:64});const{step:sa,...ra}=a.current,{step:sb,...rb}=b.current;assert.equal(sa,8);assert.equal(sb,64);assert.deepEqual(ra,rb);assert.deepEqual(p,before);assert(Object.isFrozen(a.parameters));
 assert.deepEqual(run(find('stable exact fixed phase')).history,run(find('homogeneous gauge negative')).history);
});
test('a shared frequency shift preserves relative dynamics while moving the mean frame',()=>{
 const a=run(find('locking from alignment')),b=run(find('mean frame shifted'));
 for(let i=0;i<a.history.length;i++){for(const key of ['relativePhase','phaseChange','relativeAngularVelocity','relativeCos','relativeSin','firstHarmonicSquared']as const)assert.deepEqual(a.history[i][key],b.history[i][key]);assert.equal(b.history[i].meanPhase.value!-a.history[i].meanPhase.value!,2*a.history[i].time.value!);}
});
test('zero observation time leaves initial phase and wrapping intact without defining a finite-window rate',()=>{
 const r=run(find('zero time'));assert.equal(r.current.phaseChange.resolution,'structural-zero');assert.equal(r.current.phaseBranch.signedCount,0);assert.equal(r.current.completedRelativeTurns.signedCount,0);assert.equal(r.current.finiteWindowRelativeFrequency,null);assert.equal(r.current.relativeCos.value,-1);
});
test('owned integer preparation validation rejects unsupported rates, projective origins and extra fields',()=>{
 const p=find('locking from alignment');assert(valid(p));for(const patch of [{initialHalfA:0,initialHalfB:0},{couplingNumerator:-1},{couplingNumerator:17},{omega1Numerator:9},{sampleSteps:0},{sampleSteps:65},{timeNumerator:5},{omega2Denominator:0},{meanPhaseNumerator:9},{omega1Numerator:NaN},{omega1Numerator:.5}]){assert(!valid({...p,...patch}));assert.throws(()=>run({...p,...patch}));}
 assert(!valid({...p,extra:0}));assert(!valid({...p,[Symbol('extra')]:1}));assert(!valid(Object.assign(Object.create({inherited:0}),p)));
});
test('independent reference evidence inventory is explicit',()=>{assert.equal(fixture.cases.length,38);console.log(JSON.stringify({independentPreparations:fixture.cases.length,scalarCertificates:certificates}));assert.equal(certificates,8428);});
