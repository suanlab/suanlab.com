import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {povmInstrumentModel as model,isPovmInstrumentParameters as valid,type PovmInstrumentParameters as P} from '../src/lib/learning-povm-instrument';
import {povmInstrumentKeys as keys} from '../src/lib/learning-povm-instrument-input';import {createPovmAlgebraic} from '../src/lib/learning-povm-algebraic';
import {f,one,zero} from '../src/lib/learning-quantum-hilbert-core';import {uncertaintyCovarianceModel} from '../src/lib/learning-uncertainty-covariance';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/povm-instrument-reference.json','utf8')) as {cases:{label:string;input:P;expected:any}[]};
type F={n:bigint;d:bigint};const fraction=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,eq=(a:F,b:F)=>a.n*b.d===b.n*a.d;
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return fraction(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),k=Number(m[4]||0)-(m[3]||'').length;return k>=0?fraction(n*BigInt(10)**BigInt(k)):fraction(n,BigInt(10)**BigInt(-k));}
const endpoint=(q:NonNullable<Q['bounds']>['lower'])=>fraction(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));let directedChecks=0,exactChecks=0,maxCoefficientDigits=0;
function check(actual:unknown,expected:string,path:string,exact=false){assert(isMasterObservation(actual),path+' shape');const q=actual as Q;assert(q.bounds,path);const lo=endpoint(q.bounds.lower),hi=endpoint(q.bounds.upper);assert(le(lo,hi),path+' order');maxCoefficientDigits=Math.max(maxCoefficientDigits,...[q.bounds.lower,q.bounds.upper].flatMap(b=>[b.numerator.coefficient.length,b.denominator.coefficient.length]));
 const decimal=expected.startsWith('decimal:'),x=parse(decimal?expected.slice(8):expected),absolute=x.n<0?-x.n:x.n,scale=absolute>x.d?absolute:x.d;
 if(decimal){const eps=BigInt(10)**BigInt(200),lower=fraction(x.n*eps-scale,x.d*eps),upper=fraction(x.n*eps+scale,x.d*eps);assert(le(lo,upper)&&le(lower,hi),path+' reference '+expected);}
 else{assert(le(lo,x)&&le(x,hi),path+' contains '+expected);if(exact){assert(eq(lo,x)&&eq(hi,x),path+' exact');exactChecks++;}}
 assert(le(fraction(hi.n*lo.d-lo.n*hi.d,hi.d*lo.d),fraction(scale,x.d*BigInt(10)**BigInt(125))),path+' width');if(!exact)directedChecks++;
}
function compare(actual:unknown,expected:unknown,path='result'){
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){const exact=!expected.startsWith('decimal:');check(actual,expected,path,exact);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}assert.equal(actual,expected,path);
}
function zeros(value:unknown,exact:boolean,path='residual'){
 if(value===null)return;
 if(isMasterObservation(value)){check(value,'0',path,exact);if(exact){assert(isMasterObservation(value,true));assert.equal((value as Q).sign,0,path);}return;}
 if(Array.isArray(value)){value.forEach((v,i)=>zeros(v,exact,path+'.'+i));return;}
 assert(value&&typeof value==='object',path);const v=value as Record<string,unknown>;
 if(Object.hasOwn(v,'rationalPart')){assert.equal(v.exactSign,0,path);zeros(v.rationalPart,true,path+'.rational');zeros(v.rootPart,true,path+'.root');zeros(v.observation,true,path+'.observation');return;}
 for(const[k,x]of Object.entries(v))zeros(x,exact,path+'.'+k);
}
for(const c of refs.cases)test('independent POVM/instrument reference: '+c.label,()=>{
 const x=model(c.input);compare(x,c.expected,c.label);zeros(x.exactResiduals,true);for(const r of x.instruments){zeros(r.exactResiduals,true);zeros(r.directedResiduals,false);assert(r.nonselectiveDeterminant.exactSign>=0);assert(r.dataProcessingSquaredSlack.exactSign>=0);assert(r.disturbance.rmsSquared.exactSign>=0);assert.notEqual(r.rawOzawa.relation,'contradiction');if(r.calibratedOzawa)assert.notEqual(r.calibratedOzawa.relation,'contradiction');}assert.doesNotThrow(()=>JSON.stringify(x));
});
const input=(label:string)=>refs.cases.find(c=>c.label===label)!.input,M=(label:string)=>model(input(label)),near=(x:number|null,y:number)=>{assert.notEqual(x,null);assert(Math.abs(x!-y)<1e-14*Math.max(1,Math.abs(y)),`${x} != ${y}`);};
test('one uninformative POVM can leave a state intact or erase it,with the same outcome probabilities',()=>{
 const x=M('positive X strength 0/1'),[l,f,p]=x.instruments;assert.deepEqual(x.effectProbabilities.map(v=>v.value),[.5,.5]);assert.deepEqual(l.nonselectiveBloch.map(v=>v.observation.value),[1,0,0]);assert.deepEqual(p.nonselectiveBloch.map(v=>v.observation.value),[0,0,0]);assert(l.classification.channelIdentity&&f.classification.channelIdentity);assert.equal(l.channel.choiRank,1);assert.equal(p.channel.choiRank,4);assert.equal(l.disturbance.rmsSquared.exactSign,0);near(p.disturbance.rmsSquared.observation.value,2);assert.equal(x.calibratedMeter,null);near(x.rawMeter.rmsErrorSquared.value,2);assert(!l.classification.globallyPreparesOwnOutcomeEigenstate);assert(p.classification.globallyPreparesOwnOutcomeEigenstate);
});
test('a fixed selected density does not establish zero RMS observable disturbance',()=>{
 const x=M('maximally mixed strength 1/1').instruments[0];assert(x.classification.selectedStateUnchanged);assert.equal(x.traceDistanceToInitial.sign,0);near(x.disturbance.rmsSquared.observation.value,2);assert(!x.classification.rmsBUndisturbedOnSelectedInput);assert(!x.classification.channelIdentity);
});
test('readout variance,intrinsic POVM noise and RMS error are distinct,and calibration has a cost',()=>{
 const x=M('positive Z strength 3/5');near(x.preparation.varianceA.value,0);near(x.rawMeter.variance.value,16/25);near(x.rawMeter.intrinsicNoiseExpectation.value,16/25);near(x.rawMeter.rmsErrorSquared.value,4/5);assert(!x.rawMeter.unbiasedForEveryInput);assert(x.calibratedMeter!.unbiasedForEveryInput);near(x.calibratedMeter!.variance.value,16/9);near(x.calibratedMeter!.rmsErrorSquared.value,16/9);near(x.calibratedMeter!.mean.value,1);near(x.calibratedMeter!.values[1].value,5/3);
});
test('sharp measurement has zero isometry RMS error without satisfying the naive product relation',()=>{
 const x=M('positive Y strength 1/1'),l=x.instruments[0];near(x.preparation.commutatorMagnitudeHalf.value,1);near(x.rawMeter.rmsErrorSquared.value,0);near(l.disturbance.rmsSquared.observation.value,2);assert(l.rawOzawa.naiveProductBoundViolated);assert(l.rawOzawa.correctedSlack.value!>0);near(l.rawOzawa.correctedLeft.value,Math.SQRT2);assert(x.preparation.robertsonSquaredSlack.sign===0);
});
test('impossible outcomes retain null conditionals for all three instruments',()=>{
 for(const label of ['zero outcome positive Z','zero outcome negative Z']){const x=M(label);for(const r of x.instruments){const o=r.outcomes.find(o=>o.probability.sign===0)!;assert(o);assert.equal(o.conditionalDensity,null);assert.equal(o.conditionalBloch,null);assert.equal(o.conditionalPurity,null);assert.equal(o.followupConditionalProbabilities,null);for(const v of o.unnormalizedBloch)assert.equal(v.exactSign,0);for(const v of o.followupJointProbabilities)assert.equal(v.exactSign,0);}}
});
test('projective effects do not force a repeatable square-root update or a unital feedback channel',()=>{
 const flip=M('sharp flip feedback'),feedback=flip.instruments[1];assert(flip.classification.sharpEffects);assert.deepEqual(flip.instruments[0].nonselectiveBloch.map(v=>v.observation.value),[0,0,1]);assert.deepEqual(feedback.nonselectiveBloch.map(v=>v.observation.value),[0,0,-1]);near(feedback.traceDistanceToInitial.value,1);near(feedback.disturbance.rmsSquared.observation.value,4);near(feedback.selectedInputSharpARepeatLeak.observation.value,1);assert(!feedback.classification.globallyPreparesOwnOutcomeEigenstate);const reset=M('sharp reset feedback').instruments[1];assert(!reset.classification.channelUnital);assert.deepEqual(reset.nonselectiveBloch.map(v=>v.observation.value),[0,0,1]);
});
test('feedback quaternion signs and axis representative phases leave all physical accounts unchanged',()=>{
 const raw=M('irrational retention'),sign=M('same global feedback sign'),phase=M('raw axis phase');assert.deepEqual(raw.effects,sign.effects);assert.deepEqual(raw.effects,phase.effects);for(let i=0;i<3;i++){assert.deepEqual(raw.instruments[i].nonselectiveDensity,sign.instruments[i].nonselectiveDensity);assert.deepEqual(raw.instruments[i].channel,phase.instruments[i].channel);assert.deepEqual(raw.instruments[i].disturbance,sign.instruments[i].disturbance);}
 const p=input('positive X strength 0/1'),quarter=model({...p,minusReal:1,minusZ:1,plusReal:1,plusZ:1}).instruments[1];assert.deepEqual(quarter.nonselectiveBloch.map(v=>v.observation.value),[0,1,0]);assert.equal(quarter.channel.choiRank,1);
});
test('tiny strength and near compatibility keep a positive disturbance and finite calibrated values',()=>{
 const x=M('near compatible tiny disturbance'),l=x.instruments[0];assert.equal(l.disturbance.rmsSquared.exactSign,1);assert(l.disturbance.rmsSquared.observation.value!>0);assert(l.disturbance.rms.value!>0&&l.disturbance.rms.value!<1e-10);assert(x.calibratedMeter!.rmsErrorSquared.value!>1e11);assert(x.calibratedMeter!.rmsError.value!==null);assert.equal(M('near zero conditional mass').instruments[0].outcomes[0].probability.sign,1);
});
test('preparation covariance and isometry error-disturbance use the same state without conflating inequalities',()=>{
 const ref=JSON.parse(readFileSync('scripts/fixtures/uncertainty-covariance-reference.json','utf8')).cases.find((c:any)=>c.label==='pure Z Pauli XY').input;const p={...ref,component0Real0:1,component0Real1:0,component0Imag1:1,aReal00:1,aReal11:-1,aReal01:0,aImag01:0,bReal00:0,bReal11:0,bReal01:1,bImag01:0},cov=uncertaintyCovarianceModel(p),x=M('positive Y strength 1/1');assert.deepEqual(cov.density,x.initialDensity);assert.deepEqual(cov.moments.varianceA,x.preparation.varianceA);assert.deepEqual(cov.moments.varianceB,x.preparation.varianceB);near(cov.moments.commutatorHalfOverI.value,1);assert.equal(cov.bounds.robertsonSlack.sign,0);assert(x.instruments[0].rawOzawa.naiveProductBoundViolated);
});
test('exact quadratic-field arithmetic folds rational roots,retains signs and rejects invalid roots/divisors',()=>{
 const rational=createPovmAlgebraic(f(BigInt(16),BigInt(25)));assert.equal(rational.observe(rational.make(zero,one)).rootPart.sign,0);near(rational.observe(rational.make(zero,one)).observation.value,.8);const a=createPovmAlgebraic(f(BigInt(3),BigInt(4))),h=a.make(zero,one);assert(a.eq(a.square(h),a.from(f(BigInt(3),BigInt(4)))));assert.equal(a.sign(a.sub(a.one,h)),1);assert.equal(a.sign(a.sub(h,a.one)),-1);assert(a.eq(a.div(h,h),a.one));const m=[[a.complex(a.one),a.complex(h)],[a.complex(h),a.complex(a.square(h))]];assert.equal(a.rank(m),1);assert.throws(()=>a.div(a.one,a.zero),RangeError);assert.throws(()=>a.rootOf(a.neg(a.one)),RangeError);assert.throws(()=>createPovmAlgebraic(f(BigInt(-1))),RangeError);
});
test('22 owned bounded integers enforce the exact state and nonzero apparatus directions',()=>{
 const p=input('irrational retention');assert.equal(keys.length,22);assert(valid(p));for(const key of keys){for(const bad of [NaN,Infinity,1.5,1000001,-1000001]){assert(!valid({...p,[key]:bad}));assert.throws(()=>model({...p,[key]:bad}),RangeError);}const v={...p}as Record<string,number>;delete v[key];assert(!valid(v));}
 for(const bad of [null,[],Object.create(p),{...p,blochX:1},{...p,blochDenominator:0},{...p,strengthDenominator:0},{...p,strengthNumerator:-1},{...p,strengthNumerator:3},{...p,axisReal0:0},{...p,followReal0:0,followReal1:0},{...p,minusReal:0},{...p,plusReal:0},{...p,extra:1},{...p,[Symbol('foreign')]:1}])assert(!valid(bad));const before=structuredClone(p);assert(Object.isFrozen(model(p).parameters));assert.deepEqual(p,before);
});
test('independent reference inventory preserves exact algebraic coefficients and reporting width',()=>{
 assert(exactChecks>5000&&directedChecks>1000);mkdirSync('.runtime/learning/quantum-povm-instrument',{recursive:true});writeFileSync('.runtime/learning/quantum-povm-instrument/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,instruments:refs.cases.length*3,exactRationalCertificates:exactChecks,directedObservationChecks:directedChecks,maxCoefficientDigits,maximumRelativeWidthAgainstMaxOneReference:1e-125},null,2)+'\n');
});
