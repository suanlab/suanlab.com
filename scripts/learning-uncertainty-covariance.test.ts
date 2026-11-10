import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {uncertaintyCovarianceModel as model,isUncertaintyCovarianceParameters as valid,type UncertaintyCovarianceParameters as P} from '../src/lib/learning-uncertainty-covariance';
import {createUncertaintyCovarianceMath} from '../src/lib/learning-uncertainty-covariance-math';
import {f,c,add,one,mul} from '../src/lib/learning-quantum-hilbert-core';
import {uncertaintyCovarianceKeys as keys} from '../src/lib/learning-uncertainty-covariance-input';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/uncertainty-covariance-reference.json','utf8')) as {cases:{label:string;input:P;expected:any}[]};
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


function zeros(value:unknown,exact:boolean,path='residual'){if(value===null)return;if(isMasterObservation(value)){check(value,'0',path,exact);if(exact){assert(isMasterObservation(value,true));assert.equal((value as Q).sign,0,path);}return;}if(Array.isArray(value)){value.forEach((v,i)=>zeros(v,exact,path+'.'+i));return;}assert(value&&typeof value==='object',path);for(const[k,v]of Object.entries(value))zeros(v,exact,path+'.'+k);}
for(const ref of refs.cases)test('independent preparation covariance reference: '+ref.label,()=>{
 const result=model(ref.input);compare(result,ref.expected,ref.label);zeros(result.exactResiduals,true);zeros(result.directedResiduals,false);zeros(result.quantumSpectrum.directedResiduals,false);zeros(result.realSpectrum.directedResiduals,false);
 assert(result.bounds.schrodingerSlack.sign!==-1);assert(result.bounds.robertsonSlack.sign!==-1);assert.equal(result.quantumSpectrum.parts.reduce((n,p)=>n+p.rank,0),2);assert.equal(result.realSpectrum.parts.reduce((n,p)=>n+p.rank,0),2);assert.doesNotThrow(()=>JSON.stringify(result));
});
const M=(label:string)=>model(refs.cases.find(c=>c.label===label)!.input),near=(x:number|null,y:number)=>{assert.notEqual(x,null);assert(Math.abs(x!-y)<1e-14,`${x} != ${y}`);};
test('pure XY qubit needs the covariance term even when its mean commutator is zero',()=>{
 const x=M('pure XY covariance saturation');near(x.moments.varianceA.value,16/25);near(x.moments.varianceB.value,9/25);near(x.moments.symmetricCovariance.value,-12/25);near(x.bounds.varianceProduct.value,144/625);assert.equal(x.bounds.robertsonSquaredBound.sign,0);near(x.bounds.schrodingerSquaredBound.value,144/625);assert.equal(x.bounds.schrodingerSlack.sign,0);assert(x.classification.schrodingerEquality);assert(!x.classification.robertsonEquality);assert(!x.classification.operatorsCommute);assert(x.classification.commutatorMeanZero);
});
test('saturation in pure qubits is not a universal claim about pure states or commuting observables',()=>{
 const q=M('qutrit pure strict'),c=M('commuting qutrit strict');assert(q.classification.pureState&&c.classification.pureState);near(q.bounds.schrodingerSlack.value,1);assert(!q.classification.schrodingerEquality);assert(c.classification.operatorsCommute);near(c.moments.varianceA.value,2/3);near(c.moments.varianceB.value,2/9);near(c.bounds.schrodingerSlack.value,4/27);assert(!c.classification.schrodingerEquality);
});
test('operator noncommutation, mean commutator and zero variances have distinct meanings',()=>{
 const a=M('pure Z Pauli XY'),b=M('mixed Pauli XY'),zero=M('zero variance A nonzero B');assert(!a.classification.operatorsCommute&&!b.classification.operatorsCommute);near(a.moments.commutatorHalfOverI.value,1);assert.equal(b.moments.commutatorHalfOverI.sign,0);near(b.bounds.schrodingerSlack.value,1);assert(zero.classification.zeroVarianceA);near(zero.moments.varianceB.value,1);assert(zero.classification.schrodingerEquality);assert.equal(zero.complexProjection.coefficientBOnA,null);assert(!zero.complexProjection.coefficientUnique);near(zero.complexProjection.minimumResidualVariance.value,1);assert.equal(zero.bounds.standardizedCovariance,null);assert.equal(zero.bounds.standardizedCommutator,null);
});
test('equal density matrices keep covariance while changing within and between ensemble accounts',()=>{
 const z=M('Z ensemble maximally mixed'),x=M('X ensemble maximally mixed'),y=M('Y ensemble maximally mixed');for(const r of [x,y]){assert.deepEqual(r.density,z.density);assert.deepEqual(r.quantumCovariance,z.quantumCovariance);}near(z.varianceDecomposition.withinA.value,1);near(z.varianceDecomposition.betweenA.value,0);near(x.varianceDecomposition.withinA.value,0);near(x.varianceDecomposition.betweenA.value,1);near(y.varianceDecomposition.withinB.value,0);near(y.varianceDecomposition.betweenB.value,1);
});
test('offsets preserve centered accounts while signed scaling changes units without changing equality',()=>{
 const x=M('pure XY covariance saturation'),shift=M('shifted pure XY'),scale=M('scaled pure XY');assert.deepEqual(shift.quantumCovariance,x.quantumCovariance);assert.deepEqual(shift.bounds,x.bounds);near(shift.moments.meanA.real.value,x.moments.meanA.real.value!+7);near(shift.moments.meanB.real.value,x.moments.meanB.real.value!-11);near(scale.moments.varianceA.value,4*x.moments.varianceA.value!);near(scale.moments.varianceB.value,9*x.moments.varianceB.value!);near(scale.moments.symmetricCovariance.value,-6*x.moments.symmetricCovariance.value!);near(scale.bounds.varianceProduct.value,36*x.bounds.varianceProduct.value!);assert(scale.classification.schrodingerEquality);
});
test('complex projection, nonunique zero-vector slope and real combination query retain their own contracts',()=>{
 const x=M('pure Z Pauli XY'),same=M('commuting identical Z');near(x.complexProjection.coefficientBOnA!.real.value,0);near(x.complexProjection.coefficientBOnA!.imaginary.value,1);assert.equal(x.complexProjection.minimumResidualVariance.sign,0);assert.equal(same.linearCombination.variance.sign,0);assert.equal(same.moments.varianceA.value,1);assert.equal(same.moments.varianceB.value,1);assert.equal(same.moments.symmetricCovariance.value,1);assert(!same.classification.jointMeasurementDistributionSupplied);assert(!same.classification.measurementInstrumentSupplied);assert(!same.classification.measurementErrorSupplied);assert(!same.classification.canonicalPositionMomentumSupplied);
});
test('near dependence remains rank two and degenerate covariance eigenspaces have no preferred axis',()=>{
 const tiny=M('nearly dependent qutrit'),isotropic=M('pure Z Pauli XY'),zero=M('dimension 1 constant pair');near(tiny.bounds.schrodingerSlack.value,1e-12);assert.equal(tiny.quantumSpectrum.positiveRank,2);assert.equal(tiny.quantumSpectrum.parts[0].eigenvalue.sign,1);assert(tiny.quantumSpectrum.parts[0].eigenvalue.value!<1e-12);assert.equal(isotropic.realSpectrum.parts.length,1);assert.equal(isotropic.realSpectrum.parts[0].rank,2);assert.equal(isotropic.realSpectrum.parts[0].eigenvalue.value,1);assert.equal(zero.quantumSpectrum.positiveRank,0);assert.equal(zero.quantumSpectrum.parts[0].rank,2);assert.equal(zero.quantumSpectrum.parts[0].eigenvalue.sign,0);
});
test('75 owned bounded integers reject invalid weights,inactive entries,zero denominators and foreign fields',()=>{
 const input=refs.cases.find(c=>c.label==='pure Z Pauli XY')!.input;assert.equal(keys.length,75);assert(valid(input));for(const key of keys){for(const bad of [NaN,Infinity,1.5,1000001,-1000001]){assert(!valid({...input,[key]:bad}));assert.throws(()=>model({...input,[key]:bad}),RangeError);}const x={...input};delete x[key];assert(!valid(x));}
 for(const bad of [null,[],Object.create(input),{...input,dimension:0},{...input,dimension:5},{...input,weight0:-1},{...input,weight0:0},{...input,component0Real0:0},{...input,component0Real3:1},{...input,aReal22:1},{...input,bImag13:1},{...input,aDenominator:0},{...input,mixVDenominator:0},{...input,extra:1},{...input,[Symbol('foreign')]:1}])assert(!valid(bad));
 const copy=structuredClone(input),r=model(input);assert(Object.isFrozen(r.parameters));assert.deepEqual(input,copy);
});
test('reference inventory retains exact scalar certificates and directed width budgets',()=>{
 assert(exactChecks>1000&&directedChecks>1000);mkdirSync('.runtime/learning/quantum-uncertainty-covariance',{recursive:true});writeFileSync('.runtime/learning/quantum-uncertainty-covariance/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,dimensions:[...new Set(refs.cases.map(c=>c.input.dimension))].sort(),exactRationalCertificates:exactChecks,directedObservationChecks:directedChecks,maxCoefficientDigits,maximumRelativeWidthAgainstMaxOneReference:1e-125},null,2)+'\n');
});

test('normalized eigenspaces survive an exact gap smaller than the fixed interval grid',()=>{
 const math=createUncertaintyCovarianceMath(),epsilon=f(BigInt(1),BigInt(10)**BigInt(300)),base=math.spectrum(f(BigInt(2)),f(BigInt(3)),c(one)),tiny=math.spectrum(add(one,mul(f(BigInt(2)),epsilon)),add(one,mul(f(BigInt(3)),epsilon)),c(epsilon));
 assert(!tiny.eigenspacesMerged);assert.equal(tiny.positiveRank,2);assert.equal(tiny.gapSquared.sign,1);assert.equal(tiny.gapSquared.value,null);assert.deepEqual(tiny.parts.map(p=>p.projector),base.parts.map(p=>p.projector));zeros(tiny.directedResiduals,false);
});
test('the covariance helper rejects a negative Gram certificate instead of clipping it',()=>{
 const math=createUncertaintyCovarianceMath();assert.throws(()=>math.spectrum(f(BigInt(-1)),one,c(one)),RangeError);assert.throws(()=>math.spectrum(one,one,c(f(BigInt(2)))),RangeError);assert.throws(()=>math.signedOverRoot(one,f(BigInt(0))),RangeError);assert.throws(()=>math.root(f(BigInt(-1))),RangeError);
});
