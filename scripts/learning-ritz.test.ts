import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync}from'node:fs';
import {ritzModel as model,isRitzParameters as valid,ritzKeys as keys,type RitzParameters as P}from'../src/lib/learning-ritz';
import {createRitzIntervalContext}from'../src/lib/learning-ritz-interval';
import {isMasterObservation}from'../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/ritz-reference.json','utf8'))as{cases:{label:string;input:P;expected:any}[]};
type F={n:bigint;d:bigint};const frac=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,eq=(a:F,b:F)=>a.n*b.d===b.n*a.d;
function parse(value:string):F {
 if(value.includes('/')){const[n,d]=value.split('/');return frac(BigInt(n),BigInt(d));}
 const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value);assert(m,value);
 const n=BigInt(m[1]+m[2]+(m[3]||'')),e=Number(m[4]||0)-(m[3]||'').length;
 return e>=0?frac(n*BigInt(10)**BigInt(e)):frac(n,BigInt(10)**BigInt(-e));
}
function endpoint(b:NonNullable<Q['bounds']>['lower']):F {
 const n=BigInt(b.numerator.coefficient),d=BigInt(b.denominator.coefficient),e=b.numerator.exponent-b.denominator.exponent;
 return e>=0?frac(n*BigInt(10)**BigInt(e),d):frac(n,d*BigInt(10)**BigInt(-e));
}
let exactChecks=0,directedChecks=0;
function check(actual:unknown,expected:string,path:string) {
 assert(isMasterObservation(actual),path+' observation schema');const a=actual as Q;assert(a.bounds,path+' bounds');const lo=endpoint(a.bounds.lower),hi=endpoint(a.bounds.upper);assert(le(lo,hi),path+' order');
 const decimal=expected.startsWith('decimal:'),x=parse(decimal?expected.slice(8):expected),abs=x.n<0?-x.n:x.n,scale=abs>x.d?abs:x.d;
 if(decimal){const eps=BigInt(10)**BigInt(205),lower=frac(x.n*eps-scale,x.d*eps),upper=frac(x.n*eps+scale,x.d*eps);assert(le(lo,upper)&&le(lower,hi),path+' reference enclosure');directedChecks++;}
 else{assert(eq(lo,x)&&eq(hi,x),path+' exact rational');exactChecks++;}
 assert(le(frac(hi.n*lo.d-lo.n*hi.d,hi.d*lo.d),frac(scale,x.d*BigInt(10)**BigInt(125))),path+' fixture enclosure width');
}
function compare(actual:unknown,expected:unknown,path='result') {
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){check(actual,expected,path);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}
 assert.equal(actual,expected,path);
}
function zeros(value:unknown,exact:boolean,path='residual') {
 if(isMasterObservation(value)){const v=value as Q;assert(v.bounds);const lo=endpoint(v.bounds.lower),hi=endpoint(v.bounds.upper);assert(le(lo,frac(BigInt(0)))&&le(frac(BigInt(0)),hi),path);if(exact)assert(eq(lo,frac(BigInt(0)))&&eq(hi,frac(BigInt(0))),path+' exact');return;}
 if(Array.isArray(value)){value.forEach((x,i)=>zeros(x,exact,path+'.'+i));return;}
 assert(value&&typeof value==='object',path);for(const[k,v]of Object.entries(value))zeros(v,exact,path+'.'+k);
}
const input=(label:string)=>{const ref=refs.cases.find(c=>c.label===label);assert(ref,label);return ref.input;},M=(label:string)=>model(input(label));
for(const ref of refs.cases)test('independent nonorthogonal Ritz reference: '+ref.label,()=>{
 const x=model(ref.input);compare(x,ref.expected,ref.label);zeros(x.exactResiduals,true);zeros(x.directedResiduals,false);assert.equal(x.spectrum.reduce((n,v)=>n+v.multiplicity,0),x.columns);assert.equal(x.ambientSpectrum.reduce((n,v)=>n+v.multiplicity,0),x.dimension);assert.doesNotThrow(()=>JSON.stringify(x));
 for(const row of x.interlacing){assert(row.lowerMargin.bounds&&row.upperMargin.bounds);assert(le(frac(BigInt(0)),endpoint(row.lowerMargin.bounds.upper)));assert(le(frac(BigInt(0)),endpoint(row.upperMargin.bounds.upper)));}
 for(const v of x.nestedSubspaces)if(v.improvement){assert(v.improvement.bounds);assert(le(frac(BigInt(0)),endpoint(v.improvement.bounds.upper)));}
});
test('hand plane has exact Gram and generalized polynomial and improves under actual space inclusion',()=>{
 const x=M('hand computed plane');['1','-8/3','1'].forEach((v,i)=>check(x.characteristicPolynomial[i],v,'hand characteristic.'+i));assert.equal(x.gramDeterminant.real.value,3);assert.equal(x.nestedSubspaces[0].minimumEnergy.value,1.5);assert(Math.abs(x.nestedSubspaces[1].minimumEnergy.value!-(4-Math.sqrt(7))/3)<1e-14);assert(x.nestedSubspaces[1].improvement!.value!>1);assert.equal(x.energy.value,1.5);assert.equal(x.variance.value,2.25);assert.equal(x.gramConditionNumber.value,3);
});
test('Galerkin stationarity does not imply a whole-space eigenstate and degenerate Ritz spaces stay whole',()=>{
 const x=M('Ritz stationary but not ambient eigenstate');assert(x.classification.selectedStateRitzStationary);assert(!x.classification.selectedStateAmbientEigenstate);assert(!x.classification.spaceInvariantUnderHamiltonian);assert.equal(x.spectrum.length,1);assert.equal(x.spectrum[0].multiplicity,2);assert.equal(x.spectrum[0].energy.value,2);zeros(x.coordinateStationarityResidual,true);assert(x.outsideVariance.value!>0);assert.equal(x.projectedVariance.sign,0);assert(x.spectrum[0].fullResidualFrobeniusSquared.value!>0);
});
test('full spaces match ambient spectra and have zero outside residual; constant H has one degenerate Ritz space',()=>{
 const a=M('nonorthogonal full space');assert(a.classification.spaceIsFull);assert(a.classification.spaceInvariantUnderHamiltonian);assert.equal(a.outsideVariance.sign,0);for(let i=0;i<a.spectrum.length;i++)assert(Math.abs(a.spectrum[i].energy.value!-a.ambientSpectrum[i].energy.value!)<1e-14);
 const b=M('constant H degenerate subspace');assert.equal(b.spectrum.length,1);assert.equal(b.spectrum[0].multiplicity,2);assert(b.classification.selectedStateRitzStationary);assert(b.classification.selectedStateAmbientEigenstate);assert.equal(b.variance.sign,0);
});
test('coordinate shear and scale preserve physical states and spaces while Euclidean coefficient lengths change',()=>{
 for(const label of ['hand computed plane','nonorthogonal full space','one column negative scale']){const x=M(label);assert.deepEqual(x.state,x.basisChange.state);assert.deepEqual(x.spaceProjector,x.basisChange.spaceProjector);assert.deepEqual(x.characteristicPolynomial,x.basisChange.characteristicPolynomial);assert.notDeepEqual(x.coefficients,x.basisChange.coefficients);assert.notEqual(x.coefficientEuclideanNormSquared.value,x.basisChange.coefficientEuclideanNormSquared.value);}
 const x=M('hand computed plane'),y=M('same plane changed coordinates');assert.deepEqual(x.spaceProjector,y.spaceProjector);assert.deepEqual(x.characteristicPolynomial,y.characteristicPolynomial);assert.deepEqual(x.energy,y.energy);
});
test('exact rank keeps nearly dependent integer columns and exposes their Gram conditioning',()=>{
 const x=M('nearly dependent two columns');assert.equal(x.gramDeterminant.real.value,1);assert.equal(x.columns,2);assert(x.gramConditionNumber.value!>1e24);assert(x.gramSpectrum[0].eigenvalue.value!>0);assert(x.gramSpectrum[0].eigenvalue.value!<1e-12);
 const y=M('ill conditioned full four space');assert.equal(y.columns,4);assert.equal(y.gramDeterminant.real.value,1);assert(y.gramConditionNumber.value!>1e45);assert(y.classification.spaceIsFull);zeros(y.exactResiduals,true);
});
test('63 owned bounded fields reject rank loss,inactive entries,extra properties and unsafe numeric preparations',()=>{
 const p=input('hand computed plane');assert.equal(keys.length,63);assert(valid(p));const before=structuredClone(p);assert(Object.isFrozen(model(p).parameters));assert.deepEqual(p,before);
 for(const key of keys){const missing={...p};delete missing[key];assert(!valid(missing));for(const bad of [NaN,Infinity,1.1,1000001,-1000001])assert(!valid({...p,[key]:bad}),key);}
 const zeroCoefficients={...p};for(const key of keys)if(key.startsWith('coefficient'))zeroCoefficients[key]=0;
 const rankLoss={...p};for(let i=0;i<4;i++)for(const part of ['Real','Imag'])rankLoss[`basis${part}${i}1`]=rankLoss[`basis${part}${i}0`];
 for(const bad of [null,[],Object.create(p),{...p,dimension:0},{...p,dimension:5},{...p,columns:4},{...p,columns:0},{...p,hDenominator:0},{...p,scaleNumerator:0},{...p,scaleDenominator:0},{...p,coefficientReal3:1},{...p,basisImag33:1},{...p,hReal33:1},{...p,extra:1},{...p,[Symbol('extra')]:0},zeroCoefficients,rankLoss]){assert(!valid(bad));assert.throws(()=>model(bad as P),RangeError);}
 let accessed=false;const getter={...p};Object.defineProperty(getter,'basisReal00',{get(){accessed=true;return 1;}});assert(!valid(getter));assert(!accessed);
 const single=input('one column negative scale');assert(!valid({...single,shearReal:1}));
});
test('directed Ritz arithmetic preserves signed rounding and rejects uncertified zero divisors',()=>{
 const c=createRitzIntervalContext(),a=c.from(frac(BigInt(-1),BigInt(3))),b=c.from(frac(BigInt(7),BigInt(5)));
 check(c.observation(c.multiply(a,b)),'decimal:-0.4666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667','negative product');
 const q=c.divide(a,b);const o=c.observation(q);assert(o.bounds);assert(le(endpoint(o.bounds.lower),frac(BigInt(-5),BigInt(21)))&&le(frac(BigInt(-5),BigInt(21)),endpoint(o.bounds.upper)));assert.throws(()=>c.divide(a,{lower:-BigInt(1),upper:BigInt(1)}),RangeError);assert.throws(()=>c.from(frac(BigInt(1),BigInt(0))),RangeError);assert.throws(()=>c.range({lower:frac(BigInt(2)),upper:frac(BigInt(1))}),RangeError);
});
test('independent inventory covers every dimension/trial-column pair without changing registered coverage',()=>{
 assert.equal(new Set(refs.cases.map(c=>`${c.input.dimension}x${c.input.columns}`)).size,10);assert(refs.cases.length>=40);assert(exactChecks>2000);assert(directedChecks>2000);mkdirSync('.runtime/learning/quantum-ritz-foundation',{recursive:true});writeFileSync('.runtime/learning/quantum-ritz-foundation/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,dimensionColumnPairs:10,exactRationalCertificates:exactChecks,directedObservations:directedChecks,fixtureMaximumWidthAgainstMaxOneReference:1e-125,referenceTolerance:1e-205},null,2)+'\n');
});
