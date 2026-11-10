import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {rayleighModel as model,isRayleighParameters as valid,rayleighKeys as keys,type RayleighParameters as P} from '../src/lib/learning-rayleigh';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/rayleigh-reference.json','utf8')) as {cases:{label:string;input:P;expected:any}[]};
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
const input=(label:string)=>{const x=refs.cases.find(c=>c.label===label);assert(x,label);return x.input;},M=(label:string)=>model(input(label));
for(const ref of refs.cases)test('independent Rayleigh reference: '+ref.label,()=>{
 const x=model(ref.input);compare(x,ref.expected,ref.label);zeros(x.exactResiduals,true);zeros(x.directedResiduals,false);assert.equal(x.spectrum.reduce((n,r)=>n+r.multiplicity,0),x.dimension);assert.doesNotThrow(()=>JSON.stringify(x));
 if(x.ground.certifiedLeakageUpperBound){const upper=endpoint(x.ground.certifiedLeakageUpperBound.bounds!.lower);assert(le(upper,frac(BigInt(1))));const expected=parse(ref.expected.ground.leakage.slice(8)),eps=frac(BigInt(1),BigInt(10)**BigInt(200));assert(le(frac(expected.n*eps.d-eps.n*expected.d,expected.d*eps.d),upper),ref.label+' leakage bound');}
});
test('zero variance and stationarity distinguish ground from exact excited eigenstates',()=>{
 const a=M('ground eigenstate'),b=M('excited eigenstate');for(const x of [a,b]){assert(x.classification.exactEigenstate);assert(x.variation.stationaryInAllComplexDirections);assert.equal(x.variance.sign,0);}
 assert(a.classification.groundEigenstate);assert(!a.classification.excitedEigenstate);assert(b.classification.excitedEigenstate);assert(!b.classification.groundEigenstate);assert.equal(b.ground.overlap.value,0);assert.equal(b.ground.leakage.value,1);
});
test('degenerate ground means the entire ground eigenspace and constant H has no excitation gap',()=>{
 const a=M('ground vector in degenerate plane');assert.equal(a.ground.multiplicity,2);assert(a.classification.groundEigenstate);assert.equal(a.ground.overlap.value,1);
 const b=M('constant four dimensional');assert.equal(b.spectrum.length,1);assert.equal(b.ground.multiplicity,4);assert(b.classification.groundEigenstate);assert.equal(b.ground.gap,null);assert.equal(b.ground.energyOverGap,null);assert.equal(b.ground.certifiedLeakageUpperBound,null);
});
test('a zero path state remains undefined while a zero direction leaves every path unchanged',()=>{
 const a=M('zero path state');assert(a.classification.pathZeroVector);assert(!a.path.defined);assert.equal(a.path.moments,null);
 const b=M('zero direction');assert(!b.classification.nonzeroDirection);assert(b.path.defined);assert.equal(b.variation.firstDerivative.sign,0);assert.equal(b.variation.secondDerivative.sign,0);assert.deepEqual(b.path.moments!.energy,b.energy);
});
test('small nonzero variance and negative near-zero spectral energy are not thresholded to zero',()=>{
 const a=M('tiny positive variance');assert.equal(a.variance.sign,1);assert(a.variance.value!>0&&a.variance.value!<1e-20);assert(!a.classification.exactEigenstate);
 const b=M('negative near zero ground');assert.equal(b.ground.energy.sign,-1);assert(b.ground.energy.value!<0&&b.ground.energy.value!>-1e-10);
});
test('raw scale and phase preserve physical state observables without overwriting raw amplitudes',()=>{
 const a=M('four distinct signed roots');for(const label of ['raw state scaled','negative raw state','raw state global phase']){const b=M(label);assert.deepEqual(a.energy,b.energy);assert.deepEqual(a.variance,b.variance);for(let i=0;i<a.spectrum.length;i++){const p=a.spectrum[i].probability.value!,q=b.spectrum[i].probability.value!;assert(Math.abs(p-q)<1e-14);}assert.notDeepEqual(a.state,b.state);}
});
test('Hamiltonian shifts preserve residual/variance; negative scales reverse the spectral order',()=>{
 const a=M('four distinct signed roots'),b=M('Hamiltonian shift'),c=M('Hamiltonian positive scale'),d=M('Hamiltonian negative scale');assert.deepEqual(a.variance,b.variance);assert.deepEqual(a.residualNumerators,b.residualNumerators);assert(Math.abs(b.energy.value!-a.energy.value!-13)<1e-12);assert(Math.abs(c.variance.value!-9*a.variance.value!)<1e-10);assert(Math.abs(d.variance.value!-4*a.variance.value!)<1e-10);
 for(let i=0;i<a.spectrum.length;i++)assert(Math.abs(d.spectrum[i].energy.value!+2*a.spectrum[a.spectrum.length-1-i].energy.value!)<1e-12);
});
test('phase and scale directions are flat even for a noneigenstate; nearby distinct levels remain separate',()=>{
 for(const label of ['pure phase direction','pure scale direction']){const x=M(label);assert(!x.classification.exactEigenstate);assert.equal(x.variation.firstDerivative.sign,0);assert.equal(x.variation.secondDerivative.sign,0);assert.deepEqual(x.path.moments!.energy,x.energy);}
 const x=M('near degenerate separated blocks');assert.equal(x.spectrum.length,4);assert(x.spectrum.every(v=>v.multiplicity===1));assert(x.ground.gap!.value!>0&&x.ground.gap!.value!<1e-10);
 const y=M('complex two degenerate subspaces');assert.equal(y.spectrum.length,2);assert.deepEqual(y.spectrum.map(v=>v.multiplicity),[2,2]);
});
test('input contract rejects missing/extra/inherited/accessor/fractional/inactive/zero state preparations',()=>{
 const p=input('equal phase superposition');assert.equal(keys.length,36);assert(valid(p));const before=structuredClone(p);assert(Object.isFrozen(model(p).parameters));assert.deepEqual(p,before);
 for(const key of keys){const missing={...p};delete missing[key];assert(!valid(missing));for(const bad of [NaN,Infinity,1.5,1000001,-1000001])assert(!valid({...p,[key]:bad}),key);}
 const zero={...p};for(const key of keys)if(key.startsWith('state'))zero[key]=0;
 for(const bad of [null,[],Object.create(p),{...p,dimension:0},{...p,dimension:5},{...p,hDenominator:0},{...p,pathDenominator:0},{...p,stateReal3:1},{...p,directionImag3:1},{...p,hReal33:1},{...p,extra:0},{...p,[Symbol('extra')]:0},zero]){assert(!valid(bad));assert.throws(()=>model(bad as P),RangeError);}
 let accessed=false;const getter={...p};Object.defineProperty(getter,'stateReal0',{get(){accessed=true;return 1;}});assert(!valid(getter));assert(!accessed);
});
test('independent reference inventory records full dimensions and both exact and directed certificates',()=>{
 assert.equal(new Set(refs.cases.map(c=>c.input.dimension)).size,4);assert(refs.cases.length>=45);assert(exactChecks>500);assert(directedChecks>1000);mkdirSync('.runtime/learning/quantum-variational-foundation',{recursive:true});writeFileSync('.runtime/learning/quantum-variational-foundation/reference-checks.json',JSON.stringify({preparations:refs.cases.length,uniquePreparations:new Set(refs.cases.map(c=>JSON.stringify(c.input))).size,dimensions:[1,2,3,4],exactRationalCertificates:exactChecks,directedObservations:directedChecks,fixtureMaximumWidthAgainstMaxOneReference:1e-125,referenceTolerance:1e-205},null,2)+'\n');
});
