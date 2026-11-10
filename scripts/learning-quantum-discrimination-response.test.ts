import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire}from'node:module';
import references from './fixtures/quantum-discrimination-independent.json';
import {calculateQuantumInformation as calculate}from'../src/lib/learning-quantum-information-calculation';
import {validateQuantumDiscriminationResponse as validate}from'../src/lib/learning-quantum-information-discrimination-validator';
import {quantumInformationPresets}from'../src/lib/learning-quantum-information-presets';
const request=(r:typeof references.cases[number])=>({kind:'discrimination',bloch0:r.bloch0,bloch1:r.bloch1,prior:r.prior});
const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
for(const r of references.cases)test(`independent response certificates: ${r.name}`,()=>{
 const input=request(r),produced=calculate(input),checked=validate(input,produced);
 assert(checked.ok,JSON.stringify(checked));if(!checked.ok)return;
 assert.notEqual(checked.response,produced);assert(Object.isFrozen(checked.response));assert(Object.isFrozen(checked.response.result));
 assert(checked.checkedObservations>=61);
});
const preset=quantumInformationPresets['binary-qubit-state-discrimination'][0].input;
const base=calculate(preset);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{
 let current=value;for(const key of path){assert(current&&typeof current==='object');current=(current as Record<string,unknown>)[String(key)];}return current;
}
function setAt(value:unknown,path:Path,entry:unknown){const parent=getAt(value,path.slice(0,-1));assert(parent&&typeof parent==='object');(parent as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function reverseAt(value:unknown,path:Path){const array=getAt(value,path);assert(Array.isArray(array));array.reverse();}
const mutations:[string,(r:unknown)=>void][]=[
 ['version',r=>setAt(r,['schemaVersion'],2)],['family',r=>setAt(r,['kind'],'capacity')],
 ['request prior',r=>setAt(r,['input','prior'],'2/4')],['request state',r=>setAt(r,['input','bloch0',1],'1/5')],
 ['extra envelope',r=>setAt(r,['extra'],true)],['extra result',r=>setAt(r,['result','extra'],true)],
 ['dimension',r=>setAt(r,['result','dimension'],3)],['strategy',r=>setAt(r,['result','strategy'],'always-one')],
 ['nonunique',r=>setAt(r,['result','uniqueOptimalEffect'],!getAt(r,['result','uniqueOptimalEffect']))],
 ['gain classification',r=>setAt(r,['result','strictlyBeatsPriorOnly'],false)],
 ['state equality',r=>setAt(r,['result','exactStateEquality'],true)],['orthogonality',r=>setAt(r,['result','exactOrthogonalSupports'],true)],
 ['fidelity convention',r=>setAt(r,['result','fidelityConvention'],'root-fidelity')],['approximation contract',r=>setAt(r,['result','approximationContract'],'exact floating point')],
 ['axis removed',r=>setAt(r,['result','measurementAxis'],null)],['axis coordinate',r=>setAt(r,['result','measurementAxis',0],clone(getAt(r,['result','measurementAxis',2])))],
 ['probability coordinate',r=>reverseAt(r,['result','conditionalOutcomeProbabilities',0])],
 ['effect coordinate',r=>reverseAt(r,['result','optimalGuessZeroEffect'])],
 ['spectrum order',r=>reverseAt(r,['result','eigenvalues'])],
 ['raw matrix coordinate',r=>setAt(r,['result','weightedDifference',0,0],clone(getAt(r,['result','weightedDifference',1,1])))],
 ['rounded value',r=>setAt(r,['result','optimalSuccess','value'],0.8)],
 ['sign',r=>setAt(r,['result','gainAbovePriorOnly','sign'],-1)],['resolution',r=>setAt(r,['result','traceDistance','resolution'],'mathematical-infinity')],
 ['log display',r=>setAt(r,['result','optimalSuccess','logMagnitude'],123)],['log resolution',r=>setAt(r,['result','optimalSuccess','logResolution'],'unresolved')],
 ['denominator zero',r=>setAt(r,['result','optimalSuccess','bounds','lower','denominator','coefficient'],'0')],
 ['false lower bound',r=>setAt(r,['result','optimalSuccess','bounds','lower','numerator','coefficient'],'100')],
 ['missing scalar',r=>{const d=getAt(r,['result']);assert(d&&typeof d==='object');delete(d as Record<string,unknown>).traceNorm;}],
 ['missing bounds',r=>setAt(r,['result','traceNorm','bounds'],null)],
 ['nonnumeric display',r=>setAt(r,['result','optimalSuccess','value'],NaN)],
];
for(const[name,mutate]of mutations)test(`reject discrimination tamper: ${name}`,()=>{
 const changed=clone(base);mutate(changed);const result=validate(preset,changed);assert.equal(result.ok,false,name);
});
test('wrong response for an otherwise valid request is rejected',()=>{
 const other=quantumInformationPresets['binary-qubit-state-discrimination'][1].input;
 assert.equal(validate(preset,calculate(other)).ok,false);
 assert.equal(validate(quantumInformationPresets['quantum-channel-capacities-coding-limits'][0].input,base).ok,false);
});
test('complex off-diagonal transpose and imaginary sign errors are rejected',()=>{
 const input={kind:'discrimination',bloch0:['0','1','0'],bloch1:['0','-1','0'],prior:'1/2'},r=calculate(input);
 assert(validate(input,r).ok);
 const changed=clone(r),a=getAt(changed,['result','optimalGuessZeroEffect',0,1]),b=getAt(changed,['result','optimalGuessZeroEffect',1,0]);setAt(changed,['result','optimalGuessZeroEffect',0,1],b);setAt(changed,['result','optimalGuessZeroEffect',1,0],a);
 assert.equal(validate(input,changed).ok,false);
});
test('snapshot rejects accessors and isolates validation results from caller mutation',()=>{
 let calls=0;const getter={get schemaVersion(){calls++;return 1;},kind:'discrimination',input:preset,result:base.result};
 assert.equal(validate(preset,getter).ok,false);assert.equal(calls,0);
 const produced=clone(base),checked=validate(preset,produced);assert(checked.ok);if(!checked.ok)return;
 setAt(produced,['result','optimalSuccess','value'],-1);assert.notEqual(checked.response.result.optimalSuccess.value,-1);
});
test('nonphysical request, cyclic response and oversized coefficient fail',()=>{
 assert.equal(validate({kind:'discrimination',bloch0:['1','1','0'],bloch1:['0','0','0'],prior:'1/2'},base).ok,false);
 const cycle=clone(base);setAt(cycle,['extra'],cycle);assert.equal(validate(preset,cycle).ok,false);
 const huge=clone(base);setAt(huge,['result','optimalSuccess','bounds','lower','numerator','coefficient'],'1'.repeat(4097));assert.equal(validate(preset,huge).ok,false);
});
for(let index=0;index<20;index++)test(`near-coincident32-digit states remain verifiable: ${index}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(index+2),input={kind:'discrimination',bloch0:[`1/${D}`,`1/${D-BigInt(1)}`,'0'],bloch1:[`1/${D+BigInt(1)}`,`1/${D}`,'0'],prior:'1/2'};
 const result=validate(input,calculate(input));assert(result.ok,JSON.stringify(result));
});
test('validator dependency bundle excludes the producer, calculation entry and fixtures',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild')as typeof import('esbuild');
 const result=await build({entryPoints:['src/lib/learning-quantum-information-discrimination-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const path of Object.keys(result.metafile!.inputs)){
  assert(!path.includes('/fixtures/'));assert(!path.endsWith('learning-quantum-information-calculation.ts'));assert(!path.endsWith('learning-quantum-information-discrimination.ts'));
 }
});

import {f,add,sub,observe,type F}from'../src/lib/learning-quantum-hilbert-core';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
const irrationalInput={kind:'discrimination',bloch0:['1/5','2/5','1/5'],bloch1:['1/3','-1/6','1/4'],prior:'1/2'};
const irrational=calculate(irrationalInput);
function observationPaths(value:unknown,prefix:(string|number)[]=[]):(string|number)[][]{
 if(!value||typeof value!=='object')return[];
 if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[prefix];
 return Object.entries(value).flatMap(([key,child])=>observationPaths(child,[...prefix,Array.isArray(value)?Number(key):key]));
}
function decode(x:PartitionRational):F{const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));}
function modifyObservation(response:unknown,path:(string|number)[],change:(value:PartitionObservation)=>PartitionObservation){
 const obs=getAt(response,path);assert(obs&&typeof obs==='object');setAt(response,path,change(obs as PartitionObservation));
}
const allPaths=observationPaths(irrational.result,['result']);
for(const path of allPaths)test(`reject coherently reencoded false interval: ${path.join('.')}`,()=>{
 const changed=clone(irrational);
 modifyObservation(changed,path,obs=>{assert(obs.bounds);return observe({lower:add(decode(obs.bounds.lower),f(BigInt(1),BigInt(8))),upper:add(decode(obs.bounds.upper),f(BigInt(1),BigInt(8)))});});
 assert.equal(validate(irrationalInput,changed).ok,false);
});
for(const path of allPaths)test(`reject display-only tamper: ${path.join('.')}`,()=>{
 const changed=clone(irrational);modifyObservation(changed,path,obs=>({...obs,value:obs.value===null?0:obs.value+0.125}));
 assert.equal(validate(irrationalInput,changed).ok,false);
});
test('valid alternative tight interval is accepted; an overly wide one is rejected',()=>{
 const modify=(bits:number)=>{const changed=clone(irrational),margin=f(BigInt(1),BigInt(1)<<BigInt(bits));modifyObservation(changed,['result','traceNorm'],obs=>{assert(obs.bounds);return observe({lower:sub(decode(obs.bounds.lower),margin),upper:add(decode(obs.bounds.upper),margin)});});return changed;};
 const tight=validate(irrationalInput,modify(400));assert(tight.ok,JSON.stringify(tight));
 const wide=validate(irrationalInput,modify(100));assert.equal(wide.ok,false);
});
test('a false bound below display resolution is still rejected algebraically',()=>{
 const changed=clone(irrational),delta=f(BigInt(1),BigInt(10)**BigInt(100));
 modifyObservation(changed,['result','traceNorm'],obs=>{assert(obs.bounds);const shifted=observe({lower:add(decode(obs.bounds.lower),delta),upper:add(decode(obs.bounds.upper),delta)});assert.equal(shifted.value,obs.value);return shifted;});
 assert.equal(validate(irrationalInput,changed).ok,false);
});
test('the mutation inventory reaches every numerical coordinate',()=>{assert.equal(allPaths.length,64);});
