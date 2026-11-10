import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import reference from './fixtures/quantum-information-protocols-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumBb84Response as validate} from '../src/lib/learning-quantum-information-bb84-validator';
import {fieldRational as rat,fieldRationalWire,fieldPair,fieldComplex,fieldMultiply,fieldDivideReal,fieldEqual,fieldOne} from '../src/lib/learning-quantum-information-reference-field';
import {snapshotManyBodyWire} from '../src/lib/learning-many-body-owned-wire';
import {informationEndpoint,createInformationObservationChecks} from '../src/lib/learning-quantum-information-response-checks';
import {f,zero,one,add,observe,type F,type R} from '../src/lib/learning-quantum-hilbert-core';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const scalar=(re='0',im='0',rootRe='0',rootIm='0')=>({real:[re,rootRe],imaginary:[im,rootIm]});
const input={kind:'bb84',input:[scalar('3/5'),scalar('0','4/5')],comparison:[scalar('1'),scalar('1')],attackFraction:'3/5',eveZProbability:'2/3',sampleSize:20};
const reply=calculate(input);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient),s=x.numerator.exponent-x.denominator.exponent;return s>=0?f(n<<BigInt(s),d):f(n,d<<BigInt(-s));}
function range(value:unknown,path:Path):R{const x=getAt(value,path)as PartitionObservation;assert(x.bounds);return{lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};}
function observations(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[path];return Object.entries(value).flatMap(([k,v])=>observations(v,[...path,Array.isArray(value)?Number(k):k]));}
function coefficients(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(typeof value==='string'&&/^-?\d+(?:\/\d+)?$/.test(value))return[path];if(!value||typeof value!=='object')return[];return Object.entries(value).flatMap(([k,v])=>coefficients(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const [i,r]of reference.teleports.entries())test(`BB84 clone/overlap exact vector ${r.name}`,()=>{
 const x={...input,input:r.input,comparison:reference.teleports[(i+1)%reference.teleports.length].input},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
 if(checked.ok){assert.equal(checked.checkedExactCoefficients,168);assert.equal(checked.checkedObservations,91);assert(Object.isFrozen(checked.response.result.cloning.joint));}
});
for(const r of reference.bb84)for(const sampleSize of [0,20,256])test(`BB84 posterior reference ${r.attack}/${r.eveZ}/${sampleSize}`,()=>{
 const x={...input,attackFraction:r.attack,eveZProbability:r.eveZ,sampleSize},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 assert.equal(getAt(p,['result','attack','siftedQber','value']),Number(r.qber.split('/')[0])/Number(r.qber.split('/')[1]??1));
});
const coefficientPaths=[...coefficients(getAt(reply,['result','cloning']),['result','cloning']),...coefficients(getAt(reply,['result','overlap']),['result','overlap'])];
for(const path of coefficientPaths)test(`reject exact field mutation ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path);assert(typeof old==='string');setAt(changed,path,fieldRationalWire(add(rat(old),one)));assert.equal(validate(input,changed).ok,false);
});
const observationPaths=observations(getAt(reply,['result','attack']),['result','attack']);
for(const path of observationPaths)test(`reject BB84 reencoded false probability ${path.join('.')}`,()=>{
 const changed=clone(reply),old=range(changed,path);setAt(changed,path,observe({lower:add(old.lower,rat('1/8')),upper:add(old.upper,rat('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of observationPaths)test(`reject BB84 false display ${path.join('.')}`,()=>{
 const changed=clone(reply),obs=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...obs,value:obs.value===null?0:obs.value+0.125});assert.equal(validate(input,changed).ok,false);
});
for(let row=0;row<80;row++)test(`reject changed BB84 branch record ${row}`,()=>{
 const changed=clone(reply),path=['result','attack','rows',row,'bobBit'],bit=getAt(changed,path);setAt(changed,path,bit===0?1:0);assert.equal(validate(input,changed).ok,false);
});
for(const [name,path,value]of[
 ['version',['schemaVersion'],2],['family',['kind'],'teleportation'],['request fraction',['input','attackFraction'],'6/10'],
 ['clone flag',['result','cloning','isExactProductClone'],true],['overlap flag',['result','overlap','compatibleWithClosedUnitaryCloning'],true],
 ['sample size',['result','attack','sampleSize'],21],['contract',['result','attack','contract'],'zero error certifies secrecy'],
 ['assumptions',['result','attack','assumptions'],[]],['extra',['result','extra'],0],['null Eve',['result','attack','rows',0,'eveBit'],0],
 ['attack flag',['result','attack','rows',2,'attacked'],false],['Alice basis',['result','attack','rows',2,'aliceBasis'],'x'],
 ['Bob basis',['result','attack','rows',2,'bobBasis'],'x'],['Eve basis',['result','attack','rows',2,'eveBasis'],'x'],
 ['Alice bit',['result','attack','rows',2,'aliceBit'],1],
]as [string,Path,unknown][])test(`reject BB84 protocol metadata ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
for(const denominator of ['99999999999999999999999999999999',String(BigInt(1)<<BigInt(106))])test(`largest IID fraction budget ${denominator}`,()=>{
 const x={...input,attackFraction:`1/${denominator}`,sampleSize:256},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 const result=getAt(p,['result','attack','probabilityAtLeastOneSampleError']);
 assert.throws(()=>createInformationObservationChecks().exact(result,zero,'$'));
 const changed=clone(p);setAt(changed,['result','attack','probabilityAtLeastOneSampleError'],observe({lower:zero,upper:zero}));assert.equal(validate(x,changed).ok,false);
});
test('wire and endpoint defaults remain bounded; explicit BB84 budgets are bounded too',()=>{
 assert.throws(()=>snapshotManyBodyWire('x'.repeat(8193)),RangeError);assert.equal(snapshotManyBodyWire('x'.repeat(8193),16384),'x'.repeat(8193));
 for(const n of [0,-1,16385,NaN,1.5])assert.throws(()=>snapshotManyBodyWire({},n),RangeError);
 for(const b of [{coefficientDigits:16385,binaryExponent:8192},{coefficientDigits:4096,binaryExponent:32769},{coefficientDigits:0,binaryExponent:8192}])assert.throws(()=>createInformationObservationChecks(b),RangeError);
 const obs=observe({lower:one,upper:one});assert(obs.bounds);const huge=clone(obs.bounds.lower);huge.numerator.coefficient='1'.repeat(10001);
 assert.throws(()=>informationEndpoint(huge,'$',{coefficientDigits:10000,binaryExponent:32768}));
});
test('largest tiny attack remains positive despite rounded display',()=>{
 const x={...input,attackFraction:'1/99999999999999999999999999999999',sampleSize:256},p=calculate(x);assert(validate(x,p).ok);
 assert.equal(getAt(p,['result','attack','siftedQber','sign']),1);
});
test('exact field inversion survives Pell cancellation without a floating epsilon',()=>{
 let a=BigInt(1),b=BigInt(1);while((a+BigInt(2)*b).toString().length<=32){[a,b]=[a+BigInt(2)*b,a+b];}
 const tiny=fieldComplex(fieldPair(f(a),f(-b))),inverse=fieldDivideReal(fieldOne,tiny.re);assert(fieldEqual(fieldMultiply(tiny,inverse),fieldOne));
 const x={...input,input:[scalar(String(a),'0',String(-b)),scalar('1')]},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('basis rays clone but a superposition fails; overlap is invariant under scale and phase',()=>{
 for(const v of [[scalar('1'),scalar()],[scalar(),scalar('0','3')]]){
  const x={...input,input:v},p=calculate(x);assert(validate(x,p).ok);assert.equal(getAt(p,['result','cloning','isExactProductClone']),true);
 }
 const x={...input,input:[scalar('1'),scalar('1')],comparison:[scalar('0','7'),scalar('0','7')]},p=calculate(x);assert(validate(x,p).ok);
 assert.equal(getAt(p,['result','cloning','isExactProductClone']),false);assert.equal(getAt(p,['result','overlap','compatibleWithClosedUnitaryCloning']),true);
});
test('comparison states are bound even though attack probabilities do not depend on them',()=>{
 assert.equal(validate({...input,comparison:[scalar('1'),scalar()]},reply).ok,false);
 const changed=clone(reply);setAt(changed,['input','comparison'],[scalar('1'),scalar()]);assert.equal(validate({...input,comparison:[scalar('1'),scalar()]},changed).ok,false);
});
test('zero vectors, invalid probabilities and sample sizes rejected',()=>{
 for(const x of [{...input,input:[scalar(),scalar()]},{...input,comparison:[scalar(),scalar()]},{...input,attackFraction:'-1/2'},{...input,eveZProbability:'3/2'},{...input,sampleSize:257}])assert.equal(validate(x,reply).ok,false);
});
test('owned replies reject getters, cycles and sparse arrays; accepted packet is isolated',()=>{
 let calls=0;const getter={schemaVersion:1,kind:'bb84',input,get result(){calls++;return reply.result;}};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
 const sparse=clone(reply);delete (getAt(sparse,['result','attack','rows'])as unknown[])[0];assert.equal(validate(input,sparse).ok,false);
 const original=clone(reply),checked=validate(input,original);assert(checked.ok);if(!checked.ok)return;
 setAt(original,['result','attack','sampleSize'],99);assert.equal(checked.response.result.attack.sampleSize,20);
});
test('noncanonical endpoint encoding cannot bypass the metadata contract',()=>{
 const changed=clone(reply),obs=getAt(changed,['result','attack','siftProbability'])as PartitionObservation;assert(obs.bounds);
 for(const side of ['lower','upper']as const){const x=obs.bounds[side];x.numerator.coefficient=String(BigInt(x.numerator.coefficient)*BigInt(2));x.numerator.exponent--;}
 assert.equal(validate(input,changed).ok,false);
});
test('exact coordinate coverage is explicit',()=>{const checked=validate(input,reply);assert(checked.ok);assert.equal(coefficientPaths.length,168);assert.equal(observationPaths.length,91);});
test('validator bundle excludes production protocol, scalar, matrix and circuit engines',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');const r=await build({entryPoints:['src/lib/learning-quantum-information-bb84-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|protocols|bb84)\.ts$|learning-qubit-(?:circuit-exact|scalar|channel-algebra)\.ts$|\/fixtures\//.test(p));
 assert(!r.outputFiles[0].text.includes('function runQubitCircuit'));
});
