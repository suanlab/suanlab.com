import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import reference from './fixtures/quantum-information-protocols-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumTeleportationResponse as validate} from '../src/lib/learning-quantum-information-teleportation-validator';
import {fieldRational as rat,fieldRationalWire,referenceFieldVector,fieldMultiply,fieldAdd,fieldConjugate,fieldPair,fieldComplex,fieldZero,fieldOne,fieldEqual,fieldMatrixWire} from '../src/lib/learning-quantum-information-reference-field';
import {f,one,add} from '../src/lib/learning-quantum-hilbert-core';
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const scalar=(re='0',im='0',sqrtRe='0',sqrtIm='0')=>({real:[re,sqrtRe],imaginary:[im,sqrtIm]});
const input={kind:'teleportation',input:[scalar('3/5'),scalar('0','4/5')],a:1,b:1,order:'x-then-z'},reply=calculate(input);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function coefficients(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(typeof value==='string'&&/^-?\d+(?:\/\d+)?$/.test(value))return[path];if(!value||typeof value!=='object')return[];return Object.entries(value).flatMap(([k,v])=>coefficients(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const[i,r]of reference.teleports.entries())test(`independent teleportation response ${r.name}`,()=>{
 const message=reference.dense[i%8],x={kind:'teleportation',input:r.input,a:message.a,b:message.b,order:message.order},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 for(let branch=0;branch<4;branch++){
  const actual=getAt(p,['result','teleportation','branches',branch]);
  for(const k of ['rawBob','correctedBob','probability','bobDensity','correctedDensity'])assert.deepEqual(getAt(actual,[k]),getAt(r.branches[branch],[k]));
 }
 if(checked.ok){assert(Object.isFrozen(checked.response.result.teleportation.prefixes));assert.equal(checked.checkedExactCoefficients,602+16*(message.a+message.b));}
});
for(const r of reference.dense)test(`independent dense coding response ${r.a}${r.b}/${r.order}`,()=>{
 const x={...input,a:r.a,b:r.b,order:r.order},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 for(const k of ['encodedState','decodedState','decodedDensity']as const)assert.deepEqual(getAt(p,['result','denseCoding',k]),r[k]);
});
const paths=coefficients(reply.result,['result']);
for(const path of paths)test(`reject false teleportation/dense coefficient ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path);assert(typeof old==='string');setAt(changed,path,fieldRationalWire(add(rat(old),one)));assert.equal(validate(input,changed).ok,false);
});
for(const[name,path,value]of[
 ['version',['schemaVersion'],2],['family',['kind'],'bb84'],['input spelling',['input','input',0,'real',0],'6/10'],
 ['message request',['input','a'],0],['order request',['input','order'],'z-then-x'],['extra',['result','extra'],false],
 ['site order',['result','teleportation','siteOrder'],['Bob','Alice','Input']],['gate target',['result','teleportation','gates',0,'target'],0],
 ['gate control',['result','teleportation','gates',1,'control'],0],['branch record',['result','teleportation','branches',3,'a'],0],
 ['correction chronology',['result','teleportation','branches',3,'correctionGates'],[{gate:'Z',target:0},{gate:'X',target:0}]],
 ['teleport classical bits',['result','teleportation','resources','classicalBitsSent'],1],['teleport ebits',['result','teleportation','resources','sharedEbits'],0],
 ['teleport copies',['result','teleportation','resources','quantumInputCopies'],2],['teleport contract',['result','teleportation','contract'],'instant transport'],
 ['dense index',['result','denseCoding','decodedIndex'],2],['dense message',['result','denseCoding','message'],[0,1]],['dense order',['result','denseCoding','order'],'z-then-x'],
 ['encoding order',['result','denseCoding','encoding'],[{gate:'Z',target:0},{gate:'X',target:0}]],
 ['dense ebits',['result','denseCoding','resources','sharedEbits'],0],['dense transmitted qubits',['result','denseCoding','resources','qubitsSent'],0],
 ['dense classical bits',['result','denseCoding','resources','classicalMessageBits'],1],['dense contract',['result','denseCoding','contract'],'unassisted two-bit transmission'],
]as [string,Path,unknown][])test(`reject protocol classification ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
test('Pauli order retains global phase while density and message agree',()=>{
 const x={...input,order:'z-then-x'},p=calculate(x);assert(validate(x,p).ok);
 const before=getAt(reply,['result','denseCoding','decodedState'])as {real:string[];imaginary:string[]}[],after=getAt(p,['result','denseCoding','decodedState'])as typeof before;
 assert.equal(before[3].real[0],'1');assert.equal(after[3].real[0],'-1');
 assert.deepEqual(getAt(reply,['result','denseCoding','decodedDensity']),getAt(p,['result','denseCoding','decodedDensity']));
 const changed=clone(p);setAt(changed,['result','denseCoding','decodedState'],before);assert.equal(validate(x,changed).ok,false);
});
test('changing every final dense vector phase still violates the raw prefix contract',()=>{
 const changed=clone(reply);
 for(const prefix of [['result','denseCoding','encodedState'],['result','denseCoding','decodedState']]){
  const v=getAt(changed,prefix)as {real:string[];imaginary:string[]}[];
  for(const z of v)for(const key of ['real','imaginary']as const)z[key]=z[key].map(x=>fieldRationalWire(f(-rat(x).n,rat(x).d)));
 }
 assert.equal(validate(input,changed).ok,false);
});
test('raw branch scale is preserved even when all densities are unchanged',()=>{
 const changed=clone(reply),v=getAt(changed,['result','teleportation','branches',0,'correctedBob'])as {real:string[];imaginary:string[]}[];
 for(const z of v)for(const key of ['real','imaginary']as const)z[key]=z[key].map(x=>fieldRationalWire(add(rat(x),rat(x))));
 assert.equal(validate(input,changed).ok,false);
});
test('Bob has no input/message information before communication',()=>{
 const expected=fieldMatrixWire([[fieldComplex(fieldPair(f(BigInt(1),BigInt(2)))),fieldZero],[fieldZero,fieldComplex(fieldPair(f(BigInt(1),BigInt(2))))]]);
 for(const v of [[scalar('1'),scalar()],[scalar(),scalar('1')],[scalar('1'),scalar('0','1')]])for(const a of[0,1])for(const b of[0,1]){
  const x={...input,input:v,a,b},p=calculate(x);assert(validate(x,p).ok);
  assert.deepEqual(getAt(p,['result','teleportation','bobBeforeMessage']),expected);assert.deepEqual(getAt(p,['result','denseCoding','bobBeforeTransmission']),expected);
 }
 const changed=clone(reply);setAt(changed,['result','teleportation','bobBeforeMessage'],getAt(changed,['result','teleportation','inputDensity']));assert.equal(validate(input,changed).ok,false);
});
test('all four corrected branch operators preserve a reference-entangled state',()=>{
 const columns=[[scalar('1'),scalar()],[scalar(),scalar('1')]].map(v=>calculate({...input,input:v}));
 for(const p of columns)assert(validate(getAt(p,['input']),p).ok);
 const half=fieldComplex(fieldPair(f(BigInt(1),BigInt(2))));
 const branchOperators=[0,1,2,3].map(branch=>{
  const c=columns.map(p=>referenceFieldVector(getAt(p,['result','teleportation','branches',branch,'correctedBob'])as {real:string[];imaginary:string[]}[]));
  const k=[[c[0][0],c[1][0]],[c[0][1],c[1][1]]];
  for(let i=0;i<2;i++)for(let j=0;j<2;j++)assert(fieldEqual(k[i][j],i===j?half:fieldZero));return k;
 });
 // Check the complete operator-basis action, including off-diagonal coherences.
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)for(let a=0;a<2;a++)for(let b=0;b<2;b++){
  const image=branchOperators.reduce((s,k)=>fieldAdd(s,fieldMultiply(k[a][i],fieldConjugate(k[b][j]))),fieldZero);
  assert(fieldEqual(image,a===i&&b===j?fieldOne:fieldZero));
 }
});
for(let index=0;index<6;index++)test(`large unnormalized field input ${index}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(index*10+1),x={...input,input:[scalar(`1/${D}`,`1/${D-BigInt(1)}`,`-1/${D-BigInt(2)}`,`1/${D-BigInt(3)}`),scalar(`1/${D-BigInt(4)}`,`-1/${D-BigInt(5)}`,`1/${D-BigInt(6)}`,`1/${D-BigInt(7)}`)]},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('invalid zero input, message, order and sparse records rejected',()=>{
 for(const x of [{...input,input:[scalar(),scalar()]},{...input,a:2},{...input,b:-1},{...input,order:'simultaneous'}])assert.equal(validate(x,reply).ok,false);
 const changed=clone(reply);delete(getAt(changed,['result','teleportation','branches'])as unknown[])[0];assert.equal(validate(input,changed).ok,false);
});
test('wire ownership rejects getters and cycles without executing code',()=>{
 let calls=0;const getter={schemaVersion:1,kind:'teleportation',input,get result(){calls++;return reply.result;}};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const changed=clone(reply);setAt(changed,['extra'],changed);assert.equal(validate(input,changed).ok,false);
 const mutable=clone(reply),checked=validate(input,mutable);assert(checked.ok);if(!checked.ok)return;
 setAt(mutable,['result','denseCoding','decodedIndex'],0);assert.equal(checked.response.result.denseCoding.decodedIndex,3);
});
test('exact coefficient count covers every prefix, branch, density and probability',()=>{
 const checked=validate(input,reply);assert(checked.ok);assert.equal(paths.length,634);if(checked.ok)assert.equal(checked.checkedExactCoefficients,paths.length);
});
test('validator emitted bundle excludes producer engines and fixtures',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');const r=await build({entryPoints:['src/lib/learning-quantum-information-teleportation-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|protocols)\.ts$|learning-qubit-(?:circuit-exact|scalar|channel-algebra)\.ts$|\/fixtures\//.test(p));
 assert(!r.outputFiles[0].text.includes('function runQubitCircuit'));
});
