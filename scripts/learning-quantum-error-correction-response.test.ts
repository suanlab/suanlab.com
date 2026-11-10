import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import reference from './fixtures/quantum-information-resources-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumErrorCorrectionResponse as validate} from '../src/lib/learning-quantum-information-error-correction-validator';
import {fieldRational as rat,fieldRationalWire,fieldPair,fieldComplex,fieldAdd,fieldEqual,fieldZero,fieldOne} from '../src/lib/learning-quantum-information-reference-field';
import {f,zero,one,add,observe,type F,type R} from '../src/lib/learning-quantum-hilbert-core';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const scalar=(re='0',im='0',sqrtRe='0',sqrtIm='0')=>({real:[re,sqrtRe],imaginary:[im,sqrtIm]});
const input={kind:'error-correction',input:[scalar('3/5'),scalar('0','4/5')],selectedError:'YIX',physicalXProbability:'1/10',errors:['III','XII','IXI','IIX']},reply=calculate(input);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient),s=x.numerator.exponent-x.denominator.exponent;return s>=0?f(n<<BigInt(s),d):f(n,d<<BigInt(-s));}
function range(value:unknown,path:Path):R{const x=getAt(value,path)as PartitionObservation;assert(x.bounds);return{lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};}
function coefficients(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(typeof value==='string'&&/^-?\d+(?:\/\d+)?$/.test(value))return[path];if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[];return Object.entries(value).flatMap(([k,v])=>coefficients(v,[...path,Array.isArray(value)?Number(k):k]));}
function observations(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[path];return Object.entries(value).flatMap(([k,v])=>observations(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const r of reference.cases)test(`independent code/syndrome response ${r.name}`,()=>{
 const x={...input,input:r.input,selectedError:r.error},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 for(const k of ['encodedDensity','erroredDensity','branches']as const)assert.deepEqual(getAt(p,['result','experiment',k]),r[k]);
 if(checked.ok){assert.equal(checked.checkedObservations,10);assert(Object.isFrozen(checked.response.result.experiment.branches));}
});
for(const r of reference.noise)test(`independent reference channel response ${r.name}`,()=>{
 const x={...input,input:r.input,physicalXProbability:r.p},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 for(const k of ['logicalOutput','normalizedLogicalChoi','inputSquaredFidelity','entanglementFidelity']as const)assert.deepEqual(getAt(p,['result','experiment',k]),r[k]);
});
for(const r of reference.kl)test(`independent code-image inner product ${r.words}`,()=>{
 const x={...input,errors:r.words},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));assert.equal(getAt(p,['result','condition','conditionHolds']),r.conditionHolds);
 for(let i=0;i<r.pairs.length;i++)for(const k of ['a','b','compressed','scalarOnCode','c']as const)assert.deepEqual(getAt(p,['result','condition','pairs',i,k]),r.pairs[i][k]);
});
const paths=coefficients(reply.result,['result']);
for(const path of paths)test(`reject false error-correction coefficient ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path);assert(typeof old==='string');setAt(changed,path,fieldRationalWire(add(rat(old),one)));assert.equal(validate(input,changed).ok,false);
});
const probabilityPaths=observations(reply.result,['result']);
for(const path of probabilityPaths)test(`reject reencoded logical/noise probability ${path.join('.')}`,()=>{
 const changed=clone(reply),old=range(changed,path);setAt(changed,path,observe({lower:add(old.lower,rat('1/8')),upper:add(old.upper,rat('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of probabilityPaths)test(`reject logical/noise display mutation ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...old,value:old.value===null?0:old.value+0.125});assert.equal(validate(input,changed).ok,false);
});
for(const[name,path,value]of[
 ['version',['schemaVersion'],2],['family',['kind'],'majorization'],['request spelling',['input','physicalXProbability'],'2/20'],
 ['selected error',['result','experiment','selectedError'],'XII'],['extra',['result','extra'],0],
 ['syndrome',['result','experiment','branches',0,'syndrome'],[1,1]],['correction',['result','experiment','branches',0,'correction'],'XII'],
 ['noise error',['result','experiment','noise',0,'error'],'XXX'],['noise syndrome',['result','experiment','noise',0,'syndrome'],[1,0]],
 ['noise correction',['result','experiment','noise',0,'correction'],'IXI'],['condition flag',['result','condition','conditionHolds'],false],
 ['word order',['result','condition','words'],['XII','III','IXI','IIX']],['pair index',['result','condition','pairs',0,'a'],1],
 ['pair words',['result','condition','pairs',0,'errors'],['XII','III']],['scalar flag',['result','condition','pairs',0,'scalarOnCode'],false],
 ['scalar missing',['result','condition','pairs',0,'c'],null],['first witness',['result','condition','firstWitness'],{}],
 ['experiment scope',['result','experiment','contract'],'corrects every single Pauli'],['condition scope',['result','condition','contract'],'a recovery has been constructed'],
]as [string,Path,unknown][])test(`reject error correction classification ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
test('impossible branches require zero raw vectors and null conditional states',()=>{
 const branches=getAt(reply,['result','experiment','branches'])as {conditionalLogicalDensity:unknown}[];assert.equal(branches.filter(b=>b.conditionalLogicalDensity===null).length,3);
 for(let i=0;i<4;i++)if(branches[i].conditionalLogicalDensity===null){
  const changed=clone(reply);setAt(changed,['result','experiment','branches',i,'conditionalLogicalDensity'],getAt(reply,['result','experiment','logicalOutput']));assert.equal(validate(input,changed).ok,false);
 }
});
test('a single known unitary is KL-correctable even if the selected X recovery fails',()=>{
 const x={...input,input:[scalar('1'),scalar('1')],selectedError:'ZII',errors:['ZII']},p=calculate(x);assert(validate(x,p).ok);
 assert.equal(getAt(p,['result','condition','conditionHolds']),true);
 assert.notDeepEqual(getAt(p,['result','experiment','branches',0,'conditionalLogicalDensity']),getAt(calculate({...x,selectedError:'III'}),['result','experiment','branches',0,'conditionalLogicalDensity']));
 const withIdentity={...x,errors:['III','ZII']},q=calculate(withIdentity);assert(validate(withIdentity,q).ok);assert.equal(getAt(q,['result','condition','conditionHolds']),false);
});
test('degenerate stabilizer errors and imaginary scalar cross terms retain their phases',()=>{
 for(const errors of [['III','ZZI','ZIZ','IZZ'],['XII','YZI']]){
  const x={...input,errors},p=calculate(x);assert(validate(x,p).ok);assert.equal(getAt(p,['result','condition','conditionHolds']),true);
 }
 const x={...input,errors:['XII','YZI']},p=calculate(x);
 assert.deepEqual(getAt(p,['result','condition','pairs',1,'c','imaginary']),['1','0']);assert.deepEqual(getAt(p,['result','condition','pairs',2,'c','imaginary']),['-1','0']);
 const changed=clone(p);setAt(changed,['result','condition','pairs',1,'c','imaginary',0],'-1');assert.equal(validate(x,changed).ok,false);
});
test('the first failing pair is exact and cannot be hidden by a correct global flag',()=>{
 const x={...input,errors:['III','XII','ZII','XXX']},p=calculate(x);assert(validate(x,p).ok);assert.equal(getAt(p,['result','condition','conditionHolds']),false);
 for(const value of [null,getAt(p,['result','condition','pairs',0])]){const changed=clone(p);setAt(changed,['result','condition','firstWitness'],value);assert.equal(validate(x,changed).ok,false);}
});
for(const errors of [Array(16).fill('III'),['III','XII','IXI','IIX','XXX','YII','IYI','IIY','ZII','IZI','IIZ','XXI','XIX','IXX','ZZI','YYY']])test(`sixteen-operator boundary ${errors[1]}`,()=>{
 const x={...input,errors},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));assert.equal((getAt(p,['result','condition','pairs'])as unknown[]).length,256);
 const changed=clone(p);setAt(changed,['result','condition','pairs',255,'compressed',0,0,'real',0],'2');assert.equal(validate(x,changed).ok,false);
});
test('plus-state fidelity cannot replace reference entanglement fidelity',()=>{
 const x={...input,input:[scalar('1'),scalar('1')]},p=calculate(x);assert(validate(x,p).ok);
 assert.deepEqual(getAt(p,['result','experiment','inputSquaredFidelity','real']),['1','0']);assert.deepEqual(getAt(p,['result','experiment','entanglementFidelity','real']),['243/250','0']);
 const changed=clone(p);setAt(changed,['result','experiment','entanglementFidelity'],getAt(p,['result','experiment','inputSquaredFidelity']));assert.equal(validate(x,changed).ok,false);
});
for(const p of ['0','1','1/2','1/99999999999999999999999999999999','99999999999999999999999999999998/99999999999999999999999999999999'])test(`IID channel endpoint/tiny probability ${p}`,()=>{
 const x={...input,physicalXProbability:p},produced=calculate(x),checked=validate(x,produced);assert(checked.ok,JSON.stringify(checked));
 if(p.startsWith('1/999')){const changed=clone(produced);setAt(changed,['result','experiment','logicalXProbability'],observe({lower:zero,upper:zero}));assert.equal(validate(x,changed).ok,false);}
});
test('normalized Choi is trace one with maximally mixed reference marginal',()=>{
 const p=calculate({...input,physicalXProbability:'2/7'}),raw=getAt(p,['result','experiment','normalizedLogicalChoi'])as {real:string[];imaginary:string[]}[][];
 const m=raw.map(row=>row.map(z=>fieldComplex([rat(z.real[0]),rat(z.real[1])],[rat(z.imaginary[0]),rat(z.imaginary[1])])));
 assert(fieldEqual(m.reduce((s,row,i)=>fieldAdd(s,row[i]),fieldZero),fieldOne));
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)assert(fieldEqual(fieldAdd(m[2*i][2*j],m[2*i+1][2*j+1]),i===j?fieldComplex(fieldPair(f(BigInt(1),BigInt(2)))):fieldZero));
});
for(let index=0;index<4;index++)test(`large exact input coefficients ${index}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(index*10+1),x={...input,input:[scalar(`1/${D}`,`1/${D-BigInt(1)}`,`-1/${D-BigInt(2)}`,`1/${D-BigInt(3)}`),scalar(`1/${D-BigInt(4)}`,`-1/${D-BigInt(5)}`,`1/${D-BigInt(6)}`,`1/${D-BigInt(7)}`)]},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('request and operator-family ordering remain bound',()=>{
 assert.equal(validate({...input,selectedError:'III'},reply).ok,false);
 const changed=clone(reply),x={...input,errors:[...input.errors].reverse()};setAt(changed,['input','errors'],x.errors);assert.equal(validate(x,changed).ok,false);
});
test('invalid state, family, Pauli and physical probabilities rejected',()=>{
 for(const x of [{...input,input:[scalar(),scalar()]},{...input,selectedError:'IIII'},{...input,errors:[]},{...input,errors:Array(17).fill('III')},{...input,physicalXProbability:'3/2'},{...input,errors:['__proto__']}])assert.equal(validate(x,reply).ok,false);
});
test('descriptor ownership rejects getters, cycles and sparse rows and freezes success',()=>{
 let calls=0;const getter={schemaVersion:1,kind:'error-correction',input,get result(){calls++;return reply.result;}};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
 const sparse=clone(reply);delete(getAt(sparse,['result','experiment','branches'])as unknown[])[0];assert.equal(validate(input,sparse).ok,false);
 const mutable=clone(reply),checked=validate(input,mutable);assert(checked.ok);if(!checked.ok)return;
 setAt(mutable,['result','condition','conditionHolds'],false);assert.equal(checked.response.result.condition.conditionHolds,true);
});
test('all generic packet numerical coordinates are counted',()=>{
 const checked=validate(input,reply);assert(checked.ok);assert.equal(paths.length,1176);assert.equal(probabilityPaths.length,10);if(checked.ok){assert.equal(checked.checkedExactCoefficients,paths.length);assert.equal(checked.checkedObservations,10);}
});
test('emitted validator excludes dense operator, channel and protocol producers',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');const r=await build({entryPoints:['src/lib/learning-quantum-information-error-correction-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|error-correction|protocols)\.ts$|learning-qubit-(?:circuit-exact|scalar|channel-algebra)\.ts$|\/fixtures\//.test(p));
 assert(!r.outputFiles[0].text.includes('function repetitionCodeExperiment'));
});
