import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import references from './fixtures/quantum-information-entropy-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumRelativeEntropyResponse as validate} from '../src/lib/learning-quantum-information-relative-validator';
import {f,zero,add,sub,le,observe,type F,type R} from '../src/lib/learning-quantum-hilbert-core';
import {binaryProduct} from '../src/lib/learning-binary-products';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown {let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F {const n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient),s=x.numerator.exponent-x.denominator.exponent;return s>=0?f(n<<BigInt(s),d):f(n,d<<BigInt(-s));}
function range(value:unknown,path:Path):R {const x=getAt(value,path)as PartitionObservation;assert(x.bounds);return{lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};}
function coordinates(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[path];return Object.entries(value).flatMap(([k,v])=>coordinates(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const r of references.relative)test(`relative response matrix reference: ${r.name}`,()=>{
 const input={kind:'relative-entropy',rho:r.rho,sigma:r.sigma},checked=validate(input,calculate(input));assert(checked.ok,JSON.stringify(checked));
 if(checked.ok)assert(Object.isFrozen(checked.response.result.before));
});
const input={kind:'relative-entropy',rho:['1/5','2/5','1/5'],sigma:['1/3','-1/6','1/4']},reply=calculate(input),paths=coordinates(reply.result,['result']);
for(const path of paths)test(`reject false relative interval: ${path.join('.')}`,()=>{
 const changed=clone(reply),old=range(changed,path);setAt(changed,path,observe({lower:add(old.lower,parse('1/8')),upper:add(old.upper,parse('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of paths)test(`reject relative display corruption: ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...old,value:old.value===null?0:old.value+0.125});assert.equal(validate(input,changed).ok,false);
});
for(const [name,path,value] of [
 ['version',['schemaVersion'],2],['family',['kind'],'ensemble'],['channel',['result','channel'],'different-channels'],
 ['interpretation',['result','interpretation'],'general recovery proven'],['contract',['result','before','contract'],'entrywise logarithm'],
 ['classification',['result','before','status'],'infinite'],['reason',['result','before','reason'],'equal-states'],
 ['raw spelling',['input','rho',0],'2/10'],['extra',['result','extra'],0],
]as [string,Path,unknown][])test(`reject relative structure: ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
for(const [rho,sigma]of[
 [['0','0','1'],['1','0','0']], [['0','0','0'],['0','0','1']], [['0','0','-1'],['0','0','1']],
 [['3/5','4/5','0'],['0','0','1']],
])test(`exact support failure ${rho}/${sigma}`,()=>{
 const x={kind:'relative-entropy',rho,sigma},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 const before=getAt(p,['result','before'])as {status:string;value:unknown};assert.equal(before.status,'infinite');assert.equal(before.value,null);
 const changed=clone(p);setAt(changed,['result','before','value'],observe({lower:zero,upper:zero}));assert.equal(validate(x,changed).ok,false);
});
for(const rho of [['0','0','0'],['3/5','4/5','0'],['1/5','2/5','1/5']])test(`equal states have exact zero: ${rho}`,()=>{
 const x={kind:'relative-entropy',rho,sigma:rho},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 const changed=clone(p),e=f(BigInt(1),BigInt(1)<<BigInt(240));setAt(changed,['result','before','value'],observe({lower:zero,upper:e}));assert.equal(validate(x,changed).ok,false);
});
for(let i=0;i<20;i++)test(`full-rank reference near boundary: ${i}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(i),x={kind:'relative-entropy',rho:['0','0','-1'],sigma:['0','0',`${D-BigInt(1)}/${D}`]},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 assert.equal(getAt(p,['result','before','status']),'finite');assert((getAt(p,['result','before','value','value'])as number)>100);
});
for(let i=0;i<12;i++)test(`nearly coincident states cannot become exact zero: ${i}`,()=>{
 const D=BigInt(10)**BigInt(30)-BigInt(i+1),x={kind:'relative-entropy',rho:['0','0',`1/${D}`],sigma:['0','0',`1/${D+BigInt(1)}`]},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 const changed=clone(p);for(const stage of ['before','after']){setAt(changed,['result',stage,'value'],observe({lower:zero,upper:zero}));setAt(changed,['result',stage,'rawValue'],observe({lower:zero,upper:zero}));}
 assert.equal(validate(x,changed).ok,false);
});
test('finite and infinite before/after combinations and noncommuting example',()=>{
 for(const [rho,sigma]of [[['0','0','1'],['1','0','0']],[['0','0','-1'],['0','0','1']],[['0','0','3/5'],['4/5','0','0']],[['1/5','2/5','1/5'],['0','0','0']]]){
  const x={kind:'relative-entropy',rho,sigma},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 }
 const x={kind:'relative-entropy',rho:['0','0','3/5'],sigma:['4/5','0','0']},p=calculate(x);
 assert(Math.abs((getAt(p,['result','before','value','value'])as number)-1.0150374992788438)<1e-15);
});
test('ordered arguments and the dephased pair are bound independently',()=>{
 const changed=clone(reply);setAt(changed,['input','rho'],input.sigma);setAt(changed,['input','sigma'],input.rho);
 assert.equal(validate({...input,rho:input.sigma,sigma:input.rho},changed).ok,false);
 const swap=clone(reply),before=getAt(swap,['result','before']);setAt(swap,['result','before'],getAt(swap,['result','after']));setAt(swap,['result','after'],before);assert.equal(validate(input,swap).ok,false);
});
test('same binary64 display cannot hide a false exact divergence',()=>{
 const n=getAt(reply,['result','before','value','value']);assert(typeof n==='number');const b=binaryProduct(n),r=b.exponent>=0?f(b.coefficient<<BigInt(b.exponent)):f(b.coefficient,BigInt(1)<<BigInt(-b.exponent));
 const changed=clone(reply),fake=observe({lower:r,upper:r});assert.equal(fake.value,n);
 setAt(changed,['result','before','value'],fake);setAt(changed,['result','before','rawValue'],fake);assert.equal(validate(input,changed).ok,false);
});
test('consistent alternate raw/value enclosures accepted; isolated changes rejected',()=>{
 const old=range(reply,['result','before','rawValue']),margin=f(BigInt(1),BigInt(1)<<BigInt(240)),expanded={lower:sub(old.lower,margin),upper:add(old.upper,margin)},changed=clone(reply);
 assert(le(zero,expanded.lower));setAt(changed,['result','before','rawValue'],observe(expanded));assert.equal(validate(input,changed).ok,false);
 setAt(changed,['result','before','value'],observe(expanded));const checked=validate(input,changed);assert(checked.ok,JSON.stringify(checked));
});
test('complementary spectra and spectral weights must move consistently',()=>{
 for(const field of ['sigmaSpectrum','spectralWeights']) {
  const changed=clone(reply),old=range(reply,['result','before',field,0]),margin=f(BigInt(1),BigInt(1)<<BigInt(240));
  setAt(changed,['result','before',field,0],observe({lower:sub(old.lower,margin),upper:add(old.upper,margin)}));assert.equal(validate(input,changed).ok,false);
 }
});
test('descriptor ownership rejects getter and cycles without execution',()=>{
 let calls=0;const getter={schemaVersion:1,kind:'relative-entropy',input,get result(){calls++;return reply.result;}};
 assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
 const mutable=clone(reply),checked=validate(input,mutable);assert(checked.ok);if(!checked.ok)return;
 setAt(mutable,['result','channel'],'mutated');assert.equal(checked.response.result.channel,'complete-z-dephasing');
});
test('invalid physical inputs and malformed wire values rejected',()=>{
 for(const rho of [['1','1','0'],['0','0','2']])assert.equal(validate({...input,rho},reply).ok,false);
 for(const x of [null,[],{...reply,result:null}])assert.equal(validate(input,x).ok,false);
 const changed=clone(reply);setAt(changed,['result','before','rawValue','bounds','lower','denominator','coefficient'],'0');assert.equal(validate(input,changed).ok,false);
});
test('all numerical coordinates checked for generic relative reply',()=>{
 const checked=validate(input,reply);assert(checked.ok);if(checked.ok)assert.equal(checked.checkedObservations,paths.length);assert.equal(paths.length,12);
});
test('validator bundle excludes producer models and evaluators',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');const r=await build({entryPoints:['src/lib/learning-quantum-information-relative-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|relative-entropy|discrimination|entropies)\.ts$|\/fixtures\//.test(p));
 assert(!r.outputFiles[0].text.includes('function sqrtRange'));assert(!r.outputFiles[0].text.includes('createTranscendentalContext'));
});
