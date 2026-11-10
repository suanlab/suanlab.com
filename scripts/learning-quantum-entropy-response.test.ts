import test from'node:test';
import assert from'node:assert/strict';
import{createRequire}from'node:module';
import{calculateQuantumInformation as calculate}from'../src/lib/learning-quantum-information-calculation';
import{validateQuantumEntropyResponse as validate}from'../src/lib/learning-quantum-information-entropy-validator';
import{independentInformationLog2 as logarithm}from'../src/lib/learning-quantum-information-reference-log';
import{quantumInformationPresets}from'../src/lib/learning-quantum-information-presets';
import{f,add,sub,le,observe,type F}from'../src/lib/learning-quantum-hilbert-core';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
import joint from'./fixtures/quantum-joint-entropy-independent.json';
import logs from'./fixtures/quantum-information-reference-log-independent.json';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const input=(c='1/2',t='1/2',p='1/5',copies=2,retainedRank='3')=>({kind:'entropy',correlationWeight:c,coherence:t,probabilityOne:p,copies,retainedRank});
function decimal(s:string):F{const[m,e='0']=s.toLowerCase().split('e'),negative=m.startsWith('-'),[a,b='']=m.replace(/^[-+]/,'').split('.'),n=BigInt(a+b)*(negative?BigInt(-1):BigInt(1)),k=Number(e)-b.length;return k>=0?f(n*BigInt(10)**BigInt(k)):f(n,BigInt(10)**BigInt(-k));}
for(const r of logs.cases)test(`independent log corroboration: ${r.value}`,()=>{
 const range=logarithm(parse(r.value)),center=decimal(r.log2),allowance=decimal(logs.allowance);
 assert(le(range.lower,add(center,allowance)));assert(le(sub(center,allowance),range.upper));assert(le(sub(range.upper,range.lower),f(BigInt(1),BigInt(1)<<BigInt(500))));
});
test('binary powers have exact logarithms and zero is outside the domain',()=>{
 for(const k of [-4096,-64,-1,0,1,64,4096]){
  const x=k>=0?f(BigInt(1)<<BigInt(k)):f(BigInt(1),BigInt(1)<<BigInt(-k)),range=logarithm(x);
  assert.deepEqual(range,{lower:f(BigInt(k)),upper:f(BigInt(k))});
 }
 assert.throws(()=>logarithm(f(BigInt(0))),RangeError);
});
for(const r of joint.cases)test(`joint-state response certificate: ${r.c}/${r.t}`,()=>{
 const request=input(r.c,r.t),reply=calculate(request),checked=validate(request,reply);
 assert(checked.ok,JSON.stringify(checked));if(checked.ok){assert.equal(checked.checkedObservations,72);assert(Object.isFrozen(checked.response));}
});
for(const p of ['0','1/5','1/2','4/5','1'])for(let n=1;n<=6;n++)test(`all small-block ranks: p=${p},n=${n}`,()=>{
 for(let rank=0;rank<=2**n;rank++){
  const request=input('1','1',p,n,String(rank)),checked=validate(request,calculate(request));assert(checked.ok,JSON.stringify({rank,checked}));
 }
});
for(const p of ['0','1','1/2','1/10','1/10000000000000000000000000000000','9999999999999999999999999999999/10000000000000000000000000000000'])for(const rank of [BigInt(0),BigInt(1),(BigInt(1)<<BigInt(63))+BigInt(1),(BigInt(1)<<BigInt(64))-BigInt(1),BigInt(1)<<BigInt(64)])test(`64-copy exact response: p=${p},rank=${rank}`,()=>{
 const request=input('1','1',p,64,String(rank)),checked=validate(request,calculate(request));assert(checked.ok,JSON.stringify(checked));if(checked.ok)assert.equal(checked.checkedObservations,134);
});
for(const c of ['1/10000000000000000000000000000000','9999999999999999999999999999999/10000000000000000000000000000000'])for(const t of ['0','1/10000000000000000000000000000000','1'])test(`entropy boundary certificate: c=${c},t=${t}`,()=>{
 const request=input(c,t),checked=validate(request,calculate(request));assert(checked.ok,JSON.stringify(checked));
});
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let current=value;for(const key of path){assert(current&&typeof current==='object');current=(current as Record<string,unknown>)[String(key)];}return current;}
function setAt(value:unknown,path:Path,entry:unknown){const parent=getAt(value,path.slice(0,-1));assert(parent&&typeof parent==='object');(parent as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));}
function paths(value:unknown,prefix:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[prefix];return Object.entries(value).flatMap(([key,child])=>paths(child,[...prefix,Array.isArray(value)?Number(key):key]));}
function changeObservation(value:unknown,path:Path,change:(obs:PartitionObservation)=>PartitionObservation){const obs=getAt(value,path);assert(obs&&typeof obs==='object');setAt(value,path,change(obs as PartitionObservation));}
const request=input(),reply=calculate(request),coordinates=paths(reply.result,['result']);
for(const path of coordinates)test(`reject coherently shifted entropy coordinate: ${path.join('.')}`,()=>{
 const changed=clone(reply);changeObservation(changed,path,obs=>{assert(obs.bounds);return observe({lower:add(decode(obs.bounds.lower),parse('1/8')),upper:add(decode(obs.bounds.upper),parse('1/8'))});});assert.equal(validate(request,changed).ok,false);
});
for(const path of coordinates)test(`reject entropy display-only mutation: ${path.join('.')}`,()=>{
 const changed=clone(reply);changeObservation(changed,path,obs=>({...obs,value:obs.value===null?0:obs.value+0.125}));assert.equal(validate(request,changed).ok,false);
});
const mutations:[string,Path,unknown][]=[['version',['schemaVersion'],2],['family',['kind'],'capacity'],['raw request',['input','correlationWeight'],'2/4'],['extra result',['result','extra'],true],['PPT label',['result','joint','entangled'],true],['state contract',['result','joint','contract'],'general entropy solver'],['weight interpretation',['result','block','interpretation'],'entanglement fidelity'],['rank',['result','block','retainedRank'],'2'],['dimension',['result','block','dimension'],'8'],['memory',['result','block','requiredQubits'],1],['type order',['result','block','groups',0,'ones'],1],['multiplicity',['result','block','groups',0,'multiplicity'],'9'],['retained count',['result','block','groups',1,'retained'],'0']];
for(const[name,path,value]of mutations)test(`reject entropy structural tamper: ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(request,changed).ok,false);});
test('raw original request and output snapshots remain separate',()=>{
 const produced=clone(reply),checked=validate(request,produced);assert(checked.ok);if(!checked.ok)return;
 setAt(produced,['result','joint','entangled'],true);assert.equal(checked.response.result.joint.entangled,false);
 const other=input('1','1');assert.equal(validate(other,reply).ok,false);
});
test('getters, cycles and unsupported input family fail',()=>{
 let calls=0;const getter={get result(){calls++;return reply.result;},schemaVersion:1,kind:'entropy',input:request};assert.equal(validate(request,getter).ok,false);assert.equal(calls,0);
 const cyclic=clone(reply);setAt(cyclic,['extra'],cyclic);assert.equal(validate(request,cyclic).ok,false);
 assert.equal(validate(quantumInformationPresets['binary-qubit-state-discrimination'][0].input,reply).ok,false);
});
test('all representative numerical entropy coordinates are covered',()=>{assert.equal(coordinates.length,72);});
test('valid alternative entropy intervals require consistent raw/clipped differences',()=>{
 const make=(bits:number,rebind=true)=>{
  const changed=clone(reply),margin=f(BigInt(1),BigInt(1)<<BigInt(bits));changeObservation(changed,['result','joint','dephasedJointEntropy'],obs=>{assert(obs.bounds);return observe({lower:sub(decode(obs.bounds.lower),margin),upper:add(decode(obs.bounds.upper),margin)});});
  if(rebind){
   const read=(path:Path)=>{const obs=getAt(changed,path)as PartitionObservation;assert(obs.bounds);return{lower:decode(obs.bounds.lower),upper:decode(obs.bounds.upper)};};
   const a=read(['result','joint','dephasedJointEntropy']),b=read(['result','joint','jointEntropy']),raw={lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)};
   setAt(changed,['result','joint','rawDephasingEntropyGain'],observe(raw));setAt(changed,['result','joint','dephasingEntropyGain'],observe({lower:le(raw.lower,parse('0'))?parse('0'):raw.lower,upper:le(parse('2'),raw.upper)?parse('2'):raw.upper}));
  }
  return changed;
 };
 assert.equal(validate(request,make(240,false)).ok,false);
 const valid=validate(request,make(240));assert(valid.ok,JSON.stringify(valid));assert.equal(validate(request,make(180)).ok,false);
});
test('validator bundle excludes production models and the production log evaluator',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');
 const result=await build({entryPoints:['src/lib/learning-quantum-information-entropy-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const path of Object.keys(result.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|entropies|joint-entropy)\.ts$|learning-quantum-source-block\.ts$|\/fixtures\//.test(path));
 assert(!result.outputFiles[0].text.includes('createTranscendentalContext'));
});

test('an unresolved tiny coherence gain cannot be replaced by an exact zero',()=>{
 const x=input('1/10000000000000000000000000000000','1/10000000000000000000000000000000'),produced=calculate(x);assert(produced.kind==='entropy');
 assert.equal(produced.result.joint.dephasingEntropyGain.value,null);
 const changed=clone(produced);setAt(changed,['result','joint','dephasingEntropyGain'],observe({lower:parse('0'),upper:parse('0')}));
 assert.equal(validate(x,changed).ok,false);
});
test('a false exact entropy below display resolution is rejected',()=>{
 const changed=clone(reply),falsePoint=add(parse('7/4'),f(BigInt(1),BigInt(10)**BigInt(100))),obs=observe({lower:falsePoint,upper:falsePoint});
 assert.equal(obs.value,getAt(reply,['result','joint','jointEntropy','value']));setAt(changed,['result','joint','jointEntropy'],obs);assert.equal(validate(request,changed).ok,false);
});
test('a nonzero source eigenvalue below Number range cannot become exact zero',()=>{
 const x=input('1','1','1/10000000000000000000000000000000',64,'0'),produced=calculate(x);assert(produced.kind==='entropy');
 const last=produced.result.block.groups[64].eigenvalue;assert.equal(last.value,null);assert.equal(last.sign,1);
 const changed=clone(produced);setAt(changed,['result','block','groups',64,'eigenvalue'],observe({lower:parse('0'),upper:parse('0')}));assert.equal(validate(x,changed).ok,false);
});
test('same displayed retained weight cannot hide a lost rank bit',()=>{
 const rank=(BigInt(1)<<BigInt(63))+BigInt(1),x=input('1','1','1/2',64,String(rank)),produced=calculate(x);assert(produced.kind==='entropy');
 assert.equal(produced.result.block.retainedWeight.value,0.5);
 const changed=clone(produced);setAt(changed,['result','block','retainedWeight'],observe({lower:parse('1/2'),upper:parse('1/2')}));assert.equal(validate(x,changed).ok,false);
});

test('clipped entropy gain cannot overwrite a raw uncertainty interval',()=>{
 const x=input('4/5','0'),produced=calculate(x);assert(produced.kind==='entropy');const original=validate(x,produced);assert(original.ok,JSON.stringify(original));
 const raw=produced.result.joint.rawDephasingEntropyGain;assert(raw.bounds);assert(decode(raw.bounds.lower).n<BigInt(0));
 const changed=clone(produced);setAt(changed,['result','joint','rawDephasingEntropyGain'],clone(produced.result.joint.dephasingEntropyGain));assert.equal(validate(x,changed).ok,false);
});
