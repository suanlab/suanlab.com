import test from'node:test';
import assert from'node:assert/strict';
import{createRequire}from'node:module';
import references from'./fixtures/quantum-information-entropy-independent.json';
import{calculateQuantumInformation as calculate}from'../src/lib/learning-quantum-information-calculation';
import{validateQuantumEnsembleResponse as validate}from'../src/lib/learning-quantum-information-ensemble-validator';
import{independentUnitRoot,referencePoint as point,referenceAdd as plus,referenceSubtract as minus,referenceScale as scale,referenceClip as clip}from'../src/lib/learning-quantum-information-reference-qubit';
import{f,zero,add,sub,mul,eq,le,observe,type F,type R}from'../src/lib/learning-quantum-hilbert-core';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const request=(r:typeof references.ensembles[number])=>({kind:'ensemble',bloch0:r.bloch0,bloch1:r.bloch1,prior:r.prior,measurement:r.measurement});
for(const r of references.ensembles)test(`independent ensemble response: ${r.name}`,()=>{
 const input=request(r),produced=calculate(input),checked=validate(input,produced);assert(checked.ok,JSON.stringify(checked));
 if(checked.ok){assert.equal(checked.checkedObservations,23);assert(Object.isFrozen(checked.response.result));}
});
for(let numerator=0;numerator<=31;numerator++)test(`bisection root certificate: ${numerator}/31`,()=>{
 const x=f(BigInt(numerator),BigInt(31)),root=independentUnitRoot(x);
 assert(le(mul(root.lower,root.lower),x));assert(le(x,mul(root.upper,root.upper)));
 assert(le(sub(root.upper,root.lower),f(BigInt(1),BigInt(1)<<BigInt(768))));
});
test('non-dyadic rational roots are recognized exactly',()=>{
 for(const square of ['4/25','9/25','16/25','49/625']){const x=parse(square),r=independentUnitRoot(x);assert(eq(r.lower,r.upper));assert(eq(mul(r.lower,r.lower),x));}
 assert.deepEqual(independentUnitRoot({n:BigInt(0),d:BigInt(3)}),point(zero));
 assert.throws(()=>independentUnitRoot(parse('-1/2')),RangeError);assert.throws(()=>independentUnitRoot(parse('3/2')),RangeError);
});
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let current=value;for(const key of path){assert(current&&typeof current==='object');current=(current as Record<string,unknown>)[String(key)];}return current;}
function setAt(value:unknown,path:Path,entry:unknown){const parent=getAt(value,path.slice(0,-1));assert(parent&&typeof parent==='object');(parent as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));}
function rangeOf(value:unknown,path:Path):R{const obs=getAt(value,path)as PartitionObservation;assert(obs.bounds);return{lower:decode(obs.bounds.lower),upper:decode(obs.bounds.upper)};}
function paths(value:unknown,prefix:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[prefix];return Object.entries(value).flatMap(([k,v])=>paths(v,[...prefix,Array.isArray(value)?Number(k):k]));}
const input={kind:'ensemble',bloch0:['1/5','2/5','1/5'],bloch1:['1/3','-1/6','1/4'],prior:'1/2',measurement:'helstrom'},reply=calculate(input),coordinates=paths(reply.result,['result']);
for(const path of coordinates)test(`reject reencoded false ensemble interval: ${path.join('.')}`,()=>{
 const changed=clone(reply),range=rangeOf(changed,path);setAt(changed,path,observe({lower:add(range.lower,parse('1/8')),upper:add(range.upper,parse('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of coordinates)test(`reject ensemble display mutation: ${path.join('.')}`,()=>{
 const changed=clone(reply),obs=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...obs,value:obs.value===null?0:obs.value+0.125});assert.equal(validate(input,changed).ok,false);
});
const changes:[string,Path,unknown][]=[['version',['schemaVersion'],2],['family',['kind'],'entropy'],['measurement',['result','measurement'],'z'],['raw request',['input','prior'],'2/4'],['extra',['result','extra'],true],['contract',['result','approximationContract'],'accessible information optimized']];
for(const[name,path,value]of changes)test(`reject ensemble structural mutation: ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
function rebindObservableIntervals(value:unknown){
 const state0=rangeOf(value,['result','stateEntropies',0]),state1=rangeOf(value,['result','stateEntropies',1]),average=rangeOf(value,['result','averageStateEntropy']);
 const rawChi=minus(average,plus(scale(state0,parse('1/2')),scale(state1,parse('1/2'))));
 const chi=clip(rawChi,zero,rangeOf(value,['result','classicalLabelEntropy']).upper),rawInformation=rangeOf(value,['result','rawMeasurementInformation']),information=clip(rawInformation,zero,chi.upper);
 setAt(value,['result','rawHolevoInformation'],observe(rawChi));setAt(value,['result','holevoInformation'],observe(chi));setAt(value,['result','selectedMeasurementInformation'],observe(information));setAt(value,['result','holevoGap'],observe(clip(minus(chi,information))));
}
test('valid alternative uncertainty requires consistent derived intervals',()=>{
 const changed=clone(reply),old=rangeOf(changed,['result','averageStateEntropy']),margin=f(BigInt(1),BigInt(1)<<BigInt(240));
 setAt(changed,['result','averageStateEntropy'],observe({lower:sub(old.lower,margin),upper:add(old.upper,margin)}));
 assert.equal(validate(input,changed).ok,false);
 rebindObservableIntervals(changed);const checked=validate(input,changed);assert(checked.ok,JSON.stringify(checked));
});
test('clipped measurement information cannot overwrite its raw difference',()=>{
 const x={kind:'ensemble',bloch0:['1/5','0','1/5'],bloch1:['-1/5','0','1/5'],prior:'1/2',measurement:'z'},produced=calculate(x);
 assert(validate(x,produced).ok);const raw=rangeOf(produced,['result','rawMeasurementInformation']);assert(raw.lower.n<BigInt(0));
 const changed=clone(produced);setAt(changed,['result','rawMeasurementInformation'],clone(getAt(changed,['result','selectedMeasurementInformation'])));assert.equal(validate(x,changed).ok,false);
});
test('a zero Holevo value cannot hide the raw entropy subtraction uncertainty',()=>{
 const x={kind:'ensemble',bloch0:['1/5','2/5','1/5'],bloch1:['1/5','2/5','1/5'],prior:'3/10',measurement:'z'},produced=calculate(x);
 assert(validate(x,produced).ok);assert(rangeOf(produced,['result','rawHolevoInformation']).lower.n<BigInt(0));
 const changed=clone(produced);setAt(changed,['result','rawHolevoInformation'],clone(getAt(changed,['result','holevoInformation'])));assert.equal(validate(x,changed).ok,false);
});
for(const measurement of ['z','helstrom'])test(`small positive information cannot become exact zero: ${measurement}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(1),x={kind:'ensemble',bloch0:['0','0',`1/${D}`],bloch1:['0','0',`1/${D+BigInt(1)}`],prior:'1/2',measurement},produced=calculate(x);
 const checked=validate(x,produced);assert(checked.ok,JSON.stringify(checked));
 const changed=clone(produced);setAt(changed,['result','selectedMeasurementInformation'],observe(point(zero)));assert.equal(validate(x,changed).ok,false);
});
for(let index=0;index<10;index++)for(const measurement of ['z','helstrom'])test(`near-coincident matrix reference: ${index}/${measurement}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(index+2),x={kind:'ensemble',bloch0:[`1/${D}`,`1/${D-BigInt(1)}`,'0'],bloch1:[`1/${D+BigInt(1)}`,`1/${D}`,'0'],prior:'1/2',measurement};
 const result=validate(x,calculate(x));assert(result.ok,JSON.stringify(result));
});
test('boundary priors and measurement-dependent values stay request-bound',()=>{
 for(const prior of ['0','1'])for(const measurement of ['z','helstrom']){
  const x={...input,prior,measurement};const checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
 }
 assert.equal(validate({...input,measurement:'z'},reply).ok,false);
});
test('ownership rejects getters and isolates accepted packets',()=>{
 let calls=0;const getter={get result(){calls++;return reply.result;},schemaVersion:1,kind:'ensemble',input};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const produced=clone(reply),checked=validate(input,produced);assert(checked.ok);if(!checked.ok)return;
 setAt(produced,['result','measurement'],'z');assert.equal(checked.response.result.measurement,'helstrom');
});
test('invalid physical input and cyclic packet fail',()=>{
 assert.equal(validate({...input,bloch0:['1','1','0']},reply).ok,false);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
});
test('all reported ensemble numerical coordinates are checked',()=>{assert.equal(coordinates.length,23);});
test('validator emitted bundle excludes producer models and root/log evaluators',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');
 const r=await build({entryPoints:['src/lib/learning-quantum-information-ensemble-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const path of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|ensemble|entropies|discrimination)\.ts$|\/fixtures\//.test(path));
 assert(!r.outputFiles[0].text.includes('createTranscendentalContext'));assert(!r.outputFiles[0].text.includes('function sqrtRange'));
});

for(let index=0;index<5;index++)for(const measurement of ['z','helstrom'])test(`near-pure rational states: ${index}/${measurement}`,()=>{
 const D=BigInt(10)**BigInt(15)-BigInt(index),pure=(d:bigint)=>{const denominator=d*d+BigInt(1);return[`${BigInt(2)*d}/${denominator}`,'0',`${d*d-BigInt(1)}/${denominator}`];};
 const x={kind:'ensemble',bloch0:pure(D),bloch1:pure(D+BigInt(1)),prior:'1/2',measurement},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('internal average coefficients can grow beyond the external32-digit budget',()=>{
 const D=BigInt(10)**BigInt(31),x={kind:'ensemble',bloch0:[`1/${D-BigInt(1)}`,'0','0'],bloch1:['0',`1/${D}`,'0'],prior:`1/${D+BigInt(1)}`,measurement:'helstrom'};
 const checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});


import{binaryProduct}from'../src/lib/learning-binary-products';
test('an internally coherent rounded entropy point is still mathematically false',()=>{
 const changed=clone(reply),value=getAt(reply,['result','averageStateEntropy','value']);assert(typeof value==='number');
 const binary=binaryProduct(value),rational=binary.exponent>=0?f(binary.coefficient<<BigInt(binary.exponent)):f(binary.coefficient,BigInt(1)<<BigInt(-binary.exponent)),forged=observe(point(rational));
 assert.equal(forged.value,value);setAt(changed,['result','averageStateEntropy'],forged);rebindObservableIntervals(changed);
 assert.equal(validate(input,changed).ok,false);
});
