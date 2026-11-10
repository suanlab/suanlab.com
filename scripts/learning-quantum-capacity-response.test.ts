import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import reference from './fixtures/quantum-information-capacity-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumCapacityResponse as validate} from '../src/lib/learning-quantum-information-capacity-validator';
import {validateQuantumInformationResponse as dispatch} from '../src/lib/learning-quantum-information-response';
import {quantumInformationPresets} from '../src/lib/learning-quantum-information-presets';
import {quantumInformationKinds} from '../src/lib/learning-quantum-information-input';
import {f,zero,one,two,add,sub,observe,type F,type R} from '../src/lib/learning-quantum-hilbert-core';
import {referencePoint as point,referenceAdd as plus,referenceSubtract as minus,referenceClip as clip} from '../src/lib/learning-quantum-information-reference-qubit';
import {binaryProduct} from '../src/lib/learning-binary-products';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const rat=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const input={kind:'capacity',probability:'1/10',input:['1/5','-2/5','1/3']},reply=calculate(input);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient),s=x.numerator.exponent-x.denominator.exponent;return s>=0?f(n<<BigInt(s),d):f(n,d<<BigInt(-s));}
function range(value:unknown,path:Path):R{const x=getAt(value,path)as PartitionObservation;assert(x.bounds);return{lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};}
function observations(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[path];return Object.entries(value).flatMap(([k,v])=>observations(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const r of reference.cases)test(`capacity response independent Stinespring/Choi ${r.name}`,()=>{
 const x={kind:'capacity',probability:r.p,input:r.bloch},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
 if(checked.ok){assert.equal(checked.checkedObservations,77);assert(Object.isFrozen(checked.response.result.capacitiesByFamilyTheorem));}
});
const paths=observations(reply.result,['result']);
for(const path of paths)test(`reject reencoded false capacity observation ${path.join('.')}`,()=>{
 const changed=clone(reply),old=range(changed,path);setAt(changed,path,observe({lower:add(old.lower,rat('1/8')),upper:add(old.upper,rat('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of paths)test(`reject false capacity display ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...old,value:old.value===null?0:old.value+0.125});assert.equal(validate(input,changed).ok,false);
});
for(const[name,path,value]of[
 ['version',['schemaVersion'],2],['family',['kind'],'ensemble'],['request spelling',['input','probability'],'2/20'],
 ['positivity',['result','quantumCapacityStrictlyPositive'],false],['scope',['result','contract'],'generic channel optimizer'],['assumptions',['result','assumptions'],[]],['extra',['result','extra'],false],
]as [string,Path,unknown][])test(`reject capacity metadata ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
function rebind(value:unknown){
 const h=range(value,['result','phaseErrorEntropy']),quantum=clip(minus(point(one),h));
 setAt(value,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse'],observe(quantum));setAt(value,['result','xBasisMessageInformation'],observe(quantum));setAt(value,['result','capacitiesByFamilyTheorem','entanglementAssistedClassicalBitsPerUse'],observe(minus(point(two),h)));
 const raw=minus(range(value,['result','outputEntropy']),range(value,['result','environmentEntropy']));setAt(value,['result','rawSelectedInputCoherentInformation'],observe(raw));setAt(value,['result','selectedInputCoherentInformation'],observe(raw));setAt(value,['result','selectedInputReferenceMutualInformation'],observe(plus(range(value,['result','inputEntropy']),raw)));
}
test('valid alternate entropy intervals require consistent derived information',()=>{
 const changed=clone(reply),path=['result','phaseErrorEntropy'],old=range(changed,path),margin=f(BigInt(1),BigInt(1)<<BigInt(240));
 setAt(changed,path,observe({lower:sub(old.lower,margin),upper:add(old.upper,margin)}));assert.equal(validate(input,changed).ok,false);
 rebind(changed);const checked=validate(input,changed);assert(checked.ok,JSON.stringify(checked));
});
test('the same rounded entropy point is false even with coherently rebound rates',()=>{
 const changed=clone(reply),path=['result','phaseErrorEntropy'],n=getAt(reply,[...path,'value']);assert(typeof n==='number');const b=binaryProduct(n),r=b.exponent>=0?f(b.coefficient<<BigInt(b.exponent)):f(b.coefficient,BigInt(1)<<BigInt(-b.exponent));
 setAt(changed,path,observe(point(r)));rebind(changed);assert.equal(validate(input,changed).ok,false);
});
test('raw entropy subtraction cannot be replaced with a structural zero',()=>{
 const x={...input,input:['3/5','4/5','0']},p=calculate(x);assert(validate(x,p).ok);
 const raw=range(p,['result','rawSelectedInputCoherentInformation']);assert(raw.lower.n<BigInt(0)&&raw.upper.n>BigInt(0));
 const changed=clone(p);setAt(changed,['result','rawSelectedInputCoherentInformation'],observe(point(zero)));assert.equal(validate(x,changed).ok,false);
});
for(const probability of ['0','1','1/2'])test(`identity/full dephasing/Z endpoint ${probability}`,()=>{
 const x={...input,probability},p=calculate(x);assert(validate(x,p).ok);
 if(probability==='1/2'){assert.equal(getAt(p,['result','quantumCapacityStrictlyPositive']),false);assert.equal(getAt(p,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse','value']),0);assert.equal(getAt(p,['result','selectedInputCoherentInformation','value']),0);}
 else{assert.equal(getAt(p,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse','value']),1);assert.equal(getAt(p,['result','capacitiesByFamilyTheorem','entanglementAssistedClassicalBitsPerUse','value']),2);}
});
for(const v of [['3/5','4/5','0'],['0','0','1'],['0','1','0']])test(`pure input information is distinct from family capacity ${v}`,()=>{
 const x={...input,input:v},p=calculate(x);assert(validate(x,p).ok);assert.equal(getAt(p,['result','selectedInputCoherentInformation','value']),0);assert.equal(getAt(p,['result','selectedInputReferenceMutualInformation','value']),0);
 assert((getAt(p,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse','value'])as number)>0.5);
 const changed=clone(p);setAt(changed,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse'],getAt(p,['result','selectedInputCoherentInformation']));assert.equal(validate(x,changed).ok,false);
});
for(let i=0;i<10;i++)test(`tiny positive capacity hidden by rounded phase entropy ${i}`,()=>{
 const D=BigInt(10)**BigInt(30)-BigInt(i+1),x={...input,probability:`${D+BigInt(1)}/${BigInt(2)*D}`,input:['0','0','0']},p=calculate(x);assert(validate(x,p).ok);
 assert.equal(getAt(p,['result','phaseErrorEntropy','value']),1);assert.equal(getAt(p,['result','quantumCapacityStrictlyPositive']),true);
 const q=range(p,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse']);assert(q.lower.n>BigInt(0));
 const changed=clone(p);setAt(changed,['result','capacitiesByFamilyTheorem','quantumQubitsPerUse'],observe(point(zero)));setAt(changed,['result','xBasisMessageInformation'],observe(point(zero)));assert.equal(validate(x,changed).ok,false);
});
test('environment coherence keeps the input Z sign and is Hermitian',()=>{
 const x={...input,input:['1/5','0','-1/3']},p=calculate(x);assert(validate(x,p).ok);
 const r=range(p,['result','environmentDensity',0,1,'real']);assert(r.upper.n<BigInt(0));
 assert.deepEqual(getAt(p,['result','environmentDensity',0,1]),getAt(p,['result','environmentDensity',1,0]));
 const changed=clone(p);setAt(changed,['result','environmentDensity',1,0,'real'],observe({lower:add(r.lower,rat('1/8')),upper:add(r.upper,rat('1/8'))}));assert.equal(validate(x,changed).ok,false);
});
test('capacity is input-independent and p/1-p symmetric',()=>{
 const a=calculate({...input,input:['0','0','0']}),x={...input,probability:'9/10',input:['0','0','1']},b=calculate(x);assert(validate(x,b).ok);
 assert.deepEqual(getAt(a,['result','capacitiesByFamilyTheorem']),getAt(b,['result','capacitiesByFamilyTheorem']));assert.notDeepEqual(getAt(a,['result','selectedInputCoherentInformation']),getAt(b,['result','selectedInputCoherentInformation']));
});
for(let i=0;i<8;i++)test(`internal output coefficients exceed external input digit budget ${i}`,()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(i+1),x={...input,probability:`1/${D+BigInt(1)}`,input:[`1/${D-BigInt(1)}`,'0','0']},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('invalid mathematical input, malformed responses, getters and cycles fail',()=>{
 const x={...input,input:['1','1','0']},p=clone(reply);setAt(p,['input'],x);assert.equal(validate(x,p).ok,false);
 for(const probability of ['-1/2','3/2'])assert.equal(validate({...input,probability},reply).ok,false);
 let calls=0;const getter={schemaVersion:1,kind:'capacity',input,get result(){calls++;return reply.result;}};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
 const mutable=clone(reply),checked=validate(input,mutable);assert(checked.ok);if(!checked.ok)return;setAt(mutable,['result','quantumCapacityStrictlyPositive'],false);assert.equal(checked.response.result.quantumCapacityStrictlyPositive,true);
});
for(const presets of Object.values(quantumInformationPresets))for(const preset of presets)test(`unified validator accepts draft example ${preset.id}/${preset.input.kind}`,()=>{
 const p=calculate(preset.input),checked=dispatch(preset.input,p);assert(checked.ok,JSON.stringify(checked));
 if(checked.ok){assert(checked.checkedObservations+checked.checkedExactCoefficients>0);assert(Object.isFrozen(checked.response.result));}
 for(const kind of quantumInformationKinds)if(kind!==preset.input.kind){const changed=clone(p);setAt(changed,['kind'],kind);assert.equal(dispatch(preset.input,changed).ok,false);}
});
test('unified request ownership rejects unknown families and getters',()=>{
 let calls=0;assert.equal(dispatch({get kind(){calls++;return 'capacity';}},reply).ok,false);assert.equal(calls,0);
 for(const x of [null,{},[],{kind:'unknown'}])assert.equal(dispatch(x,reply).ok,false);
 assert.equal(dispatch(input,{...reply,input:{...input,probability:'2/20'}}).ok,false);
});
test('all capacity numeric coordinates are checked',()=>{const checked=validate(input,reply);assert(checked.ok);assert.equal(paths.length,77);if(checked.ok)assert.equal(checked.checkedObservations,77);});
test('capacity and unified validation bundles exclude all producer evaluators',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');
 for(const entry of ['capacity-validator','response']){
  const r=await build({entryPoints:[`src/lib/learning-quantum-information-${entry}.ts`],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
  for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|capacity|discrimination|ensemble|entropies|joint-entropy|relative-entropy|bb84|protocols|error-correction|majorization|bell-dilution)\.ts$|learning-qubit-(?:circuit-exact|scalar|channel-algebra)\.ts$|\/fixtures\//.test(p));
  assert(!r.outputFiles[0].text.includes('createTranscendentalContext'));assert(!r.outputFiles[0].text.includes('function sqrtRange'));
 }
});
