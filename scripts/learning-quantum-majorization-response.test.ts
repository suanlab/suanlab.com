import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import reference from './fixtures/quantum-information-resources-independent.json';
import {calculateQuantumInformation as calculate} from '../src/lib/learning-quantum-information-calculation';
import {validateQuantumMajorizationResponse as validate} from '../src/lib/learning-quantum-information-majorization-validator';
import {fieldRational as rat,fieldRationalWire} from '../src/lib/learning-quantum-information-reference-field';
import {f,zero,one,add,sub,observe,type F,type R} from '../src/lib/learning-quantum-hilbert-core';
import {binaryProduct} from '../src/lib/learning-binary-products';
import type {PartitionObservation,PartitionRational} from '../src/lib/learning-partition-exact-core';
const clone=<T>(x:T):T=>JSON.parse(JSON.stringify(x));
const input={kind:'majorization',mode:'compare',source:['1/2','1/4','1/4'],target:['2/5','2/5','1/5']},reply=calculate(input);
const dilutionInput={kind:'majorization',mode:'bell-dilution',a:'4/5',b:'3/5'},dilutionReply=calculate(dilutionInput);
type Path=readonly(string|number)[];
function getAt(value:unknown,path:Path):unknown{let x=value;for(const key of path){assert(x&&typeof x==='object');x=(x as Record<string,unknown>)[String(key)];}return x;}
function setAt(value:unknown,path:Path,entry:unknown){const p=getAt(value,path.slice(0,-1));assert(p&&typeof p==='object');(p as Record<string,unknown>)[String(path[path.length-1])]=entry;}
function decode(x:PartitionRational):F{const n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient),s=x.numerator.exponent-x.denominator.exponent;return s>=0?f(n<<BigInt(s),d):f(n,d<<BigInt(-s));}
function range(value:unknown,path:Path):R{const x=getAt(value,path)as PartitionObservation;assert(x.bounds);return{lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};}
function observations(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(!value||typeof value!=='object')return[];if(Object.hasOwn(value,'bounds')&&Object.hasOwn(value,'resolution'))return[path];return Object.entries(value).flatMap(([k,v])=>observations(v,[...path,Array.isArray(value)?Number(k):k]));}
function coefficients(value:unknown,path:(string|number)[]=[]):(string|number)[][]{if(typeof value==='string'&&/^-?\d+(?:\/\d+)?$/.test(value))return[path];if(!value||typeof value!=='object')return[];return Object.entries(value).flatMap(([k,v])=>coefficients(v,[...path,Array.isArray(value)?Number(k):k]));}
for(const[i,r]of reference.majorization.entries())test(`permutation convex hull comparison ${i}`,()=>{
 const x={...input,source:r.source,target:r.target},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 assert.equal(getAt(p,['result','comparison','forward','deterministic']),r.forward);assert.equal(getAt(p,['result','comparison','reverse','deterministic']),r.reverse);
 assert.deepEqual(range(p,['result','comparison','forward','optimalStochasticSuccess']),{lower:rat(r.pForward),upper:rat(r.pForward)});
 assert.deepEqual(range(p,['result','comparison','reverse','optimalStochasticSuccess']),{lower:rat(r.pReverse),upper:rat(r.pReverse)});
 if(checked.ok)assert(Object.isFrozen(checked.response.result));
});
const paths=observations(reply.result,['result']);
for(const path of paths)test(`reject false majorization interval ${path.join('.')}`,()=>{
 const changed=clone(reply),old=range(changed,path);setAt(changed,path,observe({lower:add(old.lower,rat('1/8')),upper:add(old.upper,rat('1/8'))}));assert.equal(validate(input,changed).ok,false);
});
for(const path of paths)test(`reject majorization display corruption ${path.join('.')}`,()=>{
 const changed=clone(reply),old=getAt(changed,path)as PartitionObservation;setAt(changed,path,{...old,value:old.value===null?0:old.value+0.125});assert.equal(validate(input,changed).ok,false);
});
const dilutionPaths=coefficients(dilutionReply.result,['result']);
for(const path of dilutionPaths)test(`reject Bell dilution coefficient ${path.join('.')}`,()=>{
 const changed=clone(dilutionReply),old=getAt(changed,path);assert(typeof old==='string');setAt(changed,path,fieldRationalWire(add(rat(old),one)));assert.equal(validate(dilutionInput,changed).ok,false);
});
for(const[name,path,value]of[
 ['version',['schemaVersion'],2],['family',['kind'],'capacity'],['mode',['result','mode'],'bell-dilution'],['raw spelling',['input','source',0],'2/4'],
 ['dimension',['result','comparison','dimension'],4],['source sorting',['result','comparison','sourceSortingIndices'],[1,0,2]],['target sorting',['result','comparison','targetSortingIndices'],[1,0,2]],
 ['source rank',['result','comparison','sourceSchmidtRank'],2],['target rank',['result','comparison','targetSchmidtRank'],2],['relation',['result','comparison','relation'],'equivalent'],
 ['forward',['result','comparison','forward','deterministic'],true],['reverse',['result','comparison','reverse','deterministic'],true],
 ['first violation',['result','comparison','forward','firstViolation'],2],['reverse violation',['result','comparison','reverse','firstViolation'],1],
 ['prefix k',['result','comparison','forward','prefixes',0,'k'],2],['prefix flag',['result','comparison','forward','prefixes',0,'satisfied'],true],
 ['tail first',['result','comparison','forward','tails',0,'first'],2],['tail applies',['result','comparison','forward','tails',0,'applicable'],false],
 ['scope',['result','comparison','contract'],'general mixed-state LOCC solver'],['extra',['result','extra'],0],
]as [string,Path,unknown][])test(`reject majorization metadata ${name}`,()=>{const changed=clone(reply);setAt(changed,path,value);assert.equal(validate(input,changed).ok,false);});
test('incomparable states have nonreciprocal optimal rates and ratios above one',()=>{
 assert(validate(input,reply).ok);assert.equal(getAt(reply,['result','comparison','relation']),'incomparable');
 assert.deepEqual(range(reply,['result','comparison','forward','optimalStochasticSuccess']),{lower:rat('5/6'),upper:rat('5/6')});
 assert.deepEqual(range(reply,['result','comparison','reverse','optimalStochasticSuccess']),{lower:rat('4/5'),upper:rat('4/5')});
 assert.deepEqual(range(reply,['result','comparison','forward','tails',2,'ratio']),{lower:rat('5/4'),upper:rat('5/4')});
});
test('stable sorting, unequal dimensions and zero target tails are explicit',()=>{
 const x={...input,source:['0','1/4','1/2','1/4'],target:['1','0']},p=calculate(x);assert(validate(x,p).ok);
 assert.deepEqual(getAt(p,['result','comparison','sourceSortingIndices']),[2,1,3,0]);assert.deepEqual(getAt(p,['result','comparison','targetSortingIndices']),[0,1]);
 for(const i of [1,2,3]){assert.equal(getAt(p,['result','comparison','forward','tails',i,'ratio']),null);assert.equal(getAt(p,['result','comparison','forward','tails',i,'applicable']),false);}
 const changed=clone(p);setAt(changed,['result','comparison','forward','tails',1,'ratio'],observe({lower:one,upper:one}));assert.equal(validate(x,changed).ok,false);
});
for(const [source,target]of[
 [['1'],Array(16).fill('1/16')], [Array(16).fill('1/16'),['1']],
 [Array(16).fill('1/16'),Array(8).fill('1/8')], [Array(8).fill('1/8'),Array(16).fill('1/16')],
 [Array(16).fill('1/16'),Array(16).fill('1/16')],
 [['0','0','1'],['1']],
])test(`dimension/rank boundary ${source.length}/${target.length}/${source[0]}`,()=>{
 const x={...input,source,target},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 if(source.length===16)assert.equal(getAt(p,['result','comparison','sourceEntropy','value']),4);
});
for(let i=0;i<8;i++)test(`tiny exact prefix violation ${i}`,()=>{
 const D=BigInt(10)**BigInt(30)-BigInt(i),x={...input,source:[`${D+BigInt(1)}/${BigInt(2)*D}`,`${D-BigInt(1)}/${BigInt(2)*D}`],target:['1/2','1/2']},p=calculate(x);assert(validate(x,p).ok);
 assert.equal(getAt(p,['result','comparison','forward','deterministic']),false);assert.equal(getAt(p,['result','comparison','forward','firstViolation']),1);
 assert.equal(getAt(p,['result','comparison','forward','prefixes',0,'source','value']),getAt(p,['result','comparison','forward','prefixes',0,'target','value']));
 const changed=clone(p);setAt(changed,['result','comparison','forward','prefixes',0,'targetMinusSource'],observe({lower:zero,upper:zero}));assert.equal(validate(x,changed).ok,false);
});
test('tiny entropy and tiny optimal stochastic rate cannot become zero',()=>{
 const D=BigInt(10)**BigInt(31)-BigInt(1),x={...input,source:[`${D-BigInt(1)}/${D}`,`1/${D}`],target:['1/2','1/2']},p=calculate(x);assert(validate(x,p).ok);
 for(const path of [['result','comparison','sourceEntropy'],['result','comparison','forward','optimalStochasticSuccess']]){
  const changed=clone(p);setAt(changed,path,observe({lower:zero,upper:zero}));assert.equal(validate(x,changed).ok,false);
 }
});
test('entropy is a numerical enclosure and does not determine the partial order',()=>{
 const changed=clone(reply),path=['result','comparison','targetEntropy'],old=range(changed,path),margin=f(BigInt(1),BigInt(1)<<BigInt(240));
 setAt(changed,path,observe({lower:sub(old.lower,margin),upper:add(old.upper,margin)}));const checked=validate(input,changed);assert(checked.ok,JSON.stringify(checked));
 const number=getAt(reply,[...path,'value']);assert(typeof number==='number');const b=binaryProduct(number),r=b.exponent>=0?f(b.coefficient<<BigInt(b.exponent)):f(b.coefficient,BigInt(1)<<BigInt(-b.exponent));
 setAt(changed,path,observe({lower:r,upper:r}));assert.equal(validate(input,changed).ok,false);
});
test('a conservative interval around maximal entropy may straddle four bits',()=>{
 const x={...input,source:Array(16).fill('1/16'),target:Array(16).fill('1/16')},p=calculate(x),margin=f(BigInt(1),BigInt(1)<<BigInt(240)),four=f(BigInt(4));
 setAt(p,['result','comparison','sourceEntropy'],observe({lower:sub(four,margin),upper:add(four,margin)}));
 const checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));if(checked.ok)assert.equal(checked.checkedObservations,228);
});
test('product-state entropy requires exact zero',()=>{
 const x={...input,source:['1'],target:['1']},p=calculate(x);assert(validate(x,p).ok);
 const changed=clone(p);setAt(changed,['result','comparison','sourceEntropy'],observe({lower:zero,upper:f(BigInt(1),BigInt(1)<<BigInt(240))}));assert.equal(validate(x,changed).ok,false);
});
for(const [a,b]of [['4/5','3/5'],['3/5','4/5'],['12/13','5/13'],['24/25','7/25'],['1','0'],['0','1']])test(`explicit Bell dilution ${a}/${b}`,()=>{
 const x={...dilutionInput,a,b},p=calculate(x),checked=validate(x,p);assert(checked.ok,JSON.stringify(checked));
 assert.deepEqual(getAt(p,['result','dilution','branches',0,'corrected']),getAt(p,['result','dilution','branches',1,'corrected']));
 for(const i of [0,1])assert.deepEqual(getAt(p,['result','dilution','branches',i,'correctedDensity']),getAt(p,['result','dilution','targetDensity']));
});
for(const[name,path,value]of[
 ['outcome',['result','dilution','branches',1,'outcome'],0],['communication correction',['result','dilution','branches',1,'bothPartiesApplyX'],false],
 ['scope',['result','dilution','contract'],'synthesizes every optimal protocol'],['mode',['result','mode'],'compare'],['raw spelling',['input','a'],'8/10'],
]as [string,Path,unknown][])test(`reject dilution metadata ${name}`,()=>{const changed=clone(dilutionReply);setAt(changed,path,value);assert.equal(validate(dilutionInput,changed).ok,false);});
for(let i=0;i<5;i++)test(`large rational unit-circle Bell amplitudes ${i}`,()=>{
 const t=BigInt(10)**BigInt(15)-BigInt(i+1),d=t*t+BigInt(1),x={...dilutionInput,a:`${t*t-BigInt(1)}/${d}`,b:`${BigInt(2)*t}/${d}`},checked=validate(x,calculate(x));assert(checked.ok,JSON.stringify(checked));
});
test('spectra and modes remain independently request-bound',()=>{
 assert.equal(validate({...input,source:input.target,target:input.source},reply).ok,false);assert.equal(validate(dilutionInput,reply).ok,false);
 const changed=clone(reply);setAt(changed,['input','source'],input.target);setAt(changed,['input','target'],input.source);assert.equal(validate({...input,source:input.target,target:input.source},changed).ok,false);
});
test('invalid probability sums and Bell normalization rejected',()=>{
 for(const x of [{...input,source:['1/2']},{...input,target:['1/2','1/3']},{...input,source:Array(17).fill('1/17')},{...input,source:['-1','2']},{...dilutionInput,a:'1/2',b:'1/2'},{...dilutionInput,a:'0',b:'0'}]){const changed=clone(x.mode==='compare'?reply:dilutionReply);setAt(changed,['input'],x);assert.equal(validate(x,changed).ok,false);}
 // Use an envelope bound to an invalid mathematical input as well.
 const x={...dilutionInput,a:'1/2',b:'1/2'},changed=clone(dilutionReply);setAt(changed,['input'],x);assert.equal(validate(x,changed).ok,false);
});
test('descriptor ownership rejects getters/cycles and freezes valid packets',()=>{
 let calls=0;const getter={schemaVersion:1,kind:'majorization',input,get result(){calls++;return reply.result;}};assert.equal(validate(input,getter).ok,false);assert.equal(calls,0);
 const cycle=clone(reply);setAt(cycle,['extra'],cycle);assert.equal(validate(input,cycle).ok,false);
 const sparse=clone(reply);delete(getAt(sparse,['result','comparison','source'])as unknown[])[0];assert.equal(validate(input,sparse).ok,false);
 const mutable=clone(reply),checked=validate(input,mutable);assert(checked.ok);if(!checked.ok)return;
 setAt(mutable,['result','comparison','relation'],'equivalent');assert.equal(getAt(checked.response,['result','comparison','relation']),'incomparable');
});
test('all numerical coordinates for both modes are checked',()=>{
 assert.equal(paths.length,46);assert.equal(dilutionPaths.length,324);
 const a=validate(input,reply),b=validate(dilutionInput,dilutionReply);assert(a.ok&&b.ok);if(a.ok&&b.ok){assert.equal(a.checkedObservations,46);assert.equal(b.checkedExactCoefficients,324);}
});
test('emitted validator excludes producer sorting, entropy, circuit and LOCC engines',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild');const r=await build({entryPoints:['src/lib/learning-quantum-information-majorization-validator.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 for(const p of Object.keys(r.metafile.inputs))assert(!/learning-quantum-information-(?:calculation|majorization|bell-dilution|entropies|protocols)\.ts$|learning-qubit-(?:circuit-exact|scalar|channel-algebra)\.ts$|\/fixtures\//.test(p));
 assert(!r.outputFiles[0].text.includes('createTranscendentalContext'));assert(!r.outputFiles[0].text.includes('function pureStateMajorization'));
});
