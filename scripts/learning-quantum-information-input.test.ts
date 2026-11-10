import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire}from'node:module';
import {readQuantumInformationInput as read,snapshotQuantumInformationInput as snapshot,quantumInformationInputKey as key,parseQuantumInformationDraft as draft,parseQuantumInformationStoredInput as stored,quantumInformationKinds,type QuantumInformationInput,type InformationState}from'../src/lib/learning-quantum-information-input';
import {calculateQuantumInformation as calculate}from'../src/lib/learning-quantum-information-calculation';
import discrimination from './fixtures/quantum-discrimination-independent.json';
import entropy from './fixtures/quantum-information-entropy-independent.json';
import protocols from './fixtures/quantum-information-protocols-independent.json';
import resources from './fixtures/quantum-information-resources-independent.json';
import capacities from './fixtures/quantum-information-capacity-independent.json';
import joint from './fixtures/quantum-joint-entropy-independent.json';
import {f,add,sub,mul,le,type F}from'../src/lib/learning-quantum-hilbert-core';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
const vector=(xs:string[])=>[xs[0],xs[1],xs[2]]as const;
const scalar=(re='0',im='0',root='0',rootIm='0')=>({real:[re,root],imaginary:[im,rootIm]}as const);
const state=(xs:{real:string[];imaginary:string[]}[]):InformationState=>[scalar(xs[0].real[0],xs[0].imaginary[0],xs[0].real[1],xs[0].imaginary[1]),scalar(xs[1].real[0],xs[1].imaginary[0],xs[1].real[1],xs[1].imaginary[1])];
const psi=[scalar('3/5'),scalar('0','4/5')]as const,other=[scalar('1'),scalar()]as const;
const examples:QuantumInformationInput[]=[
 {kind:'discrimination',bloch0:['0','0','1'],bloch1:['24/25','0','-7/25'],prior:'1/2'},
 {kind:'entropy',correlationWeight:'1',coherence:'1',probabilityOne:'1/5',copies:2,retainedRank:'3'},
 {kind:'ensemble',bloch0:['0','0','1'],bloch1:['24/25','0','-7/25'],prior:'1/2',measurement:'helstrom'},
 {kind:'relative-entropy',rho:['0','0','1'],sigma:['1','0','0']},
 {kind:'bb84',input:psi,comparison:other,attackFraction:'1',eveZProbability:'1/2',sampleSize:20},
 {kind:'teleportation',input:psi,a:1,b:1,order:'z-then-x'},
 {kind:'error-correction',input:psi,selectedError:'XII',physicalXProbability:'1/10',errors:['III','XII','IXI','IIX']},
 {kind:'majorization',mode:'compare',source:['1/2','1/4','1/4'],target:['2/5','2/5','1/5']},
 {kind:'majorization',mode:'bell-dilution',a:'4/5',b:'3/5'},
 {kind:'capacity',probability:'1/10',input:['1/5','2/5','1/5']},
];
for(const example of examples)test(`owned/draft/stored roundtrip: ${example.kind}${example.kind==='majorization'?'/'+example.mode:''}`,()=>{
 const own=snapshot(example);assert.deepEqual(own,example);assert.notEqual(own,example);assert(Object.isFrozen(own));
 const visit=(x:unknown)=>{if(x&&typeof x==='object'){assert(Object.isFrozen(x));for(const v of Object.values(x))visit(v);}};visit(own);
 assert.deepEqual(draft(JSON.stringify(example)),own);assert.deepEqual(stored(JSON.stringify({version:1,value:example})),own);
 const result=calculate(own);assert.equal(result.kind,own.kind);assert.equal(result.schemaVersion,1);assert.deepEqual(result.input,own);assert.doesNotThrow(()=>JSON.stringify(result));
 assert.equal(read({...example,unknown:true}),null);
});
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
function decode(x:PartitionRational):F{const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));}
const sign=(x:F)=>x.n>BigInt(0)?1:x.n<BigInt(0)?-1:0;
function compareRoot(xs:string[],bound:F){const[a,b,c]=xs.map(parse),A=sub(a,bound),sa=sign(A),sb=sign(b);if(!sb||!sign(c))return sa;if(!sa||sa===sb)return sb;const d=sign(sub(mul(A,A),mul(mul(b,b),c)));return d===0?0:d>0?sa:sb;}
function certificate(obs:PartitionObservation,expected:string[]){assert(obs.bounds);assert(compareRoot(expected,decode(obs.bounds.lower))>=0);assert(compareRoot(expected,decode(obs.bounds.upper))<=0);}
function decimal(s:string):F{const[m,e='0']=s.toLowerCase().split('e'),negative=m.startsWith('-'),[a,b='']=m.replace(/^[-+]/,'').split('.'),n=BigInt(a+b)*(negative?BigInt(-1):BigInt(1)),k=Number(e)-b.length;return k>=0?f(n*BigInt(10)**BigInt(k)):f(n,BigInt(10)**BigInt(-k));}
function decimalReference(obs:PartitionObservation,expected:string){assert(obs.bounds);const center=decimal(expected),allowance=decimal('1e-110');assert(le(decode(obs.bounds.lower),add(center,allowance)));assert(le(sub(center,allowance),decode(obs.bounds.upper)));}
for(const r of discrimination.cases)test(`wire decode -> independent discrimination: ${r.name}`,()=>{
 const out=calculate({kind:'discrimination',bloch0:vector(r.bloch0),bloch1:vector(r.bloch1),prior:r.prior});assert.equal(out.kind,'discrimination');if(out.kind!=='discrimination')return;
 assert.equal(out.result.strategy,r.strategy);assert.equal(out.result.uniqueOptimalEffect,r.uniqueOptimalEffect);
 for(const k of Object.keys(r.scalarCertificates)as(keyof typeof r.scalarCertificates)[])certificate(out.result[k],r.scalarCertificates[k]);
 for(let i=0;i<4;i++){assert(out.result.optimalGuessZeroEffect);certificate(out.result.optimalGuessZeroEffect[Math.floor(i/2)][i%2].real,r.effect[i][0]);certificate(out.result.optimalGuessZeroEffect[Math.floor(i/2)][i%2].imaginary,r.effect[i][1]);}
});
for(const r of joint.cases)test(`wire decode -> independent joint entropy: ${r.c}/${r.t}`,()=>{
 const out=calculate({kind:'entropy',correlationWeight:r.c,coherence:r.t,probabilityOne:'1/5',copies:2,retainedRank:'3'});assert.equal(out.kind,'entropy');if(out.kind!=='entropy')return;
 assert.equal(out.result.joint.entangled,r.entangled);decimalReference(out.result.joint.jointEntropy,r.jointEntropy);
 certificate(out.result.block.retainedWeight,['24/25','0','0']);
});
for(const r of entropy.ensembles)test(`wire decode -> independent ensemble: ${r.name}`,()=>{
 const out=calculate({kind:'ensemble',bloch0:vector(r.bloch0),bloch1:vector(r.bloch1),prior:r.prior,measurement:r.measurement});assert.equal(out.kind,'ensemble');if(out.kind!=='ensemble')return;
 for(const k of Object.keys(r.expected)as(keyof typeof r.expected)[])decimalReference(out.result[k],r.expected[k]);
});
for(const r of entropy.relative)test(`wire decode -> independent divergence: ${r.name}`,()=>{
 const out=calculate({kind:'relative-entropy',rho:vector(r.rho),sigma:vector(r.sigma)});assert.equal(out.kind,'relative-entropy');if(out.kind!=='relative-entropy')return;
 assert.equal(out.result.before.status,r.status);if(r.status==='finite'){assert(out.result.before.value);decimalReference(out.result.before.value,r.value!);}
});
for(const r of protocols.bb84)test(`wire decode -> independent BB84: ${r.attack}/${r.eveZ}`,()=>{
 const out=calculate({...examples[4],attackFraction:r.attack,eveZProbability:r.eveZ});assert.equal(out.kind,'bb84');if(out.kind!=='bb84')return;
 certificate(out.result.attack.siftedQber,[r.qber,'0','0']);certificate(out.result.attack.eveIdealInformationBits,[r.eveInformation,'0','0']);certificate(out.result.attack.eveGuessSuccess,[r.eveGuess,'0','0']);
 assert.deepEqual(out.result.overlap.squaredOverlap.real,['9/25','0']);
 assert.deepEqual(out.result.cloning.singleCopySquaredFidelity.real,['337/625','0']);
});
for(const r of protocols.teleports)test(`wire decode -> independent teleportation: ${r.name}`,()=>{
 const out=calculate({...examples[5],input:state(r.input)});assert.equal(out.kind,'teleportation');if(out.kind!=='teleportation')return;
 for(let j=0;j<4;j++){assert.deepEqual(out.result.teleportation.branches[j].rawBob,r.branches[j].rawBob);assert.deepEqual(out.result.teleportation.branches[j].correctedDensity,r.branches[j].correctedDensity);}
 assert.deepEqual(out.result.teleportation.bobBeforeMessage,r.bobBeforeMessage);
});
for(const r of resources.cases)test(`wire decode -> independent syndrome: ${r.name}`,()=>{
 const out=calculate({...examples[6],input:state(r.input),selectedError:r.error});assert.equal(out.kind,'error-correction');if(out.kind!=='error-correction')return;
 assert.deepEqual(out.result.experiment.branches,r.branches);assert.deepEqual(out.result.experiment.erroredDensity,r.erroredDensity);assert(out.result.condition.conditionHolds);
});
for(const[index,r]of resources.majorization.entries())test(`wire decode -> independent convex hull: ${index}`,()=>{
 const out=calculate({kind:'majorization',mode:'compare',source:r.source,target:r.target});assert.equal(out.kind,'majorization');if(out.kind!=='majorization'||out.result.mode!=='compare')return;
 assert.equal(out.result.comparison.forward.deterministic,r.forward);certificate(out.result.comparison.forward.optimalStochasticSuccess,[r.pForward,'0','0']);
});
for(const r of capacities.cases)test(`wire decode -> independent capacity: ${r.name}`,()=>{
 const out=calculate({kind:'capacity',probability:r.p,input:vector(r.bloch)});assert.equal(out.kind,'capacity');if(out.kind!=='capacity')return;
 decimalReference(out.result.capacitiesByFamilyTheorem.quantumQubitsPerUse,r.quantumRate);decimalReference(out.result.selectedInputCoherentInformation,r.expected.selectedInputCoherentInformation);
});

test('caller mutations cannot change the owned request or calculation',()=>{
 const raw=JSON.parse(JSON.stringify(examples[5])),owned=snapshot(raw),before=key(owned);
 raw.input[0].real[0]='99';raw.a=0;raw.input[1].imaginary[0]='0';
 assert.equal(key(owned),before);assert.notEqual(key(raw),before);
 const out=calculate(owned);assert.equal(out.kind,'teleportation');if(out.kind!=='teleportation')return;
 const independent=protocols.teleports.find(r=>r.input[0].real[0]==='3/5'&&r.input[1].imaginary[0]==='4/5');assert(independent);
 assert.deepEqual(out.result.teleportation.inputDensity,independent.inputDensity);
});
test('getter and toJSON methods are never invoked',()=>{
 let calls=0;const getter={get kind(){calls++;return'capacity';}};assert.equal(read(getter),null);
 const input={...examples[9],get probability(){calls++;return'1/2';}};assert.equal(read(input),null);
 const s={get real(){calls++;return['1','0'];},imaginary:['0','0']};assert.equal(read({...examples[5],input:[s,scalar()]}),null);
 const extra={...examples[9],toJSON(){calls++;return examples[9];}};assert.equal(read(extra),null);
 assert.equal(calls,0);
});
test('dense arrays reject holes, accessors, symbols and additional fields',()=>{
 const candidates:unknown[]=[new Array(3),['0','0'],['0','0','0','0']];
 const accessor=['0','0','0'];let calls=0;Object.defineProperty(accessor,'1',{get(){calls++;return'0';}});candidates.push(accessor);
 const extra=Object.assign(['0','0','0'],{extra:'0'});candidates.push(extra);
 const symbol=['0','0','0'];Object.defineProperty(symbol,Symbol('x'),{value:1});candidates.push(symbol);
 for(const candidate of candidates)assert.equal(read({...examples[9],input:candidate}),null);
 assert.equal(calls,0);
});
test('unexpected prototypes, prototype keys, cycles and unknown variants fail',()=>{
 const nullRecord=Object.assign(Object.create(null),examples[9]);assert.equal(read(nullRecord),null);
 const inherited=Object.create(examples[9]);assert.equal(read(inherited),null);
 assert.equal(read(JSON.parse('{"kind":"capacity","probability":"1/2","input":["0","0","0"],"__proto__":{}}')),null);
 assert.equal(read({...examples[9],kind:'toString'}),null);
 const cycle:unknown[]=[];cycle.push(cycle,'0','0');assert.equal(read({...examples[9],input:cycle}),null);
 assert.equal(read({...examples[8],mode:'unknown'}),null);
});
test('exact fraction grammar rejects rounded numbers, exponents and ambiguous spellings',()=>{
 for(const value of ['0.5','1e-30','NaN','Infinity','+1','-0','00','01/2','1/02','1/0','1/-2',' 1/2 ','1//2','1/2x','1'.repeat(33),0.5,null])assert.equal(read({...examples[9],probability:value}),null);
 assert(read({...examples[9],probability:'0/7'}));assert(read({...examples[9],probability:'2/4'}));
});
test('raw equivalent fractions keep different request identities',()=>{
 const a={...examples[9],probability:'1/2'},b={...examples[9],probability:'2/4'};
 assert.notEqual(key(a),key(b));const ca=calculate(a),cb=calculate(b);
 assert.equal(ca.kind,'capacity');assert.equal(cb.kind,'capacity');if(ca.kind==='capacity'&&cb.kind==='capacity')assert.deepEqual(ca.result,cb.result);
});
test('shape-valid but nonphysical inputs fail in the mathematical stage',()=>{
 const nonphysical={...examples[9],input:['1','1','0']};assert(read(nonphysical));assert.throws(()=>calculate(nonphysical),RangeError);
 const zero={...examples[5],input:[scalar(),scalar()]};assert(read(zero));assert.throws(()=>calculate(zero),RangeError);
 const spectrum={kind:'majorization',mode:'compare',source:['1/4','1/4'],target:['1']};assert(read(spectrum));assert.throws(()=>calculate(spectrum),RangeError);
 const dilution={kind:'majorization',mode:'bell-dilution',a:'1/2',b:'1/2'};assert(read(dilution));assert.throws(()=>calculate(dilution),RangeError);
});
test('every block rank boundary is compared without Number rounding',()=>{
 for(let copies=1;copies<=64;copies++){
  const maximum=BigInt(1)<<BigInt(copies);
  for(const rank of [BigInt(0),maximum-BigInt(1),maximum])assert(read({...examples[1],copies,retainedRank:String(rank)}));
  assert.equal(read({...examples[1],copies,retainedRank:String(maximum+BigInt(1))}),null);
 }
 assert.equal(read({...examples[1],copies:64,retainedRank:18446744073709551616}),null);
});
test('JSON rejects duplicate decoded keys and lossy numeric spellings',()=>{
 const valid=JSON.stringify(examples[1]);
 for(const token of ['2.0','2e0','-0','9007199254740993','1e400'])assert.equal(draft(valid.replace('"copies":2',`"copies":${token}`)),null);
 assert.equal(draft('{"kind":"capacity","kind":"capacity","probability":"0","input":["0","0","0"]}'),null);
 assert.equal(draft('{"kind":"capacity","\\u006bind":"capacity","probability":"0","input":["0","0","0"]}'),null);
 assert.equal(draft(JSON.stringify(examples[5]).replace('"real":["3/5","0"]','"real":["3/5","0"],"\\u0072eal":["3/5","0"]')),null);
 assert.equal(stored(JSON.stringify({version:2,value:examples[9]})),null);
 assert.equal(stored('{"version":1,"version":1,"value":{}}'),null);
 assert.equal(draft(' '.repeat(32769)),null);
});
test('array/probability/integer budgets fail before mathematical allocation',()=>{
 assert.equal(read({...examples[7],source:Array(17).fill('1/17')}),null);
 assert.equal(read({...examples[6],errors:Array(17).fill('III')}),null);
 assert.equal(read({...examples[6],selectedError:'XXXX'}),null);
 for(const sampleSize of [-1,-0,1.5,257,Infinity])assert.equal(read({...examples[4],sampleSize}),null);
 for(const prior of ['-1/2','3/2'])assert.equal(read({...examples[0],prior}),null);
 assert.equal(read({...examples[5],a:2}),null);
});
test('all nine families and both resource modes are represented',()=>{
 assert.deepEqual(new Set(examples.map(x=>x.kind)),new Set(quantumInformationKinds));
 assert.equal(examples.filter(x=>x.kind==='majorization').length,2);
});
test('reader ownership bundle has no numerical engine or runtime imports',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild')as typeof import('esbuild');
 const bundled=await build({entryPoints:['src/lib/learning-quantum-information-input.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,minify:true});
 assert.deepEqual(Object.keys(bundled.metafile!.inputs),['src/lib/learning-quantum-information-input.ts']);
 assert(!/\bBigInt\s*\(/.test(bundled.outputFiles![0].text));
});

for(const r of protocols.dense)test(`wire decode -> independent dense coding: ${r.a}${r.b}/${r.order}`,()=>{
 const out=calculate({...examples[5],a:r.a,b:r.b,order:r.order});assert.equal(out.kind,'teleportation');if(out.kind!=='teleportation')return;
 assert.deepEqual(out.result.denseCoding.decodedState,r.decodedState);
});
for(const r of resources.kl)test(`wire decode -> independent KL family: ${r.words.join(',')}`,()=>{
 const out=calculate({...examples[6],errors:r.words});assert.equal(out.kind,'error-correction');if(out.kind!=='error-correction')return;
 assert.equal(out.result.condition.conditionHolds,r.conditionHolds);
 for(const[index,pair]of out.result.condition.pairs.entries())assert.deepEqual(pair.compressed,r.pairs[index].compressed);
});
test('the source block rank reaches the mathematical engine without losing a bit',()=>{
 const rank=(BigInt(1)<<BigInt(63))+BigInt(1),out=calculate({...examples[1],copies:64,probabilityOne:'1/2',retainedRank:String(rank)});
 assert.equal(out.kind,'entropy');if(out.kind!=='entropy')return;
 certificate(out.result.block.retainedWeight,[`${rank}/18446744073709551616`,'0','0']);
});
test('the divergence request preserves actual/reference order and processes both states',()=>{
 const out=calculate(examples[3]);assert.equal(out.kind,'relative-entropy');if(out.kind!=='relative-entropy')return;
 assert.equal(out.result.before.status,'infinite');assert.equal(out.result.after.status,'finite');assert(out.result.after.value);
 certificate(out.result.after.value,['1','0','0']);
});

import {quantumInformationPresets}from'../src/lib/learning-quantum-information-presets';
import {quantumInformationDraftIds}from'../src/lib/learning-quantum-information-topics';
for(const[id,presets]of Object.entries(quantumInformationPresets))for(const preset of presets)test(`teaching input remains valid: ${id}/${preset.id}`,()=>{
 assert(Object.isFrozen(preset));assert(Object.isFrozen(preset.input));assert(Object.isFrozen(preset.label));
 assert(preset.label.ko.length>0&&preset.label.en.length>0);assert(!/[가-힣]/.test(preset.label.en));
 const out=calculate(preset.input);assert.equal(out.kind,preset.input.kind);assert.deepEqual(out.input,preset.input);
});
test('draft teaching inventory covers every unit with unique example identities',()=>{
 assert.deepEqual(Object.keys(quantumInformationPresets),Array.from(quantumInformationDraftIds));
 for(const[index,id]of quantumInformationDraftIds.entries()){
  const presets=quantumInformationPresets[id];assert(Object.isFrozen(presets));assert(presets.length>=4);
  assert.equal(new Set(presets.map(p=>p.id)).size,presets.length);
  assert(presets.every(p=>p.input.kind===quantumInformationKinds[index]));
 }
 assert.equal(Object.values(quantumInformationPresets).reduce((n,ps)=>n+ps.length,0),44);
});
test('teaching preset browser bundle contains only reader-safe input modules',async()=>{
 const require=createRequire(import.meta.url),{build}=require('esbuild')as typeof import('esbuild');
 const bundled=await build({entryPoints:['src/lib/learning-quantum-information-presets.ts'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,minify:true});
 assert.deepEqual(new Set(Object.keys(bundled.metafile!.inputs)),new Set(['src/lib/learning-quantum-information-input.ts','src/lib/learning-quantum-information-presets.ts']));
 assert(!/\bBigInt\s*\(/.test(bundled.outputFiles![0].text));
});
