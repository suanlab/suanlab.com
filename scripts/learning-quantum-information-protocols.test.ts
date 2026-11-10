import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import reference from './fixtures/quantum-information-protocols-independent.json';
import {teleportQubit,denseCoding,cnotCopyCounterexample,noCloningOverlap} from '../src/lib/learning-quantum-information-protocols';
import {idealBb84InterceptResend} from '../src/lib/learning-quantum-information-bb84';
import {f,one,sub,q,type F} from '../src/lib/learning-quantum-hilbert-core';
import {quadratic,circuitScalar as complex,circuitOne,circuitZero,circuitMultiply,circuitEqual,type CircuitScalar} from '../src/lib/learning-qubit-scalar';
import {identity,tensor,product,adjoint,mAdd,mEqual,zeros,real} from '../src/lib/learning-qubit-channel-algebra';
import {getLearningContent} from '../src/lib/learning-content';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const input=(list:{real:string[];imaginary:string[]}[])=>list.map(z=>complex(quadratic(parse(z.real[0]),parse(z.real[1])),quadratic(parse(z.imaginary[0]),parse(z.imaginary[1]))));
for(const item of reference.teleports)test(`independent dense teleportation: ${item.name}`,()=>{
 const result=teleportQubit(input(item.input));
 for(let j=0;j<4;j++){
  const expected=item.branches[j],branch=result.branches[j];
  for(const key of ['a','b','probability','rawBob','correctedBob','bobDensity','correctedDensity'] as const)assert.deepEqual(branch[key],expected[key]);
 }
 assert.deepEqual(result.bobBeforeMessage,item.bobBeforeMessage);
 assert.deepEqual(result.inputDensity,item.inputDensity);
 assert.doesNotThrow(()=>JSON.stringify(result));
});
for(const[index,item]of reference.copies.entries())test(`independent CNOT clone counterexample: ${item.name}`,()=>{
 const result=cnotCopyCounterexample(input(reference.teleports[index].input));
 for(const key of ['joint','first','second','isExactProductClone','singleCopySquaredFidelity'] as const)assert.deepEqual(result[key],item[key]);
});
for(const item of reference.dense)test(`independent dense coding: ${item.a}${item.b}/${item.order}`,()=>{
 const result=denseCoding(item.a,item.b,item.order as 'x-then-z'|'z-then-x');
 for(const key of ['encodedState','decodedState','decodedDensity'] as const)assert.deepEqual(result[key],item[key]);
 assert.equal(result.decodedIndex,2*item.a+item.b);
 assert.deepEqual(result.bobBeforeTransmission,teleportQubit([circuitOne,circuitZero]).bobBeforeMessage);
 assert.deepEqual(result.decodedProbabilities,Array.from({length:4},(_,i)=>[i===2*item.a+item.b?'1':'0','0']));
});
for(const item of reference.bb84)test(`independent ket/posterior BB84: f=${item.attack},e=${item.eveZ}`,()=>{
 const result=idealBb84InterceptResend(parse(item.attack),parse(item.eveZ));
 assert.equal(result.rows.length,80);
 for(const[index,row]of result.rows.entries()){
  const expected=item.rows[index];
  for(const key of ['aliceBasis','aliceBit','bobBasis','bobBit','attacked','eveBasis','eveBit'] as const)assert.deepEqual(row[key],expected[key]);
  assert.deepEqual(row.weight,q(parse(expected.weight)));
 }
 assert.deepEqual(result.totalProbability,q(one));assert.deepEqual(result.siftProbability,q(parse('1/2')));
 assert.deepEqual(result.siftedQber,q(parse(item.qber)));
 assert.deepEqual(result.eveIdealInformationBits,q(parse(item.eveInformation)));
 assert.deepEqual(result.eveGuessSuccess,q(parse(item.eveGuess)));
 const noError=sub(one,parse(item.qber));
 assert.deepEqual(result.probabilityAtLeastOneSampleError,q(sub(one,f(noError.n**BigInt(20),noError.d**BigInt(20)))));
});
test('complex record11 is corrected in chronological X then Z order',()=>{
 const psi=[complex(quadratic(parse('3/5'))),complex(quadratic(parse('0')),quadratic(parse('4/5')))];
 const result=teleportQubit(psi),branch=result.branches[3];
 assert.deepEqual(branch.correctionGates,[{gate:'X',target:0},{gate:'Z',target:0}]);
 const restored=input(branch.correctedBob),half=complex(quadratic(parse('1/2')));
 assert(restored.every((z,i)=>circuitEqual(z,circuitMultiply(psi[i],half))));
});
test('basis cloning can succeed while nonorthogonal exact cloning is forbidden',()=>{
 const zero=[circuitOne,circuitZero],oneState=[circuitZero,circuitOne],plus=[circuitOne,circuitOne];
 assert(cnotCopyCounterexample(zero).isExactProductClone);
 assert(cnotCopyCounterexample(oneState).isExactProductClone);
 assert(!cnotCopyCounterexample(plus).isExactProductClone);
 assert(noCloningOverlap(zero,oneState).compatibleWithClosedUnitaryCloning);
 assert(!noCloningOverlap(zero,plus).compatibleWithClosedUnitaryCloning);
 assert(noCloningOverlap(plus,plus).compatibleWithClosedUnitaryCloning);
});
test('basis-specific disturbance survives an unchanged aggregate QBER',()=>{
 const result=idealBb84InterceptResend(one,one,0);
 assert.deepEqual(result.basisQber.z,q(parse('0')));
 assert.deepEqual(result.basisQber.x,q(parse('1/2')));
 assert.deepEqual(result.siftedQber,q(parse('1/4')));
 assert.deepEqual(result.probabilityAtLeastOneSampleError,q(parse('0')));
});
test('bad vectors and protocol inputs are rejected before circuit execution',()=>{
 for(const state of [[circuitZero,circuitZero],[circuitOne],new Array(2),[circuitOne,undefined],[complex([{n:BigInt(1),d:BigInt(0)},parse('0')]),circuitOne]]){
  assert.throws(()=>teleportQubit(state as CircuitScalar[]),RangeError);
 }
 for(const pair of [[-1,0],[0,2],[NaN,1],[0.5,0]])assert.throws(()=>denseCoding(pair[0],pair[1]),RangeError);
 for(const p of ['-1/2','3/2'])assert.throws(()=>idealBb84InterceptResend(parse(p)),RangeError);
 for(const n of [-1,257,1.5])assert.throws(()=>idealBb84InterceptResend(one,parse('1/2'),n),RangeError);
});
test('protocol inputs are not mutated',()=>{
 const psi=[circuitOne,complex(quadratic(parse('0')),quadratic(one))];
 const serialize=()=>JSON.stringify(psi,(_k,v)=>typeof v==='bigint'?String(v):v),before=serialize();
 teleportQubit(psi);cnotCopyCounterexample(psi);noCloningOverlap(psi,[circuitOne,circuitZero]);
 assert.equal(serialize(),before);
});
for(const id of ['no-cloning-bb84-information-disturbance','teleportation-superdense-coding-resources'])test(`bilingual protocol chapter: ${id}`,async()=>{
 const content=await getLearningContent(id);
 for(const locale of ['ko','en'] as const){assert.equal(content.outline[locale].length,10);assert(!content[locale].includes('katex-error'));assert(content[locale].includes('<table'));}
 assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`,'utf8')));
});

test('corrected branch operators preserve a reference-entangled Bell input',()=>{
 const columns=[teleportQubit([circuitOne,circuitZero]),teleportQubit([circuitZero,circuitOne])];
 const rho=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>(i===0||i===3)&&(j===0||j===3)?real(1,2):circuitZero));
 let output=zeros(4);
 for(let branch=0;branch<4;branch++){
  const first=input(columns[0].branches[branch].correctedBob),second=input(columns[1].branches[branch].correctedBob);
  const K=[[first[0],second[0]],[first[1],second[1]]];
  assert(mEqual(K,[[real(1,2),circuitZero],[circuitZero,real(1,2)]]));
  const full=tensor(identity(2),K);
  output=mAdd(output,product(product(full,rho),adjoint(full)));
 }
 assert(mEqual(output,rho));
});
