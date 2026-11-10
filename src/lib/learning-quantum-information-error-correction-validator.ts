/** Independent bit/phase Pauli action, code-image inner products and closed
 * majority-noise channel. No dense producer operator or recovery engine. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, one, add, sub, mul, eq, type F } from './learning-quantum-hilbert-core';
import { fieldRational as rat, fieldPair, fieldComplex, fieldZero, fieldOne, fieldAdd, fieldConjugate, fieldMultiply, fieldEqual, referenceFieldVector, fieldDensity, fieldWire, fieldMatrixWire, checkInformationExactTree as check, type ReferenceFieldComplex as C } from './learning-quantum-information-reference-field';
import { informationRequire as ensure, informationRecord as record, informationList as list, createInformationObservationChecks, InformationResponseInvalid } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'error-correction'}>;
const scale=(z:C,p:F)=>fieldMultiply(z,fieldComplex(fieldPair(p)));
const bits=(n:number)=>[0,1,2].reduce((s,b)=>s+((n>>b)&1),0);
const parity=(index:number)=>[Number(((index&4)!==0)!==((index&2)!==0)),Number(((index&2)!==0)!==((index&1)!==0))];
const correction=(label:number)=>[0,1,4,2][label];
const xWord=(mask:number)=>[4,2,1].map(bit=>mask&bit?'X':'I').join('');
function image(word:string,logicalBit:number) {
 let flip=0,z=0,ys=0;
 for(let i=0;i<3;i++){const bit=1<<(2-i);if(word[i]==='X'||word[i]==='Y')flip|=bit;if(word[i]==='Z'||word[i]==='Y')z|=bit;if(word[i]==='Y')ys++;}
 const initial=logicalBit?7:0,phases=[fieldOne,fieldComplex(fieldPair(),fieldPair(one)),fieldComplex(fieldPair(f(BigInt(-1)))),fieldComplex(fieldPair(),fieldPair(f(BigInt(-1))))];
 const phase=phases[(ys+2*(bits(initial&z)%2))%4];
 return {index:initial^flip,phase};
}
function logicalOperator(word:string,label:number):C[][] {
 const images=[image(word,0),image(word,1)];
 return [0,1].map(i=>images.map(({index,phase})=>{
  const [a,b]=parity(index);return 2*a+b===label&&(index^correction(label))===7*i?phase:fieldZero;
 }));
}
const apply=(k:C[][],v:C[])=>k.map(row=>row.reduce((s,z,j)=>fieldAdd(s,fieldMultiply(z,v[j])),fieldZero));
function experiment(input:C[],word:string,p:F) {
 const encoded=Array.from({length:8},(_,i)=>i===0?input[0]:i===7?input[1]:fieldZero);
 const errorImages=[image(word,0),image(word,1)],errored=Array.from({length:8},(_,i)=>errorImages.reduce((s,x,j)=>x.index===i?fieldAdd(s,fieldMultiply(x.phase,input[j])):s,fieldZero));
 const active=parity(errorImages[0].index),activeLabel=2*active[0]+active[1];
 const branches=Array.from({length:4},(_,label)=>{
  const k=logicalOperator(word,label),raw=apply(k,input);
  return {syndrome:[Math.floor(label/2),label%2],correction:xWord(correction(label)),probability:[label===activeLabel?'1':'0','0'],logicalOperator:fieldMatrixWire(k),rawRecovered:raw.map(fieldWire),conditionalLogicalDensity:label===activeLabel?fieldMatrixWire(fieldDensity(raw)):null};
 });
 const noise=Array.from({length:8},(_,mask)=>{
  const [a,b]=parity(mask),weight=[0,1,2].reduce((s,j)=>mul(s,mask&(1<<j)?p:sub(one,p)),one);
  return {error:xWord(mask),weight,syndrome:[a,b],correction:xWord(correction(2*a+b)),logicalOperator:fieldMatrixWire(logicalOperator(xWord(mask),2*a+b))};
 });
 // Independent binomial/majority formula, rather than producer recovery sums.
 const failure=add(mul(f(BigInt(3)),mul(mul(p,p),sub(one,p))),mul(mul(p,p),p)),success=sub(one,failure);
 ensure(eq(noise.reduce((s,r)=>add(s,r.weight),zero),one),'$reference','Noise mass is not one');
 ensure(eq(noise.filter((_,mask)=>bits(mask)>=2).reduce((s,r)=>add(s,r.weight),zero),failure),'$reference','Majority formula and branch mass disagree');
 const rho=fieldDensity(input),output=[0,1].map(i=>[0,1].map(j=>fieldAdd(scale(rho[i][j],success),scale(rho[1-i][1-j],failure))));
 const choi=[0,1,2,3].map(i=>[0,1,2,3].map(j=>{
  const even=(i===0||i===3)&&(j===0||j===3),odd=(i===1||i===2)&&(j===1||j===2);
  return fieldComplex(fieldPair(mul(f(BigInt(1),BigInt(2)),even?success:odd?failure:zero)));
 }));
 const xMean=fieldAdd(rho[0][1],rho[1][0]),fidelity=fieldAdd(fieldComplex(fieldPair(success)),scale(fieldMultiply(xMean,xMean),failure));
 return {input:input.map(fieldWire),selectedError:word,encodedDensity:fieldMatrixWire(fieldDensity(encoded)),erroredDensity:fieldMatrixWire(fieldDensity(errored)),branches,noise,
  physicalXProbability:p,logicalXProbability:failure,logicalOutput:fieldMatrixWire(output),normalizedLogicalChoi:fieldMatrixWire(choi),inputSquaredFidelity:fieldWire(fidelity),entanglementFidelity:fieldWire(fieldComplex(fieldPair(success))),
  contract:'Ideal repetition encoding, projective Z0Z1/Z1Z2 syndrome and minimum-weight X recovery. Selected Pauli branches are distinct from the IID X-noise channel. An impossible branch has null conditional density. This code corrects single X errors, not arbitrary single Pauli errors. No threshold/fault tolerance, arbitrary8D PSD claim, Worker/public input reader/UI.'};
}
function condition(words:readonly string[]) {
 const images=words.map(word=>[image(word,0),image(word,1)]);
 const pairs=images.flatMap((a,i)=>images.map((b,j)=>{
  const compressed=a.map(x=>b.map(y=>x.index===y.index?fieldMultiply(fieldConjugate(x.phase),y.phase):fieldZero));
  const scalar=fieldEqual(compressed[0][0],compressed[1][1])&&fieldEqual(compressed[0][1],fieldZero)&&fieldEqual(compressed[1][0],fieldZero);
  return {a:i,b:j,errors:[words[i],words[j]],compressed:fieldMatrixWire(compressed),scalarOnCode:scalar,c:scalar?fieldWire(compressed[0][0]):null};
 }));
 return {words:Array.from(words),conditionHolds:pairs.every(p=>p.scalarOnCode),firstWitness:pairs.find(p=>!p.scalarOnCode)??null,pairs,
  contract:'Checks V†Ea†EbV=c_ab I on the repetition code. This tests correctability of the selected operator span, not arbitrary unknown noise or a constructed recovery for every passing error family.'};
}
export type ErrorCorrectionResponseValidation={ok:true;response:Reply;checkedExactCoefficients:number;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumErrorCorrectionResponse(request:unknown,response:unknown):ErrorCorrectionResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='error-correction','$request','Error correction request required');
  const owned=snapshotManyBodyWire(response,16384),d=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(d.schemaVersion===1&&d.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(d.input)===quantumInformationInputKey(input),'$.input','Original input/error family differs');
  const result=record(d.result,['experiment','condition'],'$.result'),expected=experiment(referenceFieldVector(input.input),input.selectedError,rat(input.physicalXProbability));
  const actual=record(result.experiment,Object.keys(expected),'$.result.experiment'),observations=createInformationObservationChecks();let count=0;
  for(const k of ['input','selectedError','encodedDensity','erroredDensity','branches','logicalOutput','normalizedLogicalChoi','inputSquaredFidelity','entanglementFidelity','contract']as const)count+=check(actual[k],expected[k],`$.result.experiment.${k}`);
  for(const k of ['physicalXProbability','logicalXProbability']as const)observations.exact(actual[k],expected[k],`$.result.experiment.${k}`);
  list(actual.noise,8,'$.result.experiment.noise').forEach((x,i)=>{
   const row=record(x,Object.keys(expected.noise[i]),`$.result.experiment.noise[${i}]`);
   for(const k of ['error','syndrome','correction','logicalOperator']as const)count+=check(row[k],expected.noise[i][k],`$.result.experiment.noise[${i}].${k}`);
   observations.exact(row.weight,expected.noise[i].weight,`$.result.experiment.noise[${i}].weight`);
  });
  count+=check(result.condition,condition(input.errors),'$.result.condition');
  return {ok:true,response:owned as Reply,checkedExactCoefficients:count,checkedObservations:observations.count};
 }catch(error){return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid error correction response'};}
}
