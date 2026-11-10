/** Closed-form tensor/reference validation of ideal teleportation and dense
 * coding. No circuit runner, producer protocol or channel algebra. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, type F } from './learning-quantum-hilbert-core';
import { fieldPair, fieldComplex, fieldZero, fieldOne, fieldMultiply, fieldNorm, referenceFieldVector, fieldDensity, fieldWire, fieldMatrixWire, checkInformationExactTree as check, type ReferenceFieldComplex as C } from './learning-quantum-information-reference-field';
import { informationRequire as ensure, informationRecord as record, InformationResponseInvalid } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'teleportation'}>;
const half=f(BigInt(1),BigInt(2)),h=fieldComplex(fieldPair(zero,half)),negative=fieldComplex(fieldPair(f(BigInt(-1))));
const scale=(z:C,x:F)=>fieldMultiply(z,fieldComplex(fieldPair(x)));
const minus=(z:C)=>fieldMultiply(z,negative);
const vectorWire=(v:C[])=>v.map(fieldWire);
const mixed=()=>[0,1].map(i=>[0,1].map(j=>i===j?fieldComplex(fieldPair(half)):fieldZero));
function teleportation(input:C[]) {
 const [a,b]=input,ah=fieldMultiply(a,h),bh=fieldMultiply(b,h),a2=scale(a,half),b2=scale(b,half);
 // Fixed three-site tensor expansion, independently derived for each prefix.
 const prefixes=[
  [a,fieldZero,fieldZero,fieldZero,b,fieldZero,fieldZero,fieldZero],
  [ah,fieldZero,ah,fieldZero,bh,fieldZero,bh,fieldZero],
  [ah,fieldZero,fieldZero,ah,bh,fieldZero,fieldZero,bh],
  [ah,fieldZero,fieldZero,ah,fieldZero,bh,bh,fieldZero],
  [a2,b2,b2,a2,a2,minus(b2),minus(b2),a2],
 ];
 const raw=[[a2,b2],[b2,a2],[a2,minus(b2)],[minus(b2),a2]],restored=[a2,b2],target=fieldMatrixWire(fieldDensity(input));
 return {siteOrder:['Alice input','Alice Bell half','Bob Bell half'],input:vectorWire(input),inputNormSquared:fieldWire(fieldComplex(fieldNorm(input))).real,
  prefixes:prefixes.map(vectorWire),gates:[{gate:'H',target:1},{gate:'CX',control:1,target:2},{gate:'CX',control:0,target:1},{gate:'H',target:0}],
  branches:raw.map((v,label)=>{const a=Math.floor(label/2),b=label%2;return {a,b,probability:['1/4','0'],rawBob:vectorWire(v),correctedBob:vectorWire(restored),bobDensity:fieldMatrixWire(fieldDensity(v)),correctedDensity:target,
   correctionGates:[...(b?[{gate:'X',target:0}]:[]),...(a?[{gate:'Z',target:0}]:[])]};}),
  inputDensity:target,bobBeforeMessage:fieldMatrixWire(mixed()),resources:{sharedEbits:1,classicalBitsSent:2,quantumInputCopies:1},
  contract:'Exact ideal three-qubit circuit. Branch vectors are unnormalized and retain input scale. Every branch has probability1/4; corrections execute X^b then Z^a. No finite-noise fidelity, implementation security, Worker, public input reader or UI.'};
}
function denseCoding(a:number,b:number,order:'x-then-z'|'z-then-x') {
 // Bell expansion: |j xor b,j> with phase (-1)^(a*(j xor b))
 // for X then Z, or (-1)^(a*j) for Z then X.
 const bell=(aa:number,bb:number)=>[0,1,2,3].map(index=>{
  const j=index%2,k=Math.floor(index/2);if(k!==(j^bb))return fieldZero;
  const exponent=aa*(order==='x-then-z'?k:j);return exponent%2?minus(h):h;
 });
 const initial=[fieldOne,fieldZero,fieldZero,fieldZero],afterH=[h,fieldZero,h,fieldZero];
 const prefixes=[initial,afterH,bell(0,0)],encoding:{gate:string;target:number}[]=[];
 if(order==='x-then-z'){
  if(b){encoding.push({gate:'X',target:0});prefixes.push(bell(0,b));}
  if(a){encoding.push({gate:'Z',target:0});prefixes.push(bell(a,b));}
 }else{
  if(a){encoding.push({gate:'Z',target:0});prefixes.push(bell(a,0));}
  if(b){encoding.push({gate:'X',target:0});prefixes.push(bell(a,b));}
 }
 const encoded=bell(a,b),phase=order==='z-then-x'&&a*b===1?-1:1;
 const beforeFinalH=[0,1,2,3].map(index=>index%2!==b?fieldZero:((Math.floor(index/2)*a+(phase<0?1:0))%2?minus(h):h));
 const decoded=[0,1,2,3].map(index=>index===2*a+b?(phase<0?minus(fieldOne):fieldOne):fieldZero);
 prefixes.push(beforeFinalH,decoded);
 return {message:[a,b],order,encoding,decodedState:vectorWire(decoded),encodedState:vectorWire(encoded),bobBeforeTransmission:fieldMatrixWire(mixed()),
  decodedProbabilities:[0,1,2,3].map(i=>[i===2*a+b?'1':'0','0']),decodedDensity:fieldMatrixWire([0,1,2,3].map(i=>[0,1,2,3].map(j=>i===2*a+b&&j===2*a+b?fieldOne:fieldZero))),
  decodedIndex:2*a+b,prefixes:prefixes.map(vectorWire),resources:{sharedEbits:1,qubitsSent:1,classicalMessageBits:2},
  contract:'Ideal shared Bell pair, one transmitted qubit, joint receiver decoding. Raw global phase is retained. No unassisted one-qubit two-bit capacity claim.'};
}
export type TeleportationResponseValidation={ok:true;response:Reply;checkedExactCoefficients:number}|{ok:false;path:string;reason:string};
export function validateQuantumTeleportationResponse(request:unknown,response:unknown):TeleportationResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='teleportation','$request','Teleportation request required');
  const owned=snapshotManyBodyWire(response,16384),d=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(d.schemaVersion===1&&d.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(d.input)===quantumInformationInputKey(input),'$.input','Original input/message/order differs');
  const expected={teleportation:teleportation(referenceFieldVector(input.input)),denseCoding:denseCoding(input.a,input.b,input.order)};
  const count=check(d.result,expected,'$.result');return {ok:true,response:owned as Reply,checkedExactCoefficients:count};
 }catch(error){return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid teleportation response'};}
}
