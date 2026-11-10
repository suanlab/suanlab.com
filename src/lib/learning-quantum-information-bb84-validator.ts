/** Independent CNOT tensor indices, overlap amplitudes and BB84 conditional
 * bit channels/posteriors. No production protocol or circuit evaluator. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, one, add, sub, mul, div, eq, type F } from './learning-quantum-hilbert-core';
import { fieldRational as rat, fieldPair, fieldZero, fieldComplex, fieldAdd, fieldConjugate, fieldMultiply, fieldDivideReal, fieldNorm, fieldEqual, referenceFieldVector, fieldDensity, fieldWire, fieldMatrixWire, checkInformationExactTree as check, type ReferenceFieldComplex as C } from './learning-quantum-information-reference-field';
import { informationRequire as ensure, informationRecord as record, informationList as list, createInformationObservationChecks, InformationResponseInvalid } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'bb84'}>;
type Basis='z'|'x';
const half=f(BigInt(1),BigInt(2)),eighth=f(BigInt(1),BigInt(8));
function cloning(input:C[],comparison:C[]) {
 const joint=fieldDensity([input[0],fieldZero,fieldZero,input[1]]),target=fieldDensity(input);
 const marginal=[0,1].map(i=>[0,1].map(j=>i===j?target[i][j]:fieldZero));
 const desired=[0,1,2,3].map(i=>[0,1,2,3].map(j=>fieldMultiply(target[Math.floor(i/2)][Math.floor(j/2)],target[i%2][j%2])));
 const fidelity=fieldAdd(fieldMultiply(target[0][0],target[0][0]),fieldMultiply(target[1][1],target[1][1]));
 const amplitude=input.reduce((s,z,i)=>fieldAdd(s,fieldMultiply(fieldConjugate(z),comparison[i])),fieldZero);
 const overlap=fieldDivideReal(fieldMultiply(amplitude,fieldConjugate(amplitude)),fieldMultiply(fieldComplex(fieldNorm(input)),fieldComplex(fieldNorm(comparison))).re);
 return {
  cloning:{joint:fieldMatrixWire(joint),first:fieldMatrixWire(marginal),second:fieldMatrixWire(marginal),desiredProduct:fieldMatrixWire(desired),
   isExactProductClone:fieldEqual(input[0],fieldZero)||fieldEqual(input[1],fieldZero),singleCopySquaredFidelity:fieldWire(fidelity),
   contract:'This tests the CNOT candidate with a blank |0> and one specified pure input. Failure is a counterexample, not a proof of universal no-cloning.'},
  overlap:{squaredOverlap:fieldWire(overlap),compatibleWithClosedUnitaryCloning:fieldEqual(overlap,fieldZero)||fieldEqual(overlap,fieldComplex(fieldPair(one))),
   contract:'A fixed blank and a closed unitary cloning both pure inputs would require |<psi|phi>|=|<psi|phi>|^2. This is a necessary condition; no cloner is constructed.'},
 };
}
function bb84(attack:F,eveZ:F,sampleSize:number) {
 const transition=(from:Basis,bit:number,to:Basis,out:number)=>from===to?(bit===out?one:zero):half;
 type Row={aliceBasis:Basis;aliceBit:number;bobBasis:Basis;bobBit:number;attacked:boolean;eveBasis:Basis|null;eveBit:number|null;weight:F};
 const rows:Row[]=[];
 for(const a of ['z','x']as const)for(const bit of [0,1])for(const b of ['z','x']as const){
  for(const out of [0,1])rows.push({aliceBasis:a,aliceBit:bit,bobBasis:b,bobBit:out,attacked:false,eveBasis:null,eveBit:null,weight:mul(mul(eighth,sub(one,attack)),transition(a,bit,b,out))});
  for(const e of ['z','x']as const)for(const observed of [0,1])for(const out of [0,1])rows.push({aliceBasis:a,aliceBit:bit,bobBasis:b,bobBit:out,attacked:true,eveBasis:e,eveBit:observed,
   weight:[eighth,attack,e==='z'?eveZ:sub(one,eveZ),transition(a,bit,e,observed),transition(e,observed,b,out)].reduce(mul,one)});
 }
 const mass=(selected:Row[])=>selected.reduce((n,r)=>add(n,r.weight),zero),sifted=rows.filter(r=>r.aliceBasis===r.bobBasis),sift=mass(sifted);
 const qber=div(mass(sifted.filter(r=>r.aliceBit!==r.bobBit)),sift);
 const basisQber=(basis:Basis)=>{const selected=sifted.filter(r=>r.aliceBasis===basis);return div(mass(selected.filter(r=>r.aliceBit!==r.bobBit)),mass(selected));};
 // Condition only on Eve's record and announced bases, never on Bob's private bit.
 const groups=new Map<string,[F,F]>();
 for(const r of sifted){const key=JSON.stringify([r.aliceBasis,r.attacked,r.eveBasis,r.eveBit]),weights=groups.get(key)??[zero,zero];weights[r.aliceBit]=add(weights[r.aliceBit],r.weight);groups.set(key,weights);}
 let information=zero,guessMass=zero;
 for(const [a,b]of groups.values()){
  const total=add(a,b);if(eq(total,zero))continue;
  const deterministic=eq(a,zero)||eq(b,zero);if(!deterministic&&!eq(a,b))throw new RangeError('Unexpected BB84 posterior');
  if(deterministic)information=add(information,div(total,sift));
  guessMass=add(guessMass,deterministic?total:mul(half,total));
 }
 let noError=one;for(let j=0;j<sampleSize;j++)noError=mul(noError,sub(one,qber));
 return {attackFraction:attack,eveZProbability:eveZ,totalProbability:mass(rows),siftProbability:sift,siftedQber:qber,basisQber:{z:basisQber('z'),x:basisQber('x')},
  eveKnownSiftedFraction:information,eveIdealInformationBits:information,eveGuessSuccess:div(guessMass,sift),sampleSize,probabilityAtLeastOneSampleError:sub(one,noError),
  rows,
  assumptions:['Uniform independent Alice bits and Z/X bases; uniform independent Bob bases','Independent attack decisions and Eve basis choices; projective intercept/resend; ideal noiseless single-photon qubits','After sifting Eve knows public bases and her attack flags/outcomes; wrong-basis observations reveal no Alice-bit information','Authenticated public classical discussion is required in real QKD; error correction, privacy amplification and finite-key security are not modeled'],
  contract:'Exact finite enumeration and IID sample-error probability for this attack only. Eve information is the known-bit/unknown-bit flagged model after basis announcement. A zero-error sample does not certify secrecy. No modern QKD security threshold, key rate, implementation loophole analysis, Worker, public input reader or UI.'};
}
export type Bb84ResponseValidation={ok:true;response:Reply;checkedExactCoefficients:number;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumBb84Response(request:unknown,response:unknown):Bb84ResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='bb84','$request','BB84 request required');
  const owned=snapshotManyBodyWire(response,16384),d=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(d.schemaVersion===1&&d.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(d.input)===quantumInformationInputKey(input),'$.input','Original request differs');
  const states=cloning(referenceFieldVector(input.input),referenceFieldVector(input.comparison));
  const r=record(d.result,['cloning','overlap','attack'],'$.result');
  const count=check(r.cloning,states.cloning,'$.result.cloning')+check(r.overlap,states.overlap,'$.result.overlap');
  const expected=bb84(rat(input.attackFraction),rat(input.eveZProbability),input.sampleSize),attack=record(r.attack,Object.keys(expected),'$.result.attack'),observations=createInformationObservationChecks({coefficientDigits:10000,binaryExponent:32768});
  for(const k of ['attackFraction','eveZProbability','totalProbability','siftProbability','siftedQber','eveKnownSiftedFraction','eveIdealInformationBits','eveGuessSuccess','probabilityAtLeastOneSampleError']as const)observations.exact(attack[k],expected[k],`$.result.attack.${k}`);
  const basis=record(attack.basisQber,['z','x'],'$.result.attack.basisQber');
  for(const k of ['z','x']as const)observations.exact(basis[k],expected.basisQber[k],`$.result.attack.basisQber.${k}`);
  list(attack.rows,80,'$.result.attack.rows').forEach((x,i)=>{
   const row=record(x,Object.keys(expected.rows[i]),`$.result.attack.rows[${i}]`);
   for(const k of ['aliceBasis','aliceBit','bobBasis','bobBit','attacked','eveBasis','eveBit']as const)ensure(row[k]===expected.rows[i][k],`$.result.attack.rows[${i}].${k}`,'Wrong canonical branch record');
   observations.exact(row.weight,expected.rows[i].weight,`$.result.attack.rows[${i}].weight`);
  });
  for(const k of ['sampleSize','assumptions','contract']as const)check(attack[k],expected[k],`$.result.attack.${k}`);
  return {ok:true,response:owned as Reply,checkedExactCoefficients:count,checkedObservations:observations.count};
 }catch(error){return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid BB84 response'};}
}
