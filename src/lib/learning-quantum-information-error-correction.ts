/** Exact three-qubit repetition code: syndrome recovery and finite Pauli KL checks. */
import { f, zero, one, sub, mul, sum, q, type F } from './learning-quantum-hilbert-core';
import { requireProtocolVector } from './learning-quantum-information-protocols';
import { circuitNormSquared, type CircuitScalar as C } from './learning-qubit-scalar';
import { Z,O,complex,quadratic,real,zeros,identity,tensor,product,adjoint,mAdd,mScale,mWire,trace,equal,times,conjugate,divide,qDiv,qSign,densityFromVector,circuitScalarWire,type M } from './learning-qubit-channel-algebra';
const paulis:Record<string,M>={I:identity(2),X:[[Z,O],[O,Z]],Y:[[Z,complex(quadratic(zero),quadratic(f(BigInt(-1))))],[complex(quadratic(zero),quadratic(one)),Z]],Z:[[O,Z],[Z,real(-1)]]};
const code: M=Array.from({length:8},(_,i)=>[i===0?O:Z,i===7?O:Z]);
function requirePauli(word:string){if(typeof word!=='string'||!/^[IXYZ]{3}$/.test(word))throw new RangeError('A three-site I/X/Y/Z Pauli word is required');}
const operator=(word:string)=>tensor(tensor(paulis[word[0]],paulis[word[1]]),paulis[word[2]]);
const syndrome=(index:number)=>[((index>>2)&1)^((index>>1)&1),((index>>1)&1)^(index&1)];
const correctionMask=(a:number,b:number)=>a?(b?2:4):(b?1:0);
const xWord=(mask:number)=>[4,2,1].map(bit=>mask&bit?'X':'I').join('');
const projector=(a:number,b:number):M=>Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>i===j&&syndrome(i)[0]===a&&syndrome(i)[1]===b?O:Z));
const recovery=(a:number,b:number)=>product(product(adjoint(code),operator(xWord(correctionMask(a,b)))),projector(a,b));
const density=(v:readonly C[])=>{const norm=circuitNormSquared(v);return v.map(a=>v.map(b=>divide(times(a,conjugate(b)),norm)));};
export function repetitionKnillLaflamme(words:readonly string[]){
 if(!Array.isArray(words)||words.length<1||words.length>16)throw new RangeError('Select one to16 Pauli error operators');
 Array.from(words).forEach(requirePauli);
 const images=words.map(word=>product(operator(word),code));
 const pairs=images.flatMap((a,i)=>images.map((b,j)=>{
  const compressed=product(adjoint(a),b),scalar=equal(compressed[0][0],compressed[1][1])&&equal(compressed[0][1],Z)&&equal(compressed[1][0],Z);
  return {a:i,b:j,errors:[words[i],words[j]],compressed:mWire(compressed),scalarOnCode:scalar,c:scalar?circuitScalarWire(compressed[0][0]):null};
 }));
 return {words:Array.from(words),conditionHolds:pairs.every(p=>p.scalarOnCode),firstWitness:pairs.find(p=>!p.scalarOnCode)??null,pairs,
  contract:'Checks V†Ea†EbV=c_ab I on the repetition code. This tests correctability of the selected operator span, not arbitrary unknown noise or a constructed recovery for every passing error family.'};
}
export function repetitionCodeExperiment(input:readonly C[],selectedError='XII',physicalXProbability:F=f(BigInt(1),BigInt(10))){
 requireProtocolVector(input);requirePauli(selectedError);
 const p=physicalXProbability;
 if(!p||typeof p.n!=='bigint'||typeof p.d!=='bigint'||p.d<=BigInt(0)||p.n<BigInt(0)||p.n>p.d||p.n.toString().length>32||p.d.toString().length>32)throw new RangeError('Physical X probability must lie in[0,1] with bounded rational coefficients');
 const encoded=product(code,input.map(z=>[z])).map(row=>row[0]),error=operator(selectedError),errored=product(error,encoded.map(z=>[z])).map(row=>row[0]),norm=circuitNormSquared(input);
 const branches=Array.from({length:4},(_,label)=>{
  const a=Math.floor(label/2),b=label%2,K=product(product(recovery(a,b),error),code),raw=product(K,input.map(z=>[z])).map(row=>row[0]),weight=circuitNormSquared(raw),available=qSign(weight)>0;
  return {syndrome:[a,b],correction:xWord(correctionMask(a,b)),probability:circuitScalarWire(complex(qDiv(weight,norm))).real,
   logicalOperator:mWire(K),rawRecovered:raw.map(circuitScalarWire),conditionalLogicalDensity:available?mWire(densityFromVector(raw)):null};
 });
 let output=zeros(2),choi=zeros(4),failure=zero;
 const target=densityFromVector(Array.from(input));
 const bell:M=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>(i===0||i===3)&&(j===0||j===3)?real(1,2):Z));
 const noise=Array.from({length:8},(_,mask)=>{
  const bits=[4,2,1].filter(bit=>mask&bit).length,weight=mul(f(p.n**BigInt(bits),p.d**BigInt(bits)),f(sub(one,p).n**BigInt(3-bits),sub(one,p).d**BigInt(3-bits)));
  const [a,b]=syndrome(mask),K=product(product(recovery(a,b),operator(xWord(mask))),code);
  output=mAdd(output,mScale(product(product(K,target),adjoint(K)),quadratic(weight)));
  const full=tensor(identity(2),K);choi=mAdd(choi,mScale(product(product(full,bell),adjoint(full)),quadratic(weight)));
  if(!equal(K[0][1],Z))failure=sum([failure,weight]);
  return {error:xWord(mask),weight:q(weight),syndrome:[a,b],correction:xWord(correctionMask(a,b)),logicalOperator:mWire(K)};
 });
 return {input:input.map(circuitScalarWire),selectedError,encodedDensity:mWire(density(encoded)),erroredDensity:mWire(density(errored)),branches,noise,
  physicalXProbability:q(p),logicalXProbability:q(failure),logicalOutput:mWire(output),normalizedLogicalChoi:mWire(choi),
  inputSquaredFidelity:circuitScalarWire(trace(product(target,output))),entanglementFidelity:circuitScalarWire(trace(product(bell,choi))),
  contract:'Ideal repetition encoding, projective Z0Z1/Z1Z2 syndrome and minimum-weight X recovery. Selected Pauli branches are distinct from the IID X-noise channel. An impossible branch has null conditional density. This code corrects single X errors, not arbitrary single Pauli errors. No threshold/fault tolerance, arbitrary8D PSD claim, Worker/public input reader/UI.'};
}
