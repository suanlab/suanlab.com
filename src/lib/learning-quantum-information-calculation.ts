/** Worker/CLI entry: own wire input, decode exact scalars, then enforce physical models. */
import {snapshotQuantumInformationInput,type InformationBloch,type InformationState,type QuantumInformationInput}from'./learning-quantum-information-input';
import {f,type F}from'./learning-quantum-hilbert-core';
import {quadratic,circuitScalar}from'./learning-qubit-scalar';
import {binaryQubitDiscrimination}from'./learning-quantum-information-discrimination';
import {correlatedBellEntropy}from'./learning-quantum-information-joint-entropy';
import {binarySourceBlockWeight}from'./learning-quantum-source-block';
import {binaryQubitEnsemble}from'./learning-quantum-information-ensemble';
import {qubitDephasingDataProcessing}from'./learning-quantum-information-relative-entropy';
import {idealBb84InterceptResend}from'./learning-quantum-information-bb84';
import {teleportQubit,denseCoding,cnotCopyCounterexample,noCloningOverlap}from'./learning-quantum-information-protocols';
import {repetitionCodeExperiment,repetitionKnillLaflamme}from'./learning-quantum-information-error-correction';
import {pureStateMajorization}from'./learning-quantum-information-majorization';
import {bellPairDilution}from'./learning-quantum-information-bell-dilution';
import {qubitDephasingCapacity}from'./learning-quantum-information-capacity';
const fraction=(text:string):F=>{const[n,d='1']=text.split('/');return f(BigInt(n),BigInt(d));};
const bloch=(value:InformationBloch)=>value.map(fraction)as[F,F,F];
const state=(value:InformationState)=>value.map(z=>circuitScalar(quadratic(fraction(z.real[0]),fraction(z.real[1])),quadratic(fraction(z.imaginary[0]),fraction(z.imaginary[1]))));
function response<I extends QuantumInformationInput,R>(input:I,result:R):{schemaVersion:1;kind:I['kind'];input:I;result:R}{return {schemaVersion:1 as const,kind:input.kind,input,result};}
export function calculateQuantumInformation(value:unknown){
 const input=snapshotQuantumInformationInput(value);
 switch(input.kind){
  case'discrimination':return response(input,binaryQubitDiscrimination(bloch(input.bloch0),bloch(input.bloch1),fraction(input.prior)));
  case'entropy':return response(input,{joint:correlatedBellEntropy(fraction(input.correlationWeight),fraction(input.coherence)),block:binarySourceBlockWeight(fraction(input.probabilityOne),input.copies,BigInt(input.retainedRank))});
  case'ensemble':return response(input,binaryQubitEnsemble(bloch(input.bloch0),bloch(input.bloch1),fraction(input.prior),input.measurement));
  case'relative-entropy':return response(input,qubitDephasingDataProcessing(bloch(input.rho),bloch(input.sigma)));
  case'bb84':{
   const vector=state(input.input),comparison=state(input.comparison);
   return response(input,{cloning:cnotCopyCounterexample(vector),overlap:noCloningOverlap(vector,comparison),attack:idealBb84InterceptResend(fraction(input.attackFraction),fraction(input.eveZProbability),input.sampleSize)});
  }
  case'teleportation':return response(input,{teleportation:teleportQubit(state(input.input)),denseCoding:denseCoding(input.a,input.b,input.order)});
  case'error-correction':return response(input,{experiment:repetitionCodeExperiment(state(input.input),input.selectedError,fraction(input.physicalXProbability)),condition:repetitionKnillLaflamme(input.errors)});
  case'majorization':return input.mode==='compare'?response(input,{mode:input.mode,comparison:pureStateMajorization(input.source.map(fraction),input.target.map(fraction))})
   :response(input,{mode:input.mode,dilution:bellPairDilution(fraction(input.a),fraction(input.b))});
  case'capacity':return response(input,qubitDephasingCapacity(fraction(input.probability),bloch(input.input)));
 }
}
export type QuantumInformationCalculation=ReturnType<typeof calculateQuantumInformation>;
