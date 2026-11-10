/** Independent phase-channel matrix units, environment determinant and entropy
 * reference. Family theorems are scoped explicitly; no producer capacity engine. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, one, two, add, sub, mul, div, eq, le, type F, type R } from './learning-quantum-hilbert-core';
import { independentUnitRoot, referenceDensity, referenceDeterminant, referenceQubitEntropy, referenceBinaryEntropy, referencePoint as point, referenceAdd as plus, referenceSubtract as minus, referenceScale as scale, referenceClip as clip, type ReferenceMatrix } from './learning-quantum-information-reference-qubit';
import { checkInformationExactTree as exactTree } from './learning-quantum-information-reference-field';
import { informationRequire as ensure, informationRecord as record, informationList as list, informationObservationRange as reported, createInformationObservationChecks, InformationResponseInvalid } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'capacity'}>;
const rat=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const contract='Exact rational physicality and channel/Choi data; directed512-bit spectral roots and256-bit entropy logarithms in bits. Capacity values use theorems specific to the dephasing family, not a grid search or the selected input. Exact pure-input/complete-dephasing coherent-information zeros are separated from raw intervals. No generic capacity solver, private/two-way-assisted capacity, finite-block guarantee, Worker/public input reader/UI.';
export type CapacityResponseValidation={ok:true;response:Reply;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumCapacityResponse(request:unknown,response:unknown):CapacityResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='capacity','$request','Capacity request required');
  const owned=snapshotManyBodyWire(response),envelope=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(envelope.schemaVersion===1&&envelope.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(envelope.input)===quantumInformationInputKey(input),'$.input','Original channel/input differs');
  const d=record(envelope.result,['phaseFlipProbability','inputBloch','inputDensity','outputBloch','outputDensity','environmentDensity','environmentSmallerEigenvalue','normalizedChoi','inputEntropy','outputEntropy','environmentEntropy','selectedInputCoherentInformation','rawSelectedInputCoherentInformation','selectedInputReferenceMutualInformation','capacitiesByFamilyTheorem','quantumCapacityStrictlyPositive','phaseErrorEntropy','zBasisMessageInformation','xBasisMessageInformation','xBasisEqualPriorGuessSuccess','assumptions','contract'],'$.result');
  const p=rat(input.probability),other=sub(one,p),attenuation=sub(other,p),bloch=input.input.map(rat),rho=referenceDensity(bloch);
  // N(Eij)=Eij on the diagonal and (1-2p)Eij off it.
  const output:ReferenceMatrix=rho.map((row,i)=>row.map((z,j)=>({re:mul(z.re,i===j?one:attenuation),im:mul(z.im,i===j?one:attenuation)})));
  const outputBloch=[add(output[0][1].re,output[1][0].re),sub(output[1][0].im,output[0][1].im),sub(output[0][0].re,output[1][1].re)];
  const z=sub(rho[0][0].re,rho[1][1].re),environmentDeterminant=mul(mul(p,other),sub(one,mul(z,z)));
  const environmentGap=independentUnitRoot(sub(one,mul(f(BigInt(4)),environmentDeterminant)));
  const small=clip(scale(minus(point(one),environmentGap),div(one,two)),zero,div(one,two)),coherence=scale(independentUnitRoot(mul(p,other)),z);
  const inputEntropy=referenceQubitEntropy(rho),outputEntropy=referenceQubitEntropy(output),environmentEntropy=referenceBinaryEntropy(small),h=referenceBinaryEntropy(point(p));
  const structuralZero=eq(referenceDeterminant(rho),zero)||eq(p,other),coherent=structuralZero?point(zero):minus(outputEntropy,environmentEntropy),raw=minus(outputEntropy,environmentEntropy),mutual=plus(inputEntropy,coherent);
  const quantum=clip(minus(point(one),h)),assisted=minus(point(two),h),checks=createInformationObservationChecks();
  const matrix=(value:unknown,m:ReferenceMatrix,path:string)=>list(value,m.length,path).forEach((row,i)=>list(row,m.length,`${path}[${i}]`).forEach((entry,j)=>{
   const c=record(entry,['real','imaginary'],`${path}[${i}][${j}]`);checks.exact(c.real,m[i][j].re,`${path}[${i}][${j}].real`);checks.exact(c.imaginary,m[i][j].im,`${path}[${i}][${j}].imaginary`);
  }));
  checks.exact(d.phaseFlipProbability,p,'$.result.phaseFlipProbability');
  for(const[name,values]of [['inputBloch',bloch],['outputBloch',outputBloch]]as const)list(d[name],3,`$.result.${name}`).forEach((x,i)=>checks.exact(x,values[i],`$.result.${name}[${i}]`));
  matrix(d.inputDensity,rho,'$.result.inputDensity');matrix(d.outputDensity,output,'$.result.outputDensity');
  const environment=list(d.environmentDensity,2,'$.result.environmentDensity');
  environment.forEach((row,i)=>list(row,2,`$.result.environmentDensity[${i}]`).forEach((entry,j)=>{
   const c=record(entry,['real','imaginary'],`$.result.environmentDensity[${i}][${j}]`),path=`$.result.environmentDensity[${i}][${j}]`;
   if(i===j)checks.exact(c.real,i===0?other:p,`${path}.real`);else checks.check(c.real,coherence,`${path}.real`);
   checks.exact(c.imaginary,zero,`${path}.imaginary`);
  }));
  const choi:ReferenceMatrix=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>({re:i===j&&(i===0||i===3)?div(one,two):i!==j&&(i===0||i===3)&&(j===0||j===3)?div(attenuation,two):zero,im:zero})));
  matrix(d.normalizedChoi,choi,'$.result.normalizedChoi');
  for(const[name,value]of [['environmentSmallerEigenvalue',small],['inputEntropy',inputEntropy],['outputEntropy',outputEntropy],['environmentEntropy',environmentEntropy],['phaseErrorEntropy',h],['selectedInputCoherentInformation',coherent],['rawSelectedInputCoherentInformation',raw],['selectedInputReferenceMutualInformation',mutual],['xBasisMessageInformation',quantum]]as const)checks.check(d[name],value,`$.result.${name}`);
  for(const name of ['inputEntropy','outputEntropy','environmentEntropy','phaseErrorEntropy']as const){const r=reported(d[name],`$.result.${name}`);ensure(le(zero,r.lower)&&le(r.upper,one),`$.result.${name}`,'Qubit entropy bounds must lie in[0,1]');}
  if(structuralZero){const c=reported(d.selectedInputCoherentInformation,'$coherent');ensure(eq(c.lower,zero)&&eq(c.upper,zero),'$.result.selectedInputCoherentInformation','Structural zero must be exact');}
  const capacities=record(d.capacitiesByFamilyTheorem,['classicalBitsPerUse','entanglementAssistedClassicalBitsPerUse','quantumQubitsPerUse'],'$.result.capacitiesByFamilyTheorem');
  checks.exact(capacities.classicalBitsPerUse,one,'$.result.capacitiesByFamilyTheorem.classicalBitsPerUse');
  checks.check(capacities.entanglementAssistedClassicalBitsPerUse,assisted,'$.result.capacitiesByFamilyTheorem.entanglementAssistedClassicalBitsPerUse');checks.check(capacities.quantumQubitsPerUse,quantum,'$.result.capacitiesByFamilyTheorem.quantumQubitsPerUse');
  checks.exact(d.zBasisMessageInformation,one,'$.result.zBasisMessageInformation');checks.exact(d.xBasisEqualPriorGuessSuccess,le(p,other)?other:p,'$.result.xBasisEqualPriorGuessSuccess');
  ensure(d.quantumCapacityStrictlyPositive===!eq(p,other),'$.result.quantumCapacityStrictlyPositive','Wrong exact positivity classification');
  // Bind the intervals actually shown, including retained subtraction uncertainty.
  const relation=(value:unknown,expected:R,path:string)=>{const actual=reported(value,path);ensure(eq(actual.lower,expected.lower)&&eq(actual.upper,expected.upper),path,'Observable interval arithmetic is inconsistent');};
  relation(d.rawSelectedInputCoherentInformation,minus(reported(d.outputEntropy,'$output'),reported(d.environmentEntropy,'$environment')),'$.result.rawSelectedInputCoherentInformation');
  relation(d.selectedInputCoherentInformation,structuralZero?point(zero):reported(d.rawSelectedInputCoherentInformation,'$raw'),'$.result.selectedInputCoherentInformation');
  relation(d.selectedInputReferenceMutualInformation,plus(reported(d.inputEntropy,'$input'),reported(d.selectedInputCoherentInformation,'$coherent')),'$.result.selectedInputReferenceMutualInformation');
  relation(capacities.entanglementAssistedClassicalBitsPerUse,minus(point(two),reported(d.phaseErrorEntropy,'$phaseEntropy')),'$.result.capacitiesByFamilyTheorem.entanglementAssistedClassicalBitsPerUse');
  relation(capacities.quantumQubitsPerUse,clip(minus(point(one),reported(d.phaseErrorEntropy,'$phaseEntropy'))),'$.result.capacitiesByFamilyTheorem.quantumQubitsPerUse');
  relation(d.xBasisMessageInformation,reported(capacities.quantumQubitsPerUse,'$quantum'),'$.result.xBasisMessageInformation');
  const env=list(environment[0],2,'$environment0'),envOther=list(environment[1],2,'$environment1');relation((envOther[0]as Record<string,unknown>).real,reported((env[1]as Record<string,unknown>).real,'$coherence'),'$.result.environmentDensity[1][0].real');
  exactTree(d.assumptions,['Known fixed memoryless qubit phase-flip channel (1-p)rho+p ZrhoZ','C and Q are unassisted asymptotic vanishing-error rates; CE allows preshared entanglement','Degradability and Pauli symmetry justify the single-letter quantum optimum at I/2 for this family','The channel mutual-information theorem and symmetry justify CE for this family; no finite-block encoder is supplied'],'$.result.assumptions');
  ensure(d.contract===contract,'$.result.contract','Wrong channel capacity scope');
  return {ok:true,response:owned as Reply,checkedObservations:checks.count};
 }catch(error){return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid capacity response'};}
}
