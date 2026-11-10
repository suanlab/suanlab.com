/** Worker/CLI validation entry for all nine draft families. No producer import. */
import { snapshotQuantumInformationInput } from './learning-quantum-information-input';
import { validateQuantumDiscriminationResponse } from './learning-quantum-information-discrimination-validator';
import { validateQuantumEntropyResponse } from './learning-quantum-information-entropy-validator';
import { validateQuantumEnsembleResponse } from './learning-quantum-information-ensemble-validator';
import { validateQuantumRelativeEntropyResponse } from './learning-quantum-information-relative-validator';
import { validateQuantumBb84Response } from './learning-quantum-information-bb84-validator';
import { validateQuantumTeleportationResponse } from './learning-quantum-information-teleportation-validator';
import { validateQuantumErrorCorrectionResponse } from './learning-quantum-information-error-correction-validator';
import { validateQuantumMajorizationResponse } from './learning-quantum-information-majorization-validator';
import { validateQuantumCapacityResponse } from './learning-quantum-information-capacity-validator';
const validators={discrimination:validateQuantumDiscriminationResponse,entropy:validateQuantumEntropyResponse,ensemble:validateQuantumEnsembleResponse,'relative-entropy':validateQuantumRelativeEntropyResponse,bb84:validateQuantumBb84Response,teleportation:validateQuantumTeleportationResponse,'error-correction':validateQuantumErrorCorrectionResponse,majorization:validateQuantumMajorizationResponse,capacity:validateQuantumCapacityResponse};
export function validateQuantumInformationResponse(request:unknown,response:unknown) {
 try {
  const input=snapshotQuantumInformationInput(request),result=validators[input.kind](input,response);
  if(!result.ok)return result;
  return {ok:true as const,response:result.response,checkedObservations:'checkedObservations'in result?result.checkedObservations:0,checkedExactCoefficients:'checkedExactCoefficients'in result?result.checkedExactCoefficients:0};
 }catch(error){return {ok:false as const,path:'$request',reason:error instanceof Error?error.message:'Invalid quantum information request'};}
}
export type QuantumInformationResponseValidation=ReturnType<typeof validateQuantumInformationResponse>;
