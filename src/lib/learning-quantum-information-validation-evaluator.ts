/** Independent validation Worker entry. No production mathematical evaluator. */
import {informationData,informationSuccessHeader,type InformationReceipt} from './learning-quantum-information-protocol';
import {readQuantumInformationInput,quantumInformationInputKey} from './learning-quantum-information-input';
import {validateQuantumInformationResponse} from './learning-quantum-information-response';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateInformationValidationRequest(value:unknown):InformationReceipt {
 let id:number|null=null;
 try {
  const d=informationData(value,['version','requestId','input','response']);
  if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return {version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readQuantumInformationInput(d.input);if(!input)return {version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const response=snapshotQubitWire(d.response);
  if(!informationSuccessHeader(response,input,id))return {version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateQuantumInformationResponse(input,{schemaVersion:1,kind:response.kind,input:response.input,result:response.result});
  return checked.ok?{version:1,requestId:id,phase:'validated',kind:input.kind,inputKey:quantumInformationInputKey(input),response}:{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
 }catch(error){return {version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
