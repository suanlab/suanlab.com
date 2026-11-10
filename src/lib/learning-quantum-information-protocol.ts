/** Reader-safe headers only. Header acceptance is not mathematical validation. */
import {qubitData as informationData} from './learning-qubit-input';
import {readQuantumInformationInput,quantumInformationInputKey,type QuantumInformationInput,type QuantumInformationKind} from './learning-quantum-information-input';
import type {QuantumInformationCalculation} from './learning-quantum-information-calculation';
export {informationData};
export type InformationSuccess={[K in QuantumInformationKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<QuantumInformationInput,{kind:K}>;inputKey:string;result:Extract<QuantumInformationCalculation,{kind:K}>['result']}}[QuantumInformationKind];
export type InformationFailure={version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'|'resource-limit'};
export type InformationStarted={version:1;requestId:number;phase:'started';kind:QuantumInformationKind};
export type InformationRequest={version:1;requestId:number;input:QuantumInformationInput};
export type InformationReceipt={version:1;requestId:number;phase:'validated';kind:QuantumInformationKind;inputKey:string;response:InformationSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function readInformationRequest(value:unknown):{request:InformationRequest}|{failure:InformationFailure}{
 const d=informationData(value,['version','requestId','input']);
 if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return {failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readQuantumInformationInput(d.input);
 return input?{request:{version:1,requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function informationSuccessHeader(value:unknown,input:QuantumInformationInput,id:number):value is InformationSuccess {
 const d=informationData(value,['version','requestId','ok','kind','input','inputKey','result']);
 if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!==input.kind||d.inputKey!==quantumInformationInputKey(input)||!d.result||typeof d.result!=='object'||Object.getPrototypeOf(d.result)!==Object.prototype)return false;
 const echoed=readQuantumInformationInput(d.input);return echoed!==null&&quantumInformationInputKey(echoed)===d.inputKey;
}
