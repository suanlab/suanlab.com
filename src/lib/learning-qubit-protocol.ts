/** Reader-safe envelopes. A success header is not mathematical verification. */
import { qubitData,readQubitInput,qubitInputKey,type QubitInput,type QubitKind } from './learning-qubit-input';
import type { qubitCircuitReport } from './learning-qubit-circuit-report';
import type { channelReport,environmentReport,instrumentReport,deferredMeasurementReport } from './learning-qubit-channel';
import type { QubitObservations } from './learning-qubit-observations-schema';
export type QubitResults={circuit:ReturnType<typeof qubitCircuitReport>;channel:ReturnType<typeof channelReport>;environment:ReturnType<typeof environmentReport>;instrument:ReturnType<typeof instrumentReport>;deferred:ReturnType<typeof deferredMeasurementReport>};
export type QubitSuccess={[K in QubitKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<QubitInput,{kind:K}>;inputKey:string;result:QubitResults[K];observations:QubitObservations}}[QubitKind];
export type QubitFailure={version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'|'resource-limit'};
export type QubitResponse=QubitSuccess|QubitFailure;
export type QubitStarted={version:1;requestId:number;phase:'started';kind:QubitKind};
export type QubitRequest={version:1;requestId:number;input:QubitInput};
export type QubitValidationReceipt={version:1;requestId:number;phase:'validated';kind:QubitKind;inputKey:string;response:QubitSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
const idValid=(id:unknown):id is number=>typeof id==='number'&&Number.isSafeInteger(id)&&id>0;
export function readQubitRequest(value:unknown):{request:QubitRequest}|{failure:QubitFailure}{
  const d=qubitData(value,['version','requestId','input']);
  if(!d||d.version!==1||!idValid(d.requestId))return {failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
  const input=readQubitInput(d.input);
  return input?{request:{version:1,requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function qubitSuccessHeader(value:unknown,input:QubitInput,id:number):value is QubitSuccess {
  const d=qubitData(value,['version','requestId','ok','kind','input','inputKey','result','observations']);
  if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!==input.kind||d.inputKey!==qubitInputKey(input)||!d.result||typeof d.result!=='object'||!qubitData(d.observations,['schemaVersion','coordinates','states']))return false;
  const echoed=readQubitInput(d.input);return echoed!==null&&qubitInputKey(echoed)===d.inputKey;
}
