/** Independent validator Worker entry. Does not import the producer engine. */
import {gaugeOrbitData,gaugeOrbitSuccessHeader,type GaugeOrbitReceipt} from './learning-gauge-orbits-protocol';
import {readGaugeOrbitInput,gaugeOrbitInputKey} from './learning-gauge-orbits-input';
import {validateGaugeOrbitResult} from './learning-gauge-orbits-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGaugeOrbitValidationRequest(value:unknown):GaugeOrbitReceipt{
 let id:number|null=null;
 try{
  const d=gaugeOrbitData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readGaugeOrbitInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!gaugeOrbitSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateGaugeOrbitResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'gauge-orbits',inputKey:gaugeOrbitInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
