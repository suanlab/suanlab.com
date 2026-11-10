/** Independent validator Worker entry. Does not import the producer engine. */
import {spatialSpectrumData,spatialSpectrumSuccessHeader,type SpatialSpectrumReceipt} from './learning-spatial-spectrum-protocol';
import {readSpatialSpectrumInput} from './learning-spatial-spectrum-input';
import {spatialSpectrumInputKey} from './learning-spatial-spectrum-protocol';
import {validateSpatialSpectrumResult} from './learning-spatial-spectrum-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateSpatialSpectrumValidationRequest(value:unknown):SpatialSpectrumReceipt{
 let id:number|null=null;
 try{
  const d=spatialSpectrumData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readSpatialSpectrumInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!spatialSpectrumSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateSpatialSpectrumResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'spatial-spectrum',inputKey:spatialSpectrumInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
