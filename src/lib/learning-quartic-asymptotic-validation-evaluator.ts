/** Independent validator Worker entry. Does not import the producer engine. */
import {quarticAsymptoticData,quarticAsymptoticSuccessHeader,type QuarticAsymptoticReceipt} from './learning-quartic-asymptotic-protocol';
import {readQuarticAsymptoticInput} from './learning-quartic-asymptotic-input';
import {quarticAsymptoticInputKey} from './learning-quartic-asymptotic-protocol';
import {validateQuarticAsymptoticResult} from './learning-quartic-asymptotic-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateQuarticAsymptoticValidationRequest(value:unknown):QuarticAsymptoticReceipt{
 let id:number|null=null;
 try{
  const d=quarticAsymptoticData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readQuarticAsymptoticInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!quarticAsymptoticSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateQuarticAsymptoticResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'quartic-asymptotic',inputKey:quarticAsymptoticInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
