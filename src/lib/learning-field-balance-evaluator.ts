/** Producer entry; Worker-side only. */
import {calculateFieldBalance} from './learning-field-balance';
import {fieldBalanceInputKey} from './learning-field-balance-input';
import {readFieldBalanceRequest,type FieldBalanceStarted,type FieldBalanceSuccess,type FieldBalanceFailure} from './learning-field-balance-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateFieldBalanceRequest(value:unknown,started?:(message:FieldBalanceStarted)=>void):FieldBalanceSuccess|FieldBalanceFailure{
 const parsed=readFieldBalanceRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'field-balance'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'field-balance',input,inputKey:fieldBalanceInputKey(input),result:calculateFieldBalance(input)})as FieldBalanceSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
