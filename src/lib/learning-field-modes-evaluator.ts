/** Producer entry; Worker-side only. */
import {calculateFieldModes} from './learning-field-modes';
import {fieldModesInputKey} from './learning-field-modes-input';
import {readFieldModesRequest,type FieldModesStarted,type FieldModesSuccess,type FieldModesFailure} from './learning-field-modes-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateFieldModesRequest(value:unknown,started?:(message:FieldModesStarted)=>void):FieldModesSuccess|FieldModesFailure{
 const parsed=readFieldModesRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'field-modes'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'field-modes',input,inputKey:fieldModesInputKey(input),result:calculateFieldModes(input)})as FieldModesSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
