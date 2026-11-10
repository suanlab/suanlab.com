/** Producer entry; Worker-side only. */
import {calculateScalarField} from './learning-scalar-field';
import {scalarFieldInputKey} from './learning-scalar-field-input';
import {readScalarFieldRequest,type ScalarFieldStarted,type ScalarFieldSuccess,type ScalarFieldFailure} from './learning-scalar-field-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateScalarFieldRequest(value:unknown,started?:(message:ScalarFieldStarted)=>void):ScalarFieldSuccess|ScalarFieldFailure{
 const parsed=readScalarFieldRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'scalar-field'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'scalar-field',input,inputKey:scalarFieldInputKey(input),result:calculateScalarField(input)})as ScalarFieldSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
