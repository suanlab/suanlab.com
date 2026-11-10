import{rayleighModel}from'./learning-rayleigh';
import{ritzModel}from'./learning-ritz';
import{quantumVariationalRecord,isQuantumVariationalInput,isQuantumVariationalRequestEnvelope,snapshotQuantumVariationalInput,quantumVariationalInputKey,quantumVariationalKinds,type QuantumVariationalResponse}from'./learning-quantum-variational-protocol';
export function evaluateQuantumVariationalRequest(value:unknown):QuantumVariationalResponse{
 const descriptor=quantumVariationalRecord(value)?Object.getOwnPropertyDescriptor(value,'requestId'):undefined,rawId=descriptor&&Object.hasOwn(descriptor,'value')?descriptor.value:null,requestId=Number.isSafeInteger(rawId)&&rawId>0?rawId as number:null;
 if(!isQuantumVariationalRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isQuantumVariationalInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotQuantumVariationalInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:quantumVariationalInputKey(input)};return input.kind===quantumVariationalKinds[0]?{...base,kind:input.kind,input,result:rayleighModel(input.parameters)}:{...base,kind:input.kind,input,result:ritzModel(input.parameters)};}catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
