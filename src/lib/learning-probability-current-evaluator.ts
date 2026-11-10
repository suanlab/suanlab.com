/** Worker-only model and exact linear-field sampling for display. */
import {circulatingGaussianModel}from'./learning-circulating-gaussian';
import {markovFraction as f,exactMarkovObservation as exact}from'./learning-finite-markov-exact';
import {currentRecord,isCurrentInput,isCurrentRequestEnvelope,snapshotCurrentInput,currentInputKey,type CurrentInput,type CurrentDiagram,type CurrentResponse}from'./learning-probability-current-protocol';
export function currentDiagram(input:CurrentInput):CurrentDiagram{
 const p=input.parameters,v=f(BigInt(p.diffusionNumerator),BigInt(p.relaxationNumerator)),g=f(BigInt(p.relaxationNumerator),BigInt(p.coefficientDenominator)),w=f(BigInt(p.angularNumerator),BigInt(p.angularDenominator));
 type F=typeof v;const mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),out=(a:F)=>exact(a.n,a.d);
 let L=f(BigInt(1),BigInt(1048576));while(L.n*L.n*v.d<BigInt(9)*v.n*L.d*L.d)L=mul(L,f(BigInt(2)));
 const probe=L.n>=BigInt(8)*L.d?L:f(BigInt(8));
 return{densityExtent:out(L),probeExtent:out(probe),fields:[L,probe].map(extent=>({extent:out(extent),samples:Array.from({length:49},(_,i)=>{const x=mul(extent,f(BigInt(i%7-3),BigInt(4))),y=mul(extent,f(BigInt(Math.floor(i/7)-3),BigInt(4))),gx=mul(g,x),gy=mul(g,y),wx=mul(w,x),wy=mul(w,y);return{position:[out(x),out(y)],drift:[out(neg(add(gx,wy))),out(add(neg(gy),wx))],diffusive:[out(gx),out(gy)],flow:[out(neg(wy)),out(wx)]};})}))};
}
export function evaluateCurrentRequest(v:unknown):CurrentResponse{
 const requestId=currentRecord(v)&&Number.isSafeInteger(v.requestId)&&(v.requestId as number)>0?v.requestId as number:null;
 if(!isCurrentRequestEnvelope(v))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isCurrentInput(v.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotCurrentInput(v.input);return{version:1,requestId:v.requestId,ok:true,kind:input.kind,input,inputKey:currentInputKey(input),result:circulatingGaussianModel(input.parameters),diagram:currentDiagram(input)};}catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
