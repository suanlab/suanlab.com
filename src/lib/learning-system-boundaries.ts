import type {Vec2} from './learning-simulations';
const B=BigInt,Z=B(0),SCALE=B(10000);
function integer(v:number,lo:number,hi:number){if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use integer controls in the stated system-boundary domain.');}
function absolute(v:bigint){return v<Z?-v:v;}
interface RawFrame {step:number;mass:bigint[];den:bigint;incoming:bigint;outgoing:bigint;leak:bigint;cumulativeIncoming:bigint;cumulativeOutgoing:bigint;cumulativeLeak:bigint;}
function rawHistory(a:number,b:number,p:number,leak:number):RawFrame[]{
 const frames:RawFrame[]=[{step:0,mass:[B(a),B(b),Z],den:B(1),incoming:Z,outgoing:Z,leak:Z,cumulativeIncoming:Z,cumulativeOutgoing:Z,cumulativeLeak:Z}];
 for(let n=1;n<=20;n++){const q=frames[n-1],[A0,B0,E0]=q.mass,P=B(p),L=B(leak),incoming=B(100)*P*B0,outgoing=B(100)*P*A0,mixedB=P*A0+(B(100)-P)*B0,lost=L*mixedB;
  frames.push({step:n,mass:[B(100)*((B(100)-P)*A0+P*B0),(B(100)-L)*mixedB,SCALE*E0+lost],den:SCALE*q.den,incoming,outgoing,leak:lost,cumulativeIncoming:SCALE*q.cumulativeIncoming+incoming,cumulativeOutgoing:SCALE*q.cumulativeOutgoing+outgoing,cumulativeLeak:SCALE*q.cumulativeLeak+lost});
 }
 return frames;
}
const value=(n:bigint,d:bigint)=>Number(n/d)+Number(n%d)/Number(d);
const stored=(f:RawFrame,cut:number)=>cut===0?f.mass[0]:cut===1?f.mass[0]+f.mass[1]:f.mass.reduce((a,b)=>a+b,Z);
export function boundaryBalance(p:number[]){
 if(p.length!==6)throw new Error('Use initial A/B masses, exchange/leak percentages, boundary code and query step.');const[a,b,exchange,leak,cut,query]=p;integer(a,0,10);integer(b,0,10);integer(exchange,0,100);integer(leak,0,100);integer(cut,0,2);integer(query,0,20);
 const raw=rawHistory(a,b,exchange,leak),initial=cut===0?a:a+b,history=raw.map(f=>{const mass=stored(f,cut),I=cut===0?f.cumulativeIncoming:Z,O=cut===0?f.cumulativeOutgoing:cut===1?f.cumulativeLeak:Z,change=mass-B(initial)*f.den,residual=change-I+O,lastI=cut===0?f.incoming:Z,lastO=cut===0?f.outgoing:cut===1?f.leak:Z;return{step:f.step,amounts:f.mass.map(n=>value(n,f.den)),massNumerators:f.mass.map(String),denominator:f.den.toString(),stored:value(mass,f.den),flowBasedStored:value(B(initial)*f.den+I-O,f.den),change:value(change,f.den),cumulativeIn:value(I,f.den),cumulativeOut:value(O,f.den),lastIn:f.step===0?null:value(lastI,f.den),lastOut:f.step===0?null:value(lastO,f.den),lastLeak:f.step===0?null:value(f.leak,f.den),exactResidualNumerator:residual.toString(),balanceHolds:residual===Z,totalConserved:f.mass.reduce((s,n)=>s+n,Z)===B(a+b)*f.den,nonnegative:f.mass.every(n=>n>=Z),proof:{storedNumerator:mass.toString(),changeNumerator:change.toString(),incomingNumerator:I.toString(),outgoingNumerator:O.toString(),denominator:f.den.toString()}};}),selected=history[query];
 return{a,b,exchange,leak,cut,query,initialMass:initial,initialTotal:a+b,history,selected,structurallyMassClosed:cut===0?exchange===0:cut===1?leak===0:true,allBalancesHold:history.every(f=>f.balanceHolds),allMassesNonnegative:history.every(f=>f.nonnegative),allExpandedTotalsConserved:history.every(f=>f.totalConserved),curves:[history.map(f=>[f.step,f.stored] as Vec2),history.map(f=>[f.step,f.flowBasedStored] as Vec2),history.map(f=>[f.step,a+b] as Vec2)]};
}
const observed=(f:RawFrame,sensor:number)=>sensor===0?f.mass[0]:sensor===1?f.mass[1]:sensor===2?f.mass[0]+f.mass[1]:f.mass[0]-f.mass[1];
export function observationInformation(p:number[]){
 if(p.length!==5)throw new Error('Use exchange/leak percentages, sensor code, first recorded step and last recorded step.');const[exchange,leak,sensor,start,end]=p;integer(exchange,0,100);integer(leak,0,100);integer(sensor,0,3);integer(start,0,20);integer(end,0,20);
 const worlds=[rawHistory(8,2,exchange,leak),rawHistory(2,8,exchange,leak)],steps=Array.from({length:Math.max(0,end-start+1)},(_,i)=>start+i),records=steps.map(n=>{const a=observed(worlds[0][n],sensor),b=observed(worlds[1][n],sensor),den=worlds[0][n].den;return{step:n,values:[value(a,den),value(b,den)],numerators:[a.toString(),b.toString()],denominator:den.toString(),exactEqual:a===b,absoluteDifference:value(absolute(a-b),den)};}),hasObservations=records.length>0,sameRecordedOutputs=hasObservations?records.every(f=>f.exactEqual):null,firstDistinguishingStep=records.find(f=>!f.exactEqual)?.step??null,maxRecordedDifference=hasObservations?Math.max(...records.map(f=>f.absoluteDifference)):null,roundedSamplesCoincide=hasObservations?records.every(f=>f.values[0]===f.values[1]):null,endStates=worlds.map(w=>w[end].mass.map(n=>value(n,w[end].den))),sameEndState=worlds[0][end].mass.every((n,i)=>n===worlds[1][end].mass[i]),latentA=worlds.map(w=>w.map(f=>[f.step,value(f.mass[0],f.den)] as Vec2)),curves=worlds.map((_,i)=>records.map(f=>[f.step,f.values[i]] as Vec2));
 return{exchange,leak,sensor,start,end,initialStates:[[8,2,0],[2,8,0]],endStates,sameEndState,records,hasObservations,sameRecordedOutputs,firstDistinguishingStep,maxRecordedDifference,roundedSamplesCoincide,latentA,curves};
}
