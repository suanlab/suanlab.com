/** Independent dense circuit evolution, reduced-coordinate insertion and sparse
 * Pauli actions. Never calls the product circuit runner or observation report.
 */
import type { CircuitGate } from './learning-qubit-circuit-exact';
import { f,zero,one,quadratic,complex,Z,O,plus,times,conjugate,equal,minus,divide,
  qAdd,qSub,qDiv,qSign,circuitScalarWire,type C,type Q,type M } from './learning-qubit-channel-algebra';
import { exactResponseDifferences } from './learning-qubit-channel-validator';

const sum=(values:C[])=>values.reduce(plus,Z);
const matrix=(n:number,entry:(i:number,j:number)=>C):M=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>entry(i,j)));
const eye=(n:number)=>matrix(n,(i,j)=>i===j?O:Z);
const wire=(a:M)=>a.map(row=>row.map(circuitScalarWire));
const realWire=(a:Q)=>circuitScalarWire(complex(a));
const norm=(state:readonly C[])=>state.reduce((s,z)=>qAdd(s,times(z,conjugate(z)).re),quadratic(zero));
const apply=(A:M,z:readonly C[])=>A.map(row=>sum(row.map((v,j)=>times(v,z[j]))));
const multiply=(A:M,B:M)=>matrix(A.length,(i,j)=>sum(A[i].map((v,k)=>times(v,B[k][j]))));
const dagger=(A:M)=>matrix(A.length,(i,j)=>conjugate(A[j][i]));
const density=(z:readonly C[],N:Q)=>matrix(z.length,(i,j)=>divide(times(z[i],conjugate(z[j])),N));
const purity=(A:M)=>sum(A.flatMap((row,i)=>row.map((z,j)=>times(z,A[j][i]))));
const sameMatrix=(A:M,B:M)=>A.every((row,i)=>row.every((z,j)=>equal(z,B[i][j])));
const i=complex(quadratic(zero),quadratic(one)),negativeI=complex(quadratic(zero),quadratic(f(BigInt(-1))));
const negative=complex(quadratic(f(BigInt(-1))));
const rootHalf=complex(quadratic(zero,f(BigInt(1),BigInt(2))));
const local:Record<string,M>={
  I:eye(2),X:[[Z,O],[O,Z]],Y:[[Z,negativeI],[i,Z]],Z:[[O,Z],[Z,negative]],
  H:[[rootHalf,rootHalf],[rootHalf,minus(Z,rootHalf)]],S:[[O,Z],[Z,i]],Sdg:[[O,Z],[Z,negativeI]],
  T:[[O,Z],[Z,complex(rootHalf.re,rootHalf.re)]],Tdg:[[O,Z],[Z,complex(rootHalf.re,[zero,f(BigInt(-1),BigInt(2))])]],
};
function embedded(sites:number,op:CircuitGate):M {
  if(Object.prototype.hasOwnProperty.call(local,op.gate)){
    let result:M=[[O]];
    for(let site=0;site<sites;site++){
      const block=site===op.target?local[op.gate]:eye(2),old=result;
      result=matrix(old.length*2,(a,b)=>times(old[Math.floor(a/2)][Math.floor(b/2)],block[a%2][b%2]));
    }return result;
  }
  const A=matrix(2**sites,()=>Z);
  for(let column=0;column<A.length;column++){
    const bits=column.toString(2).padStart(sites,'0').split('');let factor=O;
    if(op.gate==='SWAP')[bits[op.target],bits[op.other]]=[bits[op.other],bits[op.target]];
    else if((op.gate==='CX'||op.gate==='CZ')&&bits[op.control]==='1'){
      if(op.gate==='CX')bits[op.target]=bits[op.target]==='1'?'0':'1';
      else if(bits[op.target]==='1')factor=negative;
    }
    A[parseInt(bits.join(''),2)][column]=factor;
  }return A;
}
function inputCheck(sites:number,gates:readonly CircuitGate[],input:readonly C[]){
  if(!Number.isInteger(sites)||sites<1||sites>3||input.length!==2**sites||gates.length>16)throw new RangeError('Invalid finite circuit');
  const site=(s:number)=>Number.isInteger(s)&&s>=0&&s<sites;
  for(const op of gates){
    if(!site(op.target))throw new RangeError('Invalid target');
    if(op.gate==='CX'||op.gate==='CZ'){if(!site(op.control)||op.control===op.target)throw new RangeError('Invalid control');}
    else if(op.gate==='SWAP'){if(!site(op.other)||op.other===op.target)throw new RangeError('Invalid swap');}
    else if(!Object.prototype.hasOwnProperty.call(local,op.gate))throw new RangeError('Invalid gate');
  }
  for(const z of input)for(const part of [z.re,z.im])if(part.length!==2||part.some(v=>typeof v.n!=='bigint'||typeof v.d!=='bigint'||v.d<=BigInt(0)))throw new RangeError('Invalid scalar');
  if(qSign(norm(input))<=0)throw new RangeError('Zero input');
}
function reduceOne(state:readonly C[],N:Q,sites:number,site:number):M {
  const index=(bit:number,environment:number)=>{
    const bits=sites===1?[]:environment.toString(2).padStart(sites-1,'0').split('');
    bits.splice(site,0,String(bit));return parseInt(bits.join(''),2);
  };
  return matrix(2,(a,b)=>divide(sum(Array.from({length:2**(sites-1)},(_,env)=>times(state[index(a,env)],conjugate(state[index(b,env)])))),N));
}
function wordExpectation(state:readonly C[],N:Q,sites:number,word:Map<number,number>):C {
  let value=Z;
  state.forEach((z,column)=>{
    const bits=column.toString(2).padStart(sites,'0').split('');let factor=O;
    for(const [site,axis] of word){
      const bit=bits[site];
      if(axis===0||axis===1)bits[site]=bit==='0'?'1':'0';
      if(axis===1)factor=times(factor,bit==='0'?i:negativeI);
      else if(axis===2&&bit==='1')factor=times(factor,negative);
    }
    const destination=parseInt(bits.join(''),2);
    value=plus(value,times(conjugate(state[destination]),times(factor,z)));
  });return divide(value,N);
}
function expectedResponse(sites:number,gates:readonly CircuitGate[],input:readonly C[]){
  inputCheck(sites,gates,input);const N=norm(input),dimension=2**sites;
  let U=eye(dimension),state=input.slice();const allStates=[state];
  for(const op of gates){const A=embedded(sites,op);state=apply(A,state);allStates.push(state);U=multiply(A,U);}
  const prefixes=allStates.map((z,step)=>{
    const rho=density(z,N),reductions=Array.from({length:sites},(_,site)=>reduceOne(z,N,sites,site));
    const bloch=reductions.map(d=>[plus(d[0][1],d[1][0]),times(i,minus(d[0][1],d[1][0])),minus(d[0][0],d[1][1])]);
    const pairs=[];
    for(let first=0;first<sites;first++)for(let second=first+1;second<sites;second++){
      const correlations=Array.from({length:3},(_,a)=>Array.from({length:3},(_,b)=>wordExpectation(z,N,sites,new Map([[first,a],[second,b]]))));
      pairs.push({first,second,expectations:correlations.map(row=>row.map(circuitScalarWire)),connected:correlations.map((row,a)=>row.map((v,b)=>circuitScalarWire(minus(v,times(bloch[first][a],bloch[second][b])))))});
    }
    return {step,rawAmplitudes:z.map(circuitScalarWire),normSquared:realWire(norm(z)),normResidual:realWire(qSub(norm(z),N)),density:wire(rho),
      probabilities:z.map(v=>realWire(qDiv(times(v,conjugate(v)).re,N))),purity:circuitScalarWire(purity(rho)),
      singleQubits:reductions.map((d,site)=>({site,density:wire(d),purity:circuitScalarWire(purity(d)),bloch:bloch[site].map(circuitScalarWire)})),pauliPairs:pairs};
  });
  const residual=multiply(dagger(U),U).map((row,i)=>row.map((z,j)=>minus(z,i===j?O:Z)));
  const restored=apply(dagger(U),state);
  const overlap=divide(sum(input.map((v,j)=>times(conjugate(v),state[j]))),N);
  const inverseOverlap=divide(sum(input.map((v,j)=>times(conjugate(v),restored[j]))),N);
  return {schemaVersion:1,basisOrder:'site-zero-most-significant',gateOrder:'chronological',amplitudeConvention:'raw-with-exact-squared-norm',fidelityConvention:'squared-pure-state-overlap',
    sites,gates:gates.map(op=>({...op})),pauliOrder:['X','Y','Z'],inputNormSquared:realWire(N),prefixes,
    unitary:wire(U),unitarityResidual:wire(residual),unitaryVerified:residual.every(row=>row.every(z=>equal(z,Z))),
    overlapWithInput:circuitScalarWire(overlap),fidelityWithInput:realWire(times(overlap,conjugate(overlap)).re),
    sameAmplitudesAsInput:state.every((z,j)=>equal(z,input[j])),samePhysicalStateAsInput:sameMatrix(density(state,N),density(input,N)),
    inverseRawAmplitudes:restored.map(circuitScalarWire),inverseResidual:restored.map((z,j)=>circuitScalarWire(minus(z,input[j]))),
    inverseFidelity:realWire(times(inverseOverlap,conjugate(inverseOverlap)).re),inverseRecovered:restored.every((z,j)=>equal(z,input[j]))};
}

export function verifyQubitCircuitReport(sites:number,gates:readonly CircuitGate[],input:readonly C[],response:unknown){
  try{const errors=exactResponseDifferences(expectedResponse(sites,gates,input),response);return {valid:errors.length===0,errors};}
  catch{return {valid:false,errors:['input.or.response']};}
}
