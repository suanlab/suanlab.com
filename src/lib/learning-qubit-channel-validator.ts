/** Independent channel response check: indexed superoperator contractions and
 * pivoted Schur PSD checks, never the product channel/matrix-product engine.
 * Shared code is limited to exact field/scalar arithmetic and wire conversion.
 */
import type { QubitMap, QubitInstrument } from './learning-qubit-channel';
import { add,sub,mul,div } from './learning-quantum-hilbert-core';
import { zero,one,f,quadratic,Z,O,plus,times,conjugate,equal,minus,divide,qSign,
  circuitScalarWire,type M,type C,type F } from './learning-qubit-channel-algebra';

const sum=(xs:C[])=>xs.reduce(plus,Z);
const matrix=(n:number,entry:(i:number,j:number)=>C):M=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>entry(i,j)));
const wire=(a:M)=>a.map(row=>row.map(circuitScalarWire));
const identity=(n:number)=>matrix(n,(i,j)=>i===j?O:Z);
const trace=(a:M)=>sum(a.map((row,i)=>row[i]));
const hermitian=(a:M)=>a.every((row,i)=>row.every((z,j)=>equal(z,conjugate(a[j][i]))));
const same=(a:M,b:M)=>a.every((row,i)=>row.every((z,j)=>equal(z,b[i][j])));

/** Zero diagonal with a nonzero row cannot belong to a PSD matrix. Positive
 * pivots permit an exact Schur complement; no tolerance or eigenvalue rounding.
 */
export function channelValidatorPSD(a:M):boolean {
  if(!hermitian(a))return false;
  let remaining=a.map(row=>row.slice());
  while(remaining.length){
    if(remaining.some((row,i)=>qSign(row[i].re)<0))return false;
    for(let i=0;i<remaining.length;i++)if(qSign(remaining[i][i].re)===0&&remaining[i].some(z=>!equal(z,Z)))return false;
    const pivot=remaining.findIndex((row,i)=>qSign(row[i].re)>0);
    if(pivot<0)return true;
    const indices=remaining.map((_,i)=>i).filter(i=>i!==pivot),old=remaining;
    remaining=indices.map(i=>indices.map(j=>minus(old[i][j],divide(times(old[i][pivot],old[pivot][j]),old[pivot][pivot].re))));
  }
  return true;
}

function permutationDeterminant(a:M):C {
  let total=Z;
  const visit=(permutation:number[],available:number[])=>{
    if(!available.length){
      let inversions=0,term=O;
      for(let i=0;i<permutation.length;i++){
        term=times(term,a[i][permutation[i]]);
        for(let j=i+1;j<permutation.length;j++)if(permutation[i]>permutation[j])inversions++;
      }
      total=inversions%2?minus(total,term):plus(total,term);return;
    }
    available.forEach(j=>visit([...permutation,j],available.filter(k=>k!==j)));
  };
  visit([],a.map((_,i)=>i));return total;
}

function coefficient(map:QubitMap,a:number,b:number,i:number,j:number):C {
    if(map.kind==='transpose')return a===j&&b===i?O:Z;
    if(map.kind==='matrix-units')return map.images[2*i+j][a][b];
    return sum(map.terms.map(t=>times(times(t.operator[a][i],conjugate(t.operator[b][j])),{re:quadratic(t.weight),im:quadratic(zero)})));
}
const independentChoi=(map:QubitMap)=>matrix(4,(r,s)=>coefficient(map,r%2,s%2,Math.floor(r/2),Math.floor(s/2)));
const independentImage=(map:QubitMap,rho:M)=>matrix(2,(a,b)=>sum([0,1,2,3].map(k=>times(coefficient(map,a,b,Math.floor(k/2),k%2),rho[Math.floor(k/2)][k%2]))));
const independentDualIdentity=(map:QubitMap)=>matrix(2,(i,j)=>sum([0,1].map(a=>conjugate(coefficient(map,a,a,i,j)))));
function independentResponse(map:QubitMap,rho:M){
  const J=independentChoi(map),output=independentImage(map,rho);
  const marginal=matrix(2,(i,j)=>sum([0,1].map(a=>coefficient(map,a,a,i,j))));
  const dualI=independentDualIdentity(map);
  const unital=matrix(2,(a,b)=>sum([0,1].map(i=>coefficient(map,a,b,i,i))));
  const hp=hermitian(J),cp=channelValidatorPSD(J),tp=same(marginal,identity(2));
  const half={re:quadratic(f(BigInt(1),BigInt(2))),im:quadratic(zero)};
  const bell=J.map(row=>row.map(z=>times(z,half)));
  const minors=Array.from({length:15},(_,k)=>{
    const mask=k+1,indices=[0,1,2,3].filter(j=>(mask&(1<<j))!==0);
    return {mask,value:circuitScalarWire(permutationDeterminant(indices.map(i=>indices.map(j=>J[i][j]))))};
  });
  return {schemaVersion:1,choiConvention:'reference-output-unnormalized',input:wire(rho),choi:wire(J),principalMinors:minors,
    properties:{hermiticityPreserving:hp,completelyPositive:cp,tracePreserving:tp,
      traceNonincreasing:channelValidatorPSD(matrix(2,(i,j)=>minus(i===j?O:Z,dualI[i][j]))),unital:same(unital,identity(2)),quantumChannel:cp&&tp},
    choiOutputPartialTrace:wire(marginal),dualIdentity:wire(dualI),unitalImage:wire(unital),output:wire(output),outputTrace:circuitScalarWire(trace(output)),
    outputIsDensity:channelValidatorPSD(output)&&equal(trace(output),O),bellOutput:wire(bell),bellOutputIsDensity:channelValidatorPSD(bell)&&equal(trace(bell),O),
    referenceMarginal:wire(matrix(2,(i,j)=>sum([0,1].map(a=>bell[2*i+a][2*j+a]))))};
}

/** Strict observation comparison rejects missing/extra fields and even a valid
 * but mathematically unrelated replacement value. Key order is irrelevant.
 */
export function exactResponseDifferences(expected:unknown,actual:unknown):string[]{
  const differences:string[]=[];
  const compare=(a:unknown,b:unknown,path:string)=>{
    if(differences.length>=16)return;
    if(a===null||typeof a!=='object'){if(a!==b)differences.push(path);return;}
    if(b===null||typeof b!=='object'||Array.isArray(a)!==Array.isArray(b)){differences.push(path);return;}
    if(Array.isArray(a)){
      if(!Array.isArray(b)||a.length!==b.length){differences.push(path);return;}
      if(Object.keys(b).length!==b.length){differences.push(path);return;}
      a.forEach((v,i)=>compare(v,b[i],`${path}[${i}]`));return;
    }
    const keys=Object.keys(a),other=Object.keys(b);
    if(keys.length!==other.length||keys.some(k=>!Object.prototype.hasOwnProperty.call(b,k))){differences.push(path);return;}
    keys.forEach(k=>compare((a as Record<string,unknown>)[k],(b as Record<string,unknown>)[k],`${path}.${k}`));
  };
  try{compare(expected,actual,'response');}catch{differences.push('response.unreadable');}
  return differences;
}

export function verifyQubitChannelReport(map:QubitMap,rho:M,response:unknown){
  try{
    if(rho.length!==2||rho.some(row=>row.length!==2)||!channelValidatorPSD(rho)||!equal(trace(rho),O))return {valid:false,errors:['input.density']};
    const expected=independentResponse(map,rho),errors=exactResponseDifferences(expected,response);
    return {valid:errors.length===0,errors};
  }catch{return {valid:false,errors:['input.or.response']};}
}

const density=(rho:M)=>rho.length===2&&rho.every(row=>row.length===2)&&channelValidatorPSD(rho)&&equal(trace(rho),O);
function verified(expected:()=>unknown,response:unknown){
  try{const errors=exactResponseDifferences(expected(),response);return {valid:errors.length===0,errors};}
  catch{return {valid:false,errors:['input.or.response']};}
}
const contract=(A:M,rho:M):M=>matrix(A.length,(a,b)=>sum([0,1,2,3].map(k=>times(times(A[a][Math.floor(k/2)],rho[Math.floor(k/2)][k%2]),conjugate(A[b][k%2])))));
const gramResidual=(A:M):M=>matrix(A[0].length,(i,j)=>minus(sum(A.map(row=>times(conjugate(row[i]),row[j]))),i===j?O:Z));
const partial=(a:M,keepFirst:boolean):M=>matrix(2,(i,j)=>sum([0,1].map(b=>keepFirst?a[2*i+b][2*j+b]:a[2*b+i][2*b+j])));

export function verifyEnvironmentReport(rho:M,t:F,response:unknown){
  return verified(()=>{
    if(!density(rho)||t.n<BigInt(0)||t.n>t.d||t.d<=BigInt(0))throw new RangeError('Invalid preparation');
    const square=mul(t,t),den=add(one,square),c={re:quadratic(div(sub(one,square),den)),im:quadratic(zero)},s={re:quadratic(div(mul(f(BigInt(2)),t),den)),im:quadratic(zero)};
    const V:M=[[O,Z],[Z,s],[Z,c],[Z,Z]],U:M=[[O,Z,Z,Z],[Z,c,s,Z],[Z,minus(Z,s),c,Z],[Z,Z,Z,O]];
    const K0:M=[[O,Z],[Z,c]],K1:M=[[Z,s],[Z,Z]];
    const h={re:quadratic(zero,f(BigInt(1),BigInt(2))),im:quadratic(zero)};
    const mixed=[matrix(2,(i,j)=>times(plus(K0[i][j],K1[i][j]),h)),matrix(2,(i,j)=>times(minus(K0[i][j],K1[i][j]),h))];
    const map:QubitMap={kind:'operator-sum',terms:[K0,K1].map(operator=>({weight:one,operator}))};
    const mixmap:QubitMap={kind:'operator-sum',terms:mixed.map(operator=>({weight:one,operator}))};
    const joint=contract(V,rho),system=partial(joint,true),env=partial(joint,false);
    const channel=independentResponse(map,rho);
    return {schemaVersion:1,tensorOrder:'system-environment',isometry:wire(V),unitaryCompletion:wire(U),
      isometryResidual:wire(gramResidual(V)),unitaryResidual:wire(gramResidual(U)),joint:wire(joint),system:wire(system),environment:wire(env),
      reducedEqualsKraus:same(system,independentImage(map,rho)),mixedKraus:mixed.map(wire),mixedChannelEqualsOriginal:same(independentChoi(map),independentChoi(mixmap)),
      referenceOutput:channel.bellOutput,referenceMarginal:channel.referenceMarginal};
  },response);
}

export function verifyInstrumentReport(instrument:QubitInstrument,rho:M,correctZ:boolean,response:unknown){
  return verified(()=>{
    if(!density(rho)||instrument.outcomes.length!==2)throw new RangeError('Invalid instrument input');
    const maps:QubitMap[]=instrument.outcomes.map(terms=>({kind:'operator-sum',terms}));
    const effects=maps.map(independentDualIdentity);
    if(!maps.every(map=>channelValidatorPSD(independentChoi(map)))||!same(matrix(2,(i,j)=>plus(effects[0][i][j],effects[1][i][j])),identity(2)))throw new RangeError('Invalid instrument');
    const outputs=maps.map(map=>independentImage(map,rho));
    const forgotten=matrix(2,(i,j)=>plus(outputs[0][i][j],outputs[1][i][j]));
    const cq=matrix(4,(i,j)=>Math.floor(i/2)===Math.floor(j/2)?outputs[Math.floor(i/2)][i%2][j%2]:Z);
    const corrected=correctZ?matrix(2,(i,j)=>plus(outputs[0][i][j],outputs[1][1-i][1-j])):forgotten;
    return {schemaVersion:1,recordOrder:'record-system',correctZ,branches:outputs.map((output,i)=>{
      const p=trace(output);return {label:i,probability:circuitScalarWire(p),effect:wire(effects[i]),unnormalized:wire(output),conditional:qSign(p.re)===0?null:wire(output.map(row=>row.map(z=>divide(z,p.re))))};
    }),classicalQuantum:wire(cq),forgotten:wire(forgotten),feedforward:wire(corrected)};
  },response);
}

export function verifyDeferredMeasurementReport(rho:M,response:unknown){
  return verified(()=>{
    if(!density(rho))throw new RangeError('Invalid input');
    // The two controlled permutations transfer the input coherences to C and
    // reset B. This direct state-transfer identity avoids either circuit path.
    const coherent=matrix(4,(i,j)=>i<2&&j<2?rho[i][j]:Z);
    const dephased=matrix(4,(i,j)=>i<2&&i===j?rho[i][i]:Z);
    const recordSystem=matrix(4,(i,j)=>i===j&&i%2===0?rho[i/2][i/2]:Z);
    const reset:M=[[O,Z],[Z,Z]];
    return {schemaVersion:1,tensorOrder:'system-record',coherent:wire(coherent),dephased:wire(dephased),recordSystem:wire(recordSystem),classicalCorrected:wire(reset),system:wire(reset),record:wire(rho)};
  },response);
}
