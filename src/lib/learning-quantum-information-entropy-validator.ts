/** Independent constructed-state and finite-source certificates; Worker/CLI only. */
import {snapshotQuantumInformationInput,quantumInformationInputKey}from'./learning-quantum-information-input';
import {snapshotManyBodyWire}from'./learning-many-body-owned-wire';
import {f,zero,one,two,add,sub,mul,div,eq,le,type F,type R}from'./learning-quantum-hilbert-core';
import {independentInformationEntropy}from'./learning-quantum-information-reference-log';
import {informationRequire as ensure,informationRecord as record,informationList as list,informationObservationRange as reported,InformationResponseInvalid,createInformationObservationChecks}from'./learning-quantum-information-response-checks';
import type{QuantumInformationCalculation}from'./learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'entropy'}>;
const rat=(s:string)=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const shift=(r:R,x:F):R=>({lower:add(r.lower,x),upper:add(r.upper,x)});
const difference=(a:R,b:R):R=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
const point=(x:F):R=>({lower:x,upper:x});
const jointContract='Constructed rho=c[rho_Bell_dephased(t)]+(1-c)I4/4, where rho_Bell_dephased(t) has00/11 populations1/2 and coherence t/2. Exact rational spectra and PPT classification;256-bit directed entropy logs. Partial-transpose negativity detects all entanglement for this2x2 family. No arbitrary-state entropy/eigensolver, Worker/public input reader/UI.';
const blockContract='Largest-eigenvalue rank-constrained projector weight for a binary IID source. Equal eigenvalues use increasing Hamming weight as a canonical tie order. A zero rank has no nonzero encoding space. No encoder, decoder, entanglement-fidelity guarantee, typicality cutoff or asymptotic converse is computed.';
export type EntropyResponseValidation={ok:true;response:Reply;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumEntropyResponse(request:unknown,response:unknown):EntropyResponseValidation{
 try{
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='entropy','$request','Entropy request required');
  const owned=snapshotManyBodyWire(response),envelope=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(envelope.schemaVersion===1&&envelope.kind==='entropy','$','Wrong family/version');
  ensure(quantumInformationInputKey(envelope.input)===quantumInformationInputKey(input),'$.input','Original request differs');
  const result=record(envelope.result,['joint','block'],'$.result'),checks=createInformationObservationChecks();
  const joint=record(result.joint,['correlationWeight','coherence','density','localDensityA','localDensityB','jointSpectrum','localEntropyA','localEntropyB','jointEntropy','conditionalEntropy','mutualInformation','dephasedJointEntropy','dephasingEntropyGain','rawDephasingEntropyGain','partialTransposeEigenvalues','negativity','entangled','contract'],'$.result.joint');
  const c=rat(input.correlationWeight),t=rat(input.coherence),four=f(BigInt(4));
  // Build the density as an explicit convex mixture, then extract eigenspaces.
  const rho=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>{
   const independent=i===j?div(sub(one,c),four):zero;
   const correlated=(i===0||i===3)&&(j===0||j===3)?mul(div(c,two),i===j?one:t):zero;
   return add(independent,correlated);
  }));
  const pt=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>rho[2*Math.floor(i/2)+j%2][2*Math.floor(j/2)+i%2]));
  const eigenvalue=(matrix:F[][],v:number[]):F=>{
   const vector=v.map(x=>f(BigInt(x))),norm=vector.reduce((s,x)=>add(s,mul(x,x)),zero);
   const applied=matrix.map(row=>row.reduce((s,x,j)=>add(s,mul(x,vector[j])),zero));
   const value=div(applied.reduce((s,x,i)=>add(s,mul(x,vector[i])),zero),norm);
   ensure(applied.every((x,i)=>eq(x,mul(value,vector[i]))),'$reference','Reference eigenspace failed');return value;
  };
  const spectrum=[[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]].map(v=>eigenvalue(rho,v));
  const ptSpectrum=[[1,0,0,0],[0,0,0,1],[0,1,1,0],[0,1,-1,0]].map(v=>eigenvalue(pt,v));
  const localA=Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>[0,1].reduce((s,k)=>add(s,rho[2*i+k][2*j+k]),zero)));
  const localB=Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>[0,1].reduce((s,k)=>add(s,rho[2*k+i][2*k+j]),zero)));
  ensure([localA,localB].every(local=>local.every((row,i)=>row.every((x,j)=>eq(x,i===j?div(one,two):zero)))),'$reference','Reference marginals are not maximally mixed');
  const matrix=(name:string,values:F[][])=>list(joint[name],values.length,`$.result.joint.${name}`).forEach((row,i)=>list(row,values.length,`$.result.joint.${name}[${i}]`).forEach((v,j)=>{
   const z=record(v,['real','imaginary'],`$.result.joint.${name}[${i}][${j}]`);
   checks.exact(z.real,values[i][j],`$.result.joint.${name}[${i}][${j}].real`);checks.exact(z.imaginary,zero,`$.result.joint.${name}[${i}][${j}].imaginary`);
  }));
  checks.exact(joint.correlationWeight,c,'$.result.joint.correlationWeight');checks.exact(joint.coherence,t,'$.result.joint.coherence');
  matrix('density',rho);matrix('localDensityA',localA);matrix('localDensityB',localB);
  for(const[name,values]of [['jointSpectrum',spectrum],['partialTransposeEigenvalues',ptSpectrum]]as const)list(joint[name],4,`$.result.joint.${name}`).forEach((v,i)=>checks.exact(v,values[i],`$.result.joint.${name}[${i}]`));
  checks.exact(joint.localEntropyA,one,'$.result.joint.localEntropyA');checks.exact(joint.localEntropyB,one,'$.result.joint.localEntropyB');
  const entropy=independentInformationEntropy(spectrum),dephased=independentInformationEntropy(rho.map((row,i)=>row[i]));
  const zeroCoherence=rho.every((row,i)=>row.every((x,j)=>i===j||eq(x,zero))),gain=zeroCoherence?point(zero):difference(dephased,entropy);
  checks.check(joint.jointEntropy,entropy,'$.result.joint.jointEntropy');
  checks.check(joint.conditionalEntropy,shift(entropy,f(BigInt(-1))),'$.result.joint.conditionalEntropy');
  checks.check(joint.mutualInformation,difference(point(two),entropy),'$.result.joint.mutualInformation');
  checks.check(joint.dephasedJointEntropy,dephased,'$.result.joint.dephasedJointEntropy');
  checks.check(joint.dephasingEntropyGain,gain,'$.result.joint.dephasingEntropyGain');
  checks.check(joint.rawDephasingEntropyGain,gain,'$.result.joint.rawDephasingEntropyGain');
  const negativity=ptSpectrum.filter(x=>x.n<BigInt(0)).reduce((s,x)=>sub(s,x),zero);
  checks.exact(joint.negativity,negativity,'$.result.joint.negativity');ensure(joint.entangled===!eq(negativity,zero),'$.result.joint.entangled','Wrong exact PPT classification');
  // Preserve the observable pre-clipping uncertainty, rather than accepting
  // any independently valid replacement for a field explicitly labeled raw.
  const rawGain=difference(reported(joint.dephasedJointEntropy,'$dephased'),reported(joint.jointEntropy,'$joint'));
  const rawReported=reported(joint.rawDephasingEntropyGain,'$.result.joint.rawDephasingEntropyGain');
  ensure(eq(rawReported.lower,rawGain.lower)&&eq(rawReported.upper,rawGain.upper),'$.result.joint.rawDephasingEntropyGain','Raw entropy difference interval is inconsistent');
  const clipped=zeroCoherence?point(zero):{lower:le(rawGain.lower,zero)?zero:rawGain.lower,upper:le(two,rawGain.upper)?two:rawGain.upper};
  const clippedReported=reported(joint.dephasingEntropyGain,'$.result.joint.dephasingEntropyGain');
  ensure(eq(clippedReported.lower,clipped.lower)&&eq(clippedReported.upper,clipped.upper),'$.result.joint.dephasingEntropyGain','Clipped entropy difference interval is inconsistent');
  ensure(joint.contract===jointContract,'$.result.joint.contract','Wrong constructed-state contract');
  const block=record(result.block,['copies','dimension','retainedRank','requiredQubits','retainedWeight','discardedWeight','groups','interpretation'],'$.result.block');
  const n=input.copies,rank=BigInt(input.retainedRank),p=rat(input.probabilityOne),other=sub(one,p),dimension=BigInt(2)**BigInt(n);
  ensure(block.copies===n&&block.dimension===String(dimension)&&block.retainedRank===input.retainedRank,'$.result.block','Wrong block/rank identity');
  // Pascal addition and repeated products are separate from the producer's
  // multiplicity division recurrence, exponentiation and numerical sorting.
  let counts=[BigInt(1)];for(let step=0;step<n;step++)counts=Array.from({length:counts.length+1},(_,k)=>(counts[k-1]??BigInt(0))+(counts[k]??BigInt(0)));
  const pPower=[one],otherPower=[one];for(let k=1;k<=n;k++){pPower.push(mul(pPower[k-1],p));otherPower.push(mul(otherPower[k-1],other));}
  const ascending=Array.from({length:n+1},(_,k)=>k),order=eq(p,one)?[n,...ascending.slice(0,n)]:le(p,other)?ascending:ascending.slice().reverse();
  const groups=list(block.groups,n+1,'$.result.block.groups');let cumulative=BigInt(0),weight=zero,fullWeight=zero;
  for(const[index,ones]of order.entries()){
   const d=record(groups[index],['ones','multiplicity','retained','eigenvalue'],`$.result.block.groups[${index}]`),count=counts[ones],available=rank>cumulative?rank-cumulative:BigInt(0),kept=available<count?available:count;
   const eigen=mul(pPower[ones],otherPower[n-ones]);cumulative+=count;weight=add(weight,mul(f(kept),eigen));fullWeight=add(fullWeight,mul(f(count),eigen));
   ensure(Object.is(d.ones,ones)&&d.multiplicity===String(count)&&d.retained===String(kept),`$.result.block.groups[${index}]`,'Wrong ordered type/count/rank selection');
   checks.exact(d.eigenvalue,eigen,`$.result.block.groups[${index}].eigenvalue`);
  }
  ensure(cumulative===dimension&&eq(fullWeight,one),'$reference','Pascal count/normalization inventory failed');
  const qubits=rank===BigInt(0)?null:rank===BigInt(1)?0:(rank-BigInt(1)).toString(2).length;
  ensure(Object.is(block.requiredQubits,qubits),'$.result.block.requiredQubits','Wrong integer memory size');
  checks.exact(block.retainedWeight,weight,'$.result.block.retainedWeight');checks.exact(block.discardedWeight,sub(one,weight),'$.result.block.discardedWeight');
  ensure(block.interpretation===blockContract,'$.result.block.interpretation','Wrong projector-weight interpretation');
  return{ok:true,response:owned as Reply,checkedObservations:checks.count};
 }catch(error){return{ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid entropy response'};}
}
