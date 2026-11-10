/** Independent cq matrix/spectral reference and certified selected-measurement information. */
import{snapshotQuantumInformationInput,quantumInformationInputKey}from'./learning-quantum-information-input';
import{snapshotManyBodyWire}from'./learning-many-body-owned-wire';
import{f,zero,one,add,sub,eq,type F}from'./learning-quantum-hilbert-core';
import{referencePoint as point,referenceAdd as plus,referenceSubtract as minus,referenceScale as scale,referenceClip as clip,referenceDensity,referenceMatrixCombination,referenceQubitEntropy,referenceBinaryEntropy,referenceHelstromProbabilities,referenceCommuting}from'./learning-quantum-information-reference-qubit';
import{informationRequire as ensure,informationRecord as record,informationList as list,informationObservationRange as reported,InformationResponseInvalid,createInformationObservationChecks}from'./learning-quantum-information-response-checks';
import type{QuantumInformationCalculation}from'./learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'ensemble'}>;
const rat=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const contract='Exact rational physicality; directed512-bit spectral roots and256-bit logarithms in bits. A selected measurement is evaluated, not optimized for accessible information. Nonnegative/Holevo intervals are intersected with known analytic bounds; raw measurement information is retained. No channel capacity, Worker, public input reader or UI.';
export type EnsembleResponseValidation={ok:true;response:Reply;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumEnsembleResponse(request:unknown,response:unknown):EnsembleResponseValidation{
 try{
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='ensemble','$request','Ensemble request required');
  const owned=snapshotManyBodyWire(response),envelope=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(envelope.schemaVersion===1&&envelope.kind==='ensemble','$','Wrong response version/family');
  ensure(quantumInformationInputKey(envelope.input)===quantumInformationInputKey(input),'$.input','Original request differs');
  const d=record(envelope.result,['prior','averageBloch','measurement','conditionalOutcomeProbabilities','jointProbabilities','stateEntropies','averageStateEntropy','classicalLabelEntropy','outputEntropy','conditionalOutputEntropy','selectedMeasurementInformation','holevoInformation','holevoGap','rawMeasurementInformation','rawHolevoInformation','approximationContract'],'$.result');
  const rho0=referenceDensity(input.bloch0.map(rat)),rho1=referenceDensity(input.bloch1.map(rat)),p=rat(input.prior),other=sub(one,p),average=referenceMatrixCombination(rho0,p,rho1,other);
  const equal=rho0.every((row,i)=>row.every((z,j)=>eq(z.re,rho1[i][j].re)&&eq(z.im,rho1[i][j].im))),certain=eq(p,zero)||eq(p,one);
  const s0=referenceQubitEntropy(rho0),s1=referenceQubitEntropy(rho1),sAverage=referenceQubitEntropy(average),hPrior=referenceBinaryEntropy(point(p));
  const helstrom=input.measurement==='helstrom'?referenceHelstromProbabilities(rho0,rho1,p):null;
  const probabilities=helstrom?helstrom.probabilities:[point(rho0[0][0].re),point(rho1[0][0].re)];
  const outputZero=clip(plus(scale(probabilities[0],p),scale(probabilities[1],other))),outputEntropy=referenceBinaryEntropy(outputZero);
  const conditional=plus(scale(referenceBinaryEntropy(probabilities[0]),p),scale(referenceBinaryEntropy(probabilities[1]),other));
  const equalRows=input.measurement==='z'?eq(rho0[0][0].re,rho1[0][0].re):helstrom?.constant===true;
  const chi=equal||certain?point(zero):clip(minus(sAverage,plus(scale(s0,p),scale(s1,other))),zero,hPrior.upper);
  const information=equal||certain||equalRows?point(zero):clip(minus(outputEntropy,conditional),zero,hPrior.upper);
  const diagonal=[rho0,rho1].every(rho=>eq(rho[0][1].re,zero)&&eq(rho[0][1].im,zero));
  const sufficient=input.measurement==='z'?diagonal:helstrom?.constant===false&&referenceCommuting(rho0,rho1);
  const gap=equal||certain||sufficient?point(zero):clip(minus(chi,information));
  const checks=createInformationObservationChecks();checks.exact(d.prior,p,'$.result.prior');
  const averageBloch=[add(average[0][1].re,average[1][0].re),sub(average[1][0].im,average[0][1].im),sub(average[0][0].re,average[1][1].re)];
  list(d.averageBloch,3,'$.result.averageBloch').forEach((x,i)=>checks.exact(x,averageBloch[i],`$.result.averageBloch[${i}]`));
  ensure(d.measurement===input.measurement,'$.result.measurement','Wrong selected measurement');
  for(const name of ['conditionalOutcomeProbabilities','jointProbabilities']as const)list(d[name],2,`$.result.${name}`).forEach((row,i)=>list(row,2,`$.result.${name}[${i}]`).forEach((x,j)=>{
   const probability=j===0?probabilities[i]:minus(point(one),probabilities[i]),expected=name==='jointProbabilities'?scale(probability,i===0?p:other):probability;
   checks.check(x,expected,`$.result.${name}[${i}][${j}]`);
  }));
  list(d.stateEntropies,2,'$.result.stateEntropies').forEach((x,i)=>checks.check(x,i===0?s0:s1,`$.result.stateEntropies[${i}]`));
  for(const[name,expected]of [['averageStateEntropy',sAverage],['classicalLabelEntropy',hPrior],['outputEntropy',outputEntropy],['conditionalOutputEntropy',conditional],['selectedMeasurementInformation',information],['holevoInformation',chi],['holevoGap',gap],['rawMeasurementInformation',information],['rawHolevoInformation',chi]]as const)checks.check(d[name],expected,`$.result.${name}`);
  // Bind observable arithmetic without invoking the producer: raw intervals
  // must be the differences of the displayed entropy enclosures, before clipping.

  const relation=(value:unknown,expected:ReturnType<typeof point>,path:string)=>{const actual=reported(value,path);ensure(eq(actual.lower,expected.lower)&&eq(actual.upper,expected.upper),path,'Observable interval arithmetic is inconsistent');};
  const conditionalRows=list(d.conditionalOutcomeProbabilities,2,'$.result.conditionalOutcomeProbabilities').map((row,i)=>list(row,2,`$.result.conditionalOutcomeProbabilities[${i}]`));
  const jointRows=list(d.jointProbabilities,2,'$.result.jointProbabilities').map((row,i)=>list(row,2,`$.result.jointProbabilities[${i}]`));
  for(let i=0;i<2;i++){
   relation(conditionalRows[i][1],minus(point(one),reported(conditionalRows[i][0],'$conditional')),`$.result.conditionalOutcomeProbabilities[${i}][1]`);
   for(let j=0;j<2;j++)relation(jointRows[i][j],scale(reported(conditionalRows[i][j],'$conditional'),i===0?p:other),`$.result.jointProbabilities[${i}][${j}]`);
  }
  const entropies=list(d.stateEntropies,2,'$.result.stateEntropies');
  const rawChi=minus(reported(d.averageStateEntropy,'$average'),plus(scale(reported(entropies[0],'$state0'),p),scale(reported(entropies[1],'$state1'),other)));
  const rawInformation=minus(reported(d.outputEntropy,'$output'),reported(d.conditionalOutputEntropy,'$conditionalEntropy'));
  relation(d.rawHolevoInformation,rawChi,'$.result.rawHolevoInformation');relation(d.rawMeasurementInformation,rawInformation,'$.result.rawMeasurementInformation');
  relation(d.holevoInformation,equal||certain?point(zero):clip(rawChi,zero,reported(d.classicalLabelEntropy,'$prior').upper),'$.result.holevoInformation');
  relation(d.selectedMeasurementInformation,equal||certain?point(zero):clip(rawInformation,zero,reported(d.holevoInformation,'$holevo').upper),'$.result.selectedMeasurementInformation');
  relation(d.holevoGap,clip(minus(reported(d.holevoInformation,'$holevo'),reported(d.selectedMeasurementInformation,'$information'))),'$.result.holevoGap');
  ensure(d.approximationContract===contract,'$.result.approximationContract','Wrong selected-measurement contract');
  return{ok:true,response:owned as Reply,checkedObservations:checks.count};
 }catch(error){return{ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid ensemble response'};}
}
