/** Independent rank/hinge majorization, tail feasibility and explicit local
 * Bell dilution reference. No producer sorting, entropy or LOCC evaluator. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, one, add, sub, mul, div, eq, le, type F } from './learning-quantum-hilbert-core';
import { independentInformationEntropy } from './learning-quantum-information-reference-log';
import { fieldRational as rat, fieldPair, fieldComplex, fieldZero, fieldOne, fieldMultiply, fieldDensity, fieldWire, fieldMatrixWire, checkInformationExactTree as check } from './learning-quantum-information-reference-field';
import { informationRequire as ensure, informationRecord as record, informationList as list, informationObservationRange as reported, createInformationObservationChecks, InformationResponseInvalid } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply=Extract<QuantumInformationCalculation,{kind:'majorization'}>;
const contract='Exact descending Schmidt probability majorization and Vidal tail ratios for one bipartite pure-state copy under unrestricted LOCC. Zero target tails are skipped. Entropies use directed256-bit logarithms and do not decide the partial order. This certifies existence/optimal probability by the applicable theorems, not an implemented general LOCC protocol. No mixed-state, catalytic, approximate, asymptotic or channel-capacity claim; no Worker/public input reader/UI.';
function ordered(values:F[]) {
 ensure(eq(values.reduce(add,zero),one),'$request','Schmidt probabilities must sum exactly to one');
 // Stable descending ranks from all pairwise comparisons; no producer sort.
 const indices=values.map((x,i)=>({index:i,rank:values.filter((y,j)=>!le(y,x)||(eq(x,y)&&j<i)).length}));
 const order=Array.from({length:values.length},(_,rank)=>indices.find(x=>x.rank===rank)!.index);
 return {values:order.map(i=>values[i]),indices:order};
}
/** The hinge sums are piecewise linear in t. Their difference need only be
 * checked at the union of all breakpoints; outside it equal mass fixes the sign. */
function hingeMajorized(source:F[],target:F[]):boolean {
 const hinge=(xs:F[],t:F)=>xs.reduce((s,x)=>add(s,le(x,t)?zero:sub(x,t)),zero);
 return [...source,...target,zero].every(t=>le(hinge(source,t),hinge(target,t)));
}
function suffix(values:F[]):F[]{const result=Array.from({length:values.length+1},()=>zero);for(let i=values.length-1;i>=0;i--)result[i]=add(values[i],result[i+1]);return result;}
function direction(source:F[],target:F[]) {
 const a=suffix(source),b=suffix(target),deterministic=hingeMajorized(source,target);
 const prefixes=source.map((_,i)=>({k:i+1,source:sub(one,a[i+1]),target:sub(one,b[i+1]),gap:sub(a[i+1],b[i+1]),satisfied:le(b[i+1],a[i+1])}));
 ensure(deterministic===prefixes.every(p=>p.satisfied),'$reference','Hinge and prefix certificates disagree');
 let numerator=one,denominator=one;
 // Identify the binding inequality by cross-products, then divide only once.
 for(let i=0;i<source.length;i++)if(b[i].n>BigInt(0)&&le(mul(a[i],denominator),mul(numerator,b[i]))){numerator=a[i];denominator=b[i];}
 const optimum=div(numerator,denominator);
 ensure(a.every((x,i)=>le(mul(optimum,b[i]),x)),'$reference','Tail candidate is infeasible');
 ensure(eq(optimum,one)||a.some((x,i)=>b[i].n>BigInt(0)&&eq(x,mul(optimum,b[i]))),'$reference','No binding tail certificate');
 return {deterministic,firstViolation:prefixes.find(p=>!p.satisfied)?.k??null,prefixes,
  tails:source.map((_,i)=>({first:i+1,source:a[i],target:b[i],applicable:b[i].n>BigInt(0),ratio:b[i].n>BigInt(0)?div(a[i],b[i]):null})),optimum};
}
function compare(sourceRaw:F[],targetRaw:F[],actual:unknown) {
 const a=ordered(sourceRaw),b=ordered(targetRaw),dimension=Math.max(a.values.length,b.values.length);
 const pad=(xs:F[])=>Array.from({length:dimension},(_,i)=>xs[i]??zero),source=pad(a.values),target=pad(b.values);
 const forward=direction(source,target),reverse=direction(target,source),checks=createInformationObservationChecks();
 const d=record(actual,['dimension','source','target','sourceSortingIndices','targetSortingIndices','sourceSchmidtRank','targetSchmidtRank','sourceEntropy','targetEntropy','forward','reverse','relation','contract'],'$.result.comparison');
 for(const [k,value]of Object.entries({dimension,sourceSortingIndices:a.indices,targetSortingIndices:b.indices,sourceSchmidtRank:source.filter(x=>x.n>BigInt(0)).length,targetSchmidtRank:target.filter(x=>x.n>BigInt(0)).length,relation:forward.deterministic&&reverse.deterministic?'equivalent':forward.deterministic?'forward-only':reverse.deterministic?'reverse-only':'incomparable',contract}))check(d[k],value,`$.result.comparison.${k}`);
 for(const [name,xs]of [['source',source],['target',target]]as const){
  list(d[name],dimension,`$.result.comparison.${name}`).forEach((x,i)=>checks.exact(x,xs[i],`$.result.comparison.${name}[${i}]`));
  const entropyName=name==='source'?'sourceEntropy':'targetEntropy',path=`$.result.comparison.${entropyName}`;
  if(xs.filter(x=>x.n>BigInt(0)).length===1)checks.exact(d[entropyName],zero,path);else checks.check(d[entropyName],independentInformationEntropy(xs),path);
  const r=reported(d[entropyName],path);ensure(le(zero,r.lower)&&le(r.lower,f(BigInt(4))),path,'Entropy enclosure must be nonnegative and intersect[0,4]');
 }
 for(const [name,expected]of [['forward',forward],['reverse',reverse]]as const){
  const path=`$.result.comparison.${name}`,r=record(d[name],['deterministic','firstViolation','prefixes','tails','optimalStochasticSuccess'],path);
  check(r.deterministic,expected.deterministic,`${path}.deterministic`);check(r.firstViolation,expected.firstViolation,`${path}.firstViolation`);
  list(r.prefixes,dimension,`${path}.prefixes`).forEach((x,i)=>{
   const e=expected.prefixes[i],p=record(x,['k','source','target','targetMinusSource','satisfied'],`${path}.prefixes[${i}]`);
   check(p.k,e.k,`${path}.prefixes[${i}].k`);check(p.satisfied,e.satisfied,`${path}.prefixes[${i}].satisfied`);
   checks.exact(p.source,e.source,`${path}.prefixes[${i}].source`);checks.exact(p.target,e.target,`${path}.prefixes[${i}].target`);checks.exact(p.targetMinusSource,e.gap,`${path}.prefixes[${i}].targetMinusSource`);
  });
  list(r.tails,dimension,`${path}.tails`).forEach((x,i)=>{
   const e=expected.tails[i],p=record(x,['first','source','target','applicable','ratio'],`${path}.tails[${i}]`);
   check(p.first,e.first,`${path}.tails[${i}].first`);check(p.applicable,e.applicable,`${path}.tails[${i}].applicable`);
   checks.exact(p.source,e.source,`${path}.tails[${i}].source`);checks.exact(p.target,e.target,`${path}.tails[${i}].target`);
   if(e.ratio===null)ensure(p.ratio===null,`${path}.tails[${i}].ratio`,'Zero target tail must have null ratio');else checks.exact(p.ratio,e.ratio,`${path}.tails[${i}].ratio`);
  });
  checks.exact(r.optimalStochasticSuccess,expected.optimum,`${path}.optimalStochasticSuccess`);
 }
 return checks.count;
}
function dilution(a:F,b:F) {
 ensure(eq(add(mul(a,a),mul(b,b)),one),'$request','Bell amplitudes must satisfy a²+b²=1');
 const ca=fieldComplex(fieldPair(a)),cb=fieldComplex(fieldPair(b)),h=fieldComplex(fieldPair(zero,f(BigInt(1),BigInt(2))));
 const target=[ca,fieldZero,fieldZero,cb],restored=[fieldMultiply(ca,h),fieldZero,fieldZero,fieldMultiply(cb,h)];
 const swapped=[restored[3],fieldZero,fieldZero,restored[0]],density=fieldMatrixWire(fieldDensity(target));
 return {target:target.map(fieldWire),targetDensity:density,aliceOperators:[fieldMatrixWire([[ca,fieldZero],[fieldZero,cb]]),fieldMatrixWire([[cb,fieldZero],[fieldZero,ca]])],
  measurementCompleteness:fieldMatrixWire([[fieldOne,fieldZero],[fieldZero,fieldOne]]),
  branches:[restored,swapped].map((raw,outcome)=>({outcome,probability:['1/2','0'],raw:raw.map(fieldWire),corrected:restored.map(fieldWire),correctedDensity:density,bothPartiesApplyX:outcome===1})),
  contract:'One Bell pair to a|00>+b|11>, a,b nonnegative rational and a²+b²=1. Alice performs a two-outcome local measurement and announces its outcome; both parties apply X on outcome1. Each outcome has probability1/2. This explicit family does not synthesize general majorization/Vidal-optimal protocols; no Worker/public input reader/UI.'};
}
export type MajorizationResponseValidation={ok:true;response:Reply;checkedExactCoefficients:number;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumMajorizationResponse(request:unknown,response:unknown):MajorizationResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='majorization','$request','Majorization request required');
  const owned=snapshotManyBodyWire(response,16384),d=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(d.schemaVersion===1&&d.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(d.input)===quantumInformationInputKey(input),'$.input','Original spectra or dilution input differs');
  const r=record(d.result,input.mode==='compare'?['mode','comparison']:['mode','dilution'],'$.result');ensure(r.mode===input.mode,'$.result.mode','Wrong majorization mode');
  if(input.mode==='compare')return {ok:true,response:owned as Reply,checkedExactCoefficients:0,checkedObservations:compare(input.source.map(rat),input.target.map(rat),r.comparison)};
  const count=check(r.dilution,dilution(rat(input.a),rat(input.b)),'$.result.dilution');return {ok:true,response:owned as Reply,checkedExactCoefficients:count,checkedObservations:0};
 }catch(error){return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid majorization response'};}
}
