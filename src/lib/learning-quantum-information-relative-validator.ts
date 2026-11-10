/** Independent matrix spectral logarithm and exact support check for qubit DPI examples.
 * Worker/CLI only. Shared rational and display primitives, no producer evaluator. */
import { snapshotQuantumInformationInput, quantumInformationInputKey } from './learning-quantum-information-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { f, zero, one, two, sub, mul, div, neg, eq, le, q, type F, type R } from './learning-quantum-hilbert-core';
import { independentInformationLog2 } from './learning-quantum-information-reference-log';
import { independentUnitRoot, referencePoint as point, referenceAdd as plus, referenceSubtract as minus, referenceScale as scale, referenceDensity, referenceDeterminant, referenceTraceProduct, referenceQubitEntropy, type ReferenceMatrix } from './learning-quantum-information-reference-qubit';
import { informationRequire as ensure, informationRecord as record, informationList as list, informationSame as same, informationObservationRange as reported, InformationResponseInvalid, createInformationObservationChecks } from './learning-quantum-information-response-checks';
import type { QuantumInformationCalculation } from './learning-quantum-information-calculation';
type Reply = Extract<QuantumInformationCalculation, {kind:'relative-entropy'}>;
const rat = (s:string):F => { const [n,d='1']=s.split('/'); return f(BigInt(n),BigInt(d)); };
const contract='Umegaki D(rho||sigma) in bits, exact rational support classification; directed512-bit roots and256-bit logarithms. The logarithm is spectral, not entrywise. No arbitrary-dimensional solver, Petz recovery, Worker, public input reader or UI.';
const interpretation='The same CPTP channel acts on both states. Infinite input divergence may become finite. Separate enclosures are evidence for examples, not a proof of general data processing or recovery.';
function product(a:R,b:R):R {
 const v=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];
 return {lower:v.reduce((x,y)=>le(x,y)?x:y),upper:v.reduce((x,y)=>le(x,y)?y:x)};
}
function logarithm(x:R):R {
 if(x.lower.n<=BigInt(0))throw new RangeError('Independent full-rank spectrum unresolved');
 return {lower:independentInformationLog2(x.lower).lower,upper:independentInformationLog2(x.upper).upper};
}
function nonnegative(x:R):R {
 if(x.upper.n<BigInt(0))throw new RangeError('Independent divergence is negative');
 return {lower:x.lower.n<BigInt(0)?zero:x.lower,upper:x.upper};
}
/** Compare matrices exactly: a rank-one reference contains a unit-trace PSD
 * input support iff they are identical. Nonzero overlap alone is insufficient. */
function equalMatrices(a:ReferenceMatrix,b:ReferenceMatrix):boolean {
 return a.every((row,i)=>row.every((z,j)=>eq(z.re,b[i][j].re)&&eq(z.im,b[i][j].im)));
}
function reference(rho:ReferenceMatrix,sigma:ReferenceMatrix) {
 const equal=equalMatrices(rho,sigma),det=referenceDeterminant(sigma);
 if(eq(det,zero)&&!equal)return {status:'infinite' as const,reason:'support-not-contained' as const};
 if(equal)return {status:'finite' as const,reason:'equal-states' as const,value:point(zero)};
 const entropy=referenceQubitEntropy(rho);
 if(eq(det,div(one,f(BigInt(4)))))return {status:'finite' as const,reason:'maximally-mixed-reference' as const,value:nonnegative(minus(point(one),entropy))};
 const gap=independentUnitRoot(sub(one,mul(f(BigInt(4)),det)));
 if(gap.lower.n<=BigInt(0))throw new RangeError('Independent eigenspace unresolved');
 const spectrum=[scale(plus(point(one),gap),div(one,two)),scale(minus(point(one),gap),div(one,two))];
 // Tr(rho P+) = (Tr(rho sigma)-lambda-)/gap, written as
 // 1/2+(Tr(rho sigma)-1/2)/gap to avoid dependent interval subtraction.
 const numerator=sub(referenceTraceProduct(rho,sigma),div(one,two));
 const quotient=numerator.n<BigInt(0)?{lower:div(numerator,gap.lower),upper:div(numerator,gap.upper)}:{lower:div(numerator,gap.upper),upper:div(numerator,gap.lower)};
 const clamp=(x:R):R=>({lower:le(x.lower,zero)?zero:x.lower,upper:le(one,x.upper)?one:x.upper});
 const weightPlus=clamp(plus(point(div(one,two)),quotient)),weights=[weightPlus,clamp(minus(point(one),weightPlus))];
 const cross=plus(product(weights[0],logarithm(spectrum[0])),product(weights[1],logarithm(spectrum[1])));
 const raw=minus(scale(entropy,neg(one)),cross);
 return {status:'finite' as const,reason:'full-rank-reference' as const,value:nonnegative(raw),raw,spectrum,weights};
}
export type RelativeEntropyResponseValidation={ok:true;response:Reply;checkedObservations:number}|{ok:false;path:string;reason:string};
export function validateQuantumRelativeEntropyResponse(request:unknown,response:unknown):RelativeEntropyResponseValidation {
 try {
  const input=snapshotQuantumInformationInput(request);ensure(input.kind==='relative-entropy','$request','Relative entropy request required');
  const owned=snapshotManyBodyWire(response),envelope=record(owned,['schemaVersion','kind','input','result'],'$');
  ensure(envelope.schemaVersion===1&&envelope.kind===input.kind,'$','Wrong response version/family');
  ensure(quantumInformationInputKey(envelope.input)===quantumInformationInputKey(input),'$.input','Original ordered request differs');
  const result=record(envelope.result,['channel','before','after','interpretation'],'$.result');
  ensure(result.channel==='complete-z-dephasing','$.result.channel','Wrong channel');
  ensure(result.interpretation===interpretation,'$.result.interpretation','Wrong scope interpretation');
  const rho=input.rho.map(rat),sigma=input.sigma.map(rat),checks=createInformationObservationChecks();
  const values:Record<string,R|null>={};
  for(const stage of ['before','after'] as const) {
   const state=(bloch:F[])=>referenceDensity(stage==='before'?bloch:[zero,zero,bloch[2]]);
   const expected=reference(state(rho),state(sigma)),path=`$.result.${stage}`;
   const keys=expected.reason==='full-rank-reference'?['status','value','reason','sigmaSpectrum','spectralWeights','rawValue','contract']:['status','value','reason','sigmaSpectrum','contract'];
   const d=record(result[stage],keys,path);
   ensure(d.status===expected.status,`${path}.status`,'Wrong support classification');
   ensure(d.reason===expected.reason,`${path}.reason`,'Wrong spectral branch');
   ensure(d.contract===contract,`${path}.contract`,'Wrong relative entropy contract');
   if(expected.status==='infinite') {
    ensure(d.value===null,`${path}.value`,'Infinite divergence requires null numerical value');
    ensure(same(d.sigmaSpectrum,[q(one),q(zero)]),`${path}.sigmaSpectrum`,'Wrong exact rank-one spectrum');values[stage]=null;continue;
   }
   if(expected.reason!=='equal-states')checks.check(d.value,expected.value,`${path}.value`);
   values[stage]=reported(d.value,`${path}.value`);
   ensure(le(zero,values[stage]!.lower),`${path}.value`,'Finite divergence must be nonnegative');
   if(expected.reason==='equal-states') {
    checks.exact(d.value,zero,`${path}.value`);ensure(d.sigmaSpectrum===null,`${path}.sigmaSpectrum`,'Equal states use null spectrum');
   } else if(expected.reason==='maximally-mixed-reference') {
    ensure(same(d.sigmaSpectrum,[q(div(one,two)),q(div(one,two))]),`${path}.sigmaSpectrum`,'Wrong exact mixed spectrum');
   } else {
    list(d.sigmaSpectrum,2,`${path}.sigmaSpectrum`).forEach((x,i)=>checks.check(x,expected.spectrum[i],`${path}.sigmaSpectrum[${i}]`));
    list(d.spectralWeights,2,`${path}.spectralWeights`).forEach((x,i)=>checks.check(x,expected.weights[i],`${path}.spectralWeights[${i}]`));
    const spectrum=list(d.sigmaSpectrum,2,`${path}.sigmaSpectrum`).map((x,i)=>reported(x,`${path}.sigmaSpectrum[${i}]`));
    const weights=list(d.spectralWeights,2,`${path}.spectralWeights`).map((x,i)=>reported(x,`${path}.spectralWeights[${i}]`));
    for(const x of spectrum)ensure(x.lower.n>BigInt(0)&&le(x.upper,one),`${path}.sigmaSpectrum`,'Full-rank spectrum must be positive and at most one');
    for(const x of weights)ensure(le(zero,x.lower)&&le(x.upper,one),`${path}.spectralWeights`,'Spectral weights must be probabilities');
    for(const [name,pair] of [['sigmaSpectrum',spectrum],['spectralWeights',weights]] as const) {
     const complement=minus(point(one),pair[0]);
     ensure(eq(pair[1].lower,complement.lower)&&eq(pair[1].upper,complement.upper),`${path}.${name}[1]`,'Complement intervals must match');
    }
    checks.check(d.rawValue,expected.raw,`${path}.rawValue`);
    const clipped=nonnegative(reported(d.rawValue,`${path}.rawValue`)),actual=values[stage]!;
    ensure(eq(actual.lower,clipped.lower)&&eq(actual.upper,clipped.upper),`${path}.value`,'Value must clip only the raw lower endpoint at zero');
   }
  }
  // Interval consistency only; the independent spectral computations above
  // validate the numbers. This does not use DPI to replace those computations.
  if(values.before!==null) {
   ensure(values.after!==null,'$.result.after','A channel cannot turn finite divergence infinite');
   ensure(le(values.after!.lower,values.before!.upper),'$.result.after.value','Dephasing divergences contradict data processing');
  }
  return {ok:true,response:owned as Reply,checkedObservations:checks.count};
 } catch(error) {return {ok:false,path:error instanceof InformationResponseInvalid?error.path:'$',reason:error instanceof Error?error.message:'Invalid relative entropy response'};}
}
