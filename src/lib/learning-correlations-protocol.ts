/** Transport schema and preparation identity; imports below are erased types.
 * This validates structure, not numerical truth from an arbitrary sender.
 */
import type {binaryConnectedCorrelationsModel,BinaryCorrelationsParameters} from './learning-binary-correlations';
import type {periodicIsingRingModel,PeriodicIsingRingParameters} from './learning-periodic-ising-ring';
import type {ar1FiniteAverageModel,AR1AverageParameters} from './learning-ar1-average';
export const correlationKinds=['binary-connected-correlations-field-hessian','periodic-ising-ring-correlation-length','finite-correlated-average-ar1-uncertainty'] as const;
export type CorrelationKind=typeof correlationKinds[number];
interface PrecisionOptions {minimumPrecision?:number;maximumPrecision?:number}
export type CorrelationInput=({kind:typeof correlationKinds[0];parameters:BinaryCorrelationsParameters}|{kind:typeof correlationKinds[1];parameters:PeriodicIsingRingParameters}|{kind:typeof correlationKinds[2];parameters:AR1AverageParameters})&{options?:PrecisionOptions};
type Results={
 'binary-connected-correlations-field-hessian':ReturnType<typeof binaryConnectedCorrelationsModel>;
 'periodic-ising-ring-correlation-length':ReturnType<typeof periodicIsingRingModel>;
 'finite-correlated-average-ar1-uncertainty':ReturnType<typeof ar1FiniteAverageModel>;
};
export type CorrelationSuccess={[K in CorrelationKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<CorrelationInput,{kind:K}>;inputKey:string;result:Results[K]}}[CorrelationKind];
export type CorrelationResponse=CorrelationSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const correlationRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const precisions=[128,256,512,1024,2048,4096,8192];
const parameterKeys=(kind:CorrelationKind)=>kind===correlationKinds[0]?['couplingK','fieldOne','fieldTwo','probeOne','probeTwo']:kind===correlationKinds[1]?['spinCount','couplingK']:['sampleCount','correlationRho'];
const onlyKeys=(v:Record<string,unknown>,keys:string[])=>Object.keys(v).every(k=>keys.includes(k));
export function isCorrelationInput(v:unknown):v is CorrelationInput {
 if(!correlationRecord(v)||!onlyKeys(v,['kind','parameters','options'])||!correlationKinds.includes(v.kind as CorrelationKind)||!correlationRecord(v.parameters))return false;
 const p=v.parameters,keys=parameterKeys(v.kind as CorrelationKind);
 if(!onlyKeys(p,keys)||!keys.every(k=>finite(p[k])))return false;
 if(v.options!==undefined){
  if(!correlationRecord(v.options)||!onlyKeys(v.options,['minimumPrecision','maximumPrecision']))return false;
  if(v.options.minimumPrecision!==undefined&&!precisions.includes(v.options.minimumPrecision as number)||v.options.maximumPrecision!==undefined&&!precisions.includes(v.options.maximumPrecision as number))return false;
  if(((v.options.minimumPrecision??128)as number)>((v.options.maximumPrecision??8192)as number))return false;
 }
 if(v.kind===correlationKinds[0])return ['couplingK','fieldOne','fieldTwo'].every(k=>Math.abs(p[k] as number)<=8)&&['probeOne','probeTwo'].every(k=>Math.abs(p[k] as number)<=4);
 if(v.kind===correlationKinds[1])return Number.isInteger(p.spinCount)&&(p.spinCount as number)>=3&&(p.spinCount as number)<=128&&Math.abs(p.couplingK as number)<=8;
 return Number.isInteger(p.sampleCount)&&(p.sampleCount as number)>=1&&(p.sampleCount as number)<=512&&Math.abs(p.correlationRho as number)<=1;
}
export function snapshotCorrelationInput(v:CorrelationInput):CorrelationInput {
 if(!isCorrelationInput(v))throw new RangeError('Invalid correlations preparation');
 const options=Object.freeze({minimumPrecision:v.options?.minimumPrecision??128,maximumPrecision:v.options?.maximumPrecision??8192}),number=(n:number)=>n===0?0:n;
 if(v.kind===correlationKinds[0])return Object.freeze({kind:v.kind,parameters:Object.freeze({couplingK:number(v.parameters.couplingK),fieldOne:number(v.parameters.fieldOne),fieldTwo:number(v.parameters.fieldTwo),probeOne:number(v.parameters.probeOne),probeTwo:number(v.parameters.probeTwo)}),options});
 if(v.kind===correlationKinds[1])return Object.freeze({kind:v.kind,parameters:Object.freeze({spinCount:v.parameters.spinCount,couplingK:number(v.parameters.couplingK)}),options});
 return Object.freeze({kind:v.kind,parameters:Object.freeze({sampleCount:v.parameters.sampleCount,correlationRho:number(v.parameters.correlationRho)}),options});
}
export const correlationInputKey=(v:CorrelationInput)=>JSON.stringify(snapshotCorrelationInput(v));
const same=(a:unknown,b:unknown)=>{try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
const coefficient=(v:unknown,positive=false)=>correlationRecord(v)&&typeof v.coefficient==='string'&&v.coefficient.length<=12000000&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v.coefficient)&&v.coefficient!=='-0'&&finite(v.exponent)&&Number.isInteger(v.exponent)&&Math.abs(v.exponent)<=1100000;
const rational=(v:unknown)=>correlationRecord(v)&&coefficient(v.numerator)&&coefficient(v.denominator,true);
export function isCorrelationObservation(v:unknown):boolean {
 if(!correlationRecord(v)||!(v.value===null||finite(v.value))||!(v.sign===null||[-1,0,1].includes(v.sign as number))||!(v.logMagnitude===null||finite(v.logMagnitude))||!['exact-rational','structural-zero','interval-resolved','unresolved-at-precision-ceiling','mathematical-infinity'].includes(v.resolution as string)||!['finite','outside-number-range','not-applicable','unresolved'].includes(v.logResolution as string))return false;
 if(v.logResolution==='finite'?!finite(v.logMagnitude):v.logMagnitude!==null)return false;
 if(v.resolution==='structural-zero'){if(v.value!==0||v.sign!==0||v.logMagnitude!==null||v.logResolution!=='not-applicable')return false;}
 else if(v.sign===0||v.value===0)return false;
 if(v.resolution==='mathematical-infinity')return v.value===null&&[-1,1].includes(v.sign as number)&&v.logResolution==='not-applicable'&&v.bounds===null;
 if(v.resolution==='unresolved-at-precision-ceiling'&&v.value!==null)return false;
 if(['interval-resolved','exact-rational'].includes(v.resolution as string)&&![-1,1].includes(v.sign as number))return false;
 if(finite(v.value)&&v.value!==0&&Math.sign(v.value)!==v.sign)return false;
 if(!correlationRecord(v.bounds)||!rational(v.bounds.lower)||!rational(v.bounds.upper))return v.bounds===null&&v.resolution==='unresolved-at-precision-ceiling';
 if(v.resolution==='structural-zero'){const b=v.bounds as Record<string,Record<string,{coefficient:string}>>;return b.lower.numerator.coefficient==='0'&&b.upper.numerator.coefficient==='0';}
 return true;
}
function safeTree(v:unknown,observations:Record<string,unknown>[],seen=new Set<object>(),depth=0,budget={nodes:0}):boolean {
 if(++budget.nodes>100000||depth>32)return false;
 if(v===null||typeof v==='boolean')return true;
 if(typeof v==='string')return v.length<=12000000;
 if(typeof v==='number')return Number.isFinite(v);
 if(typeof v!=='object'||seen.has(v))return false;
 seen.add(v);let valid=true;
 if(Array.isArray(v))valid=v.length<=4096&&Array.from(v).every(x=>safeTree(x,observations,seen,depth+1,budget));
 else if(correlationRecord(v)){
  if('resolution'in v){valid=isCorrelationObservation(v);observations.push(v);}
  if('numerator'in v||'denominator'in v)valid=valid&&rational(v);
  valid=valid&&Object.values(v).every(x=>safeTree(x,observations,seen,depth+1,budget));
 }else valid=false;
 seen.delete(v);return valid;
}
const fields=(v:Record<string,unknown>,keys:string[])=>keys.every(k=>isCorrelationObservation(v[k]));
export function isCorrelationSuccessForInput(v:unknown,input:CorrelationInput,requestId:number):v is CorrelationSuccess {
 if(!correlationRecord(v)||v.version!==1||v.requestId!==requestId||v.ok!==true||v.kind!==input.kind||!isCorrelationInput(v.input)||correlationInputKey(v.input)!==correlationInputKey(input)||v.inputKey!==correlationInputKey(input)||!correlationRecord(v.result))return false;
 const result=v.result,observations:Record<string,unknown>[]=[],snapshot=snapshotCorrelationInput(input),options=snapshot.options!;
 if(!safeTree(v,observations)||!same(result.parameters,snapshot.parameters)||typeof result.fullyResolved!=='boolean'||!precisions.includes(result.precisionBits as number)||!Array.isArray(result.attempts)||typeof result.approximationContract!=='string')return false;
 const P=result.precisionBits as number;
 if(P<options.minimumPrecision!||P>options.maximumPrecision!||!result.fullyResolved&&P!==options.maximumPrecision!)return false;
 if(!same(result.attempts,precisions.filter(p=>p>=options.minimumPrecision!&&p<=P))||result.fullyResolved!==observations.every(q=>q.resolution!=='unresolved-at-precision-ceiling'))return false;
 if(input.kind===correlationKinds[0]){
  return result.scope==='two-spin-connected-covariance-and-source-hessian'
   &&fields(result,['pairMean','covarianceDeterminant','logPartition','probeMean','probeVariance','commonFieldResponsePerSpin'])
   &&Array.isArray(result.means)&&result.means.length===2&&result.means.every(isCorrelationObservation)
   &&Array.isArray(result.covariance)&&result.covariance.length===2&&result.covariance.every(row=>Array.isArray(row)&&row.length===2&&row.every(isCorrelationObservation))
   &&Array.isArray(result.states)&&result.states.length===4&&result.states.every((row,i)=>correlationRecord(row)&&same(row.spins,[i<2?-1:1,i%2?1:-1])&&fields(row,['probability','logProbability']));
 }
 if(input.kind===correlationKinds[1]){
  const N=input.parameters.spinCount,K=input.parameters.couplingK;
  return result.scope==='zero-field-periodic-nearest-neighbor-ising-ring'&&result.bondCount===N&&result.microstateCount===(BigInt(1)<<BigInt(N)).toString()&&result.oddAntiferromagnetic===(K<0&&N%2===1)
   &&result.correlationLengthStatus===(K===0?'independent-zero-range-limit':'finite-positive-infinite-chain-envelope')
   &&fields(result,['meanSpin','commonFieldResponsePerSpin','magnetizationMeanVariance','totalSpinVariance','logPartition','infiniteChainCorrelationLength','infiniteChainCommonFieldResponse','infiniteChainOneStepCorrelation','logAbsoluteOneStepCorrelation'])
   &&Array.isArray(result.correlations)&&result.correlations.length===N&&result.correlations.every((row,i)=>correlationRecord(row)&&row.offset===i&&row.distance===Math.min(i,N-i)&&fields(row,['correlation','logAbsoluteCorrelation']));
 }
 const n=input.parameters.sampleCount,rho=input.parameters.correlationRho,interior=Math.abs(rho)<1,deterministic=rho===-1&&n%2===0;
 if(result.scope!=='stationary-unit-variance-gaussian-ar1-finite-average'||!fields(result,['meanExpectation','innovationVariance','independentMeanVariance','meanVariance','totalSumVariance','standardError','varianceInflation','varianceEquivalentSampleCount']))return false;
 if(result.asymptoticStatus!==(interior?'mixing-interior':'not-applicable-nonmixing-endpoint')||result.processMode!==(rho===1?'constant-degenerate':rho===-1?'alternating-degenerate':'stationary-gaussian-ar1')||result.deterministicAverage!==deterministic)return false;
 if(!['asymptoticIntegratedCorrelationTime','asymptoticMeanVariance','finiteToAsymptoticVarianceRatio'].every(k=>interior?isCorrelationObservation(result[k]):result[k]===null))return false;
 if(deterministic&&(result.meanVariance as Record<string,unknown>).resolution!=='structural-zero'||deterministic&&(result.varianceEquivalentSampleCount as Record<string,unknown>).resolution!=='mathematical-infinity')return false;
 return Array.isArray(result.autocovariances)&&result.autocovariances.length===n&&result.autocovariances.every((row,i)=>correlationRecord(row)&&row.lag===i&&isCorrelationObservation(row.covariance))
  &&Array.isArray(result.varianceBySampleCount)&&result.varianceBySampleCount.length===n&&result.varianceBySampleCount.every((row,i)=>correlationRecord(row)&&row.sampleCount===i+1&&isCorrelationObservation(row.meanVariance));
}
