/** Lightweight transport checks; numerical imports below are erased types.
 * These check schema/input identity, not the mathematics of an arbitrary sender.
 */
import type {finiteCurieWeissModel,CurieWeissParameters} from './learning-curie-weiss';
import type {curieWeissVariationalModel,CurieWeissVariationalParameters} from './learning-curie-weiss-variational';
import type {landauSixthOrderModel,LandauSixthOrderParameters} from './learning-landau-sixth-order';
export const phaseKinds=['finite-curie-weiss-magnetization-symmetry','curie-weiss-variational-stability-limits','landau-sixth-order-coexistence-stability'] as const;
export type PhaseKind=typeof phaseKinds[number];
interface PrecisionOptions {minimumPrecision?:number;maximumPrecision?:number}
export type PhaseInput=({kind:typeof phaseKinds[0];parameters:CurieWeissParameters}|{kind:typeof phaseKinds[1];parameters:CurieWeissVariationalParameters}|{kind:typeof phaseKinds[2];parameters:LandauSixthOrderParameters})&{options?:PrecisionOptions};
type Results={
 'finite-curie-weiss-magnetization-symmetry':ReturnType<typeof finiteCurieWeissModel>;
 'curie-weiss-variational-stability-limits':ReturnType<typeof curieWeissVariationalModel>;
 'landau-sixth-order-coexistence-stability':ReturnType<typeof landauSixthOrderModel>;
};
export type PhaseSuccess={[K in PhaseKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<PhaseInput,{kind:K}>;inputKey:string;result:Results[K]}}[PhaseKind];
export type PhaseResponse=PhaseSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const phaseRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const precisions=[128,256,512,1024,2048,4096,8192];
const parameters=(kind:PhaseKind)=>kind===phaseKinds[2]?['quadraticA','quarticB','sexticC','fieldH']:kind===phaseKinds[0]?['spinCount','couplingK','fieldB']:['couplingK','fieldB'];
export function isPhaseInput(v:unknown):v is PhaseInput {
 if(!phaseRecord(v)||!phaseKinds.includes(v.kind as PhaseKind)||!phaseRecord(v.parameters))return false;
 const p=v.parameters;if(!parameters(v.kind as PhaseKind).every(k=>finite(p[k])))return false;
 if(v.options!==undefined){if(!phaseRecord(v.options))return false;if(v.options.minimumPrecision!==undefined&&!precisions.includes(v.options.minimumPrecision as number)||v.options.maximumPrecision!==undefined&&!precisions.includes(v.options.maximumPrecision as number))return false;const lo=v.options.minimumPrecision??128,hi=v.options.maximumPrecision??8192;if((lo as number)>(hi as number))return false;}
 if(v.kind===phaseKinds[2])return Math.abs(p.quadraticA as number)<=4&&Math.abs(p.quarticB as number)<=4&&(p.sexticC as number)>=.25&&(p.sexticC as number)<=4&&Math.abs(p.fieldH as number)<=4;
 return (p.couplingK as number)>=0&&(p.couplingK as number)<=8&&Math.abs(p.fieldB as number)<=4&&(v.kind!==phaseKinds[0]||Number.isInteger(p.spinCount)&&(p.spinCount as number)>=1&&(p.spinCount as number)<=128);
}
export function snapshotPhaseInput(v:PhaseInput):PhaseInput {
 if(!isPhaseInput(v))throw new RangeError('Invalid phase-transition preparation');
 const options=Object.freeze({minimumPrecision:v.options?.minimumPrecision??128,maximumPrecision:v.options?.maximumPrecision??8192}),number=(n:number)=>n===0?0:n;
 if(v.kind===phaseKinds[0])return Object.freeze({kind:v.kind,parameters:Object.freeze({spinCount:v.parameters.spinCount,couplingK:number(v.parameters.couplingK),fieldB:number(v.parameters.fieldB)}),options});
 if(v.kind===phaseKinds[1])return Object.freeze({kind:v.kind,parameters:Object.freeze({couplingK:number(v.parameters.couplingK),fieldB:number(v.parameters.fieldB)}),options});
 return Object.freeze({kind:v.kind,parameters:Object.freeze({quadraticA:number(v.parameters.quadraticA),quarticB:number(v.parameters.quarticB),sexticC:number(v.parameters.sexticC),fieldH:number(v.parameters.fieldH)}),options});
}
export const phaseInputKey=(v:PhaseInput)=>JSON.stringify(snapshotPhaseInput(v));
const same=(a:unknown,b:unknown)=>{try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
const coefficient=(v:unknown,positive=false)=>phaseRecord(v)&&typeof v.coefficient==='string'&&v.coefficient.length<=12000000&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v.coefficient)&&v.coefficient!=='-0'&&finite(v.exponent)&&Number.isInteger(v.exponent)&&Math.abs(v.exponent)<=1100000;
const rational=(v:unknown)=>phaseRecord(v)&&coefficient(v.numerator)&&coefficient(v.denominator,true);
export function isPhaseObservation(v:unknown):boolean {
 if(!phaseRecord(v)||!(v.value===null||finite(v.value))||!(v.sign===null||[-1,0,1].includes(v.sign as number))||!(v.logMagnitude===null||finite(v.logMagnitude))||!['exact-rational','structural-zero','interval-resolved','unresolved-at-precision-ceiling','mathematical-infinity'].includes(v.resolution as string)||!['finite','outside-number-range','not-applicable','unresolved'].includes(v.logResolution as string))return false;
 if(v.logResolution==='finite'?!finite(v.logMagnitude):v.logMagnitude!==null)return false;
 if(v.resolution==='structural-zero'){if(v.value!==0||v.sign!==0||v.logMagnitude!==null||v.logResolution!=='not-applicable')return false;}
 else if(v.sign===0||v.value===0)return false;
 if(v.resolution==='mathematical-infinity')return v.value===null&&[-1,1].includes(v.sign as number)&&v.logResolution==='not-applicable'&&v.bounds===null;
 if(v.resolution==='unresolved-at-precision-ceiling'&&v.value!==null)return false;
 if(['interval-resolved','exact-rational'].includes(v.resolution as string)&&![-1,1].includes(v.sign as number))return false;
 if(finite(v.value)&&v.value!==0&&Math.sign(v.value)!==v.sign)return false;
 if(!phaseRecord(v.bounds)||!rational(v.bounds.lower)||!rational(v.bounds.upper))return v.bounds===null&&v.resolution==='unresolved-at-precision-ceiling';
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
 else if(phaseRecord(v)){
  if('resolution'in v){valid=isPhaseObservation(v);observations.push(v);}
  if('numerator'in v||'denominator'in v)valid=valid&&rational(v);
  valid=valid&&Object.values(v).every(x=>safeTree(x,observations,seen,depth+1,budget));
 }else valid=false;
 seen.delete(v);return valid;
}
const fields=(v:Record<string,unknown>,keys:string[])=>keys.every(k=>isPhaseObservation(v[k]));
export function isPhaseSuccessForInput(v:unknown,input:PhaseInput,requestId:number):v is PhaseSuccess {
 if(!phaseRecord(v)||v.version!==1||v.requestId!==requestId||v.ok!==true||v.kind!==input.kind||!isPhaseInput(v.input)||phaseInputKey(v.input)!==phaseInputKey(input)||v.inputKey!==phaseInputKey(input)||!phaseRecord(v.result))return false;
 const result=v.result,observations:Record<string,unknown>[]=[],options=snapshotPhaseInput(input).options!;
 if(!safeTree(v,observations)||!same(result.parameters,snapshotPhaseInput(input).parameters)||typeof result.fullyResolved!=='boolean'||!precisions.includes(result.precisionBits as number)||!Array.isArray(result.attempts)||!result.attempts.length||typeof result.approximationContract!=='string')return false;
 const P=result.precisionBits as number,lo=options.minimumPrecision!,hi=options.maximumPrecision!;
 if(P<lo||P>hi||!result.fullyResolved&&P!==hi)return false;
 const expected=precisions.filter(p=>p>=lo&&p<=P);
 if(!same(result.attempts.map(x=>input.kind===phaseKinds[0]?x:phaseRecord(x)?x.precisionBits:null),expected))return false;
 if(result.fullyResolved&&observations.some(q=>q.resolution==='unresolved-at-precision-ceiling'))return false;
 if(input.kind===phaseKinds[0]){
  const n=input.parameters.spinCount;
  return result.scope==='finite-ferromagnetic-curie-weiss'&&result.macroStateCount===n+1&&result.microstateCount===(BigInt(1)<<BigInt(n)).toString()&&result.spinFlipSymmetric===(input.parameters.fieldB===0)
   &&fields(result,['meanMagnetization','meanAbsoluteMagnetization','meanSquareMagnetization','magnetizationVariance','fieldSusceptibility','macroEntropyNats','conditionalMicrostateEntropyNats','microstateEntropyNats','logPartition','pressurePerSpin','freeEnergyInThermalUnits','freeEnergyPerSpinInThermalUnits','meanEnergyInThermalUnits','logPartitionWithoutSelfInteraction','meanEnergyWithoutSelfInteractionInThermalUnits','logZFieldDerivative','logZCouplingDerivative'])
   &&Array.isArray(result.states)&&result.states.length===n+1&&result.states.every((x,i)=>phaseRecord(x)&&x.plusSpinCount===i&&x.totalMagnetization===2*i-n&&typeof x.multiplicity==='string'&&/^[1-9]\d*$/.test(x.multiplicity)&&fields(x,['magnetizationPerSpin','probability','logProbability']))&&phaseRecord(result.modes)&&typeof result.modes.certified==='boolean'&&Array.isArray(result.modes.candidateTotalMagnetizations)&&result.fullyResolved===(result.modes.certified&&observations.every(q=>q.resolution!=='unresolved-at-precision-ceiling'));
 }
 if(!Array.isArray(result.roots)||!Array.isArray(result.globalMinimumIndices)||result.globalMinimumIndices.some((i:unknown)=>!Number.isInteger(i)||(i as number)<0||(i as number)>=(result.roots as unknown[]).length)||new Set(result.globalMinimumIndices).size!==result.globalMinimumIndices.length)return false;
 const roots=result.roots as Record<string,unknown>[];
 if(!roots.every(phaseRecord))return false;
 if(typeof result.unresolvedReason!=='string'&&result.unresolvedReason!==null||result.fullyResolved!==(result.unresolvedReason===null))return false;
 const indices=(v:unknown)=>Array.isArray(v)&&v.every(i=>Number.isInteger(i)&&i>=0&&i<roots.length)&&new Set(v).size===v.length;
 const enclosure=(v:unknown)=>phaseRecord(v)&&rational(v.lower)&&rational(v.upper);
 const allReady=observations.every(q=>q.resolution!=='unresolved-at-precision-ceiling');
 if(input.kind===phaseKinds[1]){
  if(result.scope!=='curie-weiss-constrained-variational-limit'||roots.length>3||!Array.isArray(result.rootCountCandidates)||!result.rootCountCandidates.length||!result.rootCountCandidates.every(i=>[1,2,3].includes(i))||!Array.isArray(result.barriers))return false;
  if(result.spinodal!==null&&(!phaseRecord(result.spinodal)||!fields(result.spinodal,['turningMagnetization','fieldMagnitude'])))return false;
  if(result.minimumFreeEnergyGap!==null&&!isPhaseObservation(result.minimumFreeEnergyGap))return false;
  if(!result.fullyResolved)return roots.length===0&&result.thermodynamicLimitPressure===null&&result.globalMinimumIndices.length===0&&result.minimumFreeEnergyGap===null&&result.barriers.length===0;
  if(!allReady||result.rootCount!==roots.length||!same(result.rootCountCandidates,[result.rootCount])||!isPhaseObservation(result.thermodynamicLimitPressure)||!result.globalMinimumIndices.length)return false;
  if(!result.barriers.every(x=>phaseRecord(x)&&Number.isInteger(x.minimumIndex)&&(x.minimumIndex as number)>=0&&(x.minimumIndex as number)<roots.length&&isPhaseObservation(x.barrier)))return false;
  return roots.every((x,i)=>phaseRecord(x)&&fields(x,['magnetization','curvature','branchFieldDerivative','constrainedFreeEnergy','freeEnergyDifferenceToZero','entropyNats'])&&enclosure(x.residualEnclosure)&&x.globalMinimum===(result.globalMinimumIndices as number[]).includes(i)&&['global-minimum','metastable-minimum','unstable-maximum','critical-minimum'].includes(x.role as string)&&x.globalMinimum===(x.role==='global-minimum'||x.role==='critical-minimum')&&[1,3].includes(x.stationaryMultiplicity as number));
 }
 if(result.scope!=='phenomenological-sixth-order-landau-minimization'||result.rootsComplete!==true||result.rootCount!==roots.length||roots.length<1||roots.length>5||typeof result.globalSelectionResolved!=='boolean'||typeof result.exactCoexistence!=='boolean'||!indices(result.globalMinimumCandidates)||!Array.isArray(result.barriers))return false;
 if(result.fullyResolved!==(allReady&&result.globalSelectionResolved&&roots.every(x=>x.localKind!=='unresolved'&&((x.stationaryMultiplicity as number)%2===0||x.branchFieldDerivative!==null))))return false;
 if(result.globalSelectionResolved?!(result.globalMinimumIndices.length&&isPhaseObservation(result.minimumFreeEnergy)&&same(result.globalMinimumCandidates,result.globalMinimumIndices)):result.globalMinimumIndices.length!==0||result.minimumFreeEnergy!==null)return false;
 if(!result.barriers.every(x=>phaseRecord(x)&&Number.isInteger(x.minimumIndex)&&Number.isInteger(x.maximumIndex)&&(x.minimumIndex as number)>=0&&(x.minimumIndex as number)<roots.length&&(x.maximumIndex as number)>=0&&(x.maximumIndex as number)<roots.length&&roots[x.minimumIndex as number].localKind==='minimum'&&roots[x.maximumIndex as number].localKind==='maximum'&&isPhaseObservation(x.barrier)))return false;
 if(result.realRootMultiplicity!==roots.reduce((n,x)=>n+(x.stationaryMultiplicity as number),0)||(result.realRootMultiplicity as number)>5)return false;
 return roots.every((x,i)=>{
  if(!phaseRecord(x)||!fields(x,['magnetization','curvature','constrainedFreeEnergy','leadingStationaryDerivative'])||!enclosure(x.residualEnclosure)||!Number.isInteger(x.stationaryMultiplicity)||(x.stationaryMultiplicity as number)<1||(x.stationaryMultiplicity as number)>5||!['minimum','maximum','stationary-inflection','unresolved'].includes(x.localKind as string))return false;
  const global=result.globalSelectionResolved?(result.globalMinimumIndices as number[]).includes(i):null;
  const role=x.localKind==='minimum'?(global===null?'minimum-global-unresolved':global?'global-minimum':'metastable-minimum'):x.localKind==='maximum'?'unstable-maximum':x.localKind;
  if(x.globalMinimum!==global||x.role!==role||(result.globalSelectionResolved?!isPhaseObservation(x.freeEnergyExcess):x.freeEnergyExcess!==null))return false;
  if((x.stationaryMultiplicity as number)%2===0)return x.localKind==='stationary-inflection'&&x.branchFieldDerivative===null&&x.derivativeStatus==='fold-no-single-valued-derivative';
  return x.derivativeStatus===((x.stationaryMultiplicity as number)>1?'singular-limiting-derivative':'regular-branch')&&(x.branchFieldDerivative===null||isPhaseObservation(x.branchFieldDerivative));
 });
}
