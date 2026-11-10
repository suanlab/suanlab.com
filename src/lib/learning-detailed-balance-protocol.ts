/** Transport structure and identity checks; erased model imports never execute
 * on the main thread. This is not numerical truth certification of a sender.
 */
import type {stationaryThreeStateCycleModel,StationaryCycleParameters} from './learning-stationary-cycle';
import type {metropolisHastingsProposalModel,HastingsProposalParameters} from './learning-hastings-proposal';
import type {reversibleTwoStateMixingModel,TwoStateMixingParameters} from './learning-two-state-mixing';
export const detailedBalanceKinds=['stationary-flux-three-state-cycle','metropolis-hastings-proposal-balance','reversible-two-state-periodicity-mixing'] as const;
export type DetailedBalanceKind=typeof detailedBalanceKinds[number];
export type DetailedBalanceInput={kind:typeof detailedBalanceKinds[0];parameters:StationaryCycleParameters}|{kind:typeof detailedBalanceKinds[1];parameters:HastingsProposalParameters}|{kind:typeof detailedBalanceKinds[2];parameters:TwoStateMixingParameters};
type Results={
 'stationary-flux-three-state-cycle':ReturnType<typeof stationaryThreeStateCycleModel>;
 'metropolis-hastings-proposal-balance':ReturnType<typeof metropolisHastingsProposalModel>;
 'reversible-two-state-periodicity-mixing':ReturnType<typeof reversibleTwoStateMixingModel>;
};
export type DetailedBalanceSuccess={[K in DetailedBalanceKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<DetailedBalanceInput,{kind:K}>;inputKey:string;result:Results[K]}}[DetailedBalanceKind];
export type DetailedBalanceResponse=DetailedBalanceSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const detailedBalanceRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
type Check=(v:unknown)=>boolean;
function shape(v:unknown,spec:Record<string,Check>):v is Record<string,unknown>{return detailedBalanceRecord(v)&&Object.keys(v).length===Object.keys(spec).length&&Object.keys(spec).every(k=>Object.prototype.hasOwnProperty.call(v,k)&&spec[k](v[k]));}
const integer=(v:unknown,min=0,max=1000000):v is number=>typeof v==='number'&&Number.isInteger(v)&&v>=min&&v<=max;
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const bool:Check=v=>typeof v==='boolean';
const literal=(value:unknown):Check=>v=>v===value;
const choice=(...values:unknown[]):Check=>v=>values.includes(v);
const nullable=(check:Check):Check=>v=>v===null||check(v);
const vector=(size:number,check:Check):Check=>v=>Array.isArray(v)&&v.length===size&&Array.from(v).every(check);
const triplet=(v:unknown,positive=false):v is number[]=>vector(3,n=>integer(n,positive?1:0))(v);
export function isDetailedBalanceInput(v:unknown):v is DetailedBalanceInput {
 if(!shape(v,{kind:choice(...detailedBalanceKinds),parameters:detailedBalanceRecord}))return false;
 const p=v.parameters as Record<string,unknown>;
 if(v.kind===detailedBalanceKinds[0])return shape(p,{forwardWeight:integer,backwardWeight:integer,stayWeight:integer,initialWeights:triplet,stepCount:n=>integer(n,0,512)})
  &&(p.forwardWeight as number)+(p.backwardWeight as number)+(p.stayWeight as number)>0&&(p.initialWeights as number[]).some(n=>n>0);
 if(v.kind===detailedBalanceKinds[1])return shape(p,{targetWeights:n=>triplet(n,true),forwardWeight:integer,backwardWeight:integer,stayWeight:integer})
  &&(p.forwardWeight as number)+(p.backwardWeight as number)+(p.stayWeight as number)>0;
 return shape(p,{forwardNumerator:integer,backwardNumerator:integer,transitionDenominator:n=>integer(n,1),initialOneNumerator:integer,initialDenominator:n=>integer(n,1),stepCount:n=>integer(n,0,512)})
  &&(p.forwardNumerator as number)<=(p.transitionDenominator as number)&&(p.backwardNumerator as number)<=(p.transitionDenominator as number)&&(p.initialOneNumerator as number)<=(p.initialDenominator as number);
}
export function snapshotDetailedBalanceInput(v:DetailedBalanceInput):DetailedBalanceInput {
 if(!isDetailedBalanceInput(v))throw new RangeError('Invalid detailed-balance preparation');
 const n=(x:number)=>x+0;
 if(v.kind===detailedBalanceKinds[0])return Object.freeze({kind:v.kind,parameters:Object.freeze({forwardWeight:n(v.parameters.forwardWeight),backwardWeight:n(v.parameters.backwardWeight),stayWeight:n(v.parameters.stayWeight),initialWeights:Object.freeze(v.parameters.initialWeights.map(n)),stepCount:n(v.parameters.stepCount)})});
 if(v.kind===detailedBalanceKinds[1])return Object.freeze({kind:v.kind,parameters:Object.freeze({targetWeights:Object.freeze(v.parameters.targetWeights.map(n)),forwardWeight:n(v.parameters.forwardWeight),backwardWeight:n(v.parameters.backwardWeight),stayWeight:n(v.parameters.stayWeight)})});
 return Object.freeze({kind:v.kind,parameters:Object.freeze({forwardNumerator:n(v.parameters.forwardNumerator),backwardNumerator:n(v.parameters.backwardNumerator),transitionDenominator:v.parameters.transitionDenominator,initialOneNumerator:n(v.parameters.initialOneNumerator),initialDenominator:v.parameters.initialDenominator,stepCount:n(v.parameters.stepCount)})});
}
export const detailedBalanceInputKey=(v:DetailedBalanceInput)=>JSON.stringify(snapshotDetailedBalanceInput(v));
export function isDetailedBalanceRequestEnvelope(v:unknown):v is {version:1;requestId:number;input:unknown} {
 return shape(v,{version:literal(1),requestId:n=>Number.isSafeInteger(n)&&(n as number)>0,input:()=>true});
}
const same=(a:unknown,b:unknown)=>{if(a===b)return true;try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
/** Integer bounds need at most ~3330 digits in these input ranges. The limit
 * below covers every family without parsing large integers on the main thread.
 */
export function isDetailedBalanceObservation(v:unknown):boolean {
 if(!shape(v,{value:n=>n===null||finite(n),sign:choice(-1,0,1),logMagnitude:n=>n===null||finite(n),logResolution:choice('finite','not-applicable'),resolution:choice('exact-rational','structural-zero'),bounds:detailedBalanceRecord}))return false;
 const coefficient=(x:unknown,positive=false)=>typeof x==='string'&&x.length<=4096&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(x)&&x!=='-0';
 const term=(x:unknown,positive=false)=>shape(x,{coefficient:n=>coefficient(n,positive),exponent:literal(0)});
 const rational=(x:unknown)=>shape(x,{numerator:n=>term(n),denominator:n=>term(n,true)});
 const bounds=v.bounds as Record<string,unknown>;
 if(!shape(bounds,{lower:rational,upper:rational}))return false;
 const lo=bounds.lower as {numerator:{coefficient:string};denominator:{coefficient:string}},hi=bounds.upper as typeof lo;
 if(lo.numerator.coefficient!==hi.numerator.coefficient||lo.denominator.coefficient!==hi.denominator.coefficient)return false;
 const numerator=lo.numerator.coefficient;
 if(v.resolution==='structural-zero')return v.value===0&&v.sign===0&&v.logMagnitude===null&&v.logResolution==='not-applicable'&&numerator==='0'&&lo.denominator.coefficient==='1';
 return numerator!=='0'&&v.sign===(numerator[0]==='-'?-1:1)&&v.logResolution==='finite'&&finite(v.logMagnitude)
  &&(v.value===null||finite(v.value)&&v.value!==0&&Math.sign(v.value)===v.sign);
}
export function isDetailedBalanceSuccessForInput(v:unknown,input:DetailedBalanceInput,requestId:number):v is DetailedBalanceSuccess {
 if(!Number.isSafeInteger(requestId)||requestId<=0||!isDetailedBalanceInput(input)||!shape(v,{version:literal(1),requestId:literal(requestId),ok:literal(true),kind:literal(input.kind),input:isDetailedBalanceInput,inputKey:literal(detailedBalanceInputKey(input)),result:detailedBalanceRecord})||detailedBalanceInputKey(v.input as DetailedBalanceInput)!==detailedBalanceInputKey(input))return false;
 const result=v.result as Record<string,unknown>,snapshot=snapshotDetailedBalanceInput(input);
 const points=new WeakMap<object,boolean>();
 const q:Check=x=>{if(!detailedBalanceRecord(x))return false;if(points.has(x))return points.get(x)!;const valid=isDetailedBalanceObservation(x);points.set(x,valid);return valid;};
 const matrix=(size:number,check=q)=>vector(size,vector(size,check));
 const group=(size:number):Check=>x=>Array.isArray(x)&&x.length>0&&x.length<=size&&Array.from(x).every(n=>integer(n,0,size-1))&&new Set(x).size===x.length;
 const groups=(size:number):Check=>x=>Array.isArray(x)&&x.length>0&&x.length<=size&&Array.from(x).every(group(size));
 const partition=(size:number):Check=>x=>groups(size)(x)&&same((x as number[][]).flat().slice().sort(),Array.from({length:size},(_,i)=>i));
 const base={parameters:(x:unknown)=>same(x,snapshot.parameters),arithmetic:literal('exact-bigint-rational'),fullyResolved:literal(true),approximationContract:(x:unknown)=>typeof x==='string'&&x.length>0&&x.length<10000};
 if(input.kind===detailedBalanceKinds[0]){
  const row:Check=x=>shape(x,{step:n=>integer(n,0,input.parameters.stepCount),probabilities:vector(3,q),deviationFromUniform:vector(3,q),totalVariationFromUniform:q,contractionUpperBound:q});
  const n=input.parameters.stepCount;
  const p=input.parameters,irreducible=p.forwardWeight+p.backwardWeight>0;
  const period=irreducible&&p.stayWeight===0&&(p.forwardWeight===0||p.backwardWeight===0)?3:1;
  return shape(result,{...base,transition:matrix(3),timeReversedTransition:matrix(3),stationaryReference:vector(3,q),stationaryResidual:vector(3,q),stationaryFlux:matrix(3),stationaryCurrent:matrix(3),
   stationaryBalance:vector(3,x=>shape(x,{state:n=>integer(n,0,2),incoming:q,outgoing:q,holding:q,netOutflow:q})),
   clockwiseCurrent:q,conditionalForwardCycleProbability:q,conditionalBackwardCycleProbability:q,cycleProductDifference:q,contractionCoefficient:q,
   history:x=>vector(n+1,row)(x)&&(x as Record<string,unknown>[]).every((r,i)=>r.step===i),current:row,nextDistribution:vector(3,q),currentFlux:matrix(3),currentNetOutflow:vector(3,q),
   classification:x=>shape(x,{irreducible:literal(irreducible),stationaryReferenceIsUnique:literal(irreducible),stationaryFamily:literal(irreducible?'unique-uniform':'all-distributions'),detailedBalanceAtUniform:literal(p.forwardWeight===p.backwardWeight),
    returnPeriods:vector(3,literal(period)),uniqueClassPeriod:literal(irreducible?period:null),communicatingClasses:partition(3),convergesToUniformFromEveryInitial:literal(irreducible&&period===1),convergesToUniformForThisInitial:bool,currentDistributionIsStationary:bool})})
   &&same(result.current,(result.history as unknown[])[n])&&(result.stationaryBalance as Record<string,unknown>[]).every((r,i)=>r.state===i);
 }
 if(input.kind===detailedBalanceKinds[1]){
  const p=input.parameters;
  const acceptance:Check=x=>Array.isArray(x)&&x.length===3&&Array.from(x).every((row,i)=>Array.isArray(row)&&row.length===3&&Array.from(row).every((entry,j)=>(i===j?p.stayWeight:j===(i+1)%3?p.forwardWeight:p.backwardWeight)>0?q(entry):entry===null));
  const mode=(unique:boolean):Check=>x=>shape(x,{
   acceptance,acceptedProposal:matrix(3),rejectedProposal:matrix(3),rejectionByState:vector(3,q),selfProposalByState:vector(3,q),totalHoldingByState:vector(3,q),transition:matrix(3),
   targetFlux:matrix(3),targetCurrent:matrix(3),targetOneStep:vector(3,q),targetResidual:vector(3,q),targetInvariant:bool,targetDetailedBalance:bool,
   stationaryRank:literal(unique?2:0),stationaryFamilyDimension:literal(unique?0:2),stationaryDistribution:unique?vector(3,q):literal(null),
   stationaryFlux:unique?matrix(3):literal(null),stationaryCurrent:unique?matrix(3):literal(null),stationaryResidual:unique?vector(3,q):literal(null),stationaryTargetDistance:unique?q:literal(null),
   movementByState:vector(3,q),jumpTransition:unique?matrix(3):literal(null),jumpStationaryDistribution:unique?vector(3,q):literal(null),jumpStationaryResidual:unique?vector(3,q):literal(null),jumpTargetDistance:unique?q:literal(null),meanMovementAtStationary:unique?q:literal(null),
   contractionCoefficient:q,meanProposalAcceptanceAtTarget:q,meanProposalRejectionAtTarget:q,meanMovementAtTarget:q,
   classification:n=>shape(n,{irreducible:literal(unique),communicatingClasses:partition(3),returnPeriods:vector(3,choice(1,3)),uniqueClassPeriod:unique?choice(1,3):literal(null),stationaryIsUnique:literal(unique),targetIsUniqueStationary:bool,convergesToTargetFromEveryInitial:bool})
  });
  return shape(result,{...base,target:vector(3,q),proposal:matrix(3),corrected:mode(input.parameters.forwardWeight>0&&input.parameters.backwardWeight>0),uncorrected:mode(input.parameters.forwardWeight+input.parameters.backwardWeight>0),correctionChangesTransition:bool});
 }
 const p=input.parameters,unique=p.forwardNumerator+p.backwardNumerator>0,n=p.stepCount;
 const reference=(check:Check)=>unique?check:literal(null);
 const row:Check=x=>shape(x,{step:k=>integer(k,0,n),probabilities:vector(2,q),closedFormProbabilityOne:q,closedFormMatchesEvolution:literal(true),deviationFromStationary:reference(vector(2,q)),totalVariationFromStationary:reference(q),worstCaseTotalVariation:reference(q),forwardFlux:q,backwardFlux:q,netForwardFlux:q,cesaroProbabilityOne:q,cesaroDistanceFromStationary:reference(q)});
 const support=[...(p.backwardNumerator>0?[0]:[]),...(p.forwardNumerator>0?[1]:[])];
 const alternating=p.forwardNumerator+p.backwardNumerator===2*p.transitionDenominator;
 const limitExists=!unique||!alternating||2*p.initialOneNumerator===p.initialDenominator;
 const mixing:Check=x=>shape(x,{criterion:literal('worst-initial total variation at most 1/4'),
  status:!unique?literal('not-applicable-nonunique-stationary'):alternating?literal('never-periodic'):choice('not-reached-within-recorded-range','reached-within-recorded-range'),
  firstStep:nullable(k=>integer(k,0,n))})&&(x.status==='reached-within-recorded-range'?x.firstStep!==null:x.firstStep===null);
 return shape(result,{...base,transition:matrix(2),initial:vector(2,q),secondEigenvalue:q,absoluteContractionCoefficient:q,signedSpectralGap:q,absoluteSpectralGap:q,
  stationaryDistribution:reference(vector(2,q)),stationaryFamily:literal(unique?'unique-distribution':'all-distributions'),stationaryFamilyDimension:literal(unique?0:1),stationarySupport:x=>same(x,unique?support:null),
  stationaryBalanceFlow:reference(q),stationaryTimeReversal:x=>!unique?x===null:Array.isArray(x)&&x.length===2&&Array.from(x).every((r,i)=>support.includes(i)?vector(2,q)(r):r===null),
  stationaryIndicatorVariance:reference(q),stationaryLagOneCovariance:reference(q),stationaryLagOneCorrelation:p.forwardNumerator>0&&p.backwardNumerator>0?q:literal(null),
  history:x=>vector(n+1,row)(x)&&(x as Record<string,unknown>[]).every((r,i)=>r.step===i),current:row,nextDistribution:vector(2,q),cesaroAverage:vector(2,q),limitDistribution:limitExists?vector(2,q):literal(null),cesaroLimitDistribution:vector(2,q),
  mixingAtQuarter:mixing,
  classification:x=>shape(x,{irreducible:bool,communicatingClasses:partition(2),closedCommunicatingClasses:groups(2),returnPeriods:vector(2,nullable(choice(1,2))),uniqueClosedClassPeriod:unique?choice(1,2):literal(null),stationaryIsUnique:literal(unique),allStationaryLawsSatisfyDetailedBalance:literal(true),initialDistributionIsStationary:bool,currentDistributionIsStationary:bool,hasPointwiseLimitForThisInitial:bool,convergesToUniqueStationaryFromEveryInitial:bool,convergesToUniqueStationaryForThisInitial:unique?bool:literal(null)})
 })&&same(result.current,(result.history as unknown[])[n]);
}
