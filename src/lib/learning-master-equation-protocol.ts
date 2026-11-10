import {parseNumberJsonDraft} from './learning-json-number-input';
/** Transport shape and preparation identity checks; no numerical model imports.
 * This validates a message, not the mathematical truthfulness of its sender. */
import type {masterTwoStateModel,MasterTwoStateParameters} from './learning-master-two-state';
import type {uniformizationModel,UniformizationParameters} from './learning-master-uniformization';
import type {holdingModel,HoldingParameters} from './learning-master-holding';
export const masterKinds=['two-state-continuous-time-generator-euler','poisson-uniformization-potential-events','state-dependent-holding-time-jump-occupation']as const;
export type MasterKind=typeof masterKinds[number];
export type MasterInput={kind:typeof masterKinds[0];parameters:MasterTwoStateParameters}|{kind:typeof masterKinds[1];parameters:UniformizationParameters}|{kind:typeof masterKinds[2];parameters:HoldingParameters};
type Results={'two-state-continuous-time-generator-euler':ReturnType<typeof masterTwoStateModel>;'poisson-uniformization-potential-events':ReturnType<typeof uniformizationModel>;'state-dependent-holding-time-jump-occupation':ReturnType<typeof holdingModel>};
export type MasterSuccess={[K in MasterKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<MasterInput,{kind:K}>;inputKey:string;result:Results[K]}}[MasterKind];
export type MasterResponse=MasterSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const masterRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
type Check=(v:unknown)=>boolean;
const shape=(v:unknown,spec:Record<string,Check>):v is Record<string,unknown>=>masterRecord(v)&&Object.keys(v).length===Object.keys(spec).length&&Object.entries(spec).every(([k,c])=>Object.prototype.hasOwnProperty.call(v,k)&&c(v[k]));
const integer=(v:unknown,low=0,high=1000000):v is number=>typeof v==='number'&&Number.isInteger(v)&&v>=low&&v<=high;
const literal=(x:unknown):Check=>v=>v===x,choice=(...xs:unknown[]):Check=>v=>xs.includes(v),bool:Check=v=>typeof v==='boolean',finite:Check=v=>typeof v==='number'&&Number.isFinite(v);
const nullable=(c:Check):Check=>v=>v===null||c(v),array=(n:number,c:Check):Check=>v=>Array.isArray(v)&&v.length===n&&Array.from(v).every(c),fields=(names:string,c:Check)=>Object.fromEntries(names.split(' ').map(k=>[k,c]));
export const masterParameterKeys=[['forwardNumerator','backwardNumerator','rateDenominator','timeNumerator','timeDenominator','initialOneNumerator','initialDenominator','eulerSteps'],['clockwiseNumerator','counterclockwiseNumerator','rateDenominator','clockNumerator','clockDenominator','timeNumerator','timeDenominator','initialZeroWeight','initialOneWeight','initialTwoWeight','poissonCutoff'],['rateZeroNumerator','rateOneNumerator','rateTwoNumerator','rateDenominator','clockNumerator','clockDenominator','timeNumerator','timeDenominator','initialZeroWeight','initialOneWeight','initialTwoWeight','poissonCutoff']]as const;
export function isMasterInput(value:unknown):value is MasterInput{
 if(!shape(value,{kind:choice(...masterKinds),parameters:masterRecord}))return false;
 const family=masterKinds.indexOf(value.kind as MasterKind),p=value.parameters as Record<string,number>,keys=masterParameterKeys[family];
 if(!shape(p,Object.fromEntries(keys.map(k=>[k,(v:unknown)=>integer(v,k.endsWith('Denominator')||k==='eulerSteps'?1:0,k==='eulerSteps'?128:k==='poissonCutoff'?256:1000000)]))))return false;
 if(p.timeNumerator>10*p.timeDenominator)return false;
 if(family===0)return p.forwardNumerator<=5*p.rateDenominator&&p.backwardNumerator<=5*p.rateDenominator&&p.initialOneNumerator<=p.initialDenominator;
 if(p.clockNumerator>10*p.clockDenominator||p.initialZeroWeight+p.initialOneWeight+p.initialTwoWeight===0)return false;
 const rates=family===1?[p.clockwiseNumerator,p.counterclockwiseNumerator]:[p.rateZeroNumerator,p.rateOneNumerator,p.rateTwoNumerator];
 return rates.every(r=>r<=5*p.rateDenominator)&&(family===1?rates.reduce((a,b)=>a+b,0):Math.max(...rates))*p.clockDenominator<=p.clockNumerator*p.rateDenominator;
}
export function snapshotMasterInput(input:MasterInput):MasterInput{
 if(!isMasterInput(input))throw new RangeError('Invalid master-equation preparation');
 const p=input.parameters as unknown as Record<string,number>,keys=masterParameterKeys[masterKinds.indexOf(input.kind)];
 return Object.freeze({kind:input.kind,parameters:Object.freeze(Object.fromEntries(keys.map(k=>[k,p[k]+0])))})as unknown as MasterInput;
}
export const masterInputKey=(input:MasterInput)=>JSON.stringify(snapshotMasterInput(input));
export const isMasterRequestEnvelope=(v:unknown):v is{version:1;requestId:number;input:unknown}=>shape(v,{version:literal(1),requestId:v=>Number.isSafeInteger(v)&&(v as number)>0,input:()=>true});
const same=(a:unknown,b:unknown)=>{if(a===b)return true;try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
export function isMasterObservation(v:unknown,exactOnly=false):boolean{
 if(!shape(v,{value:nullable(finite),sign:choice(-1,0,1,null),logMagnitude:nullable(finite),logResolution:choice('finite','not-applicable','unresolved'),resolution:choice('exact-rational','structural-zero',...(!exactOnly?['interval-resolved','unresolved-at-precision-ceiling']:[])),bounds:masterRecord}))return false;
 const term=(x:unknown,den=false)=>shape(x,{coefficient:v=>typeof v==='string'&&v.length<=32768&&(den?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v)&&v!=='-0',exponent:literal(0)});
 const rational:Check=x=>shape(x,{numerator:y=>term(y),denominator:y=>term(y,true)});
 if(!shape(v.bounds,{lower:rational,upper:rational}))return false;
 const bounds=v.bounds as {lower:{numerator:{coefficient:string};denominator:{coefficient:string}};upper:{numerator:{coefficient:string};denominator:{coefficient:string}}};
 const a=bounds.lower.numerator.coefficient,b=bounds.upper.numerator.coefficient;
 if(v.resolution==='structural-zero')return v.value===0&&v.sign===0&&v.logMagnitude===null&&v.logResolution==='not-applicable'&&same(bounds.lower,bounds.upper)&&a==='0'&&bounds.lower.denominator.coefficient==='1';
 if(v.resolution==='unresolved-at-precision-ceiling'){const sign=a!=='0'&&a[0]!=='-'?1:b[0]==='-'?-1:null;return v.value===null&&v.logMagnitude===null&&v.logResolution==='unresolved'&&v.sign===sign&&!same(bounds.lower,bounds.upper);}
 if(a==='0'||b==='0'||v.sign!==(a[0]==='-'?-1:1)||v.sign!==(b[0]==='-'?-1:1)||!finite(v.logMagnitude)||v.logResolution!=='finite')return false;
 if(v.value!==null&&(v.value===0||Math.sign(v.value as number)!==v.sign))return false;
 return v.resolution==='exact-rational'?same(bounds.lower,bounds.upper):!same(bounds.lower,bounds.upper);
}
export function isMasterSuccessForInput(v:unknown,input:MasterInput,requestId:number):v is MasterSuccess{
 if(!Number.isSafeInteger(requestId)||requestId<=0||!isMasterInput(input)||!shape(v,{version:literal(1),requestId:literal(requestId),ok:literal(true),kind:literal(input.kind),input:isMasterInput,inputKey:literal(masterInputKey(input)),result:masterRecord})||masterInputKey(v.input as MasterInput)!==masterInputKey(input))return false;
 const p=input.parameters as unknown as Record<string,number>,result=v.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const q:Check=x=>{if(!masterRecord(x))return false;if(cache.has(x))return cache.get(x)!;const ok=isMasterObservation(x);cache.set(x,ok);return ok;},exact:Check=x=>isMasterObservation(x,true);
 const vector=(n:number,c:Check=q)=>array(n,c),matrix=(n:number,c:Check=q)=>array(n,vector(n,c));
 const holding:Check=x=>shape(x,{kind:literal('infinite')})||shape(x,{kind:literal('finite'),value:exact});
 const base={parameters:(x:unknown)=>same(x,snapshotMasterInput(input).parameters),precisionBits:literal(256),rateUnits:literal('s^-1'),timeUnits:literal('s'),approximationContract:(x:unknown)=>typeof x==='string'&&x.length>0&&x.length<12000};
 if(input.kind===masterKinds[0]){
  const row:Check=x=>shape(x,{step:x=>integer(x,0,p.eulerSteps),...fields('time relaxationFactor signedEulerError forwardFlux backwardFlux netFlux exactNormalizationResidual eulerNormalizationResidual',q),exactProbabilities:vector(2),eulerValues:vector(2,exact),signedDeviationFromUniqueStationary:p.forwardNumerator+p.backwardNumerator>0?q:literal(null),eulerRowIsProbability:bool});
  return shape(result,{...base,generator:matrix(2,exact),eulerStepKernel:matrix(2,exact),exactTimeKernel:matrix(2),rates:vector(2,exact),...fields('relaxationRate eulerStepTime eulerDeviationMultiplier',exact),stationaryDistribution:p.forwardNumerator+p.backwardNumerator>0?vector(2,exact):literal(null),selectedLimitDistribution:vector(2,exact),meanHoldingTimes:x=>array(2,holding)(x)&&(x as {kind:string}[]).every((h,i)=>h.kind===([p.forwardNumerator,p.backwardNumerator][i]===0?'infinite':'finite')),noJumpSurvival:vector(2),history:array(p.eulerSteps+1,row),current:row,classification:x=>shape(x,{...fields('initialLawIsStationary currentExactLawIsStationary eulerKernelIsStochastic eulerSelectedHistoryIsProbability eulerLinearModeStrictlyContracts eulerLinearModeIsMarginal eulerLinearModeExpands',bool),stationaryIsUnique:literal(p.forwardNumerator+p.backwardNumerator>0),irreducible:literal(p.forwardNumerator>0&&p.backwardNumerator>0),stationaryFamily:literal(p.forwardNumerator+p.backwardNumerator>0?'unique':'all-initial-laws'),generatorConservesMass:literal(true),eulerKernelConservesMass:literal(true)})})&&(result.history as {step:number}[]).every((r,i)=>r.step===i)&&same(result.current,(result.history as unknown[]).at(-1));
 }
 const poissonRow:Check=x=>shape(x,{cutoff:x=>integer(x,0,p.poissonCutoff),conditionalLaw:vector(3,exact),poissonWeight:q,partialLaw:vector(3),partialMass:q,omittedPoissonMass:q,exactLawEnclosures:vector(3),componentErrorUpperBound:exact});
 const mixture={history:array(p.poissonCutoff+1,poissonRow),current:poissonRow,tailProof:(x:unknown)=>shape(x,{lastSummedIndex:x=>integer(x,0,1024),firstOmittedBareCoefficient:exact,bareRemainderUpperBound:exact,relativeTargetBits:literal(256)})};
 let valid:boolean;
 if(input.kind===masterKinds[1])valid=shape(result,{...base,...mixture,generator:matrix(3,exact),potentialEventKernel:matrix(3,exact),initialLaw:vector(3,exact),...fields('exitRate potentialEventRate poissonMean potentialSelfLoopProbability expectedPotentialEvents expectedActualStateChanges',exact),stationaryDistribution:p.clockwiseNumerator+p.counterclockwiseNumerator>0?vector(3,exact):literal(null),stationaryDirectedEdgeCurrent:p.clockwiseNumerator+p.counterclockwiseNumerator>0?exact:literal(null),classification:x=>shape(x,{generatorConservesMass:literal(true),kernelIsStochastic:literal(true),stationaryIsUnique:literal(p.clockwiseNumerator+p.counterclockwiseNumerator>0),irreducible:literal(p.clockwiseNumerator+p.counterclockwiseNumerator>0),stationaryFamily:literal(p.clockwiseNumerator+p.counterclockwiseNumerator>0?'unique-uniform':'all-initial-laws'),embeddedPeriod:choice(1,3),...fields('exactLawConvergesToUniqueStationaryFromEveryInitialLaw potentialEventsEqualStateChanges positiveOmittedMass truncatedMassIsOne',bool)})});
 else {const rates=[p.rateZeroNumerator,p.rateOneNumerator,p.rateTwoNumerator],zeros=rates.flatMap((v,i)=>v===0?[i]:[]),active=zeros.length===0;valid=shape(result,{...base,...mixture,exitRates:vector(3,exact),generator:matrix(3,exact),potentialEventKernel:matrix(3,exact),actualJumpKernelRows:x=>array(3,nullable(vector(3,exact)))(x)&&(x as unknown[]).every((row,i)=>rates[i]===0?row===null:row!==null),initialLaw:vector(3,exact),meanHoldingTimes:x=>array(3,holding)(x)&&(x as {kind:string}[]).every((h,i)=>h.kind===(rates[i]===0?'infinite':'finite')),noActualJumpSurvival:vector(3),stationaryDistribution:active||zeros.length===1?vector(3,exact):literal(null),stationaryFamilyAbsorbingStates:x=>same(x,zeros),selectedLimitDistribution:vector(3,exact),longRunActualJumpVisitLaw:active?vector(3,exact):literal(null),meanCompleteCycleTime:active?exact:literal(null),stationaryMeanJumpRate:exact,stationaryPerEdgeCurrent:exact,selectedAbsorptionTargets:active?literal(null):array(3,x=>zeros.includes(x as number)),selectedMeanAbsorptionTime:active?literal(null):exact,potentialSelfLoopProbabilities:vector(3,exact),classification:x=>shape(x,{...fields('irreducible stationaryIsUnique initialLawIsStationary actualJumpSequenceTerminatesAlmostSurely actualJumpKernelDefinedOnEveryState convergesToUniqueStationaryFromEveryInitialLaw actualCountHasOneHomogeneousPoissonLawForAllInitialStates positiveOmittedMass',bool),stationaryFamily:choice('unique-holding-weighted','unique-absorbing','simplex-on-absorbing-states'),actualJumpChainPeriod:literal(active?3:null),selectedLawHasLimit:literal(true)})});}
 return valid&&(result.history as {cutoff:number}[]).every((r,i)=>r.cutoff===i)&&same(result.current,(result.history as unknown[]).at(-1));
}

/** Validate the original numeric lexemes before JSON rounding can lose digits. */
export function parseMasterStoredInput(raw:string,kind:MasterKind):MasterInput|null{
 try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:literal(1),value:isMasterInput})&&(v.value as MasterInput).kind===kind?snapshotMasterInput(v.value as MasterInput):null;}catch{return null;}
}
