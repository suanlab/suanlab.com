/** Own input/response shapes; no numerical-model evaluation in UI imports. */
import type{CirculatingGaussianParameters,circulatingGaussianModel}from'./learning-circulating-gaussian';
import {isMasterObservation}from'./learning-master-equation-protocol';
import {parseNumberJsonDraft}from'./learning-json-number-input';
export const currentKinds=['stationary-gaussian-circulating-current']as const;
export type CurrentKind=typeof currentKinds[number];
export type CurrentInput={kind:CurrentKind;parameters:CirculatingGaussianParameters};
type Q=ReturnType<typeof circulatingGaussianModel>['density'];
export interface CurrentDiagram{densityExtent:Q;probeExtent:Q;fields:{extent:Q;samples:{position:Q[];drift:Q[];diffusive:Q[];flow:Q[]}[]}[]}
export type CurrentSuccess={version:1;requestId:number;ok:true;kind:CurrentKind;input:CurrentInput;inputKey:string;result:ReturnType<typeof circulatingGaussianModel>;diagram:CurrentDiagram};
export type CurrentResponse=CurrentSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const currentRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
type Check=(v:unknown)=>boolean;
const shape=(v:unknown,s:Record<string,Check>):v is Record<string,unknown>=>currentRecord(v)&&Object.keys(v).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.prototype.hasOwnProperty.call(v,k)&&c(v[k]));
const int=(v:unknown,min=0,max=1000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max,lit=(x:unknown):Check=>v=>v===x,nullable=(c:Check):Check=>v=>v===null||c(v),array=(n:number,c:Check):Check=>v=>Array.isArray(v)&&v.length===n&&Array.from(v).every(c),fields=(s:string,c:Check)=>Object.fromEntries(s.split(' ').map(k=>[k,c]));
const same=(a:unknown,b:unknown)=>{try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
export const currentParameterKeys=['relaxationNumerator','diffusionNumerator','coefficientDenominator','angularNumerator','angularDenominator','probeXNumerator','probeYNumerator','spaceDenominator','timeNumerator','timeDenominator','eulerSteps']as const;
export function isCurrentInput(v:unknown):v is CurrentInput{
 if(!shape(v,{kind:lit(currentKinds[0]),parameters:currentRecord}))return false;
 const p=v.parameters as Record<string,number>;
 return shape(p,Object.fromEntries(currentParameterKeys.map(k=>[k,(x:unknown)=>int(x,['angularNumerator','probeXNumerator','probeYNumerator'].includes(k)?-1000000:k==='timeNumerator'?0:1,k==='eulerSteps'?64:1000000)])))
 &&p.relaxationNumerator<=2*p.coefficientDenominator&&p.diffusionNumerator<=2*p.coefficientDenominator&&Math.abs(p.angularNumerator)<=4*p.angularDenominator
 &&Math.abs(p.probeXNumerator)<=4*p.spaceDenominator&&Math.abs(p.probeYNumerator)<=4*p.spaceDenominator&&p.timeNumerator<=4*p.timeDenominator;
}
export function snapshotCurrentInput(input:CurrentInput):CurrentInput{if(!isCurrentInput(input))throw new RangeError('Invalid probability-current input');return Object.freeze({kind:input.kind,parameters:Object.freeze(Object.fromEntries(currentParameterKeys.map(k=>[k,input.parameters[k]+0])))})as unknown as CurrentInput;}
export const currentInputKey=(input:CurrentInput)=>JSON.stringify(snapshotCurrentInput(input));
export const isCurrentRequestEnvelope=(v:unknown):v is{version:1;requestId:number;input:unknown}=>shape(v,{version:lit(1),requestId:x=>Number.isSafeInteger(x)&&(x as number)>0,input:()=>true});
export function parseCurrentStoredInput(raw:string,kind:CurrentKind):CurrentInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:lit(1),value:isCurrentInput})&&(v.value as CurrentInput).kind===kind?snapshotCurrentInput(v.value as CurrentInput):null;}catch{return null;}}
/** Structural/availability validation does not prove mathematical truth. */
export function isCurrentSuccessForInput(v:unknown,input:CurrentInput,id:number):v is CurrentSuccess{
 if(!Number.isSafeInteger(id)||id<=0||!isCurrentInput(input)||!shape(v,{version:lit(1),requestId:lit(id),ok:lit(true),kind:lit(input.kind),input:isCurrentInput,inputKey:lit(currentInputKey(input)),result:currentRecord,diagram:currentRecord})||currentInputKey(v.input as CurrentInput)!==currentInputKey(input))return false;
 const p=input.parameters,r=v.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();const q:Check=x=>{if(!currentRecord(x))return false;if(cache.has(x))return cache.get(x)!;const ok=isMasterObservation(x);cache.set(x,ok);return ok;},exact:Check=x=>isMasterObservation(x,true),vec=(n:number,c:Check=q)=>array(n,c),matrix=(c:Check=q)=>array(2,vec(2,c));
 const g=BigInt(p.relaxationNumerator),d=BigInt(p.coefficientDenominator),w=BigInt(p.angularNumerator),wd=BigInt(p.angularDenominator),tn=BigInt(p.timeNumerator),td=BigInt(p.timeDenominator)*BigInt(p.eulerSteps);
 const den=d*wd*td,a=den-g*wd*tn,b=w*d*tn,norm=a*a+b*b,base=den*den,contract=norm<base,origin=p.probeXNumerator===0&&p.probeYNumerator===0;
 const row:Check=x=>shape(x,{step:x=>int(x,0,p.eulerSteps),...fields('time angleRadians exactCoordinateVariance eulerCoordinateVariance signedEulerVarianceError normalizationResidual',exact),...fields('sine cosine relaxationEnvelope',q),sdeCovXtX0:matrix(),probabilityFlowCovXtX0:matrix(),eulerCovXtX0:matrix(exact),mean:vec(2,exact),eulerLawIsProbability:lit(true)});
 const good=shape(r,{parameters:x=>same(x,snapshotCurrentInput(input).parameters),precisionBits:lit(256),...fields('stationaryCoordinateVariance probeRadiusSquared stationaryCurrentDivergence circleNormalProbabilityFlux stepTime eulerAmplificationSquared eulerNoiseVariance linearObservableAsymptoticVariance radialObservableAsymptoticVariance linearAsymptoticVarianceRatioToZeroRotation',exact),...fields('density logDensity velocityCirculation currentTangentialLineIntegral',q),...fields('driftVelocity diffusiveProbabilityVelocity probabilityFlowVelocity',vec(2,exact)),...fields('driftCurrent diffusiveCurrent totalCurrent',vec(2)),logAbsoluteCurrent:x=>array(2,nullable(q))(x)&&(x as unknown[]).every((x,i)=>p.angularNumerator===0||[p.probeYNumerator,p.probeXNumerator][i]===0?x===null:x!==null),eulerInvariantCoordinateVariance:contract?exact:lit(null),classification:x=>shape(x,{densityStrictlyPositive:lit(true),initialLawIsStationary:lit(true),coordinatesAreEvenPositions:lit(true),probeCurrentIsExactlyZero:lit(p.angularNumerator===0||origin),ordinaryStationaryCurrentVanishesEverywhere:lit(p.angularNumerator===0),eulerStrictMeanContraction:lit(contract),eulerMarginalMeanMode:lit(norm===base),eulerExpandingMeanMode:lit(norm>base),eulerGaussianLawsRemainProbabilities:lit(true),radialAsymptoticVarianceChangesWithRotation:lit(false),linearDriftRealDecayRateIndependentOfRotation:lit(true)}),history:array(p.eulerSteps+1,row),current:row,approximationContract:x=>typeof x==='string'&&x.length>0&&x.length<12000});
 return good&&(r.history as {step:number}[]).every((x,i)=>x.step===i)&&same(r.current,(r.history as unknown[]).at(-1))&&shape(v.diagram,{densityExtent:exact,probeExtent:exact,fields:array(2,x=>shape(x,{extent:exact,samples:array(49,x=>shape(x,{position:vec(2,exact),drift:vec(2,exact),diffusive:vec(2,exact),flow:vec(2,exact)}))}))});
}
