/** Owned transport shapes and preparation identities; numerical models are type-only imports. */
import type {OUMomentParameters,ouMomentModel} from './learning-ou-moments';
import type {FokkerVolumeParameters,fokkerVolumeModel} from './learning-fokker-volume';
import {isMasterObservation} from './learning-master-equation-protocol';
import {parseNumberJsonDraft} from './learning-json-number-input';
export const fokkerKinds=['ornstein-uhlenbeck-density-euler-moments','fokker-planck-cell-mass-boundaries']as const;
export type FokkerKind=typeof fokkerKinds[number];
export type FokkerInput={kind:typeof fokkerKinds[0];parameters:OUMomentParameters}|{kind:typeof fokkerKinds[1];parameters:FokkerVolumeParameters};
type Results={'ornstein-uhlenbeck-density-euler-moments':ReturnType<typeof ouMomentModel>;'fokker-planck-cell-mass-boundaries':ReturnType<typeof fokkerVolumeModel>};
export type FokkerSuccess={[K in FokkerKind]:{version:1;requestId:number;ok:true;kind:K;input:Extract<FokkerInput,{kind:K}>;inputKey:string;result:Results[K]}}[FokkerKind];
export type FokkerResponse=FokkerSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const fokkerRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
type Check=(v:unknown)=>boolean;
const shape=(v:unknown,s:Record<string,Check>):v is Record<string,unknown>=>fokkerRecord(v)&&Object.keys(v).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.prototype.hasOwnProperty.call(v,k)&&c(v[k]));
const integer=(v:unknown,min=0,max=1000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max;
const lit=(x:unknown):Check=>v=>v===x,choice=(...xs:unknown[]):Check=>v=>xs.includes(v),bool:Check=v=>typeof v==='boolean';
const array=(n:number,c:Check):Check=>v=>Array.isArray(v)&&v.length===n&&Array.from(v).every(c);
const fields=(s:string,c:Check)=>Object.fromEntries(s.split(' ').map(k=>[k,c]));
const same=(a:unknown,b:unknown)=>{try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
export const fokkerParameterKeys=[['relaxationNumerator','diffusionNumerator','coefficientDenominator','initialMeanNumerator','initialVarianceNumerator','initialDenominator','timeNumerator','timeDenominator','eulerSteps'],['cells','halfWidthNumerator','spaceDenominator','relaxationNumerator','windNumerator','diffusionNumerator','coefficientDenominator','timeNumerator','timeDenominator','eulerSteps','poissonCutoff','leftBoundary','rightBoundary','initialWeights']]as const;
/** Only the bounded rational workload predicate: no time integration or matrix solve. */
function withinVolumeWork(p:FokkerVolumeParameters){
 type Q={n:bigint;d:bigint};const q=(n:number,d=1):Q=>({n:BigInt(n),d:BigInt(d)}),add=(a:Q,b:Q):Q=>({n:a.n*b.d+b.n*a.d,d:a.d*b.d}),mul=(a:Q,b:Q):Q=>({n:a.n*b.n,d:a.d*b.d}),neg=(a:Q):Q=>({n:-a.n,d:a.d}),div=(a:Q,b:Q):Q=>({n:a.n*b.d,d:a.d*b.n}),pos=(a:Q):Q=>a.n>BigInt(0)?a:q(0),le=(a:Q,b:Q)=>a.n*b.d<=b.n*a.d;
 const L=q(p.halfWidthNumerator,p.spaceDenominator),dx=div(mul(q(2),L),q(p.cells)),gamma=q(p.relaxationNumerator,p.coefficientDenominator),D=q(p.diffusionNumerator,p.coefficientDenominator),wind=q(p.windNumerator,p.coefficientDenominator),diff=div(D,mul(dx,dx)),b=(i:number)=>add(wind,neg(mul(gamma,add(neg(L),mul(dx,q(i))))));
 return Array.from({length:p.cells},(_,i)=>add(i>0?add(div(pos(neg(b(i))),dx),diff):p.leftBoundary==='absorbing'?add(div(pos(neg(b(0))),dx),mul(q(2),diff)):q(0),i<p.cells-1?add(div(pos(b(i+1)),dx),diff):p.rightBoundary==='absorbing'?add(div(pos(b(p.cells)),dx),mul(q(2),diff)):q(0))).every(exit=>le(mul(exit,q(p.timeNumerator,p.timeDenominator)),q(64)));
}
export function isFokkerInput(v:unknown):v is FokkerInput{
 if(!shape(v,{kind:choice(...fokkerKinds),parameters:fokkerRecord}))return false;
 const family=fokkerKinds.indexOf(v.kind as FokkerKind),p=v.parameters as Record<string,unknown>;
 const specs=Object.fromEntries(fokkerParameterKeys[family].map(k=>[k,k==='initialWeights'?array(typeof p.cells==='number'?p.cells:0,x=>integer(x)):k.endsWith('Boundary')?choice('reflecting','absorbing'):(x:unknown)=>integer(x,['initialMeanNumerator','windNumerator'].includes(k)?-1000000:k.endsWith('Denominator')||['eulerSteps','halfWidthNumerator'].includes(k)?1:0,k==='eulerSteps'?128:k==='poissonCutoff'?256:1000000)]));
 if(!shape(p,specs))return false;const a=p as unknown as Record<string,number>;
 if(a.timeNumerator>10*a.timeDenominator)return false;
 if(family===0)return a.relaxationNumerator<=5*a.coefficientDenominator&&a.diffusionNumerator<=2*a.coefficientDenominator&&Math.abs(a.initialMeanNumerator)<=5*a.initialDenominator&&a.initialVarianceNumerator<=5*a.initialDenominator;
 return integer(a.cells,2,8)&&2*a.halfWidthNumerator>=a.spaceDenominator&&a.halfWidthNumerator<=4*a.spaceDenominator&&a.relaxationNumerator<=2*a.coefficientDenominator&&Math.abs(a.windNumerator)<=2*a.coefficientDenominator&&a.diffusionNumerator<=a.coefficientDenominator&&(p.initialWeights as number[]).some(x=>x>0)&&withinVolumeWork(p as unknown as FokkerVolumeParameters);
}
export function snapshotFokkerInput(input:FokkerInput):FokkerInput{
 if(!isFokkerInput(input))throw new RangeError('Invalid Fokker–Planck preparation');
 const p=input.parameters as unknown as Record<string,unknown>;
 return Object.freeze({kind:input.kind,parameters:Object.freeze(Object.fromEntries(fokkerParameterKeys[fokkerKinds.indexOf(input.kind)].map(k=>[k,Array.isArray(p[k])?Object.freeze([...(p[k]as number[])].map(x=>x+0)):typeof p[k]==='number'?(p[k]as number)+0:p[k]])))})as unknown as FokkerInput;
}
export const fokkerInputKey=(v:FokkerInput)=>JSON.stringify(snapshotFokkerInput(v));
export const isFokkerRequestEnvelope=(v:unknown):v is {version:1;requestId:number;input:unknown}=>shape(v,{version:lit(1),requestId:x=>Number.isSafeInteger(x)&&(x as number)>0,input:()=>true});
export function parseFokkerStoredInput(raw:string,kind:FokkerKind):FokkerInput|null{
 try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:lit(1),value:isFokkerInput})&&(v.value as FokkerInput).kind===kind?snapshotFokkerInput(v.value as FokkerInput):null;}catch{return null;}
}
/** Strict response structure and availability, not a proof of a sender's numerical claims. */
export function isFokkerSuccessForInput(v:unknown,input:FokkerInput,id:number):v is FokkerSuccess{
 if(!Number.isSafeInteger(id)||id<=0||!isFokkerInput(input)||!shape(v,{version:lit(1),requestId:lit(id),ok:lit(true),kind:lit(input.kind),input:isFokkerInput,inputKey:lit(fokkerInputKey(input)),result:fokkerRecord})||fokkerInputKey(v.input as FokkerInput)!==fokkerInputKey(input))return false;
 const result=v.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
 const obs:Check=x=>{if(!fokkerRecord(x))return false;if(cache.has(x))return cache.get(x)!;const ok=isMasterObservation(x);cache.set(x,ok);return ok;},exact:Check=x=>isMasterObservation(x,true),vector=(n:number,c:Check=obs)=>array(n,c),matrix=(n:number,c:Check=exact)=>array(n,vector(n,c));
 const base={parameters:(x:unknown)=>same(x,snapshotFokkerInput(input).parameters),precisionBits:lit(256),approximationContract:(x:unknown)=>typeof x==='string'&&x.length>0&&x.length<12000};
 const numbered=(rows:unknown,k:string)=>Array.isArray(rows)&&rows.every((r,i)=>fokkerRecord(r)&&r[k]===i);
 if(input.kind===fokkerKinds[0]){
  const p=input.parameters,hasStationary=p.relaxationNumerator>0,stationary=hasStationary?p.initialMeanNumerator===0&&p.initialVarianceNumerator*p.relaxationNumerator===p.diffusionNumerator*p.initialDenominator:p.diffusionNumerator===0;
  const g=BigInt(p.relaxationNumerator)*BigInt(p.timeNumerator),h=BigInt(p.coefficientDenominator)*BigInt(p.timeDenominator)*BigInt(p.eulerSteps),contract=g>BigInt(0)&&g<BigInt(2)*h;
  const row:Check=x=>{if(!fokkerRecord(x)||!integer(x.step,0,p.eulerSteps))return false;const atom=p.initialVarianceNumerator===0&&(p.diffusionNumerator===0||p.timeNumerator===0||x.step===0);return shape(x,{step:y=>integer(y,0,p.eulerSteps),...fields('time mean variance signedMeanError signedVarianceError meanRate varianceRate initialTimeCovariance',obs),...fields('emMean emVariance normalizationResidual',exact),signedVarianceDeviationFromStationary:hasStationary?obs:lit(null),stationarySDETwoTimeCovariance:hasStationary&&stationary?obs:lit(null),stationaryFrozenODETwoTimeCovariance:hasStationary&&stationary?exact:lit(null),lawKind:lit(atom?'point-mass':'ordinary-gaussian-density'),scoreDefined:lit(!atom),emLawIsProbability:lit(true)});};
  return shape(result,{...base,units:x=>shape(x,{position:lit('m'),time:lit('s'),gamma:lit('s^-1'),diffusion:lit('m^2/s'),variance:lit('m^2')}),...fields('relaxation diffusion initialMean initialVariance stepTime emMeanMultiplier emNoiseVariancePerStep',exact),stationaryVariance:hasStationary?exact:lit(null),emInvariantVariance:contract?exact:lit(null),classification:x=>shape(x,{initialLawIsStationary:lit(stationary),stationaryProbability:lit(hasStationary?'unique':p.diffusionNumerator>0?'none-on-real-line':'all-initial-laws'),stationaryLawKind:lit(!hasStationary?null:p.diffusionNumerator===0?'point-mass':'ordinary-gaussian-density'),emStrictMeanContraction:lit(contract),emMarginalMeanMode:lit(g===BigInt(0)||g===BigInt(2)*h),emExpandingMeanMode:lit(g>BigInt(2)*h),emAlwaysValidGaussianOrAtom:lit(true),emInvariantVarianceIsUnique:lit(contract),probabilityFlowDefinedAtInitialTime:lit(p.initialVarianceNumerator>0),positiveTimeRegularizesAtom:lit(p.diffusionNumerator>0&&p.initialVarianceNumerator===0)}),history:array(p.eulerSteps+1,row),current:row})&&numbered(result.history,'step')&&same(result.current,(result.history as unknown[]).at(-1));
 }
 const p=input.parameters,n=p.cells,size=n+2,stationary=p.leftBoundary==='reflecting'&&p.rightBoundary==='reflecting'&&p.diffusionNumerator>0;
 const row:Check=x=>shape(x,{step:y=>integer(y,0,p.eulerSteps),time:exact,...fields('explicitLaw implicitLaw',vector(size,exact)),...fields('explicitDensity implicitDensity',vector(n,exact)),...fields('explicitFaceFlux implicitFaceFlux',vector(n+1,exact)),...fields('explicitInteriorMass implicitInteriorMass explicitGlobalMassResidual implicitGlobalMassResidual',exact),explicitLawIsProbability:bool,implicitLawIsProbability:lit(true)});
 const mix:Check=x=>shape(x,{cutoff:y=>integer(y,0,p.poissonCutoff),conditionalLaw:vector(size,exact),partialLaw:vector(size),partialMass:obs,omittedPoissonMass:obs,exactGridLawEnclosures:vector(size)});
 return shape(result,{...base,stateOrder:lit('interior cells, escaped-left, escaped-right'),...fields('cellWidth leftEscapeRate rightEscapeRate maximumExitRate poissonMean stepTime',exact),faces:vector(n+1,exact),cellCenters:vector(n,exact),faceDrift:vector(n+1,exact),initialLaw:vector(size,exact),initialDensity:vector(n,exact),generator:matrix(size),explicitStepKernel:matrix(size),implicitStepKernel:matrix(size),diffusiveClosedStationaryLaw:stationary?vector(n,exact):lit(null),classification:x=>shape(x,{...fields('explicitKernelIsStochastic explicitHistoryIsProbability initialLawIsStationary absorbingBoundaryHasPositiveEscape',bool),implicitKernelIsStochastic:lit(true),bothBoundariesReflecting:lit(p.leftBoundary==='reflecting'&&p.rightBoundary==='reflecting'),positiveOmittedPoissonMass:bool}),history:array(p.eulerSteps+1,row),current:row,mixtures:array(p.poissonCutoff+1,mix),reference:mix,...fields('signedExplicitTerminalError signedImplicitTerminalError exactGridTerminalLaw',vector(size))})&&numbered(result.history,'step')&&numbered(result.mixtures,'cutoff')&&same(result.current,(result.history as unknown[]).at(-1))&&same(result.reference,(result.mixtures as unknown[]).at(-1))&&same(result.exactGridTerminalLaw,(result.reference as Record<string,unknown>).exactGridLawEnclosures);
}
