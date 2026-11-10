/** Identity/shape/availability checks only. No numerical evaluator in UI imports. */
import type { ThermalOuResponseParameters, thermalOuResponseModel } from './learning-thermal-ou-response';
import { isMasterObservation } from './learning-master-equation-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export const fdtKinds=['thermal-ou-fluctuation-response'] as const;
export type FdtKind=typeof fdtKinds[number];
export type FdtInput={kind:FdtKind;parameters:ThermalOuResponseParameters};
export type FdtSuccess={version:1;requestId:number;ok:true;kind:FdtKind;input:FdtInput;inputKey:string;result:ReturnType<typeof thermalOuResponseModel>};
export type FdtResponse=FdtSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const fdtRecord=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&Object.getPrototypeOf(v)===Object.prototype;
type Check=(v:unknown)=>boolean;
const shape=(v:unknown,s:Record<string,Check>):v is Record<string,unknown>=>fdtRecord(v)&&Reflect.ownKeys(v).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.prototype.hasOwnProperty.call(v,k)&&c(v[k]));
const integer=(v:unknown,min=1,max=1000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max;
const lit=(value:unknown):Check=>v=>v===value;
const array=(n:number,c:Check):Check=>v=>Array.isArray(v)&&Object.getPrototypeOf(v)===Array.prototype&&v.length===n&&Reflect.ownKeys(v).length===n+1&&Array.from({length:n},(_,i)=>Object.prototype.hasOwnProperty.call(v,i)&&c(v[i])).every(Boolean);
const fields=(keys:string,c:Check)=>Object.fromEntries(keys.split(' ').map(k=>[k,c]));
const same=(a:unknown,b:unknown)=>{try{return JSON.stringify(a)===JSON.stringify(b);}catch{return false;}};
export const fdtParameterKeys=['mobilityNumerator','stiffnessNumerator','thermalEnergyNumerator','coefficientDenominator','angularNumerator','angularDenominator','forceXNumerator','forceYNumerator','forceDenominator','impulseXNumerator','impulseYNumerator','impulseDenominator','timeNumerator','timeDenominator','sampleSteps'] as const;
export function isFdtInput(v:unknown):v is FdtInput {
  if(!shape(v,{kind:lit(fdtKinds[0]),parameters:fdtRecord}))return false;
  const p=v.parameters as Record<string,number>,signed=['angularNumerator','forceXNumerator','forceYNumerator','impulseXNumerator','impulseYNumerator'];
  return shape(p,Object.fromEntries(fdtParameterKeys.map(k=>[k,(x:unknown)=>integer(x,signed.includes(k)?-1000000:k==='timeNumerator'?0:1,k==='sampleSteps'?64:1000000)])))
    &&Math.abs(p.angularNumerator)<=4*p.angularDenominator&&Math.abs(p.forceXNumerator)<=4*p.forceDenominator&&Math.abs(p.forceYNumerator)<=4*p.forceDenominator
    &&Math.abs(p.impulseXNumerator)<=4*p.impulseDenominator&&Math.abs(p.impulseYNumerator)<=4*p.impulseDenominator&&p.timeNumerator<=4*p.timeDenominator;
}
export function snapshotFdtInput(input:FdtInput):FdtInput {
  if(!isFdtInput(input))throw new RangeError('Invalid thermal OU preparation');
  return Object.freeze({kind:input.kind,parameters:Object.freeze(Object.fromEntries(fdtParameterKeys.map(k=>[k,input.parameters[k]+0])))}) as unknown as FdtInput;
}
export const fdtInputKey=(input:FdtInput)=>JSON.stringify(snapshotFdtInput(input));
export const isFdtRequestEnvelope=(v:unknown):v is {version:1;requestId:number;input:unknown}=>shape(v,{version:lit(1),requestId:x=>integer(x,1,Number.MAX_SAFE_INTEGER),input:()=>true});
export function parseFdtStoredInput(raw:string,kind:FdtKind):FdtInput|null {
  try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:lit(1),value:isFdtInput})&&(v.value as FdtInput).kind===kind?snapshotFdtInput(v.value as FdtInput):null;}catch{return null;}
}
export function isFdtSuccessForInput(v:unknown,input:FdtInput,id:number):v is FdtSuccess {
  if(!integer(id,1,Number.MAX_SAFE_INTEGER)||!isFdtInput(input)||!shape(v,{version:lit(1),requestId:lit(id),ok:lit(true),kind:lit(input.kind),input:isFdtInput,inputKey:lit(fdtInputKey(input)),result:fdtRecord})||fdtInputKey(v.input as FdtInput)!==fdtInputKey(input))return false;
  const p=input.parameters,r=v.result as Record<string,unknown>,cache=new WeakMap<object,boolean>();
  const q:Check=x=>{if(!fdtRecord(x))return false;if(cache.has(x))return cache.get(x)!;const good=isMasterObservation(x);cache.set(x,good);return good;},exact:Check=x=>isMasterObservation(x,true);
  const vec=(c:Check=q)=>array(2,c),matrix=(c:Check=q)=>array(2,vec(c)),nullable:Check=x=>x===null||q(x);
  const row:Check=x=>{
    if(!shape(x,{step:x=>integer(x,0,p.sampleSteps),...fields('time angleRadians logRelaxationEnvelope',exact),...fields('sine cosine relaxationEnvelope',q),
      ...fields('unperturbedCovXtX0 covarianceLagDerivative stationaryCurrentCorrelation impulseResponse naiveEquilibriumResponse naiveResponseError currentResponseCorrection correctedResponse scoreResponse stepSusceptibility naiveEquilibriumStepSusceptibility',matrix()),responseIdentityResidual:matrix(exact),
      ...fields('stepMean impulseMean naiveStepMean naiveImpulseMean',vec()),logAbsoluteImpulseResponse:matrix(nullable),effectiveThermalEnergy:matrix(nullable),
      absoluteResponseLogStatus:matrix(x=>['structural-zero','unresolved-trigonometric-factor','available'].includes(x as string)),effectiveThermalEnergyStatus:matrix(x=>['undefined-zero-response','unresolved-response-denominator','available'].includes(x as string)),tZeroIsRightLimit:x=>typeof x==='boolean'}))return false;
    const zeroLag=x.step===0||p.timeNumerator===0,zeroAngle=zeroLag||p.angularNumerator===0;
    if(x.tZeroIsRightLimit!==zeroLag)return false;
    const logs=x.logAbsoluteImpulseResponse as unknown[][],ls=x.absoluteResponseLogStatus as string[][],diagnostic=x.effectiveThermalEnergy as unknown[][],ds=x.effectiveThermalEnergyStatus as string[][];
    return [0,1].every(i=>[0,1].every(j=>{
      const off=i!==j;
      return (ls[i][j]==='available'?logs[i][j]!==null:logs[i][j]===null)
        &&(ds[i][j]==='available'?diagnostic[i][j]!==null:diagnostic[i][j]===null)
        &&(off&&zeroAngle?ls[i][j]==='structural-zero'&&ds[i][j]==='undefined-zero-response':ls[i][j]!=='structural-zero'&&ds[i][j]!=='undefined-zero-response');
    }));
  };
  const good=shape(r,{parameters:x=>same(x,snapshotFdtInput(input).parameters),precisionBits:lit(256),...fields('mobility stiffness thermalEnergy relaxationRate diffusion stationaryCoordinateVariance angularRate',exact),force:vec(exact),impulse:vec(exact),initialImpulseMean:vec(exact),staticSusceptibility:matrix(exact),staticMean:vec(exact),stationaryCovarianceResidual:matrix(exact),history:array(p.sampleSteps+1,row),current:row,
    classification:x=>shape(x,{stationaryInitialization:lit(true),thermalEinsteinRelationSpecified:lit(true),coordinatesAreEvenPositions:lit(true),backgroundDetailedBalance:lit(p.angularNumerator===0),backgroundDriven:lit(p.angularNumerator!==0),forceResponseIsCausal:lit(true),rightLimitAtZero:lit(true),meanResponseIsExactlyLinearInThisFamily:lit(true),sampledPaths:lit(false),effectiveThermalEnergyIsThermometer:lit(false)}),approximationContract:x=>typeof x==='string'&&x.length>0&&x.length<12000});
  return good&&(r.history as {step:number}[]).every((x,i)=>x.step===i)&&same(r.current,(r.history as unknown[]).at(-1));
}
