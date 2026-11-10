import type {FoundationResult}from './learning-foundations';
import type {StaticVector}from './learning-electro-magnetostatics';
import {timeGaugeState,timeGaugeLoop,lorenzHarmonicState,lorenzInitialConstraintState,constantLorenzOffsetState,type GaugeTransformMode,type TimeGauge}from './learning-gauge-freedom';
export const gaugeFreedomKinds=['time-dependent-gauge-field-invariance','lorenz-gauge-source-initial-compatibility']as const;
export type GaugeFreedomKind=typeof gaugeFreedomKinds[number];
function validate(v:number[],first:boolean){const bounds=first?Array.from({length:16},(_,i)=>i===11?[.25,4]:i===12||i===13||i===14?[0,4]:i===15?[0,3]:[-4,4]):[[-4,4],[-4,4],[0,16],[0,16],[-4,4],[-4,4],[.25,4],[0,4],[0,2]];if(v.length!==bounds.length)throw new RangeError('Wrong gauge control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||i===(first?15:8)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid gauge control ${i}`);}
const scaled=(x:number|null,s:number|null)=>x===0||s===0?0:x===null||s===null||!Number.isFinite(x*s)||x*s===0?null:x*s;
function transform(v:number[]){
 const system:TimeGauge={magneticConstant:v[0],magneticRate:v[1],magneticAcceleration:v[2],gaugeAmplitude:v[3],gaugeConstant:v[4],gaugeRate:v[5],gaugeAcceleration:v[6],length:v[11],potential:v[12],mode:(['coordinated','vector-only','scalar-only','original']as GaugeTransformMode[])[v[15]]},position:[number,number,number]=[v[7],v[8],v[9]],tau=v[10],query=timeGaugeState(system,position,tau),loop=timeGaugeLoop(system,tau,v[9],v[13],v[14]);
 const spatial=Array.from({length:129},(_,i)=>{const x=-4+i/16;return{x,value:timeGaugeState(system,[x,v[8],v[9]],tau)};});
 const temporal=Array.from({length:81},(_,i)=>{const t=-4+i/10;return{tau:t,value:timeGaugeState(system,position,t),loop:timeGaugeLoop(system,t,v[9],v[13],v[14])};});
 return{system,position,tau,width:v[13],height:v[14],query,loop,spatial,temporal};
}
function constraint(v:number[]){
 const family=v[8],system={scalarAmplitude:v[0],vectorAmplitude:v[1],wavenumber:v[2],frequency:v[3],length:v[6],potential:v[7]},z=v[4],tau=v[5];
 const observe=(z:number,tau:number)=>{
  const h=family===0?lorenzHarmonicState(system,z,tau):null,i=family===1?lorenzInitialConstraintState({...system,amplitude:v[1]},z,tau):null,c=family===2?constantLorenzOffsetState(system,v[0],tau):null;
  const raw=h??i!,scales=(raw??c!).scales;
  const normalized=raw?{scalarPotential:raw.normalized.scalarPotential,vectorPotential:raw.normalized.vectorPotential,electric:raw.normalized.electric,magnetic:raw.normalized.magnetic,lorenz:raw.normalized.lorenz,assignedCharge:raw.normalized.assignedCharge,assignedCurrent:raw.normalized.assignedCurrent,fieldCharge:raw.normalized.fieldCharge,fieldCurrent:raw.normalized.fieldCurrent,gaussResidual:raw.normalized.gaussResidual,ampereResidual:raw.normalized.ampereResidual,sourceContinuity:raw.normalized.sourceContinuity,initialLorenz:raw.normalized.initialLorenz,initialLorenzTimeDerivative:raw.normalized.initialLorenzTimeDerivative,scalarWaveEquationResidual:raw.normalized.scalarWaveEquationResidual,vectorWaveEquationResidual:raw.normalized.vectorWaveEquationResidual}:{scalarPotential:c!.normalized.scalarPotential,vectorPotential:[0,0,0]as StaticVector,electric:c!.normalized.electric,magnetic:c!.normalized.magnetic,lorenz:c!.normalized.lorenz,assignedCharge:0,assignedCurrent:0,fieldCharge:0,fieldCurrent:0,gaussResidual:0,ampereResidual:0,sourceContinuity:0,initialLorenz:v[0],initialLorenzTimeDerivative:0,scalarWaveEquationResidual:0,vectorWaveEquationResidual:0};
  const condition=raw??c!;return{z,tau,scales,normalized,coefficients:h?{Q:h.normalized.electricCoefficient,C:h.normalized.lorenzCoefficient,roundedQ:h.normalized.naiveElectricCoefficient,roundedC:h.normalized.naiveLorenzCoefficient,D:h.normalized.dispersion,roundedD:h.normalized.naiveDispersion}:null,
   si:{time:scaled(tau,scales.normalizedTimeSeconds),electricZ:scaled(normalized.electric?.[2]??null,scales.electric),lorenz:scaled(normalized.lorenz,scales.lorenz),assignedCharge:scaled(normalized.assignedCharge,scales.charge),assignedCurrent:scaled(normalized.assignedCurrent,scales.current),fieldCharge:scaled(normalized.fieldCharge,scales.charge),fieldCurrent:scaled(normalized.fieldCurrent,scales.current),gauss:scaled(normalized.gaussResidual,scales.gauss),ampere:scaled(normalized.ampereResidual,scales.ampere)},
   lorenzFamilySatisfied:condition.lorenzFamilySatisfied,sourceFamilyConserved:raw?raw.assignedSourceFamilyConserved:true,maxwellFamilyCompatible:condition.assignedSourceMaxwellFamilyCompatible};
 };
 const query=observe(z,tau),segments=Math.max(128,2*Math.ceil(8*Math.max(v[2],family===1?v[2]:v[3])*16/(4*Math.PI)));
 const spatial=Array.from({length:segments+1},(_,j)=>{const z=-4+8*j/segments;return{z,value:observe(z,tau)};});
 const temporal=Array.from({length:segments+1},(_,j)=>{const t=-4+8*j/segments;return{tau:t,value:observe(z,t)};});
 return{family,system,z,tau,query,spatial,temporal};
}
export interface GaugeFreedomResult extends FoundationResult{kind:GaugeFreedomKind;electromagnetism:true;gaugeTransform?:ReturnType<typeof transform>;lorenzConstraint?:ReturnType<typeof constraint>;}
export function evaluateGaugeFreedom(kind:GaugeFreedomKind,v:number[]):GaugeFreedomResult{
 if(!(gaugeFreedomKinds as readonly string[]).includes(kind))throw new RangeError('Unknown gauge lesson');const first=kind===gaugeFreedomKinds[0];validate(v,first);
 if(first){const r=transform(v),q=r.query;return{kind,electromagnetism:true,gaugeTransform:r,curves:[],stats:{'actual E x · V/m':q.si.electric?.[0]??null,'normalized Lorenz G':q.normalized.lorenz,'physical field family unchanged · 0/1':Number(q.physicalFieldFamilyUnchanged),'loop magnetic flux · Wb':r.loop.magneticFlux}};}
 const r=constraint(v),q=r.query;return{kind,electromagnetism:true,lorenzConstraint:r,curves:[],stats:{'actual E z · V/m':q.si.electricZ,'normalized Lorenz G':q.normalized.lorenz,'normalized Gauss residual':q.normalized.gaussResidual,'assigned-source Maxwell family compatible · 0/1':q.maxwellFamilyCompatible===null?null:Number(q.maxwellFamilyCompatible)}};
}
