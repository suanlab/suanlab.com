import type{FoundationResult}from './learning-foundations';
import {openTransmissionLineState,dielectricBilayerState,dielectricBilayerField,type OpenLineExcitation}from './learning-electromagnetic-effective-models';
export const electromagneticEffectiveKinds=['transmission-line-lumped-limit-energy','layered-medium-bloch-effective-limit']as const;
export type ElectromagneticEffectiveKind=typeof electromagneticEffectiveKinds[number];
function validate(v:number[],first:boolean){const bounds=first?[[1e-9,4],[1e-12,4],[.01,4],[-4,4],[0,8],[0,1],[0,2*Math.PI],[0,1],[0,2]]:[[.25,16],[.25,16],[0,1],[.01,4],[0,8],[0,16],[0,1]];if(v.length!==bounds.length)throw new RangeError('Wrong effective-model control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||(first?(i===7||i===8):i===5)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid effective-model control ${i}`);}
function lineLab(v:number[]){
 const system={inductancePerLength:v[0],capacitancePerLength:v[1],length:v[2],inputVoltage:v[3]},excitation:OpenLineExcitation=v[7]===1?{kind:'ideal-quarter-wave-pole',order:v[8]}:{kind:'finite-phase',theta:v[4]},fraction=v[5],clock=v[6],query=openTransmissionLineState(system,excitation,fraction,clock);
 const spatial=Array.from({length:129},(_,i)=>{const u=i/128;return{u,value:openTransmissionLineState(system,excitation,u,clock)};}),temporal=Array.from({length:129},(_,i)=>{const phase=2*Math.PI*i/128;return{phase,value:openTransmissionLineState(system,excitation,fraction,phase)};}),spectrum=Array.from({length:321},(_,i)=>{const theta=i/40;return{theta,branch:Math.floor((theta+Math.PI/2)/Math.PI),value:openTransmissionLineState(system,{kind:'finite-phase',theta},fraction,clock)};}),small=Array.from({length:77},(_,i)=>{const exponent=-20+i/4,theta=10**exponent;return{exponent,value:openTransmissionLineState(system,{kind:'finite-phase',theta},fraction,clock)};});
 return{system,excitation,fraction,clock,query,spatial,temporal,spectrum,small};
}
function layerLab(v:number[]){
 const system={epsilon1:v[0],epsilon2:v[1],fraction:v[2],period:v[3]},frequency=v[4],periods=v[5],fraction=v[6],query=dielectricBilayerState(system,frequency,periods),position=periods*fraction,field=dielectricBilayerField(system,frequency,periods,position);
 const positions=[...Array.from({length:129},(_,i)=>periods*i/128),...Array.from({length:periods},(_,i)=>[i,i+system.fraction,i+1]).flat()],spatial=[...new Set(positions)].sort((a,b)=>a-b).map(x=>({x,value:dielectricBilayerField(system,frequency,periods,x)}));
 const spectrum=Array.from({length:241},(_,i)=>{const omega=i/30;return{omega,value:dielectricBilayerState(system,omega,periods)};}),counts=Array.from({length:17},(_,count)=>({count,value:dielectricBilayerState(system,frequency,count)})),longWave=Array.from({length:81},(_,i)=>{const omega=i/400;return{omega,value:dielectricBilayerState(system,omega,periods)};});
 const geometryStatus=periods>0&&system.fraction>0&&system.fraction<1&&Array.from({length:periods},(_,i)=>i+system.fraction===i||i+system.fraction===i+1).some(Boolean)?'thin-layer-below-coordinate-resolution' as const:'finite' as const;
 return{system,frequency,periods,fraction,query,position,field,spatial,spectrum,counts,longWave,geometryStatus};
}
export interface ElectromagneticEffectiveResult extends FoundationResult{kind:ElectromagneticEffectiveKind;electromagnetism:true;transmissionLine?:ReturnType<typeof lineLab>;layeredMedium?:ReturnType<typeof layerLab>;}
export function evaluateElectromagneticEffective(kind:ElectromagneticEffectiveKind,values:number[]):ElectromagneticEffectiveResult{
 if(!(electromagneticEffectiveKinds as readonly string[]).includes(kind))throw new RangeError('Unknown effective-model lesson');const first=kind===electromagneticEffectiveKinds[0];validate(values,first);
 if(first){const r=lineLab(values),q=r.query;return{kind,electromagnetism:true,transmissionLine:r,curves:[],stats:{'relative admittance correction':q.relativeAdmittanceCorrection,'cycle energy · J':q.cycleEnergy,'stable energy minus nondispersive fit · J':q.stableEnergyFitDifference,'phase amplification':q.phaseAmplification}};}
 const r=layerLab(values),q=r.query;return{kind,electromagnetism:true,layeredMedium:r,curves:[],stats:{'unit-cell half trace':q.halfTrace,'Bloch attenuation per period':q.attenuationPerPeriod,'finite-stack transmittance':q.transmittance,'principal Bloch phase · rad':q.principalBlochPhase}};
}
