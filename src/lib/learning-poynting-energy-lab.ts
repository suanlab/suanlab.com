import type {FoundationResult} from './learning-foundations';
import {inductionEnergyState as source,inductionCylinderEnergy as cylinder,transversePulseEnergy as pulse,transversePulseSlab as slab,energyOnlyNegativeControl} from './learning-poynting-energy';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from './learning-maxwell';
export const poyntingEnergyKinds=['poynting-source-work-local-balance','transverse-pulse-energy-flux-balance'] as const;
export type PoyntingEnergyKind=typeof poyntingEnergyKinds[number];
const scale=(x:number|null,f:number)=>x===null||!Number.isFinite(x*f)||x!==0&&x*f===0?null:x*f;
function validate(v:number[],sourceKind:boolean){
 const bounds=sourceKind?[[-4,4],[-4,4],[-4,4],[-2,2],[-4,4],[-4,4],[0,4],[0,4],[0,1],[.5,4],[.25,4]]:[[-4,4],[.1,4],[0,1],[-4,4],[-4,4],[0,1],[.5,4],[.25,4],[-2,2],[0,2]],selector=sourceKind?8:5;
 if(v.length!==bounds.length)throw new RangeError('Wrong energy control count');
 for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||i===selector&&!Number.isInteger(v[i]))throw new RangeError(`Invalid energy control ${i}`);
}
function sourceLab(v:number[]){
 const system={magneticOffset:v[0]*1e-6,magneticRate:v[1]*1e-6,magneticAcceleration:v[2]*1e-6,currentMode:v[8]===0?'required-current' as const:'omitted-current-negative-control' as const,permittivity:eps*v[9],permeability:mu*v[10]},time=v[3],position:[number,number,number]=[v[4],v[5],0],radius=v[6],height=v[7];
 const query=source(system,position,time),volume=cylinder(system,radius,height,time);
 const temporal=Array.from({length:81},(_,i)=>{const t=-2+4*i/80;return{time:t,query:source(system,position,t),volume:cylinder(system,radius,height,t)};});
 const radial=Array.from({length:81},(_,i)=>{const r=4*i/80;return{radius:r,query:source(system,[r,0,0],time),volume:cylinder(system,r,height,time)};});
 return{system,time,position,radius,height,query,volume,temporal,radial,energyOnlyControl:energyOnlyNegativeControl(v[4])};
}
function pulseLab(v:number[]){
 const system={amplitude:v[0],width:v[1],rightFraction:v[2],magneticMode:v[5]===0?'compatible' as const:'zero-magnetic-negative-control' as const,permittivity:eps*v[6],permeability:mu*v[7]},time=v[3]*1e-9,z=v[4],center=v[8],halfWidth=v[9];
 const start=center-halfWidth,end=center+halfWidth,geometryAvailable=halfWidth===0||start!==end;
 const query=pulse(system,z,time),window=geometryAvailable?slab(system,start,end,time):null;
 const spatial=Array.from({length:129},(_,i)=>{const x=-4+8*i/128;return{z:x,value:pulse(system,x,time),initial:pulse(system,x,0)};});
 const temporal=Array.from({length:81},(_,i)=>{const ns=-4+8*i/80;return{ns,value:pulse(system,z,ns*1e-9)};});
 return{system,time,z,center,halfWidth,start,end,geometryStatus:geometryAvailable?'finite' as const:'endpoint-representation' as const,window,query,spatial,temporal};
}
export interface PoyntingEnergyResult extends FoundationResult{kind:PoyntingEnergyKind;electromagnetism:true;unavailable?:'input-representation';sourceEnergy?:ReturnType<typeof sourceLab>;pulseEnergy?:ReturnType<typeof pulseLab>;}
export function evaluatePoyntingEnergy(kind:PoyntingEnergyKind,v:number[]):PoyntingEnergyResult{
 if(!(poyntingEnergyKinds as readonly string[]).includes(kind))throw new RangeError('Unknown energy lesson');const sourceKind=kind===poyntingEnergyKinds[0];validate(v,sourceKind);
 if((sourceKind?[0,1,2]:[3]).some(i=>v[i]!==0&&v[i]*(sourceKind?1e-6:1e-9)===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(sourceKind){const r=sourceLab(v),q=r.query;return{kind,electromagnetism:true,sourceEnergy:r,curves:[],stats:{'magnetic storage · W/m³':q.magneticStorageRate,'electric storage · W/m³':q.electricStorageRate,'grouped residual · W/m³':q.groupedResidual,'boundary outward · W':r.volume.outwardPower}};}
 const r=pulseLab(v),q=r.query;return{kind,electromagnetism:true,pulseEnergy:r,curves:[],stats:{'actual energy · pJ/m³':scale(q.energyDensity,1e12),'actual flow · mW/m²':scale(q.poyntingZ,1e3),'actual slab · pJ/m²':scale(r.window?.energyPerArea??null,1e12),'analytic compatible ratio · m/s':q.analyticCompatibleVelocity}};
}
