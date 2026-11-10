import type {FoundationResult} from './learning-foundations';
import {maxwellPlaneWaveState as wave,maxwellPlaneWaveRectangle as rectangle,maxwellPlaneWaveBox as box,maxwellRadialState as radial,maxwellRadialSphere as sphere,maxwellRadialBalance as balance,teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from './learning-maxwell';
import type {StaticVector} from './learning-electro-magnetostatics';
export const maxwellKinds=['maxwell-plane-wave-local-integral-laws','maxwell-charge-displacement-continuity'] as const;
export type MaxwellKind=typeof maxwellKinds[number];
const scale=(x:number|null,f:number)=>x===null||!Number.isFinite(x*f)||x!==0&&x*f===0?null:x*f;
function validate(v:number[],plane:boolean){
 const bounds=plane?[[0,4],[0,16],[0,1],[-Math.PI,Math.PI],[.5,1.5],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[.5,4],[.25,4]]:[[-4,4],[-4,4],[-2,2],[0,4],[-4,4],[-4,4],[-4,4],[-4,4],[.5,4],[-2,2]];
 if(v.length!==bounds.length)throw new RangeError('Wrong Maxwell control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||plane&&i===2&&!Number.isInteger(v[i]))throw new RangeError(`Invalid Maxwell control ${i}`);
}
function planeLab(v:number[]){
 const system={amplitude:v[0],wavenumber:v[1],direction:(v[2]===1?1:-1) as 1|-1,phaseOffset:v[3],frequencyRatio:v[4],permittivity:eps*v[10],permeability:mu*v[11]},time=v[5]*1e-9,point:[number,number,number]=[0,0,v[6]];
 const query=wave(system,point,time),faraday=rectangle(system,'faraday-xz',v[7],v[8],v[9],time),ampere=rectangle(system,'ampere-yz',v[7],v[8],v[9],time),closed=box(system,Math.abs(v[7]),Math.abs(v[7]),Math.min(v[8],v[9]),Math.max(v[8],v[9]),time);
 const spatial=Array.from({length:257},(_,i)=>{const z=-4+8*i/256;return{z,value:wave(system,[0,0,z],time)};}),temporal=Array.from({length:257},(_,i)=>{const ns=-4+8*i/256;return{ns,value:wave(system,point,ns*1e-9)};});
 return{system,time,point,query,faraday,ampere,closed,spatial,temporal};
}
function radialLab(v:number[]){
 const system={initialGradient:v[0],gradientRate:v[1],chargeOffset:v[7]*1e-12,permittivity:eps*v[8]},point=v.slice(4,7) as StaticVector,time=v[2],radius=v[3],query=radial(system,point,time),surface=sphere(system,radius,time),interval=balance(system,radius,v[9],time);
 const temporal=Array.from({length:81},(_,i)=>{const t=-2+4*i/80;return{time:t,local:radial(system,point,t),surface:sphere(system,radius,t)};}),profile=Array.from({length:81},(_,i)=>{const x=-4+8*i/80;return{x,local:radial(system,[x,0,0],time)};});
 return{system,point,time,radius,query,surface,interval,temporal,profile};
}
export interface MaxwellResult extends FoundationResult{kind:MaxwellKind;electromagnetism:true;unavailable?:'input-representation';planeMaxwell?:ReturnType<typeof planeLab>;radialMaxwell?:ReturnType<typeof radialLab>;}
export function evaluateMaxwell(kind:MaxwellKind,v:number[]):MaxwellResult{
 if(!(maxwellKinds as readonly string[]).includes(kind))throw new RangeError('Unknown Maxwell lesson');const plane=kind===maxwellKinds[0];validate(v,plane);
 const conversion=plane?[[5,1e-9]]:[[7,1e-12]];if(conversion.some(([i,f])=>v[i]!==0&&v[i]*f===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(plane){const r=planeLab(v);return{kind,electromagnetism:true,planeMaxwell:r,curves:[],stats:{'E x · V/m':r.query.electricField?.[0]??null,'B y · nT':scale(r.query.magneticField?.[1]??null,1e9),'Faraday residual · V':r.faraday.residual,'Ampere residual · T m':r.ampere.residual}};}
 const r=radialLab(v);return{kind,electromagnetism:true,radialMaxwell:r,curves:[],stats:{'rho · pC/m³':scale(r.query.chargeDensity,1e12),'outward conduction · pA':scale(r.surface.outwardConductionCurrent,1e12),'Gauss residual · V/m²':r.query.gaussConstraintResidual,'Delta Q · pC':scale(r.interval.chargeChange,1e12)}};
}
