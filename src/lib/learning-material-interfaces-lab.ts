import type {FoundationResult} from './learning-foundations';
import {dielectricInterfaceState as field,dielectricInterfaceMeasures as measures,normalInterfaceState as wave} from './learning-material-interfaces';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from './learning-maxwell';
export const materialInterfacesKinds=['dielectric-interface-free-bound-charge','normal-incidence-impedance-interface'] as const;
export type MaterialInterfacesKind=typeof materialInterfacesKinds[number];
const scale=(x:number|null,f:number)=>x===null||!Number.isFinite(x*f)||x!==0&&x*f===0?null:x*f;
function product(...v:(number|null)[]):number|null{if(v.includes(0))return 0;if(v.includes(null))return null;let q=1;for(const x of v){q*=x!;if(!Number.isFinite(q)||q===0)return null;}return q;}
function validate(v:number[],staticKind:boolean){
 const bounds=staticKind?[[-40,40],[-40,40],[-4,4],[-4,4],[.5,4],[.5,4],[-4,4],[-4,4],[0,4],[0,4],[0,4],[0,1]]:[[-4,4],[0,16],[.5,4],[.5,4],[.5,4],[.5,4],[-4,4],[-4,4],[0,1],[0,1]];
 if(v.length!==bounds.length)throw new RangeError('Wrong material-interface control count');
 for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||(staticKind?i===11:i===8||i===9)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid interface control ${i}`);
}
function dielectricLab(v:number[]){
 const system={normalDisplacement1:v[0]*1e-9,freeSurfaceCharge:v[1]*1e-9,tangentialElectric:v[2],tangentialJump:v[3],permittivity1:eps*v[4],permittivity2:eps*v[5],normalMode:v[11]===0?'specified-charge' as const:'equal-normal-field-negative-control' as const},x=v[6],z=v[7],span=v[8],depth=v[9],halfHeight=v[10];
 const query=field(system,[x,0,z]),box=measures(system,span,depth,halfHeight);
 const spatial=Array.from({length:129},(_,i)=>{const z=-4+8*i/128;return{z,value:field(system,[x,0,z])};});
 const tangential=Array.from({length:81},(_,i)=>{const x=-4+8*i/80;return{x,value:field(system,[x,0,0])};});
 return{system,x,z,span,depth,halfHeight,query,box,spatial,tangential};
}
function waveLab(v:number[]){
 const normalized=v[9]===1,system={amplitude:v[0],wavenumber1:v[1],permittivity1:(normalized?1:eps)*v[2],permittivity2:(normalized?1:eps)*v[3],permeability1:(normalized?1:mu)*v[4],permeability2:(normalized?1:mu)*v[5],reflectionMode:v[8]===0?'matched-boundary' as const:'forced-zero-reflection' as const},time=v[6]*1e-9,z=v[7],query=wave(system,z,time);
 const spatialSegments=Math.max(128,2*Math.ceil(8*Math.max(system.wavenumber1,query.wavenumber2??0)*16/(4*Math.PI)));
 const temporalSegments=Math.max(80,2*Math.ceil(8e-9*(query.omega??0)*16/(4*Math.PI)));
 const spatial=Array.from({length:spatialSegments+1},(_,i)=>{
  const z=-4+8*i/spatialSegments,value=wave(system,z,time),space=product(system.wavenumber1,z),temporal=product(value.omega,time);
  const forward=space===null||temporal===null?null:space-temporal,backward=space===null||temporal===null?null:space+temporal;
  const incidentElectric=product(system.amplitude,forward===null?null:Math.cos(forward)),reflectedElectric=product(system.amplitude,value.reflection,backward===null?null:Math.cos(backward));
  return{z,value,incidentElectric,reflectedElectric,incidentH:scale(incidentElectric,1/value.impedance1),reflectedH:scale(reflectedElectric,-1/value.impedance1)};
 });
 const temporal=Array.from({length:temporalSegments+1},(_,i)=>{const ns=-4+8*i/temporalSegments;return{ns,value:wave(system,z,ns*1e-9)};});
 return{system,time,z,normalizedCoefficients:normalized,query,spatial,temporal};
}
export interface MaterialInterfacesResult extends FoundationResult{kind:MaterialInterfacesKind;electromagnetism:true;unavailable?:'input-representation';dielectricBoundary?:ReturnType<typeof dielectricLab>;normalInterface?:ReturnType<typeof waveLab>;}
export function evaluateMaterialInterfaces(kind:MaterialInterfacesKind,v:number[]):MaterialInterfacesResult{
 if(!(materialInterfacesKinds as readonly string[]).includes(kind))throw new RangeError('Unknown material-interface lesson');const staticKind=kind===materialInterfacesKinds[0];validate(v,staticKind);
 if((staticKind?[0,1]:[6]).some(i=>v[i]!==0&&v[i]*1e-9===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 if(staticKind){const r=dielectricLab(v),q=r.query;return{kind,electromagnetism:true,dielectricBoundary:r,curves:[],stats:{'normal E jump · V/m':q.stableNormalEJump,'bound surface charge · nC/m²':scale(q.boundSurfaceCharge,1e9),'loop circulation · V':r.box.electricCirculation,'analytic normal D jump · nC/m²':scale(q.requiredFreeSurfaceCharge,1e9)}};}
 const r=waveLab(v),q=r.query;return{kind,electromagnetism:true,normalInterface:r,curves:[],stats:{'actual reflection r':q.reflection,'actual transmittance T':q.transmittance,'required sheet work · W/m²':q.boundary.requiredSheetWork,'compatible reflection r':q.referenceReflection}};
}
