import type {FoundationResult} from './learning-foundations';
import {electrostaticSphere,magnetostaticCylinder,type StaticVector} from './learning-electro-magnetostatics';
export const electroMagnetostaticsKinds=['gauss-symmetry-electrostatic-sphere','ampere-symmetry-steady-current-cylinder'] as const;
export type ElectroMagnetostaticsKind=typeof electroMagnetostaticsKinds[number];
const bounds=(electric:boolean)=>[[.25,4],[-4,4],[-4,4],[-4,4],[-4,4],...Array.from({length:3},()=>electric?[-20,20]:[-2,2])];
function validate(v:number[],electric:boolean){if(v.length!==8)throw new RangeError('Expected eight static-field controls');v.forEach((x,i)=>{const [lo,hi]=bounds(electric)[i];if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`Invalid static-field control ${i}`);});}
const scale=(v:number|null,f:number)=>v===null?null:v*f;
function makeLab(v:number[],electric:boolean){
 const source=v[1]*(electric?1e-9:1),background=v.slice(5).map(x=>x*(electric?1:1e-6)) as StaticVector,observation=v.slice(2,5) as StaticVector;
 const a=v[0],at=(p:StaticVector)=>electric?electrostaticSphere({radius:a,totalCharge:source,backgroundField:background},p):magnetostaticCylinder({radius:a,totalCurrent:source,backgroundField:background},p);
 const query=at(observation),extent=Math.max(1,2*a,Math.abs(observation[0]),Math.abs(observation[1]));
 const grid=Array.from({length:81},(_,i)=>{const x=extent*(-1+(i%9)/4),y=extent*(-1+Math.floor(i/9)/4);return{x,y,value:at([x,y,observation[2]])};});
 const maxRadius=Math.max(4*a,query.radialDistance),radii=[...new Set([...Array.from({length:241},(_,i)=>maxRadius*i/240),a,query.radialDistance])].sort((a,b)=>a-b);
 const radial=radii.map(radius=>({radius,value:at([radius,0,0])}));
 const angular=Array.from({length:129},(_,i)=>{const angle=2*Math.PI*i/128,co=Math.cos(angle),si=Math.sin(angle),r=query.radialDistance;
  // Electric: an equatorial section of a 3D closed measuring sphere.
  // Magnetic: a complete oriented circle at the observation height.
  const value=at([r*co,r*si,electric?0:observation[2]]),unit:StaticVector=electric?[co,si,0]:[-si,co,0];
  return{angle,source:value.sourceField===null?null:value.sourceField.reduce((sum,x,j)=>sum+x*unit[j],0),background:background.reduce((sum,x,j)=>sum+x*unit[j],0),total:value.totalField===null?null:value.totalField.reduce((sum,x,j)=>sum+x*unit[j],0)};
 });
 return{query,grid,radial,angular,observation,extent,electric};
}
export type StaticFieldLab=ReturnType<typeof makeLab>;
export interface ElectroMagnetostaticsResult extends FoundationResult{kind:ElectroMagnetostaticsKind;electromagnetism:true;unavailable?:'input-representation';statics?:StaticFieldLab;}
export function evaluateElectroMagnetostatics(kind:ElectroMagnetostaticsKind,v:number[]):ElectroMagnetostaticsResult{
 if(!(electroMagnetostaticsKinds as readonly string[]).includes(kind))throw new RangeError('Unknown static-field lesson');
 const electric=kind===electroMagnetostaticsKinds[0];validate(v,electric);
 const conversions=[{value:v[1],factor:electric?1e-9:1},...v.slice(5).map(value=>({value,factor:electric?1:1e-6}))];
 if(conversions.some(x=>x.value!==0&&x.value*x.factor===0))return{kind,electromagnetism:true,unavailable:'input-representation',curves:[],stats:{'SI observation':null}};
 const r=makeLab(v,electric),q=r.query,f=electric?1:1e6;
 const unit=electric?'E · N/C':'B · µT';
 const common=Object.fromEntries(['x','y','z'].flatMap((axis,j)=>[[`source ${axis} ${unit}`,scale(q.sourceField?.[j]??null,f)],[`total ${axis} ${unit}`,scale(q.totalField?.[j]??null,f)]]));
 return{kind,electromagnetism:true,statics:r,curves:[],stats:electric?{...common,'r · m':q.radialDistance,'V source · V':'sourcePotential'in q?q.sourcePotential:null,'flux · N m²/C':'closedElectricFlux'in q?q.closedElectricFlux:null}:{...common,'r · m':q.radialDistance,'circulation · µT m':'circulation'in q?scale(q.circulation,1e6):null}};
}
