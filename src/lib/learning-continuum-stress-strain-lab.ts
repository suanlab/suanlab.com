import {planeStressAt,planeStressTraction,rectangleStressBalance,affineDeformationState,affineDeformedPoint,infinitesimalRotationControl,type AffineDeformation} from './learning-continuum-stress-strain';
export const continuumStressStrainKinds=['cauchy-traction-stress-balance','finite-deformation-small-strain-rotation']as const;
export type ContinuumStressStrainKind=typeof continuumStressStrainKinds[number];
export const isContinuumStressStrainKind=(id:string):id is ContinuumStressStrainKind=>(continuumStressStrainKinds as readonly string[]).includes(id);
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length)throw new RangeError('Wrong stress/strain control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid stress/strain control ${i}`);}
export function tractionStressLab(v:number[]){
  validate(v,[[-8,8],[-8,8],[-8,8],[-4,4],[-4,4],[-4,4],[-2,2],[-2,2],[-Math.PI,Math.PI],[0,4],[-1,1],[-1,1],[0,1],[0,1],[.25,2],[-8,8],[-8,8]]);
  const stress={normalX:v[0],normalY:v[1],shear:v[2],gradientX:v[3],gradientY:v[4],crossGradient:v[5]},x=v[6],y=v[7],angle=v[8],patchArea=v[9];
  const rectangle={centerX:v[10],centerY:v[11],halfWidth:v[12],halfHeight:v[13],depth:v[14],bodyForce:[v[15],v[16]]as[number,number]};
  const query=planeStressTraction(stress,x,y,angle,patchArea),balance=rectangleStressBalance(stress,rectangle);
  const orientations=Array.from({length:181},(_,i)=>{const angle=-Math.PI+2*Math.PI*i/180;return{angle,value:planeStressTraction(stress,x,y,angle,patchArea)};});
  const spatial=Array.from({length:121},(_,i)=>{const x=-2+i/30;return{x,stress:planeStressAt(stress,x,y)};});
  return{stress,x,y,angle,patchArea,rectangle,query,balance,orientations,spatial};
}
export function finiteStrainLab(v:number[]){
  validate(v,[[.5,2],[.5,2],[-1,1],[-Math.PI,Math.PI],[-1,1],[-1,1],[-Math.PI,Math.PI],[-1,1],[-Math.PI,Math.PI]]);
  const model:AffineDeformation={stretchX:v[0],stretchY:v[1],shear:v[2],rotation:v[3],translationX:v[4],translationY:v[5]},direction=v[6],omega=v[7],extraRotation=v[8];
  const query=affineDeformationState(model,direction),rotated=affineDeformationState({...model,rotation:model.rotation+extraRotation},direction),control=infinitesimalRotationControl(omega);
  const grid=Array.from({length:6},(_,i)=>-1+2*i/5).flatMap(a=>[
    {reference:[[-1,a],[1,a]],current:[affineDeformedPoint(model,-1,a),affineDeformedPoint(model,1,a)],rotated:[affineDeformedPoint({...model,rotation:model.rotation+extraRotation},-1,a),affineDeformedPoint({...model,rotation:model.rotation+extraRotation},1,a)]},
    {reference:[[a,-1],[a,1]],current:[affineDeformedPoint(model,a,-1),affineDeformedPoint(model,a,1)],rotated:[affineDeformedPoint({...model,rotation:model.rotation+extraRotation},a,-1),affineDeformedPoint({...model,rotation:model.rotation+extraRotation},a,1)]}]);
  const directions=Array.from({length:181},(_,i)=>{const angle=-Math.PI+2*Math.PI*i/180;return{angle,value:affineDeformationState(model,angle)};});
  const rotations=Array.from({length:121},(_,i)=>{const angle=-Math.PI+2*Math.PI*i/120;return{angle,value:affineDeformationState({...model,rotation:angle},direction)};});
  const shears=Array.from({length:81},(_,i)=>{const shear=-1+i/40;return{shear,value:affineDeformationState({...model,shear},direction)};});
  const tinyRotations=Array.from({length:121},(_,i)=>{const logAngle=-30+i/4,angle=10**logAngle;return{logAngle,angle,value:affineDeformationState({...model,stretchX:1,stretchY:1,shear:0,rotation:angle}),roundedCorrection:Math.cos(angle)-1};});
  const controls=Array.from({length:81},(_,i)=>{const omega=-1+i/40;return{omega,value:infinitesimalRotationControl(omega)};});
  return{model,direction,omega,extraRotation,query,rotated,control,grid,directions,rotations,shears,tinyRotations,controls};
}
