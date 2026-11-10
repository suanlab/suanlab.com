import {binaryProduct,binarySum,binaryMultiply,binaryObservation,binaryRatioObservation} from './learning-binary-products';
/** Plane-stress and affine-kinematic teaching families, not a constitutive or FEM solver. */
export type MechanicalValue=number|null;
export type MechanicalVector=[MechanicalValue,MechanicalValue];
export type MechanicalMatrix=[MechanicalVector,MechanicalVector];
export interface AffinePlaneStress {normalX:number;normalY:number;shear:number;gradientX:number;gradientY:number;crossGradient:number;}
export interface StressRectangle {centerX:number;centerY:number;halfWidth:number;halfHeight:number;depth:number;bodyForce:[number,number];}
export interface AffineDeformation {stretchX:number;stretchY:number;shear:number;rotation:number;translationX:number;translationY:number;}
function bounded(v:number,min:number,max:number,name:string){if(!Number.isFinite(v)||v<min||v>max)throw new RangeError(`Invalid ${name}`);}
const available=(x:number):MechanicalValue=>Number.isFinite(x)?x:null;
const add=(...xs:MechanicalValue[]):MechanicalValue=>xs.some(x=>x===null)?null:available((xs as number[]).reduce((a,b)=>a+b,0));
function mul(a:MechanicalValue,b:MechanicalValue):MechanicalValue{
  if(a===0||b===0)return 0;
  if(a===null||b===null)return null;
  const p=a*b;return p===0&&a!==0&&b!==0?null:available(p);
}
const negate=(a:MechanicalValue)=>a===null?null:-a;
const dot=(a:MechanicalVector,b:MechanicalVector)=>add(mul(a[0],b[0]),mul(a[1],b[1]));
const cross=(a:MechanicalVector,b:MechanicalVector)=>add(mul(a[0],b[1]),negate(mul(a[1],b[0])));
/** Eigenvalues of the supplied Cartesian components. Scale before halving;
 * recover the smaller root from the unprojected determinant to avoid cancellation.
 */
function principalStress(stress:MechanicalMatrix){
  const a=stress[0][0],b=stress[0][1],d=stress[1][1];
  if(a===null||b===null||d===null)return {values:[null,null]as MechanicalVector,direction:null,status:'arithmetic-unavailable' as const};
  const isotropic=a===d&&b===0;
  const direction=isotropic?null:Math.atan2(2*b,a-d)/2;
  if(b===0)return {values:[Math.max(a,d),Math.min(a,d)]as MechanicalVector,direction,status:isotropic?'nonunique-isotropic' as const:'defined' as const};
  const scale=Math.max(Math.abs(a),Math.abs(b),Math.abs(d));
  const mean=(a/scale+d/scale)/2,radius=Math.hypot((a/scale-d/scale)/2,b/scale);
  const farRaw=binaryProduct(scale,mean+(mean<0?-radius:radius));
  const far=binaryObservation(farRaw).value!;
  const determinant=binarySum([binaryProduct(a,d),binaryProduct(-1,b,b)]);
  const near=binaryRatioObservation(determinant,farRaw).value;
  return {values:(mean<0?[near,far]:[far,near])as MechanicalVector,direction,status:'defined' as const};
}
function stressValid(s:AffinePlaneStress){for(const k of ['normalX','normalY','shear']as const)bounded(s[k],-8,8,k);for(const k of ['gradientX','gradientY','crossGradient']as const)bounded(s[k],-4,4,k);}
export function planeStressAt(s:AffinePlaneStress,x:number,y:number):MechanicalMatrix{
  stressValid(s);bounded(x,-4,4,'x');bounded(y,-4,4,'y');
  const a=add(s.normalX,mul(s.gradientX,x)),b=add(s.shear,mul(s.crossGradient,y)),d=add(s.normalY,mul(s.gradientY,y));
  return[[a,b],[b,d]];
}
export function planeStressTraction(s:AffinePlaneStress,x:number,y:number,angle:number,area=1){
  bounded(angle,-2*Math.PI,2*Math.PI,'normal angle');bounded(area,0,4,'patch area');
  const stress=planeStressAt(s,x,y),normal:[number,number]=[Math.cos(angle),Math.sin(angle)],tangent:[number,number]=[-normal[1],normal[0]];
  const traction:MechanicalVector=[dot(stress[0],normal),dot(stress[1],normal)],normalStress=dot(traction,normal),signedShear=dot(traction,tangent);
  const shearVector:MechanicalVector=[mul(signedShear,tangent[0]),mul(signedShear,tangent[1])];
  const force:MechanicalVector=area===0?[0,0]:[mul(area,traction[0]),mul(area,traction[1])];
  const principal=principalStress(stress);
  return{x,y,angle,area,stress,normal,tangent,traction,normalStress,signedShear,shearVector,force,
    principalValues:principal.values,principalDirection:principal.direction,
    directionStatus:principal.status,
    patchStatus:area===0?'zero-area-no-face-observation' as const:'finite-reference-patch' as const};
}
/** Active rotation of a rounded Cartesian tensor; not an additional physical loading. */
export function rotatePlaneStress(matrix:MechanicalMatrix,angle:number):MechanicalMatrix{
  bounded(angle,-2*Math.PI,2*Math.PI,'rotation');
  if(!Array.isArray(matrix)||matrix.length!==2)throw new RangeError('Expected2x2 tensor');
  for(let i=0;i<2;i++){if(!Array.isArray(matrix[i])||matrix[i].length!==2)throw new RangeError('Expected tensor row');for(let j=0;j<2;j++){const x=matrix[i][j];if(x!==null&&!Number.isFinite(x))throw new RangeError('Nonfinite tensor');}}
  const c=Math.cos(angle),s=Math.sin(angle),Q=[[c,-s],[s,c]];
  const values=[0,1].map(i=>[0,1].map(j=>add(...[0,1].flatMap(a=>[0,1].map(b=>mul(mul(Q[i][a],matrix[a][b]),Q[j][b]))))));
  return values as MechanicalMatrix;
}
export function rectangleStressBalance(s:AffinePlaneStress,r:StressRectangle){
  stressValid(s);bounded(r.centerX,-1,1,'center x');bounded(r.centerY,-1,1,'center y');bounded(r.halfWidth,0,1,'half width');bounded(r.halfHeight,0,1,'half height');bounded(r.depth,.25,2,'depth');
  if(!Array.isArray(r.bodyForce)||r.bodyForce.length!==2)throw new RangeError('Expected body force vector');for(let i=0;i<2;i++)bounded(r.bodyForce[i],-8,8,'body force');
  const volume=mul(mul(mul(4,r.halfWidth),r.halfHeight),r.depth),divergence:[number,number]=[s.gradientX+s.crossGradient,s.gradientY];
  const volumeForce:MechanicalVector=[mul(volume,divergence[0]),mul(volume,divergence[1])],bodyForce:MechanicalVector=[mul(volume,r.bodyForce[0]),mul(volume,r.bodyForce[1])];
  const status=r.halfWidth===0||r.halfHeight===0?'zero-area':r.halfWidth<1e-8||r.halfHeight<1e-8?'below-supported-geometry':'regular';
  const empty={volume,divergence,volumeForce,bodyForce,geometryStatus:status,faces:[] as {name:string;area:number;force:MechanicalVector;moment:MechanicalValue;center:[number,number];normal:[number,number]}[],surfaceForce:null as MechanicalVector|null,surfaceMoment:null as MechanicalValue,forceResidual:null as MechanicalVector|null,totalForce:null as MechanicalVector|null,totalMoment:null as MechanicalValue,omittedBodyForceControl:null as MechanicalVector|null};
  if(status!=='regular')return empty;
  const definitions:[string,[number,number],[number,number],number,[number,number]][]=[
    ['left',[r.centerX-r.halfWidth,r.centerY],[-1,0],r.halfHeight,[0,1]],['right',[r.centerX+r.halfWidth,r.centerY],[1,0],r.halfHeight,[0,1]],
    ['bottom',[r.centerX,r.centerY-r.halfHeight],[0,-1],r.halfWidth,[1,0]],['top',[r.centerX,r.centerY+r.halfHeight],[0,1],r.halfWidth,[1,0]]];
  const faces=definitions.map(([name,center,normal,half,tangent])=>{
    const samples=[-1/Math.sqrt(3),1/Math.sqrt(3)].map(z=>{
      const x=center[0]+half*z*tangent[0],y=center[1]+half*z*tangent[1],stress=planeStressAt(s,x,y),weight=half*r.depth;
      const f:MechanicalVector=[mul(weight,dot(stress[0],normal)),mul(weight,dot(stress[1],normal))];
      return{force:f,moment:cross([x-r.centerX,y-r.centerY],f)};
    });
    return{name,center,normal,area:2*half*r.depth,force:[add(...samples.map(p=>p.force[0])),add(...samples.map(p=>p.force[1]))]as MechanicalVector,moment:add(...samples.map(p=>p.moment))};
  });
  const surfaceForce:MechanicalVector=[add(...faces.map(p=>p.force[0])),add(...faces.map(p=>p.force[1]))],surfaceMoment=add(...faces.map(p=>p.moment));
  const totalForce:MechanicalVector=[add(surfaceForce[0],bodyForce[0]),add(surfaceForce[1],bodyForce[1])];
  // Moments are about the rectangle centre; uniform body force has zero moment there.
  return{volume,divergence,volumeForce,bodyForce,geometryStatus:status,faces,surfaceForce,surfaceMoment,
    forceResidual:[add(surfaceForce[0],negate(volumeForce[0])),add(surfaceForce[1],negate(volumeForce[1]))]as MechanicalVector,
    totalForce,totalMoment:surfaceMoment,omittedBodyForceControl:surfaceForce};
}
function deformationValid(d:AffineDeformation){bounded(d.stretchX,.5,2,'stretch x');bounded(d.stretchY,.5,2,'stretch y');bounded(d.shear,-1,1,'shear');bounded(d.rotation,-2*Math.PI,2*Math.PI,'rotation');bounded(d.translationX,-1,1,'translation x');bounded(d.translationY,-1,1,'translation y');}
export function affineDeformedPoint(d:AffineDeformation,X:number,Y:number):[number,number]{
  deformationValid(d);bounded(X,-2,2,'reference x');bounded(Y,-2,2,'reference y');
  const c=Math.cos(d.rotation),s=Math.sin(d.rotation),a=d.stretchX*X+d.shear*Y,b=d.stretchY*Y;
  return[c*a-s*b+d.translationX,s*a+c*b+d.translationY];
}
export function affineDeformationState(d:AffineDeformation,direction=0){
  deformationValid(d);bounded(direction,-2*Math.PI,2*Math.PI,'reference direction');
  const {stretchX:a,stretchY:b,shear:g,rotation:theta}=d,c=Math.cos(theta),s=Math.sin(theta);
  const F:[[number,number],[number,number]]=[[c*a,c*g-s*b],[s*a,s*g+c*b]];
  const cosMinusOne=theta===0?0:mul(-2,mul(Math.sin(theta/2),Math.sin(theta/2)));
  // If theta/2 itself vanished, the nonzero second-order term is unavailable.
  const rotationCorrection=theta!==0&&cosMinusOne===0?null:cosMinusOne;
  const H:MechanicalMatrix=[[add(a-1,mul(a,rotationCorrection)),add(mul(c,g),-s*b)],[s*a,add(b-1,mul(b,rotationCorrection),mul(s,g))]];
  const shearStrain=mul(.5,add(mul(c,g),mul(s,a-b)));
  const infinitesimal:MechanicalMatrix=[[H[0][0],shearStrain],[shearStrain,H[1][1]]];
  // Retain the complete tensor polynomial before projecting individual components.
  const greenRaw=[binarySum([binaryProduct(.5,a,a),binaryProduct(-.5)]),binaryProduct(.5,a,g),
    binarySum([binaryProduct(.5,b,b),binaryProduct(.5,g,g),binaryProduct(-.5)])];
  const [green11,green12,green22]=greenRaw.map(v=>binaryObservation(v).value);
  const green:MechanicalMatrix=[[green11,green12],[green12,green22]];
  const roundedGreen:MechanicalMatrix=[[.5*(F[0][0]**2+F[1][0]**2-1),.5*(F[0][0]*F[0][1]+F[1][0]*F[1][1])],[.5*(F[0][0]*F[0][1]+F[1][0]*F[1][1]),.5*(F[0][1]**2+F[1][1]**2-1)]];
  const n:[number,number]=[Math.cos(direction),Math.sin(direction)],mapped=affineDeformedPoint({...d,translationX:0,translationY:0},n[0],n[1]);
  const directionalGreen=binaryObservation(binarySum([binaryMultiply(greenRaw[0],binaryProduct(n[0],n[0])),
    binaryMultiply(greenRaw[1],binaryProduct(2,n[0],n[1])),binaryMultiply(greenRaw[2],binaryProduct(n[1],n[1]))])).value;
  const roundedStretch=Math.hypot(...mapped),directionalLinear=dot(n,[dot(infinitesimal[0],n),dot(infinitesimal[1],n)]);
  const stableEngineeringStrain=directionalGreen===null?null:2*directionalGreen/(Math.sqrt(1+2*directionalGreen)+1);
  const referenceCorners:[number,number][]=[[-1,-1],[1,-1],[1,1],[-1,1]],corners=referenceCorners.map(p=>affineDeformedPoint(d,...p));
  const roundedPolygonArea=Math.abs(corners.reduce((sum,p,i)=>{const next=corners[(i+1)%4];return sum+p[0]*next[1]-p[1]*next[0];},0))/2;
  return{...d,direction,F,H,green,infinitesimal,roundedGreen,normal:n,mappedDirection:mapped,roundedStretch,directionalGreen,directionalLinear,stableEngineeringStrain,roundedEngineeringStrain:roundedStretch-1,
    determinant:a*b,roundedDeterminant:F[0][0]*F[1][1]-F[0][1]*F[1][0],referenceArea:4,area:4*a*b,roundedPolygonArea,referenceCorners,corners,
    rotationCorrection,arithmeticStatus:[...green.flat(),...infinitesimal.flat(),stableEngineeringStrain].some(x=>x===null)?'some-observations-unavailable' as const:'available' as const};
}
/** I+omega J is an affine stretch/rotation, not an exact finite rigid rotation. */
export function infinitesimalRotationControl(omega:number){
  bounded(omega,-1,1,'infinitesimal rotation parameter');const square=mul(omega,omega),greenScalar=mul(.5,square),roundedStretch=Math.hypot(1,omega);
  return{omega,F:[[1,-omega],[omega,1]],greenScalar,roundedStretch,stableEngineeringStrain:square===null?null:square/(roundedStretch+1)};
}
