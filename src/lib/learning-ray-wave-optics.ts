/** Ideal scalar paraxial optics in one homogeneous medium.
 * Rays use (x,theta) in metres/radians; field uses exp(+ikz-iwt), q_waist=-i*zR.
 * Browser-safe pure calculations. No finite aperture, aberration or Maxwell solver.
 */
export type OpticsComplex = readonly [number, number];
export type RayMatrix = readonly [readonly [number, number], readonly [number, number]];
export interface ParaxialRayInput {
  objectDistance: number; screenDistance: number; focalLength: number | null;
  objectHeight: number; centralAngle: number; fanHalfAngle: number;
}
export interface GaussianBeamInput {
  wavelength: number; waistRadius: number; amplitude: number;
  preDistance: number; postDistance: number; focalLength: number | null;
}
const within = (name: string, x: number, lo: number, hi: number) => {
  if (!Number.isFinite(x) || x < lo || x > hi) throw new RangeError(`${name} must be finite in [${lo}, ${hi}]`);
  return x;
};
function focal(f: number | null) {
  if (f !== null && (!Number.isFinite(f) || Math.abs(f) < .005 || Math.abs(f) > 2)) throw new RangeError('focalLength must be null or have magnitude in [0.005, 2] m');
  return f;
}
export function multiplyRayMatrices(a: RayMatrix, b: RayMatrix): RayMatrix {
  return [[a[0][0]*b[0][0]+a[0][1]*b[1][0], a[0][0]*b[0][1]+a[0][1]*b[1][1]],
    [a[1][0]*b[0][0]+a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1]]];
}
export const freeRayMatrix = (distance: number): RayMatrix => {
  within('distance', distance, -Number.MAX_VALUE, Number.MAX_VALUE); return [[1,distance],[0,1]];
};
export const thinLensRayMatrix = (f: number | null): RayMatrix => {
  focal(f); return [[1,0],[f === null ? 0 : -1/f,1]];
};
export function rayTransferMatrix(before: number, after: number, f: number | null): RayMatrix {
  return multiplyRayMatrices(freeRayMatrix(after),multiplyRayMatrices(thinLensRayMatrix(f),freeRayMatrix(before)));
}
function rayApply(m: RayMatrix, x: number, theta: number): [number, number] {
  return [m[0][0]*x+m[0][1]*theta,m[1][0]*x+m[1][1]*theta];
}
const det = (m: RayMatrix) => m[0][0]*m[1][1]-m[0][1]*m[1][0];
export function paraxialRayExperiment(input: ParaxialRayInput) {
  const s = {...input};
  within('objectDistance',s.objectDistance,.005,2);within('screenDistance',s.screenDistance,0,2);
  within('objectHeight',s.objectHeight,-.005,.005);within('centralAngle',s.centralAngle,-.1,.1);within('fanHalfAngle',s.fanHalfAngle,0,.1);focal(s.focalLength);
  const u=s.objectDistance,z=s.screenDistance,f=s.focalLength;
  // Only exact focal equality is singular; nearby values retain their finite image.
  const imageDistance = f === null || u === f ? null : f*u/(u-f);
  const magnification = imageDistance === null ? null : -imageDistance/u;
  const imageHeight = magnification === null ? null : magnification*s.objectHeight;
  const imageKind = f === null ? 'no-lens' : imageDistance === null ? 'at-infinity' : imageDistance > 0 ? 'real' : 'virtual';
  const matrix=rayTransferMatrix(u,z,f),rays=Array.from({length:9},(_,i)=>{
    const theta=s.centralAngle+s.fanHalfAngle*(i/4-1),lensX=s.objectHeight+u*theta,outputTheta=theta-(f===null?0:lensX/f);
    return {inputAngle:theta,lensHeight:lensX,outputAngle:outputTheta,screenHeight:lensX+z*outputTheta,
      imageBackExtension:imageDistance!==null&&imageDistance<0 ? {coordinate:imageDistance,height:lensX+imageDistance*outputTheta} : null};
  });
  const heights=rays.map(r=>r.screenHeight),meanScreenHeight=heights.reduce((a,b)=>a+b,0)/heights.length;
  const rmsScreenSpread=Math.sqrt(heights.reduce((a,b)=>a+(b-meanScreenHeight)**2,0)/heights.length);
  const imageMatrix=imageDistance===null ? null : rayTransferMatrix(u,imageDistance,f);
  return {input:s,matrix,determinant:det(matrix),rays,imageDistance,imageHeight,magnification,imageKind,imageMatrix,
    imageDistanceDerivative: f===null||imageDistance===null ? null : -((f/(u-f))**2),
    screen:{meanHeight:meanScreenHeight,rmsSpread:rmsScreenSpread,span:Math.max(...heights)-Math.min(...heights)},
    chiefRay:rayApply(matrix,s.objectHeight,s.centralAngle),
    maxSlope:Math.max(...rays.flatMap(r=>[Math.abs(r.inputAngle),Math.abs(r.outputAngle)])),
    convention:'(x,theta), same refractive index; virtual images are backward extrapolations; no aperture or aberrations'};
}
function divide(a: OpticsComplex,b: OpticsComplex): OpticsComplex {
  const norm=b[0]*b[0]+b[1]*b[1];
  if (!(norm>0) || !Number.isFinite(norm)) throw new RangeError('Invalid complex denominator');
  return [(a[0]*b[0]+a[1]*b[1])/norm,(a[1]*b[0]-a[0]*b[1])/norm];
}
function cmul(a: OpticsComplex,b: OpticsComplex): OpticsComplex { return [a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]]; }
export function transformGaussianQ(q: OpticsComplex,m: RayMatrix): OpticsComplex {
  if (!q.every(Number.isFinite) || !(q[1]<0) || !m.flat().every(Number.isFinite)) throw new RangeError('q must have finite negative imaginary part and matrix must be finite');
  if (Math.abs(det(m)-1)>1e-9) throw new RangeError('Gaussian q map requires a same-index determinant-one matrix');
  const denominator: OpticsComplex=[m[1][0]*q[0]+m[1][1],m[1][0]*q[1]];
  const quotient=divide([m[0][0]*q[0]+m[0][1],m[0][0]*q[1]],denominator);
  // Im((Aq+B)/(Cq+D))=det(M)*Im(q)/|Cq+D|² avoids subtractive cancellation.
  const result: OpticsComplex=[quotient[0],det(m)*q[1]/(denominator[0]**2+denominator[1]**2)];
  if (!(result[1]<0)) throw new RangeError('Gaussian q lost its physical half-plane');
  return result;
}
function geometry(q: OpticsComplex,k: number) {
  const norm=q[0]**2+q[1]**2,invCurvature=q[0]/norm;
  return {q,width:Math.sqrt(2*norm/(-k*q[1])),inverseCurvature:invCurvature,
    curvature:invCurvature===0?null:1/invCurvature,
    coefficient:[-k*q[1]/(2*norm),-k*q[0]/(2*norm)] as OpticsComplex};
}
export function gaussianBeamExperiment(input: GaussianBeamInput) {
  const s={...input};within('wavelength',s.wavelength,200e-9,2e-6);within('waistRadius',s.waistRadius,5e-6,.002);
  within('amplitude',s.amplitude,0,2);within('preDistance',s.preDistance,0,2);within('postDistance',s.postDistance,0,2);focal(s.focalLength);
  const k=2*Math.PI/s.wavelength,zR=k*s.waistRadius**2/2,q0:OpticsComplex=[0,-zR];
  const beforeQ:OpticsComplex=[s.preDistance,-zR],afterQ=transformGaussianQ(beforeQ,thinLensRayMatrix(s.focalLength));
  const lensBefore=geometry(beforeQ,k),lensAfter=geometry(afterQ,k),matrix=rayTransferMatrix(s.preDistance,s.postDistance,s.focalLength);
  // Compose the already calculated lens state with free translation. This avoids
  // cancellation in a second global Möbius quotient at a very narrow new waist.
  const output=geometry([afterQ[0]+s.postDistance,afterQ[1]],k);
  // 2D field prefactor = product of two 1D Fresnel factors, with continuous forward Gouy phase.
  const preDen:OpticsComplex=[1,s.preDistance/zR],aLens=lensAfter.coefficient;
  const postDen:OpticsComplex=[1-2*s.postDistance*aLens[1]/k,2*s.postDistance*aLens[0]/k];
  const prefactor=divide(divide([s.amplitude,0],preDen),postDen);
  const gouyAdvance=Math.atan2(preDen[1],preDen[0])+Math.atan2(postDen[1],postDen[0]);
  const intensityIntegral=s.amplitude**2*Math.PI*s.waistRadius**2/2;
  const newWaistCoordinate=-afterQ[0],newRayleigh=-afterQ[1],newWaistRadius=Math.sqrt(2*newRayleigh/k);
  const divergence=s.wavelength/(Math.PI*newWaistRadius);
  const sampleField=(radius:number)=>{
    within('radius',radius,0,1e6);
    const phase=-output.coefficient[1]*radius**2,magnitude=Math.exp(-output.coefficient[0]*radius**2);
    const value=cmul(prefactor,[magnitude*Math.cos(phase),magnitude*Math.sin(phase)]);
    return {radius,field:value,intensity:value[0]**2+value[1]**2,phase:s.amplitude===0||magnitude===0?null:Math.atan2(value[1],value[0])};
  };
  const radialProfile=Array.from({length:97},(_,i)=>sampleField(i/96*3*output.width));
  const longitudinal=Array.from({length:129},(_,i)=>{
    const coordinate=-s.preDistance+(s.preDistance+s.postDistance)*i/128;
    const m=coordinate<0?freeRayMatrix(s.preDistance+coordinate):rayTransferMatrix(s.preDistance,coordinate,s.focalLength);
    const g=geometry(transformGaussianQ(q0,m),k);
    return {coordinate,width:g.width,inverseCurvature:g.inverseCurvature,geometricalSignedRadius:m[0][0]*s.waistRadius,
      diffractionRadius:Math.abs(m[0][1])*s.wavelength/(Math.PI*s.waistRadius)};
  });
  const initialVariance=s.waistRadius**2/4,initialAngleVariance=(s.wavelength/(Math.PI*s.waistRadius))**2/4;
  const [A,B]=matrix[0],[C,D]=matrix[1];
  const covariance={xx:A*A*initialVariance+B*B*initialAngleVariance,
    xTheta:A*C*initialVariance+B*D*initialAngleVariance,thetaTheta:C*C*initialVariance+D*D*initialAngleVariance};
  // Preserve a factor M*diag(sigma_x,sigma_theta) as well as raw second moments.
  // Computing the tiny determinant of a highly correlated covariance by subtraction
  // can lose all significant digits; report this, never repair/project the moments.
  const covarianceFactor: RayMatrix=[[A*Math.sqrt(initialVariance),B*Math.sqrt(initialAngleVariance)],
    [C*Math.sqrt(initialVariance),D*Math.sqrt(initialAngleVariance)]];
  const expectedDeterminant=(s.wavelength/(4*Math.PI))**2;
  const directDeterminant=covariance.xx*covariance.thetaTheta-covariance.xTheta**2;
  const covarianceDeterminantEvaluation={direct:directDeterminant,expected:expectedDeterminant,
    relativeResidual:(directDeterminant-expectedDeterminant)/expectedDeterminant,
    cancellationRatio:(Math.abs(covariance.xx*covariance.thetaTheta)+covariance.xTheta**2)/expectedDeterminant};
  return {input:s,k,rayleigh:zR,matrix,determinant:det(matrix),inputQ:q0,lensBefore,lensAfter,output,prefactor,gouyAdvance,
    onAxisIntensity:prefactor[0]**2+prefactor[1]**2,onAxisPhase:s.amplitude===0?null:Math.atan2(prefactor[1],prefactor[0]),
    intensityIntegral,intensityIntegralUnits:'relative-field-squared metre-squared',
    fractionWithinWidth:s.amplitude===0?null:1-Math.exp(-2),
    newWaist:{coordinate:newWaistCoordinate,radius:newWaistRadius,rayleigh:newRayleigh,
      location:newWaistCoordinate<0?'backward-extrapolated':newWaistCoordinate===0?'at-lens':'downstream'},
    divergence,lensPhaseSlope:s.focalLength===null?0:lensBefore.width/Math.abs(s.focalLength),
    geometricRadius:Math.abs(A)*s.waistRadius,diffractionRadius:Math.abs(B)*s.wavelength/(Math.PI*s.waistRadius),
    covariance,covarianceFactor,covarianceDeterminantEvaluation,emittance:s.wavelength/(4*Math.PI),radialProfile,longitudinal,sampleField,
    convention:'exp(+ikz-iwt), q waist=-i*zR, 2D axisymmetric scalar field; geometry at zero amplitude describes the reference mode, not a nonzero beam'};
}
