import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {paraxialRayExperiment,gaussianBeamExperiment,type ParaxialRayInput,type GaussianBeamInput} from './learning-ray-wave-optics';
export const rayWaveOpticsKinds=['paraxial-ray-matrices-thin-lens-imaging','gaussian-beam-focusing-ray-wave-limit'] as const;
export type RayWaveOpticsKind=typeof rayWaveOpticsKinds[number];
function validate(v:number[],bounds:number[][],integer:number){
 if(v.length!==bounds.length)throw new RangeError('Wrong ray/wave optics control count');
 for(let i=0;i<bounds.length;i++){const x=v[i];if(!Number.isFinite(x)||x<bounds[i][0]||x>bounds[i][1]||i===integer&&!Number.isInteger(x))throw new RangeError(`Invalid optics control ${i}`);}
}
const lens=(mode:number,magnitude:number)=>mode===2?null:(mode===1?-1:1)*magnitude/1000;
function rayLab(v:number[]){
 validate(v,[[5,2000],[0,2000],[5,2000],[0,2],[-5,5],[-.1,.1],[0,.1]],3);
 const system:ParaxialRayInput={objectDistance:v[0]/1000,screenDistance:v[1]/1000,focalLength:lens(v[3],v[2]),objectHeight:v[4]/1000,centralAngle:v[5],fanHalfAngle:v[6]},query=paraxialRayExperiment(system);
 const sensorScan=Array.from({length:97},(_,i)=>{const distance=2*i/96,B=system.focalLength===null?system.objectDistance+distance:system.objectDistance+distance-system.objectDistance*distance/system.focalLength;return{distance,span:2*Math.abs(B)*system.fanHalfAngle};});
 // Include and split at the exact singular object distance; never join across it.
 const grid=[...new Set([...Array.from({length:129},(_,i)=>.005+1.995*i/128),...(system.focalLength!==null&&system.focalLength>=.005&&system.focalLength<=2?[system.focalLength]:[])])].sort((a,b)=>a-b);
 const imageScan=grid.map(u=>{const f=system.focalLength,position=f===null||u===f?null:f*u/(u-f);return{objectDistance:u,imageDistance:position,magnification:position===null?null:-position/u,sensitivity:f===null||position===null?null:-((f/(u-f))**2)};});
 return{system,query,sensorScan,imageScan};
}
function gaussianLab(v:number[]){
 validate(v,[[200,2000],[5,2000],[0,2],[0,2000],[0,2000],[5,2000],[0,2]],6);
 const system:GaussianBeamInput={wavelength:v[0]/1e9,waistRadius:v[1]/1e6,amplitude:v[2],preDistance:v[3]/1000,postDistance:v[4]/1000,focalLength:lens(v[6],v[5])},query=gaussianBeamExperiment(system);
 // Resolve the waist on its own Rayleigh scale and include the actual observation.
 // A fixed 2 m grid can miss an entire millimetre-scale focus and mislead interpolation.
 const scanEnd=Math.min(2,Math.max(.01,2*system.postDistance,Math.max(0,query.newWaist.coordinate)+4*query.newWaist.rayleigh));
 const local=[-4,-2,-1,-.5,0,.5,1,2,4].map(a=>query.newWaist.coordinate+a*query.newWaist.rayleigh).filter(z=>z>=0&&z<=scanEnd);
 const distances=[...new Set([...Array.from({length:65},(_,i)=>scanEnd*i/64),system.postDistance,...local])].sort((a,b)=>a-b);
 const distanceScan=distances.map(distance=>{const q=gaussianBeamExperiment({...system,postDistance:distance});return{distance,width:q.output.width,inverseCurvature:q.output.inverseCurvature,gouyAdvance:q.gouyAdvance,onAxisIntensity:q.onAxisIntensity,intensityIntegral:q.intensityIntegral,covarianceDeterminantResidual:q.covarianceDeterminantEvaluation.relativeResidual};});
 return{system,query,distanceScan};
}
export interface RayWaveOpticsResult extends FoundationResult {kind:RayWaveOpticsKind;rayWaveOptics:true;ray?:ReturnType<typeof rayLab>;gaussian?:ReturnType<typeof gaussianLab>;}
export function evaluateRayWaveOptics(kind:RayWaveOpticsKind,v:number[]):RayWaveOpticsResult{
 if(kind===rayWaveOpticsKinds[0]){const r=rayLab(v),q=r.query;return{kind,rayWaveOptics:true,ray:r,curves:q.rays.map(p=>[[-r.system.objectDistance,r.system.objectHeight],[0,p.lensHeight],[r.system.screenDistance,p.screenHeight]]as Vec2[]),stats:{'image z · mm':q.imageDistance===null?null:q.imageDistance*1000,'image x · mm':q.imageHeight===null?null:q.imageHeight*1000,'magnification · 1':q.magnification,'sensor span · mm':q.screen.span*1000,'sensor rms · mm':q.screen.rmsSpread*1000,'max slope · rad':q.maxSlope,'dv/du · 1':q.imageDistanceDerivative,'det M · 1':q.determinant}};}
 if(kind!==rayWaveOpticsKinds[1])throw new RangeError('Unknown ray/wave optics lesson');
 const r=gaussianLab(v),q=r.query;return{kind,rayWaveOptics:true,gaussian:r,curves:[q.longitudinal.map(p=>[p.coordinate,p.width]as Vec2)],stats:{'w out · µm':q.output.width*1e6,'waist z · mm':q.newWaist.coordinate*1000,'waist w · µm':q.newWaist.radius*1e6,'I axis · rel':q.onAxisIntensity,'integral · rel m²':q.intensityIntegral,'Gouy advance · rad':q.gouyAdvance,'inverse R · 1/m':q.output.inverseCurvature,'divergence · rad':q.divergence,'lens slope · rad':q.lensPhaseSlope,'cov det residual · 1':q.covarianceDeterminantEvaluation.relativeResidual}};
}
