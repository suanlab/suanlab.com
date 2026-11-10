import { nonuniformVolumeSimulation, type NonuniformVolumeParameters } from './learning-continuum-volume-discretization';
import { burgersRiemannSimulation, type BurgersParameters } from './learning-continuum-godunov';
export const continuumConservativeKinds = ['nonuniform-volume-shared-flux-mass-norm','godunov-burgers-conservative-shock-entropy'] as const;
export type ContinuumConservativeKind = typeof continuumConservativeKinds[number];
export const isContinuumConservativeKind = (id:string):id is ContinuumConservativeKind => (continuumConservativeKinds as readonly string[]).includes(id);
export const volumeControlBounds = [[3,6],[.25,4],[.25,4],[-.75,.75],[-2,2],[0,2],[0,4],[-1,1],[1,3],[0,1],[0,.25],[0,64]];
export const godunovControlBounds = [[3,6],[.5,4],[-2,2],[-2,2],[1,3],[0,.25],[0,64]];
function validate(v:number[],bounds:number[][],integers:number[]) {
  if(v.length!==bounds.length)throw new RangeError('Wrong conservative control count');
  for(let i=0;i<bounds.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||integers.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid conservative control ${i}`);
}
export function volumeDiscretizationLab(v:number[]) {
  validate(v,volumeControlBounds,[0,8,11]);
  const parameters:NonuniformVolumeParameters={cellCountExponent:v[0],length:v[1],area:v[2],meshWarp:v[3],velocity:v[4],diffusivity:v[5],meanConcentration:v[6],relativeContrast:v[7],mode:v[8],phaseCycles:v[9],timeStep:v[10],steps:v[11]};
  return nonuniformVolumeSimulation(parameters);
}
export function godunovDiscretizationLab(v:number[]) {
  validate(v,godunovControlBounds,[0,4,6]);
  const parameters:BurgersParameters={cellCountExponent:v[0],length:v[1],leftState:v[2],rightState:v[3],interfaceFraction:v[4]/4,timeStep:v[5],steps:v[6]};
  const result=burgersRiemannSimulation(parameters),L=parameters.length,x0=L*parameters.interfaceFraction;
  // Separate paths across a shock: do not interpolate through the jump.
  const exactSegments:[number,number][][]=[];
  const piece=(a:number,b:number,qa:number,qb=qa)=>{if(b>a)exactSegments.push([[a,qa],[b,qb]]);};
  const time=result.time.value;
  if(result.waveType==='constant')piece(0,L,parameters.leftState);
  else if(time===0){piece(0,x0,parameters.leftState);piece(x0,L,parameters.rightState);}
  else if(result.waveType==='shock'){
    const x=result.exactShockPosition!.value!;
    piece(0,Math.min(L,x),parameters.leftState);piece(Math.max(0,x),L,parameters.rightState);
  }else if(time!==null){
    const a=result.rarefactionLeftPosition!.value!,b=result.rarefactionRightPosition!.value!;
    piece(0,Math.min(L,a),parameters.leftState);
    const lo=Math.max(0,a),hi=Math.min(L,b);
    const q=(x:number)=>x<=a?parameters.leftState:x>=b?parameters.rightState:(x-x0)/time;
    piece(lo,hi,q(lo),q(hi));piece(Math.max(0,b),L,parameters.rightState);
  }
  return {...result,exactSegments};
}
/** Scale an observation without first multiplying its rounded linear value. */
export function conservativeScaledObservation(q:{value:number|null;sign:number;logMagnitude:number|null},factor:number) {
  if(!Number.isFinite(factor))throw new RangeError('Invalid observation factor');
  if(!q.sign||factor===0)return{value:0,sign:0,logMagnitude:null};
  const sign=q.sign*Math.sign(factor),logMagnitude=q.logMagnitude!+Math.log(Math.abs(factor)),linear=sign*Math.exp(logMagnitude);
  return{value:linear===0||!Number.isFinite(linear)?null:linear,sign,logMagnitude};
}
