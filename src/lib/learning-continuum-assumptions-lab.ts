import {coarseGrainedParticleState,coarseGrainedIntervalMass,continuumScaleWindow,independentCountFluctuations} from './learning-continuum-assumptions';
export const continuumAssumptionKinds=['particle-kernel-continuum-fields','continuum-scale-separation-counting'] as const;
export type ContinuumAssumptionKind=typeof continuumAssumptionKinds[number];
export const isContinuumAssumptionKind=(id:string):id is ContinuumAssumptionKind=>(continuumAssumptionKinds as readonly string[]).includes(id);
function validate(values:number[],bounds:number[][],integers:number[]) {
  if(values.length!==bounds.length)throw new RangeError('Wrong assumption control count');
  for(let i=0;i<values.length;i++)if(!Number.isFinite(values[i])||values[i]<bounds[i][0]||values[i]>bounds[i][1]||integers.includes(i)&&!Number.isInteger(values[i]))throw new RangeError(`Invalid assumption control ${i}`);
}
export function particleAssumptionLab(v:number[]) {
  validate(v,[[.1,4],[0,4],[-1,1],[0,1],[.02,2],[-6,6],[-1,1],[-.5,.5],[-2,2],[0,1],[1,2]],[10]);
  const [mass2,separation,drift,spread,h,x,time,bandwidthRate,center,halfWidth,count]=v;
  const particles=[{mass:1,position:-separation/2,velocity:drift+spread},...(count===2?[{mass:mass2,position:separation/2,velocity:drift-spread}]:[])];
  const query=coarseGrainedParticleState(particles,x,time,h,bandwidthRate);
  const positions=new Set(Array.from({length:161},(_,i)=>-6+12*i/160));
  for(const p of particles)for(let i=0;i<=32;i++){const a=p.position+p.velocity*time-h+2*h*i/32;if(a>=-6&&a<=6)positions.add(a);}
  const spatial=[...positions].sort((a,b)=>a-b).map(x=>({x,value:coarseGrainedParticleState(particles,x,time,h,bandwidthRate)}));
  const left=center-halfWidth,right=center+halfWidth;
  const widthStatus=halfWidth===0?'zero-length':right===left||right-left<1e-8?'unsupported-positive-width':'observed';
  const observe=(t:number)=>widthStatus==='unsupported-positive-width'?{mass:null,meanDensity:null,width:right-left}:coarseGrainedIntervalMass(particles,left,right,t,h);
  const interval=observe(time),totalMass=particles.reduce((n,p)=>n+p.mass,0);
  const temporal=Array.from({length:81},(_,i)=>{const time=-1+i/40;return{time,...observe(time)};});
  const widths=Array.from({length:100},(_,i)=>{const h=.02+1.98*i/99;return{h,value:coarseGrainedParticleState(particles,x,time,h,bandwidthRate)};});
  return{particles,h,x,time,bandwidthRate,center,halfWidth,left,right,count,query,spatial,temporal,widths,interval,totalMass,widthStatus};
}
export function scaleAssumptionLab(v:number[]) {
  validate(v,[[0,30],[-12,3],[-9,6],[-12,6],[1e-4,1],[1e-4,.5],[1,1000],[1,1000],[0,1]],[7]);
  const [logDensity,logMeanFreePath,logMacroLength,logAveragingLength,epsilon,eta,factor,totalCount,p]=v;
  const input={numberDensity:10**logDensity,meanFreePath:10**logMeanFreePath,macroLength:10**logMacroLength,averagingLength:10**logAveragingLength,relativeFluctuationTolerance:epsilon,relativeResolutionTolerance:eta,collisionSeparationFactor:factor};
  const query=continuumScaleWindow(input),count=independentCountFluctuations(totalCount,p);
  const lengths=Array.from({length:145},(_,i)=>{const logLength=-12+i/8;return{logLength,value:continuumScaleWindow({...input,averagingLength:10**logLength})};});
  const fractions=Array.from({length:101},(_,i)=>{const p=i/100;return{p,value:independentCountFluctuations(totalCount,p)};});
  return{input,query,totalCount,p,count,lengths,fractions};
}
