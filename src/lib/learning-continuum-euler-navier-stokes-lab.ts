import { linearizedEulerAt, taylorGreenAt, taylorGreenCell, taylorGreenDecay, taylorGreenParticle, type FlowValue } from './learning-continuum-euler-navier-stokes';
export const continuumFlowKinds = ['linearized-euler-acoustic-characteristics', 'taylor-green-navier-stokes-viscous-decay'] as const;
export type ContinuumFlowKind = typeof continuumFlowKinds[number];
export const isContinuumFlowKind = (id: string): id is ContinuumFlowKind => (continuumFlowKinds as readonly string[]).includes(id);
export const acousticControlBounds = [[.25,4],[.25,4],[-4,4],[1.1,1.67],[.25,4],[-.03,.03],[-.03,.03],[-.03,.03],[-Math.PI,Math.PI],[0,4],[-4,4],[-1,1],[0,1]];
export const vortexControlBounds = [[.25,4],[0,4],[.25,4],[0,1],[-4,4],[.25,2],[0,4],[-1,1],[-1,1],[-1,1],[-1,1]];
function validate(v: number[], bounds: number[][]) {
  if (v.length !== bounds.length) throw new RangeError('Wrong flow control count');
  for (let i=0;i<v.length;i++) if (!Number.isFinite(v[i]) || v[i]<bounds[i][0] || v[i]>bounds[i][1]) throw new RangeError(`Invalid flow control ${i}`);
}
const sum = (...xs: FlowValue[]): FlowValue => xs.includes(null) ? null : (xs as number[]).reduce((a,b)=>a+b,0);
const neg = (x: FlowValue): FlowValue => x===null?null:-x;
function product(...xs: FlowValue[]): FlowValue {
  if(xs.includes(0))return 0;if(xs.includes(null))return null;
  let r=1;for(const x of xs as number[]){r*=x;if(r===0||!Number.isFinite(r))return null;}return r;
}
export function acousticFlowLab(v: number[]) {
  validate(v,acousticControlBounds);
  const parameters={density:v[0],soundSpeed:v[1],meanVelocity:v[2],gamma:v[3],waveNumber:v[4],rightAmplitude:v[5],leftAmplitude:v[6],contactAmplitude:v[7],phase:v[8]},time=v[9],x=v[10];
  const query=linearizedEulerAt(parameters,x,time),halfAmplitude=[v[5],v[6],v[7]].some(a=>a!==0&&a/2===0)?null:linearizedEulerAt({...parameters,rightAmplitude:v[5]/2,leftAmplitude:v[6]/2,contactAmplitude:v[7]/2},x,time);
  const spatial=Array.from({length:241},(_,i)=>{const x=-4+i/30;return{x,value:linearizedEulerAt(parameters,x,time)};});
  const temporal=Array.from({length:81},(_,i)=>{const time=i/20;return{time,value:linearizedEulerAt(parameters,x,time)};});
  const characteristics=[parameters.meanVelocity+parameters.soundSpeed,parameters.meanVelocity-parameters.soundSpeed,parameters.meanVelocity].map((speed,i)=>({speed,index:i,points:[[0,0],[4*speed,4]] as [number,number][]}));
  const left=v[11]-v[12],right=v[11]+v[12],count=128;
  const nodes=Array.from({length:count+1},(_,i)=>({weight:i===0||i===count?1:i%2?4:2,value:linearizedEulerAt(parameters,left+2*v[12]*i/count,time)}));
  const supported=v[12]===0||v[12]>=1e-8;
  const integral=(get:(q:typeof query)=>FlowValue)=>!supported?null:v[12]===0?0:product(2,v[12],1/(3*count),sum(...nodes.map(n=>product(n.weight,get(n.value)))));
  const l=nodes[0].value,r=nodes[count].value;
  const flux=(q:typeof query)=>[
    sum(product(parameters.meanVelocity,q.perturbation.density),product(parameters.density,q.perturbation.velocity)),
    sum(product(parameters.meanVelocity,q.perturbation.velocity),q.perturbation.pressure===null?null:q.perturbation.pressure/parameters.density),
    sum(product(parameters.meanVelocity,q.perturbation.pressure),product(parameters.density,parameters.soundSpeed,parameters.soundSpeed,q.perturbation.velocity))];
  const keys=['density','velocity','pressure'] as const,lf=flux(l),rf=flux(r);
  const window=keys.map((key,i)=>{const rate=integral(q=>q.dt[key]),storage=integral(q=>q.perturbation[key]),netFlux=!supported?null:v[12]===0?0:sum(rf[i],neg(lf[i]));return{key,storage,rate,leftFlux:lf[i],rightFlux:rf[i],netFlux,residual:sum(rate,netFlux)};});
  const energyRate=(q:typeof query)=>sum(product(parameters.density,q.perturbation.velocity,q.dt.velocity),product(q.perturbation.pressure,q.dt.pressure,1/(parameters.density*parameters.soundSpeed**2)));
  const energyWindow={storage:integral(q=>q.acousticEnergy),rate:integral(energyRate),netFlux:!supported?null:v[12]===0?0:sum(r.energyFlux,neg(l.energyFlux))};
  return{parameters,time,x,query,halfAmplitude,spatial,temporal,characteristics,windowLeft:left,windowRight:right,window,energyWindow:{...energyWindow,residual:sum(energyWindow.rate,energyWindow.netFlux)},windowStatus:!supported?'below-supported-width' as const:v[12]===0?'collapsed-zero-width' as const:'finite-window' as const};
}
export function taylorGreenFlowLab(v: number[]) {
  validate(v,vortexControlBounds);
  const parameters={density:v[0],amplitude:v[1],waveNumber:v[2],viscosity:v[3],pressureOffset:v[4],depth:v[5]},time=v[6],x=v[7],y=v[8];
  const query=taylorGreenAt(parameters,x,y,time),cell=taylorGreenCell(parameters,time),start:[number,number]=[v[9],v[10]];
  const particle=taylorGreenParticle(parameters,start,time,256),coarseParticle=taylorGreenParticle(parameters,start,time,128);
  const endpoint=particle.points.at(-1)!,coarseEndpoint=coarseParticle.points.at(-1)!;
  const endpointDifference=particle.status==='integrated'&&coarseParticle.status==='integrated'?Math.hypot(endpoint.x-coarseEndpoint.x,endpoint.y-coarseEndpoint.y):null;
  const label=(px:number,py:number)=>Math.cos(parameters.waveNumber*px)*Math.cos(parameters.waveNumber*py);
  const grid=Array.from({length:81},(_,i)=>{const x=-cell.side/2+(i%9)*cell.side/8,y=-cell.side/2+Math.floor(i/9)*cell.side/8;return{x,y,value:taylorGreenAt(parameters,x,y,time)};});
  const temporal=Array.from({length:81},(_,i)=>{const time=i/20;return{time,value:taylorGreenCell(parameters,time),point:taylorGreenAt(parameters,x,y,time)};});
  const profile=Array.from({length:121},(_,i)=>{const x=-cell.side/2+i*cell.side/120;return{x,value:taylorGreenAt(parameters,x,0,time)};});
  const tinyViscosities=Array.from({length:121},(_,i)=>{const logViscosity=-30+i/4,viscosity=10**logViscosity;return{logViscosity,value:taylorGreenDecay({...parameters,viscosity},time)};});
  const contours=parameters.amplitude===0?[]:[-.75,-.25,.25,.75].flatMap(level=>[-1,1].map(sign=>({level,points:Array.from({length:241},(_,i)=>{
    const x=-cell.side/2+i*cell.side/240,ratio=level/Math.cos(parameters.waveNumber*x);return Math.abs(ratio)>1?null:[x,sign*Math.acos(ratio)/parameters.waveNumber]as[number,number];
  })})));
  return{parameters,time,x,y,start,query,cell,grid,profile,temporal,tinyViscosities,particle,coarseParticle,endpointDifference,
    initialLabel:label(...start),endpointLabel:label(endpoint.x,endpoint.y),contours};
}
