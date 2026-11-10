import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {periodicWaveChain,wavePulsePoint,wavePulseEnergyBalance,type PulseMedium} from './learning-wave-equation';
export const waveEquationKinds=['coupled-chain-continuum-wave-limit','wave-initial-data-reflection-energy-flux'] as const;
export type WaveEquationKind=typeof waveEquationKinds[number];
function validate(v:number[],bounds:number[][]){if(v.length!==bounds.length+1)throw new RangeError('Wrong wave-equation control count.');for(let i=0;i<bounds.length;i++)if(!Number.isInteger(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1])throw new RangeError(`Invalid wave-equation control${i}.`);const f=v[v.length-1];if(!Number.isFinite(f)||f<0||f>1)throw new RangeError('Query fraction must be between0 and1.');}
function chainLab(v:number[]){
 validate(v,[[1,16],[1,16],[0,4],[-10,10],[-5,5],[-10,10],[-5,5],[0,8]]);
 const system={density:v[0]/4,tension:v[1]/4,length:8,particles:2**(v[2]+3)},duration=v[7],time=duration*v[8];
 const harmonics=[{mode:0,positionCos:0,positionSin:0,velocityCos:v[6]/100,velocitySin:0},{mode:1,positionCos:v[3]/100,positionSin:0,velocityCos:0,velocitySin:0},{mode:2,positionCos:0,positionSin:0,velocityCos:v[5]/100,velocitySin:0},{mode:3,positionCos:0,positionSin:v[4]/100,velocityCos:0,velocitySin:0}];
 const query=periodicWaveChain(system,harmonics,time),initial=periodicWaveChain(system,harmonics,0),frequency=Math.max(...query.spectrum.map(a=>a.continuumFrequency));
 const count=Math.max(128,Math.ceil(frequency*duration/.15));
 const history=Array.from({length:duration===0?1:count+1},(_,i)=>{const r=periodicWaveChain(system,harmonics,duration*i/count);return {time:r.time,energy:r.energy,modalEnergy:r.modalEnergy.reduce((a,b)=>a+b,0),continuumEnergy:r.continuumEnergy,rms:r.rmsDisplacementError,momentum:r.momentum};});
 const field=Array.from({length:257},(_,i)=>{const x=system.length*i/256;let displacement=0,velocity=0;for(const a of query.spectrum){const p=a.waveNumber*x;displacement+=a.continuumCos.q*Math.cos(p)+a.continuumSin.q*Math.sin(p);velocity+=a.continuumCos.v*Math.cos(p)+a.continuumSin.v*Math.sin(p);}return {x,displacement,velocity};});
 const refinements=[8,16,32,64,128].map(particles=>{const r=periodicWaveChain({...system,particles},harmonics,time);return {particles,spacing:r.spacing,mass:r.mass,stiffness:r.stiffness,rms:r.rmsDisplacementError,energy:r.energy};});
 return {system,harmonics,duration,query,initial,history,field,refinements};
}
function pulseLab(v:number[]){
 // Fraction is last; center2m/halfwidth<=1.5m keeps every boundary choice compatible.
 validate(v,[[1,16],[1,16],[0,2],[0,2],[-10,10],[-30,30],[2,6],[0,8],[0,32]]);
 const system:PulseMedium={density:v[0]/4,tension:v[1]/4,boundary:(['whole-line','fixed','free']as const)[v[2]],direction:(['left','right','split']as const)[v[3]],amplitude:v[4]/100,velocityAmplitude:v[5]/100,width:v[6]/4,center:2};
 const duration=v[7],time=duration*v[9],at=v[8]/4,c=Math.sqrt(system.tension/system.density),left=system.boundary==='whole-line'?-8:0,right=8;
 const query=wavePulsePoint(system,at,time),budget=wavePulseEnergyBalance(system,[0,4],time);
 const spatialCount=Math.max(256,Math.ceil((right-left)*24/system.width)),grid=Array.from({length:spatialCount+1},(_,i)=>left+(right-left)*i/spatialCount);
 const edges=[system.center-system.width,system.center+system.width];if(system.boundary!=='whole-line')edges.push(...edges.map(x=>-x));
 grid.push(at,...edges.flatMap(x=>[x-c*time,x+c*time]).filter(x=>x>=left&&x<=right));
 const xs=[...new Set(grid)].sort((a,b)=>a-b),field=xs.map(x=>wavePulsePoint(system,x,time)),initial=xs.map(x=>wavePulsePoint(system,x,0));
 const count=Math.max(128,Math.ceil(duration*c*20/system.width));
 const times=Array.from({length:duration===0?1:count+1},(_,i)=>duration*i/count);
 times.push(...edges.flatMap(x=>[...([0,4,at].flatMap(y=>[(x-y)/c,(y-x)/c]))]).filter(t=>t>=0&&t<=duration));
 const history=[...new Set(times)].sort((a,b)=>a-b).map(t=>{const b=wavePulseEnergyBalance(system,[0,4],t),q=wavePulsePoint(system,at,t);return {...b,pointDisplacement:q.displacement,pointVelocity:q.velocity,leftFlux:wavePulsePoint(system,0,t).flux,rightFlux:wavePulsePoint(system,4,t).flux};});
 return {system,c,duration,time,at,left,right,query,budget,field,initial,history};
}
export interface WaveEquationResult extends FoundationResult {kind:WaveEquationKind;waveEquation:true;chain?:ReturnType<typeof chainLab>;pulse?:ReturnType<typeof pulseLab>;}
export function evaluateWaveEquation(kind:WaveEquationKind,values:number[]):WaveEquationResult{
 if(kind===waveEquationKinds[0]){const r=chainLab(values),q=r.query;return {kind,waveEquation:true,chain:r,curves:[r.history.map(p=>[p.time,p.rms]as Vec2)],stats:{'t · s':q.time,'N':r.system.particles,'a · m':q.spacing,'m · kg':q.mass,'κ · N/m':q.stiffness,'c · m/s':q.c,'E chain · J':q.energy,'E continuum · J':q.continuumEnergy,'RMS · m':q.rmsDisplacementError,'p · kg m/s':q.momentum}};}
 if(kind!==waveEquationKinds[1])throw new RangeError('Unknown wave-equation lesson.');
 const r=pulseLab(values),q=r.query,b=r.budget;return {kind,waveEquation:true,pulse:r,curves:[r.field.map(p=>[p.x,p.displacement]as Vec2)],stats:{'t · s':r.time,'x · m':r.at,'c · m/s':r.c,'u · m':q.displacement,'ut · m/s':q.velocity,'ux · 1':q.gradient,'e · J/m':q.energyDensity,'J · W':q.flux,'E[0,4] · J':b.energyNow,'E total · J':b.totalEnergyNow,'ΔE−Win+Wout · J':b.residual}};
}
