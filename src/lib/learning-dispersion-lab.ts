import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {gaussianDispersivePacket,latticeDispersivePacket,type GaussianPacket,type LatticePacket} from './learning-dispersion';
export const dispersionKinds=['dispersive-wave-packets-phase-group-velocity','lattice-wave-packet-transport-dispersion'] as const;
export type DispersionKind=typeof dispersionKinds[number];
function controls(v:number[],ranges:number[][],integers:number[]=[]){
 if(v.length!==ranges.length)throw new RangeError('Wrong packet control count.');
 for(let i=0;i<ranges.length;i++)if(!Number.isFinite(v[i])||v[i]<ranges[i][0]||v[i]>ranges[i][1]||integers.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid packet control ${i}.`);
}
function gaussianLab(v:number[]){
 controls(v,[[0,2],[.5,2],[0,8],[0,16],[-2,2],[-1,1],[0,4],[-4,4],[-8,8]],[8]);
 const system:GaussianPacket={amplitude:v[0],sigma:v[1],carrierWaveNumber:v[2],carrierFrequency:v[3],groupVelocity:v[4],dispersion:v[5],center:v[7],phase:v[8]*Math.PI/8};
 const time=v[6],center=system.center+system.groupVelocity*time,query=gaussianDispersivePacket(system,center,time),initial=gaussianDispersivePacket(system,system.center,0);
 const lo=Math.min(system.center-4*system.sigma,center-4*query.amplitudeWidth),hi=Math.max(system.center+4*system.sigma,center+4*query.amplitudeWidth);
 const maximumPhaseGradient=Math.max(system.carrierWaveNumber,Math.abs(system.carrierWaveNumber+query.chirp*(lo-center)),Math.abs(system.carrierWaveNumber+query.chirp*(hi-center)));
 // A quadratic phase has a linear gradient; endpoint maxima bound EVERY step.
 // This is display sampling, not the independent Fourier quadrature.
 const intervals=Math.max(512,Math.ceil((hi-lo)*maximumPhaseGradient/.15));
 const field=Array.from({length:intervals+1},(_,i)=>{const x=lo+(hi-lo)*i/intervals;return{x,current:gaussianDispersivePacket(system,x,time),initial:gaussianDispersivePacket(system,x,0)};});
 const evolution=Array.from({length:129},(_,i)=>{const t=i/32;return gaussianDispersivePacket(system,system.center+system.groupVelocity*t,t);});
 const spectrum=Array.from({length:129},(_,i)=>{const p=(-4+8*i/128)/system.sigma,k=system.carrierWaveNumber+p;return{k,amplitude:system.amplitude*system.sigma/Math.sqrt(2*Math.PI)*Math.exp(-(system.sigma**2)*p**2/2),omega:system.carrierFrequency+system.groupVelocity*p+system.dispersion*p*p/2,tangent:system.carrierFrequency+system.groupVelocity*p};});
 return{system,time,query,initial,field,evolution,spectrum,display:{lo,hi,intervals,maximumPhaseGradient,maximumPhaseStep:(hi-lo)*maximumPhaseGradient/intervals}};
}
function latticeLab(v:number[]){
 controls(v,[[0,1],[.25,4],[.25,4],[0,.2],[1,15],[.2,5],[0,8],[0,8],[-Math.PI,Math.PI]],[0]);
 const system:LatticePacket={particles:v[0]===0?32:64,length:8,density:v[1],tension:v[2],amplitude:v[3],modeCenter:v[4],bandwidth:v[5],center:v[6],phase:v[8]};
 const time=v[7],query=latticeDispersivePacket(system,time),initial=latticeDispersivePacket(system,0);
 const evolution=Array.from({length:65},(_,i)=>{const t=i/8,q=latticeDispersivePacket(system,t);return{time:t,energy:q.energy,norm:q.analyticNorm,transport:q.energyTransportVelocity,weighted:q.energyWeightedGroupVelocity,carrier:q.carrierGroupVelocity,analyticCenter:q.analyticCenter.position,energyCenter:q.energyCenter.position,analyticResultant:q.analyticCenter.resultant,energyResultant:q.energyCenter.resultant};});
 const dispersion=Array.from({length:129},(_,i)=>{const k=Math.PI/query.spacing*i/128;return{k,frequency:2*query.c/query.spacing*Math.sin(k*query.spacing/2),continuum:query.c*k,phase:k===0?query.c:2*query.c/query.spacing*Math.sin(k*query.spacing/2)/k,group:query.c*Math.cos(k*query.spacing/2),curvature:-query.c*query.spacing/2*Math.sin(k*query.spacing/2)};});
 return{system,time,query,initial,evolution,dispersion};
}
export interface DispersionResult extends FoundationResult{kind:DispersionKind;dispersivePackets:true;gaussian?:ReturnType<typeof gaussianLab>;lattice?:ReturnType<typeof latticeLab>;}
export function evaluateDispersion(kind:DispersionKind,v:number[]):DispersionResult{
 if(kind===dispersionKinds[0]){const r=gaussianLab(v),q=r.query;return{kind,dispersivePackets:true,gaussian:r,curves:[r.field.map(p=>[p.x,p.current.realField]as Vec2)],stats:{'t · s':q.time,'center · m':q.intensityCenter,'width · m':q.amplitudeWidth,'SD I · m':q.intensityStandardDeviation,'peak I · rel':q.intensity,'norm · m':q.integratedNorm,'vp carrier · m/s':q.carrierPhaseVelocity,'vg · m/s':q.groupVelocity,'v crest local · m/s':q.localCrestVelocity,'chirp · 1/m²':q.chirp}};}
 if(kind!==dispersionKinds[1])throw new RangeError('Unknown packet lesson.');const r=latticeLab(v),q=r.query;return{kind,dispersivePackets:true,lattice:r,curves:[q.points.map(p=>[p.x,p.displacement]as Vec2)],stats:{'t · s':q.time,'a · m':q.spacing,'E · J':q.energy,'norm · m³':q.analyticNorm,'vp carrier · m/s':q.carrierPhaseVelocity,'vg carrier · m/s':q.carrierGroupVelocity,'vE · m/s':q.energyTransportVelocity,'vg energy mean · m/s':q.energyWeightedGroupVelocity,'center I · m':q.analyticCenter.position,'center E · m':q.energyCenter.position,'R I · 1':q.analyticCenter.resultant,'R E · 1':q.energyCenter.resultant,'slope bound · 1':q.initialSlopeBound}};
}
