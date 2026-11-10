import type{FoundationResult}from'./learning-foundations';import type{Vec2}from'./learning-simulations';
import{jonesPolarizationExperiment,jonesPolarizationObservation,stokesPolarizationExperiment,stokesPolarizationObservation,type JonesPolarizationExperiment,type StokesPolarizationExperiment,type PolarizationOrder,type PolarizationPopulation}from'./learning-polarization';
export const polarizationKinds=['jones-polarization-waveplates-analyzers','stokes-coherency-polarization-tomography']as const;
export type PolarizationKind=typeof polarizationKinds[number];
const orders:PolarizationOrder[]=['retarder-only','analyzer-only','retarder-then-analyzer','analyzer-then-retarder'];
function controls(v:number[],bounds:number[][],integers:number[]){if(v.length!==bounds.length)throw new RangeError('Wrong polarization control count.');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||integers.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid polarization control ${i}.`);}
function jonesLab(v:number[]){
 controls(v,[[0,2],[0,2],[-Math.PI,Math.PI],[-Math.PI,Math.PI],[-Math.PI,Math.PI],[-2*Math.PI,2*Math.PI],[-Math.PI,Math.PI],[0,3],[0,2*Math.PI]],[7]);
 const system:JonesPolarizationExperiment={amplitudeX:v[0],amplitudeY:v[1],relativePhase:v[2],globalPhase:v[3],retarderAngle:v[4],retardance:v[5],analyzerAngle:v[6],order:orders[v[7]],observationPhase:v[8]},query=jonesPolarizationExperiment(system);
 const retardanceScan=Array.from({length:65},(_,i)=>{const retardance=-Math.PI+2*Math.PI*i/64,q=jonesPolarizationObservation({...system,retardance});return{retardance,intensity:q.output.intensity,transmittance:q.transmittance,stokes:q.output.stokes,major:q.output.majorAxis,minor:q.output.minorAxis};});
 const commonRadius=Math.max(Math.sqrt(query.input.intensity),Math.sqrt(query.intermediate.intensity),Math.sqrt(query.output.intensity));
 return{system,query,retardanceScan,commonRadius:commonRadius===0?1:1.15*commonRadius};
}
function stokesLab(v:number[]){
 controls(v,[[0,4],[0,4],[0,1],[0,1],[-Math.PI,Math.PI],[-2*Math.PI,2*Math.PI],[-Math.PI,Math.PI],[0,3],[0,2*Math.PI]],[1,7]);
 const intensity=v[0],amplitude=Math.sqrt(intensity),half=Math.sqrt(intensity/2),p=v[2];
 const make=(amplitudeX:number,amplitudeY:number,relativePhase:number,globalPhase:number,weight:number):PolarizationPopulation=>({amplitudeX,amplitudeY,relativePhase,globalPhase,weight});
 const populations:PolarizationPopulation[][]=[
  [make(amplitude,0,0,0,p),make(0,amplitude,0,0,1-p)],
  [make(half,half,Math.PI/2,0,p),make(half,half,-Math.PI/2,0,1-p)],
  [make(half,half,0,0,p),make(half,half,Math.PI,0,1-p)],
  [make(amplitude,0,0,0,p),make(amplitude,0,0,Math.PI,1-p)],
  [make(Math.sqrt(3*intensity/4),Math.sqrt(intensity/4),Math.PI/3,0,1)]
 ];
 const system:StokesPolarizationExperiment={population:populations[v[1]],retainedPolarization:v[3],retarderAngle:v[4],retardance:v[5],analyzerAngle:v[6],order:orders[v[7]]},query=stokesPolarizationExperiment(system);
 const depolarizationScan=Array.from({length:33},(_,i)=>{const retained=i/32,q=stokesPolarizationObservation({...system,retainedPolarization:retained});return{retained,inputDegree:q.input.degree,depolarizedDegree:q.depolarized.degree,outputDegree:q.output.degree,outputIntensity:q.output.intensity,transmittance:q.transmittance};});
 const representativeIndex=query.outputPopulation.findIndex(r=>r.probability>0&&r.field.some(z=>z[0]!==0||z[1]!==0)),representative=representativeIndex<0?null:query.outputPopulation[representativeIndex];
 const real=(phase:number):Vec2=>representative?[representative.field[0][0]*Math.cos(phase)+representative.field[0][1]*Math.sin(phase),representative.field[1][0]*Math.cos(phase)+representative.field[1][1]*Math.sin(phase)]:[0,0];
 const representativeTrajectory=representative?Array.from({length:129},(_,i)=>real(2*Math.PI*i/128)):[];
 const matrixResidual=[query.outputViaMatrix.xx-query.output.coherency.xx,query.outputViaMatrix.yy-query.output.coherency.yy,...query.outputViaMatrix.xy.map((x,i)=>x-query.output.coherency.xy[i])];
 return{system,query,depolarizationScan,source:v[1],firstProbability:p,observationPhase:v[8],representativeIndex,representative,representativeTrajectory,representativePoint:representative?real(v[8]):null,representativeRadius:representative?Math.max(1e-12,Math.sqrt(query.input.intensity)*1.15):1,matrixResidual};
}
export interface PolarizationResult extends FoundationResult{kind:PolarizationKind;polarizationOptics:true;jones?:ReturnType<typeof jonesLab>;stokes?:ReturnType<typeof stokesLab>;}
export function evaluatePolarization(kind:PolarizationKind,v:number[]):PolarizationResult{
 if(kind===polarizationKinds[0]){const r=jonesLab(v),q=r.query;return{kind,polarizationOptics:true,jones:r,curves:[q.trajectory.map(p=>p.output)],stats:{'I in · rel':q.input.intensity,'I out · rel':q.output.intensity,'transmittance · 1':q.transmittance,'mean real square · rel':q.meanRealSquare,'P out · 1':q.output.degree,'major · rel':q.output.majorAxis,'minor · rel':q.output.minorAxis,'orientation · rad':q.output.orientation,'ellipticity · 1':q.output.ellipticityRatio,'S3 out · rel':q.output.stokes[3],'τ · rad':q.observationPhase}};}
 if(kind!==polarizationKinds[1])throw new RangeError('Unknown polarization lesson.');const r=stokesLab(v),q=r.query;return{kind,polarizationOptics:true,stokes:r,curves:[q.analyzerScan.map(p=>[p.angle,p.intensity]as Vec2)],stats:{'I in · rel':q.input.intensity,'I out · rel':q.output.intensity,'transmittance · 1':q.transmittance,'P in · 1':q.input.degree,'P depol · 1':q.depolarized.degree,'P out · 1':q.output.degree,'S1 out · rel':q.output.stokes[1],'S2 out · rel':q.output.stokes[2],'S3 out · rel':q.output.stokes[3],'DOLP out · 1':q.output.degreeLinear,'DOCP out · 1':q.output.degreeCircular}};
}
