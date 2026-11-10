import type {FoundationResult}from './learning-foundations';import type{Vec2}from'./learning-simulations';
import{coherentSuperpositionDetector,coherentDetectorPhaseScan,finiteApertureDiffraction,type InterferenceExperiment,type ApertureExperiment}from'./learning-interference-diffraction';
export const interferenceKinds=['coherent-superposition-detector-averages','finite-aperture-fresnel-fraunhofer']as const;
export type InterferenceKind=typeof interferenceKinds[number];
function controls(v:number[],ranges:number[][],continuous:number[]=[]){if(v.length!==ranges.length)throw new RangeError('Wrong interference control count.');for(let i=0;i<ranges.length;i++)if(!Number.isFinite(v[i])||v[i]<ranges[i][0]||v[i]>ranges[i][1]||!continuous.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid interference control${i}.`);}
function detectorLab(v:number[]){
 controls(v,[[0,2],[0,2],[0,2],[0,2],[-8,8],[0,16],[0,2],[0,16],[0,16]],[0,1]);
 const population=v[6]===0?[{phase:0,weight:1}]:v[6]===1?[{phase:0,weight:v[5]},{phase:Math.PI,weight:16-v[5]}]:[0,1,2,3].map(i=>({phase:i*Math.PI/2,weight:1}));
 const system:InterferenceExperiment={amplitude1:v[0],amplitude2:v[1],omega1:2*Math.PI*v[2],omega2:2*Math.PI*v[3],phaseDifference:v[4]*Math.PI/8,population,windowDuration:v[7]/4,timeCenter:v[8]/4},query=coherentSuperpositionDetector(system);
 const phaseScan=coherentDetectorPhaseScan(system),pointScan=coherentDetectorPhaseScan({...system,windowDuration:0}),incoherent=coherentDetectorPhaseScan({...system,population:[{phase:0,weight:1},{phase:Math.PI,weight:1}]}),count=Math.max(128,Math.ceil(Math.max(system.omega1,system.omega2)*system.windowDuration/.12));
 const timeWave=Array.from({length:system.windowDuration===0?1:count+1},(_,i)=>{const offset=system.windowDuration*(i/count-.5),phase=system.phaseDifference+population.find(p=>p.weight>0)!.phase;const u1=system.amplitude1*Math.cos(-system.omega1*(system.timeCenter+offset)),u2=system.amplitude2*Math.cos(phase-system.omega2*(system.timeCenter+offset));return {offset,u1,u2,total:u1+u2,squared:(u1+u2)**2};});
 const exposures=Array.from({length:65},(_,i)=>{const duration=4*i/64,q=coherentSuperpositionDetector({...system,windowDuration:duration});return {duration,mean:q.meanIntensity,realMean:q.realSquareMean,halfMean:q.meanIntensity/2,formulaResidual:q.realMeanFormulaResidual,refinementDifference:q.realMeanRefinementDifference,visibility:q.visibility,windowFactor:q.windowFactor};});
 return {system,query,phaseScan,pointScan,incoherent,timeWave,exposures};
}
function diffractionLab(v:number[]){
 controls(v,[[1,2],[400,800],[20,200],[0,5],[4,16],[0,2],[0,2],[-8,8],[-2,2]],[5,6,8]);
 const system:ApertureExperiment={slits:v[0]as 1|2,wavelength:v[1]/1e9,slitWidth:v[2]/1e6,fresnelNumber:[.01,.1,.5,1,4,5][v[3]],separationRatio:v[4]/4,amplitude1:v[5],amplitude2:v[6],relativePhase:v[7]*Math.PI/8};
 const query=finiteApertureDiffraction(system,v[8]);
 const framed=(q:ReturnType<typeof finiteApertureDiffraction>)=>{const angle=Math.PI*q.eta*q.eta/q.fresnelNumber-Math.PI/4,co=Math.cos(angle),si=Math.sin(angle),scale=Math.sqrt(q.fresnelNumber),frame=(z:Vec2):Vec2=>[(z[0]*co+z[1]*si)/scale,(z[1]*co-z[0]*si)/scale];return {...q,apertureFresnel:frame(q.fresnel),apertureFraunhofer:frame(q.fraunhofer)};};
 const field=Array.from({length:513},(_,i)=>framed(finiteApertureDiffraction(system,-2+4*i/512)));
 const limits=[.01,.1,.5,1,4,5].map(fresnelNumber=>{const q=finiteApertureDiffraction({...system,fresnelNumber},v[8]);return {F:fresnelNumber,distance:q.distance,scaledFieldError:q.complexFieldDifference/Math.sqrt(fresnelNumber),scaledIntensityError:q.intensityDifference/fresnelNumber,transmitted:q.expectedTransmittedNorm,discardedPhase:q.discardedAperturePhase};});
 const mask=system.slits===1?[{center:0,amplitude:system.amplitude1,phase:0}]:[{center:-system.separationRatio/2,amplitude:system.amplitude1,phase:0},{center:system.separationRatio/2,amplitude:system.amplitude2,phase:system.relativePhase}];
 return {system,query:framed(query),field,limits,mask};
}
export interface InterferenceResult extends FoundationResult{kind:InterferenceKind;interferenceOptics:true;detector?:ReturnType<typeof detectorLab>;diffraction?:ReturnType<typeof diffractionLab>;}
export function evaluateInterference(kind:InterferenceKind,v:number[]):InterferenceResult{
 if(kind===interferenceKinds[0]){const r=detectorLab(v),q=r.query;return {kind,interferenceOptics:true,detector:r,curves:[r.phaseScan.map(p=>[p.phase,p.intensity]as Vec2)],stats:{'t0 · s':q.timeCenter,'T · s':q.windowDuration,'I mean · rel':q.meanIntensity,'I point · rel':q.instantaneousIntensity,'real mean · rel':q.realSquareMean,'I mean/2 · rel':q.meanIntensity/2,'cross · rel':q.crossTerm,'S · 1':q.windowFactor,'V · 1':q.visibility,'Re γ12 · 1':q.mutualCoherence12?.[0]??null,'Im γ12 · 1':q.mutualCoherence12?.[1]??null}};}
 if(kind!==interferenceKinds[1])throw new RangeError('Unknown interference lesson.');const r=diffractionLab(v),q=r.query;return {kind,interferenceOptics:true,diffraction:r,curves:[r.field.map(p=>[p.eta,p.fresnelIntensity]as Vec2)],stats:{'η · 1':q.eta,'x · m':q.x,'z · m':q.distance,'F · 1':q.fresnelNumber,'IF · rel':q.fresnelIntensity,'IFF · rel':q.fraunhoferIntensity,'IF/F · rel':q.scaledFresnelIntensity,'IFF/F · rel':q.scaledFraunhoferIntensity,'norm expected · m':q.expectedTransmittedNorm,'|UF−UFF| · rel':q.complexFieldDifference,'ΦA · rad':q.discardedAperturePhase,'geometry · 1':q.maximumTransverseRatio}};
}
