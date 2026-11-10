import type {FoundationResult} from './learning-foundations';
import type {Vec2} from './learning-simulations';
import {coherentFourierImaging,incoherentFourierImaging,fourierPupilTransfer,type FourierPupil,type FourierMode,type CoherentFourierInput,type IncoherentFourierInput} from './learning-fourier-optics';
export const fourierOpticsKinds=['coherent-fourier-pupil-spatial-filtering','incoherent-psf-otf-image-formation'] as const;
export type FourierOpticsKind=typeof fourierOpticsKinds[number];
function validate(v:number[],bounds:number[][],integers:number[]=[]){if(v.length!==bounds.length)throw new RangeError('Wrong Fourier-optics control count');for(let i=0;i<v.length;i++)if(!Number.isFinite(v[i])||v[i]<bounds[i][0]||v[i]>bounds[i][1]||integers.includes(i)&&!Number.isInteger(v[i]))throw new RangeError(`Invalid Fourier-optics control ${i}`);}
function pupil(v:number[]):FourierPupil{return{wavelength:v[0]/1e9,focalLength1:.1,focalLength2:v[1]/10,referenceHalfWidth:.001,apertureFraction:v[2],obstructionFraction:v[3],amplitudeTransmission:v[4],quadraticPhase:v[5],linearPhase:v[6]};}
function maskSegments(s:FourierPupil){
 const reference=s.referenceHalfWidth/(s.wavelength*s.focalLength1),cutoff=reference*s.apertureFraction,stop=cutoff*s.obstructionFraction;
 const intervals:[number,number][]=[[-cutoff,-stop],[stop,cutoff]],segments=intervals.filter(([lo,hi])=>hi>lo).map(([lo,hi])=>Array.from({length:65},(_,i)=>{const frequency=lo+(hi-lo)*(i+.5)/65,value=fourierPupilTransfer(s,frequency);return{frequency,value};}));
 return{reference,segments};
}
function coherentLab(v:number[]){
 validate(v,[[200,2000],[.5,2],[0,1],[0,1],[0,1],[-8,8],[-Math.PI,Math.PI],[0,5],[0,1],[50,500],[-.5,.5],[-Math.PI,Math.PI]],[7]);
 const beta=v[8],m=(order:number,amplitude:number,phase=0):FourierMode=>({order,amplitude,phase});
 const preparations:FourierMode[][]=[[m(0,1),m(-1,beta/2),m(1,beta/2)],[m(0,1),m(-1,beta/2,Math.PI/2),m(1,beta/2,Math.PI/2)],[m(3,beta)],[m(1,beta),m(1,beta,Math.PI)],[m(0,0)],[m(0,.7,.3),m(2,beta,-.4),m(-1,.3,.8)]];
 const system:CoherentFourierInput={...pupil(v),period:v[9]/1e6,modes:preparations[v[7]],globalPhase:v[11],queryCoordinate:v[10]*v[9]/1e6},query=coherentFourierImaging(system);
 return{system,query,preparation:v[7],mask:maskSegments(system)};
}
function incoherentLab(v:number[]){
 validate(v,[[200,2000],[.5,2],[0,1],[0,1],[0,1],[-8,8],[-Math.PI,Math.PI],[0,4],[0,2],[0,2],[0,2.5],[0,1],[-Math.PI,Math.PI],[-4,4]]);
 const common=pupil(v),ref=common.referenceHalfWidth/(common.wavelength*common.focalLength1),half=v[7]/(2*ref);
 const system:IncoherentFourierInput={...common,sources:[{position:-half,weight:v[8]},{position:half,weight:v[9]}],queryCoordinate:v[13]/ref,gratingMean:1,gratingModulation:v[11],gratingFrequency:v[10]*ref,gratingPhase:v[12]},query=incoherentFourierImaging(system);
 return{system,query,referenceFrequency:ref,mask:maskSegments(system)};
}
export interface FourierOpticsResult extends FoundationResult {kind:FourierOpticsKind;fourierOptics:true;coherent?:ReturnType<typeof coherentLab>;incoherent?:ReturnType<typeof incoherentLab>;}
export function evaluateFourierOptics(kind:FourierOpticsKind,v:number[]):FourierOpticsResult{
 if(kind===fourierOpticsKinds[0]){const r=coherentLab(v),q=r.query;return{kind,fourierOptics:true,coherent:r,curves:[q.profile.map(p=>[p.coordinate,p.outputIntensity]as Vec2)],stats:{'mean I in · rel':q.inputMeanIntensity,'mean I out · rel':q.outputMeanIntensity,'coherent T · 1':q.transmittance,'reference T · 1':q.pupil.referenceThroughput,'I query · rel':q.query.outputIntensity,'actual I query · rel':q.query.actualOutputIntensity,'actual x · µm':q.query.actualCoordinate*1e6,'magnification · 1':q.pupil.magnification,'cutoff · cycles/mm':q.pupil.cutoff/1000,'phase query · rad':q.query.outputPhase}};}
 if(kind!==fourierOpticsKinds[1])throw new RangeError('Unknown Fourier-optics lesson');const r=incoherentLab(v),q=r.query;
 return{kind,fourierOptics:true,incoherent:r,curves:[q.gratingProfile.map(p=>[p.coordinate,p.gratingOutput]as Vec2)],stats:{'reference T · 1':q.pupil.referenceThroughput,'point integral · rel':q.pointImageIntegral,'point T · 1':q.pointTransmittance,'point I query · rel/m':q.query.pointIntensity,'actual point I · rel/m':q.query.actualPointIntensity,'grating I query · rel':q.query.gratingOutput,'grating mean · rel':q.gratingReferenceMean,'contrast · 1':q.gratingObservedContrast,'OTF real · 1':q.gratingTransfer.conditional?.[0]??null,'OTF imag · 1':q.gratingTransfer.conditional?.[1]??null,'MTF · 1':q.gratingTransfer.mtf,'OTF phase · rad':q.gratingTransfer.phase}};
}
