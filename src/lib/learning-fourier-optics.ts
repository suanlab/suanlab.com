/** Continuous 1D scalar paraxial 4F imaging. FT kernel exp(-i 2pi nu x).
 * Actual output x'=m*x, m=-f2/f1; relative squared-field units, not watts.
 * No DFT, detector pixels, 2D circular pupil or hidden aperture renormalization.
 */
export type FourierComplex = [number,number];
export interface FourierPupil {
 wavelength:number; focalLength1:number; focalLength2:number; referenceHalfWidth:number;
 apertureFraction:number; obstructionFraction:number; amplitudeTransmission:number;
 quadraticPhase:number; linearPhase:number;
}
export interface FourierMode {order:number;amplitude:number;phase:number;}
export interface CoherentFourierInput extends FourierPupil {period:number;modes:FourierMode[];globalPhase:number;queryCoordinate:number;}
export interface IncoherentSource {position:number;weight:number;}
export interface IncoherentFourierInput extends FourierPupil {
 sources:IncoherentSource[];queryCoordinate:number;gratingMean:number;gratingModulation:number;gratingFrequency:number;gratingPhase:number;
}
const tau=2*Math.PI,epsilon=64*Number.EPSILON;
function bound(name:string,v:number,lo:number,hi:number){if(!Number.isFinite(v)||v<lo||v>hi)throw new RangeError(`${name} must be finite in [${lo}, ${hi}]`);}
const add=(a:FourierComplex,b:FourierComplex):FourierComplex=>[a[0]+b[0],a[1]+b[1]];
const scale=(a:FourierComplex,b:number):FourierComplex=>[a[0]*b,a[1]*b];
const multiply=(a:FourierComplex,b:FourierComplex):FourierComplex=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const norm=(a:FourierComplex)=>a[0]**2+a[1]**2;
function exponential(v:number):FourierComplex {
 const n=v/(Math.PI/2);
 if(Number.isInteger(n)){const j=((n%4)+4)%4;return [[1,0],[0,1],[-1,0],[0,-1]][j] as FourierComplex;}
 return [Math.cos(v),Math.sin(v)];
}
function sinc(v:number){return Math.abs(v)<1e-4?1-v*v/6+v**4/120-v**6/5040:Math.sin(v)/v;}
function phase(v:FourierComplex){return v[0]===0&&v[1]===0?null:Math.atan2(v[1],v[0]);}
function pupilSetup(s:FourierPupil){
 bound('wavelength',s.wavelength,200e-9,2e-6);bound('focalLength1',s.focalLength1,.005,.5);bound('focalLength2',s.focalLength2,.005,.5);bound('referenceHalfWidth',s.referenceHalfWidth,10e-6,.005);
 for(const k of['apertureFraction','obstructionFraction','amplitudeTransmission']as const)bound(k,s[k],0,1);
 bound('quadraticPhase',s.quadraticPhase,-8,8);bound('linearPhase',s.linearPhase,-Math.PI,Math.PI);
 const referenceCutoff=s.referenceHalfWidth/(s.wavelength*s.focalLength1),cutoff=referenceCutoff*s.apertureFraction;
 const bandwidth=2*cutoff*(1-s.obstructionFraction),pupilEnergy=bandwidth*s.amplitudeTransmission**2;
 const intervals:[number,number][]=[[-1,-s.obstructionFraction],[s.obstructionFraction,1]].filter(([a,b])=>b>a) as [number,number][];
 return {referenceCutoff,cutoff,bandwidth,pupilEnergy,referenceThroughput:s.apertureFraction*(1-s.obstructionFraction)*s.amplitudeTransmission**2,
  magnification:-s.focalLength2/s.focalLength1,paraxialSlope:s.referenceHalfWidth*s.apertureFraction/s.focalLength1,intervals};
}
function transfer(s:FourierPupil,p:ReturnType<typeof pupilSetup>,frequency:number):FourierComplex{
 if(p.cutoff===0||s.amplitudeTransmission===0||s.obstructionFraction===1)return [0,0];
 const t=frequency/p.cutoff;
 if(Math.abs(t)>=1||Math.abs(t)<s.obstructionFraction)return [0,0];
 return scale(exponential(s.quadraticPhase*t*t+s.linearPhase*t),s.amplitudeTransmission);
}
export function fourierPupilTransfer(s:FourierPupil,frequency:number){
 const p=pupilSetup(s);bound('frequency',frequency,-1e8,1e8);return transfer(s,p,frequency);
}
function otf(s:FourierPupil,p:ReturnType<typeof pupilSetup>,frequency:number){
 let raw:FourierComplex=[0,0];
 if(p.pupilEnergy>0){
  if(frequency===0)raw=[p.pupilEnergy,0];
  else {
   const d=frequency/p.cutoff;
   if(Math.abs(d)<2)for(const a of p.intervals)for(const b of p.intervals){
    const lo=Math.max(a[0],b[0]+d),hi=Math.min(a[1],b[1]+d);
    if(hi>lo){const length=hi-lo,mid=(lo+hi)/2,angle=2*s.quadraticPhase*d*mid-s.quadraticPhase*d*d+s.linearPhase*d;
     raw=add(raw,scale(exponential(angle),s.amplitudeTransmission**2*p.cutoff*length*sinc(s.quadraticPhase*d*length)));
    }
   }
  }
 }
 const conditional:FourierComplex|null=p.pupilEnergy>0?[raw[0]/p.pupilEnergy,raw[1]/p.pupilEnergy]:null;
 return {frequency,raw,referenceTransfer:scale(raw,1/(2*p.referenceCutoff)),conditional,mtf:conditional?Math.hypot(...conditional):null,phase:conditional?phase(conditional):null};
}
export function fourierPupilOTF(s:FourierPupil,frequency:number){const p=pupilSetup(s);bound('frequency',frequency,-1e8,1e8);return otf(s,p,frequency);}
const nodes=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363];
const weights=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
function integral(s:FourierPupil,z:number,lo:number,hi:number,steps:number):FourierComplex{
 // Exact elementary branch for a purely linear pupil phase, including signed sinc.
 if(s.quadraticPhase===0){const length=hi-lo,mid=(lo+hi)/2;return scale(exponential(z*mid),length*sinc(z*length/2));}
 let result:FourierComplex=[0,0];const step=(hi-lo)/steps;
 for(let i=0;i<steps;i++){const mid=lo+(i+.5)*step;for(let j=0;j<4;j++){const a=mid-step*nodes[j]/2,b=mid+step*nodes[j]/2;
  result=add(result,scale(add(exponential(s.quadraticPhase*a*a+z*a),exponential(s.quadraticPhase*b*b+z*b)),weights[j]*step/2));
 }}return result;
}
function kernel(s:FourierPupil,p:ReturnType<typeof pupilSetup>,x:number){
 bound('scaled point-spread coordinate',x*p.referenceCutoff,-64,64);
 let field:FourierComplex=[0,0],coarse:FourierComplex=[0,0],panels=0;
 if(p.pupilEnergy>0){const z=s.linearPhase+tau*p.cutoff*x;
  for(const [lo,hi]of p.intervals){const steps=Math.max(1,Math.ceil((hi-lo)*(2*Math.abs(s.quadraticPhase)+Math.abs(z))/4));
   field=add(field,scale(integral(s,z,lo,hi,2*steps),s.amplitudeTransmission*p.cutoff));
   coarse=add(coarse,scale(integral(s,z,lo,hi,steps),s.amplitudeTransmission*p.cutoff));panels+=2*steps;
  }
 }
 const intensity=norm(field),normalized=p.pupilEnergy>0?intensity/p.pupilEnergy:null;
 return {coordinate:x,actualCoordinate:p.magnification*x,field,fieldPhase:phase(field),rawIntensity:intensity,conditionalPSF:normalized,
  referenceIntensity:intensity/(2*p.referenceCutoff),actualConditionalPSF:normalized===null?null:normalized/Math.abs(p.magnification),
  refinementDifference:Math.hypot(field[0]-coarse[0],field[1]-coarse[1]),quadraturePanels:panels,quadratureStatus:'refinement-comparison-not-certified-error-bound' as const};
}
export function fourierPointSpread(s:FourierPupil,x:number){const p=pupilSetup(s);return kernel(s,p,x);}
export function coherentFourierImaging(input:CoherentFourierInput){
 const s={...input},p=pupilSetup(s);bound('period',s.period,20e-6,.002);bound('globalPhase',s.globalPhase,-4*Math.PI,4*Math.PI);bound('queryCoordinate',s.queryCoordinate,-.002,.002);
 if(!Array.isArray(s.modes)||s.modes.length<1||s.modes.length>33)throw new RangeError('Use 1–33 finite Fourier preparations');
 const combined=new Map<number,FourierComplex>();
 for(let i=0;i<s.modes.length;i++){const m=s.modes[i];if(!m)throw new RangeError('Missing Fourier mode');bound('order',m.order,-16,16);if(!Number.isInteger(m.order))throw new RangeError('Fourier orders must be integers');bound('mode amplitude',m.amplitude,0,2);bound('mode phase',m.phase,-4*Math.PI,4*Math.PI);
  combined.set(m.order,add(combined.get(m.order)??[0,0],scale(exponential(m.phase),m.amplitude)));
 }
 const modes=[...combined].sort((a,b)=>a[0]-b[0]).map(([order,coefficient])=>{const frequency=order/s.period,mask=transfer(s,p,frequency),outputCoefficient=multiply(coefficient,mask);return{order,frequency,fourierPlaneCoordinate:s.wavelength*s.focalLength1*frequency,coefficient,transfer:mask,outputCoefficient};});
 const inputMean=modes.reduce((a,m)=>a+norm(m.coefficient),0),outputMean=modes.reduce((a,m)=>a+norm(m.outputCoefficient),0);
 const observe=(x:number)=>{bound('coordinate',x,-.002,.002);let before:FourierComplex=[0,0],after:FourierComplex=[0,0];
  for(const m of modes){const clock=exponential(tau*m.frequency*x);before=add(before,multiply(m.coefficient,clock));after=add(after,multiply(m.outputCoefficient,clock));}
  before=multiply(before,exponential(s.globalPhase));after=multiply(after,exponential(s.globalPhase));const actualField=scale(after,1/Math.sqrt(Math.abs(p.magnification)));
  return{coordinate:x,actualCoordinate:p.magnification*x,inputField:before,outputField:after,actualOutputField:actualField,inputIntensity:norm(before),outputIntensity:norm(after),actualOutputIntensity:norm(actualField),inputPhase:phase(before),outputPhase:phase(after)};
 };
 const profile=Array.from({length:257},(_,i)=>observe(s.period*(i/256-.5)));
 return {input:{...s,modes:s.modes.map(m=>({...m}))},pupil:p,modes,profile,query:observe(s.queryCoordinate),inputMeanIntensity:inputMean,outputMeanIntensity:outputMean,
  transmittance:inputMean===0?null:outputMean/inputMean,inputPeriodIntegral:inputMean*s.period,outputPeriodIntegral:outputMean*s.period,actualImagePeriod:Math.abs(p.magnification)*s.period,
  fieldOfViewSlope:Math.max(s.period/2,Math.abs(s.queryCoordinate))/s.focalLength1,observe,convention:'continuous 1D periodic finite Fourier field; actual x=m*x_reduced; one-dimensional flux-preserving relay amplitude 1/sqrt(|m|); carrier/global relay phase removed'};
}
export function incoherentFourierImaging(input:IncoherentFourierInput){
 const s={...input},p=pupilSetup(s);bound('query scaled coordinate',s.queryCoordinate*p.referenceCutoff,-16,16);
 if(!Array.isArray(s.sources)||s.sources.length<1||s.sources.length>8)throw new RangeError('Use 1–8 nonnegative incoherent point sources');
 for(let i=0;i<s.sources.length;i++){const a=s.sources[i];if(!a)throw new RangeError('Missing incoherent source');bound('source scaled position',a.position*p.referenceCutoff,-16,16);bound('source weight',a.weight,0,2);}
 s.sources=s.sources.map(a=>({...a}));
 bound('grating mean',s.gratingMean,0,2);bound('grating modulation',s.gratingModulation,0,1);bound('grating frequency',s.gratingFrequency,0,2.5*p.referenceCutoff);bound('grating phase',s.gratingPhase,-4*Math.PI,4*Math.PI);
 const measuredOTF=otf(s,p,s.gratingFrequency),rawModulation=s.gratingModulation*(measuredOTF.mtf??0);
 if(rawModulation>1+epsilon)throw new RangeError('Constructed passive pupil violated its modulation bound');
 const evaluatedModulation=rawModulation>1?1:rawModulation;
 const gratingObserve=(x:number)=>{
  const objectPhase=tau*s.gratingFrequency*x+s.gratingPhase,observedPhase=objectPhase+(measuredOTF.phase??0);
  const positive=(modulation:number,angle:number)=>(1-modulation)+2*modulation*exponential(angle/2)[0]**2;
  const gratingInput=s.gratingMean*positive(s.gratingModulation,objectPhase);
  const gratingOutput=s.gratingMean*p.referenceThroughput*positive(evaluatedModulation,observedPhase);
  return{gratingInput,gratingOutput,actualGratingOutput:gratingOutput/Math.abs(p.magnification)};
 };
 const observe=(x:number)=>{bound('observation scaled coordinate',x*p.referenceCutoff,-16,16);
  const contributions=s.sources.map(a=>{const h=kernel(s,p,x-a.position);return{sourcePosition:a.position,weight:a.weight,intensity:a.weight*h.referenceIntensity,conditionalPSF:h.conditionalPSF,refinementDifference:h.refinementDifference};});
  const pointIntensity=contributions.reduce((a,b)=>a+b.intensity,0);
  return{coordinate:x,actualCoordinate:p.magnification*x,contributions,pointIntensity,actualPointIntensity:pointIntensity/Math.abs(p.magnification),...gratingObserve(x)};
 };
 const psfProfile=Array.from({length:129},(_,i)=>kernel(s,p,16*(i/128-.5)/p.referenceCutoff));
 const pointImageProfile=Array.from({length:129},(_,i)=>observe(32*(i/128-.5)/p.referenceCutoff));
 // The point-source view and grating preparation have different useful scales.
 // Resolve at most four grating cycles with 129 samples instead of aliasing
 // the highest supported grating into a sparse fixed-window display.
 const gratingHalfSpan=s.gratingFrequency===0?16/p.referenceCutoff:Math.min(16/p.referenceCutoff,2/s.gratingFrequency);
 const gratingProfile=Array.from({length:129},(_,i)=>{const coordinate=2*gratingHalfSpan*(i/128-.5);return{coordinate,actualCoordinate:p.magnification*coordinate,...gratingObserve(coordinate)};});
 const transferScan=Array.from({length:129},(_,i)=>otf(s,p,5*p.referenceCutoff*(i/128-.5)));
 const totalSourceWeight=s.sources.reduce((a,b)=>a+b.weight,0);
 const gratingObjectMean=s.gratingFrequency===0?gratingObserve(0).gratingInput:s.gratingMean;
 return {input:{...s,sources:s.sources.map(a=>({...a}))},pupil:p,psfProfile,pointImageProfile,gratingProfile,gratingHalfSpan,transferScan,gratingTransfer:measuredOTF,query:observe(s.queryCoordinate),totalSourceWeight,
  pointImageIntegral:p.referenceThroughput*totalSourceWeight,pointTransmittance:totalSourceWeight===0?null:p.referenceThroughput,
  gratingObjectMean,gratingReferenceMean:gratingObjectMean*p.referenceThroughput,gratingActualMean:gratingObjectMean*p.referenceThroughput/Math.abs(p.magnification),
  gratingObservedContrast:s.gratingMean===0||p.referenceThroughput===0||s.gratingFrequency===0?null:evaluatedModulation,
  modulationRoundoffCorrection:evaluatedModulation-rawModulation,fieldOfViewSlope:16/(p.referenceCutoff*s.focalLength1),observe,
  convention:'continuous 1D incoherent point-source powers and a separate nonnegative grating preparation; relative throughput is against the fixed clear reference slit; no detector pixels or finite-window normalization'};
}
