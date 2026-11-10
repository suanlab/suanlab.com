/** Scalar linear fields. Intensities are normalized squared-field quantities,
 * not mechanical displacement energy or an unspecified SI optical irradiance.
 */
export type WaveComplex=[number,number];
export interface PhaseWeight {phase:number;weight:number;}
export interface InterferenceExperiment {amplitude1:number;amplitude2:number;omega1:number;omega2:number;phaseDifference:number;timeCenter:number;windowDuration:number;population:PhaseWeight[];}
export interface ApertureExperiment {wavelength:number;slitWidth:number;fresnelNumber:number;separationRatio:number;amplitude1:number;amplitude2:number;relativePhase:number;slits:1|2;}
function bound(x:number,a:number,b:number,name:string){if(!Number.isFinite(x)||x<a||x>b)throw new RangeError(`${name} outside supported range.`);}
const sum=(a:WaveComplex,b:WaveComplex):WaveComplex=>[a[0]+b[0],a[1]+b[1]];
const scale=(a:WaveComplex,b:number):WaveComplex=>[a[0]*b,a[1]*b];
const multiply=(a:WaveComplex,b:WaveComplex):WaveComplex=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const exponential=(x:number):WaveComplex=>[Math.cos(x),Math.sin(x)];
const norm=(a:WaveComplex)=>a[0]*a[0]+a[1]*a[1];
function sinc(x:number){return Math.abs(x)<1e-4?1-x*x/6+x**4/120:Math.sin(x)/x;}
const nodes=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363],weights=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
function quadrature(f:(x:number)=>WaveComplex,a:number,b:number,subdivisions:number):WaveComplex{
 let result:WaveComplex=[0,0];const step=(b-a)/subdivisions;
 for(let i=0;i<subdivisions;i++){const mid=a+(i+.5)*step;for(let j=0;j<4;j++)result=sum(result,scale(sum(f(mid-nodes[j]*step/2),f(mid+nodes[j]*step/2)),weights[j]*step/2));}
 return result;
}
function checkInterference(s:InterferenceExperiment){
 for(const a of[s.amplitude1,s.amplitude2])bound(a,0,2,'Amplitude');for(const w of[s.omega1,s.omega2])bound(w,0,16,'Angular frequency');bound(s.phaseDifference,-4*Math.PI,4*Math.PI,'Relative phase');bound(s.timeCenter,-16,16,'Window center');bound(s.windowDuration,0,8,'Window duration');
 if(!Array.isArray(s.population)||s.population.length<1||s.population.length>32)throw new RangeError('Supply1–32 phase populations.');
 let total=0;for(let i=0;i<s.population.length;i++){const p=s.population[i];if(!p)throw new RangeError('Missing phase population.');bound(p.phase,-4*Math.PI,4*Math.PI,'Population phase');bound(p.weight,0,16,'Population weight');total+=p.weight;}if(total===0)throw new RangeError('Positive total phase weight required.');return s.population.map(p=>({...p,probability:p.weight/total}));
}
function detectorSetup(s:InterferenceExperiment,spatialPhase:number){
 const population=checkInterference(s);bound(spatialPhase,-8*Math.PI,8*Math.PI,'Spatial phase');
 const a=s.amplitude1,b=s.amplitude2,T=s.windowDuration,center=s.timeCenter,difference=s.omega2-s.omega1,windowFactor=sinc(difference*T/2),basePhase=s.phaseDifference+spatialPhase-difference*center;
 const phaseMoment=population.reduce<WaveComplex>((out,p)=>sum(out,scale(exponential(p.phase),p.probability)),[0,0]);
 // Positive forms avoid subtracting two nearly equal intensities at a dark fringe.
 const pointIntensity=(phase:number)=>(a-b)**2+4*a*b*Math.cos(phase/2)**2;
 const meanAt=(offset:number)=>population.reduce((out,p)=>{const phase=basePhase+p.phase+offset;return out+p.probability*((1+windowFactor)/2*pointIntensity(phase)+(1-windowFactor)/2*((a-b)**2+4*a*b*Math.sin(phase/2)**2));},0);
 return {population,a,b,T,center,difference,windowFactor,basePhase,phaseMoment,meanAt,pointIntensity};
}
/** Efficient phase scan shares the same positive detector formula as the full
 * real-field quadrature. It does not perform a second real-time integration.
 */
export function coherentDetectorPhaseScan(s:InterferenceExperiment,points=129){
 if(!Number.isInteger(points)||points<2||points>1025)throw new RangeError('Use2–1025 phase samples.');
 const setup=detectorSetup(s,0);
 return Array.from({length:points},(_,i)=>{const phase=-Math.PI+2*Math.PI*i/(points-1);return {phase,intensity:setup.meanAt(phase-s.phaseDifference)};});
}
export function coherentSuperpositionDetector(s:InterferenceExperiment,spatialPhase=0){
 const {population,a,b,T,center,difference,windowFactor,basePhase,phaseMoment,meanAt,pointIntensity}=detectorSetup(s,spatialPhase);
 const modulation=multiply(exponential(basePhase),scale(phaseMoment,windowFactor)),peakOffset=-Math.atan2(modulation[1],modulation[0]),candidatePeak=meanAt(peakOffset),candidateOpposite=meanAt(peakOffset+Math.PI),maximum=Math.max(candidatePeak,candidateOpposite),minimum=Math.min(candidatePeak,candidateOpposite),meanIntensity=meanAt(0),baseline=a*a+b*b;
 const subdivisions=Math.max(1,Math.ceil(Math.max(s.omega1,s.omega2)*T/.4));
 let realSquareMean=0,coarseRealSquareMean=0,realSquareMeanFormula=0,instantaneousIntensity=0;
 const realizations=population.map(p=>{
  const theta1=-s.omega1*center,theta2=s.phaseDifference+spatialPhase+p.phase-s.omega2*center,field1=scale(exponential(theta1),a),field2=scale(exponential(theta2),b),field=sum(field1,field2);
  // Integrate a unit exposure coordinate instead of subtracting t0±T/2.
  // A tiny positive window must not collapse when t0 is much larger than T.
  const squared=(u:number):WaveComplex=>[(a*Math.cos(theta1-s.omega1*T*u)+b*Math.cos(theta2-s.omega2*T*u))**2,0];
  const coarse=T===0?squared(0)[0]:quadrature(squared,-.5,.5,subdivisions)[0];
  const refined=T===0?coarse:quadrature(squared,-.5,.5,2*subdivisions)[0];
  const formula=a*a/2*(1+sinc(s.omega1*T)*Math.cos(2*theta1))+b*b/2*(1+sinc(s.omega2*T)*Math.cos(2*theta2))+a*b*(sinc(difference*T/2)*Math.cos(theta2-theta1)+sinc((s.omega1+s.omega2)*T/2)*Math.cos(theta1+theta2));
  realSquareMean+=p.probability*refined;coarseRealSquareMean+=p.probability*coarse;realSquareMeanFormula+=p.probability*formula;instantaneousIntensity+=p.probability*pointIntensity(theta2-theta1);
  return {...p,field1,field2,field,realField:field[0],instantaneousIntensity:norm(field),realSquareMean:refined};
 });
 const degree=a*b===0?null:[modulation[0],-modulation[1]]as WaveComplex;
 return {population,realizations,timeCenter:center,windowDuration:T,windowFactor,phaseMoment,mutualCoherence12:degree,baseline,instantaneousIntensity,meanIntensity,crossTerm:meanIntensity-baseline,maximum,minimum,
  visibility:baseline===0?null:(maximum-minimum)/(maximum+minimum),realSquareMean,realSquareMeanFormula,realMeanFormulaResidual:realSquareMean-realSquareMeanFormula,
  realMeanRefinementDifference:Math.abs(realSquareMean-coarseRealSquareMean),realQuadratureSubdivisions:T===0?0:2*subdivisions,
  quadratureStatus:'refinement-comparison-not-certified-bound' as const};
}
function apertureData(s:ApertureExperiment){
 bound(s.wavelength,400e-9,800e-9,'Wavelength');bound(s.slitWidth,20e-6,200e-6,'Slit width');bound(s.fresnelNumber,.01,5,'Fresnel number');bound(s.separationRatio,1,4,'Separation/width');for(const a of[s.amplitude1,s.amplitude2])bound(a,0,2,'Aperture amplitude');bound(s.relativePhase,-4*Math.PI,4*Math.PI,'Slit phase');if(s.slits!==1&&s.slits!==2)throw new RangeError('Choose one or two nonoverlapping slits.');
 const slits=s.slits===1?[{center:0,amplitude:s.amplitude1,phase:0}]:[{center:-s.separationRatio/2,amplitude:s.amplitude1,phase:0},{center:s.separationRatio/2,amplitude:s.amplitude2,phase:s.relativePhase}];
 return slits;
}
/** eta=x*w/(lambda*z), with w the full width of each rectangular slit.
 * Both fields retain the same observation-plane phase; only exp(ikz) is omitted.
 * Fresnel is a paraxial model, not the exact Maxwell boundary-value solution.
 */
export function finiteApertureDiffraction(s:ApertureExperiment,eta:number){
 const slits=apertureData(s);bound(eta,-2,2,'Scaled observation coordinate');const F=s.fresnelNumber,w=s.slitWidth,distance=w*w/(s.wavelength*F),x=eta*w/F,prefactor=scale(exponential(Math.PI*eta*eta/F-Math.PI/4),Math.sqrt(F));
 let fresnel:WaveComplex=[0,0],far:WaveComplex=[0,0],coarse:WaveComplex=[0,0],subdivisions=0;
 const contributions=slits.map(slit=>{
  const lower=slit.center-.5,upper=slit.center+.5,maxCoordinate=Math.max(Math.abs(lower),Math.abs(upper)),steps=Math.max(1,Math.ceil(2*Math.PI*(F*maxCoordinate+Math.abs(eta))/.5));
  const phase=(u:number)=>scale(exponential(Math.PI*F*u*u-2*Math.PI*eta*u+slit.phase),slit.amplitude);
  const integral=quadrature(phase,lower,upper,steps),refined=quadrature(phase,lower,upper,2*steps),fourier=scale(exponential(slit.phase-2*Math.PI*eta*slit.center),slit.amplitude*sinc(Math.PI*eta));
  const nearField=multiply(prefactor,refined),farField=multiply(prefactor,fourier);fresnel=sum(fresnel,nearField);far=sum(far,farField);coarse=sum(coarse,multiply(prefactor,integral));subdivisions+=2*steps;
  return {center:slit.center*w,width:w,amplitude:slit.amplitude,phase:slit.phase,fresnel:nearField,fraunhofer:farField};
 });
 const apertureHalfExtent=Math.max(0,...slits.filter(p=>p.amplitude>0).map(p=>Math.abs(p.center)+.5))*w,transverseExtent=Math.abs(x)+apertureHalfExtent,nearIntensity=norm(fresnel),farIntensity=norm(far),difference:WaveComplex=[fresnel[0]-far[0],fresnel[1]-far[1]];
 return {eta,x,distance,wavelength:s.wavelength,slitWidth:w,fresnelNumber:F,contributions,fresnel,fraunhofer:far,fresnelIntensity:nearIntensity,fraunhoferIntensity:farIntensity,
  scaledFresnelIntensity:nearIntensity/F,scaledFraunhoferIntensity:farIntensity/F,complexFieldDifference:Math.hypot(...difference),intensityDifference:nearIntensity-farIntensity,
  refinementDifference:Math.hypot(fresnel[0]-coarse[0],fresnel[1]-coarse[1]),quadratureSubdivisions:subdivisions,
  expectedTransmittedNorm:w*slits.reduce((out,p)=>out+p.amplitude**2,0),apertureHalfExtent,
  discardedAperturePhase:Math.PI*apertureHalfExtent**2/(s.wavelength*distance),maximumTransverseRatio:transverseExtent/distance,
  fourthOrderDistancePhaseEstimate:2*Math.PI/s.wavelength*transverseExtent**4/(8*distance**3),
  quadratureStatus:'refinement-comparison-not-certified-bound' as const};
}
