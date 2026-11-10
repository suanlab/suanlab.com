import {binaryProduct,binarySum,binaryMultiply,binaryNegate,binaryRatioObservation,binaryZero,type BinaryValue} from './learning-binary-products';
/** Bounded teaching models. Particle smoothing is not a collisional fluid solver. */
export interface BallisticParticle { mass:number; position:number; velocity:number; }
function bounded(value:number,min:number,max:number,name:string) {
  if (!Number.isFinite(value)||value<min||value>max) throw new RangeError(`Invalid ${name}`);
}
function bandwidth(h:number){bounded(h,.02,2,'bandwidth');}
function particlesValid(particles:readonly BallisticParticle[]) {
  if(!Array.isArray(particles)||particles.length<1||particles.length>64)throw new RangeError('Expected 1–64 particles');
  for(let i=0;i<particles.length;i++) {
    const p=particles[i];if(!p)throw new RangeError('Missing particle');
    bounded(p.mass,.1,4,'particle mass');bounded(p.position,-4,4,'initial position');bounded(p.velocity,-2,2,'particle velocity');
  }
}
/** Rational kernel evaluations retain tiny derivatives before projection. */
function kernelBinary(displacement:number,h:number) {
  bounded(displacement,-16,16,'displacement');bandwidth(h);
  const denominator=binaryProduct(16,h,h,h,h,h),spatialDenominator=binaryProduct(4,h,h,h,h,h),widthDenominator=binaryProduct(16,h,h,h,h,h,h);
  if(Math.abs(displacement)>=h)return{value:binaryZero,spatial:binaryZero,width:binaryZero,denominator,spatialDenominator,widthDenominator};
  const b=binarySum([binaryProduct(h,h),binaryProduct(-1,displacement,displacement)]);
  const b5=binarySum([binaryProduct(h,h),binaryProduct(-5,displacement,displacement)]);
  return{value:binaryMultiply(binaryProduct(15),binaryMultiply(b,b)),
    spatial:binaryMultiply(binaryProduct(-15,displacement),b),width:binaryMultiply(binaryProduct(-15),binaryMultiply(b,b5)),
    denominator,spatialDenominator,widthDenominator};
}
function kernelProjection(raw:ReturnType<typeof kernelBinary>){
  const observations={value:binaryRatioObservation(raw.value,raw.denominator),spatialDerivative:binaryRatioObservation(raw.spatial,raw.spatialDenominator),bandwidthDerivative:binaryRatioObservation(raw.width,raw.widthDenominator)};
  return{value:observations.value.value!,spatialDerivative:observations.spatialDerivative.value,bandwidthDerivative:observations.bandwidthDerivative.value,observations};
}
/** Normalized compact quartic kernel; C1 at its support boundary (not C2). */
export function compactMassKernel(displacement:number,h:number) {
  const kernel=kernelProjection(kernelBinary(displacement,h));
  return {value:kernel.value,spatialDerivative:kernel.spatialDerivative,bandwidthDerivative:kernel.bandwidthDerivative};
}
export function compactMassKernelObservations(displacement:number,h:number) {return kernelProjection(kernelBinary(displacement,h)).observations;}
export function coarseGrainedParticleState(particles:readonly BallisticParticle[],x:number,time:number,h:number,bandwidthRate=0) {
  particlesValid(particles);bounded(x,-8,8,'observation position');bounded(time,-1,1,'time');bandwidth(h);bounded(bandwidthRate,-.5,.5,'bandwidth rate');
  const contributions=particles.map(p=>{
    const position=p.position+p.velocity*time,raw=kernelBinary(x-position,h),kernel=kernelProjection(raw);
    const weight=binaryRatioObservation(binaryMultiply(binaryProduct(p.mass),raw.value),raw.denominator).value!;
    return{position,mass:p.mass,velocity:p.velocity,weight,kernel,raw};
  });
  const accumulate=(get:(p:typeof contributions[number])=>BinaryValue)=>binarySum(contributions.map(get));
  const densityNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass),p.raw.value));
  const fluxNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass,p.velocity),p.raw.value));
  const momentumNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass,p.velocity,p.velocity),p.raw.value));
  const densityXNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass),p.raw.spatial));
  const fluxXNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass,p.velocity),p.raw.spatial));
  const momentumXNumerator=accumulate(p=>binaryMultiply(binaryProduct(p.mass,p.velocity,p.velocity),p.raw.spatial));
  const smoothingNumerator=accumulate(p=>binaryMultiply(binaryProduct(bandwidthRate,p.mass),p.raw.width));
  const momentumSmoothingNumerator=accumulate(p=>binaryMultiply(binaryProduct(bandwidthRate,p.mass,p.velocity),p.raw.width));
  const spatialScale=binaryProduct(4,h),den=contributions[0].raw;
  const densityTNumerator=binarySum([binaryNegate(binaryMultiply(spatialScale,fluxXNumerator)),smoothingNumerator]);
  const momentumTNumerator=binarySum([binaryNegate(binaryMultiply(spatialScale,momentumXNumerator)),momentumSmoothingNumerator]);
  const observed=densityNumerator.coefficient!==binaryZero.coefficient;
  const velocityObservation=observed?binaryRatioObservation(fluxNumerator,densityNumerator):null;
  const pairs:BinaryValue[]=[];
  for(let i=0;i<contributions.length;i++)for(let j=i+1;j<contributions.length;j++){
    const a=contributions[i],b=contributions[j];
    const jump=binarySum([binaryProduct(a.velocity),binaryProduct(-b.velocity)]);
    pairs.push(binaryMultiply(binaryProduct(a.mass,b.mass),binaryMultiply(binaryMultiply(a.raw.value,b.raw.value),binaryMultiply(jump,jump))));
  }
  const meanTransportObservation=observed?binaryRatioObservation(binaryMultiply(fluxNumerator,fluxNumerator),binaryMultiply(den.denominator,densityNumerator)):null;
  const dispersionObservation=observed?binaryRatioObservation(binarySum(pairs),binaryMultiply(den.denominator,densityNumerator)):null;
  const observations={density:binaryRatioObservation(densityNumerator,den.denominator),flux:binaryRatioObservation(fluxNumerator,den.denominator),velocity:velocityObservation,
    momentumFlux:binaryRatioObservation(momentumNumerator,den.denominator),meanMomentumTransport:meanTransportObservation,dispersion:dispersionObservation,densityX:binaryRatioObservation(densityXNumerator,den.spatialDenominator),
    densityT:binaryRatioObservation(densityTNumerator,den.widthDenominator),fluxX:binaryRatioObservation(fluxXNumerator,den.spatialDenominator),smoothingRate:binaryRatioObservation(smoothingNumerator,den.widthDenominator),
    continuityResidual:binaryRatioObservation(binarySum([densityTNumerator,binaryMultiply(spatialScale,fluxXNumerator)]),den.widthDenominator),momentumT:binaryRatioObservation(momentumTNumerator,den.widthDenominator),
    momentumFluxX:binaryRatioObservation(momentumXNumerator,den.spatialDenominator),momentumSmoothingRate:binaryRatioObservation(momentumSmoothingNumerator,den.widthDenominator),
    momentumResidual:binaryRatioObservation(binarySum([momentumTNumerator,binaryMultiply(spatialScale,momentumXNumerator)]),den.widthDenominator)};
  return {x,time,bandwidth:h,bandwidthRate,density:observations.density.value!,flux:observations.flux.value,
    velocity:velocityObservation?.value??null,meanMomentumTransport:meanTransportObservation?.value??null,momentumFlux:observations.momentumFlux.value,dispersion:dispersionObservation?.value??null,
    densityX:observations.densityX.value,densityT:observations.densityT.value,fluxX:observations.fluxX.value,
    smoothingRate:observations.smoothingRate.value,continuityResidual:observations.continuityResidual.value,
    momentumT:observations.momentumT.value,momentumFluxX:observations.momentumFluxX.value,
    momentumSmoothingRate:observations.momentumSmoothingRate.value,momentumResidual:observations.momentumResidual.value,
    observations,contributions:contributions.map(p=>({position:p.position,mass:p.mass,velocity:p.velocity,weight:p.weight,kernel:p.kernel})),status:observed?'observed' as const:'empty-support' as const,
    arithmeticStatus:Object.values(observations).some(q=>q?.sign!==0&&q?.value===null)?'linear-value-unavailable' as const:'finite' as const};
}
/** Exact kernel integral on a finite interval, using a centred polynomial rather than CDF subtraction. */
export function coarseGrainedIntervalMass(particles:readonly BallisticParticle[],left:number,right:number,time:number,h:number) {
  particlesValid(particles);bounded(left,-8,8,'left boundary');bounded(right,-8,8,'right boundary');bounded(time,-1,1,'time');bandwidth(h);
  if(right<left||right>left&&right-left<1e-8)throw new RangeError('Reversed or sub-range interval');
  if(right===left)return{mass:0,meanDensity:null,width:0};
  const mass=particles.reduce((sum,p)=>{
    const position=p.position+p.velocity*time;
    const a=Math.max(-1,(left-position)/h),b=Math.min(1,(right-position)/h);
    if(a>=b)return sum;
    const midpoint=(a+b)/2,half=(b-a)/2,base=(1-midpoint)*(1+midpoint);
    return sum+p.mass*15/16*2*half*(base*base+half*half*(2*midpoint*midpoint-2/3)+half**4/5);
  },0);
  return{mass,meanDensity:mass/(right-left),width:right-left};
}
export interface ContinuumScaleInput {
  /** Number density /m^3, mean free path m, macroscopic variation length m. */
  numberDensity:number;meanFreePath:number;macroLength:number;averagingLength:number;
  /** User-selected tolerances, not universal fluid validity thresholds. */
  relativeFluctuationTolerance:number;relativeResolutionTolerance:number;collisionSeparationFactor:number;
}
export function continuumScaleWindow(input:ContinuumScaleInput) {
  const {numberDensity:n,meanFreePath:lambda,macroLength:L,averagingLength:ell,
    relativeFluctuationTolerance:epsilon,relativeResolutionTolerance:eta,collisionSeparationFactor:factor}=input;
  bounded(n,1,1e30,'number density');bounded(lambda,1e-12,1e3,'mean free path');bounded(L,1e-9,1e6,'macro length');bounded(ell,1e-12,1e6,'averaging length');
  bounded(epsilon,1e-4,1,'fluctuation tolerance');bounded(eta,1e-4,.5,'resolution tolerance');bounded(factor,1,1e3,'collision separation factor');
  // Homogeneous Poisson count benchmark in a cube, not a universal correlated-fluid law.
  const expectedCount=n*ell**3,relativeCountStd=1/Math.sqrt(expectedCount),knudsen=lambda/L;
  const countLowerLength=Math.cbrt(1/n)*Math.pow(epsilon,-2/3),collisionLowerLength=factor*lambda,upperLength=eta*L;
  const lowerLength=Math.max(countLowerLength,collisionLowerLength);
  const windowExists=lowerLength<=upperLength;
  const countSatisfied=ell>=countLowerLength,collisionSatisfied=ell>=collisionLowerLength,resolutionSatisfied=ell<=upperLength;
  return{...input,expectedCount,relativeCountStd,knudsen,countLowerLength,collisionLowerLength,lowerLength,upperLength,
    windowExists,selectedSatisfies:countSatisfied&&collisionSatisfied&&resolutionSatisfied,countSatisfied,collisionSatisfied,resolutionSatisfied,
    ratioToMacro:ell/L,ratioToMeanFreePath:ell/lambda,meanSpacing:Math.cbrt(1/n),
    // This is only the intersection of declared proxy inequalities.
    status:windowExists?'declared-window-exists' as const:'no-declared-window' as const};
}
/** Fixed N uniform independent particles: binomial correction to the Poisson benchmark. */
export function independentCountFluctuations(totalCount:number,volumeFraction:number) {
  bounded(totalCount,1,1e12,'total particle count');if(!Number.isInteger(totalCount))throw new RangeError('Integer count required');
  bounded(volumeFraction,0,1,'volume fraction');
  const mean=totalCount*volumeFraction,variance=mean*(1-volumeFraction);
  return {mean,variance,relativeStd:mean===0?null:Math.sqrt(variance)/mean,
    poissonRelativeStd:mean===0?null:1/Math.sqrt(mean)};
}
