/** Constant-coefficient teaching families, not variable-material or wall solvers. */
export type DiffusionValue = number | null;
interface SignedLog { sign: number; log: number; }
const zero: SignedLog = {sign:0,log:0};
function bound(v:number,lo:number,hi:number,name:string){if(!Number.isFinite(v)||v<lo||v>hi)throw new RangeError(`Invalid ${name}`);}
function quantity(...factors:number[]):SignedLog{
  if(factors.includes(0))return zero;
  return{sign:factors.reduce((s,x)=>s*Math.sign(x),1),log:factors.reduce((s,x)=>s+Math.log(Math.abs(x)),0)};
}
const shift=(q:SignedLog,log:number):SignedLog=>q.sign===0?zero:{sign:q.sign,log:q.log+log};
const negate=(q:SignedLog):SignedLog=>({sign:-q.sign,log:q.log});
const times=(a:SignedLog,b:SignedLog):SignedLog=>a.sign===0||b.sign===0?zero:{sign:a.sign*b.sign,log:a.log+b.log};
function add(...qs:SignedLog[]):SignedLog{
  const nonzero=qs.filter(q=>q.sign!==0);if(!nonzero.length)return zero;
  let max=-Infinity;for(const q of nonzero)max=Math.max(max,q.log);
  const scaled=nonzero.reduce((s,q)=>s+q.sign*Math.exp(q.log-max),0);
  return scaled===0?zero:{sign:Math.sign(scaled),log:max+Math.log(Math.abs(scaled))};
}
function value(q:SignedLog):DiffusionValue{
  if(q.sign===0)return 0;const v=q.sign*Math.exp(q.log);return v===0||!Number.isFinite(v)?null:v;
}
const observedSum=(...vs:DiffusionValue[]):DiffusionValue=>vs.includes(null)?null:(vs as number[]).reduce((a,b)=>a+b,0);
/** log(abs(expm1(-z))) retains a positive under-range z in logarithmic form. */
function decayState(logExponent:number|null){
  if(logExponent===null)return{exponent:0 as DiffusionValue,roundedFactor:1,logChange:null as number|null};
  const z=Math.exp(logExponent),exponent=z===0?null:z;
  return{exponent,roundedFactor:Math.exp(-z),logChange:z===0?logExponent:Math.log(-Math.expm1(-z))};
}

export interface HeatModeParameters{
  heatCapacity:number;conductivity:number;meanTemperature:number;contrast:number;waveNumber:number;phase:number;area:number;
}
function validHeat(p:HeatModeParameters){
  bound(p.heatCapacity,.25,4,'volumetric heat capacity');bound(p.conductivity,0,4,'conductivity');
  bound(p.meanTemperature,250,400,'mean absolute temperature');bound(p.contrast,-.2,.2,'temperature contrast');
  bound(p.waveNumber,.25,4,'wave number');bound(p.phase,-Math.PI,Math.PI,'phase');bound(p.area,.25,4,'cross-sectional area');
}
function heatFamily(p:HeatModeParameters,time:number){
  validHeat(p);bound(time,0,4,'time');
  const logRate=p.conductivity===0?null:Math.log(p.conductivity)-Math.log(p.heatCapacity)+2*Math.log(p.waveNumber);
  const logExponent=logRate===null||time===0?null:logRate+Math.log(time),decay=decayState(logExponent);
  const initial=quantity(p.meanTemperature,p.contrast),amplitude=shift(initial,-(decay.exponent??0));
  const amplitudeChange=decay.logChange===null?zero:negate(shift(initial,decay.logChange));
  return{logRate,logExponent,decay,initial,amplitude,amplitudeChange};
}
export function heatModeAt(p:HeatModeParameters,x:number,time:number){
  const f=heatFamily(p,time);bound(x,-16,16,'x');const phase=p.waveNumber*x+p.phase;
  const perturbation=times(f.amplitude,quantity(Math.cos(phase)));
  const gradient=times(f.amplitude,quantity(-p.waveNumber,Math.sin(phase)));
  const laplacian=times(perturbation,quantity(-p.waveNumber,p.waveNumber));
  const timeDerivative=f.logRate===null?zero:negate(shift(perturbation,f.logRate));
  const heatFlux=times(negate(gradient),quantity(p.conductivity));
  const energyRate=times(timeDerivative,quantity(p.heatCapacity));
  const fluxGradient=times(negate(laplacian),quantity(p.conductivity));
  const theta=value(perturbation),rate=value(energyRate),divergence=value(fluxGradient);
  // Validated |contrast| <= .2 bounds |theta| <= 80 K: an unavailable
  // perturbation can only be below range, far below one ulp of 250–400 K.
  // Keep its null diagnostic while independently projecting the finite total.
  const temperature=p.meanTemperature+(theta??0);
  return{x,time,phase,temperature,roundedTemperature:temperature,perturbation:theta,
    gradient:value(gradient),laplacian:value(laplacian),timeDerivative:value(timeDerivative),heatFlux:value(heatFlux),
    volumetricEnergyRate:rate,fluxGradient:divergence,localBalanceResidual:observedSum(rate,divergence),
    amplitude:value(f.amplitude),amplitudeChange:value(f.amplitudeChange),decayFactor:f.decay.roundedFactor,
    exponent:f.decay.exponent,decayRate:f.logRate===null?0:value({sign:1,log:f.logRate}),stableRelativeDecayChange:f.decay.logChange===null?0:value({sign:-1,log:f.decay.logChange}),
    temperatureLowerBound:p.meanTemperature*(1-Math.abs(p.contrast)),
    residualMeaning:'analytic-family-diagnostic-not-independent-validation' as const};
}
/** Sensible-energy reference is explicitly 273.15 K; the quadratic norm has different units. */
export function heatModeCell(p:HeatModeParameters,time:number){
  const f=heatFamily(p,time),length=2*Math.PI/p.waveNumber,volume=p.area*length;
  const norm=shift(times(f.amplitude,f.amplitude),Math.log(volume/4));
  const normRate=f.logRate===null?zero:negate(shift(norm,f.logRate+Math.log(2)));
  const normDecay=decayState(f.logExponent===null?null:f.logExponent+Math.log(2));
  const initialNorm=shift(times(f.initial,f.initial),Math.log(volume/4));
  return{length,volume,referenceTemperature:273.15,sensibleEnergy:p.heatCapacity*volume*(p.meanTemperature-273.15),
    perturbationHeat:0,sensibleEnergyRate:0,quadraticNorm:value(norm),quadraticNormRate:value(normRate),
    gradientNormDissipation:value(negate(normRate)),stableQuadraticNormChange:value(normDecay.logChange===null?zero:negate(shift(initialNorm,normDecay.logChange)))};
}
/** Symmetric finite physical window; stable trigonometric identities preserve its specified width. */
export function heatModeWindow(p:HeatModeParameters,center:number,halfWidth:number,time:number){
  const f=heatFamily(p,time);bound(center,-4,4,'window center');bound(halfWidth,0,2,'window half-width');
  const h=p.waveNumber*halfWidth,sinc=h===0?1:Math.sin(h)/h,phase=p.waveNumber*center+p.phase;
  const integralFactor=quantity(2,halfWidth,sinc,Math.cos(phase),p.area,p.heatCapacity);
  const modeHeat=times(f.amplitude,integralFactor),energyRate=f.logRate===null?zero:negate(shift(modeHeat,f.logRate));
  const left=center-halfWidth,right=center+halfWidth;
  const power=(x:number)=>times(f.amplitude,quantity(p.area,p.conductivity,p.waveNumber,Math.sin(p.waveNumber*x+p.phase)));
  const leftPower=value(negate(power(left))),rightPower=value(power(right));
  const stableNetPower=value(negate(energyRate)),rate=value(energyRate),baseline=quantity(p.heatCapacity,p.area,2,halfWidth,p.meanTemperature-273.15);
  return{center,halfWidth,left,right,time,referenceTemperature:273.15,modeHeat:value(modeHeat),
    sensibleEnergy:value(add(baseline,modeHeat)),energyRate:rate,stableNetPower,
    roundedLeftOutwardPower:leftPower,roundedRightOutwardPower:rightPower,
    roundedFaceNetPower:halfWidth===0?0:observedSum(leftPower,rightPower),analyticBalanceResidual:observedSum(rate,stableNetPower),
    stableModeHeatChange:value(times(f.amplitudeChange,integralFactor)),
    omittedBoundaryFluxResidual:rate,geometryStatus:halfWidth===0?'zero-width' as const:left===right?'endpoints-round-together' as const:'finite-window' as const};
}

export interface GaussianDiffusionParameters{
  mass:number;area:number;initialSigma:number;initialMean:number;velocity:number;diffusivity:number;
}
function validGaussian(p:GaussianDiffusionParameters){
  bound(p.mass,0,4,'mass');bound(p.area,.25,4,'area');bound(p.initialSigma,.125,2,'initial standard deviation');
  bound(p.initialMean,-2,2,'initial mean');bound(p.velocity,-2,2,'velocity');bound(p.diffusivity,0,1,'diffusivity');
}
export function gaussianDiffusionState(p:GaussianDiffusionParameters,time:number){
  validGaussian(p);bound(time,0,4,'time');const initialVariance=p.initialSigma**2,delta=quantity(2,p.diffusivity,time);
  const varianceChange=value(delta),variance=initialVariance+(varianceChange??0),sigma=Math.sqrt(variance),meanChange=value(quantity(p.velocity,time));
  const mean=p.initialMean+(meanChange??0),etaLog=delta.sign===0?null:delta.log-Math.log(initialVariance);
  const root=etaLog===null?1:Math.sqrt(1+Math.exp(etaLog));
  const relativePeakChange=etaLog===null?zero:{sign:-1,log:etaLog-Math.log(root)-Math.log(root+1)};
  const logInitialPeak=p.mass===0?null:Math.log(p.mass)-Math.log(p.area)-Math.log(p.initialSigma)-.5*Math.log(2*Math.PI);
  return{time,mean,variance,sigma,initialVariance,varianceChange,stableMeanChange:meanChange,
    stableSigmaChange:value(delta.sign===0?zero:shift(delta,-Math.log(sigma+p.initialSigma))),
    stableRelativePeakChange:value(relativePeakChange),
    stablePeakChange:logInitialPeak===null?0:value(shift(relativePeakChange,logInitialPeak)),
    roundedVarianceChange:variance-initialVariance,roundedSigmaChange:sigma-p.initialSigma,
    declaredShapeMean:mean,declaredShapeVariance:variance,massWeightedMean:p.mass===0?null:mean,massWeightedVariance:p.mass===0?null:variance,
    normalizationStatus:p.mass===0?'zero-mass-no-normalized-moments' as const:'positive-mass' as const};
}
function gaussianLogs(p:GaussianDiffusionParameters,x:number,time:number){
  const state=gaussianDiffusionState(p,time);bound(x,-16,16,'x');const delta=x-state.mean,z=delta/state.sigma;
  const c=p.mass===0?zero:{sign:1,log:Math.log(p.mass)-Math.log(p.area)-Math.log(state.sigma)-.5*Math.log(2*Math.PI)-z*z/2};
  const dx=times(c,quantity(-delta,1/state.variance));
  const dxx=times(c,quantity((z*z-1)/state.variance));
  const advection=times(c,quantity(p.velocity)),diffusion=times(negate(dx),quantity(p.diffusivity));
  const dt=add(times(negate(dx),quantity(p.velocity)),times(dxx,quantity(p.diffusivity)));
  const fluxGradient=add(times(dx,quantity(p.velocity)),times(negate(dxx),quantity(p.diffusivity)));
  return{state,x,time,z,c,dx,dxx,dt,advection,diffusion,flux:add(advection,diffusion),fluxGradient};
}
export function gaussianDiffusionAt(p:GaussianDiffusionParameters,x:number,time:number){
  const q=gaussianLogs(p,x,time),dt=value(q.dt),div=value(q.fluxGradient);
  return{x,time,z:q.z,state:q.state,concentration:value(q.c),logConcentration:q.c.sign===0?null:q.c.log,
    spatialDerivative:value(q.dx),secondDerivative:value(q.dxx),timeDerivative:dt,
    advectiveFlux:value(q.advection),diffusiveFlux:value(q.diffusion),flux:value(q.flux),fluxGradient:div,
    localBalanceResidual:observedSum(dt,div),residualMeaning:'analytic-family-diagnostic-not-independent-validation' as const};
}
const x16=[.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499];
const w16=[.1894506104550685,.1826034150449236,.16915651939500254,.14959598881657673,.12462897125553387,.09515851168249278,.06225352393864789,.027152459411754095];
const x8=[.1834346424956498,.525532409916329,.7966664774136267,.9602898564975363];
const w8=[.362683783378362,.3137066458778873,.2223810344533745,.1012285362903763];
/** Physical-width quadrature in log space; never subtract nearly equal CDF values. */
export function gaussianDiffusionWindow(p:GaussianDiffusionParameters,left:number,right:number,time:number){
  const state=gaussianDiffusionState(p,time);bound(left,-16,16,'left boundary');bound(right,-16,16,'right boundary');
  if(!(right>left))throw new RangeError('Expected left < right');
  const width=right-left,closest=state.mean<left?left-state.mean:state.mean>right?state.mean-right:0;
  // Tail-dominated intervals change on the shorter sigma/|z| scale. Never clip them.
  const panels=Math.max(1,Math.ceil(width/state.sigma),Math.ceil(width/state.sigma*(closest/state.sigma)/4)),step=width/panels;
  const logNormalization=-Math.log(state.sigma)-.5*Math.log(2*Math.PI),inverseVariance=1/state.variance;
  const integrate=(nodes:number[],weights:number[],kind:'fraction'|'rate'):SignedLog=>{
    if(kind==='rate'&&p.mass===0)return zero;
    const terms:SignedLog[]=[];
    for(let j=0;j<panels;j++)for(let i=0;i<nodes.length;i++)for(const sign of[-1,1]){
      const x=left+(j+.5+sign*.5*nodes[i])*step,delta=x-state.mean,z=delta/state.sigma,logPdf=logNormalization-z*z/2;
      const integrand=kind==='fraction'?{sign:1,log:logPdf}:times({sign:1,log:Math.log(p.mass)+logPdf},add(quantity(p.velocity,delta,inverseVariance),quantity(p.diffusivity,(z*z-1)*inverseVariance)));
      terms.push(shift(integrand,Math.log(width)-Math.log(2*panels)+Math.log(weights[i])));
    }
    return add(...terms);
  };
  const fraction=integrate(x16,w16,'fraction'),coarse=integrate(x8,w8,'fraction'),mass=times(fraction,quantity(p.mass)),rate=integrate(x16,w16,'rate');
  const l=gaussianLogs(p,left,time),r=gaussianLogs(p,right,time),net=times(add(r.flux,negate(l.flux)),quantity(p.area));
  return{left,right,time,panels,fraction:value(fraction),logFraction:fraction.log,massInside:value(mass),logMassInside:mass.sign===0?null:mass.log,
    massRate:value(rate),leftOutwardFlux:value(times(negate(l.flux),quantity(p.area))),rightOutwardFlux:value(times(r.flux,quantity(p.area))),netOutflow:value(net),
    balanceResidual:observedSum(value(rate),value(net)),
    logQuadratureDifference:Math.abs(fraction.log-coarse.log),
    quadratureMeaning:'order-comparison-diagnostic-not-certified-error-bound' as const};
}
