/** Exact smooth teaching families; equal-mass reversible labels and an open growth model. */
import {gaussianDiffusionAt,gaussianDiffusionWindow,type GaussianDiffusionParameters}from './learning-continuum-diffusion';
import {binaryProduct as product,binarySum as sum,binaryMultiply as multiply,binaryNegate as negative,binaryRatioObservation as ratio,binaryObservation as observe,type BinaryValue}from './learning-binary-products';
export type ReactionValue=number|null;
interface SignedLog{sign:number;log:number;binary?:{numerator:BinaryValue;denominator:BinaryValue};}
const zero:SignedLog={sign:0,log:0};
function bound(x:number,lo:number,hi:number,name:string){if(!Number.isFinite(x)||x<lo||x>hi)throw new RangeError(`Invalid ${name}`);}
function quantity(...xs:number[]):SignedLog{if(xs.includes(0))return zero;return{sign:xs.reduce((s,x)=>s*Math.sign(x),1),log:xs.reduce((s,x)=>s+Math.log(Math.abs(x)),0)};}
function binary(numerator:BinaryValue,denominator=product(1)):SignedLog{
 const q=ratio(numerator,denominator),magnitude=q.value===null?null:Math.abs(q.value);
 let log=q.logMagnitude??0;
 if(magnitude!==null&&magnitude>=2**-1022)log=Math.log(magnitude);
 if(magnitude!==null&&magnitude>=.5&&magnitude<=2){
  const absolute=q.sign<0?negative(numerator):numerator;
  const delta=ratio(sum([absolute,negative(denominator)]),denominator).value;
  if(delta!==null)log=Math.log1p(delta);
 }
 return{sign:q.sign,log,binary:{numerator,denominator}};
}
const times=(a:SignedLog,b:SignedLog):SignedLog=>a.sign===0||b.sign===0?zero:a.binary&&b.binary?binary(multiply(a.binary.numerator,b.binary.numerator),multiply(a.binary.denominator,b.binary.denominator)):{sign:a.sign*b.sign,log:a.log+b.log};
const shift=(q:SignedLog,log:number):SignedLog=>q.sign===0?zero:log===0?q:{sign:q.sign,log:q.log+log};
const negate=(q:SignedLog):SignedLog=>q.binary?binary(negative(q.binary.numerator),q.binary.denominator):{sign:-q.sign,log:q.log};
function add(...qs:SignedLog[]):SignedLog{
 let max=-Infinity;for(const q of qs)if(q.sign!==0)max=Math.max(max,q.log);if(max===-Infinity)return zero;
 const scaled=qs.reduce((s,q)=>q.sign===0?s:s+q.sign*Math.exp(q.log-max),0);return scaled===0?zero:{sign:Math.sign(scaled),log:max+Math.log(Math.abs(scaled))};
}
function value(q:SignedLog):ReactionValue{if(q.binary)return ratio(q.binary.numerator,q.binary.denominator).value;if(q.sign===0)return 0;const v=q.sign*Math.exp(q.log);return v===0||!Number.isFinite(v)?null:v;}
const logValue=(q:SignedLog)=>q.sign===0?null:q.log;
const observedSum=(...xs:ReactionValue[]):ReactionValue=>xs.includes(null)?null:(xs as number[]).reduce((a,b)=>a+b,0);
const observedNegate=(x:ReactionValue)=>x===null?null:-x;
export interface ReversibleKinetics{forwardRate:number;reverseRate:number;initialFractionA:number;}
export interface ReversibleTransport extends GaussianDiffusionParameters,ReversibleKinetics{}
function kinetics(p:ReversibleKinetics,time:number){
 bound(p.forwardRate,0,128,'forward rate');bound(p.reverseRate,0,128,'reverse rate');bound(p.initialFractionA,0,1,'initial fraction');bound(time,0,4,'time');
 const rawRate=sum([product(p.forwardRate),product(p.reverseRate)]),rawA=product(p.initialFractionA),rawB=sum([product(1),negative(rawA)]);
 const rawDerivative=sum([multiply(product(p.reverseRate),rawB),negative(multiply(product(p.forwardRate),rawA))]);
 const rawExponent=multiply(rawRate,product(time)),exponent=observe(rawExponent);
 const rate=observe(rawRate).value!,logRate=rate===0?null:Math.log(rate);
 const logExponent=exponent.logMagnitude,z=exponent.value??0;
 const logMix=logExponent===null?null:z===0?logExponent:Math.log(-Math.expm1(-z));
 const initialA=quantity(p.initialFractionA),initialB=p.initialFractionA===1?zero:{sign:1,log:Math.log1p(-p.initialFractionA)};
 const equilibriumA=rate===0?null:binary(product(p.reverseRate),rawRate);
 const equilibriumB=rate===0?null:binary(product(p.forwardRate),rawRate);
 const initialDerivative=binary(rawDerivative);
 if(logExponent===null||logExponent<-18){
  // Retain the first corrections before binary64 projection: e^-z can round
  // to one while these terms decide which side of a subnormal midpoint wins.
  // The fourth-order exponential remainder is < 7e-42 on this branch.
  const z2=multiply(rawExponent,rawExponent),z3=multiply(z2,rawExponent);
  const denominator=product(24),phi=sum([denominator,multiply(product(-12),rawExponent),multiply(product(4),z2),negative(z3)]);
  const mix=multiply(rawExponent,phi),decay=sum([denominator,negative(mix)]),fractionDenominator=multiply(rawRate,denominator);
  const a=rate===0?binary(rawA):binary(sum([multiply(multiply(rawRate,rawA),decay),multiply(product(p.reverseRate),mix)]),fractionDenominator);
  const b=rate===0?binary(rawB):binary(sum([multiply(multiply(rawRate,rawB),decay),multiply(product(p.forwardRate),mix)]),fractionDenominator);
  const derivative=binary(multiply(rawDerivative,decay),denominator),change=binary(multiply(multiply(rawDerivative,product(time)),phi),denominator);
  return{rate,logExponent,z,logMix,a,b,equilibriumA,equilibriumB,derivative,change};
 }
 const a=initialDerivative.sign===0?initialA:add(shift(initialA,-z),equilibriumA===null||logMix===null?zero:shift(equilibriumA,logMix));
 const b=initialDerivative.sign===0?initialB:add(shift(initialB,-z),equilibriumB===null||logMix===null?zero:shift(equilibriumB,logMix));
 const derivative=shift(initialDerivative,-z),change=logMix===null||logRate===null?zero:shift(initialDerivative,logMix-logRate);
 return{rate,logExponent,z,logMix,a,b,equilibriumA,equilibriumB,derivative,change};
}
export function reversibleReactionState(p:ReversibleKinetics,time:number){
 const k=kinetics(p,time),a=value(k.a),b=value(k.b),change=value(k.change);
 return{time,rate:k.rate,exponent:k.logExponent!==null&&k.z===0?null:k.z,
  fractionA:a,fractionB:b,logFractionA:logValue(k.a),logFractionB:logValue(k.b),
  equilibriumA:k.equilibriumA===null?null:value(k.equilibriumA),equilibriumB:k.equilibriumB===null?null:value(k.equilibriumB),
  logEquilibriumA:k.equilibriumA===null?null:logValue(k.equilibriumA),equilibriumStatus:k.rate===0?'nonunique-all-fractions-stationary' as const:'unique' as const,
  fractionRateA:value(k.derivative),fractionRateSign:k.derivative.sign,logFractionRateMagnitude:logValue(k.derivative),stableFractionChangeA:change,stableFractionChangeB:observedNegate(change),
  roundedFractionChangeA:a===null?null:a-p.initialFractionA,roundedComplementB:a===null?null:1-a,
  roundedFractionSum:observedSum(a,b)};
}
function reactionLogs(p:ReversibleTransport,x:number,time:number,k=kinetics(p,time)){
 const total=gaussianDiffusionAt(p,x,time),s=total.state,delta=x-s.mean,z=delta/s.sigma;
 const c=total.logConcentration===null?zero:{sign:1,log:total.logConcentration};
 const dx=times(c,quantity(-delta,1/s.variance)),dxx=times(c,quantity((z*z-1)/s.variance));
 const dt=add(times(negate(dx),quantity(p.velocity)),times(dxx,quantity(p.diffusivity)));
 const flux=add(times(c,quantity(p.velocity)),times(negate(dx),quantity(p.diffusivity)));
 const fluxGradient=add(times(dx,quantity(p.velocity)),times(negate(dxx),quantity(p.diffusivity)));
 const source=times(c,k.derivative);
 const a={concentration:times(c,k.a),dx:times(dx,k.a),dt:add(times(dt,k.a),source),flux:times(flux,k.a),fluxGradient:times(fluxGradient,k.a),source};
 const b={concentration:times(c,k.b),dx:times(dx,k.b),dt:add(times(dt,k.b),negate(source)),flux:times(flux,k.b),fluxGradient:times(fluxGradient,k.b),source:negate(source)};
 return{total,k,a,b,forward:times(a.concentration,quantity(p.forwardRate)),reverse:times(b.concentration,quantity(p.reverseRate))};
}
export function reversibleReactionAt(p:ReversibleTransport,x:number,time:number){
 const q=reactionLogs(p,x,time),state=reversibleReactionState(p,time);
 const species=(s:typeof q.a)=>{const dt=value(s.dt),div=value(s.fluxGradient),source=value(s.source);return{concentration:value(s.concentration),logConcentration:logValue(s.concentration),spatialDerivative:value(s.dx),timeDerivative:dt,flux:value(s.flux),fluxGradient:div,source,
  sourceSign:s.source.sign,logSourceMagnitude:logValue(s.source),balanceResidual:observedSum(dt,div,observedNegate(source)),omittedReactionResidual:observedSum(dt,div)};};
 const a=species(q.a),b=species(q.b),mass=binary(product(p.mass));
 return{x,time,state,total:q.total,a,b,forwardTurnover:value(q.forward),reverseTurnover:value(q.reverse),
  roundedTurnoverDifference:observedSum(value(q.reverse),observedNegate(value(q.forward))),
  globalMassA:value(times(mass,q.k.a)),globalMassB:value(times(mass,q.k.b)),globalMass:p.mass,
  globalSourceA:value(times(mass,q.k.derivative)),globalSourceB:value(negate(times(mass,q.k.derivative))),
  totalSource:0,roundedSourceSum:observedSum(a.source,b.source),roundedConcentrationSum:observedSum(a.concentration,b.concentration),
  massNormalizedFractionA:p.mass===0?null:state.fractionA,
  residualMeaning:'analytic-family-diagnostic-not-independent-validation' as const};
}
const nodes=[.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499];
const weights=[.1894506104550685,.1826034150449236,.16915651939500254,.14959598881657673,.12462897125553387,.09515851168249278,.06225352393864789,.027152459411754095];
function integrate(fn:(x:number)=>SignedLog,left:number,right:number,panels:number):SignedLog{
 const width=right-left,terms:SignedLog[]=[];
 for(let j=0;j<panels;j++)for(let i=0;i<nodes.length;i++)for(const sign of[-1,1]){
  const x=left+(j+.5+sign*.5*nodes[i])*width/panels;terms.push(shift(fn(x),Math.log(width)-Math.log(2*panels)+Math.log(weights[i])));
 }
 return add(...terms);
}
export function reversibleReactionWindow(p:ReversibleTransport,left:number,right:number,time:number){
 const total=gaussianDiffusionWindow(p,left,right,time),k=kinetics(p,time),mass=total.logMassInside===null?zero:{sign:1,log:total.logMassInside};
 const l=reactionLogs(p,left,time,k),r=reactionLogs(p,right,time,k);
 const species=(kind:'a'|'b')=>{
  const fraction=k[kind],source=times(mass,kind==='a'?k.derivative:negate(k.derivative));
  const rate=integrate(x=>times(reactionLogs(p,x,time,k)[kind].dt,quantity(p.area)),left,right,total.panels);
  const net=times(add(r[kind].flux,negate(l[kind].flux)),quantity(p.area));
  const observedRate=value(rate),observedNet=value(net),observedSource=value(source);
  return{massInside:value(times(mass,fraction)),massRate:observedRate,netOutflow:observedNet,integratedSource:observedSource,
   balanceResidual:observedSum(observedRate,observedNet,observedNegate(observedSource)),omittedReactionResidual:observedSum(observedRate,observedNet)};
 };
 const a=species('a'),b=species('b');
 return{left,right,time,total,a,b,totalIntegratedSource:0,roundedSpeciesMass:observedSum(a.massInside,b.massInside),roundedIntegratedSourceSum:observedSum(a.integratedSource,b.integratedSource)};
}

export interface FisherFrontParameters{diffusivity:number;growthRate:number;carryingConcentration:number;bulkVelocity:number;initialFront:number;area:number;}
function front(p:FisherFrontParameters,time:number){
 bound(p.diffusivity,.0001,4,'positive diffusivity');bound(p.growthRate,.0001,4,'positive growth rate');
 bound(p.carryingConcentration,.25,4,'carrying concentration');bound(p.bulkVelocity,-2,2,'bulk velocity');bound(p.initialFront,-2,2,'initial front');bound(p.area,.25,4,'area');bound(time,0,4,'time');
 const lambda=Math.sqrt(p.growthRate/(6*p.diffusivity)),relativeSpeed=5*Math.sqrt(p.diffusivity*p.growthRate/6),laboratorySpeed=p.bulkVelocity+relativeSpeed;
 const displacement=value(quantity(laboratorySpeed,time));
 return{lambda,relativeSpeed,laboratorySpeed,center:p.initialFront+(displacement??0),stableDisplacement:displacement,stableRelativeDisplacement:value(quantity(relativeSpeed,time))};
}
const softplus=(x:number)=>x>0?x+Math.log1p(Math.exp(-x)):Math.log1p(Math.exp(x));
function frontLogs(p:FisherFrontParameters,x:number,time:number,f=front(p,time)){
 bound(x,-16,16,'x');const z=f.lambda*(x-f.center),logW=-softplus(z),logS=-softplus(-z),w=Math.exp(logW),s=Math.exp(logS);
 const c={sign:1,log:Math.log(p.carryingConcentration)+2*logW},deficit={sign:1,log:Math.log(p.carryingConcentration)+logS+Math.log1p(w)};
 const dx=shift(times(c,quantity(-2,f.lambda)),logS),dxx=shift(times(c,quantity(2,f.lambda,f.lambda,3*s-1)),logS);
 const dt=times(dx,quantity(-f.laboratorySpeed)),source=shift(c,Math.log(p.growthRate)+logS+Math.log1p(w));
 const advection=times(c,quantity(p.bulkVelocity)),diffusion=times(negate(dx),quantity(p.diffusivity));
 const fluxGradient=add(times(dx,quantity(p.bulkVelocity)),times(negate(dxx),quantity(p.diffusivity)));
 return{f,z,logW,logS,c,deficit,dx,dxx,dt,source,advection,diffusion,flux:add(advection,diffusion),fluxGradient};
}
export function fisherFrontAt(p:FisherFrontParameters,x:number,time:number){
 const q=frontLogs(p,x,time),c=q.z<0?p.carryingConcentration*Math.exp(2*q.logW):value(q.c),dt=value(q.dt),div=value(q.fluxGradient),source=value(q.source);
 return{x,time,...q.f,phase:q.z,concentration:c,logConcentration:q.c.log,stableDeficit:value(q.deficit),logDeficit:q.deficit.log,
  roundedDeficit:c===null?null:p.carryingConcentration-c,spatialDerivative:value(q.dx),secondDerivative:value(q.dxx),timeDerivative:dt,
  advectiveFlux:value(q.advection),diffusiveFlux:value(q.diffusion),flux:value(q.flux),fluxGradient:div,source,logSource:q.source.log,
  roundedLogisticSource:c===null?null:p.growthRate*c*(1-c/p.carryingConcentration),
  balanceResidual:observedSum(dt,div,observedNegate(source)),omittedSourceResidual:observedSum(dt,div),
  globalMassStatus:'diverges-on-entire-line' as const,residualMeaning:'analytic-family-diagnostic-not-independent-validation' as const};
}
export function fisherLevelPosition(p:FisherFrontParameters,level:number,time:number){
 const f=front(p,time);bound(level,.01,.99,'relative concentration level');
 return f.center+Math.log(1/Math.sqrt(level)-1)/f.lambda;
}
export function fisherFrontWindow(p:FisherFrontParameters,left:number,right:number,time:number){
 const f=front(p,time);bound(left,-16,16,'left');bound(right,-16,16,'right');if(!(right>left))throw new RangeError('Expected left < right');
 const panels=Math.max(1,Math.ceil((right-left)*f.lambda/2));
 const observe=(key:'c'|'dt'|'source')=>integrate(x=>times(frontLogs(p,x,time,f)[key],quantity(p.area)),left,right,panels);
 const mass=observe('c'),rate=observe('dt'),source=observe('source'),l=frontLogs(p,left,time,f),r=frontLogs(p,right,time,f),net=times(add(r.flux,negate(l.flux)),quantity(p.area));
 return{left,right,time,panels,massInside:value(mass),logMassInside:logValue(mass),massRate:value(rate),netOutflow:value(net),integratedSource:value(source),logIntegratedSource:logValue(source),
  balanceResidual:observedSum(value(rate),value(net),observedNegate(value(source))),omittedSourceResidual:observedSum(value(rate),value(net)),
  integrationMeaning:'physical-window-quadrature-not-a-certified-error-bound' as const};
}
