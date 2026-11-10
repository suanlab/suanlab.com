/** Declared linear thermoelastic mode and steady constant-property thermoelectric bar. */
import { binaryProduct, binaryMultiply, binarySum, binaryNegate, binaryRatioObservation, type BinaryValue } from './learning-binary-products';
export interface MultiphysicsObservation { value: number | null; sign: number; logMagnitude: number | null }
interface Q { sign: number; log: number }
type Vector = [Q,Q,Q];
type Matrix = [Vector,Vector,Vector];
const zero: Q = {sign:0,log:0};
const ln2=Math.log(2);
function q(...factors:number[]):Q {
  return factors.includes(0)?zero:{sign:factors.reduce((a,v)=>a*Math.sign(v),1),log:factors.reduce((a,v)=>a+Math.log(Math.abs(v)),0)};
}
const neg=(a:Q):Q=>a.sign===0?zero:{sign:-a.sign,log:a.log};
const shift=(a:Q,log:number):Q=>a.sign===0?zero:{sign:a.sign,log:a.log+log};
const mul=(a:Q,b:Q):Q=>a.sign===0||b.sign===0?zero:{sign:a.sign*b.sign,log:a.log+b.log};
function add(...terms:Q[]):Q {
  let result=zero;
  for(const a of terms){
    if(a.sign===0)continue;
    if(result.sign===0){result=a;continue;}
    const high=result.log>=a.log?result:a,low=result.log>=a.log?a:result;
    if(high.sign===low.sign)result={sign:high.sign,log:high.log+Math.log1p(Math.exp(low.log-high.log))};
    else if(high.log===low.log)result=zero;
    else result={sign:high.sign,log:high.log+Math.log(-Math.expm1(low.log-high.log))};
  }
  return result;
}
function observe(a:Q):MultiphysicsObservation {
  if(a.sign===0)return{value:0,sign:0,logMagnitude:null};
  const n=a.log<=Math.log(Number.MIN_VALUE)-ln2?0:a.sign*Math.exp(a.log);
  return{value:n===0||!Number.isFinite(n)?null:n,sign:a.sign,logMagnitude:a.log};
}
function bound(v:number,lo:number,hi:number,name:string){if(!Number.isFinite(v)||v<lo||v>hi)throw new RangeError(`Invalid ${name}`);}
const mat=(f:(i:number,j:number)=>Q):Matrix=>Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>f(i,j))) as Matrix;
const matrixProduct=(a:Matrix,b:Matrix)=>mat((i,j)=>add(...[0,1,2].map(k=>mul(a[i][k],b[k][j]))));
const apply=(a:Matrix,b:Vector):Vector=>a.map(row=>add(...row.map((v,j)=>mul(v,b[j])))) as Vector;
/** Directly propagate exp(Bt)-I, retaining tiny off-diagonal and diagonal changes. */
function exponentialPair(B:Matrix,time:Q) {
  if(time.sign===0)return{propagator:mat((i,j)=>i===j?q(1):zero),change:mat(()=>zero)};
  const numericNorm=Math.max(...B.map(row=>row.reduce((s,x)=>s+(observe(x).value===null?0:Math.abs(observe(x).value!)),0)));
  const z=numericNorm*(observe(time).value??0),squarings=z>.25?Math.ceil(Math.log2(z/.25)):0;
  const step=mat((i,j)=>shift(mul(B[i][j],time),-squarings*ln2));
  let term=step,change=step;
  for(let order=2;order<=32;order++){
    const next=matrixProduct(term,step);term=mat((i,j)=>shift(next[i][j],-Math.log(order)));
    change=mat((i,j)=>add(change[i][j],term[i][j]));
  }
  let propagator=mat((i,j)=>add(i===j?q(1):zero,change[i][j]));
  for(let i=0;i<squarings;i++){
    const square=matrixProduct(change,change);change=mat((j,k)=>add(shift(change[j][k],ln2),square[j][k]));
    propagator=matrixProduct(propagator,propagator);
  }
  return{propagator,change};
}
export interface ThermoelasticParameters {
  elasticModulus:number; density:number; heatCapacity:number; referenceTemperature:number;
  coupling:number; conductivity:number; wavenumber:number; area:number;
  initialDisplacement:number; initialVelocity:number; initialTemperatureDeviation:number;
}
export type ThermoelasticCoupling = 'reciprocal' | 'one-way';
function validateElastic(p:ThermoelasticParameters){
  bound(p.elasticModulus,.5,4,'elastic modulus');bound(p.density,.5,2,'density');bound(p.heatCapacity,.5,2,'volumetric heat capacity');
  bound(p.referenceTemperature,250,350,'reference absolute temperature');bound(p.coupling,0,.05,'thermal stress coefficient');bound(p.conductivity,0,1,'thermal conductivity');
  bound(p.wavenumber,.25,2,'positive wavenumber');bound(p.area,.25,4,'area');bound(p.initialDisplacement,-.001,.001,'initial displacement amplitude');
  bound(p.initialVelocity,-.001,.001,'initial velocity amplitude');bound(p.initialTemperatureDeviation,-.02,.02,'initial temperature amplitude');
}
function elasticSystem(p:ThermoelasticParameters,mode:ThermoelasticCoupling){
  if(mode!=='reciprocal'&&mode!=='one-way')throw new RangeError('Unknown thermoelastic coupling');
  const E=p.elasticModulus,rho=p.density,C=p.heatCapacity,T=p.referenceTemperature,k=p.wavenumber;
  const w=q(k,Math.sqrt(E/rho)),g=q(p.coupling,k,Math.sqrt(T/(rho*C))),d=shift(q(p.conductivity,k,k),-Math.log(C));
  const B:Matrix=[[zero,w,zero],[neg(w),zero,g],[zero,mode==='reciprocal'?neg(g):zero,neg(d)]];
  const initial:Vector=[q(k,Math.sqrt(E),p.initialDisplacement),q(Math.sqrt(rho),p.initialVelocity),q(Math.sqrt(C/T),p.initialTemperatureDeviation)];
  return{B,initial,g,d,volume:2*Math.PI*p.area/k};
}
function elasticState(p:ThermoelasticParameters,time:Q,mode:ThermoelasticCoupling){
  const s=elasticSystem(p,mode),pair=exponentialPair(s.B,time),change=apply(pair.change,s.initial),y=apply(pair.propagator,s.initial);
  const dy=apply(s.B,y),square=(v:Q)=>mul(v,v),energy=(v:Q)=>shift(square(v),Math.log(s.volume/4));
  const mechanical=add(energy(y[0]),energy(y[1])),thermal=energy(y[2]);
  const exchange=shift(mul(s.g,mul(y[1],y[2])),Math.log(s.volume/2)),loss=shift(mul(s.d,square(y[2])),Math.log(s.volume/2));
  const componentChanges=change.map((delta,i)=>shift(mul(delta,add(shift(s.initial[i],ln2),delta)),Math.log(s.volume/4)));
  return{...s,y,dy,change,mechanical,thermal,availability:add(mechanical,thermal),exchange,loss,
    mechanicalChange:add(componentChanges[0],componentChanges[1]),thermalChange:componentChanges[2],endpointChange:add(...componentChanges),
    displacement:shift(y[0],-Math.log(p.wavenumber*Math.sqrt(p.elasticModulus))),velocity:shift(y[1],-.5*Math.log(p.density)),temperature:shift(y[2],.5*Math.log(p.referenceTemperature/p.heatCapacity))};
}
export function thermoelasticAt(p:ThermoelasticParameters,time:number,phase=0,mode:ThermoelasticCoupling='reciprocal'){
  validateElastic(p);bound(time,0,4,'time');bound(phase,0,2*Math.PI,'spatial phase');
  return observeElasticState(p,time,phase,mode,elasticState(p,q(time),mode));
}
/** Share the same amplitude solve over a physical spatial profile. */
export function thermoelasticProfile(p:ThermoelasticParameters,time:number,phases:number[],mode:ThermoelasticCoupling='reciprocal'){
  validateElastic(p);bound(time,0,4,'time');
  for(let i=0;i<phases.length;i++)bound(phases[i],0,2*Math.PI,'spatial phase');
  const state=elasticState(p,q(time),mode);
  return phases.map(phase=>observeElasticState(p,time,phase,mode,state));
}
function observeElasticState(p:ThermoelasticParameters,time:number,phase:number,mode:ThermoelasticCoupling,r:ReturnType<typeof elasticState>){
  const k=p.wavenumber,cos=Math.cos(phase),sin=Math.sin(phase);
  const strain=mul(q(k,cos),r.displacement),rate=mul(q(k,cos),r.velocity),temperature=mul(q(cos),r.temperature);
  const stress=add(mul(q(p.elasticModulus),strain),neg(mul(q(p.coupling),temperature)));
  const gradient=neg(mul(q(k,sin),r.temperature)),temperatureRate=shift(r.dy[2],.5*Math.log(p.referenceTemperature/p.heatCapacity));
  const initialAvailability=add(...r.initial.map(y=>shift(mul(y,y),Math.log(r.volume/4))));
  const amplitude=(i:number,scale:number)=>observe(shift(r.y[i],scale));
  return{time,phase,mode,cellLength:2*Math.PI/k,cellVolume:r.volume,
    displacementAmplitude:observe(r.displacement),velocityAmplitude:observe(r.velocity),temperatureAmplitude:observe(r.temperature),
    displacementChange:observe(shift(r.change[0],-Math.log(k*Math.sqrt(p.elasticModulus)))),velocityChange:observe(shift(r.change[1],-.5*Math.log(p.density))),temperatureChange:observe(shift(r.change[2],.5*Math.log(p.referenceTemperature/p.heatCapacity))),
    accelerationAmplitude:observe(shift(r.dy[1],-.5*Math.log(p.density))),temperatureRateAmplitude:observe(temperatureRate),temperatureRate:observe(mul(q(cos),temperatureRate)),
    displacement:observe(mul(q(sin),r.displacement)),velocity:observe(mul(q(sin),r.velocity)),strain:observe(strain),strainRate:observe(rate),temperatureDeviation:observe(temperature),absoluteTemperature:{...observe(add(q(p.referenceTemperature),temperature)),value:p.referenceTemperature+(observe(temperature).value??0)},
    temperatureGradient:observe(gradient),heatFlux:observe(neg(mul(q(p.conductivity),gradient))),stress:observe(stress),
    mechanicalAvailability:observe(r.mechanical),thermalAvailability:observe(r.thermal),totalAvailability:observe(r.availability),initialAvailability:observe(initialAvailability),
    mechanicalExchangeRate:observe(r.exchange),thermalExchangeRate:observe(mode==='reciprocal'?neg(r.exchange):zero),conductionAvailabilityLoss:observe(r.loss),
    totalAvailabilityRate:observe(mode==='reciprocal'?neg(r.loss):add(r.exchange,neg(r.loss))),
    endpointAvailabilityChange:observe(r.endpointChange),mechanicalEndpointChange:observe(r.mechanicalChange),thermalEndpointChange:observe(r.thermalChange),
    reciprocalStrainBound:mode==='reciprocal'?observe(initialAvailability.sign===0?zero:{sign:1,log:.5*(initialAvailability.log+Math.log(4/r.volume/p.elasticModulus))}):null,
    normalizedStrainAmplitude:amplitude(0,0),normalizedVelocityAmplitude:amplitude(1,0),normalizedTemperatureAmplitude:amplitude(2,0),
    energyMeaning:'quadratic-availability-not-total-internal-energy' as const};
}
const nodes=[.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,.755404408355003,.8656312023878318,.9445750230732326,.9894009349916499];
const weights=[.1894506104550685,.1826034150449236,.16915651939500254,.14959598881657673,.12462897125553387,.09515851168249278,.06225352393864789,.027152459411754095];
export function thermoelasticWindow(p:ThermoelasticParameters,start:number,end:number,mode:ThermoelasticCoupling='reciprocal'){
  validateElastic(p);bound(start,0,4,'start time');bound(end,0,4,'end time');if(end<start)throw new RangeError('Expected start <= end');
  const system=elasticSystem(p,mode),width=end-start,norm=Math.max(...system.B.map(row=>row.reduce((v,a)=>v+Math.abs(observe(a).value??0),0))),panels=Math.max(1,Math.ceil(width*norm/2));
  const exchanges:Q[]=[],losses:Q[]=[];
  if(width!==0)for(let j=0;j<panels;j++)for(let i=0;i<nodes.length;i++)for(const sign of[-1,1]){
    const fraction=(j+.5+sign*.5*nodes[i])/panels,time=add(q(start),q(width,fraction)),state=elasticState(p,time,mode),scale=q(width,weights[i]/(2*panels));
    exchanges.push(mul(state.exchange,scale));losses.push(mul(state.loss,scale));
  }
  const exchange=add(...exchanges),loss=add(...losses),availabilityChange=mode==='reciprocal'?neg(loss):add(exchange,neg(loss));
  return{start,end,mode,panels,mechanicalExchange:observe(exchange),conductionAvailabilityLoss:observe(loss),balanceDerivedAvailabilityChange:observe(availabilityChange),
    thermalExchange:observe(mode==='reciprocal'?neg(exchange):zero),meaning:'independent-time-quadrature-of-declared-family-not-an-error-bound' as const};
}

export interface ThermoelectricParameters {
  length:number;area:number;resistivity:number;conductivity:number;seebeck:number;
  leftTemperature:number;rightTemperature:number;currentDensity:number;referencePotential:number;
}
function validateElectric(p:ThermoelectricParameters){
  bound(p.length,.25,4,'length');bound(p.area,.25,4,'area');bound(p.resistivity,.01,4,'resistivity');bound(p.conductivity,.1,4,'conductivity');
  bound(p.seebeck,-.05,.05,'Seebeck coefficient');bound(p.leftTemperature,250,350,'left absolute temperature');bound(p.rightTemperature,250,350,'right absolute temperature');
  bound(p.currentDensity,-10,10,'prescribed current density');bound(p.referencePotential,-4,4,'reference potential');
}
/** Exact rational operations for the constant-property bar. Keep cancellation
 * between supplied binary inputs until the final observation; logs alone can
 * lose the sign of a small voltage or incorrectly classify terminal work. */
interface ElectricFraction { numerator: BinaryValue; denominator: BinaryValue }
const fraction=(...factors:number[]):ElectricFraction=>({numerator:binaryProduct(...factors),denominator:binaryProduct(1)});
const fractionNegate=(a:ElectricFraction):ElectricFraction=>({...a,numerator:binaryNegate(a.numerator)});
const fractionMultiply=(a:ElectricFraction,b:ElectricFraction):ElectricFraction=>({numerator:binaryMultiply(a.numerator,b.numerator),denominator:binaryMultiply(a.denominator,b.denominator)});
const fractionDivide=(a:ElectricFraction,b:ElectricFraction):ElectricFraction=>({numerator:binaryMultiply(a.numerator,b.denominator),denominator:binaryMultiply(a.denominator,b.numerator)});
function fractionSum(...terms:ElectricFraction[]):ElectricFraction {
 return terms.reduce((a,b)=>({numerator:binarySum([binaryMultiply(a.numerator,b.denominator),binaryMultiply(b.numerator,a.denominator)]),denominator:binaryMultiply(a.denominator,b.denominator)}),fraction(0));
}
const fractionObservation=(a:ElectricFraction):MultiphysicsObservation=>binaryRatioObservation(a.numerator,a.denominator);
/** Current is prescribed. The terminal potential drop is dependent, never independently imposed. */
export function thermoelectricAt(p:ThermoelectricParameters,xi:number){
  validateElectric(p);bound(xi,0,1,'dimensionless position');
  const{length:L,area:A,resistivity:r,conductivity:K,seebeck:S,currentDensity:j,leftTemperature:TL,rightTemperature:TR}=p;
  const f=fraction,add=fractionSum,mul=fractionMultiply,div=fractionDivide,neg=fractionNegate,obs=fractionObservation;
  const delta=add(f(TR),neg(f(TL))),complement=add(f(1),neg(f(xi)));
  const joule=f(r,j,j),rise=div(mul(f(r,j,j,L,L,xi),complement),f(2,K));
  const temperature=add(mul(f(TL),complement),f(TR,xi),rise),temperatureChange=add(mul(delta,f(xi)),rise);
  const gradient=add(div(delta,f(L)),div(mul(f(r,j,j,L),add(f(1),f(-2,xi))),f(2,K)));
  const field=add(f(r,j),mul(f(S),gradient)),peltier=mul(f(S,j),temperature),fourier=neg(mul(f(K),gradient)),flux=add(peltier,fourier);
  const voltage=add(f(r,j,L),mul(f(S),delta)),partialVoltage=add(f(r,j,L,xi),mul(f(S),temperatureChange));
  const power=mul(f(A,j),voltage),joulePower=f(A,r,j,j,L),reversiblePower=mul(f(A,S,j),delta);
  const leftFlux=add(f(S,TL,j),neg(div(mul(f(K),delta),f(L))),f(-.5,r,j,j,L));
  const rightFlux=add(f(S,TR,j),neg(div(mul(f(K),delta),f(L))),f(.5,r,j,j,L));
  const leftHeat=mul(f(A),leftFlux),rightHeat=mul(f(A),rightFlux),partialPower=mul(f(A,j),partialVoltage);
  const observedLeft=obs(leftHeat),observedRight=obs(rightHeat),observedPower=obs(power);
  return{xi,position:obs(f(L,xi)),current:obs(f(A,j)),currentDensity:obs(f(j)),resistance:obs(div(f(r,L),f(A))),
    temperature:obs(temperature),stableJouleTemperatureRise:obs(rise),temperatureGradient:obs(gradient),temperatureCurvature:obs(neg(div(joule,f(K)))),
    electricField:obs(field),potential:obs(add(f(p.referencePotential),neg(partialVoltage))),stablePotentialChange:obs(neg(partialVoltage)),terminalVoltageDrop:obs(voltage),
    peltierHeatFlux:obs(peltier),fourierHeatFlux:obs(fourier),totalHeatFlux:obs(flux),
    localJouleHeating:obs(joule),localElectricalWork:obs(mul(f(j),field)),heatFluxGradient:obs(add(joule,mul(f(S,j),gradient))),
    leftHeatInput:observedLeft,rightHeatOutput:observedRight,electricalPowerInput:observedPower,joulePower:obs(joulePower),reversibleElectricalPower:obs(reversiblePower),
    stableBoundaryHeatDifference:observedPower,roundedBoundaryHeatDifference:observedLeft.value===null||observedRight.value===null?null:observedRight.value-observedLeft.value,
    partialElectricalPower:obs(partialPower),partialJoulePower:obs(f(A,r,j,j,L,xi)),
    leftCoolingCOP:observedLeft.sign>0&&observedPower.sign>0?obs(div(leftHeat,power)):null,
    leftCoolingStatus:observedLeft.sign<=0?'no-positive-left-heat-input' as const:observedPower.sign<=0?'no-positive-electrical-input' as const:'defined' as const,
    electricalRegime:observedPower.sign<0?'generation' as const:observedPower.sign===0?'zero-terminal-work' as const:'electrical-input' as const,
    fieldMeaning:'steady-constant-property-single-leg-with-declared-terminal-ports' as const};
}
