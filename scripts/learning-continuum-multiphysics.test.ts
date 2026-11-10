import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {thermoelasticAt as thermal,thermoelasticWindow as window,thermoelectricAt as electric,type ThermoelasticParameters,type ThermoelectricParameters,type ThermoelasticCoupling,type MultiphysicsObservation} from '../src/lib/learning-continuum-multiphysics';
interface Ref {value:string;sign:number;logMagnitude:string|null}
interface ThermalCase {parameters:ThermoelasticParameters;time:number;phase:number;mode:ThermoelasticCoupling;expected:Record<string,Ref>;window?:Record<string,Ref>}
interface ElectricCase {parameters:ThermoelectricParameters;xi:number;expected:Record<string,Ref>;integrals?:Record<string,Ref>;cop?:Ref}
const refs=JSON.parse(readFileSync('scripts/fixtures/continuum-multiphysics-independent.json','utf8')) as {thermalCases:ThermalCase[];electricCases:ElectricCase[];thermalLimits:ThermalCase[];electricLimits:ElectricCase[]};
const p:ThermoelasticParameters={elasticModulus:2,density:1,heatCapacity:1,referenceTemperature:300,coupling:.02,conductivity:.1,wavenumber:1,area:1,initialDisplacement:.001,initialVelocity:.001,initialTemperatureDeviation:.02};
const e:ThermoelectricParameters={length:1,area:1,resistivity:1,conductivity:1,seebeck:.01,leftTemperature:300,rightTemperature:310,currentDensity:.9,referencePotential:0};
function near(a:number|null,b:number,tol=4e-9){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(Math.abs(b),1e-290),`${a} != ${b}`);}
function observation(actual:MultiphysicsObservation,r:Ref,label:string){
 assert.equal(actual.sign,r.sign,label+' sign');
 if(r.sign===0){assert.equal(actual.value,0,label);assert.equal(actual.logMagnitude,null);return;}
 assert.ok(actual.logMagnitude!==null,label);assert.ok(Math.abs(actual.logMagnitude!-Number(r.logMagnitude))<2e-7,label+' log');
 const n=Number(r.value);
 if(n===0||!Number.isFinite(n))assert.equal(actual.value,null,label+' unavailable');
 else if(Math.abs(n)<1e-300){assert.notEqual(actual.value,null);assert.ok(Math.abs(actual.value!-n)<=8*Number.MIN_VALUE+Math.abs(n)*2e-8,label);}
 else near(actual.value,n,2e-8);
}
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
test('24 physical-variable matrix exponential/eigensystem references verify both thermoelastic coupling models',()=>{
 assert.equal(refs.thermalCases.length,24);
 for(const [i,row]of refs.thermalCases.entries()){const q=thermal(row.parameters,row.time,row.phase,row.mode) as unknown as Record<string,MultiphysicsObservation>;for(const[key,r]of Object.entries(row.expected))observation(q[key],r,`${i}:${key}`);}
});
test('24 independent time quadratures verify reciprocal exchange, conduction loss and endpoint availability change',()=>{
 for(const [i,row]of refs.thermalCases.entries()){const r=window(row.parameters,0,row.time,row.mode) as unknown as Record<string,MultiphysicsObservation>;for(const[key,v]of Object.entries(row.window!))observation(r[key],v,`${i}:${key}`);}
});
test('24 thermoelectric profiles verify independently differentiated temperature, potential and heat/electric work',()=>{
 assert.equal(refs.electricCases.length,24);
 for(const [i,row]of refs.electricCases.entries()){const q=electric(row.parameters,row.xi) as unknown as Record<string,MultiphysicsObservation>;for(const[key,r]of Object.entries(row.expected))observation(q[key],r,`${i}:${key}`);}
});
test('physical electrical/Joule/partial-work integrals match actual boundary and terminal observations',()=>{
 for(const row of refs.electricCases){const q=electric(row.parameters,row.xi);observation(q.electricalPowerInput,row.integrals!.electricalWork,'electrical work');observation(q.joulePower,row.integrals!.jouleHeating,'Joule heat');observation(q.partialElectricalPower,row.integrals!.partialWork,'partial work');}
});
test('eight high-precision thermoelastic limits retain small amplitudes, tiny changes and absent coupling',()=>{
 assert.equal(refs.thermalLimits.length,8);
 for(const [i,row]of refs.thermalLimits.entries()){const q=thermal(row.parameters,row.time,row.phase,row.mode) as unknown as Record<string,MultiphysicsObservation>;for(const[key,r]of Object.entries(row.expected))observation(q[key],r,`${i}:${key}`);}
});
test('eight high-precision thermoelectric limits preserve tiny powers, exact zeros and conditional cooling ratios',()=>{
 assert.equal(refs.electricLimits.length,8);
 for(const [i,row]of refs.electricLimits.entries()){const q=electric(row.parameters,row.xi),actual=q as unknown as Record<string,MultiphysicsObservation>;for(const[key,r]of Object.entries(row.expected))observation(actual[key],r,`${i}:${key}`);if(row.cop)observation(q.leftCoolingCOP!,row.cop,'COP');else assert.equal(q.leftCoolingCOP,null);}
});
test('initial worked mode distinguishes stored availability, reversible exchange and conduction dissipation',()=>{
 const q=thermal(p,0);near(q.totalAvailability.value,Math.PI/2*(3e-6+.0004/300));near(q.stress.value,.0016);
 near(q.mechanicalExchangeRate.value,Math.PI*.02*.001*.02);near(q.thermalExchangeRate.value,-Math.PI*.02*.001*.02);
 near(q.conductionAvailabilityLoss.value,Math.PI*.1*.0004/300);near(q.totalAvailabilityRate.value,-q.conductionAvailabilityLoss.value!);
 assert.equal(q.displacementChange.value,0);assert.equal(q.temperatureChange.value,0);assert.ok(q.reciprocalStrainBound!.value!<.01);
});
test('a reciprocal adiabatic mode exchanges availability without conduction loss; zero coupling decouples both models',()=>{
 const a=thermal({...p,conductivity:0},1),b=thermal({...p,conductivity:0},0);near(a.totalAvailability.value,b.totalAvailability.value!);assert.equal(a.conductionAvailabilityLoss.value,0);assert.equal(window({...p,conductivity:0},0,1).balanceDerivedAvailabilityChange.value,0);
 const c=thermal({...p,coupling:0},1),d=thermal({...p,coupling:0},1,0,'one-way');assert.deepEqual(c.displacementAmplitude,d.displacementAmplitude);assert.deepEqual(c.temperatureAmplitude,d.temperatureAmplitude);
 near(c.temperatureAmplitude.value,.02*Math.exp(-.1));assert.equal(c.mechanicalExchangeRate.value,0);
});
test('rapid decoupled thermal decay retains its small nonzero tail without subtracting from one',()=>{
 const q=thermal({...p,coupling:0,conductivity:1,wavenumber:2,heatCapacity:.5},4);
 near(q.temperatureAmplitude.value,.02*Math.exp(-32),1e-10);
 assert.ok(Math.abs(q.temperatureAmplitude.logMagnitude!-(Math.log(.02)-32))<1e-10);
});
test('one-way omission changes the actual state and leaves a declared exchange defect in its energy balance',()=>{
 const a=thermal(p,1),b=thermal(p,1,0,'one-way');assert.notEqual(a.temperatureAmplitude.value,b.temperatureAmplitude.value);assert.equal(b.thermalExchangeRate.value,0);assert.equal(b.reciprocalStrainBound,null);
 near(b.totalAvailabilityRate.value,b.mechanicalExchangeRate.value!-b.conductionAvailabilityLoss.value!);
 const w=window(p,0,1,'one-way');near(w.balanceDerivedAvailabilityChange.value,b.totalAvailability.value!-b.initialAvailability.value!,1e-8);
});
test('tiny times retain generated temperature, second-order velocity change and cubic conduction loss',()=>{
 const parameters={...p,initialDisplacement:0,initialTemperatureDeviation:0};
 for(const t of[1e-20,1e-200,Number.MIN_VALUE]){
  const q=thermal(parameters,t),w=window(parameters,0,t);
  assert.equal(q.temperatureAmplitude.sign,-1);assert.equal(q.velocityChange.sign,-1);assert.equal(w.conductionAvailabilityLoss.sign,1);
  const leadingLog=Math.log(2*Math.PI)+Math.log(.1)+Math.log(300)+2*Math.log(.02)+2*Math.log(.001)-Math.log(6)+3*Math.log(t);
  assert.ok(Math.abs(w.conductionAvailabilityLoss.logMagnitude!-leadingLog)<1e-8);
 }
});
test('real field finite differences verify thermoelastic momentum and entropy equations',()=>{
 const time=.4,phase=.7,h=1e-4,q=thermal(p,time,phase),lo=thermal(p,time,phase-h),hi=thermal(p,time,phase+h);
 const sigmaX=(hi.stress.value!-lo.stress.value!)/(2*h/p.wavenumber);near(sigmaX,p.density*q.accelerationAmplitude.value!*Math.sin(phase),1e-8);
 const thetaXX=-(p.wavenumber**2)*q.temperatureDeviation.value!;near(p.heatCapacity*q.temperatureRateAmplitude.value!*Math.cos(phase),p.conductivity*thetaXX-p.referenceTemperature*p.coupling*q.strainRate.value!);
});
test('reference potential shifts only the potential; actual heat and electric work remain invariant',()=>{
 const a=electric(e,.3),b=electric({...e,referencePotential:3},.3);near(b.potential.value!-a.potential.value!,3);
 for(const key of['electricField','totalHeatFlux','terminalVoltageDrop','electricalPowerInput','stablePotentialChange']as const)assert.deepEqual(a[key],b[key]);
});
test('worked bar has physical heat imbalance0.9W, Joule0.81W and reversible electrical work0.09W',()=>{
 const q=electric(e,.5);near(q.temperature.value,305.10125);near(q.leftHeatInput.value,-7.705);near(q.rightHeatOutput.value,-6.805);
 near(q.terminalVoltageDrop.value,1);near(q.electricalPowerInput.value,.9);near(q.joulePower.value,.81);near(q.reversibleElectricalPower.value,.09);assert.equal(q.leftCoolingCOP,null);
});
test('cooling, generation and zero terminal work use different port conditions',()=>{
 const q=electric({...e,resistivity:.01,conductivity:.1,seebeck:.05,currentDensity:10},.5);near(q.leftHeatInput.value,148.5);near(q.rightHeatOutput.value,154.5);near(q.electricalPowerInput.value,6);near(q.leftCoolingCOP!.value,24.75);
 const g=electric({...e,currentDensity:-.05},.5);assert.equal(g.electricalRegime,'generation');near(g.electricalPowerInput.value,-.0025);assert.equal(g.leftCoolingCOP,null);
 const z=electric({...e,currentDensity:0},.5);assert.equal(z.electricalRegime,'zero-terminal-work');assert.equal(z.electricalPowerInput.value,0);assert.equal(z.joulePower.value,0);near(z.totalHeatFlux.value,-10);near(z.terminalVoltageDrop.value,.1);
});
test('tiny Joule power survives rounded equal boundary heat fluxes and small temperature rise survives a rounded absolute temperature',()=>{
 const q=electric({...e,rightTemperature:300,currentDensity:1e-200},.5);
 assert.equal(q.temperature.value,300);assert.equal(q.stableJouleTemperatureRise.value,null);assert.equal(q.stableJouleTemperatureRise.sign,1);
 assert.equal(q.electricalPowerInput.value,null);assert.equal(q.electricalPowerInput.sign,1);assert.equal(q.roundedBoundaryHeatDifference,0);assert.equal(q.stableBoundaryHeatDifference.sign,1);
});
test('actual spatial differences verify thermoelectric local work and temperature curvature',()=>{
 const h=1e-4,xi=.4,q=electric(e,xi),lo=electric(e,xi-h),hi=electric(e,xi+h);
 near(-(hi.potential.value!-lo.potential.value!)/(2*h*e.length),q.electricField.value!,1e-8);
 near((hi.totalHeatFlux.value!-lo.totalHeatFlux.value!)/(2*h*e.length),q.localElectricalWork.value!,1e-8);
 near(q.temperatureCurvature.value,-e.resistivity*e.currentDensity**2/e.conductivity);
});
test('entropy-flux differentiation cancels reversible terms and leaves nonnegative resistance/conduction production',()=>{
 for(const parameters of[e,{...e,currentDensity:-.05},{...e,resistivity:.01,conductivity:.1,seebeck:.05,currentDensity:10}]){
  const xi=.4,h=1e-4,q=electric(parameters,xi),lo=electric(parameters,xi-h),hi=electric(parameters,xi+h),T=q.temperature.value!;
  const derivative=(hi.totalHeatFlux.value!/hi.temperature.value!-lo.totalHeatFlux.value!/lo.temperature.value!)/(2*h*parameters.length);
  const production=parameters.resistivity*parameters.currentDensity**2/T+parameters.conductivity*(q.temperatureGradient.value!/T)**2;
  assert.ok(production>=0);near(derivative,production,2e-6);
 }
});
test('zero windows, exact zero initial mode and invalid physical inputs stay explicit',()=>{
 const w=window(p,1,1);assert.equal(w.mechanicalExchange.value,0);assert.equal(w.conductionAvailabilityLoss.value,0);
 const z=thermal({...p,initialDisplacement:0,initialVelocity:0,initialTemperatureDeviation:0},1);assert.equal(z.totalAvailability.value,0);assert.equal(z.reciprocalStrainBound!.value,0);
 assert.throws(()=>window(p,1,0),RangeError);assert.throws(()=>thermal({...p,wavenumber:0},0),RangeError);assert.throws(()=>thermal({...p,conductivity:-1},0),RangeError);
 for(const bad of[0,-1,Infinity,NaN])assert.throws(()=>electric({...e,resistivity:bad},.5),RangeError);
 for(const input of[0,Number.MIN_VALUE,1e-200,.02])finiteTree(thermal({...p,initialTemperatureDeviation:input},1));
 for(const currentDensity of[0,Number.MIN_VALUE,1e-200,10])finiteTree(electric({...e,currentDensity},.5));
});
