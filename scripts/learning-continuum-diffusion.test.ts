import test from 'node:test';
import assert from 'node:assert/strict';
import {heatModeAt as heat,heatModeCell as cell,heatModeWindow as thermalWindow,gaussianDiffusionAt as gaussian,gaussianDiffusionState as state,gaussianDiffusionWindow as massWindow,type HeatModeParameters,type GaussianDiffusionParameters,type DiffusionValue}from '../src/lib/learning-continuum-diffusion';
import refs from './fixtures/continuum-diffusion-independent.json';
const h:HeatModeParameters={heatCapacity:2,conductivity:.4,meanTemperature:300,contrast:.1,waveNumber:1.2,phase:.3,area:1.5};
const g:GaussianDiffusionParameters={mass:2,area:1.5,initialSigma:.5,initialMean:-.2,velocity:.7,diffusivity:.15};
function near(a:DiffusionValue,b:number,tolerance=2e-10){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tolerance*Math.max(1,Math.abs(b)),`${a} vs ${b}`);}
function relative(a:DiffusionValue,b:number){assert.notEqual(a,null);assert.ok(Math.abs(a!/b-1)<2e-11,`${a} vs ${b}`);}
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
test('24 independent heat fields and actual spatial/time derivatives match physical Fourier flux',()=>{
 assert.equal(refs.heatCases.length,24);for(const r of refs.heatCases){const q=heat(r.parameters,r.x,r.time),e=r.expected;
  for(const key of ['temperature','gradient','laplacian','timeDerivative','heatFlux']as const)near(q[key],e[key]);near(q.localBalanceResidual,0);assert.ok(q.temperature!>0);
 }
});
test('24 independently integrated heat windows agree with storage rate and oriented physical face power',()=>{
 for(const r of refs.heatCases){const q=thermalWindow(r.parameters,r.center,r.halfWidth,r.time),e=r.expected;
  near(q.modeHeat,e.modeHeat);near(q.energyRate,e.energyRate);near(q.stableNetPower,e.netPower);near(q.roundedFaceNetPower,e.netPower);near(q.analyticBalanceResidual,0);
 }
});
test('24 physical period-cell integrals distinguish quadratic norm loss from conserved sensible energy',()=>{
 for(const r of refs.heatCases){const q=cell(r.parameters,r.time),e=r.expected;near(q.quadraticNorm,e.quadraticNorm);near(q.gradientNormDissipation,e.gradientNormDissipation);near(q.quadraticNormRate,-q.gradientNormDissipation!);assert.equal(q.sensibleEnergyRate,0);assert.equal(q.perturbationHeat,0);}
});
test('actual temperature differences and finite-window energy differences verify signs and units',()=>{
 const x=.4,t=.6,dt=1e-5,dx=1e-4,q=heat(h,x,t);
 near((heat(h,x+dx,t).temperature!-heat(h,x-dx,t).temperature!)/(2*dx),q.gradient!,2e-7);
 near((heat(h,x,t+dt).temperature!-heat(h,x,t-dt).temperature!)/(2*dt),q.timeDerivative!,2e-8);
 const fluxDerivative=(-heat(h,x+2*dx,t).heatFlux!+8*heat(h,x+dx,t).heatFlux!-8*heat(h,x-dx,t).heatFlux!+heat(h,x-2*dx,t).heatFlux!)/(12*dx);
 near(h.heatCapacity*q.timeDerivative!+fluxDerivative,0,2e-8);
 const a=thermalWindow(h,.2,.7,t-dt),b=thermalWindow(h,.2,.7,t+dt),mid=thermalWindow(h,.2,.7,t);
 near((b.sensibleEnergy!-a.sensibleEnergy!)/(2*dt),mid.energyRate!,2e-8);
});
test('worked heat example has explicit physical watts, conserved heat and decreasing K² m³ norm',()=>{
 const p={heatCapacity:2,conductivity:1,meanTemperature:300,contrast:.1,waveNumber:1,phase:0,area:2};
 const q=heat(p,Math.PI/2,0),w=thermalWindow(p,0,Math.PI/2,0),c=cell(p,0);
 near(q.heatFlux,30);near(w.modeHeat,240);near(w.energyRate,-120);near(w.stableNetPower,120);
 near(c.quadraticNorm,900*Math.PI);near(c.sensibleEnergy,2*4*Math.PI*(300-273.15));
 assert.ok(cell(p,1).quadraticNorm!<c.quadraticNorm!);near(cell(p,1).sensibleEnergy,c.sensibleEnergy);
});
test('zero conductivity and uniform temperature are separate exact controls',()=>{
 const noConduction=heat({...h,conductivity:0},.4,3),initial=heat({...h,conductivity:0},.4,0);
 near(noConduction.perturbation,initial.perturbation!);near(noConduction.heatFlux,0);near(noConduction.timeDerivative,0);
 const uniform=heat({...h,contrast:0},.4,3);near(uniform.temperature,300);near(uniform.gradient,0);near(uniform.heatFlux,0);near(cell({...h,contrast:0},3).quadraticNorm,0);
});
test('heat flux points down temperature gradients and diffusivity is conductivity divided by capacity',()=>{
 for(const x of [-1,-.3,.2,1]){const q=heat(h,x,.4);assert.ok(q.heatFlux!*q.gradient!<=0);near(q.timeDerivative,h.conductivity/h.heatCapacity*q.laplacian!);}
 const sameDiffusivity=heat({...h,heatCapacity:4,conductivity:.8},.4,.7),original=heat(h,.4,.7);near(sameDiffusivity.temperature,original.temperature!);near(sameDiffusivity.heatFlux,2*original.heatFlux!);
});
test('small heat changes retain stable nonzero values below rounded factor equality',()=>{
 for(const r of refs.tinyCases.slice(0,2)){const p={...h,heatCapacity:1,conductivity:Number(r.diffusivity),waveNumber:1},q=heat(p,0,Number(r.time));assert.equal(q.decayFactor,1);relative(q.stableRelativeDecayChange,Number(r.heatDecayChange));assert.notEqual(q.amplitudeChange,0);}
 const q=heat({...h,heatCapacity:1,conductivity:1e-200,waveNumber:1},0,1e-200);assert.equal(q.exponent,null);assert.equal(q.stableRelativeDecayChange,null);assert.equal(q.amplitudeChange,null);
});
test('log-domain scaling rescues absolute heat observations without inventing zero relative change',()=>{
 const p={...h,heatCapacity:4,conductivity:Number.MIN_VALUE,meanTemperature:400,contrast:.2,waveNumber:.25},q=heat(p,0,1);
 assert.equal(q.exponent,null);assert.equal(q.stableRelativeDecayChange,null);assert.equal(q.amplitudeChange,-Number.MIN_VALUE);
 const rapid=heat({...h,heatCapacity:.25,conductivity:4,meanTemperature:400,contrast:.2,waveNumber:4,phase:0},0,749/256);
 assert.equal(rapid.decayFactor,0);assert.equal(rapid.amplitude,Number.MIN_VALUE);assert.ok(rapid.timeDerivative!<0);
});
test('collapsed heat windows differ from nonzero specified widths whose endpoints round together',()=>{
 const zero=thermalWindow(h,.3,0,.5);assert.equal(zero.geometryStatus,'zero-width');near(zero.modeHeat,0);near(zero.energyRate,0);
 const tiny=thermalWindow(h,.3,1e-20,.5);assert.equal(tiny.geometryStatus,'endpoints-round-together');assert.equal(tiny.left,tiny.right);assert.notEqual(tiny.modeHeat,0);assert.notEqual(tiny.stableNetPower,0);assert.equal(tiny.roundedFaceNetPower,0);
});
test('24 independent Gaussian concentration fields, derivatives and separated fluxes agree',()=>{
 assert.equal(refs.gaussianCases.length,24);for(const r of refs.gaussianCases){const q=gaussian(r.parameters,r.x,r.time),e=r.expected;
  for(const key of ['concentration','logConcentration','spatialDerivative','secondDerivative','timeDerivative','advectiveFlux','diffusiveFlux']as const)near(q[key],e[key]);near(q.localBalanceResidual,0);
 }
});
test('24 independent finite-window integrals and fluxes verify actual mass budgets',()=>{
 for(const r of refs.gaussianCases){const q=massWindow(r.parameters,r.left,r.right,r.time),e=r.expected.window;
  for(const key of ['fraction','logFraction','massInside','massRate','netOutflow']as const)near(q[key],e[key]);near(q.balanceResidual,0);
 }
});
test('independently integrated full-domain moments match conserved mass, mean advection and variance growth',()=>{
 for(const r of refs.gaussianCases){const q=state(r.parameters,r.time),[m,m1,m2]=r.expected.moments;near(m,r.parameters.mass);near(q.massWeightedMean,m1/m);near(q.massWeightedVariance,m2/m-(m1/m)**2);}
});
test('finite differences of actual concentrations and window masses check the diffusion PDE',()=>{
 const x=.4,t=.7,dt=1e-5,dx=1e-4,q=gaussian(g,x,t);
 const c0=q.concentration!,cm=gaussian(g,x-dx,t).concentration!,cp=gaussian(g,x+dx,t).concentration!;
 near((cp-cm)/(2*dx),q.spatialDerivative!,2e-8);
 near((gaussian(g,x,t+dt).concentration!-gaussian(g,x,t-dt).concentration!)/(2*dt),q.timeDerivative!,2e-9);
 near(q.timeDerivative!+g.velocity*(cp-cm)/(2*dx)-g.diffusivity*(cp-2*c0+cm)/(dx*dx),0,2e-8);
 const a=massWindow(g,-.7,1.1,t-dt),b=massWindow(g,-.7,1.1,t+dt),mid=massWindow(g,-.7,1.1,t);
 near((b.massInside!-a.massInside!)/(2*dt),mid.massRate!,2e-9);
});
test('Gaussian spreading lowers its peak without losing total mass and outward diffusion reverses at the center',()=>{
 const p={...g,velocity:0,initialMean:0},initial=gaussian(p,0,0),later=gaussian(p,0,1);
 assert.ok(later.concentration!<initial.concentration!);near(later.state.variance,.55);
 assert.ok(gaussian(p,-.4,1).diffusiveFlux!<0);assert.ok(gaussian(p,.4,1).diffusiveFlux!>0);near(later.diffusiveFlux,0);
});
test('zero diffusivity is pure advection and zero mass has no mass-normalized moments',()=>{
 const p={...g,diffusivity:0},q=gaussian(p,.4,.7),initial=gaussian(p,.4-g.velocity*.7,0);
 near(q.concentration,initial.concentration!);near(q.diffusiveFlux,0);near(q.state.varianceChange,0);
 const empty=gaussian({...g,mass:0},.4,1);near(empty.concentration,0);near(empty.flux,0);assert.equal(empty.logConcentration,null);assert.equal(empty.state.massWeightedMean,null);assert.equal(empty.state.massWeightedVariance,null);assert.equal(massWindow({...g,mass:0},9,9.1,0).massInside,0);
});
test('six scaled tail-window references retain logs and tiny physical widths without CDF subtraction',()=>{
 assert.equal(refs.tailCases.length,6);for(const r of refs.tailCases){const q=massWindow(r.parameters,r.left,r.right,0),e=r.expected;near(q.logFraction,Number(e.logFraction),2e-12);
  for(const key of ['fraction','massInside']as const){const expected=Number(e[key]);if(expected===0)assert.equal(q[key],null);else if(expected<1e-300){assert.notEqual(q[key],null);assert.ok(Math.abs(q[key]!-expected)<=Number.MIN_VALUE);}else relative(q[key],expected);}
 }
 assert.ok(massWindow(refs.tailCases[0].parameters,9,9.1,0).fraction!>0);
 const tiny=massWindow(refs.tailCases[5].parameters,0,Number.MIN_VALUE,0);assert.equal(tiny.fraction,null);assert.equal(tiny.massInside,2*Number.MIN_VALUE);
});
test('a field derivative can remain representable after concentration underflows',()=>{
 const p={...g,mass:1,area:1,initialSigma:.125,initialMean:0,velocity:0,diffusivity:0},q=gaussian(p,4.84,0);
 assert.equal(q.concentration,null);assert.ok(q.logConcentration! < -745);assert.ok(q.spatialDerivative!<0);assert.notEqual(q.spatialDerivative,null);
 const distant=gaussian(p,15,0);assert.equal(distant.concentration,null);assert.ok(Number.isFinite(distant.logConcentration));near(distant.flux,0);
});
test('small Gaussian variance, standard deviation and peak changes agree with stable high precision references',()=>{
 for(const r of refs.tinyCases){const q=state({...g,initialSigma:.5,diffusivity:Number(r.diffusivity)},Number(r.time));assert.equal(q.roundedVarianceChange,0);assert.equal(q.roundedSigmaChange,0);
  for(const[key,ref]of [['varianceChange','varianceChange'],['stableSigmaChange','sigmaChange'],['stableRelativePeakChange','relativePeakChange']]as const){const expected=Number(r[ref]);if(expected===0)assert.equal(q[key],null);else relative(q[key],expected);}
 }
});
test('constant cross-sectional area changes concentration and physical flux but not normalized spreading',()=>{
 const q=gaussian(g,.4,.7),r=gaussian({...g,area:g.area*2},.4,.7);near(r.concentration,q.concentration!/2);near(r.flux,q.flux!/2);assert.equal(r.state.variance,q.state.variance);
 near(massWindow({...g,area:g.area*2},-.7,1.1,.7).massInside,massWindow(g,-.7,1.1,.7).massInside!);
});
test('invalid inputs are rejected and accepted extremes never leak NaN or infinity',()=>{
 for(const bad of [NaN,Infinity,-1,5])assert.throws(()=>heat({...h,heatCapacity:bad},0,0),RangeError);
 assert.throws(()=>gaussian({...g,initialSigma:0},0,0),RangeError);assert.throws(()=>gaussian({...g,diffusivity:-1},0,0),RangeError);assert.throws(()=>massWindow(g,1,1,0),RangeError);assert.throws(()=>thermalWindow(h,0,-1,0),RangeError);
 for(const mass of [0,Number.MIN_VALUE,1e-200,4])for(const diffusivity of [0,Number.MIN_VALUE,1])finiteTree(gaussian({...g,mass,diffusivity},16,4));
 for(const contrast of [0,Number.MIN_VALUE,1e-200,.2])for(const conductivity of [0,Number.MIN_VALUE,4])finiteTree(heat({...h,contrast,conductivity,heatCapacity:.25,waveNumber:4},1,4));
});
