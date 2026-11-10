import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/poynting-energy-independent.json';
import {inductionEnergyState as source,inductionCylinderEnergy as cylinder,transversePulseEnergy as pulse,transversePulseSlab as slab,energyOnlyNegativeControl} from '../src/lib/learning-poynting-energy';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from '../src/lib/learning-maxwell';

function close(a:number|null,b:number,tol=2e-9){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);}
function relative(a:number|null,b:number,tol=5e-14){assert.notEqual(a,null);assert.ok(Math.abs(a!/b-1)<tol,`${a} != ${b}`);}
function finite(x:unknown){if(typeof x==='number')assert.ok(Number.isFinite(x));else if(Array.isArray(x))x.forEach(finite);else if(x&&typeof x==='object')Object.values(x).forEach(finite);}
const s={magneticOffset:1,magneticRate:.5,magneticAcceleration:.2,permittivity:1,permeability:1};
const p={amplitude:1,width:1,rightFraction:.5,permittivity:1,permeability:1};

test('81 independent vector energy/cross-product and cylinder quadrature references',()=>{
 for(const r of refs.source.sourceCases){const q=source(r,[.2,-.3,.4],r.time),v=cylinder(r,r.radius,r.height,r.time);
  close(q.energyDensity,r.energyDensity);q.poynting!.forEach((x,i)=>close(x,r.poynting[i]));close(q.fieldToMatterPowerDensity,r.fieldToMatterPowerDensity);
  close(v.totalEnergy,r.cylinderEnergy);close(v.energyRate,r.independentEnergyRate);close(v.outwardPower,r.outwardPoyntingPower);close(v.fieldToMatterPower,r.fieldToMatterPower);
  close(q.groupedResidual,0);close(v.groupedResidual,0);
 }
});
test('source power sign distinguishes material work from prescribed supply',()=>{
 const q=source(s,[1,0,0],1);assert.ok(q.fieldToMatterPowerDensity!<0);assert.ok(q.electricStorageRate!>0);assert.ok(q.poynting![0]<0);
 const reverse=source({...s,magneticAcceleration:-.2},[1,0,0],0);assert.ok(reverse.fieldToMatterPowerDensity!>0);
});
test('zero acceleration changes magnetic storage through boundary flux with zero source work',()=>{
 const q=cylinder({...s,magneticAcceleration:0},1,1,1);assert.equal(q.electricStorageRate,0);assert.equal(q.fieldToMatterPower,0);assert.ok(q.magneticStorageRate!>0);close(q.outwardPower,-q.magneticStorageRate!);
});
test('omitted current reveals SI electric defect hidden by ordinary total residual',()=>{
 const config={magneticOffset:1e-6,magneticRate:.5e-6,magneticAcceleration:.2e-6};
 const a=source(config,[.2,-.3,0],.7),b=source({...config,currentMode:'omitted-current-negative-control'},[.2,-.3,0],.7);
 assert.equal(b.naiveTotalResidual,0);relative(b.groupedResidual,refs.physicalScale.missingConductionCurrent.highPrecisionResidual);
 assert.equal(a.groupedResidual,0);assert.equal(b.maxwellFamilyCompatible,false);assert.ok(b.ampereResidual!.some(x=>x!==0));assert.deepEqual(b.conductionCurrent,[0,0,0]);assert.deepEqual(b.totalCurrent,b.displacementCurrent);
 assert.equal(a.permittivity,eps);assert.equal(a.permeability,mu);
});
test('the spatial origin alone cannot certify a missing-current family',()=>{const q=source({...s,currentMode:'omitted-current-negative-control'},[0,0,0],0);assert.equal(q.groupedResidual,0);assert.equal(q.maxwellFamilyCompatible,false);});
test('empty cylinders have exact zero integrals but tiny nonzero volumes may be unavailable',()=>{
 for(const [R,H]of[[0,1],[1,0]]){const q=cylinder(s,R,H,0);assert.equal(q.totalEnergy,0);assert.equal(q.outwardPower,0);assert.equal(q.groupedResidual,0);}
 const q=cylinder(s,Number.MIN_VALUE,1,0);assert.equal(q.volume,null);assert.equal(q.totalEnergy,null);
});
test('54 independent actual-field pulse energies, fluxes and slab integrals',()=>{
 for(const r of refs.source.pulseCases){const q=pulse(r,r.z,r.time),v=slab(r,r.slabStart,r.slabEnd,r.time);
  close(q.energyDensity,r.energyDensity);close(q.poyntingZ,r.poyntingZ);
  if(r.conditionalEnergyVelocity===null)assert.equal(q.observedEnergyVelocity,null);else close(q.observedEnergyVelocity,r.conditionalEnergyVelocity);
  close(v.energyPerArea,r.slabEnergyPerArea);close(v.energyRatePerArea,r.independentSlabEnergyRate);close(v.stableOutwardPowerPerArea,r.outwardPowerPerArea);
  close(q.wholeLineActualEnergyPerArea,r.wholeLineEnergyPerArea);close(q.energyResidual,0);close(v.balanceResidual,0);
 }
});
test('compatible total energy cancels cross terms but electric/magnetic parts still interfere',()=>{
 const q=pulse(p,0,0);assert.equal(q.magneticEnergyDensity,0);close(q.electricEnergyDensity,.5);close(q.energyDensity,q.componentEnergyReference!);
 assert.equal(q.poyntingZ,0);assert.equal(q.observedEnergyVelocity,0);
});
test('same initial E does not imply the same energy when initial B differs',()=>{
 const split=pulse(p,1,0),right=pulse({...p,rightFraction:1},1,0);assert.equal(split.electricValue,right.electricValue);
 relative(right.energyDensity,2*split.energyDensity!);relative(right.wholeLineActualEnergyPerArea,2*split.wholeLineActualEnergyPerArea!);
});
test('single traveling waves have signed energy velocity c, not merely a nonnegative speed',()=>{
 for(const beta of[0,1]){const q=pulse({...p,rightFraction:beta},1,.3);close(q.observedEnergyVelocity,beta===0?-1:1);close(q.analyticCompatibleVelocity,beta===0?-1:1);}
});
test('balanced tiny time retains Poynting flow lost by subtracting squared profiles',()=>{
 const r=refs.cancellation.balancedFlux,q=pulse(p,r.z,r.time);assert.equal(q.naiveComponentFluxReference,0);relative(q.poyntingZ,r.highPrecisionPoynting);relative(q.analyticCompatibleVelocity,r.conditionalEnergyVelocity);
});
test('nine180-digit thin-slab references retain the true physical width',()=>{
 for(const r of refs.cancellation.tinySlabs){const q=slab({...p,rightFraction:r.rightFraction},r.slabStart,r.slabEnd,r.time);relative(q.energyPerArea,r.highPrecisionEnergyPerArea);relative(q.stableOutwardPowerPerArea,r.highPrecisionOutwardPowerPerArea);assert.ok(q.width>0);}
});
test('whole-line per-area energy is independent of propagation time',()=>{
 for(const beta of[0,.2,.5,1]){const a=pulse({...p,rightFraction:beta},0,0),b=pulse({...p,rightFraction:beta},2,1);assert.equal(a.wholeLineActualEnergyPerArea,b.wholeLineActualEnergyPerArea);}
});
test('absolute energy underflow is not zero; analytic family velocity is separately available',()=>{
 for(const r of refs.cancellation.absoluteEnergyUnderflow){const q=pulse({...p,amplitude:r.amplitude,rightFraction:r.rightFraction},0,0);assert.equal(q.energyDensity,null);assert.equal(q.observedEnergyVelocity,null);assert.equal(q.energyStatus,'arithmetic-range');close(q.analyticCompatibleVelocity,r.analyticConditionalVelocity);}
});
test('zero amplitude has zero energy and an undefined conditional velocity',()=>{
 const q=pulse({...p,amplitude:0},4,0);assert.equal(q.energyDensity,0);assert.equal(q.poyntingZ,0);assert.equal(q.observedEnergyVelocity,null);assert.equal(q.analyticCompatibleVelocity,null);assert.equal(q.observedVelocityStatus,'zero-energy-undefined');assert.equal(q.wholeLineActualEnergyPerArea,0);
});
test('unresolved Gaussian tails remain unavailable while exact symmetric flux can remain zero',()=>{
 const q=pulse({...p,width:.1},4,0);assert.equal(q.energyDensity,null);assert.equal(q.poyntingZ,0);assert.equal(q.observedEnergyVelocity,null);assert.equal(q.analyticCompatibleVelocity,0);
});
test('zero-B negative control uses actual fields rather than compatible reference energies',()=>{
 const q=pulse({...p,rightFraction:1,magneticMode:'zero-magnetic-negative-control'},1,0);assert.equal(q.magneticEnergyDensity,0);assert.equal(q.poyntingZ,0);
 relative(q.energyDensity,.5*q.componentEnergyReference!);assert.ok(q.energyResidual!>0);assert.equal(q.admissibleSourceFreeSolution,false);assert.equal(q.actualAnalyticVelocity,0);assert.equal(q.analyticCompatibleVelocity,1);
 relative(q.wholeLineActualEnergyPerArea,.5*q.wholeLineCompatibleEnergyPerArea!);
});
test('a stationary energy-only negative control passes energy balance but fails Faraday everywhere',()=>{
 for(const x of[-1,0,1]){const q=energyOnlyNegativeControl(x);assert.equal(q.energyResidual,0);assert.equal(q.faradayResidualZ,1);assert.equal(q.maxwellFamilyCompatible,false);}
});
test('fixed slab boundary orientation and empty intervals are explicit',()=>{
 const q=slab({...p,rightFraction:1},-.5,.8,.3);close(q.stableOutwardPowerPerArea,q.rightPoynting!-q.leftPoynting!);
 const empty=slab(p,1,1,0);assert.equal(empty.energyPerArea,0);assert.equal(empty.energyRatePerArea,0);assert.equal(empty.stableOutwardPowerPerArea,0);assert.equal(empty.balanceResidual,0);
});
test('signed amplitude changes fields but not quadratic energy and flux',()=>{
 const a=pulse(p,1,.4),b=pulse({...p,amplitude:-1},1,.4);assert.equal(a.energyDensity,b.energyDensity);assert.equal(a.poyntingZ,b.poyntingZ);
});
test('invalid domains, coefficient signs, representations and excessive displacement fail explicitly',()=>{
 assert.throws(()=>source({...s,currentMode:'bad' as 'required-current'},[0,0,0],0),RangeError);
 assert.throws(()=>cylinder(s,-1,1,0),RangeError);assert.throws(()=>cylinder(s,1,NaN,0),RangeError);assert.throws(()=>pulse({...p,permittivity:0},0,0),RangeError);
 assert.throws(()=>slab(p,1,-1,0),RangeError);assert.throws(()=>slab(p,-5,1,0),RangeError);assert.throws(()=>pulse(p,0,9),RangeError);assert.throws(()=>energyOnlyNegativeControl(Infinity),RangeError);
});
test('observations contain finite numbers or explicit nulls throughout the supported cases',()=>{
 for(const r of refs.source.sourceCases){finite(source(r,[.2,-.3,0],r.time));finite(cylinder(r,r.radius,r.height,r.time));}
 for(const r of refs.source.pulseCases){finite(pulse(r,r.z,r.time));finite(slab(r,r.slabStart,r.slabEnd,r.time));}
});
