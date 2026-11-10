import test from 'node:test';import assert from 'node:assert/strict';
import refs from './fixtures/material-interfaces-independent.json';
import {dielectricInterfaceState as field,dielectricInterfaceMeasures as measure,normalInterfaceCoefficients as coefficients,normalInterfaceState as wave} from '../src/lib/learning-material-interfaces';
import {teachingMaxwellPermittivity as eps,teachingMaxwellPermeability as mu} from '../src/lib/learning-maxwell';
function close(a:number|null,b:number,rel=3e-10,abs=1e-23){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=abs+rel*Math.abs(b),`${a} != ${b}`);}
function relative(a:number|null,b:number,tol=1e-14){assert.notEqual(a,null);assert.ok(Math.abs(a!/b-1)<tol,`${a} != ${b}`);}
function finite(v:unknown){if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finite);}
const d={permittivity1:eps,permittivity2:4*eps,normalDisplacement1:2e-9,freeSurfaceCharge:0,tangentialElectric:1};
const w={permittivity1:1,permittivity2:4,permeability1:1,permeability2:1,amplitude:1,wavenumber1:1};
test('243 independent actual-vector dielectric surface and loop integrals match two-sided fields',()=>{
 for(const r of refs.dielectricCases){const s={...r,permittivity1:r.epsilon1,permittivity2:r.epsilon2},q=field(s,[.3,0,.4]),p=measure(s,1,r.pillboxArea,.4),l=measure(s,.6,1,.4);
  for(const [value,expected]of[[q.medium1.electricField,r.electric1],[q.medium2.electricField,r.electric2],[q.medium1.polarization,r.polarization1],[q.medium2.polarization,r.polarization2]]as const)value!.forEach((x,i)=>close(x,expected[i]));
  close(q.boundSurfaceCharge,r.boundSurfaceCharge);close(p.stableDFlux,r.independentDFlux);close(p.stableVacuumEFlux,r.independentVacuumEFlux);close(l.electricCirculation,r.independentElectricCirculation,1e-10,1e-12);assert.equal(q.electrostaticFamilyCompatible,true);
 }
});
test('normal E may jump while normal D is continuous without free surface charge',()=>{
 const q=field(d,[0,0,.3]);assert.equal(q.medium1.displacement![2],q.medium2.displacement![2]);relative(q.medium1.electricField![2]/q.medium2.electricField![2],4);assert.ok(q.boundSurfaceCharge!<0);assert.equal(q.normalSourceResidual,0);
});
test('vacuum on both sides has exact zero polarization charge even with nonzero free charge',()=>{
 const q=field({...d,permittivity2:eps,freeSurfaceCharge:1e-9},[0,0,0]);assert.equal(q.boundSurfaceCharge,0);assert.deepEqual(q.medium1.polarization,[0,0,0]);assert.deepEqual(q.medium2.polarization,[0,0,0]);close(q.totalSurfaceCharge,1e-9);
});
test('interface queries return both limits rather than a fabricated average',()=>{
 const q=field(d,[.3,0,0]);assert.equal(q.region,'interface-two-sided');assert.equal(q.electricField,null);assert.equal(q.displacement,null);assert.equal(q.potential,null);assert.notEqual(q.medium1.electricField![2],q.medium2.electricField![2]);assert.equal(q.interfacePotential1,q.interfacePotential2);
 const left=field(d,[.3,0,-.4]),right=field(d,[.3,0,.4]);assert.deepEqual(left.electricField,left.medium1.electricField);assert.deepEqual(right.electricField,right.medium2.electricField);
});
test('tangential mismatch creates electrostatic circulation and an x-dependent potential jump',()=>{
 const s={...d,tangentialJump:.3},q=field(s,[.3,0,0]),m=measure(s,.6,1,.4);close(q.stablePotentialJump,-.09);close(m.electricCirculation,-.18);assert.equal(q.electrostaticFamilyCompatible,false);close(m.loopEdgeIntegrals!.reduce<number>((n,x)=>n+x!,0),-.18,1e-10,1e-12);
});
test('equal normal fields require a different free surface charge when permittivity changes',()=>{
 const s={...d,permittivity2:2*eps,normalMode:'equal-normal-field-negative-control' as const},q=field(s,[0,0,.4]);assert.equal(q.stableNormalEJump,0);close(q.requiredFreeSurfaceCharge,2e-9);close(q.boundSurfaceCharge,-2e-9);assert.equal(q.electrostaticFamilyCompatible,false);
 const supported=field({...s,freeSurfaceCharge:2e-9},[0,0,.4]);assert.equal(supported.electrostaticFamilyCompatible,true);assert.equal(supported.normalSourceResidual,0);
});
test('two180-digit tiny free-charge and material jumps survive rounded face subtraction',()=>{
 for(const r of refs.tinyDielectricJumps){const q=field(r,[0,0,.3]);relative(q.stableNormalEJump,r.normalEJump);if(r.boundSurfaceCharge===0)assert.equal(q.boundSurfaceCharge,0);else relative(q.boundSurfaceCharge,r.boundSurfaceCharge);}
 const r=refs.tinyDielectricJumps[0],q=field(r,[0,0,.3]);assert.equal(q.naiveNormalDJump,0);assert.ok(q.stableNormalEJump!>0);const p=measure(r,1,1,.3);assert.equal(p.naiveDFlux,0);assert.equal(p.stableDFlux,r.freeSurfaceCharge);
});
test('zero area,collapsed height and geometric representation loss remain distinct',()=>{
 const zero=measure(d,0,1,.4);assert.equal(zero.stableDFlux,0);assert.equal(zero.normalizationStatus,'zero-area-undefined');
 const collapse=measure({...d,tangentialJump:.3},1,1,0);assert.equal(collapse.stableDFlux,null);assert.equal(collapse.electricCirculation,null);assert.equal(collapse.vertices,null);close(collapse.limitingElectricCirculation,-.3);assert.equal(collapse.geometryStatus,'collapsed-height-two-sided-limit');
 const tiny=measure(d,Number.MIN_VALUE,1,.4);assert.equal(tiny.vertices,null);assert.equal(tiny.geometryStatus,'arithmetic-range');
});
test('unrepresentable source comparisons are unknown rather than certified compatible',()=>{
 const q=field({...d,normalDisplacement1:Number.MIN_VALUE,normalMode:'equal-normal-field-negative-control'},[0,0,.3]);assert.equal(q.normalSourceResidual,null);assert.equal(q.electrostaticFamilyCompatible,null);
});
test('81 independent E/H boundary solves and signed phase-averaged fluxes match coefficients',()=>{
 for(const r of refs.waveCases){const s={permittivity1:r.epsilon1,permittivity2:r.epsilon2,permeability1:r.mu1,permeability2:r.mu2,amplitude:1,wavenumber1:1},q=coefficients(s);
  close(q.reflection,r.independentReflectionAmplitude,1e-9,2e-16);close(q.transmission,r.independentTransmissionAmplitude);close(q.incidentCycleMean,r.incidentAveragePower);close(q.reflectedSignedCycleMean,r.reflectedSignedAveragePower,1e-8,1e-20);close(q.transmittedCycleMean,r.transmittedAveragePower);close(q.reflectance!+q.transmittance!,1);assert.equal(q.noSheetFamilyCompatible,true);
 }
});
test('96 independent spacetime differences verify bulk fields,energy and both first-order equations',()=>{
 for(const r of refs.spacetimeCases){const q=wave(r,r.z,r.time);
  for(const side of['medium1','medium2']as const){const s=q[side],v=r[side];close(s.electricValue,v.electricValue,1e-10,1e-12);close(s.magneticIntensity,v.magneticIntensity,1e-10,1e-12);close(s.energyDensity,v.energyDensity,1e-10,1e-12);close(s.poyntingZ,v.poyntingZ,1e-10,1e-12);for(const[a,b]of[[s.electricSpatialDerivative,v.electricZ],[s.electricTimeDerivative,v.electricT],[s.magneticIntensitySpatialDerivative,v.magneticIntensityZ],[s.magneticIntensityTimeDerivative,v.magneticIntensityT]])close(a,b!,2e-9,2e-10);close(s.faradayResidualY,0,1e-9,1e-12);close(s.ampereResidualX,0,1e-9,1e-12);}
 }
});
test('seven180-digit impedance references include a product cancellation missed by naive square roots',()=>{
 for(const r of refs.highPrecisionImpedanceCases){const q=coefficients(r);if(r.reflection===0)assert.equal(q.reflection,0);else relative(q.reflection,r.reflection);relative(q.transmission,r.transmission);if(r.reflectance===0)assert.equal(q.reflectance,0);else relative(q.reflectance,r.reflectance);relative(q.transmittance,r.transmittance);}
 const q=coefficients(refs.highPrecisionImpedanceCases[2]);assert.equal(q.naiveReflection,0);assert.ok(q.reflection<0);assert.ok(q.reflectance!>0);assert.equal(q.impedanceMatched,false);
});
test('amplitude and power transmission differ when impedance changes',()=>{
 const q=coefficients({...w,permittivity2:.25});assert.ok(q.transmission>1);assert.ok(q.transmittance!<1);close(q.reflectance!+q.transmittance!,1);
});
test('impedance matching can suppress reflection while speed and wavelength change',()=>{
 const q=coefficients({...w,permittivity2:2,permeability2:2});assert.equal(q.reflection,0);assert.equal(q.transmission,1);assert.notEqual(q.medium1.speed,q.medium2.speed);relative(q.wavenumber2,2);assert.equal(q.impedanceMatched,true);
});
test('interface samples separate E/H side limits and do not average different B',()=>{
 const q=wave({...w,permittivity2:2,permeability2:2},0,.3);assert.equal(q.electricValue,null);assert.equal(q.magneticIntensity,null);close(q.medium1.electricValue,q.medium2.electricValue!);close(q.medium1.magneticIntensity,q.medium2.magneticIntensity!);relative(q.medium2.magneticValue,2*q.medium1.magneticValue!);
});
test('forcing no reflection can require a sheet current rather than an impossible Maxwell system',()=>{
 for(const r of refs.waveCases){const s={permittivity1:r.epsilon1,permittivity2:r.epsilon2,permeability1:r.mu1,permeability2:r.mu2,amplitude:1,wavenumber1:1,reflectionMode:'forced-zero-reflection' as const},q=coefficients(s),v=wave(s,0,0);
  close(q.requiredSheetCurrentAmplitude,r.forcedZeroReflectionRequiredSheetCurrentX,1e-8,1e-18);close(v.boundary.requiredSheetWork,2*r.forcedZeroReflectionSheetWork,1e-8,1e-18);assert.equal(q.noSheetFamilyCompatible,q.impedanceMatched);assert.equal(v.boundary.electricLeft,v.boundary.electricRight);
 }
});
test('required sheet work has absorption or supply sign depending on the two impedances',()=>{
 const a=wave({...w,reflectionMode:'forced-zero-reflection'},0,0),b=wave({...w,permittivity2:.25,reflectionMode:'forced-zero-reflection'},0,0);assert.ok(a.boundary.requiredSheetWork!<0);assert.ok(b.boundary.requiredSheetWork!>0);close(a.boundary.naivePowerDifference,a.boundary.requiredSheetWork!);
});
test('zero frequency has a static field and no cycle average',()=>{
 const q=wave({...w,wavenumber1:0},-.4,2);assert.equal(q.cycleStatus,'no-cycle-static-family');assert.equal(q.incidentCycleMean,null);assert.equal(q.observedCycleReflectance,null);assert.equal(q.medium1.electricTimeDerivative,0);assert.ok(q.medium1.poyntingZ!>0);
});
test('a nonzero frequency lost to arithmetic is distinct from the static family',()=>{
 const q=wave({...w,permittivity1:4,permeability1:4,wavenumber1:Number.MIN_VALUE},0,0);assert.equal(q.omega,null);assert.equal(q.cycleStatus,'arithmetic-range');assert.equal(q.incidentCycleMean,null);assert.ok(q.boundary.electricLeft!>0);
});
test('zero amplitude gives zero fields and undefined measured power ratios while formal coefficients remain',()=>{
 const q=wave({...w,amplitude:0,reflectionMode:'forced-zero-reflection'},0,0);assert.equal(q.medium1.energyDensity,0);assert.equal(q.medium2.poyntingZ,0);assert.equal(q.incidentCycleMean,0);assert.equal(q.observedCycleTransmittance,null);assert.equal(q.noSheetFamilyCompatible,true);assert.ok(q.referenceReflectance!>0);
});
test('underflow of nonzero absolute energy is unavailable rather than physical zero',()=>{
 const q=wave({...w,amplitude:1e-200},-.4,.3);assert.ok(q.medium1.electricValue!>0);assert.equal(q.medium1.energyDensity,null);assert.equal(q.incidentCycleMean,null);assert.equal(q.observedCycleReflectance,null);assert.ok(q.reflectance!>0);
});
test('signed amplitude reverses fields but preserves quadratic energy and flux',()=>{
 const a=wave(w,-.4,.3),b=wave({...w,amplitude:-1},-.4,.3);assert.equal(a.medium1.electricValue,-b.medium1.electricValue!);assert.equal(a.medium1.energyDensity,b.medium1.energyDensity);assert.equal(a.medium1.poyntingZ,b.medium1.poyntingZ);
});
test('invalid inputs,sparse positions,unknown modes and phase overbudget fail explicitly',()=>{
 assert.throws(()=>field({...d,permittivity1:0},[0,0,0]),RangeError);assert.throws(()=>field(d,[0,,0]as unknown as [number,number,number]),RangeError);assert.throws(()=>field({...d,normalMode:'bad' as 'specified-charge'},[0,0,0]),RangeError);assert.throws(()=>measure(d,-1,1,1),RangeError);assert.throws(()=>wave({...w,reflectionMode:'bad'as 'matched-boundary'},0,0),RangeError);assert.throws(()=>wave({...w,amplitude:NaN},0,0),RangeError);assert.throws(()=>wave({amplitude:1,wavenumber1:1},0,1),RangeError);assert.throws(()=>wave({...w,permittivity1:1e-12,permeability1:1e-12,permittivity2:4,permeability2:4},4,0),RangeError);
});
test('all selected observations are finite or explicit nulls and do not mutate input positions',()=>{
 const p:[number,number,number]=[.2,.3,.4],q=field(d,p);q.position[0]=2;assert.equal(p[0],.2);
 for(const r of refs.spacetimeCases)finite(wave(r,r.z,r.time));for(const r of refs.tinyDielectricJumps)finite(field(r,[0,0,.3]));finite(measure(d,0,1,0));finite(wave({...w,amplitude:1e-200},0,0));
});
