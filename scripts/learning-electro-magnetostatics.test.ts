import test from 'node:test';import assert from 'node:assert/strict';
import refs from './fixtures/electro-magnetostatics-independent.json';
import {electrostaticSphere as sphere,magnetostaticCylinder as cylinder,teachingVacuumPermeability,type StaticVector,type UniformChargedSphere,type UniformSteadyCylinder} from '../src/lib/learning-electro-magnetostatics';
const close=(a:number,b:number,tol=3e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const relative=(a:number,b:number,tol=3e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(Number.MIN_VALUE,Math.abs(b)),`${a} != ${b}`);
const S:UniformChargedSphere={radius:1,totalCharge:1,coulombConstant:1},C:UniformSteadyCylinder={radius:1,totalCurrent:1,permeability:1};
test('63 spherical fields and potentials agree with independent shell quadrature and potential gradients',()=>{
 for(const r of refs.spheres){const q=sphere({radius:r.radius,totalCharge:r.totalCharge,coulombConstant:1},[r.observationRadius,0,0]);close(q.sourcePotential!,r.potentialOverK);close(q.sourceRadialField!,r.gaussPredictionOverK);close(q.sourceField![0],r.negativePotentialGradientOverK,r.boundary?2e-5:2e-9);close(q.gaussResidual!,0);}
});
test('24 three-dimensional total fields agree with independent potential quadrature and finite differences',()=>{
 for(const r of refs.vectorSpheres){const q=sphere({radius:r.radius,totalCharge:r.totalCharge,coulombConstant:1,backgroundField:r.backgroundField as StaticVector},r.observation as StaticVector);close(q.totalPotential!,r.totalPotential);q.totalField!.forEach((x,i)=>close(x,r.negativeIndependentGradient[i],2e-9));}
});
test('the worked sphere example preserves SI field,potential,enclosed charge and flux',()=>{
 const q=sphere({radius:1,totalCharge:1e-9},[.5,0,0]);close(q.sourceRadialField!,4.5);close(q.sourcePotential!,12.375);relative(q.enclosedCharge!,.125e-9);close(q.closedElectricFlux!,4.5*Math.PI);close(q.sourceVolumeDensity!,3e-9/(4*Math.PI));
});
test('a regular charged center has zero field and undefined radial direction,not a source singularity',()=>{
 const q=sphere(S,[0,0,0]);assert.deepEqual(q.sourceField,[0,0,0]);assert.equal(q.sourceFieldDirection,null);assert.equal(q.radialUnit,null);close(q.sourcePotential!,1.5);close(q.sourceDivergence!,3);close(q.sourceRadialDerivative!,1);assert.equal(q.sourceFieldStatus,'finite');
});
test('a collapsed Gaussian surface has zero flux but supplies no flux-normalized point field',()=>{
 const q=sphere({...S,backgroundField:[.1,.2,.3]},[0,0,0]);assert.equal(q.gaussianSurfaceArea,0);assert.equal(q.enclosedCharge,0);assert.equal(q.closedElectricFlux,0);assert.equal(q.meanRadialFromFlux,null);assert.equal(q.normalizationStatus,'zero-surface-measure');assert.deepEqual(q.totalField,[.1,.2,.3]);
});
test('sphere interface fields and potential are continuous while classical field derivatives are unavailable',()=>{
 const q=sphere({radius:1,totalCharge:1e-9},[1,0,0]);assert.equal(q.region,'boundary');close(q.sourceRadialField!,9);close(q.sourcePotential!,9);assert.equal(q.sourceRadialDerivative,null);assert.equal(q.sourceDivergence,null);assert.equal(q.densityAtObservation,null);assert.equal(q.derivativeStatus,'boundary-one-sided-only');close(q.interfaceLimits.radialDerivativeInside!,9);close(q.interfaceLimits.radialDerivativeOutside!,-18);close(q.interfaceLimits.divergenceInside!,27);assert.equal(q.interfaceLimits.divergenceOutside,0);close(q.gaussResidual!,0);
});
test('near-interface observations remain interior or exterior without tolerance snapping',()=>{
 const a=sphere(S,[1-1e-12,0,0]),b=sphere(S,[1+1e-12,0,0]);assert.equal(a.region,'interior');assert.equal(b.region,'exterior');assert.equal(a.derivativeStatus,'finite');assert.equal(b.derivativeStatus,'finite');close(a.sourceRadialDerivative!,1);close(b.sourceRadialDerivative!,-2/(1+1e-12)**3);
 const x=cylinder(C,[1-1e-12,0,0]),y=cylinder(C,[1+1e-12,0,0]);assert.equal(x.region,'interior');assert.equal(y.region,'exterior');assert.ok(x.sourceRadialDerivative!>0&&y.sourceRadialDerivative!<0);
});
test('zero sources remove the interface derivative jump and retain a supplied background',()=>{
 const a=sphere({...S,totalCharge:0,backgroundField:[0,0,2]},[1,0,0]);assert.equal(a.derivativeStatus,'finite');assert.equal(a.sourceRadialDerivative,0);assert.equal(a.sourceDivergence,0);assert.deepEqual(a.sourceField,[0,0,0]);assert.deepEqual(a.totalField,[0,0,2]);assert.deepEqual(a.totalFieldDirection,[0,0,1]);
 const b=cylinder({...C,totalCurrent:0,backgroundField:[.1,.2,.3]},[1,0,0]);assert.equal(b.derivativeStatus,'finite');assert.equal(b.sourceRadialDerivative,0);assert.equal(b.sourceCurlZ,0);assert.deepEqual(b.sourceField,[0,0,0]);assert.deepEqual(b.totalField,[.1,.2,.3]);
});
test('charge reversal changes source field,potential and flux without reversing the background',()=>{
 const a=sphere({...S,backgroundField:[.3,0,0]},[.5,0,0]),b=sphere({...S,totalCharge:-1,backgroundField:[.3,0,0]},[.5,0,0]);close(b.sourceRadialField!,-a.sourceRadialField!);close(b.sourcePotential!,-a.sourcePotential!);close(b.closedElectricFlux!,-a.closedElectricFlux!);close(a.totalField![0]+b.totalField![0],.6);
});
test('a common spatial dilation rescales isolated electric field,potential and density differently',()=>{
 const a=sphere(S,[.3,.4,.2]),b=sphere({...S,radius:2},[.6,.8,.4]);a.sourceField!.forEach((x,i)=>close(b.sourceField![i],x/4));close(b.sourcePotential!,a.sourcePotential!/2);close(b.sourceVolumeDensity!,a.sourceVolumeDensity!/8);close(b.enclosedCharge!,a.enclosedCharge!);close(b.closedElectricFlux!,a.closedElectricFlux!);
});
test('sphere field vectors rotate covariantly in three dimensions',()=>{
 const rotate=([x,y,z]:StaticVector):StaticVector=>[z,x,y],p:StaticVector=[.3,.4,.2],bg:StaticVector=[.1,-.2,.3];const a=sphere({...S,backgroundField:bg},p),b=sphere({...S,backgroundField:rotate(bg)},rotate(p));assert.deepEqual(b.totalField,rotate(a.totalField!));close(b.sourcePotential!,a.sourcePotential!);
});
test('uniform electric background changes a point field while preserving the closed flux',()=>{
 const a=sphere({radius:1,totalCharge:1e-9},[.5,0,0]),b=sphere({radius:1,totalCharge:1e-9,backgroundField:[3,0,0]},[.5,0,0]);close(b.totalField![0],7.5);close(b.meanRadialFromFlux!,4.5);close(b.closedElectricFlux!,a.closedElectricFlux!);close(b.totalPotential!,10.875);assert.notEqual(b.totalField![0],b.meanRadialFromFlux);
});
test('numerical closed-sphere integration verifies total-field flux with a symmetry-breaking background',()=>{
 const source={...S,backgroundField:[.3,-.2,.1]as StaticVector};
 for(const r of[.5,1,2]){let flux=0;const n=12,m=24;for(let i=0;i<n;i++){const z=-1+(i+.5)*2/n,h=Math.sqrt(1-z*z);for(let j=0;j<m;j++){const phi=2*Math.PI*(j+.5)/m,normal:StaticVector=[h*Math.cos(phi),h*Math.sin(phi),z],q=sphere(source,normal.map(x=>r*x)as StaticVector);flux+=q.totalField!.reduce((v,x,k)=>v+x*normal[k],0)*4*Math.PI*r*r/(n*m);}}close(flux,sphere(source,[r,0,0]).closedElectricFlux!,2e-12);}
});
test('three-dimensional divergence agrees with Cartesian field differences away from the interface',()=>{
 for(const p of[[.1,.2,.3],[2,1,1]]as StaticVector[]){let divergence=0;const h=1e-4;for(let j=0;j<3;j++){const a=[...p]as StaticVector,b=[...p]as StaticVector;a[j]+=h;b[j]-=h;divergence+=(sphere(S,a).sourceField![j]-sphere(S,b).sourceField![j])/(2*h);}close(divergence,sphere(S,p).sourceDivergence!,2e-8);}
});
test('interface potential differences have reduced convergence order despite a high-order stencil',()=>{
 const errors=[];for(const r of refs.boundaryConvergence){const h=r.step,v=(x:number)=>sphere(S,[x,0,0]).sourcePotential!,g=-(-v(1+2*h)+8*v(1+h)-8*v(1-h)+v(1-2*h))/(12*h);close(g,r.negativePotentialGradient,2e-10);errors.push(Math.abs(g-1));}
 for(let i=0;i<errors.length-1;i++)assert.ok(errors[i]/errors[i+1]>1.8&&errors[i]/errors[i+1]<2.2);
});
test('tiny nonzero radii preserve an available electric field and flux when enclosed charge underflows',()=>{
 const q=sphere({radius:1,totalCharge:1},[1e-110,0,0]);assert.equal(q.region,'interior');relative(q.sourceRadialField!,9e-101);assert.equal(q.enclosedCharge,null);assert.equal(q.enclosedChargeStatus,'numerical-range');assert.ok(q.closedElectricFlux!>0);assert.equal(q.sourceFieldStatus,'finite');
});
test('unavailable small-source observations remain distinct from a true zero source',()=>{
 const q=sphere({...S,radius:4,totalCharge:Number.MIN_VALUE},[4,0,0]);assert.equal(q.totalCharge,Number.MIN_VALUE);assert.equal(q.enclosedCharge,Number.MIN_VALUE);assert.equal(q.sourceField,null);assert.equal(q.sourceFieldStatus,'numerical-range');assert.equal(q.sourceVolumeDensity,null);assert.equal(q.gaussResidual,null);
});
test('63 magnetic fields agree with independent Biot-Savart filament integrals',()=>{
 for(const r of refs.cylinders){const q=cylinder({radius:r.radius,totalCurrent:r.totalCurrent,permeability:1},[r.observationRadius,0,0]);close(q.sourceField![1],r.biotSavartFieldOverMu0);close(q.sourceAzimuthalField!,r.amperePredictionOverMu0);close(q.ampereResidual!,0);assert.equal(q.sourceField![2],0);}
});
test('the worked current-cylinder example preserves SI field,current density and circulation',()=>{
 const q=cylinder({radius:1,totalCurrent:1},[.5,0,0]);assert.equal(teachingVacuumPermeability,4*Math.PI*1e-7);relative(q.sourceAzimuthalField!,1e-7);close(q.enclosedCurrent!,.25);relative(q.circulation!,Math.PI*1e-7);relative(q.sourceCurlZ!,4e-7);close(q.sourceCurrentDensity!,1/Math.PI);
});
test('the current axis has a finite zero vector and nonzero curl without a unique azimuthal direction',()=>{
 const q=cylinder(C,[0,0,3]);assert.deepEqual(q.sourceField,[0,0,0]);assert.equal(q.azimuthalUnit,null);assert.equal(q.sourceFieldDirection,null);close(q.sourceCurlZ!,1/Math.PI);assert.equal(q.circulation,0);assert.equal(q.enclosedCurrent,0);assert.equal(q.meanAzimuthalFromCirculation,null);assert.equal(q.normalizationStatus,'zero-loop-length');
});
test('a nonzero magnetic background survives at the current axis despite zero collapsed-loop circulation',()=>{
 const q=cylinder({...C,backgroundField:[.1,.2,.3]},[0,0,0]);assert.deepEqual(q.totalField,[.1,.2,.3]);assert.equal(q.circulation,0);assert.equal(q.meanAzimuthalFromCirculation,null);
});
test('current-cylinder interface fields are continuous while derivative and curl retain one-sided limits',()=>{
 const q=cylinder({radius:1,totalCurrent:1},[1,0,0]);assert.equal(q.region,'boundary');relative(q.sourceAzimuthalField!,2e-7);assert.equal(q.sourceRadialDerivative,null);assert.equal(q.sourceCurlZ,null);assert.equal(q.currentDensityAtObservation,null);assert.equal(q.derivativeStatus,'boundary-one-sided-only');relative(q.interfaceLimits.radialDerivativeInside!,2e-7);relative(q.interfaceLimits.radialDerivativeOutside!,-2e-7);relative(q.interfaceLimits.curlInside!,4e-7);assert.equal(q.interfaceLimits.curlOutside,0);relative(q.expectedAmpereCirculation!,teachingVacuumPermeability);
});
test('current sign and the right-hand basis determine the full Cartesian magnetic direction',()=>{
 const a=cylinder(C,[1,0,0]),b=cylinder(C,[0,1,0]),c=cylinder({...C,totalCurrent:-1},[1,0,0]);assert.ok(a.sourceField![1]>0);assert.ok(b.sourceField![0]<0);close(c.sourceField![1],-a.sourceField![1]);close(c.circulation!,-a.circulation!);close(c.sourceCurrentDensity!,-a.sourceCurrentDensity!);
});
test('the infinite-cylinder source is independent of axial position',()=>{
 for(const z of[-12,0,12])assert.deepEqual(cylinder(C,[.3,.4,z]).sourceField,cylinder(C,[.3,.4,0]).sourceField);
});
test('common cylinder dilation changes field by inverse length and density by inverse area',()=>{
 const a=cylinder(C,[.3,.4,1]),b=cylinder({...C,radius:2},[.6,.8,2]);a.sourceField!.forEach((x,i)=>close(b.sourceField![i],x/2));close(b.sourceCurrentDensity!,a.sourceCurrentDensity!/4);close(b.enclosedCurrent!,a.enclosedCurrent!);close(b.circulation!,a.circulation!);
});
test('outside sphere and cylinder fields have different radial powers and different integral geometry',()=>{
 const e1=sphere(S,[2,0,0]),e2=sphere(S,[4,0,0]),b1=cylinder(C,[2,0,0]),b2=cylinder(C,[4,0,0]);close(e1.sourceRadialField!,4*e2.sourceRadialField!);close(b1.sourceAzimuthalField!,2*b2.sourceAzimuthalField!);close(e1.closedElectricFlux!,e2.closedElectricFlux!);close(b1.circulation!,b2.circulation!);
});
test('a uniform magnetic background changes local field without changing Ampere circulation',()=>{
 const source={radius:1,totalCurrent:1},a=cylinder(source,[.5,0,0]),b=cylinder({...source,backgroundField:[0,3e-7,0]},[.5,0,0]);relative(b.totalField![1],4e-7);relative(b.meanAzimuthalFromCirculation!,1e-7);relative(a.circulation!,b.circulation!);
});
test('independent circular line summation verifies total-field circulation with a uniform background',()=>{
 const source={...C,backgroundField:[.2,-.1,.4]as StaticVector};for(const r of[.5,1,2]){let integral=0;const n=96;for(let i=0;i<n;i++){const theta=2*Math.PI*(i+.5)/n,co=Math.cos(theta),si=Math.sin(theta),q=cylinder(source,[r*co,r*si,0]);integral+=(-q.totalField![0]*si+q.totalField![1]*co)*r*2*Math.PI/n;}close(integral,cylinder(source,[r,0,0]).circulation!,3e-12);}
});
test('Cartesian curl and divergence remain regular at the axis and agree away from the interface',()=>{
 const h=1e-4;for(const[x,y]of[[0,0],[.2,.3],[2,1]]){const xp=cylinder(C,[x+h,y,0]).sourceField!,xm=cylinder(C,[x-h,y,0]).sourceField!,yp=cylinder(C,[x,y+h,0]).sourceField!,ym=cylinder(C,[x,y-h,0]).sourceField!;const curl=(xp[1]-xm[1]-yp[0]+ym[0])/(2*h),div=(xp[0]-xm[0]+yp[1]-ym[1])/(2*h);close(curl,cylinder(C,[x,y,0]).sourceCurlZ!,2e-8);close(div,0,2e-8);}
});
test('tiny nonzero current radii retain a representable field without inventing a finite circulation',()=>{
 const q=cylinder({...C,radius:.25},[Number.MIN_VALUE,0,0]);assert.ok(q.sourceField![1]>0);assert.equal(q.sourceFieldStatus,'finite');assert.equal(q.enclosedCurrent,null);assert.equal(q.circulation,null);assert.equal(q.normalizationStatus,'numerical-range');assert.ok(q.amperianLoopLength!>0);
});
test('zero source current is not confused with underflowed nonzero current',()=>{
 const a=cylinder({...C,totalCurrent:0},[1,0,0]),b=cylinder({radius:4,totalCurrent:Number.MIN_VALUE},[1,0,0]);assert.deepEqual(a.sourceField,[0,0,0]);assert.equal(a.sourceFieldStatus,'finite');assert.equal(b.sourceField,null);assert.equal(b.sourceFieldStatus,'numerical-range');assert.equal(b.totalCurrent,Number.MIN_VALUE);
});
test('returned observations and source/background vectors do not alias caller input or each other',()=>{
 const p:StaticVector=[.5,0,0],bg:StaticVector=[1,2,3],q=sphere({...S,backgroundField:bg},p);p[0]=2;bg[0]=20;assert.equal(q.observation[0],.5);assert.equal(q.backgroundField[0],1);const total=q.totalField![0];q.sourceField![0]=100;assert.equal(q.totalField![0],total);
});
test('available and unavailable cases never leak nonfinite numbers into model output',()=>{
 const check=(v:unknown):void=>{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(check);};for(const x of[0,Number.MIN_VALUE,1e-110,1,16]){check(sphere({...S,totalCharge:Number.MIN_VALUE},[x,0,0]));check(cylinder({...C,totalCurrent:Number.MIN_VALUE},[x,0,0]));}
});
test('source radii,3D geometry,coefficients and backgrounds are validated without silent clamping',()=>{
 for(const radius of[0,-1,5,NaN]){assert.throws(()=>sphere({...S,radius},[0,0,0]));assert.throws(()=>cylinder({...C,radius},[0,0,0]));}
 assert.throws(()=>sphere(S,[Infinity,0,0]));assert.throws(()=>sphere(S,new Array(3)as StaticVector));assert.throws(()=>sphere({...S,coulombConstant:0},[1,0,0]));assert.throws(()=>sphere({...S,backgroundField:[1e7,0,0]},[1,0,0]));assert.throws(()=>cylinder({...C,permeability:0},[1,0,0]));assert.throws(()=>cylinder({...C,backgroundField:[2,0,0]},[1,0,0]));assert.throws(()=>cylinder({...C,totalCurrent:Infinity},[1,0,0]));
});
