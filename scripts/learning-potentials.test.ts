import test from 'node:test';import assert from 'node:assert/strict';
import refs from './fixtures/potentials-independent.json';
import {quadraticPotentialObservation as observe,quadraticPotentialPathWork as work,cylinderVectorPotential as potential,cylinderPotentialRectangle as rectangle,type PotentialMatrix,type QuadraticPotential,type CylinderPotential}from '../src/lib/learning-potentials';
import {magnetostaticCylinder,type StaticVector}from '../src/lib/learning-electro-magnetostatics';
const close=(a:number,b:number,tol=4e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const relative=(a:number,b:number,tol=5e-12)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(Number.MIN_VALUE,Math.abs(b)),`${a} != ${b}`);
const S:QuadraticPotential={hessian:[[2,0,0],[0,-1,0],[0,0,0]],linear:[1,0,0],reference:5};
const C:CylinderPotential={radius:1,current:1,permeability:1};
test('36 quadratic potentials,fields and divergences agree with independent NumPy references',()=>{
 for(const r of refs.scalarPaths){const s={hessian:r.matrix as PotentialMatrix,linear:r.linear as StaticVector,reference:r.referenceConstant},q=observe(s,r.end as StaticVector);close(q.potential!,r.endPotential);q.field!.forEach((x,i)=>close(x,r.endField[i]));close(q.divergence,r.divergence);}
});
test('straight and curved product path integrals agree with 36 independent SciPy path pairs',()=>{
 for(const r of refs.scalarPaths){const q=work({hessian:r.matrix as PotentialMatrix,linear:r.linear as StaticVector,reference:r.referenceConstant},r.start as StaticVector,r.end as StaticVector,r.bezierControl as StaticVector);close(q.potentialDifference!,r.potentialDifference);close(q.straightFieldIntegral!,r.fieldIntegrals[0]);close(q.bezierFieldIntegral!,r.fieldIntegrals[1]);close(q.bezierResidual!,0);}
});
test('36 gauged vector potentials and curls match independent 3D references',()=>{
 for(const r of refs.vectorCurls){const q=potential({radius:r.radius,current:r.current,permeability:1,gaugeQuadratic:r.gaugeQuadratic,gaugeCubic:r.gaugeCubic},r.observation as StaticVector);q.vectorPotential!.forEach((x,i)=>close(x,r.vectorPotential[i]));q.magneticField!.forEach((x,i)=>close(x,r.independentCurl[i],2e-9));}
});
test('24 actual gauged rectangle line integrals agree with independent magnetic surface fluxes',()=>{
 for(const r of refs.rectangles){const q=rectangle({radius:r.radius,current:r.current,permeability:1,gaugeQuadratic:.7,gaugeCubic:-.3,gaugeAxialConstant:.2},r.left,r.right,r.bottom,r.top,r.y);close(q.circulation!,r.independentMagneticFluxOverMu);close(q.gaugeClosedIntegral!,0);}
});
test('worked scalar example preserves field,energy and distinct field/external work signs',()=>{
 const q=work(S,[0,0,0],[1,1,0],[0,2,1],2e-9);close(q.initial.potential!,5);close(q.final.potential!,6.5);assert.deepEqual(q.final.field,[-3,1,0]);close(q.potentialDifference!,1.5);relative(q.fieldWorkBezier!,-3e-9);relative(q.potentialEnergyChange!,3e-9);relative(q.quasistaticExternalWork!,3e-9);close(q.final.divergence,-1);relative(q.final.chargeDensity!,-1/(4*Math.PI*9e9));
});
test('reference shifts change absolute potentials but not fields,work or potential differences',()=>{
 const a=work(S,[0,0,0],[1,1,0],[0,2,1],2),b=work({...S,reference:100},[0,0,0],[1,1,0],[0,2,1],2);assert.deepEqual(a.final.field,b.final.field);close(b.final.potential!-a.final.potential!,95);assert.equal(a.potentialDifference,b.potentialDifference);assert.equal(a.fieldWorkBezier,b.fieldWorkBezier);assert.equal(a.potentialEnergyChange,b.potentialEnergyChange);
});
test('a small nonzero potential difference survives rounding of the two shifted endpoint potentials',()=>{
 const s:QuadraticPotential={hessian:[[0,0,0],[0,0,0],[0,0,0]],linear:[1,0,0],reference:100};const q=work(s,[0,0,0],[1e-16,0,0],[.5e-16,0,0]);assert.equal(q.naiveEndpointDifference,0);assert.equal(q.potentialDifference,1e-16);relative(q.straightFieldIntegral!,-1e-16);
});
test('charge sign and zero charge alter force and energy without altering the prescribed field',()=>{
 const a=observe(S,[1,1,0],2),b=observe(S,[1,1,0],-2),z=observe(S,[1,1,0],0);assert.deepEqual(a.field,b.field);assert.deepEqual(a.field,z.field);a.force!.forEach((x,i)=>close(b.force![i],-x));close(b.potentialEnergy!,-a.potentialEnergy!);assert.deepEqual(z.force,[0,0,0]);assert.equal(z.potentialEnergy,0);
});
test('reversed and closed curved paths retain orientation and zero endpoint work',()=>{
 const a=work(S,[0,0,0],[1,1,0],[0,2,1]),b=work(S,[1,1,0],[0,0,0],[0,2,1]),z=work(S,[1,1,0],[1,1,0],[2,-1,3]);close(a.bezierFieldIntegral!,-b.bezierFieldIntegral!);close(z.potentialDifference!,0);close(z.bezierFieldIntegral!,0);
});
test('constant-coordinate paths at the domain boundary do not drift outside the domain',()=>{
 for(const value of [-4,4]){const q=work(S,[value,value,value],[value,value,value],[value,value,value]);assert.equal(q.bezierFieldIntegral,0);assert.equal(q.potentialDifference,0);}
});
test('lost quadratic terms are unavailable while a representable linear field remains finite',()=>{
 const q=observe({hessian:[[1,0,0],[0,0,0],[0,0,0]],linear:[0,0,0],reference:0},[1e-200,0,0]);assert.equal(q.potential,null);assert.equal(q.potentialStatus,'arithmetic-range');assert.equal(q.field![0],-1e-200);assert.equal(q.fieldStatus,'finite');
});
test('exact zero coefficients remain exact physical zeros at tiny positions',()=>{
 const q=observe({hessian:[[0,0,0],[0,0,0],[0,0,0]],linear:[0,0,0],reference:0},[Number.MIN_VALUE,0,0]);assert.equal(q.potential,0);assert.deepEqual(q.field,[0,0,0]);assert.equal(q.potentialStatus,'finite');
});
test('Poisson density underflow is distinct from a representable nonzero divergence',()=>{
 const q=observe({hessian:[[Number.MIN_VALUE,0,0],[0,0,0],[0,0,0]],linear:[0,0,0],reference:0},[0,0,0]);assert.equal(q.divergence,-Number.MIN_VALUE);assert.equal(q.chargeDensity,null);assert.equal(q.chargeDensityStatus,'arithmetic-range');assert.deepEqual(q.field,[0,0,0]);
});
test('scalar gradients and Hessians agree with independent Cartesian finite differences',()=>{
 const x:StaticVector=[.7,-.3,.4],h=1e-5,q=observe(S,x);for(let j=0;j<3;j++){const a=[...x]as StaticVector,b=[...x]as StaticVector;a[j]+=h;b[j]-=h;close(q.field![j],-(observe(S,a).potential!-observe(S,b).potential!)/(2*h),2e-10);}
});
test('nonsymmetric Hessians,malformed vectors and nonfinite or unsupported domains are rejected',()=>{
 assert.throws(()=>observe({...S,hessian:[[1,2,0],[0,1,0],[0,0,1]]},[0,0,0]),RangeError);assert.throws(()=>observe(S,[0,0]as unknown as StaticVector),RangeError);assert.throws(()=>observe({...S,reference:Infinity},[0,0,0]),RangeError);assert.throws(()=>work(S,[0,0,0],[5,0,0],[0,0,0]),RangeError);assert.throws(()=>observe(S,[0,0,0],5),RangeError);
});
test('worked cylinder example has regular axis potential and correctly oriented curl',()=>{
 const c={radius:1,current:1},axis=potential(c,[0,0,0]),half=potential(c,[.5,0,0]),outside=potential(c,[2,0,0]);relative(axis.sourceAz!,1e-7);assert.deepEqual(axis.magneticField,[0,0,0]);relative(half.sourceAz!,7.5e-8);relative(half.magneticField![1],1e-7);relative(outside.sourceAz!,-2e-7*Math.log(2));
});
test('vector potential remains differentiable at the current interface while second derivatives do not',()=>{
 const q=potential(C,[1,0,0]);assert.equal(q.sourceAz,0);assert.deepEqual(q.sourceAzGradient,[-1/(2*Math.PI),0,0]);close(q.magneticField![1],1/(2*Math.PI));assert.equal(q.sourceAzSecondRadialDerivative,null);assert.equal(q.sourceAzLaplacian,null);assert.equal(q.secondDerivativeStatus,'boundary-one-sided-only');close(q.interfaceLimits.secondRadialInside!,-q.interfaceLimits.secondRadialOutside!);
});
test('zero current removes the second derivative interface jump even with a nonzero gauge',()=>{
 const q=potential({...C,current:0,gaugeQuadratic:1},[1,0,0]);assert.equal(q.sourceAzSecondRadialDerivative,0);assert.equal(q.sourceAzLaplacian,0);assert.equal(q.secondDerivativeStatus,'finite');assert.deepEqual(q.magneticField,[0,0,0]);assert.deepEqual(q.vectorPotential,[1,0,0]);
});
test('gauge transformations alter A and its divergence but not B at the same physical observation',()=>{
 const a=potential(C,[.3,.4,.7]),b=potential({...C,gaugeQuadratic:.2,gaugeCubic:.3,gaugeAxialConstant:.4},[.3,.4,.7]);assert.notDeepEqual(a.vectorPotential,b.vectorPotential);assert.deepEqual(a.magneticField,b.magneticField);close(a.divergenceA,0);close(b.divergenceA,.6);b.vectorPotential!.forEach((x,i)=>close(x,a.sourceVectorPotential![i]+b.gaugeGradient![i]));
});
test('constant axial A changes open path values while closed rectangle circulation is unchanged',()=>{
 const a=rectangle(C,.25,1.5,-.5,.5),b=rectangle({...C,gaugeAxialConstant:2},.25,1.5,-.5,.5);assert.notEqual(a.edgeIntegrals[1],b.edgeIntegrals[1]);close(a.circulation!,b.circulation!);close(b.gaugeClosedIntegral!,0);
});
test('rectangle orientation and degenerate closed geometries are explicit',()=>{
 const a=rectangle(C,.25,1.5,-.5,.5),b=rectangle(C,1.5,.25,-.5,.5),z=rectangle({...C,gaugeCubic:.5},.25,.25,-.5,.5);close(a.circulation!,-b.circulation!);assert.ok(a.orientedAreaY<0&&b.orientedAreaY>0);close(z.circulation!,0);assert.equal(z.orientedAreaY,0);
});
test('potential-derived B agrees with the preceding independently tested Ampere model across 3D observations',()=>{
 for(const x of [0,.2,1,2])for(const y of [0,.4])for(const z of [-2,0,3]){const p:StaticVector=[x,y,z],a=potential(C,p),b=magnetostaticCylinder({radius:1,totalCurrent:1,permeability:1},p);a.magneticField!.forEach((v,i)=>close(v,b.sourceField![i]));}
});
test('Cartesian finite differences of the actual transformed A retain its magnetic curl and divergence',()=>{
 const s={...C,gaugeQuadratic:.7,gaugeCubic:-.2},x:StaticVector=[.3,.4,.6],h=1e-5,q=potential(s,x),columns=[0,1,2].map(j=>{const a=[...x]as StaticVector,b=[...x]as StaticVector;a[j]+=h;b[j]-=h;return potential(s,a).vectorPotential!.map((v,i)=>(v-potential(s,b).vectorPotential![i])/(2*h));});
 const [dx,dy,dz]=columns,curl=[dy[2]-dz[1],dz[0]-dx[2],dx[1]-dy[0]];curl.forEach((v,i)=>close(v,q.magneticField![i],2e-10));close(dx[0]+dy[1]+dz[2],q.divergenceA,2e-10);
});
test('a tiny nonzero field and a lost potential/gradient term keep distinct arithmetic states',()=>{
 const q=potential(C,[1e-200,0,0]);assert.ok(q.sourceAz!>0);relative(q.magneticField![1],1e-200/(2*Math.PI));const r=potential({...C,current:Number.MIN_VALUE,permeability:1e-12},[1,0,0]);assert.equal(r.sourceAz,0);assert.equal(r.magneticField,null);assert.equal(r.magneticFieldStatus,'arithmetic-range');
});
test('returned vectors and path endpoints do not alias inputs or one another',()=>{
 const p:StaticVector=[.3,.4,.5],q=potential({...C,gaugeQuadratic:1},p);q.position[0]=2;assert.equal(p[0],.3);q.gaugeGradient![0]=3;assert.notEqual(q.vectorPotential![0],3);const w=work(S,p,[1,1,0],[0,2,1]);w.start[0]=3;assert.equal(p[0],.3);
});
test('all finite available outputs remain JSON-safe,with explicit unavailable values rather than NaN',()=>{
 const scan=(x:unknown):void=>{if(typeof x==='number')assert.ok(Number.isFinite(x));else if(x&&typeof x==='object')Object.values(x).forEach(scan);};for(const p of [[0,0,0],[1,0,0],[1e-200,0,0],[4,-4,4]]as StaticVector[]){scan(potential(C,p));scan(observe(S,p));}scan(rectangle(C,.25,2,-1,1));
});
test('vector potential domains and sparse/nonfinite gauge inputs are rejected',()=>{
 assert.throws(()=>potential({...C,radius:0},[0,0,0]),RangeError);assert.throws(()=>potential({...C,gaugeCubic:NaN},[0,0,0]),RangeError);assert.throws(()=>potential(C,[NaN,0,0]),RangeError);assert.throws(()=>potential(C,[0,0]as unknown as StaticVector),RangeError);assert.throws(()=>rectangle(C,-5,1,0,1),RangeError);
});
