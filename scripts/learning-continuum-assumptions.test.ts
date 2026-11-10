import test from 'node:test';
import assert from 'node:assert/strict';
import {compactMassKernel,coarseGrainedParticleState as state,coarseGrainedIntervalMass as interval,continuumScaleWindow as scales,independentCountFluctuations as count,type BallisticParticle} from '../src/lib/learning-continuum-assumptions';
import references from './fixtures/continuum-assumptions-independent.json';
const near=(actual:number|null,expected:number|null)=>{
  if(expected===null){assert.equal(actual,null);return;}
  assert.notEqual(actual,null);assert.ok(Math.abs(actual!-expected)<=3e-11*Math.max(1e-4,Math.abs(expected)),`${actual} vs ${expected}`);
};
const pair:BallisticParticle[]=[{mass:1,position:0,velocity:1},{mass:1,position:0,velocity:-1}];
const input={numberDensity:1e24,meanFreePath:1e-7,macroLength:1e-3,averagingLength:1e-5,relativeFluctuationTolerance:.01,relativeResolutionTolerance:.1,collisionSeparationFactor:10};
test('48 definition-level independent quadratures and field derivatives match particle observations',()=>{
  assert.equal(references.particleCases.length,48);
  for(const r of references.particleCases){const actual=state(r.particles,r.x,r.time,r.h,r.bandwidthRate);
    for(const [key,value]of Object.entries(r.expected)){if(key==='intervalMass')near(interval(r.particles,r.left,r.right,r.time,r.h).mass,value);else near((actual as unknown as Record<string,number|null>)[key],value);}
  }
});
test('compact kernel is normalized, symmetric, nonnegative and C1 across its support edge',()=>{
  for(const h of [.02,.1,.7,2]){
    near(interval([{mass:1,position:0,velocity:0}],-3,3,0,h).mass,1);
    for(const q of [-1,-.75,-.1,0,.1,.75,1]){const a=compactMassKernel(q*h,h),b=compactMassKernel(-q*h,h);assert.ok(b.spatialDerivative!==null);near(a.value,b.value);near(a.spatialDerivative,-b.spatialDerivative);assert.ok(a.value>=0);}
    for(const q of [-1,1])assert.deepEqual(compactMassKernel(q*h,h),{value:0,spatialDerivative:0,bandwidthDerivative:0});
    const inside=compactMassKernel(h*(1-1e-8),h);assert.ok(inside.spatialDerivative!==null);assert.ok(Math.abs(inside.spatialDerivative)<1e-3);
  }
});
test('global mass stays fixed while the finite displayed window can exclude mass',()=>{
  for(const t of [-1,-.25,0,.5,1])near(interval(pair,-8,8,t,.4).mass,2);
  near(interval(pair,-.5,.5,0,1).mass,203/128);
  assert.ok(interval(pair,-.5,.5,1,1).mass<interval(pair,-.5,.5,0,1).mass);
});
test('opposite particle velocities have zero mean but nonzero unresolved momentum transport',()=>{
  const r=state(pair,0,0,1);near(r.density,15/8);near(r.velocity,0);near(r.flux,0);near(r.dispersion,15/8);near(r.momentumFlux,15/8);
});
test('uniform velocity gives exact zero dispersion and mass-weighted rather than unweighted velocity',()=>{
  near(state(pair.map(p=>({...p,velocity:.7})),0,0,1).dispersion,0);
  near(state([{mass:1,position:0,velocity:-1},{mass:3,position:0,velocity:1}],0,0,1).velocity,.5);
});
test('a spatially empty support has zero mass density but undefined normalized velocity and dispersion',()=>{
  const r=state(pair,8,0,.02);assert.equal(r.density,0);assert.equal(r.flux,0);assert.equal(r.velocity,null);assert.equal(r.dispersion,null);assert.equal(r.status,'empty-support');
});
test('fixed kernel supports exact continuity and ballistic momentum balance without asserting collisional closure',()=>{
  for(const x of [-.9,-.3,0,.3,.9]){const r=state(pair,x,.2,.7);near(r.continuityResidual,0);near(r.momentumResidual,0);}
});
test('changing only smoothing width creates an explicit observational term in both balances',()=>{
  const r=state([{mass:1,position:0,velocity:0}],0,0,1,.25);near(r.densityT,-15/64);near(r.continuityResidual,r.smoothingRate);near(r.smoothingRate,-15/64);
  const moving=state([{mass:1,position:0,velocity:1}],.2,0,1,.25);near(moving.momentumResidual,moving.momentumSmoothingRate);assert.notEqual(moving.momentumSmoothingRate,0);
});
test('density and relative velocity transform under a consistent Galilean boost',()=>{
  const x=.2,t=.3,b=.5,a=state(pair,x,t,.7),r=state(pair.map(p=>({...p,velocity:p.velocity+b})),x+b*t,t,.7);
  assert.ok(a.flux!==null);near(a.density,r.density);near(r.flux,a.flux+b*a.density);near(r.velocity,a.velocity!+b);near(r.dispersion,a.dispersion);
});
test('partitioning an observation interval preserves total mass without renormalizing the display',()=>{
  for(const t of [-.7,0,.7])near(interval(pair,-2,2,t,.5).mass,interval(pair,-2,.13,t,.5).mass+interval(pair,.13,2,t,.5).mass);
  const z=interval(pair,0,0,0,1);assert.equal(z.mass,0);assert.equal(z.meanDensity,null);
});
test('independent time differences of interval mass agree with physical fixed-boundary inflow',()=>{
  const time=.23,h=.8,left=-.4,right=.6,dt=1e-5;
  const derivative=(interval(pair,left,right,time+dt,h).mass-interval(pair,left,right,time-dt,h).mass)/(2*dt);
  const l=state(pair,left,time,h).flux,r=state(pair,right,time,h).flux;assert.ok(l!==null&&r!==null);const inflow=l-r;
  assert.ok(Math.abs(derivative-inflow)<2e-8);
});
test('a moving observation interval uses relative rather than laboratory flux',()=>{
  const time=.23,h=.8,speed=.3,dt=1e-5;
  const mass=(t:number)=>interval(pair,-.4+speed*t,.6+speed*t,t,h).mass;
  const left=state(pair,-.4+speed*time,time,h),right=state(pair,.6+speed*time,time,h);
  assert.ok(left.flux!==null&&right.flux!==null);const relative=left.flux-speed*left.density-right.flux+speed*right.density;
  assert.ok(Math.abs((mass(time+dt)-mass(time-dt))/(2*dt)-relative)<2e-8);
  assert.ok(Math.abs(relative-(left.flux-right.flux))>.01);
});
test('supported positive interval widths remain positive without broadening or accepting unsupported scales',()=>{
  assert.ok(interval(pair,0,1e-8,0,1).mass>0);
  assert.throws(()=>interval(pair,0,1e-12,0,1),RangeError);
  assert.throws(()=>interval(pair,1,-1,0,1),RangeError);
});
test('kernel derivatives match central differences in position and bandwidth',()=>{
  const x=.13,h=.7,d=1e-5,k=compactMassKernel(x,h);assert.ok(k.spatialDerivative!==null&&k.bandwidthDerivative!==null);
  assert.ok(Math.abs(k.spatialDerivative-(compactMassKernel(x+d,h).value-compactMassKernel(x-d,h).value)/(2*d))<2e-8);
  assert.ok(Math.abs(k.bandwidthDerivative-(compactMassKernel(x,h+d).value-compactMassKernel(x,h-d).value)/(2*d))<2e-8);
});
test('worked scale window distinguishes collision length, counting noise and macroscopic resolution',()=>{
  const r=scales(input);near(r.expectedCount,1e9);near(r.knudsen,1e-4);near(r.relativeCountStd,1/Math.sqrt(1e9));near(r.lowerLength,1e-6);near(r.upperLength,1e-4);assert.ok(r.windowExists&&r.selectedSatisfies);
});
test('shrinking the macroscopic length can eliminate all declared averaging scales',()=>{
  const r=scales({...input,macroLength:1e-6});assert.equal(r.windowExists,false);assert.equal(r.selectedSatisfies,false);assert.ok(r.lowerLength>r.upperLength);
});
test('a feasible interval does not make an arbitrary selected width acceptable',()=>{
  const small=scales({...input,averagingLength:1e-9}),large=scales({...input,averagingLength:1e-2});
  assert.ok(small.windowExists&&large.windowExists);assert.equal(small.countSatisfied,false);assert.equal(small.collisionSatisfied,false);assert.equal(large.resolutionSatisfied,false);
});
test('scale-window witnesses satisfy the declared inequalities and each limit has its own mechanism',()=>{
  for(const numberDensity of [1e12,1e18,1e24])for(const meanFreePath of [1e-9,1e-7,1e-5])for(const macroLength of [1e-5,1e-3,.1]){
    const r=scales({...input,numberDensity,meanFreePath,macroLength});
    if(r.windowExists){const witness=Math.sqrt(r.lowerLength*r.upperLength),w=scales({...input,numberDensity,meanFreePath,macroLength,averagingLength:witness});assert.ok(w.selectedSatisfies);}
    else assert.ok(r.countLowerLength>r.upperLength||r.collisionLowerLength>r.upperLength);
  }
});
test('common spatial unit dilation with inverse cubic number density preserves all dimensionless diagnostics',()=>{
  const a=scales(input),b=scales({...input,numberDensity:input.numberDensity/1e9,meanFreePath:input.meanFreePath*1e3,macroLength:input.macroLength*1e3,averagingLength:input.averagingLength*1e3});
  for(const key of ['expectedCount','relativeCountStd','knudsen','ratioToMacro','ratioToMeanFreePath']as const)near(a[key],b[key]);assert.equal(a.selectedSatisfies,b.selectedSatisfies);
});
test('95 independent enumerated binomial laws verify finite-reservoir count moments',()=>{
  assert.equal(references.binomialCases.length,95);for(const r of references.binomialCases){const c=count(r.n,r.p);near(c.mean,r.mean);near(c.variance,r.variance);near(c.relativeStd,r.relativeStd);}
});
test('zero probability has undefined relative noise; the full fixed reservoir has exact zero count noise',()=>{
  assert.equal(count(1000,0).relativeStd,null);assert.equal(count(1000,1).relativeStd,0);
  near(count(1000,.1).relativeStd,Math.sqrt(90)/100);near(count(1000,.1).poissonRelativeStd,.1);
});
test('out-of-range, sparse, fractional and nonfinite inputs fail instead of changing the requested experiment',()=>{
  assert.throws(()=>state(new Array(2),0,0,1),RangeError);assert.throws(()=>state([],0,0,1),RangeError);
  assert.throws(()=>state(pair,0,0,0),RangeError);assert.throws(()=>state(pair,NaN,0,1),RangeError);assert.throws(()=>compactMassKernel(0,1e-20),RangeError);
  for(const key of Object.keys(input)){assert.throws(()=>scales({...input,[key]:NaN}),RangeError);const missing={...input}as Record<string,number>;delete missing[key];assert.throws(()=>scales(missing as unknown as typeof input),RangeError);}
  assert.throws(()=>count(1.2,.5),RangeError);assert.throws(()=>count(1000,-.1),RangeError);assert.throws(()=>count(1000,NaN),RangeError);
});
test('all supported extreme scales remain finite without mutating the input',()=>{
  for(const n of [1,1e30])for(const ell of [1e-12,1e6]){const v={...input,numberDensity:n,averagingLength:ell},before={...v},r=scales(v);assert.deepEqual(v,before);Object.values(r).filter(x=>typeof x==='number').forEach(x=>assert.ok(Number.isFinite(x)));}
  const before=JSON.stringify(pair);state(pair,0,.3,.4,.1);interval(pair,-1,1,.3,.4);assert.equal(JSON.stringify(pair),before);
});
