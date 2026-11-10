import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/charge-current-fields-reference.json';
import { pointChargeField as field, directedSurfaceCurrent as surface, transportedChargeObservation as observe,
  gaussianChargeInterval as interval, transportedChargeBalance as balance, teachingCoulombConstant,
  type PointCharge, type ChargeVector, type TransportedGaussianCharge } from '../src/lib/learning-charge-current-fields';
const known=(value:number|null):number=>{assert.notEqual(value,null);return value!;};
const close = (a: number|null, b: number|null, tol = 3e-12) => assert.ok(a !== null && b !== null && Number.isFinite(a) && Math.abs(a-b) <= tol*Math.max(1, Math.abs(b)), `${a} != ${b}`);
const relative = (a: number|null, b: number, tol = 3e-12) => assert.ok(a !== null && Number.isFinite(a) && Math.abs(a-b) <= tol*Math.max(Number.MIN_VALUE, Math.abs(b)), `${a} != ${b}`);
const p: PointCharge = { position: [0,0], charge: 1 };
const s: TransportedGaussianCharge = { totalCharge: 1, sigma: .3, initialCenter: 0, velocity: .4, area: 2 };
test('48 direct vector sums agree with independent NumPy references and potential gradients', () => {
  for (const r of refs.fields) {
    const q = field(r.sources as PointCharge[], r.observation as ChargeVector, .25, 1);
    assert.equal(q.status, 'finite');q.field!.forEach((v,i)=>{close(v,r.fieldOverK[i]);close(v,r.negativePotentialGradient[i],2e-10);close(q.force![i],.25*v);});close(q.potential!,r.potentialOverK);
  }
});
test('like-charge midpoint cancels the vector field while scalar potential remains positive', () => {
  const q=field([{position:[-1,0],charge:1},{position:[1,0],charge:1}],[0,0],1,1);
  assert.deepEqual(q.field,[0,0]);assert.equal(q.potential,2);assert.equal(q.fieldDirection,null);assert.deepEqual(q.force,[0,0]);
});
test('opposite-charge midpoint cancels potential but reinforces field direction', () => {
  const q=field([{position:[-1,0],charge:1},{position:[1,0],charge:-1}],[0,0],1,1);
  assert.deepEqual(q.field,[2,0]);assert.equal(q.potential,0);assert.deepEqual(q.fieldDirection,[1,0]);
});
test('the planar view uses inverse-square field and inverse-distance potential,not two-dimensional logarithmic electrostatics', () => {
  const a=field([p],[1,0],1,1),b=field([p],[2,0],1,1);close(a.field![0],4*b.field![0]);close(a.potential!,2*b.potential!);
});
test('source-charge sign changes field and potential while test-charge sign changes only force', () => {
  const a=field([p],[1,2],.5,1),b=field([{...p,charge:-1}],[1,2],.5,1),c=field([p],[1,2],-.5,1);
  a.field!.forEach((v,i)=>{close(b.field![i],-v);close(c.field![i],v);close(c.force![i],-a.force![i]);});close(b.potential!,-a.potential!);close(c.potential!,a.potential!);
});
test('a zero test charge retains the nonzero source field and yields zero force', () => {
  const q=field([p],[1,0],0,1);assert.deepEqual(q.field,[1,0]);assert.deepEqual(q.force,[0,0]);
});
test('empty and zero source sets have finite zero fields,including at a declared zero source', () => {
  for(const sources of[[],[{...p,charge:0}]]){const q=field(sources,[0,0]);assert.equal(q.status,'finite');assert.deepEqual(q.field,[0,0]);assert.equal(q.potential,0);assert.equal(q.fieldDirection,null);}
});
test('a nonzero source collision is explicitly singular even with a zero test charge', () => {
  const q=field([p],[0,0],0);assert.equal(q.status,'source-singularity');assert.equal(q.field,null);assert.equal(q.potential,null);assert.equal(q.force,null);
});
test('raw coincident opposite sources remain individually singular at their shared position', () => {
  assert.equal(field([p,{...p,charge:-1}],[0,0]).status,'source-singularity');assert.deepEqual(field([p,{...p,charge:-1}],[1,0]).field,[0,0]);
});
test('small nonzero distances are neither clamped nor treated as source collisions', () => {
  const a=field([p],[1e-13,0],1,1);assert.equal(a.status,'finite');relative(a.field![0],1e26);relative(a.potential!,1e13);
});
test('numerical overflow is distinct from source coincidence and never leaks NaN or infinity', () => {
  const q=field([p],[Number.MIN_VALUE,0],1,1);assert.equal(q.status,'numerical-range');assert.ok(q.contributions[0].distance>0);assert.equal(q.field,null);assert.equal(q.potential,null);
  const check=(value:unknown):void=>{if(typeof value==='number')assert.ok(Number.isFinite(value));else if(value&&typeof value==='object')Object.values(value).forEach(check);};check(q);
});
test('a representable potential survives an unrepresentable field without inventing a force',()=>{const q=field([p],[1e-160,0],1,1);assert.equal(q.field,null);assert.equal(q.force,null);relative(q.potential!,1e160);assert.equal(q.potentialStatus,'finite');assert.equal(q.fieldStatus,'numerical-range');});
test('underflowed nonzero source ratios are unavailable rather than falsely zero sources', () => {
  const q=field([{position:[0,0],charge:Number.MIN_VALUE}],[2,0],1,1);assert.equal(q.status,'numerical-range');assert.equal(q.contributions[0].status,'numerical-range');
});
test('common translation preserves field and potential and a common rotation rotates the field', () => {
  const sources:PointCharge[]=[{position:[-.5,.2],charge:1},{position:[.7,-.3],charge:-.6}],x:ChargeVector=[2,1];
  const angle=.7,R=([a,b]:ChargeVector):ChargeVector=>[a*Math.cos(angle)-b*Math.sin(angle),a*Math.sin(angle)+b*Math.cos(angle)];
  const q=field(sources,x,1,1),rot=field(sources.map(p=>({...p,position:R(p.position)})),R(x),1,1);
  R(q.field!).forEach((v,i)=>close(rot.field![i],v));close(rot.potential!,q.potential!);
  const shift=([a,b]:ChargeVector):ChargeVector=>[a+1,b-2],moved=field(sources.map(p=>({...p,position:shift(p.position)})),shift(x),1,1);
  q.field!.forEach((v,i)=>close(moved.field![i],v));close(moved.potential!,q.potential!);
});
test('uniform geometric dilation rescales field and potential by different powers', () => {
  const a=field([p],[1,2],1,1),b=field([p],[2,4],1,1);a.field!.forEach((v,i)=>close(b.field![i],v/4));close(b.potential!,a.potential!/2);
});
test('a planar Coulomb slice has nonzero two-component divergence away from the source',()=>{
  const h=1e-4,x:ChargeVector=[1,2],component=(a:number,b:number,i:number)=>field([p],[a,b],1,1).field![i];
  const divergence=(component(x[0]+h,x[1],0)-component(x[0]-h,x[1],0))/(2*h)
    +(component(x[0],x[1]+h,1)-component(x[0],x[1]-h,1))/(2*h);
  const missingNormalDerivative=1/Math.hypot(...x)**3;close(divergence,-missingNormalDerivative,2e-9);close(divergence+missingNormalDerivative,0,2e-9);
});
test('the teaching vacuum constant is explicitly rounded and applied to SI charge and force', () => {
  assert.equal(teachingCoulombConstant,9e9);const q=field([{...p,charge:1e-9}],[1,0],2e-9);close(q.field![0],9);relative(q.force![0],18e-9);close(q.potential!,9);
});
test('surface-normal reversal reverses reported current without changing the current-density vector', () => {
  const a=surface([-2,0],3,0),b=surface([-2,0],3,Math.PI);close(a.current,-6);close(b.current,6);close(surface([-2,0],3,Math.PI/2).current,0);assert.equal(surface([2,0],0,0).current,0);
});
test('59 representable Gaussian control volumes match independent references;one lost-tail balance remains unavailable', () => {
  let finiteCases=0,rangeCases=0;
  for(const r of refs.currents){
    const q=interval(r.system,r.left,r.right,r.time),b=balance(r.system,r.left,r.right,r.startTime,r.time);
    if(q.chargeInside===null){
      // A positive Gaussian tail is bounded by width times its largest PDF.
      const center=r.system.initialCenter+r.system.velocity*r.time,leftZ=(r.left-center)/r.system.sigma,rightZ=(r.right-center)/r.system.sigma;
      assert.ok(leftZ*rightZ>0);const nearest=Math.min(Math.abs(leftZ),Math.abs(rightZ));
      const logUpper=Math.log(Math.abs(r.system.totalCharge)*(r.right-r.left)/r.system.sigma/Math.sqrt(2*Math.PI))-nearest*nearest/2;
      assert.ok(logUpper<Math.log(Number.MIN_VALUE)-10);assert.equal(r.chargeInside,0);
      assert.equal(q.fractionStatus,'numerical-underflow');assert.equal(q.netInflow,null);
      assert.equal(b.chargeChange,null);assert.equal(b.integratedNetInflow,null);assert.equal(b.balanceResidual,null);rangeCases++;
    }else{
      close(q.chargeInside,r.chargeInside);close(q.chargeInside,r.independentSpaceIntegral);close(q.netInflow,r.netInflow);close(q.netInflow,r.independentTimeDerivative,5e-8);close(b.chargeChange,r.chargeChange);close(b.integratedNetInflow,r.independentFlowIntegral);close(b.balanceResidual,0);finiteCases++;
    }
  }
  assert.equal(finiteCases,59);assert.equal(rangeCases,1);
});
test('176 representable local observations match independent differences;four nonzero underflowed tails are unavailable', () => {
  let finiteCases=0,rangeCases=0;
  for(const r of refs.currents)for(const v of r.observations){
    const q=observe(r.system,v.x,r.time);
    if(q.density===null){
      const logDensity=Math.log(Math.abs(r.system.totalCharge)/(r.system.area*r.system.sigma*Math.sqrt(2*Math.PI)))-q.z*q.z/2;
      assert.ok(logDensity<Math.log(Number.MIN_VALUE)-10);assert.notEqual(r.system.velocity,0);
      assert.ok(v.density===0);assert.equal(q.currentDensity,null);assert.equal(q.timeDerivative,null);assert.equal(q.currentDivergence,null);assert.equal(q.localContinuityResidual,null);rangeCases++;
    }else{
      close(q.density,v.density);close(q.currentDensity,v.currentDensity);close(q.timeDerivative,v.independentTimeDerivative,2e-7);close(q.currentDivergence,v.independentCurrentDivergence,2e-7);close(q.localContinuityResidual,0);finiteCases++;
    }
  }
  assert.equal(finiteCases,176);assert.equal(rangeCases,4);
});
test('negative charge moving right has leftward conventional current', () => {
  const q=observe({...s,totalCharge:-1},.1,0);assert.ok(known(q.density)<0);assert.ok(known(q.currentDensity)<0);assert.ok(s.velocity>0);
});
test('changing area at fixed total charge changes density and current density but not charge or boundary current', () => {
  const a={...s,area:.1},b={...s,area:4};const x=observe(a,.1,.2),y=observe(b,.1,.2);close(x.density,40*known(y.density));close(x.currentDensity,40*known(y.currentDensity));close(x.positiveXCurrent,y.positiveXCurrent);close(interval(a,-.5,.8,.2).chargeInside,interval(b,-.5,.8,.2).chargeInside);
});
test('stationary and zero-charge preparations preserve their different physical densities', () => {
  const still=observe({...s,velocity:0},0,1),empty=observe({...s,totalCharge:0},0,1);assert.ok(known(still.density)>0);assert.equal(still.currentDensity,0);assert.equal(empty.density,0);assert.equal(empty.currentDensity,0);close(balance({...s,velocity:0},-.5,.8,-1,1).chargeChange,0);
});
test('outward normals on two ends give charge-change rate equal to negative net outflow', () => {
  const q=interval(s,-.5,.8,.6);close(q.leftOutwardCurrent,-known(observe(s,-.5,.6).positiveXCurrent));close(q.rightOutwardCurrent,observe(s,.8,.6).positiveXCurrent);close(q.netInflow,-known(q.netOutflow));
});
test('forward,reverse and zero-time balances preserve signed integrated flow', () => {
  const a=balance(s,-.5,.8,-1,1),b=balance(s,-.5,.8,1,-1),c=balance(s,-.5,.8,.3,.3);close(a.chargeChange,-known(b.chargeChange));close(a.integratedNetInflow,-known(b.integratedNetInflow));assert.equal(c.chargeChange,0);assert.equal(c.integratedNetInflow,0);close(a.balanceResidual,0);
});
test('finite-window Gaussian mass is not normalized to one', () => {
  const q=interval(s,-.1,.1,0);assert.ok(known(q.fraction)>0&&known(q.fraction)<.3);assert.ok(known(q.chargeInside)<s.totalCharge);
});
test('seven independent tail and narrow-interval integrals retain relative accuracy', () => {
  for(const r of refs.tailIntervals){const q=interval({...s,sigma:1,initialCenter:0,velocity:0},r.left,r.right,0);assert.ok(known(q.fraction)>0);relative(known(q.fraction),r.fraction,8e-13);}
});
test('physical interval width survives standardized endpoint cancellation', () => {
  const left=.8,right=.8+Number.EPSILON/2,q=interval({...s,sigma:.1,velocity:0},left,right,0);
  const expected=(right-left)/.1*Math.exp(-32)/Math.sqrt(2*Math.PI);relative(known(q.fraction),expected,3e-13);assert.ok(known(q.fraction)>0);
});
test('underflowed positive Gaussian interval mass is labeled rather than claimed to be exactly absent', () => {
  const q=interval({...s,sigma:.1,velocity:0},15,16,0);assert.equal(q.fraction,null);assert.equal(q.fractionStatus,'numerical-underflow');assert.equal(q.chargeInside,null);
});
test('nonzero Gaussian tails do not become zero density,current or continuity certificates',()=>{
 const q=observe({...s,sigma:.1,area:1,velocity:1},4,0);
 assert.equal(q.density,null);assert.equal(q.currentDensity,null);assert.equal(q.positiveXCurrent,null);
 assert.equal(q.spatialDerivative,null);assert.equal(q.timeDerivative,null);assert.equal(q.currentDivergence,null);assert.equal(q.localContinuityResidual,null);
 assert.equal(q.densityStatus,'numerical-range');assert.equal(q.currentDensityStatus,'numerical-range');
});
test('stationary and zero-charge preparations preserve their exact zeros despite unavailable tails',()=>{
 const still=observe({...s,sigma:.1,velocity:0},4,0),empty=observe({...s,sigma:.1,totalCharge:0},4,0);
 assert.equal(still.density,null);assert.equal(still.currentDensity,0);assert.equal(still.positiveXCurrent,0);
 assert.equal(still.timeDerivative,0);assert.equal(still.currentDivergence,0);assert.equal(still.localContinuityResidual,0);
 for(const key of ['density','currentDensity','spatialDerivative','timeDerivative','currentDivergence','positiveXCurrent']as const)assert.equal(empty[key],0);
});
test('normalization in log space retains density when the unnormalized exponential underflows',()=>{
 const q=observe({...s,totalCharge:4,area:.1,sigma:.1,velocity:1},3.865,0);
 assert.equal(Math.exp(-q.z*q.z/2),0);assert.ok(q.density!==null&&q.density>0);assert.equal(q.densityStatus,'finite');assert.equal(q.currentDensity,q.density);
});
test('nonzero current product loss is unavailable while its density and exact stationary current remain separate',()=>{
 const q=observe({...s,totalCharge:1,area:1,sigma:1,velocity:Number.MIN_VALUE},0,0);
 assert.ok(q.density!==null&&q.density>0);assert.equal(q.currentDensity,null);assert.equal(q.currentDensityStatus,'numerical-range');
 assert.equal(observe({...s,velocity:0},0,0).currentDensity,0);
});
test('lost interval charge does not erase an available positive probability or certify its balance',()=>{
 const q=interval({...s,totalCharge:Number.MIN_VALUE,velocity:0},-.01,.01,0);
 assert.ok(q.fraction!==null&&q.fraction>0);assert.equal(q.fractionStatus,'finite');assert.equal(q.chargeInside,null);assert.equal(q.chargeInsideStatus,'numerical-range');
});
test('zero-duration and stationary balances stay exactly zero even with unavailable endpoint charges',()=>{
 for(const [velocity,start,end]of [[0,-1,1],[1,0,0]]){
  const q=balance({...s,sigma:.1,velocity},15,16,start,end);
  assert.equal(q.initial.chargeInside,null);assert.equal(q.final.chargeInside,null);
  assert.equal(q.chargeChange,0);assert.equal(q.integratedNetInflow,0);assert.equal(q.balanceResidual,0);
 }
});
test('maximum-speed narrow transport with lost endpoint tails does not certify a zero balance', () => {
  for(const v of[-2,2]){const q=balance({...s,sigma:.1,velocity:v},-.5,.8,-4,4);assert.equal(q.initial.chargeInside,null);assert.equal(q.final.chargeInside,null);assert.equal(q.chargeChange,null);assert.equal(q.integratedNetInflow,null);assert.equal(q.balanceResidual,null);assert.equal(q.flowQuadratureDifference,null);}
});
test('zero and invalid geometric widths,cross sections,nonfinite coordinates and unsupported data are rejected', () => {
  for(const bad of[{...s,sigma:0},{...s,area:0},{...s,velocity:3},{...s,totalCharge:NaN}])assert.throws(()=>observe(bad,0,0));
  assert.throws(()=>interval(s,1,1,0));assert.throws(()=>interval(s,1,-1,0));assert.throws(()=>observe(s,0,5));assert.throws(()=>field([p],[NaN,0]));assert.throws(()=>field([p],new Array(2) as ChargeVector));assert.throws(()=>field([p],[0,1],1,0));assert.throws(()=>surface([Infinity,0],1,0));assert.throws(()=>surface([1,0],-1,0));
});

test('nonzero test force below representable range does not become an exact zero observation',()=>{
 const source:PointCharge={position:[0,0],charge:1e-30},q=field([source],[1,0],1e-310);
 relative(q.field![0],9e-21);relative(q.potential!,9e-21);assert.equal(q.fieldStatus,'finite');assert.equal(q.potentialStatus,'finite');assert.equal(q.force,null);assert.equal(q.forceStatus,'numerical-range');assert.equal(q.status,'numerical-range');
 const resolved=field([source],[1,0],1e-300);assert.ok(resolved.force![0]>0);assert.equal(resolved.forceStatus,'finite');
});
test('exact zero charge and exact zero field retain physical zero forces after range checks',()=>{
 assert.deepEqual(field([{position:[0,0],charge:1e-30}],[1,0],0).force,[0,0]);
 assert.deepEqual(field([{position:[0,0],charge:0}],[1,0],1e-310).force,[0,0]);
 assert.deepEqual(field([{position:[-1,0],charge:1},{position:[1,0],charge:1}],[0,0],1e-310).force,[0,0]);
});
test('SI scaling loss keeps normalized values available without inventing zero E or potential',()=>{
 const q=field([{position:[0,0],charge:Number.MIN_VALUE}],[1,0],1,1e-12);assert.equal(q.fieldOverK![0],Number.MIN_VALUE);assert.equal(q.potentialOverK,Number.MIN_VALUE);assert.equal(q.field,null);assert.equal(q.potential,null);assert.equal(q.fieldStatus,'numerical-range');assert.equal(q.potentialStatus,'numerical-range');
});
test('an unrepresentable nonzero field component is unavailable rather than a false axis alignment',()=>{
 const q=field([{position:[0,0],charge:1e-30}],[1,1e-300],1,1);assert.equal(q.field,null);assert.equal(q.fieldStatus,'numerical-range');assert.notEqual(q.potential,null);
});

test('current through a nonzero patch remains unavailable when its nonzero product underflows',()=>{
 const q=surface([1e-200,0],1e-200,0);assert.equal(q.densityThroughNormal,1e-200);assert.equal(q.current,null);assert.equal(q.status,'numerical-range');assert.equal(surface([1e-200,0],0,0).current,0);assert.equal(surface([1e-200,0],1,0).current,1e-200);
});
test('directed projection loss and current overflow are explicit without fabricated finite values',()=>{
 assert.throws(()=>surface(new Array(2) as ChargeVector,1,0),RangeError);const q=surface([Number.MIN_VALUE,0],1,Math.acos(.25));assert.equal(q.densityThroughNormal,null);assert.equal(q.current,null);assert.equal(q.status,'numerical-range');const large=surface([Number.MAX_VALUE,0],4,0);assert.equal(large.densityThroughNormal,Number.MAX_VALUE);assert.equal(large.current,null);
});
