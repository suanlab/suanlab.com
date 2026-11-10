import test from 'node:test';
import assert from 'node:assert/strict';
import {planeStressAt,planeStressTraction as traction,rectangleStressBalance as balance,rotatePlaneStress,affineDeformationState as strain,affineDeformedPoint,infinitesimalRotationControl,type AffinePlaneStress,type AffineDeformation,type MechanicalValue,type MechanicalMatrix} from '../src/lib/learning-continuum-stress-strain';
import references from './fixtures/continuum-stress-strain-independent.json';
const near=(actual:MechanicalValue,expected:number,tolerance=5e-12)=>{assert.notEqual(actual,null);assert.ok(Math.abs(actual!-expected)<=tolerance*Math.max(1,Math.abs(expected)),`${actual} vs ${expected}`);};
const nearRelative=(actual:MechanicalValue,expected:number)=>{assert.notEqual(actual,null);assert.ok(Math.abs(actual!-expected)<=1e-12*Math.abs(expected),`${actual} vs ${expected}`);};
const tensor:AffinePlaneStress={normalX:3,normalY:-1,shear:1,gradientX:1,gradientY:-.5,crossGradient:2};
const rectangle={centerX:.2,centerY:-.1,halfWidth:1,halfHeight:.5,depth:2,bodyForce:[-3,.5]as[number,number]};
const identity:AffineDeformation={stretchX:1,stretchY:1,shear:0,rotation:0,translationX:0,translationY:0};
const isotropic:AffinePlaneStress={normalX:-2,normalY:-2,shear:0,gradientX:0,gradientY:0,crossGradient:0};
function allFiniteOrMissing(value:unknown):void{if(typeof value==='number')assert.ok(Number.isFinite(value));else if(value&&typeof value==='object')Object.values(value).forEach(allFiniteOrMissing);}
test('48 independent actual-face and physical-volume quadratures match forces and moments',()=>{
  assert.equal(references.stressCases.length,48);for(const r of references.stressCases){const actual=balance(r.stress,{...r.rectangle,bodyForce:r.rectangle.bodyForce as[number,number]}),expected=r.expected;assert.ok(actual.surfaceForce);
    for(let j=0;j<2;j++){near(actual.surfaceForce![j],expected.surfaceForce[j]);near(actual.volumeForce[j],expected.volumeForce[j]);}
    near(actual.surfaceMoment,expected.surfaceMoment);for(let i=0;i<4;i++){for(let j=0;j<2;j++)near(actual.faces[i].force[j],expected.faces[i].force[j]);near(actual.faces[i].moment,expected.faces[i].moment);}
  }
});
test('48 independent tractions and normal/shear decompositions match arbitrary oriented faces',()=>{
  for(const r of references.stressCases){const q=traction(r.stress,r.x,r.y,r.angle);for(let j=0;j<2;j++)near(q.traction[j],r.expected.traction[j]);near(q.normalStress,r.expected.normalStress);near(q.signedShear,r.expected.signedShear);}
});
test('worked tilted-face stress separates vector traction, normal stress, signed shear and force',()=>{
  const q=traction({...tensor,gradientX:0,gradientY:0,crossGradient:0},0,0,Math.PI/4,2);
  near(q.traction[0],2*Math.SQRT2);near(q.traction[1],0);near(q.normalStress,2);near(q.signedShear,-2);near(q.force[0],4*Math.SQRT2);near(q.principalValues[0],1+Math.sqrt(5));near(q.principalValues[1],1-Math.sqrt(5));
});
test('in-plane isotropic compression points inward and has no uniquely preferred principal direction',()=>{
  for(const angle of [-2,-.4,0,.7,2]){const q=traction(isotropic,0,0,angle);near(q.normalStress,-2);near(q.signedShear,0);near(q.traction[0],-2*q.normal[0]);near(q.traction[1],-2*q.normal[1]);assert.equal(q.principalDirection,null);assert.equal(q.directionStatus,'nonunique-isotropic');}
});
test('normal reversal reverses traction and shear vector while preserving scalar normal stress',()=>{
  const a=traction(tensor,.3,-.4,.2),b=traction(tensor,.3,-.4,.2+Math.PI);for(let j=0;j<2;j++){near(b.traction[j],-a.traction[j]!);near(b.shearVector[j],-a.shearVector[j]!);}near(b.normalStress,a.normalStress!);near(b.signedShear,a.signedShear!);
});
test('active tensor and normal rotation preserve normal/shear stresses and principal values',()=>{
  const base={...tensor,gradientX:0,gradientY:0,crossGradient:0},q=traction(base,0,0,.3),angle=.7,rotated=rotatePlaneStress(q.stress,angle);
  const r=traction({normalX:rotated[0][0]!,normalY:rotated[1][1]!,shear:rotated[0][1]!,gradientX:0,gradientY:0,crossGradient:0},0,0,.3+angle);
  near(r.normalStress,q.normalStress!);near(r.signedShear,q.signedShear!);near(r.principalValues[0],q.principalValues[0]!);near(r.principalValues[1],q.principalValues[1]!);
});
test('uniform stress has nonzero individual face forces but zero resultant and central moment',()=>{
  const q=balance({...tensor,gradientX:0,gradientY:0,crossGradient:0},{...rectangle,bodyForce:[0,0]});near(q.surfaceForce![0],0);near(q.surfaceForce![1],0);near(q.surfaceMoment,0);assert.ok(q.faces.some(f=>Math.hypot(...f.force as[number,number])>0));
});
test('the affine worked field balances a declared body force; omitting it leaves the same-field imbalance',()=>{
  const q=balance(tensor,rectangle);near(q.volume,4);near(q.surfaceForce![0],12);near(q.surfaceForce![1],-2);near(q.totalForce![0],0);near(q.totalForce![1],0);near(q.surfaceMoment,0);near(q.omittedBodyForceControl![0],12);near(q.omittedBodyForceControl![1],-2);
  const noBody=balance(tensor,{...rectangle,bodyForce:[0,0]});assert.deepEqual(noBody.surfaceForce,q.surfaceForce);near(noBody.totalForce![0],12);
});
test('surface force equals the integrated divergence rather than a sum of absolute face magnitudes',()=>{
  for(const r of references.stressCases){const q=balance(r.stress,{...r.rectangle,bodyForce:[0,0]});near(q.forceResidual![0],0);near(q.forceResidual![1],0);near(q.surfaceMoment,0);}
});
test('geometric dilation with inverse stress gradients scales force by area',()=>{
  const base={...rectangle,halfWidth:.3,halfHeight:.2,depth:.6},a=balance(tensor,base),b=balance({...tensor,gradientX:tensor.gradientX/2,gradientY:tensor.gradientY/2,crossGradient:tensor.crossGradient/2},{...base,centerX:base.centerX*2,centerY:base.centerY*2,halfWidth:.6,halfHeight:.4,depth:1.2,bodyForce:[-1.5,.25]});
  near(b.volume,8*a.volume!);near(b.surfaceForce![0],4*a.surfaceForce![0]!);near(b.surfaceForce![1],4*a.surfaceForce![1]!);
});
test('zero patch area does not erase the mathematical local traction',()=>{
  const q=traction(isotropic,0,0,0,0);assert.deepEqual(q.force,[0,0]);assert.deepEqual(q.traction,[-2,0]);assert.equal(q.patchStatus,'zero-area-no-face-observation');
});
test('a collapsed body has no manufactured actual faces or surface balance',()=>{
  const q=balance(tensor,{...rectangle,halfWidth:0});assert.equal(q.volume,0);assert.deepEqual(q.faces,[]);assert.equal(q.surfaceForce,null);assert.equal(q.totalMoment,null);assert.equal(q.geometryStatus,'zero-area');
});
test('a positive sub-range body is neither widened nor replaced by exact zero area',()=>{
  const q=balance(tensor,{...rectangle,halfWidth:1e-10});assert.ok(q.volume!>0);assert.equal(q.surfaceForce,null);assert.equal(q.geometryStatus,'below-supported-geometry');
  assert.equal(balance(tensor,{...rectangle,halfWidth:1e-200,halfHeight:1e-200}).volume,null);
});
test('nonzero stress and patch products lost to underflow remain unavailable',()=>{
  const q=traction({...isotropic,normalX:Number.MIN_VALUE,normalY:0},0,0,0,.25);assert.equal(q.traction[0],Number.MIN_VALUE);assert.equal(q.force[0],null);assert.equal(q.force[1],0);
  assert.equal(planeStressAt({...isotropic,gradientX:Number.MIN_VALUE},.25,0)[0][0],null);
});
test('48 independent mapped-point lengths, inner products and polygon areas match affine strains',()=>{
  assert.equal(references.deformationCases.length,48);for(const r of references.deformationCases){const q=strain(r.model,r.angle);for(let i=0;i<2;i++)for(let j=0;j<2;j++){near(q.green[i][j],r.expected.green[i][j]);near(q.infinitesimal[i][j],r.expected.infinitesimal[i][j]);}
    near(q.roundedStretch,r.expected.stretch);near(q.stableEngineeringStrain,r.expected.engineeringStrain);near(q.area,r.expected.area);near(q.roundedPolygonArea,r.expected.area);for(let i=0;i<4;i++)for(let j=0;j<2;j++)near(q.corners[i][j],r.expected.corners[i][j]);
  }
});
test('identity and translations have zero strain while translated point coordinates change',()=>{
  const a=strain(identity),b=strain({...identity,translationX:.7,translationY:-.2});assert.deepEqual(a.green,[[0,0],[0,0]]);assert.deepEqual(a.infinitesimal,[[0,0],[0,0]]);assert.deepEqual(b.green,a.green);assert.deepEqual(b.F,a.F);assert.notDeepEqual(b.corners,a.corners);near(b.area,4);
});
test('finite rigid rotation preserves lengths and Green strain but invalidates a large-angle linear strain reading',()=>{
  const q=strain({...identity,rotation:Math.PI/3},.4);assert.deepEqual(q.green,[[0,0],[0,0]]);near(q.roundedStretch,1);near(q.stableEngineeringStrain,0);near(q.infinitesimal[0][0],-.5);near(q.infinitesimal[1][1],-.5);near(q.area,4);
});
test('simple shear has a second-order Green component absent from infinitesimal strain',()=>{
  const q=strain({...identity,shear:.4},Math.PI/2);near(q.green[0][0],0);near(q.green[0][1],.2);near(q.green[1][1],.08);near(q.infinitesimal[1][1],0);near(q.stableEngineeringStrain,Math.sqrt(1.16)-1);near(q.directionalLinear,0);
});
test('uniaxial engineering strain and Green strain answer different questions',()=>{
  const q=strain({...identity,stretchX:1.2,stretchY:.8});near(q.green[0][0],.22);near(q.infinitesimal[0][0],.2);near(q.stableEngineeringStrain,.2);near(q.determinant,.96);near(q.area,3.84);
});
test('superposed spatial rigid rotation leaves reference Green strain and actual lengths unchanged',()=>{
  const d={...identity,stretchX:1.3,stretchY:.8,shear:.4,rotation:.2},a=strain(d,.7),b=strain({...d,rotation:.9},.7);assert.deepEqual(b.green,a.green);near(b.stableEngineeringStrain,a.stableEngineeringStrain!);near(b.roundedStretch,a.roundedStretch);near(b.area,a.area);assert.notDeepEqual(b.infinitesimal,a.infinitesimal);
});
test('directional length can be observed from two translated mapped points without using a strain tensor',()=>{
  const d={...identity,stretchX:.7,stretchY:1.6,shear:-.4,rotation:.7,translationX:.8,translationY:-.3},angle=.42,n=[Math.cos(angle),Math.sin(angle)],a=affineDeformedPoint(d,0,0),b=affineDeformedPoint(d,n[0],n[1]),q=strain(d,angle);near(Math.hypot(b[0]-a[0],b[1]-a[1]),q.roundedStretch);
});
test('I plus a skew increment is not an exact finite rigid rotation',()=>{
  const q=infinitesimalRotationControl(.5);near(q.greenScalar,.125);near(q.stableEngineeringStrain,Math.sqrt(1.25)-1);assert.ok(q.roundedStretch>1);assert.deepEqual(strain({...identity,rotation:.5}).green,[[0,0],[0,0]]);
});
test('140-digit references retain tiny finite-rotation corrections and skew-control extensions',()=>{
  assert.equal(references.tinyCases.length,3);for(const r of references.tinyCases){const q=strain({...identity,rotation:r.angle}),control=infinitesimalRotationControl(r.angle);nearRelative(q.infinitesimal[0][0],r.cosMinusOne);nearRelative(control.greenScalar,r.linearControlGreen);nearRelative(control.stableEngineeringStrain,r.linearControlStrain);assert.equal(q.F[0][0]-1,0);assert.ok(q.infinitesimal[0][0]!<0);assert.equal(control.roundedStretch-1,0);}
});
test('a stretch one representable step above unity remains a nonzero family observation',()=>{
  const q=strain({...identity,stretchX:1+Number.EPSILON});assert.ok(q.green[0][0]!>0);nearRelative(q.stableEngineeringStrain,Number.EPSILON);
});
test('unrepresentable second-order rotation and shear terms remain unavailable instead of physical zero',()=>{
  const q=strain({...identity,rotation:1e-200});assert.equal(q.infinitesimal[0][0],null);assert.deepEqual(q.green,[[0,0],[0,0]]);assert.equal(q.arithmeticStatus,'some-observations-unavailable');assert.equal(infinitesimalRotationControl(1e-200).greenScalar,null);
  const shear=strain({...identity,shear:Number.MIN_VALUE},.7);assert.equal(shear.green[0][1],null);assert.equal(shear.green[1][1],null);assert.equal(shear.stableEngineeringStrain,null);
});
test('missing, sparse, nonfinite and out-of-range inputs are rejected without clamping into another experiment',()=>{
  assert.throws(()=>traction({...tensor,normalX:NaN},0,0,0),RangeError);assert.throws(()=>traction(tensor,5,0,0),RangeError);assert.throws(()=>traction(tensor,0,0,0,-1),RangeError);assert.throws(()=>balance(tensor,{...rectangle,bodyForce:new Array(2) as[number,number]}),RangeError);
  assert.throws(()=>rotatePlaneStress(new Array(2)as MechanicalMatrix,0),RangeError);assert.throws(()=>rotatePlaneStress([[NaN,0],[0,1]],0),RangeError);
  for(const key of Object.keys(identity)){const missing={...identity}as Record<string,number>;delete missing[key];assert.throws(()=>strain(missing as unknown as AffineDeformation),RangeError);}
  assert.throws(()=>strain({...identity,stretchX:0}),RangeError);assert.throws(()=>strain(identity,Infinity),RangeError);assert.throws(()=>affineDeformedPoint(identity,3,0),RangeError);assert.throws(()=>infinitesimalRotationControl(2),RangeError);
});
test('observations are finite or explicitly unavailable and inputs remain unchanged',()=>{
  const saved=JSON.stringify({tensor,rectangle,identity});for(const angle of [-Math.PI,0,Math.PI]){allFiniteOrMissing(traction(tensor,0,0,angle));allFiniteOrMissing(balance(tensor,rectangle));allFiniteOrMissing(strain({...identity,rotation:angle}));}assert.equal(JSON.stringify({tensor,rectangle,identity}),saved);
});
