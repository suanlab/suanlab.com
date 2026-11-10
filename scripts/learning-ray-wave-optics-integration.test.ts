import test from 'node:test';import assert from 'node:assert/strict';import{createHash}from'node:crypto';
import * as data from '../src/data/interactive-book';import prior from './fixtures/pre-ray-wave-optics-registry-hashes.json';
import{wavesKinds,evaluateWaves}from'../src/lib/learning-waves';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const close=(a:number,b:number,tol=3e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const [ray,gauss]=[wavesKinds[12],wavesKinds[13]],R=(i:number)=>evaluateWaves(ray,data.wavesLabs[ray].presets[i].values).ray!,G=(i:number)=>evaluateWaves(gauss,data.wavesLabs[gauss].presets[i].values).gaussian!;
test('all454 prior units,guides,lab entries and relative ordering are preserved as canonical data',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.equal(units.length,454);assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('real-image default and guided displaced sensor retain the same preparation and distinct physical observations',()=>{
 const a=R(0).query,b=R(7).query;close(a.imageDistance!,.2);close(a.imageHeight!,-.001);close(a.magnification!,-1);close(a.screen.span,0);close(b.imageDistance!,a.imageDistance!);close(b.screen.span,.001);close(b.screen.meanHeight,-.0005);close(R(8).query.screen.span,0);close(R(9).query.imageHeight!,-.001);close(R(10).query.imageHeight!,0);
});
test('virtual and negative-lens examples retain backwards intersections while actual output coordinates stay forward',()=>{
 const a=R(1).query,b=R(5).query;assert.equal(a.imageKind,'virtual');close(a.imageDistance!,-.1);close(a.magnification!,2);close(b.imageDistance!,-1/15);close(b.magnification!,1/3);a.rays.forEach(p=>close(p.imageBackExtension!.height,.002));assert.equal(R(6).query.imageKind,'no-lens');assert.equal(R(6).query.imageDistance,null);
});
test('exact focal placement and nearby huge finite images remain distinct without crossing the scan pole',()=>{
 assert.equal(R(2).query.imageKind,'at-infinity');assert.equal(R(2).query.imageDistance,null);close(R(3).query.imageDistance!,100.1);close(R(4).query.imageDistance!,-99.9);for(const i of[0,2,3,4]){const r=R(i),f=r.system.focalLength!;const pole=r.imageScan.findIndex(p=>p.objectDistance===f);assert.ok(pole>=0);assert.equal(r.imageScan[pole].imageDistance,null);assert.ok(r.imageScan.slice(0,pole).every(p=>p.imageDistance!<0));assert.ok(r.imageScan.slice(pole+1).every(p=>p.imageDistance!>0));}
});
test('Gaussian presets distinguish input widths,geometric focal plane and actual finite waist',()=>{
 const a=G(0).query,b=G(1).query,c=G(2).query;close(a.newWaist.coordinate,.004033999568132059);close(a.newWaist.radius,35.739054751761906e-6);close(a.output.width,79.57747154594765e-6);close(b.newWaist.coordinate,.019068277191795473);close(b.newWaist.radius,25.9005872888485e-6);close(c.output.width,c.newWaist.radius);assert.ok(Math.abs(c.output.inverseCurvature)<1e-9);close(a.intensityIntegral,c.intensityIntegral);close(G(8).query.output.width,2*b.output.width);
});
test('zero field removes actual phase and conditional normalization while retaining reference-mode geometry',()=>{
 const a=G(0).query,b=G(5).query;close(a.output.width,b.output.width);assert.equal(b.onAxisIntensity,0);assert.equal(b.intensityIntegral,0);assert.equal(b.onAxisPhase,null);assert.equal(b.fractionWithinWidth,null);assert.ok(b.radialProfile.every(p=>p.phase===null&&p.intensity===0));assert.ok(G(5).distanceScan.every(p=>p.onAxisIntensity===0&&p.intensityIntegral===0));assert.equal(G(6).query.output.curvature,null);assert.equal(G(6).query.output.inverseCurvature,0);
});
test('absent and negative lenses have their own physical transformations and extrapolated waist labels',()=>{
 const a=G(3),b=G(4);assert.equal(a.system.focalLength,null);close(a.query.output.q[0],.04);assert.ok(b.query.newWaist.coordinate<0);assert.equal(b.query.newWaist.location,'backward-extrapolated');assert.ok(b.query.longitudinal.every(p=>p.coordinate>=0));assert.equal(G(7).system.preDistance,.03);
});
test('area normalization and directly exposed covariance residual survive every physical-unit scan',()=>{
 for(let i=0;i<12;i++){const r=G(i),q=r.query;close(q.onAxisIntensity*Math.PI*q.output.width**2/2,q.intensityIntegral);for(const p of r.distanceScan)close(p.intensityIntegral,q.intensityIntegral);}
 assert.ok(Math.abs(G(11).query.covarianceDeterminantEvaluation.relativeResidual)>.001);assert.ok(G(10).query.divergence>.1||G(10).query.lensPhaseSlope>.1);assert.ok(R(11).query.maxSlope>.1);assert.ok(G(9).query.lensPhaseSlope>.1);
});
test('continuous controls retain precision through unit conversions and physical endpoints are accepted without rounding',()=>{
 const v=[...data.wavesLabs[ray].presets[0].values];v[6]=.0073123456789;const a=evaluateWaves(ray,v);assert.equal(a.ray!.system.fanHalfAngle,v[6]);const g=[...data.wavesLabs[gauss].presets[0].values];g[2]=.73123456789;assert.equal(evaluateWaves(gauss,g).gaussian!.system.amplitude,g[2]);for(const wavelength of[200,2000])for(const width of[5,2000])assert.doesNotThrow(()=>evaluateWaves(gauss,[wavelength,width,1,0,0,20,0]));assert.throws(()=>evaluateWaves(ray,[...v.slice(0,3),.5,...v.slice(4)]));assert.throws(()=>evaluateWaves(gauss,[500,40,1,0,20,20,3]));for(const [kind,index]of[[ray,3],[gauss,6]]as const){const sparse=[...data.wavesLabs[kind].presets[0].values];delete sparse[index];assert.throws(()=>evaluateWaves(kind,sparse));}
});

test('adaptive observation scans retain the selected plane and resolve the finite waist on its Rayleigh scale',()=>{
 for(let i=0;i<12;i++){const r=G(i),q=r.query;const selected=r.distanceScan.find(p=>p.distance===r.system.postDistance);assert.ok(selected);close(selected.width,q.output.width);close(selected.onAxisIntensity,q.onAxisIntensity);
  if(q.newWaist.coordinate>=0&&q.newWaist.coordinate<=r.distanceScan.at(-1)!.distance){const waist=r.distanceScan.find(p=>p.distance===q.newWaist.coordinate);assert.ok(waist);close(waist.width,q.newWaist.radius);assert.ok(Math.abs(waist.inverseCurvature)<1e-5);}
 }
 const a=G(0);assert.ok(a.distanceScan.at(-1)!.distance<=.05);assert.ok(Math.max(...a.distanceScan.map(p=>p.onAxisIntensity))>1.25);
});
