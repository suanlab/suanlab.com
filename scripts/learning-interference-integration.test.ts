import test from'node:test';import assert from'node:assert/strict';import{createHash}from'node:crypto';import*as data from'../src/data/interactive-book';import prior from'./fixtures/pre-interference-registry-hashes.json';import{evaluateWaves,wavesKinds}from'../src/lib/learning-waves';import{finiteApertureDiffraction}from'../src/lib/learning-interference-diffraction';
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex'),close=(a:number,b:number,tol=3e-10)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);const [detector,diffraction]=[wavesKinds[6],wavesKinds[7]];
const getD=(i:number)=>evaluateWaves(detector,data.wavesLabs[detector].presets[i].values).detector!;
const getA=(i:number)=>evaluateWaves(diffraction,data.wavesLabs[diffraction].presets[i].values).diffraction!;
test('all448 prior units,guides,lab entries and relative order remain identical as canonical data',()=>{
 const ids=new Set(prior.ids),units=data.learningUnits.filter(u=>ids.has(u.id));assert.deepEqual(units.map(u=>u.id),prior.ids);assert.equal(hash(units),prior.units);assert.equal(hash(Object.fromEntries(units.map(u=>[u.id,data.learningGuides[u.id]]))),prior.guides);const labs=Object.fromEntries(Object.entries(prior.labEntries).map(([name,keys])=>[name,Object.fromEntries(keys.map(k=>[k,(data as Record<string,any>)[name][k]]))]));assert.equal(hash(labs),prior.labs);
});
test('the two observationally identical presets preserve exact noninteger amplitudes and distinct populations',()=>{
 const a=getD(0),b=getD(1);close(a.query.meanIntensity,3);close(a.query.visibility!,.5);close(b.system.amplitude1,Math.sqrt((2+Math.sqrt(3))/2));assert.equal(a.system.population.length,2);assert.equal(b.system.population.length,1);a.phaseScan.forEach((p,i)=>close(p.intensity,b.phaseScan[i].intensity));
});
test('detector guides retain negative sinc phase reversal,cycle averaging and unavailable coherence',()=>{
 const a=getD(2),b=getD(3);close(a.query.windowFactor,-2/(3*Math.PI));close(a.query.meanIntensity,2-4/(3*Math.PI));close(b.query.meanIntensity,2+4/(3*Math.PI));close(a.query.visibility!,b.query.visibility!);const c=getD(4),d=getD(5);close(c.query.meanIntensity,4);close(c.query.realSquareMean,2);close(d.query.realSquareMean,4);assert.equal(d.timeWave.length,1);assert.equal(d.query.realQuadratureSubdivisions,0);assert.equal(getD(6).query.mutualCoherence12,null);assert.equal(getD(6).query.visibility,0);assert.equal(getD(7).query.visibility,null);
});
test('touching and twice-wide presets preserve the same physical distance,query point and complex fields',()=>{
 const a=getA(2),b=getA(3);close(a.query.distance,.005);close(a.query.distance,b.query.distance);close(a.query.x,b.query.x);close(a.query.fresnel[0],b.query.fresnel[0]);close(a.query.fresnel[1],b.query.fresnel[1]);close(a.query.fraunhoferIntensity,b.query.fraunhoferIntensity);close(a.query.expectedTransmittedNorm,b.query.expectedTransmittedNorm);
});
test('aperture guides distinguish translated phase,opposite-phase darkness and unchanged transmitted norm',()=>{
 const a=getA(4),b=getA(5);close(a.query.fraunhoferIntensity,b.query.fraunhoferIntensity);assert.ok(Math.hypot(a.query.fraunhofer[0]-b.query.fraunhofer[0],a.query.fraunhofer[1]-b.query.fraunhofer[1])>.1);const c=getA(6),d=getA(7);assert.ok(c.query.fresnelIntensity<1e-25&&c.query.fraunhoferIntensity<1e-25);close(c.query.expectedTransmittedNorm,.0001);close(d.query.fraunhoferIntensity,4);close(c.query.expectedTransmittedNorm,d.query.expectedTransmittedNorm);
});
test('aperture plot frame removes the same phase from both fields and preserves scaled intensities',()=>{
 const a=getA(0);for(const q of a.field.filter((_,i)=>i%32===0)){close(q.apertureFresnel[0]**2+q.apertureFresnel[1]**2,q.scaledFresnelIntensity);close(q.apertureFraunhofer[0]**2+q.apertureFraunhofer[1]**2,q.scaledFraunhoferIntensity);close(Math.hypot(q.apertureFresnel[0]-q.apertureFraunhofer[0],q.apertureFresnel[1]-q.apertureFraunhofer[1]),q.complexFieldDifference/Math.sqrt(q.fresnelNumber));}
 const values=data.wavesLabs[diffraction].presets[0].values,b=evaluateWaves(diffraction,[...values.slice(0,-1),.73123456789]).diffraction!;assert.deepEqual(a.field,b.field);close(b.query.eta,.73123456789);const q=finiteApertureDiffraction(b.system,b.query.eta);close(q.fresnelIntensity,b.query.fresnelIntensity);
});
test('physical-unit endpoint conversion and control validation do not clamp meaningful continuous values',()=>{
 const max=getA(11);assert.equal(max.system.wavelength,800e-9);assert.equal(max.system.slitWidth,20e-6);for(const id of[detector,diffraction]){const values=data.wavesLabs[id].controls.map(c=>c.initial);assert.throws(()=>evaluateWaves(id,new Array(values.length)));assert.throws(()=>evaluateWaves(id,values.map((v,i)=>i===0?NaN:v)));}const a=data.wavesLabs[detector].presets[1].values;assert.throws(()=>evaluateWaves(detector,[...a.slice(0,2),.5,...a.slice(3)]));
});
