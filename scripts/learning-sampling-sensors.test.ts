import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/sampling-sensors-independent.json';
import {pixelApertureSampling as pixel,photonCountingSensor as sensor,pixelSinc,sensorNormalCdf,sensorPoissonDistribution as poisson,sensorQuantize,calibratedPhotonExposure as exposure} from '../src/lib/learning-sampling-sensors';
import type {PixelSamplingInput,PhotonSensorInput} from '../src/lib/learning-sampling-sensors';
const p:PixelSamplingInput={pitch:10e-6,apertureWidth:10e-6,pixelHeight:10e-6,wavelength:550e-9,irradiance:.003,modulation:.8,frequency:30000,phase:.7,velocity:0,exposure:.01,centerTime:0,origin:0,radius:8,queryCoordinate:0};
const s:PhotonSensorInput={incidentPhotonMean:10,quantumEfficiency:.6,exposure:.01,darkRate:20,readSigma:2,gain:.7,offset:8,bits:8,fullWell:8192};
const close=(a:number,b:number,tol=2e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
test('48 independently integrated space/time pixel preparations and432physical photon observations agree',()=>{
 assert.equal(refs.provenance.productImports,false);assert.equal(refs.pixels.length,48);
 for(const r of refs.pixels){const a=pixel(r.system);for(const q of r.samples){const z=a.samples.find(v=>v.index===q.index)!;close(z.limitAverage,q.limitAverage,1e-13);close(z.energy,q.energy,1e-25);close(z.incidentPhotonMean,q.incidentPhotonMean,3e-11);}}
});
test('36independent Poisson/Gaussian quantized distributions and clipped moments agree without mass renormalization',()=>{
 assert.equal(refs.sensors.length,36);for(const r of refs.sensors){const a=sensor(r.system);r.bins.forEach((v,i)=>close(a.bins[i].probability,v,3e-11));for(const k of ['adcMean','adcVariance','storedMean','storedVariance','fullWellOverflowProbability','adcLowerClippingProbability','adcUpperClippingProbability'] as const)close(a[k],r[k],2e-9);assert.ok(Math.abs(a.coveredProbability-1)<2e-11);assert.ok(a.bins.every(q=>q.probability>=0));}
});
test('77independent normal CDF values including quadrature transitions and extreme tails agree',()=>{
 for(const r of refs.normal)close(sensorNormalCdf(r.z),r.cdf,4e-15);
 assert.equal(sensorNormalCdf(-Infinity),0);assert.equal(sensorNormalCdf(Infinity),1);assert.throws(()=>sensorNormalCdf(NaN));assert.throws(()=>sensorNormalCdf(undefined as unknown as number));
});
test('sinc retains negative side lobes,exact integer zeros and physically distinct neighbouring frequencies',()=>{
 assert.equal(pixelSinc(0),1);assert.equal(pixelSinc(1),0);assert.ok(pixelSinc(1.5)<0);assert.ok(pixelSinc(1-1e-10)>0);assert.ok(pixelSinc(1+1e-10)<0);close(pixelSinc(-.7),pixelSinc(.7));assert.throws(()=>pixelSinc(Infinity));
});
test('subpixel averaging attenuates a grating by its original spatial frequency before sampling',()=>{
 const a=pixel(p);close(a.spatialTransfer,Math.sin(.3*Math.PI)/(.3*Math.PI));assert.equal(a.temporalTransfer,1);close(a.query.limitAverage,.003*(1+.8*a.spatialTransfer*Math.cos(.7)));assert.notEqual(a.query.limitAverage,a.query.pointIrradiance);
});
test('signed box transfer can invert a high-frequency grating contrast rather than using an MTF absolute value',()=>{
 const a=pixel({...p,frequency:150000,phase:0});assert.ok(a.spatialTransfer<0);assert.ok(a.query.limitAverage<p.irradiance);assert.ok(a.query.pointIrradiance>p.irradiance);
});
test('finite pixel apertures do not make an ideal Nyquist brick wall',()=>{
 const a=pixel({...p,frequency:80000});assert.equal(a.strictlyBelowNyquist,false);assert.ok(Math.abs(a.spatialTransfer)>.2);close(a.aliasFrequency,-20000);
});
test('alias frequency and phase account for nonzero grid origin and motion center time',()=>{
 const a=pixel({...p,frequency:130000,origin:3e-6,velocity:.0002,centerTime:.03});close(a.aliasFrequency,30000);a.samples.forEach(q=>close(q.limitAverage,q.aliasAverage,2e-14));assert.ok(Math.abs(a.observe(1.2e-6).limitAverage-a.observe(1.2e-6).aliasAverage)>1e-5);
});
test('Nyquist-boundary quadrature phase can hide a nonzero continuous modulation',()=>{
 const a=pixel({...p,frequency:50000,phase:Math.PI/2});assert.equal(a.strictlyBelowNyquist,false);a.samples.forEach(q=>close(q.pointIrradiance,p.irradiance,1e-15));assert.ok(Math.abs(a.observe(2.5e-6).pointIrradiance-p.irradiance)>.001);
});
test('temporal exposure averages moving irradiance independently from the spatial aperture',()=>{
 const a=pixel({...p,frequency:25000,velocity:.001,exposure:.04});assert.equal(a.temporalTransfer,0);a.samples.forEach(q=>close(q.limitAverage,p.irradiance));assert.notEqual(a.spatialTransfer,0);
});
test('zero aperture and zero exposure retain only the point/instantaneous limit,with zero physical collected energy',()=>{
 for(const v of [{...p,apertureWidth:0},{...p,exposure:0}]){const a=pixel(v);assert.equal(a.apertureAverageDefined,false);a.samples.forEach(q=>{assert.equal(q.averageIrradiance,null);assert.equal(q.energy,0);assert.equal(q.incidentPhotonMean,0);assert.ok(q.limitAverage>=0);});}
});
test('finite sinc reconstruction interpolates all retained samples but does not assert perfect between-sample recovery',()=>{
 const a=pixel({...p,frequency:20000});a.samples.forEach(q=>close(a.observe(q.coordinate).finiteReconstruction,q.limitAverage,2e-14));const q=a.observe(4.3*p.pitch);assert.ok(Math.abs(q.finiteReconstruction-q.limitAverage)>1e-6);
});
test('display profiles resolve highest original frequency rather than aliasing it into a fabricated smooth curve',()=>{
 const a=pixel({...p,frequency:4/p.pitch,radius:32});const step=a.profile[1].coordinate-a.profile[0].coordinate;assert.ok(step*4/p.pitch<=1/23.99);
});
test('radiometric calibration uses photon energy with exact SI h and c and correct area/time units',()=>{
 const a=exposure(.001,25e-12,.01,550e-9);close(a.incidentPhotonMean,692.1910280371228,1e-13);close(a.photonEnergy,3.611719740270779e-19,1e-30);close(exposure(.001,25e-12,.02,550e-9).incidentPhotonMean,2*a.incidentPhotonMean);assert.equal(exposure(0,25e-12,.01,550e-9).incidentPhotonMean,0);
});
test('SI irradiance in the previous relative optical model cannot enter through an unvalidated unit coercion',()=>{
 assert.throws(()=>exposure(.001,25e-12,.01,550));assert.throws(()=>pixel({...p,pitch:10}));assert.throws(()=>pixel({...p,apertureWidth:20e-6}));
});
test('pixel observation closes over a defensive input snapshot',()=>{
 const input={...p};const a=pixel(input),before=a.observe(1e-6);input.frequency=0;input.irradiance=10;assert.deepEqual(a.observe(1e-6),before);a.samples[0].limitAverage=1e6;assert.deepEqual(a.observe(1e-6),before);
});
test('all pixel scalar controls reject missing,NaN,infinite and invalid integer inputs',()=>{
 for(const key of Object.keys(p) as (keyof PixelSamplingInput)[]){assert.throws(()=>pixel({...p,[key]:NaN}));assert.throws(()=>pixel({...p,[key]:undefined}));}
 assert.throws(()=>pixel({...p,radius:2.1}));assert.throws(()=>pixel({...p,modulation:1.01}));assert.throws(()=>pixel({...p,frequency:-1}));assert.throws(()=>pixel(p).observe(Infinity));
});
test('Poisson centered recurrence is finite and unnormalized from zero through the largest supported mean',()=>{
 for(const mean of [0,.00001,.1,1,16,206,256,4096,4224]){const a=poisson(mean);assert.ok(a.values.every(q=>q.probability>=0&&Number.isFinite(q.probability)));close(a.coveredProbability,1,2e-11);close(a.values.reduce((v,q)=>v+q.count*q.probability,0),mean,2e-11);assert.ok(a.omittedProbabilityBound>=0&&a.omittedProbabilityBound<1e-10);}
});
test('QE produces photoelectron mean and shot variance while dark exposure and read variance retain electron units',()=>{
 const a=sensor(s);close(a.photoMean,6);close(a.darkMean,.2);close(a.electronMean,6.2);close(a.electronVariance,10.2);close(a.unsaturatedAnalogMean,12.34);close(a.unsaturatedAnalogVariance,4.998);close(a.unsaturatedSignalToNoise!,6/Math.sqrt(10.2));
});
test('gain scales mean signal and standard deviation equally before quantization and clipping',()=>{
 const a=sensor(s),b=sensor({...s,gain:1.4});close(b.unsaturatedAnalogMean-s.offset,2*(a.unsaturatedAnalogMean-s.offset));close(b.unsaturatedAnalogVariance,4*a.unsaturatedAnalogVariance);assert.equal(a.unsaturatedSignalToNoise,b.unsaturatedSignalToNoise);
});
test('dark-mean subtraction does not remove independent dark shot variance',()=>{
 const a=sensor({...s,darkRate:0}),b=sensor(s);close(b.electronVariance-a.electronVariance,.2);close((b.unsaturatedAnalogMean-s.gain*b.darkMean),a.unsaturatedAnalogMean);
});
test('zero photon QE retains dark/read preparation while zero exposure forbids an inconsistent integrated photon mean',()=>{
 const a=sensor({...s,quantumEfficiency:0});assert.equal(a.photoMean,0);close(a.electronMean,.2);assert.ok(a.adcVariance>0);assert.throws(()=>sensor({...s,exposure:0}));const b=sensor({...s,exposure:0,incidentPhotonMean:0});assert.equal(b.electronMean,0);close(b.electronVariance,4);
});
test('a completely deterministic dark camera has zero exact quantized variance,not the uniform 1/12 approximation',()=>{
 const a=sensor({...s,incidentPhotonMean:0,darkRate:0,readSigma:0,offset:8.2});assert.equal(a.adcMean,8);assert.equal(a.adcVariance,0);assert.equal(a.unsaturatedSignalToNoise,null);assert.equal(a.quantizationVarianceApproximation,1/12);
});
test('full-well clipping occurs before read noise and differs from ADC clipping after gain',()=>{
 const a=sensor({...s,incidentPhotonMean:100,quantumEfficiency:1,readSigma:0,fullWell:12,gain:1,offset:0});close(a.storedMean,12,1e-11);close(a.adcMean,12,1e-11);assert.ok(a.fullWellOverflowProbability>.999999);assert.equal(a.adcUpperClippingProbability,0);
 const b=sensor({...s,incidentPhotonMean:100,quantumEfficiency:1,readSigma:0,fullWell:8192,bits:4,gain:1,offset:0});assert.ok(b.adcUpperClippingProbability>.999999);close(b.adcMean,15,1e-10);assert.ok(b.storedMean>99);assert.equal(b.fullWellOverflowProbability,0);
});
test('rail occupancy is distinct from an out-of-range rounded value requiring clipping',()=>{
 const a=sensor({...s,incidentPhotonMean:0,darkRate:0,readSigma:0,offset:15,bits:4});assert.equal(a.upperRailOccupancy,1);assert.equal(a.adcUpperClippingProbability,0);
 const b=sensor({...s,incidentPhotonMean:0,darkRate:0,readSigma:0,offset:0,bits:4});assert.equal(b.lowerRailOccupancy,1);assert.equal(b.adcLowerClippingProbability,0);
});
test('exact half-code ties and unsigned endpoints follow the declared rounding convention',()=>{
 assert.equal(sensorQuantize(.5,4),1);assert.equal(sensorQuantize(-.5,4),0);assert.equal(sensorQuantize(14.5,4),15);assert.equal(sensorQuantize(15.5,4),15);assert.equal(sensorQuantize(-8,4),0);assert.throws(()=>sensorQuantize(Infinity,4));assert.throws(()=>sensorQuantize(0,4.5));
});
test('shot-limited signal-to-noise grows with square root rather than with gain',()=>{
 const a=sensor({...s,incidentPhotonMean:25,quantumEfficiency:1,darkRate:0,readSigma:0});const b=sensor({...s,incidentPhotonMean:100,quantumEfficiency:1,darkRate:0,readSigma:0});assert.equal(a.unsaturatedSignalToNoise,5);assert.equal(b.unsaturatedSignalToNoise,10);
});
test('sensor input validation covers all missing scalar values and physical or discrete domain violations',()=>{
 for(const key of Object.keys(s) as (keyof PhotonSensorInput)[]){assert.throws(()=>sensor({...s,[key]:NaN}));assert.throws(()=>sensor({...s,[key]:undefined}));}
 for(const q of [{quantumEfficiency:1.1},{readSigma:-1},{gain:0},{bits:7.5},{fullWell:2.5},{exposure:-1}])assert.throws(()=>sensor({...s,...q}));assert.throws(()=>poisson(-1));assert.throws(()=>poisson(Infinity));
});

for(const id of ['pixel-aperture-sampling-aliasing','photon-counting-exposure-sensor-noise'])test(id+' has substantive bilingual theory,worked exercises and valid actual strict KaTeX rendering',async()=>{
 const {getLearningContent}=await import('../src/lib/learning-content');const a=await getLearningContent(id);
 assert.ok(a.outline.en.length>=16);assert.equal(a.outline.en.length,a.outline.ko.length);assert.ok(a.en.length>20000);assert.ok(a.ko.length>15000);assert.doesNotMatch(a.en,/[가-힣]/);assert.doesNotMatch(a.en,/katex-error/);assert.doesNotMatch(a.ko,/katex-error/);
});
