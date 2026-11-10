import test from 'node:test';import assert from 'node:assert/strict';
import refs from './fixtures/interference-diffraction-independent.json';
import{coherentDetectorPhaseScan,coherentSuperpositionDetector as interference,finiteApertureDiffraction as aperture,type InterferenceExperiment,type ApertureExperiment}from'../src/lib/learning-interference-diffraction';
const close=(a:number,b:number,tol=2e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=2e-10)=>a.forEach((v,i)=>close(v,b[i],tol));
const base:InterferenceExperiment={amplitude1:1,amplitude2:1,omega1:2*Math.PI,omega2:2*Math.PI,phaseDifference:0,timeCenter:0,windowDuration:1,population:[{phase:0,weight:1}]};
const slit:ApertureExperiment={wavelength:500e-9,slitWidth:50e-6,fresnelNumber:1,separationRatio:2,amplitude1:1,amplitude2:1,relativePhase:0,slits:1};
test('48 independently integrated field populations reproduce means,phase moments and observed contrast',()=>{
 assert.equal(refs.interference.length,48);assert.equal(refs.provenance.symbolicNonidentifiability,true);
 for(const r of refs.interference){const a=interference(r.system);for(const k of['meanIntensity','realSquareMean','baseline','minimum','maximum']as const)close(a[k],r[k]);vector(a.phaseMoment,r.phaseMoment);if(r.visibility===null)assert.equal(a.visibility,null);else close(a.visibility!,r.visibility);assert.ok(a.minimum>=0&&a.maximum>=a.minimum);assert.ok(a.realSquareMean>=0);assert.ok(a.realMeanRefinementDifference<1e-10);close(a.realMeanFormulaResidual,0,1e-10);}
});
test('36 independent slit systems and468 observation points agree with direct kernel and special-function references',()=>{
 assert.equal(refs.apertures.length,36);assert.ok(refs.provenance.maxDirectSpecialDifference<2e-11);
 for(const row of refs.apertures)for(const e of row.states){const a=aperture(row.system as ApertureExperiment,e.eta);vector(a.fresnel,e.fresnel,4e-11);vector(a.fraunhofer,e.fraunhofer,4e-11);close(a.fresnelIntensity,e.fresnelIntensity);close(a.fraunhoferIntensity,e.fraunhoferIntensity);close(a.expectedTransmittedNorm,row.expectedTransmittedNorm);assert.ok(a.refinementDifference<2e-11);}
});
test('coherent bright and dark fringes add fields before taking the normalized squared magnitude',()=>{
 close(interference(base).meanIntensity,4);const a=interference({...base,phaseDifference:Math.PI});assert.ok(a.meanIntensity<1e-28);close(a.crossTerm,-2);close(a.visibility!,1);close(interference({...base,phaseDifference:Math.PI/2}).meanIntensity,2);
});
test('finite phase populations average realizations rather than squaring the mean field',()=>{
 const a=interference({...base,population:[{phase:0,weight:1},{phase:Math.PI,weight:1}]});close(a.meanIntensity,2);close(a.visibility!,0);close(a.phaseMoment[0],0);assert.equal(a.realizations.length,2);close(a.realizations.reduce((sum,p)=>sum+p.probability*p.instantaneousIntensity,0),a.instantaneousIntensity);
});
test('amplitude imbalance and partial coherence can generate the identical complete fringe',()=>{
 const unequal={...base,amplitude1:Math.sqrt((2+Math.sqrt(3))/2),amplitude2:Math.sqrt((2-Math.sqrt(3))/2)},partial={...base,population:[{phase:0,weight:3},{phase:Math.PI,weight:1}]};
 for(const phase of[0,.3,1.2,Math.PI,4.3,2*Math.PI]){const a=interference({...unequal,phaseDifference:phase}),b=interference({...partial,phaseDifference:phase});close(a.meanIntensity,2+Math.cos(phase));close(a.meanIntensity,b.meanIntensity);close(a.visibility!,.5);close(b.visibility!,.5);}
});
test('zero exposure means an instantaneous observation,not a divide-by-zero or cycle average',()=>{
 const a=interference({...base,omega2:3,windowDuration:0,timeCenter:.3});close(a.meanIntensity,a.instantaneousIntensity);close(a.realSquareMean,a.realizations[0].realField**2);close(a.windowFactor,1);
});
test('real-field square averaging differs from normalized analytic intensity until carrier terms average out',()=>{
 const cycle=interference(base);close(cycle.meanIntensity,4);close(cycle.realSquareMean,2);
 const dc=interference({...base,omega1:0,omega2:0,phaseDifference:Math.PI/2});close(dc.meanIntensity,2);close(dc.realSquareMean,1);
});
test('negative finite-window sinc reverses fringe phase without inventing negative intensity',()=>{
 const s={...base,omega1:0,omega2:2*Math.PI,windowDuration:1.5},a=interference(s),b=interference({...s,phaseDifference:Math.PI});assert.ok(a.windowFactor<0);assert.ok(a.meanIntensity<2&&b.meanIntensity>2);close(a.visibility!,Math.abs(a.windowFactor));assert.ok(a.mutualCoherence12![0]<0);
});
test('time-center evolution and relative phase describe the same deterministic fringe shift',()=>{
 const a=interference({...base,omega2:3,timeCenter:1}),b=interference({...base,omega2:3,phaseDifference:base.omega1-3,timeCenter:0});close(a.meanIntensity,b.meanIntensity);
});
test('missing branches and complete darkness distinguish unavailable coherence from zero contrast',()=>{
 const a=interference({...base,amplitude2:0});assert.equal(a.mutualCoherence12,null);assert.equal(a.visibility,0);close(a.meanIntensity,1);
 const b=interference({...base,amplitude1:0,amplitude2:0});assert.equal(b.mutualCoherence12,null);assert.equal(b.visibility,null);assert.equal(b.meanIntensity,0);assert.equal(b.realSquareMean,0);
});
test('population weight normalization preserves a common rescaling and zero-weight realizations',()=>{
 const p=[{phase:.3,weight:1},{phase:2.4,weight:2}],a=interference({...base,population:p}),b=interference({...base,population:p.map(x=>({...x,weight:x.weight*4})).concat({phase:-2,weight:0})});close(a.meanIntensity,b.meanIntensity);vector(a.phaseMoment,b.phaseMoment);
});
test('single-slit far field has the full-width sinc zeros without per-curve peak normalization',()=>{
 const a=aperture({...slit,fresnelNumber:.1},0);close(a.fraunhoferIntensity,.1);for(const x of[-2,-1,1,2])assert.ok(aperture(slit,x).fraunhoferIntensity<1e-29);close(aperture({...slit,amplitude1:.5},0).fraunhoferIntensity,.25);
});
test('two touching equal-phase slits reproduce one slit twice as wide at the same distance',()=>{
 const s={...slit,slits:2 as const,separationRatio:1},wide={...slit,slitWidth:100e-6,fresnelNumber:4};for(const eta of[-.7,0,.3,.8]){const a=aperture(s,eta),b=aperture(wide,2*eta);close(a.distance,b.distance);vector(a.fresnel,b.fresnel);vector(a.fraunhofer,b.fraunhofer);close(a.expectedTransmittedNorm,b.expectedTransmittedNorm);}
});
test('blocking a displaced slit changes phase,not the far-field intensity of the remaining slit',()=>{
 for(const eta of[-1.3,-.4,0,.7]){const a=aperture({...slit,slits:2,amplitude2:0},eta),b=aperture(slit,eta);close(a.fraunhoferIntensity,b.fraunhoferIntensity);close(a.expectedTransmittedNorm,b.expectedTransmittedNorm);}
});
test('translation of a single active slit translates its Fresnel field in physical coordinates',()=>{
 const shifted={...slit,slits:2 as const,amplitude2:0,fresnelNumber:.5};for(const eta of[-.8,0,.4]){const a=aperture(shifted,eta),b=aperture({...slit,fresnelNumber:.5},eta+.5);vector(a.fresnel,b.fresnel);}
});
test('an antisymmetric double aperture has an on-axis dark field with unchanged transmitted norm',()=>{
 const a=aperture({...slit,slits:2,relativePhase:Math.PI},0),b=aperture({...slit,slits:2},0);assert.ok(a.fresnelIntensity<1e-28);assert.ok(a.fraunhoferIntensity<1e-28);close(a.expectedTransmittedNorm,b.expectedTransmittedNorm);assert.ok(b.fraunhoferIntensity>0);
});
test('the far-field comparison converges at fixed scaled coordinate and retains observation-plane phase',()=>{
 for(const eta of[-1.3,-.4,0,.7]){const a=aperture({...slit,fresnelNumber:.1},eta),b=aperture({...slit,fresnelNumber:.01},eta);assert.ok(b.complexFieldDifference/Math.sqrt(.01)<a.complexFieldDifference/Math.sqrt(.1)/5);close(a.scaledFraunhoferIntensity,b.scaledFraunhoferIntensity);}
});
test('propagation geometry,aperture power and physical approximation diagnostics remain distinct',()=>{
 const a=aperture(slit,.3),b=aperture({...slit,fresnelNumber:.1},.3);close(a.distance,.005);close(b.distance,10*a.distance);close(a.expectedTransmittedNorm,50e-6);close(a.discardedAperturePhase,Math.PI/4);assert.ok(a.maximumTransverseRatio>0&&a.fourthOrderDistancePhaseEstimate>0);close(a.scaledFresnelIntensity,a.fresnelIntensity/a.fresnelNumber);
});
test('all blocked apertures have zero fields and no invented transmitted norm',()=>{
 const a=aperture({...slit,slits:2,amplitude1:0,amplitude2:0},1);vector(a.fresnel,[0,0]);vector(a.fraunhofer,[0,0]);assert.equal(a.expectedTransmittedNorm,0);assert.equal(a.apertureHalfExtent,0);
});
test('invalid physical parameters,populations and overlapping aperture choices are rejected',()=>{
 for(const value of[-1,NaN,Infinity])assert.throws(()=>interference({...base,windowDuration:value}));assert.throws(()=>interference({...base,population:[]}));assert.throws(()=>interference({...base,population:new Array(1)}));assert.throws(()=>interference({...base,population:[{phase:0,weight:0}]}));assert.throws(()=>interference({...base,population:[{phase:NaN,weight:1}]}));assert.throws(()=>interference(base,Infinity));
 for(const [key,value]of Object.entries({wavelength:0,slitWidth:0,fresnelNumber:0,separationRatio:.5,relativePhase:NaN,amplitude1:-1,slits:3}))assert.throws(()=>aperture({...slit,[key]:value},0));assert.throws(()=>aperture(slit,2.1));
});
test('nearly dark fringes retain nonnegative small intensity rather than subtracting rounded bright totals',()=>{
 const d=1e-9,a=interference({...base,phaseDifference:Math.PI+d});assert.ok(a.meanIntensity>0&&a.meanIntensity<2e-18);assert.ok(a.minimum>=0&&a.visibility!<=1);
});
test('uniform phase populations keep a physical contrast even when only roundoff modulation remains',()=>{
 for(let n=2;n<=16;n++){const population=Array.from({length:n},(_,i)=>({phase:2*Math.PI*i/n,weight:1})),a=interference({...base,population});close(a.meanIntensity,2);assert.ok(a.visibility!>=0&&a.visibility!<=1);close(a.visibility!,0);}
});
for(const id of ['coherent-superposition-detector-averages','finite-aperture-fresnel-fraunhofer'])for(const lang of ['ko','en']as const)test(`${id}.${lang}: detailed bilingual theory,six explained exercises and strict math render`,async()=>{
 const{readFile}=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');const raw=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),rendered=await getLearningContent(id);
 assert.ok(raw.length>6500);assert.equal(rendered.outline[lang].length,16);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);assert.equal(new Set(rendered.outline[lang].map(h=>h.id)).size,16);assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));
 assert.equal((raw.match(/^### (?:Exercise|문제) [1-6]/gm)||[]).length,6);assert.equal((raw.match(lang==='en'?/\*\*Solution:\*\*/g:/\*\*해설:\*\*/g)||[]).length,6);assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);
});
test('a tiny positive exposure at a large center retains its instantaneous limit instead of collapsing endpoints',()=>{
 const ref=refs.tinyExposure,a=interference(ref.system),b=interference({...ref.system,windowDuration:0});assert.equal(ref.decimalPrecision,70);assert.ok(a.realSquareMean>.05);close(a.realSquareMean,ref.realSquareMean,1e-13);close(a.meanIntensity,ref.meanIntensity,1e-13);close(a.realSquareMean,b.realSquareMean,1e-13);
});
test('efficient absolute-phase scan agrees with complete detector evaluations and retains canonical phase endpoints',()=>{
 for(const row of refs.interference.slice(0,12)){const s=row.system,scan=coherentDetectorPhaseScan(s,33);assert.equal(scan[0].phase,-Math.PI);assert.equal(scan[32].phase,Math.PI);for(const i of[0,8,16,24,32])close(scan[i].intensity,interference({...s,phaseDifference:scan[i].phase}).meanIntensity);}
 assert.throws(()=>coherentDetectorPhaseScan(base,1));assert.throws(()=>coherentDetectorPhaseScan(base,2.5));assert.equal(interference({...base,windowDuration:0}).realQuadratureSubdivisions,0);
});
