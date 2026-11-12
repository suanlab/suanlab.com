import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {unified} from 'unified';import remarkParse from 'remark-parse';import remarkMath from 'remark-math';import remarkRehype from 'remark-rehype';import rehypeKatex from 'rehype-katex';import rehypeStringify from 'rehype-stringify';
import refs from './fixtures/fourier-optics-independent.json';
import {coherentFourierImaging as coherent,incoherentFourierImaging as incoherent,fourierPupilTransfer as transfer,fourierPupilOTF as otf,fourierPointSpread as psf,type FourierPupil,type CoherentFourierInput,type IncoherentFourierInput} from '../src/lib/learning-fourier-optics';
const p:FourierPupil={wavelength:500e-9,focalLength1:.1,focalLength2:.1,referenceHalfWidth:.001,apertureFraction:1,obstructionFraction:0,amplitudeTransmission:1,quadraticPhase:0,linearPhase:0};
const c:CoherentFourierInput={...p,period:100e-6,modes:[{order:0,amplitude:1,phase:0},{order:1,amplitude:.25,phase:0},{order:-1,amplitude:.25,phase:0}],globalPhase:0,queryCoordinate:0};
const s:IncoherentFourierInput={...p,sources:[{position:0,weight:1}],queryCoordinate:0,gratingMean:1,gratingModulation:.8,gratingFrequency:15000,gratingPhase:0};
const close=(a:number,b:number,tol=5e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const complex=(a:readonly number[],b:readonly number[],tol=5e-10)=>a.forEach((v,i)=>close(v,b[i],tol));
test('48 independent continuous pupils,816direct pupil integrals and1200shifted-pupil autocorrelations agree',()=>{
 assert.equal(refs.provenance.productImports,false);assert.equal(refs.pupils.length,48);
 for(const r of refs.pupils){for(const v of r.psf){const a=psf(r.system,v.coordinate);complex(a.field,v.field,3e-10);close(a.rawIntensity,v.intensity,3e-9);assert.ok(a.refinementDifference<2e-8*Math.max(1,Math.hypot(...a.field)));}for(const v of r.otf){const a=otf(r.system,v.frequency);complex(a.raw,v.value,3e-10);}}
});
test('48finite harmonic preparations and816independent field observations retain duplicate-order interference and Parseval means',()=>{
 assert.equal(refs.coherent.length,48);for(const r of refs.coherent){const a=coherent(r.system);close(a.inputMeanIntensity,r.inputMean);close(a.outputMeanIntensity,r.outputMean);for(const v of r.observations){const z=a.observe(v.coordinate);complex(z.inputField,v.before);complex(z.outputField,v.after);}}
});
test('48incoherent systems and432positive image observations agree with independent pupil integrals',()=>{
 assert.equal(refs.incoherent.length,48);for(const r of refs.incoherent){const a=incoherent(r.system);close(a.pointImageIntegral,r.pointIntegral);close(a.gratingReferenceMean,r.gratingMean);for(const v of r.observations){const q=a.observe(v.coordinate);close(q.pointIntensity,v.pointIntensity,3e-9);close(q.gratingOutput,v.gratingIntensity,3e-9);assert.ok(q.contributions.every(c=>c.intensity>=0));}}
});
test('physical Fourier-plane coordinates use cycles/m and relay magnification includes parity and line-density scaling',()=>{
 const a=coherent({...c,focalLength2:.2,queryCoordinate:25e-6});close(a.pupil.magnification,-2);close(a.query.actualCoordinate,-50e-6);close(a.query.actualOutputIntensity,a.query.outputIntensity/2);close(a.actualImagePeriod,200e-6);close(a.modes.find(m=>m.order===1)!.fourierPlaneCoordinate,.0005);close(a.outputPeriodIntegral,a.outputMeanIntensity*a.actualImagePeriod/2);
});
test('clear coherent imaging preserves a three-mode complex field while narrower pupils discard sidebands rather than rescaling the field',()=>{
 const a=coherent(c),b=coherent({...c,apertureFraction:.4});close(a.inputMeanIntensity,1.125);close(a.outputMeanIntensity,1.125);close(a.query.outputIntensity,2.25);close(b.outputMeanIntensity,1);close(b.transmittance!,8/9);close(b.pupil.referenceThroughput,.4);assert.notEqual(b.transmittance,b.pupil.referenceThroughput);b.profile.forEach(q=>close(q.outputIntensity,1));
});
test('blocking the zero order creates a squared cosine intensity harmonic without inventing an extended coherent field passband',()=>{
 const a=coherent({...c,obstructionFraction:.2});close(a.outputMeanIntensity,.125);close(a.transmittance!,1/9);close(a.query.outputIntensity,.25);close(a.observe(c.period/4).outputIntensity,0);complex(a.modes.find(m=>m.order===0)!.transfer,[0,0]);
});
test('weak complex phase preparation converts phase to first-order visible intensity without changing mean power',()=>{
 const base={...c,modes:[{order:0,amplitude:1,phase:0},{order:1,amplitude:.1,phase:Math.PI/2},{order:-1,amplitude:.1,phase:Math.PI/2}]};
 const a=coherent(base),b=coherent({...base,quadraticPhase:2*Math.PI});close(a.query.inputIntensity,1.04);close(b.query.outputIntensity,.64);close(b.observe(c.period/2).outputIntensity,1.44);close(a.outputMeanIntensity,b.outputMeanIntensity);close(b.outputMeanIntensity,1.02);
});
test('a linear pupil phase shifts the reduced field with the declared sign and actual image inversion changes its coordinate',()=>{
 const a=coherent({...c,linearPhase:Math.PI/2,focalLength2:.2,queryCoordinate:-12.5e-6});close(a.query.outputIntensity,2.25);close(a.query.actualCoordinate,25e-6);close(a.query.actualOutputIntensity,1.125);
});
test('common input phase changes the actual complex field but leaves all intensities and period integrals unchanged',()=>{
 const a=coherent(c),b=coherent({...c,globalPhase:.7});close(a.query.outputIntensity,b.query.outputIntensity);close(a.outputPeriodIntegral,b.outputPeriodIntegral);assert.ok(Math.hypot(a.query.outputField[0]-b.query.outputField[0],a.query.outputField[1]-b.query.outputField[1])>.1);
});
test('duplicate preparations combine before energy accounting,including exact destructive cancellation with no normalized transmission',()=>{
 const a=coherent({...c,modes:[{order:1,amplitude:1,phase:0},{order:1,amplitude:1,phase:Math.PI}]});assert.equal(a.inputMeanIntensity,0);assert.equal(a.outputMeanIntensity,0);assert.equal(a.transmittance,null);assert.equal(a.query.outputPhase,null);
});
test('hard pupil edges are excluded exactly while physically distinct neighbouring frequencies are not snapped into the boundary',()=>{
 complex(transfer(p,20000),[0,0]);complex(transfer(p,-20000),[0,0]);assert.ok(transfer(p,20000-1e-8)[0]>.99);complex(transfer(p,20000+1e-8),[0,0]);complex(transfer({...p,obstructionFraction:.5},10000),[1,0]);complex(transfer({...p,obstructionFraction:.5},10000-1e-8),[0,0]);
});
test('clear 1Dslit yields signed sinc amplitude,nonnegative intensity and triangular conditional OTF',()=>{
 const a=psf(p,0),b=psf(p,37.5e-6);complex(a.field,[40000,0]);close(a.conditionalPSF!,40000);assert.ok(b.field[0]<0);assert.ok(b.rawIntensity>0);for(const nu of[0,10000,20000,30000,40000,50000])close(otf(p,nu).conditional![0],Math.max(0,1-nu/40000));
});
test('incoherent cutoff differs from coherent field cutoff without applying an amplitude mask to image intensity',()=>{
 complex(transfer(p,30000),[0,0]);close(otf(p,30000).mtf!,.25);const a=incoherent({...s,gratingFrequency:30000});close(a.gratingObservedContrast!,.2);close(a.query.gratingOutput,1.2);
});
test('raw pupil energy,reference throughput and conditional PSF normalization remain separate under aperture and amplitude loss',()=>{
 const a=incoherent({...s,apertureFraction:.5,amplitudeTransmission:.5,sources:[{position:0,weight:1},{position:10e-6,weight:.4}]});close(a.pupil.pupilEnergy,5000);close(a.pupil.referenceThroughput,.125);close(a.pointImageIntegral,.175);const h=psf({...p,apertureFraction:.5,amplitudeTransmission:.5},0);close(h.conditionalPSF!,20000);close(h.referenceIntensity,2500);close(a.gratingReferenceMean,.125);
});
test('defocus-like phase pupils can have negative real OTF and invert contrast while PSF and image intensities remain nonnegative',()=>{
 const a=incoherent({...s,quadraticPhase:6}),v=.625*Math.sin(5.625)/5.625;close(a.gratingTransfer.conditional![0],v);assert.ok(v<0);close(a.gratingTransfer.mtf!,-v);close(Math.abs(a.gratingTransfer.phase!),Math.PI);assert.ok(a.query.gratingOutput<1);assert.ok(a.observe(1/(2*s.gratingFrequency)).gratingOutput>1);assert.ok(a.psfProfile.every(v=>v.rawIntensity>=0));assert.ok(a.pointImageProfile.every(v=>v.pointIntensity>=0&&v.gratingOutput>=0));
});
test('a linear pupil phase changes OTF phase while preserving its magnitude and mirrors the actual image displacement',()=>{
 const a=otf(p,15000),b=otf({...p,linearPhase:.8},15000);close(a.mtf!,b.mtf!);close(b.phase!,.6);close(b.conditional![0],a.mtf!*Math.cos(.6));close(b.conditional![1],a.mtf!*Math.sin(.6));
});
test('fully blocked apertures,complete central obstruction and zero amplitude have no conditional PSF or OTF',()=>{
 for(const patch of[{apertureFraction:0},{obstructionFraction:1},{amplitudeTransmission:0}]){const a=incoherent({...s,...patch}),h=psf({...p,...patch},0);assert.equal(a.pupil.referenceThroughput,0);assert.equal(a.pointImageIntegral,0);assert.equal(a.gratingTransfer.conditional,null);assert.equal(a.gratingTransfer.mtf,null);assert.equal(a.gratingObservedContrast,null);assert.equal(h.conditionalPSF,null);assert.equal(h.fieldPhase,null);assert.ok(a.pointImageProfile.every(v=>v.pointIntensity===0&&v.gratingOutput===0));}
});
test('zero sources preserve device reference responses while observed point transmittance is undefined',()=>{
 const a=incoherent({...s,sources:[{position:0,weight:0}]});assert.equal(a.pointImageIntegral,0);assert.equal(a.pointTransmittance,null);assert.ok(a.gratingTransfer.conditional!==null);assert.ok(a.pointImageProfile.every(v=>v.pointIntensity===0));
});
test('zero-frequency gratings are constant preparations rather than fringes with an invented Michelson contrast',()=>{
 const a=incoherent({...s,gratingFrequency:0,gratingModulation:1,gratingPhase:Math.PI});assert.equal(a.gratingObjectMean,0);assert.equal(a.gratingReferenceMean,0);assert.equal(a.gratingActualMean,0);assert.equal(a.gratingObservedContrast,null);assert.ok(a.pointImageProfile.every(v=>v.gratingInput===0&&v.gratingOutput===0));
 const b=incoherent({...s,gratingFrequency:0,gratingModulation:.5,gratingPhase:0});close(b.gratingObjectMean,1.5);close(b.gratingReferenceMean,1.5);
});
test('positive point-source intensities add without coherent cross terms',()=>{
 const a=incoherent({...s,sources:[{position:0,weight:1},{position:0,weight:1}]});close(a.query.pointIntensity,80000);assert.notEqual(a.query.pointIntensity,160000);close(a.pointImageIntegral,2);close(a.query.contributions.reduce((v,p)=>v+p.intensity,0),a.query.pointIntensity);
});
test('image density scales with magnification while the whole-line point-source integral is invariant',()=>{
 const a=incoherent(s),b=incoherent({...s,focalLength2:.2});close(a.pointImageIntegral,b.pointImageIntegral);close(b.query.actualPointIntensity,a.query.pointIntensity/2);close(b.query.actualGratingOutput,a.query.gratingOutput/2);
});
test('finite-window PSF samples are not secretly normalized into whole-line unit mass',()=>{
 assert.equal(refs.finiteWindowMasses.length,5);for(const r of refs.finiteWindowMasses){close(r.numericalMass,r.analyticMass);assert.ok(r.numericalMass<1);}assert.ok(refs.finiteWindowMasses[0].numericalMass<.4);assert.ok(refs.finiteWindowMasses.at(-1)!.numericalMass>.98);
});
test('grating display resolves the highest supported spatial frequency independently of the fixed point-source window',()=>{
 for(const frequency of[0,100,15000,50000]){const a=incoherent({...s,gratingFrequency:frequency});assert.equal(a.gratingProfile.length,129);const step=a.gratingProfile[1].coordinate-a.gratingProfile[0].coordinate;assert.ok(frequency*step<=1/16);a.gratingProfile.forEach(q=>close(q.gratingOutput,a.observe(q.coordinate).gratingOutput));assert.equal(a.pointImageProfile.length,129);}
});
test('quadrature refinement is exposed as a difference rather than a certified error bound',()=>{
 const a=psf({...p,quadraticPhase:8,linearPhase:1.1},20e-6);assert.equal(a.quadratureStatus,'refinement-comparison-not-certified-error-bound');assert.ok(a.quadraturePanels>0);assert.ok(Number.isFinite(a.refinementDifference));
});
test('fresh outputs and cloned preparations do not mutate input modes or source weights',()=>{
 const before=JSON.stringify([c,s]),a=coherent(c),b=incoherent(s);a.input.modes[0].amplitude=0;b.input.sources[0].weight=0;assert.equal(JSON.stringify([c,s]),before);assert.notEqual(coherent(c).input.modes,c.modes);const external={...s,sources:[{position:0,weight:1}]},stable=incoherent(external),value=stable.observe(0).pointIntensity;external.sources[0].weight=0;close(stable.observe(0).pointIntensity,value);
});
test('malformed,sparse,nonfinite or physically invalid preparations are rejected without changing the experiment',()=>{
 for(const patch of[{wavelength:0},{focalLength1:0},{apertureFraction:-.1},{obstructionFraction:1.1},{quadraticPhase:NaN},{amplitudeTransmission:2}])assert.throws(()=>coherent({...c,...patch}));
 assert.throws(()=>coherent({...c,modes:[]}));assert.throws(()=>coherent({...c,modes:new Array(2)}));assert.throws(()=>coherent({...c,modes:[{order:.5,amplitude:1,phase:0}]}));assert.throws(()=>incoherent({...s,sources:new Array(2)}));assert.throws(()=>incoherent({...s,sources:[{position:0,weight:-1}]}));assert.throws(()=>incoherent({...s,gratingModulation:2}));assert.throws(()=>psf(p,1));assert.throws(()=>otf(p,Infinity));
});
for(const id of ['coherent-fourier-pupil-spatial-filtering','incoherent-psf-otf-image-formation'])for(const lang of ['ko','en']as const)test(`${id}.${lang} detailed theory,worked examples,six explained exercises and strict actual Markdown render`,async()=>{
 const raw=fs.readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');assert.ok(raw.length>(lang==='en'?11000:7000));const research=raw.match(/^## (?:Research extension:|연구 확장:).*\n[\s\S]*?(?=^## )/gm)||[];assert.equal(research.length,1);const original=raw.replace(research[0],'');assert.equal((raw.match(/^## /gm)||[]).length,17);assert.equal((original.match(/^## /gm)||[]).length,16);assert.equal((raw.match(/^### /gm)||[]).length,11);assert.equal((original.match(/^### /gm)||[]).length,6);assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);const file=await unified().use(remarkParse).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(raw);assert.equal(file.messages.length,0);const{getLearningContent}=await import('../src/lib/learning-content');const rendered=await getLearningContent(id);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));
});
