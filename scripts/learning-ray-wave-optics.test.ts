import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import refs from './fixtures/ray-wave-optics-independent.json';
import { paraxialRayExperiment as rays, gaussianBeamExperiment as beam, rayTransferMatrix, freeRayMatrix, thinLensRayMatrix, multiplyRayMatrices, transformGaussianQ, type GaussianBeamInput, type ParaxialRayInput } from '../src/lib/learning-ray-wave-optics';
const close=(a:number,b:number,tol=3e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*Math.max(Math.abs(b),1e-12),`${a} != ${b}`);
const vector=(a:readonly number[],b:readonly number[],tol=3e-10)=>a.forEach((v,i)=>close(v,b[i],tol));
const r:ParaxialRayInput={objectDistance:.2,screenDistance:.2,focalLength:.1,objectHeight:.001,centralAngle:0,fanHalfAngle:.01};
const g:GaussianBeamInput={wavelength:500e-9,waistRadius:40e-6,amplitude:1,preDistance:0,postDistance:.02,focalLength:.02};
test('product-independent sequential rays cover real,virtual,infinite and absent-lens cases',()=>{
 assert.equal(refs.provenance.productImports,false);assert.equal(refs.rays.length,288);
 for(const v of refs.rays){const a=rays(v.system);if(v.imageDistance===null)assert.equal(a.imageDistance,null);else close(a.imageDistance!,v.imageDistance);if(v.imageHeight===null)assert.equal(a.imageHeight,null);else close(a.imageHeight!,v.imageHeight);a.rays.forEach((point,i)=>{const expected=v.rays[i];for(const k of ['inputAngle','lensHeight','outputAngle','screenHeight']as const)close(point[k],expected[k],2e-8);});}
});
test('96 direct two-dimensional Fresnel systems and288complex field observations agree with the q model',()=>{
 assert.equal(refs.beams.length,96);assert.ok(refs.provenance.maxQuadratureClosedFieldDifference<2e-9);assert.equal(refs.provenance.symbolicImagingDerivative,'0');
 for(const v of refs.beams){const a=beam(v.system);close(a.output.width,v.width);vector(a.output.coefficient,v.coefficient);a.prefactor.forEach((x,i)=>assert.ok(Math.abs(x-v.prefactor[i])<2e-12*(1+Math.abs(v.prefactor[i]))));close(a.intensityIntegral,v.intensityIntegral,2e-9);close(a.newWaist.coordinate,v.waistCoordinate,2e-8);close(a.newWaist.radius,v.waistRadius,2e-8);for(const p of v.observations){const actual=a.sampleField(p.radius);actual.field.forEach((x,i)=>assert.ok(Math.abs(x-p.field[i])<2e-9,`${x} vs ${p.field[i]}`));assert.ok(Math.abs(actual.intensity-p.intensity)<2e-9);}}
});
test('matrix composition uses height then slope and the imaging condition is B=0',()=>{
 const a=rays(r);close(a.imageDistance!,.2);close(a.magnification!,-1);close(a.imageHeight!,-.001);assert.equal(a.imageKind,'real');close(a.matrix[0][1],0);close(a.determinant,1);close(a.screen.rmsSpread,0,1e-3);close(a.screen.meanHeight,-.001);vector(a.matrix[0],[-1,0]);vector(a.matrix[1],[-10,-1]);
 const wrong=multiplyRayMatrices(thinLensRayMatrix(.1),freeRayMatrix(.2));assert.notDeepEqual(wrong,a.matrix);
});
test('object inside the focal length has a virtual upright enlarged image and backward extrapolations only',()=>{
 const a=rays({...r,objectDistance:.05});assert.equal(a.imageKind,'virtual');close(a.imageDistance!,-.1);close(a.magnification!,2);close(a.imageHeight!,.002);assert.ok(a.rays.every(p=>p.imageBackExtension!==null));a.rays.forEach(p=>close(p.imageBackExtension!.height,.002));assert.ok(a.screen.span>0);
});
test('negative focal length diverges rays and has a virtual reduced image',()=>{
 const a=rays({...r,focalLength:-.1});assert.equal(a.imageKind,'virtual');close(a.imageDistance!,-1/15);close(a.magnification!,1/3);close(a.imageHeight!,.001/3);assert.ok(a.rays[8].outputAngle>a.rays[0].outputAngle);
});
test('exact focal placement has an infinite image,while neighboring floating inputs retain huge signed finite images',()=>{
 const a=rays({...r,objectDistance:.1});assert.equal(a.imageKind,'at-infinity');for(const k of ['imageDistance','imageHeight','magnification','imageMatrix','imageDistanceDerivative']as const)assert.equal(a[k],null);a.rays.forEach(p=>close(p.outputAngle,-.01));
 for(const du of [-Number.EPSILON*.1,Number.EPSILON*.1]){const b=rays({...r,objectDistance:.1+du});assert.ok(Number.isFinite(b.imageDistance!)&&Math.abs(b.imageDistance!)>1e12);assert.equal(Math.sign(b.imageDistance!),Math.sign(du));}
});
test('a displaced sensor has a finite blur depending on the fan,not an invented perfect image',()=>{
 const a=rays({...r,screenDistance:.15});assert.ok(a.screen.span>0);close(a.screen.span,2*Math.abs(a.matrix[0][1])*.01);const zero=rays({...r,screenDistance:.15,fanHalfAngle:0});close(zero.screen.rmsSpread,0);assert.equal(zero.rays.length,9);
});
test('no lens is free propagation and does not manufacture an image',()=>{
 const a=rays({...r,focalLength:null});assert.equal(a.imageKind,'no-lens');assert.equal(a.imageDistance,null);vector(a.matrix[0],[1,.4]);vector(a.matrix[1],[0,1]);a.rays.forEach(p=>close(p.outputAngle,p.inputAngle));
});
test('image-distance derivative is a physical conditioning diagnostic and agrees with central differences',()=>{
 const u=.23,h=1e-6,a=rays({...r,objectDistance:u}),plus=rays({...r,objectDistance:u+h}),minus=rays({...r,objectDistance:u-h});close(a.imageDistanceDerivative!,(plus.imageDistance!-minus.imageDistance!)/(2*h),2e-9);assert.ok(Math.abs(rays({...r,objectDistance:.10001}).imageDistanceDerivative!)>1e7);
});
test('Gaussian free propagation recovers width,curvature,on-axis amplitude and full two-dimensional Gouy phase',()=>{
 const a=beam({...g,focalLength:null,preDistance:.01,postDistance:.03}),z=.04,zr=a.rayleigh;close(a.output.width,g.waistRadius*Math.sqrt(1+(z/zr)**2));close(a.output.curvature!,z*(1+(zr/z)**2));close(a.gouyAdvance,Math.atan(z/zr));vector(a.prefactor,[1/(1+(z/zr)**2),-(z/zr)/(1+(z/zr)**2)]);close(a.onAxisIntensity,(g.waistRadius/a.output.width)**2);
});
test('the chosen q half-plane yields a decaying Gaussian and is preserved by determinant-one optics',()=>{
 for(const f of[-.005,.005,.1,null])for(const u of[0,2])for(const z of[0,2]){const a=beam({...g,focalLength:f,preDistance:u,postDistance:z,waistRadius:.002});assert.ok(a.output.q[1]<0&&a.output.coefficient[0]>0);close(a.output.width**2,4*a.covariance.xx,2e-9);}
 assert.throws(()=>transformGaussianQ([0,1],freeRayMatrix(1)));assert.throws(()=>transformGaussianQ([0,-1],[[2,0],[0,1]]));
});
test('thin lens leaves radius and on-axis phase unchanged but adds negative quadratic phase and shifts inverse curvature',()=>{
 const a=beam({...g,preDistance:.03,postDistance:0}),b=beam({...g,preDistance:.03,postDistance:0,focalLength:null});close(a.lensAfter.width,a.lensBefore.width);close(a.output.width,b.output.width);vector(a.prefactor,b.prefactor);close(a.lensAfter.inverseCurvature,a.lensBefore.inverseCurvature-1/g.focalLength!);close(a.output.coefficient[1]-b.output.coefficient[1],a.k/(2*g.focalLength!));
});
test('a waist at a positive lens focuses before the geometric focal plane with nonzero minimum width',()=>{
 const a=beam(g),f=g.focalLength!,zr=a.rayleigh;close(a.newWaist.coordinate,f*zr*zr/(zr*zr+f*f));close(a.newWaist.radius,g.waistRadius*f/Math.sqrt(zr*zr+f*f));assert.ok(a.newWaist.coordinate<f&&a.newWaist.radius>0);const at=beam({...g,postDistance:a.newWaist.coordinate});close(at.output.width,at.newWaist.radius);assert.ok(Math.abs(at.output.inverseCurvature)<1e-10);assert.ok(Number.isFinite(at.onAxisIntensity));
});
test('a diverging lens has a backward-extrapolated waist,not a forward propagation to negative distance',()=>{
 const a=beam({...g,focalLength:-.02});assert.ok(a.newWaist.coordinate<0);assert.equal(a.newWaist.location,'backward-extrapolated');assert.ok(a.longitudinal.every(p=>p.coordinate>=-g.preDistance));assert.throws(()=>beam({...g,postDistance:-.01}));
});
test('at geometric focus the ray contribution vanishes but finite diffraction remains',()=>{
 const a=beam({...g,waistRadius:120e-6});close(a.geometricRadius,0);close(a.output.width,a.diffractionRadius);close(a.output.width,g.wavelength*g.focalLength!/(Math.PI*(120e-6)));assert.ok(a.output.width>0);const b=beam({...g,waistRadius:120e-6,wavelength:1e-6});close(b.output.width,2*a.output.width);
});
test('width-square decomposes into geometric and diffraction contributions over all sampled planes',()=>{
 for(const f of[-.04,.02,null]){const a=beam({...g,preDistance:.03,postDistance:.08,focalLength:f});a.longitudinal.forEach(p=>close(p.width**2,p.geometricalSignedRadius**2+p.diffractionRadius**2));close(a.output.width**2,a.geometricRadius**2+a.diffractionRadius**2);}
});
test('two-dimensional power normalization and conditional encircled fraction are retained under focusing',()=>{
 for(const v of refs.beams){const a=beam(v.system);close(a.onAxisIntensity*Math.PI*a.output.width**2/2,a.intensityIntegral,3e-9);if(v.system.amplitude>0)close(a.fractionWithinWidth!,1-Math.exp(-2));else assert.equal(a.fractionWithinWidth,null);}
});
test('zero amplitude retains reference-mode geometry but has zero field and no observed phase or conditional fraction',()=>{
 const a=beam({...g,amplitude:0}),b=beam(g);close(a.output.width,b.output.width);assert.equal(a.onAxisPhase,null);assert.equal(a.intensityIntegral,0);assert.equal(a.onAxisIntensity,0);assert.equal(a.fractionWithinWidth,null);a.radialProfile.forEach(p=>{vector(p.field,[0,0]);assert.equal(p.phase,null);});
});
test('second-moment covariance transports with the ray matrix and retains diffraction emittance',()=>{
 for(const v of refs.beams){const a=beam(v.system),s=a.covariance;close(s.xx*s.thetaTheta-s.xTheta*s.xTheta,a.emittance**2,3e-8);close(4*s.xx,a.output.width**2);}
});
test('strong covariance cancellation is reported without projecting raw moments or claiming a directly computed invariant',()=>{
 let exposed=false;
 for(const wavelength of[200e-9,2e-6])for(const waistRadius of[5e-6,.002])for(const preDistance of[0,2])for(const postDistance of[0,2])for(const focalLength of[-2,-.005,.005,2,null]){
  const a=beam({...g,wavelength,waistRadius,preDistance,postDistance,focalLength}),v=a.covariance,d=a.covarianceDeterminantEvaluation;
  assert.equal(d.direct,v.xx*v.thetaTheta-v.xTheta*v.xTheta);close(d.expected,a.emittance**2);
  assert.ok(d.cancellationRatio>=1-1e-12);
  a.covarianceFactor.forEach((row,i)=>{const value=row[0]**2+row[1]**2;close(value,i===0?v.xx:v.thetaTheta);});
  if(Math.abs(d.relativeResidual)>.001){exposed=true;assert.ok(d.cancellationRatio>1e12);}
 }
 assert.ok(exposed,'extreme allowed preparations must expose rather than hide lost determinant precision');
});
test('nonparaxial combinations remain explicit diagnostics rather than falsely certified exact Maxwell solutions',()=>{
 const a=beam({...g,wavelength:2e-6,waistRadius:5e-6,preDistance:2,postDistance:2,focalLength:.005});assert.ok(a.divergence>.1||a.lensPhaseSlope>.1);assert.ok(Number.isFinite(a.output.width));assert.ok(a.radialProfile.every(p=>p.field.every(Number.isFinite)));const b=rays({...r,objectDistance:2,focalLength:.005,fanHalfAngle:.1});assert.ok(b.maxSlope>.1);
});
test('invalid physical domains and nonfinite numbers throw instead of clamping into a different experiment',()=>{
 for(const bad of[{wavelength:0},{waistRadius:0},{amplitude:-1},{postDistance:-1},{preDistance:Infinity},{focalLength:0},{focalLength:.0049},{wavelength:NaN}])assert.throws(()=>beam({...g,...bad}));
 for(const bad of[{objectDistance:0},{screenDistance:-1},{objectHeight:.01},{centralAngle:NaN},{fanHalfAngle:-.1},{focalLength:Infinity}])assert.throws(()=>rays({...r,...bad}));
 assert.throws(()=>beam(g).sampleField(-1));
});
for(const id of ['paraxial-ray-matrices-thin-lens-imaging','gaussian-beam-focusing-ray-wave-limit'])for(const lang of ['ko','en'])test(`substantive ${lang} manuscript ${id} renders strict KaTeX with worked reasoning and exercises`,async()=>{
 const source=fs.readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8');assert.ok(source.length>(lang==='en'?10000:6500));assert.ok((source.match(/^## /gm)||[]).length>=14);assert.ok((source.match(/^### /gm)||[]).length>=6);assert.ok(source.includes('https://'));const file=await unified().use(remarkParse).use(remarkMath).use(remarkRehype).use(rehypeKatex,{strict:'error'}).use(rehypeStringify).process(source);assert.equal(file.messages.length,0);const html=String(file);assert.ok(html.includes('katex'));assert.ok(!html.includes('katex-error'));const {getLearningContent}=await import('../src/lib/learning-content');const rendered=await getLearningContent(id);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);assert.ok(rendered[lang as 'ko'|'en'].includes('katex-display'));assert.ok(!rendered[lang as 'ko'|'en'].includes('katex-error'));if(lang==='en')assert.doesNotMatch(source,/[가-힣]/);
});
