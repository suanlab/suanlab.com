import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {harmonicKinds,evaluateHarmonic as evaluate,discreteFourier,radixTwoFFT,linearConvolution,circularConvolution,squarePartial,transformROC,sampledAlias,haarTransform,inverseHaar,localSignal,firResponse,deconvolutionExperiment} from '../src/lib/learning-harmonic';
import {harmonicLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
import type {Complex} from '../src/lib/learning-complex';
import {remark} from 'remark';
import remarkMath from 'remark-math';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const real=(x:number[]):Complex[]=>x.map(v=>[v,0]);
const sumsq=(x:number[])=>x.reduce((s,a)=>s+a*a,0);
function integrate(f:(x:number)=>number,lo:number,hi:number,n=12000){let s=f(lo)+f(hi);for(let j=1;j<n;j++)s+=(j%2?4:2)*f(lo+(hi-lo)*j/n);return s*(hi-lo)/(3*n);}
test('square-wave Fourier coefficients follow independent integrals and jump midpoints remain zero',()=>{
 for(let k=1;k<=23;k++)close(integrate(t=>Math.sin(k*t),0,Math.PI)*2/Math.PI,k%2?4/(Math.PI*k):0,1e-10);
 for(const terms of [1,4,12]){close(squarePartial(terms,0),0);close(squarePartial(terms,Math.PI),0);close(squarePartial(terms,-.4),-squarePartial(terms,.4));}close(squarePartial(1,Math.PI/2),4/Math.PI);
});
test('Gaussian transform real and imaginary integrals match the independent analytic expression',()=>{
 for(const sigma of [.3,1,2])for(const w of [0,.7,2]){const f=(t:number)=>Math.exp(-t*t/(2*sigma*sigma));close(integrate(t=>f(t)*Math.cos(w*t),-12*sigma,12*sigma),sigma*Math.sqrt(2*Math.PI)*Math.exp(-sigma*sigma*w*w/2),1e-10);close(integrate(t=>f(t)*Math.sin(w*t),-12*sigma,12*sigma),0);}
});
test('linear and circular convolution have independently worked boundary outputs',()=>{
 for(const r of [-1,0,.5,1]){assert.deepEqual(linearConvolution([1,0,0,2],[1,r]),[1,r,0,2,2*r]);assert.deepEqual(circularConvolution([1,0,0,2],[1,r,0,0]),[1+2*r,r,0,2]);for(const mode of [0,1])assert.ok((evaluate('convolution-boundary-theorem',[r,mode,0]).stats['spectral reconstruction maximum error'] as number)<1e-12);}
 assert.equal(evaluate('convolution-boundary-theorem',[.5,1,4]).stats['selected output'],null);
});
test('DFT impulse, constant and complex-exponential coefficients match closed forms',()=>{
 for(const N of [4,8,16,64]){const impulse=radixTwoFFT(real(Array.from({length:N},(_,n)=>Number(n===0))));impulse.forEach(z=>{close(z[0],1);close(z[1],0);});const tone:Complex[]=Array.from({length:N},(_,n)=>[Math.cos(2*Math.PI*3*n/N),Math.sin(2*Math.PI*3*n/N)]);radixTwoFFT(tone).forEach((z,k)=>{close(z[0],k===3?N:0,1e-10);close(z[1],0);});discreteFourier(real(Array(N).fill(1))).forEach((z,k)=>{close(z[0],k===0?N:0,1e-10);close(z[1],0);});}
});
test('recursive FFT agrees with an independent trigonometric sum and roundtrips complex samples',()=>{
 for(const N of [1,2,4,8,16,32,64,128]){const x:Complex[]=Array.from({length:N},(_,n)=>[Math.sin(n*.31),Math.cos(n*.27)]),fast=radixTwoFFT(x);fast.forEach((z,k)=>{let re=0,im=0;x.forEach(([a,b],n)=>{const theta=-2*Math.PI*k*n/N;re+=a*Math.cos(theta)-b*Math.sin(theta);im+=a*Math.sin(theta)+b*Math.cos(theta);});close(z[0],re,2e-11);close(z[1],im,2e-11);});radixTwoFFT(fast,true).forEach((z,n)=>{close(z[0],x[n][0]);close(z[1],x[n][1]);});}
});
test('Laplace ROC values match causal and anti-causal improper integrals',()=>{
 const causal=transformROC(0,0,1,0,.5),anti=transformROC(0,1,1,-2,.5);for(const [r,side,re] of [[causal,0,0],[anti,1,-2]] as const){assert.equal(r.inside,true);const f=(t:number)=>(side===0?1:-1)*Math.exp(-(1+re)*t);close(r.transform![0],integrate(t=>f(t)*Math.cos(.5*t),side===0?0:-40,side===0?40:0),1e-9);close(r.transform![1],integrate(t=>-f(t)*Math.sin(.5*t),side===0?0:-40,side===0?40:0),1e-9);}
});
test('Laplace continuation outside ROC and algebraic poles are never reported as transforms',()=>{
 const r=transformROC(0,1,1,0,0);assert.equal(r.inside,false);close(r.rational![0],1);close(r.rational![1],0);assert.equal(r.transform,null);const pole=transformROC(0,0,1,-1,0);assert.equal(pole.pole,true);assert.equal(pole.rational,null);assert.equal(pole.transform,null);
});
test('bilateral Z values match independently truncated right and negative left geometric series',()=>{
 const right=transformROC(1,0,.5,1,0),left=transformROC(1,1,.5,.25,0);let r=0,l=0;for(let n=0;n<80;n++){r+=.5**n;l-=.5**(n+1);}close(right.transform![0],r);close(left.transform![0],l);transformROC(1,1,.5,0,0).transform!.forEach(x=>close(x,0));assert.equal(transformROC(1,0,.5,.5,0).transform,null);assert.equal(transformROC(1,0,.5,0,0).transform,null);
});
test('signed aliases preserve samples across phases and the Nyquist sine disappears exactly',()=>{
 for(const fs of [2,8,16])for(const f of [0,1,4,7,9,20])for(const phase of [0,1,2,3]){const r=sampledAlias(f,fs,phase);assert.ok(r.alias>=-fs/2&&r.alias<fs/2);r.times.forEach(t=>close(r.signal(t),r.reconstruction(t)));}
 assert.equal(sampledAlias(7,8,0).alias,-1);assert.ok(sampledAlias(4,8,1).samples.every(x=>x===0));assert.equal(evaluate('sampling-aliasing',[4,8,1]).stats['1{tone strictly below Nyquist}'],0);
});
test('Haar known pairs, all basis-vector roundtrips and orthonormality preserve energy',()=>{
 const a=haarTransform([1,1]),b=haarTransform([1,-1]);close(a[0],Math.SQRT2);close(a[1],0);close(b[0],0);close(b[1],Math.SQRT2);
 for(const N of [2,4,8,16,32,64]){const vectors=Array.from({length:N},(_,j)=>haarTransform(Array.from({length:N},(_,n)=>Number(n===j))));for(let j=0;j<N;j++){inverseHaar(vectors[j]).forEach((v,n)=>close(v,Number(n===j)));for(let k=0;k<N;k++)close(vectors[j].reduce((s,v,n)=>s+v*vectors[k][n],0),Number(j===k));}const x=Array.from({length:N},(_,n)=>Math.cos(n*.7));close(sumsq(haarTransform(x)),sumsq(x));}
});
test('local windows identify early and late tones and preserve windowed Fourier energy',()=>{
 for(const center of [0,8,16,24,31])for(const power of [2,3,4]){const r=localSignal(center,power);close(r.spectrum.reduce((s,z)=>s+z[0]**2+z[1]**2,0)/32,sumsq(r.windowed));close(sumsq(r.haar),sumsq(r.block));assert.equal(r.padded,r.block.filter((_,j)=>r.start+j<0||r.start+j>=32).length);}
 assert.equal(evaluate('stft-haar-localization',[0,8,4,2]).stats['dominant nonnegative STFT bin'],1);assert.equal(evaluate('stft-haar-localization',[0,24,4,6]).stats['dominant nonnegative STFT bin'],6);assert.equal(evaluate('stft-haar-localization',[1,8,3,31]).stats['selected coefficient magnitude'],null);
});
test('FIR complex response matches a direct steady-state sum at every time index',()=>{
 for(const r of [0,.25,.5,1])for(const w of [0,Math.PI/3,Math.PI/2,Math.PI]){const [re,im]=firResponse(r,w);for(let n=-2;n<=16;n++)close((1-r)*Math.cos(w*n)+r*Math.cos(w*(n-1)),re*Math.cos(w*n)-im*Math.sin(w*n));}
 close(Math.hypot(...firResponse(.5,Math.PI/2)),Math.SQRT1_2);assert.deepEqual(firResponse(.5,Math.PI),[0,0]);assert.equal(evaluate('fir-frequency-response',[.5,Math.PI,0]).stats['phase response at query (rad)'],null);
});
test('Parseval holds for fractional cycles, DC shifts and both normalization conventions',()=>{
 for(const Npower of [2,3,5])for(const cycles of [0,1,1.25,8])for(const dc of [-1,0,1])for(const mode of [0,1]){const out=evaluate('parseval-normalization',[Npower,cycles,dc,mode]);close(out.stats['time-domain total energy'] as number,out.stats['frequency energy after correct normalization'] as number,1e-10);}
 close(evaluate('parseval-normalization',[3,1,0,0]).stats['displayed-spectrum raw energy'] as number,32);close(evaluate('parseval-normalization',[3,1,1,0]).stats['time-domain total energy'] as number,12);
});
test('Gaussian energy widths match independently integrated second moments and Hz units',()=>{
 for(const sigma of [.25,.5,1,2]){const density=(t:number,s:number)=>Math.exp(-t*t/(2*s*s))/(s*Math.sqrt(2*Math.PI)),dw=1/(2*sigma);for(const width of [sigma,dw]){close(integrate(t=>density(t,width),-10*width,10*width),1);close(integrate(t=>t*t*density(t,width),-10*width,10*width),width*width,1e-9);}const r=evaluate('gaussian-time-frequency-localization',[sigma,3,1]);close(r.stats['time-frequency uncertainty product'] as number,.5);close((r.stats['time standard deviation'] as number)*(r.stats['frequency standard deviation in Hz'] as number),1/(4*Math.PI));}
});
test('invertible noiseless blur is recovered and its exact nullspace is not identifiable',()=>{
 for(const r of [0,.25,.49,.75,1]){const out=deconvolutionExperiment(r,0,0,0);out.estimate!.forEach((x,n)=>close(x,out.truth[n],1e-12));close(out.mse!,0);}
 const nullspace=circularConvolution(Array.from({length:8},(_,n)=>(-1)**n),[.5,.5,0,0,0,0,0,0]);assert.ok(nullspace.every(x=>x===0));const direct=deconvolutionExperiment(.5,0,0,0);assert.equal(direct.estimate,null);assert.equal(direct.mse,null);const pseudo=deconvolutionExperiment(.5,.05,0,1);close(pseudo.mse!,.25);close(pseudo.residualMSE!,.05**2);
});
test('near-zero deconvolution amplifies a known alternating noise to an independently derived MSE',()=>{
 const r=deconvolutionExperiment(.49,.05,0,0);r.estimate!.forEach((x,n)=>close(x-r.truth[n],2.5*(-1)**n,1e-12));close(r.mse!,6.25);close(r.residualMSE!,0);
});
test('Tikhonov estimates satisfy the time-domain stationarity equation independently of FFT',()=>{
 for(const r of [.25,.5,.75])for(const lambda of [.01,.1,1]){const out=deconvolutionExperiment(r,.05,lambda,1),z=out.estimate!,residual=circularConvolution(z,out.h).map((v,n)=>v-out.observed[n]);for(let j=0;j<8;j++){const adjoint=(1-r)*residual[j]+r*residual[(j+1)%8];close(adjoint+lambda*z[j],0);}}
 close(deconvolutionExperiment(.5,.05,.1,1).mse!,.2554989525838,1e-12);
});
test('signal models reject nonfinite, out-of-range and fractional integer controls',()=>{
 for(const kind of harmonicKinds){const spec=harmonicLabs[kind],values=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));for(let j=0;j<values.length;j++){const bad=[...values];bad[j]=spec.controls[j].max+1;assert.throws(()=>evaluate(kind,bad));if(spec.controls[j].step!=='any'){bad[j]=spec.controls[j].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
 for(const f of [()=>radixTwoFFT(real([1,2,3])),()=>discreteFourier([[NaN,0]]),()=>haarTransform([1,2,3]),()=>inverseHaar([]),()=>linearConvolution([],[1]),()=>circularConvolution([1],[1,2]),()=>transformROC(0,0,0,0,0)])assert.throws(f);
});
for(const kind of harmonicKinds)test(`${kind}: presets expose finite geometry and all guide targets exist`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);
 for(const g of learningGuides[kind].experiments)assert.ok(harmonicLabs[kind].presets[g.preset]);
 for(const p of harmonicLabs[kind].presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[]),...(r.holes??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}assert.ok(!/[가-힣]/.test(r.statement??''));for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of harmonicKinds)for(const lang of ['ko','en'])test(`${kind}.${lang}: explanatory source includes math, exercises and primary further reading`,async()=>{
 const text=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8');const tree=remark().use(remarkMath).parse(text);assert.ok(tree.children.filter(n=>n.type==='heading').length>=4);assert.ok(tree.children.some(n=>n.type==='math'));const rendered=await getLearningContent(kind);assert.ok(rendered[lang as 'ko'|'en'].length>4000);assert.ok(!rendered[lang as 'ko'|'en'].includes('katex-error'));assert.ok(text.includes('https://'));assert.ok(text.includes(lang==='ko'?'직접 풀기':'Exercise'));if(lang==='en'){assert.ok(!/[가-힣]/.test(text));assert.ok(text.split(/\s+/).length>=350);}
});
