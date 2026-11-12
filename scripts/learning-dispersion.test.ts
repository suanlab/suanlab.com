import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/dispersion-independent.json';
import {gaussianDispersivePacket as gaussian,latticeDispersivePacket as lattice,periodicPacketCenter,
  type GaussianPacket,type LatticePacket} from '../src/lib/learning-dispersion';
const close=(a:number,b:number,tol=3e-10)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=3e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const g:GaussianPacket={amplitude:.7,sigma:1,center:.3,carrierWaveNumber:3.2,carrierFrequency:5.3,groupVelocity:1.1,dispersion:.9,phase:.2};
const s:LatticePacket={particles:32,length:8,density:.8,tension:1.7,amplitude:.08,modeCenter:4,bandwidth:2,center:2.96,phase:.4};
test('36 independently Fourier-integrated Gaussian systems reproduce180 complex fields and three derivative integrals',()=>{
 assert.equal(refs.gaussians.length,36);assert.equal(refs.provenance.productImports,false);assert.equal(refs.provenance.symbolicEnvelopePdeResidual,'0');
 for(const r of refs.gaussians){for(const p of r.states){const a=gaussian(r.system,p.x,r.time);for(const key of['field','envelopeDerivativeX','envelopeDerivativeXX','envelopeDerivativeT']as const)vector(a[key],p[key]);}
 const a=gaussian(r.system,0,r.time);close(a.integratedNorm,r.norm);if(r.intensityVariance!==null)close(a.intensityVariance,r.intensityVariance);}
});
test('36 real chain states agree with an independently exponentiated physical equation and FFT analytic reconstruction',()=>{
 assert.equal(refs.chains.length,36);
 for(const r of refs.chains){const a=lattice(r.system as LatticePacket,r.time);vector(a.position,r.position);vector(a.velocity,r.velocity);vector(a.assignedEnergy,r.assignedEnergy);vector(a.linkFlux,r.linkFlux);a.points.forEach((p,i)=>vector(p.analyticField,r.analyticField[i]));close(a.energy,r.energy);close(a.modalEnergy,r.modalEnergy);close(a.analyticNorm,r.analyticNorm);
 for(const key of['energyTransportVelocity','energyWeightedGroupVelocity']as const)if(r[key]===null)assert.equal(a[key],null);else close(a[key]!,r[key]!);
 for(const key of['analyticCenter','energyCenter']as const){close(a[key].resultant,r[key].resultant);if(r[key].position===null)assert.equal(a[key].position,null);else {const delta=Math.abs(a[key].position!-r[key].position!);close(Math.min(delta,r.system.length-delta)*a[key].resultant,0);}}}
});
test('zero time recovers the prescribed amplitude Gaussian and full carrier phase',()=>{
 for(const x of[-2,.3,1.7]){const a=gaussian(g,x,0),e=g.amplitude*Math.exp(-((x-g.center)**2)/(2*g.sigma**2)),p=g.carrierWaveNumber*x+g.phase;vector(a.field,[e*Math.cos(p),e*Math.sin(p)]);close(a.amplitudeWidth,g.sigma);close(a.intensityStandardDeviation,g.sigma/Math.sqrt(2));}
});
test('linear dispersion translates the envelope without spreading while carrier phase remains independent',()=>{
 const system={...g,dispersion:0};for(const t of[-2,0,1.5]){const a=gaussian(system,system.center+system.groupVelocity*t+.4,t),b=gaussian(system,system.center+.4,0);close(a.envelope,b.envelope);close(a.amplitudeWidth,system.sigma);close(a.carrierPhaseVelocity!,g.carrierFrequency/g.carrierWaveNumber);close(a.groupVelocity,g.groupVelocity);}
});
test('quadratic dispersion preserves norm and grows intensity variance with the correct amplitude-width convention',()=>{
 const a=gaussian(g,0,0),b=gaussian(g,0,2);close(a.integratedNorm,b.integratedNorm);close(b.intensityVariance,g.sigma**2*(1+(g.dispersion*2/g.sigma**2)**2)/2);assert.ok(b.amplitudeWidth>a.amplitudeWidth);close(gaussian(g,g.center+g.groupVelocity*2,2).intensity,g.amplitude**2/Math.hypot(1,g.dispersion*2/g.sigma**2));
});
test('negative dispersion reverses the envelope chirp without changing the intensity profile',()=>{
 const a=gaussian(g,1.8,1),b=gaussian({...g,dispersion:-g.dispersion},1.8,1);close(a.intensity,b.intensity);close(a.chirp,-b.chirp);assert.notEqual(a.phase,b.phase);
});
test('the envelope satisfies the advective dispersive PDE at independent positions and times',()=>{
 for(const x of[-1,.7,2])for(const t of[-1,0,.8]){const a=gaussian(g,x,t),dx=a.envelopeDerivativeX,dxx=a.envelopeDerivativeXX,dt=a.envelopeDerivativeT;close(-(dt[1]+g.groupVelocity*dx[1])+g.dispersion*dxx[0]/2,0);close(dt[0]+g.groupVelocity*dx[0]+g.dispersion*dxx[1]/2,0);}
});
test('reported full local phase gradients agree with finite differences and differ from a carrier-only crest speed',()=>{
 const x=1.8,t=.7,h=1e-5,a=gaussian(g,x,t);close(a.localWaveNumber!,(gaussian(g,x+h,t).phase-gaussian(g,x-h,t).phase)/(2*h),2e-8);close(a.localAngularFrequency!,-(gaussian(g,x,t+h).phase-gaussian(g,x,t-h).phase)/(2*h),2e-8);close(a.localCrestVelocity!,a.localAngularFrequency!/a.localWaveNumber!);assert.ok(Math.abs(a.localCrestVelocity!-a.carrierPhaseVelocity!)>.01);
});
test('a zero carrier wave number has undefined carrier phase speed,not infinite speed',()=>{
 const a=gaussian({...g,carrierWaveNumber:0},g.center,0);assert.equal(a.carrierPhaseVelocity,null);assert.equal(a.localCrestVelocity,null);close(a.groupVelocity,g.groupVelocity);
});
test('zero Gaussian field has zero norm and no observable intensity center or local phase',()=>{
 const a=gaussian({...g,amplitude:0},.4,.7);vector(a.field,[0,0]);assert.equal(a.integratedNorm,0);assert.equal(a.intensityCenter,null);assert.equal(a.observedPhase,null);assert.equal(a.localWaveNumber,null);assert.equal(a.localCrestVelocity,null);
});
test('packet spectra are positive resolved modes and preserve the prescribed peak-amplitude bound',()=>{
 const a=lattice(s,0);assert.equal(a.modes.length,15);close(a.modes.reduce((v,m)=>v+m.amplitude,0),s.amplitude);assert.ok(a.modes.every(m=>m.mode>0&&m.mode<s.particles/2));assert.ok(a.position.every(q=>Math.abs(q)<=s.amplitude+1e-14));
});
test('the declared all-phase slope bound bounds actual link slopes without certifying linear material behavior',()=>{
 for(const t of[0,.7,4]){const a=lattice(s,t);for(let i=0;i<s.particles;i++)assert.ok(Math.abs(a.position[(i+1)%s.particles]-a.position[i])/a.spacing<=a.initialSlopeBound+1e-12);}
 const a=lattice({...s,amplitude:.2,modeCenter:14,bandwidth:1},0);assert.ok(a.initialSlopeBound>1);assert.ok(a.energy>0);
});
test('actual spring forces agree with the modal acceleration of every real mass',()=>{
 const a=lattice(s,1.3);for(let i=0;i<s.particles;i++)close(a.force[i],a.mass*a.points[i].acceleration);close(a.mass,s.density*s.length/s.particles);close(a.stiffness,s.tension*s.particles/s.length);
});
test('local energy derivatives equal incoming minus outgoing link power and their sum vanishes',()=>{
 const a=lattice(s,.8),h=1e-6,b=lattice(s,.8+h),c=lattice(s,.8-h);
 a.assignedEnergyRate.forEach((rate,i)=>{close(rate,a.linkFlux[(i+s.particles-1)%s.particles]-a.linkFlux[i]);close(rate,(b.assignedEnergy[i]-c.assignedEnergy[i])/(2*h),2e-8);});close(a.assignedEnergyRate.reduce((v,x)=>v+x,0),0);
});
test('global energy and analytic norm are separately conserved over time with their own dimensions',()=>{
 const a=lattice(s,0);for(const t of[.7,4,16]){const b=lattice(s,t);close(a.energy,b.energy);close(b.energy,b.modalEnergy);close(a.analyticNorm,b.analyticNorm);close(b.analyticNorm,b.expectedAnalyticNorm);}assert.ok(Math.abs(a.energy-a.analyticNorm)>.001);
});
test('transport from real physical link power equals the energy-weighted group speed,not a centroid derivative',()=>{
 for(const bandwidth of[.2,2,5])for(const t of[0,1.2,4]){const a=lattice({...s,bandwidth},t);close(a.energyTransportVelocity!,a.energyWeightedGroupVelocity!);assert.ok(a.energyTransportVelocity!>=0&&a.energyTransportVelocity!<=a.c);}
});
test('a narrow spectrum approaches carrier group speed but a wide spectrum can differ substantially',()=>{
 const a=lattice({...s,modeCenter:10,bandwidth:.2},1),b=lattice({...s,modeCenter:10,bandwidth:5},1);close(a.energyTransportVelocity!,a.carrierGroupVelocity,1e-8);assert.ok(Math.abs(b.energyTransportVelocity!-b.carrierGroupVelocity)>.01);assert.ok(a.carrierPhaseVelocity>a.carrierGroupVelocity);
});
test('doubling amplitude quadruples physical energy and analytic norm without changing transport speed',()=>{
 const a=lattice({...s,amplitude:.04},1),b=lattice({...s,amplitude:.08},1);close(b.energy,4*a.energy);close(b.analyticNorm,4*a.analyticNorm);close(a.energyTransportVelocity!,b.energyTransportVelocity!);
});
test('fixed peak normalization does not pretend bandwidth changes conserve initial energy',()=>{
 const a=lattice({...s,bandwidth:.2},0),b=lattice({...s,bandwidth:5},0);assert.ok(Math.abs(a.energy-b.energy)>.001);close(a.modes.reduce((v,m)=>v+m.amplitude,0),b.modes.reduce((v,m)=>v+m.amplitude,0));
});
test('a whole-site initial translation rotates the real mass data and preserves total energy',()=>{
 const shift=3,a=lattice(s,1.2),b=lattice({...s,center:s.center+shift*s.length/s.particles},1.2);b.position.forEach((q,i)=>close(q,a.position[(i-shift+s.particles)%s.particles]));close(a.energy,b.energy);close(a.energyTransportVelocity!,b.energyTransportVelocity!);
});
test('periodic endpoints specify the same initial packet,not an absorbing or reflecting boundary',()=>{
 const a=lattice({...s,center:0},.7),b=lattice({...s,center:s.length},.7);vector(a.position,b.position);vector(a.velocity,b.velocity);
});
test('circular centers cross the boundary correctly and return no arbitrary angle for uniform or absent weights',()=>{
 const x=[0,2,4,6];assert.equal(periodicPacketCenter([1,1,1,1],x,8).position,null);assert.equal(periodicPacketCenter([0,0,0,0],x,8).position,null);const a=periodicPacketCenter([1,0,0,1],x,8);close(a.position!,7);assert.ok(a.resultant>.7);
});
test('an empty lattice field has neither energy transport nor an observable circular center',()=>{
 const a=lattice({...s,amplitude:0},1);assert.equal(a.energy,0);assert.equal(a.energyTransportVelocity,null);assert.equal(a.energyWeightedGroupVelocity,null);assert.equal(a.analyticCenter.position,null);assert.equal(a.energyCenter.position,null);
});
test('unsupported nonfinite,negative,aliased or ill-sized physical inputs are rejected',()=>{
 for(const patch of[{sigma:0},{amplitude:-1},{dispersion:NaN},{carrierWaveNumber:Infinity},{center:9}])assert.throws(()=>gaussian({...g,...patch},0,0));
 for(const patch of[{particles:16},{modeCenter:16},{bandwidth:0},{center:9},{density:0},{amplitude:.3},{phase:NaN}])assert.throws(()=>lattice({...s,...patch}as LatticePacket,0));
 assert.throws(()=>gaussian(g,129,0));assert.throws(()=>gaussian(g,0,5));assert.throws(()=>lattice(s,-1));assert.throws(()=>periodicPacketCenter([1],[0,1],8));assert.throws(()=>periodicPacketCenter([-1],[0],8));assert.throws(()=>periodicPacketCenter(new Array(2),[0,1],8));
});
for(const id of['dispersive-wave-packets-phase-group-velocity','lattice-wave-packet-transport-dispersion'])for(const lang of['ko','en']as const)test(`${id}.${lang}: detailed theory,six explained exercises and strict KaTeX rendering`,async()=>{
 const{readFile}=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');const raw=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),rendered=await getLearningContent(id);
 assert.ok(raw.length>6500);assert.equal(rendered.outline[lang].length,17);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);assert.equal(new Set(rendered.outline[lang].map(h=>h.id)).size,17);assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));
 const sections=raw.match(/^## /gm)||[],research=raw.match(/^## (?:Research extension:|연구 확장:).*\n[\s\S]*?(?=^## )/gm)||[];assert.equal(research.length,1);assert.equal(sections.length,17);assert.equal((raw.replace(research[0],'').match(/^## /gm)||[]).length,16);
 assert.equal((raw.match(/^### (?:Exercise|문제) [1-6]/gm)||[]).length,6);assert.equal((raw.match(lang==='en'?/\*\*Solution:\*\*/g:/\*\*해설:\*\*/g)||[]).length,6);assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);
});

test('small nonzero phase gradients retain finite crest ratios and exact zeros remain distinct',()=>{
 for(const k of[-1e-13,1e-13,1e-12-1e-25,1e-12,1e-12+1e-25]){const a=gaussian({...g,carrierWaveNumber:k,carrierFrequency:8,dispersion:0},g.center,0);assert.equal(a.localCrestVelocityStatus,'finite');close(a.localCrestVelocity!,8/k,2e-15);assert.equal(a.carrierPhaseVelocityStatus,'finite');close(a.carrierPhaseVelocity!,8/k,2e-15);}
 const zero=gaussian({...g,carrierWaveNumber:0},g.center,0);assert.equal(zero.localCrestVelocityStatus,'zero-phase-gradient');assert.equal(zero.carrierPhaseVelocityStatus,'zero-carrier-wave-number');assert.equal(gaussian({...g,amplitude:0},g.center,0).localCrestVelocityStatus,'zero-field');
});
test('unrepresentable phase-speed ratios are explicit unavailable values instead of infinite observations',()=>{
 const a=gaussian({...g,carrierWaveNumber:Number.MIN_VALUE,carrierFrequency:8,dispersion:0},g.center,0);assert.equal(a.localWaveNumber,Number.MIN_VALUE);assert.equal(a.localCrestVelocity,null);assert.equal(a.localCrestVelocityStatus,'unrepresentable-ratio');assert.equal(a.carrierPhaseVelocity,null);assert.equal(a.carrierPhaseVelocityStatus,'unrepresentable-ratio');assert.ok(Object.values(a).filter(v=>typeof v==='number').every(v=>Number.isFinite(v)));
});
