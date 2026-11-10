import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/wave-equation-independent.json';
import {periodicWaveChain as chain,wavePulsePoint as point,wavePulseEnergyBalance as balance,compactWaveBump as bump,type ChainMedium,type WaveHarmonic,type PulseMedium} from '../src/lib/learning-wave-equation';
const close=(a:number,b:number,tol=2e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol*(1+Math.abs(b)),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=2e-9)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const medium:ChainMedium={density:1,tension:1,length:8,particles:16};
const harmonics:WaveHarmonic[]=[{mode:0,positionCos:.01,positionSin:0,velocityCos:.02,velocitySin:0},{mode:1,positionCos:.05,positionSin:-.02,velocityCos:.01,velocitySin:.03},{mode:3,positionCos:-.02,positionSin:.01,velocityCos:.02,velocitySin:0}];
const pulse:PulseMedium={density:1,tension:1,amplitude:.1,velocityAmplitude:0,width:1,center:2,direction:'left',boundary:'whole-line'};
function finite(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(Array.isArray(v))v.forEach(finite);else if(v&&typeof v==='object')Object.values(v).forEach(finite);}
test('24 independent circulant matrix exponentials, ODE and continuum FFT solutions match120 chain states',()=>{
 assert.equal(refs.chains.length,24);
 for(const row of refs.chains){assert.ok(row.odeExpmDifference<2e-9);for(const expected of row.states){const a=chain(row.medium,row.harmonics,expected.time);vector(a.position,expected.position);vector(a.velocity,expected.velocity);vector(a.force,expected.force);vector(a.assignedEnergy,expected.assignedEnergy);vector(a.linkFlux,expected.linkFlux);close(a.energy,expected.energy);close(a.momentum,expected.momentum);vector(a.continuum.map(p=>p.displacement),expected.continuumPosition);vector(a.continuum.map(p=>p.velocity),expected.continuumVelocity);vector(a.continuum.map(p=>p.gradient),expected.continuumGradient);close(a.continuumEnergy,expected.continuumEnergy);close(a.rmsDisplacementError,expected.rmsDisplacementError);}}
});
test('27 independent polynomial image sources match648 displacement/velocity/energy/flux points',()=>{
 assert.equal(refs.pulses.length,27);assert.equal(refs.provenance.symbolicPde,true);assert.equal(refs.provenance.symbolicEnergyConservation,true);
 for(const row of refs.pulses)for(const e of row.states){const a=point(row.medium as PulseMedium,e.x,e.time);for(const key of ['displacement','velocity','gradient','acceleration','energyDensity','flux']as const)close(a[key],e[key],2e-10);}
});
test('108 separately integrated space-energy and time-flux balances agree with real QUADPACK integrals',()=>{
 for(const row of refs.pulses)for(const e of row.balances){const a=balance(row.medium as PulseMedium,[e.window[0],e.window[1]],e.time);for(const key of ['energy0','energyNow','integratedInflow','integratedOutflow','totalEnergy0','totalEnergyNow']as const)close(a[key],e[key],2e-10);close(a.residual,0,2e-11);assert.ok(Math.abs(e.independentResidual)<1e-9);}
});
test('fixed material refinement changes particle mass and spring stiffness while preserving wave speed',()=>{
 const a=chain(medium,harmonics,0),b=chain({...medium,particles:32},harmonics,0);close(a.mass,2*b.mass);close(b.stiffness,2*a.stiffness);close(a.c,b.c);close(a.mass*a.x.length,medium.density*medium.length);
});
test('chain superposition reconstructs independent initial position and velocity coefficients',()=>{
 const a=chain(medium,harmonics,0);
 for(let i=0;i<a.x.length;i++){const x=a.x[i];const expected=(kind:'position'|'velocity')=>harmonics.reduce((sum,h)=>sum+h[`${kind}Cos`]*Math.cos(2*Math.PI*h.mode*x/medium.length)+h[`${kind}Sin`]*Math.sin(2*Math.PI*h.mode*x/medium.length),0);close(a.position[i],expected('position'));close(a.velocity[i],expected('velocity'));}
});
test('the zero mode is affine drift with kinetic energy and conserved transverse momentum',()=>{
 const h=[{mode:0,positionCos:.1,positionSin:0,velocityCos:.05,velocitySin:0}],a=chain(medium,h,12);a.position.forEach(q=>close(q,.7));a.velocity.forEach(v=>close(v,.05));a.force.forEach(f=>close(f,0));close(a.energy,.01);close(a.momentum,.4);assert.equal(a.spectrum[0].frequencyRatio,null);
});
test('chain physical energy equals resolved modal energy and stays constant',()=>{
 const initial=chain(medium,harmonics,0);
 for(const t of [0,.3,5,64]){const a=chain(medium,harmonics,t);close(a.energy,initial.energy,1e-12);close(a.energy,a.modalEnergy.reduce((s,v)=>s+v,0),1e-12);close(a.continuumEnergy,initial.continuumEnergy,1e-12);close(a.momentum,initial.momentum,1e-12);}
});
test('signed link flux satisfies each actual particle assigned-energy derivative',()=>{
 const a=chain(medium,harmonics,1.7),dt=1e-5,left=chain(medium,harmonics,1.7-dt),right=chain(medium,harmonics,1.7+dt);
 a.assignedEnergyRate.forEach((rate,i)=>{close(rate,a.linkFlux[(i+15)%16]-a.linkFlux[i],1e-12);close(rate,(right.assignedEnergy[i]-left.assignedEnergy[i])/(2*dt),1e-10);});
});
test('the discrete force accelerates the exact spectral trajectory',()=>{
 const t=1.3,dt=2e-5,a=chain(medium,harmonics,t),l=chain(medium,harmonics,t-dt),r=chain(medium,harmonics,t+dt);a.force.forEach((f,i)=>close(f/a.mass,(r.velocity[i]-l.velocity[i])/(2*dt),1e-9));
});
test('continuum samples obey the PDE and physical characteristic flux identity',()=>{
 for(const a of chain(medium,harmonics,.8).continuum){close(a.acceleration,a.curvature,1e-13);close(a.energyDensity,(a.velocity*a.velocity+a.gradient*a.gradient)/2,1e-13);close(a.flux,-a.velocity*a.gradient,1e-13);}
});
test('long-wave refinement converges quadratically for the same physical initial field and time',()=>{
 const h=harmonics.filter(h=>h.mode<2);const errors=[16,32,64,128].map(particles=>chain({...medium,particles},h,2).rmsDisplacementError);
 errors.slice(1).forEach((e,i)=>assert.ok(errors[i]/e>3.85&&errors[i]/e<4.15));
 const ratios=[16,32,64].map(particles=>chain({...medium,particles},h,2).spectrum[1].frequencyRatio!);assert.ok(ratios[0]<ratios[1]&&ratios[1]<ratios[2]&&ratios[2]<1);
});
test('resolved near-Nyquist motion remains dispersive instead of being mislabeled a continuum mode',()=>{
 const a=chain({...medium,particles:8},[{mode:3,positionCos:.05,positionSin:0,velocityCos:0,velocitySin:0}],2);assert.ok(a.spectrum[0].frequencyRatio!<.8);assert.ok(a.rmsDisplacementError>.01);
});
test('compact profile has C2 support boundaries and the independently known total integral',()=>{
 for(const x of [-1,1,-2,2]){const a=bump(x);assert.equal(a.value,0);assert.equal(a.first,0);assert.equal(a.second,0);}close(bump(1).integral,32/35,1e-15);assert.equal(bump(-1).integral,0);close(bump(0).value,1);close(bump(0).second,-6);
});
test('dAlembert solution recovers independent initial displacement and velocity',()=>{
 const s={...pulse,velocityAmplitude:.2};for(const x of [.3,1.2,1.75,2,2.6,4]){const b=bump((x-s.center)/s.width),a=point(s,x,0);close(a.displacement,s.amplitude*b.value);close(a.velocity,s.amplitude*b.first/s.width+.2*b.value);}
});
test('left and right traveling data produce translated pulses with opposite signed energy flux',()=>{
 for(const direction of ['left','right']as const){const sign=direction==='left'?1:-1,s={...pulse,direction},t=.8,x=2.4-sign*t,a=point(s,x,t),b=bump(.4);close(a.displacement,.1*b.value);close(a.velocity,sign*.1*b.first);close(a.gradient,.1*b.first);close(a.flux,-sign*a.energyDensity);assert.ok(a.energyDensity>0);}
});
test('zero initial velocity splits a displacement pulse into two half-height pulses',()=>{
 const s={...pulse,direction:'split' as const};close(point(s,4,2).displacement,.05);close(point(s,0,2).displacement,.05);close(point(s,2,2).displacement,0);
});
test('velocity-only compact data leave a displacement plateau without local energy',()=>{
 const s={...pulse,amplitude:0,velocityAmplitude:.2,direction:'split' as const},a=point(s,2,3);close(a.displacement,.2*32/35/2);close(a.velocity,0);close(a.gradient,0);close(a.energyDensity,0);assert.ok(balance(s,[-2,6],3).totalEnergyNow>0);
});
test('finite propagation leaves points outside the initial dependence cone exactly undisturbed',()=>{
 const a=point({...pulse,velocityAmplitude:.2},20,1);for(const k of ['displacement','velocity','gradient','energyDensity','flux']as const)close(a[k],0,0);vector(a.initialDependencyInterval,[19,21]);
});
test('fixed and free ends preserve their own compatibility condition and have zero boundary power',()=>{
 for(const boundary of ['fixed','free']as const)for(const direction of ['left','right','split']as const)for(const t of [0,.3,1.5,2,2.4,4]){
  const a=point({...pulse,boundary,direction,velocityAmplitude:.2},0,t);close(boundary==='fixed'?a.displacement:a.gradient,0,1e-13);close(a.flux,0,1e-13);
 }
});
test('after reflection a fixed end inverts displacement while a free end preserves its sign',()=>{
 for(const boundary of ['fixed','free']as const){const a=point({...pulse,boundary},2,4);close(a.displacement,boundary==='fixed'?-.1:.1,1e-13);close(point({...pulse,boundary},2.4,4).flux,point({...pulse,boundary},2.4,4).energyDensity,1e-13);}
});
test('characteristic energy densities sum to physical energy and yield signed flux',()=>{
 for(const t of [.1,.8,2.2])for(const x of [.5,1.7,2.3,4]){const a=point({...pulse,velocityAmplitude:.17,boundary:'free'},x,t);close(a.energyDensity,a.rightEnergyDensity+a.leftEnergyDensity,1e-13);close(a.flux,a.rightEnergyDensity-a.leftEnergyDensity,1e-13);close(a.displacement,a.rightDisplacement+a.leftDisplacement,1e-13);}
});
test('local wave acceleration, spatial curvature and independent finite differences satisfy the PDE',()=>{
 const s={...pulse,velocityAmplitude:.2,direction:'split' as const},x=2.2,t=.3,h=2e-5,a=point(s,x,t);close(a.acceleration,a.curvature,1e-13);close((point(s,x,t+h).velocity-point(s,x,t-h).velocity)/(2*h),a.acceleration,1e-8);close((point(s,x+h,t).gradient-point(s,x-h,t).gradient)/(2*h),a.curvature,1e-8);
});
test('energy continuity holds from independent time and space differences',()=>{
 const x=2.2,t=.3,h=2e-5,s={...pulse,velocityAmplitude:.2};close((point(s,x,t+h).energyDensity-point(s,x,t-h).energyDensity)/(2*h)+(point(s,x+h,t).flux-point(s,x-h,t).flux)/(2*h),0,2e-8);
});
test('a finite window loses energy while whole-line energy stays constant',()=>{
 const s={...pulse,direction:'right' as const},a=balance(s,[0,4],4);assert.ok(a.energy0>0);close(a.energyNow,0,1e-13);close(a.integratedInflow,0,1e-13);close(a.integratedOutflow,a.energy0,1e-12);close(a.totalEnergyNow,a.totalEnergy0,1e-12);close(a.residual,0,1e-12);
});
test('zero duration, zero motion and negative amplitude retain physical meanings and finite output',()=>{
 const z={...pulse,amplitude:0};const a=balance(z,[-1,5],0);finite(a);assert.equal(a.totalEnergyNow,0);assert.equal(a.netBoundaryWork,0);const b=balance(pulse,[-1,5],0);assert.equal(b.integratedInflow,0);close(b.energyNow,b.energy0);close(balance({...pulse,amplitude:-.1},[-1,5],2).totalEnergyNow,b.totalEnergy0,1e-12);
});
test('the slope indicator is the actual initial profile maximum bound and does not assert global linearity',()=>{
 const a=balance(pulse,[-1,5],0),expected=Math.abs(point(pulse,2+1/Math.sqrt(5),0).gradient);close(a.initialSlopeBound,expected,1e-13);
});
test('unsupported medium values, unresolved/Nyquist/duplicate/sparse modes and invalid windows are rejected',()=>{
 for(const key of ['density','tension']as const)for(const value of [0,NaN,Infinity,-1])assert.throws(()=>chain({...medium,[key]:value},harmonics,0));
 for(const particles of [0,7,12,NaN])assert.throws(()=>chain({...medium,particles},harmonics,0));
 for(const h of [[],new Array(1),[harmonics[1],harmonics[1]],[{...harmonics[1],mode:8}],[{...harmonics[0],positionSin:1}],[{...harmonics[1],velocityCos:NaN}]])assert.throws(()=>chain(medium,h,0));
 for(const t of [-1,65,NaN])assert.throws(()=>point(pulse,2,t));assert.throws(()=>point({...pulse,boundary:'fixed',center:.5},0,1));assert.throws(()=>point({...pulse,boundary:'free'},-1,1));
 for(const w of [[1,1],[2,1],[-1,3],[NaN,3],new Array(2)])assert.throws(()=>balance({...pulse,boundary:'fixed'},w as [number,number],1));assert.throws(()=>bump(Infinity));
});
for(const id of ['coupled-chain-continuum-wave-limit','wave-initial-data-reflection-energy-flux'])for(const lang of ['ko','en']as const)test(`${id}.${lang}: substantive bilingual theory and six explained exercises render strict math`,async()=>{
 const{readFile}=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
 const raw=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),rendered=await getLearningContent(id);
 assert.ok(raw.length>6500);assert.ok(rendered.outline[lang].length>=16);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);
 assert.equal(new Set(rendered.outline[lang].map(h=>h.id)).size,rendered.outline[lang].length);assert.equal(rendered.outline[lang].length,(raw.match(/^## /gm)||[]).length);
 assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));
 assert.equal((raw.match(/^### (?:Exercise|문제) [1-6]/gm)||[]).length,6);assert.equal((raw.match(lang==='en'?/\*\*Solution:\*\*/g:/\*\*해설:\*\*/g)||[]).length,6);
 assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);
});
