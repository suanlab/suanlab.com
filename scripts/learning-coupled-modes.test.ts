import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/coupled-modes-independent.json';
import {coupledModeSpectrum as spectrum,coupledModeState as state,massWeightedCoupledNormalModes as normal,coupledBeatingState as beat,coupledModeBeatingEnergyTransfer as beating,type CoupledInitialState} from '../src/lib/learning-coupled-modes';
const system={mass1:1,mass2:2,stiffness1:1,stiffness2:1,coupling:1};
const initial:CoupledInitialState={position:[1,0],velocity:[0,0]};
const equal={mass:1,stiffness:1,coupling:1/16,amplitude:1};
const close=(a:number,b:number,tolerance=3e-9)=>assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tolerance*(1+Math.abs(b)),`${a} != ${b}`);
const vector=(a:number[],b:number[],tolerance=3e-9)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tolerance));};
const matrix=(a:number[][],b:number[][],tolerance=3e-9)=>a.forEach((v,i)=>vector(v,b[i],tolerance));
function finite(value:unknown):void{if(typeof value==='number')assert.ok(Number.isFinite(value));else if(Array.isArray(value))value.forEach(finite);else if(value&&typeof value==='object')Object.values(value).forEach(finite);}
const multiply=(a:number[][],b:number[][])=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));
test('120 independent generalized spectra, projectors and 1080 physical initial-value samples agree',()=>{
 assert.equal(refs.normal.length,120);
 for(const row of refs.normal){const init:CoupledInitialState={position:[row.initial.position[0],row.initial.position[1]],velocity:[row.initial.velocity[0],row.initial.velocity[1]]},s=spectrum(row.system);
  assert.equal(s.rank,row.rank);assert.equal(s.degenerate,row.degenerate);vector(s.squaredFrequencies,row.squaredFrequencies,2e-12);
  if(row.projectors)row.projectors.forEach((p,i)=>matrix(s.projectors[i],p,2e-11));
  assert.ok(row.refinementDifference<2e-7);assert.ok(row.independentExpmDifference<2e-9);
  for(const expected of row.states){const actual=state(row.system,init,expected.time);vector(actual.position,expected.position);vector(actual.velocity,expected.velocity);if(row.degenerate)close(actual.modalEnergySum,expected.modalEnergies[0]+expected.modalEnergies[1]);else vector(actual.modalEnergies,expected.modalEnergies);vector(actual.assignedEnergy,expected.assignedEnergy);close(actual.energy,expected.energy);close(actual.couplingEnergy,expected.couplingEnergy);close(actual.flux12,expected.flux12);}
 }
});
test('72 independent beating systems and 648 physical states, real flux integrals and approximation trajectories agree',()=>{
 assert.equal(refs.beating.length,72);assert.equal(refs.provenance.symbolicFluxIdentities,true);
 for(const row of refs.beating){assert.ok(row.refinementDifference<2e-6);for(const expected of row.states){const actual=beat(row.system,expected.time);vector(actual.position,expected.position);vector(actual.velocity,expected.velocity);vector(actual.bareEnergy,expected.bareEnergy);vector(actual.assignedEnergy,expected.assignedEnergy);close(actual.couplingEnergy,expected.couplingEnergy);close(actual.flux12,expected.flux12);close(actual.netTransfer,expected.netTransfer);close(actual.transferResidual,0,1e-10);if(expected.approximation){assert.ok(actual.approximation);vector(actual.approximation.position,expected.approximation.position,1e-10);vector(actual.approximation.velocity,expected.approximation.velocity,1e-10);}else assert.equal(actual.approximation,null);}
  if(row.firstEnvelopeSwap){const r=beating(row.system,0,0).firstEnvelopeSwap!;close(r.time,row.firstEnvelopeSwap.time,1e-11);vector(r.state.position,row.firstEnvelopeSwap.position,2e-10);vector(r.state.velocity,row.firstEnvelopeSwap.velocity,2e-10);}
 }
});
test('unequal masses require mass orthogonality rather than Euclidean mode orthogonality',()=>{
 const s=spectrum(system);vector(s.squaredFrequencies,[(3-Math.sqrt(3))/2,(3+Math.sqrt(3))/2],1e-13);matrix(s.massGram,[[1,0],[0,1]],1e-13);
 assert.ok(Math.abs(s.modes[0][0]*s.modes[1][0]+s.modes[0][1]*s.modes[1][1])>.1);
 for(let j=0;j<2;j++)close(s.modes[j][1]/s.modes[j][0],2-s.squaredFrequencies[j],1e-13);
});
test('physical mode projectors resolve identity, are idempotent and self-adjoint in the mass metric',()=>{
 const s=spectrum(system),[a,b]=s.projectors;matrix(a.map((r,i)=>r.map((v,j)=>v+b[i][j])),[[1,0],[0,1]],1e-13);
 for(const p of [a,b]){matrix(multiply(p,p),p,1e-13);const mp=multiply(s.mass,p);close(mp[0][1],mp[1][0],1e-13);}
 matrix(multiply(a,b),[[0,0],[0,0]],1e-13);assert.ok(Math.abs(a[0][1]-a[1][0])>.1);
});
test('modal position and velocity contributions reconstruct arbitrary initial data at t=0',()=>{
 const init:CoupledInitialState={position:[1,-.5],velocity:[.25,-.75]},r=state(system,init,0);vector(r.position,init.position,1e-13);vector(r.velocity,init.velocity,1e-13);close(r.energy,r.modalEnergySum,1e-13);
});
test('a pure selected mode retains its fixed shape and its other mode has zero energy',()=>{
 const s=spectrum(system),v=s.modes[0],r=state(system,{position:v,velocity:[0,0]},1.25);
 vector(r.position,v.map(x=>x*Math.cos(s.frequencies[0]*1.25)),1e-13);close(r.modalEnergies[1],0,1e-13);close(r.modalEnergies[0],s.squaredFrequencies[0]/2,1e-13);
});
test('a zero-frequency translation includes initial drift and does not divide by frequency',()=>{
 const s={...system,stiffness1:0,stiffness2:0},init:CoupledInitialState={position:[1,1],velocity:[.5,.5]},r=state(s,init,12);
 assert.equal(spectrum(s).zeroModeCount,1);assert.equal(spectrum(s).periods[0],null);vector(r.position,[7,7],1e-13);vector(r.velocity,[.5,.5],1e-13);close(r.energy,.375,1e-13);
});
test('two unconnected free coordinates preserve both independent affine motions',()=>{
 const s={...system,stiffness1:0,stiffness2:0,coupling:0},init:CoupledInitialState={position:[1,-1],velocity:[.5,-.25]},r=state(s,init,128);
 assert.equal(spectrum(s).rank,0);assert.equal(spectrum(s).zeroModeCount,2);assert.equal(spectrum(s).degenerate,true);vector(r.position,[65,-33]);vector(r.velocity,init.velocity);finite(r);
});
test('a repeated eigenvalue has many valid mass-normalized bases but one physical initial-value solution',()=>{
 const s={...system,stiffness1:1,stiffness2:2,coupling:0},init:CoupledInitialState={position:[1,-.5],velocity:[.25,.75]},t=.7,r=state(s,init,t),angle=.37,c=Math.cos(angle),z=Math.sin(angle),basis=[[c,z/Math.sqrt(2)],[-z,c/Math.sqrt(2)]];
 assert.equal(spectrum(s).degenerate,true);
 const reconstructed=[0,0];for(const v of basis){const a=v[0]*init.position[0]+2*v[1]*init.position[1],b=v[0]*init.velocity[0]+2*v[1]*init.velocity[1],coordinate=a*Math.cos(t)+b*Math.sin(t);for(let i=0;i<2;i++)reconstructed[i]+=v[i]*coordinate;}
 vector(r.position,reconstructed,1e-13);
});
test('mode sign reversal preserves reconstructed physical position and modal energy',()=>{
 const s=spectrum(system),r=state(system,initial,2),reconstructed=[0,0];for(let j=0;j<2;j++)for(let i=0;i<2;i++)reconstructed[i]+=(-s.modes[j][i])*(-r.coordinates[j]);vector(reconstructed,r.position,1e-13);
});
test('a structurally repeated rational eigenvalue stays repeated despite determinant-quotient roundoff',()=>{
 const s=spectrum({mass1:.75,mass2:1.5,stiffness1:.25,stiffness2:.5,coupling:0});
 assert.equal(s.degenerate,true);assert.equal(s.squaredFrequencies[0],s.squaredFrequencies[1]);close(s.squaredFrequencies[0],1/3,1e-15);
});
test('the actual trajectory derivative satisfies both coupled Newton equations',()=>{
 const h=1e-5,t=.8,a=state(system,initial,t-h),b=state(system,initial,t),c=state(system,initial,t+h);
 vector(c.position.map((v,i)=>(v-a.position[i])/(2*h)),b.velocity,1e-9);
 vector(c.velocity.map((v,i)=>(v-a.velocity[i])/(2*h)),b.acceleration,1e-9);
 close(b.force[0]+b.force[1],-system.stiffness1*b.position[0]-system.stiffness2*b.position[1],1e-13);
});
test('half-coupling local energies obey independently differentiated signed flux balances',()=>{
 const h=1e-5,t=.7,a=state(system,initial,t-h),b=state(system,initial,t),c=state(system,initial,t+h);
 close((c.assignedEnergy[0]-a.assignedEnergy[0])/(2*h),-b.flux12,1e-9);close((c.assignedEnergy[1]-a.assignedEnergy[1])/(2*h),b.flux12,1e-9);
 close(b.assignedEnergyRates[0],-b.flux12,1e-13);close(b.assignedEnergyRates[1],b.flux12,1e-13);
});
test('coupling energy is a necessary physical contribution, not a plotting correction',()=>{
 const r=state(system,initial,0);close(r.bareEnergy[0],.5,1e-13);close(r.bareEnergy[1],0,1e-13);close(r.couplingEnergy,.5,1e-13);close(r.energy,1,1e-13);vector(r.assignedEnergy,[.75,.25],1e-13);
});
test('an envelope swap is not generally an exact energy swap',()=>{
 const r=beating(equal,64,0),swap=r.firstEnvelopeSwap!;assert.ok(swap.withinWindow);close(swap.time,51.7900388576685,1e-12);close(swap.state.position[0],0,1e-12);close(swap.state.velocity[0],.0302976668213452,1e-11);close(swap.state.bareEnergy[0],.000458974307408621,1e-11);assert.ok(swap.state.bareEnergy[0]>0);
});
test('a landmark beyond the observation window is computed and explicitly labeled without clamping',()=>{
 const r=beating({...equal,mass:4,stiffness:4},8,4);assert.ok(r.firstEnvelopeSwap!.time>128);assert.equal(r.firstEnvelopeSwap!.withinWindow,false);assert.equal(r.query.time,4);assert.equal(r.states.at(-1)!.time,8);
});
test('signed local flux can initially run opposite to the eventual visible envelope exchange',()=>{
 const r=beat(equal,.1);assert.ok(r.flux12<0);assert.ok(r.netTransfer<0);close(r.flux12,r.analyticFlux,1e-13);
});
test('integrated flux is differentiated independently of endpoint-energy identities',()=>{
 const h=1e-5,t=3,a=beat(equal,t-h),b=beat(equal,t),c=beat(equal,t+h);close((c.netTransfer-a.netTransfer)/(2*h),b.flux12,1e-10);close(b.transferResidual,0,1e-12);
});
test('modal energies remain constant while energies assigned to physical masses change',()=>{
 const s={mass1:1,mass2:1,stiffness1:1,stiffness2:1,coupling:1/16},a=state(s,initial,0),b=state(s,initial,20);vector(a.modalEnergies,b.modalEnergies,1e-13);assert.ok(Math.abs(a.assignedEnergy[0]-b.assignedEnergy[0])>.05);close(a.energy,b.energy,1e-13);
});
test('without anchors, center and relative motion differ from a weak-coupling beat approximation',()=>{
 const s={...equal,stiffness:0,coupling:1},r=beating(s,8,2);assert.equal(r.frequencies[0],0);assert.equal(r.query.approximation,null);assert.equal(r.weakCouplingRatio,null);close(r.query.flux12,0,1e-13);close(r.query.netTransfer,0,1e-13);vector(r.query.assignedEnergy,[.25,.25],1e-13);assert.equal(r.interpretation,'zero-mode-and-relative-motion');
});
test('zero coupling has no finite exchange period and both analytic models reduce to one uncoupled oscillator',()=>{
 const s={...equal,coupling:0},r=beating(s,8,2);assert.equal(r.spectralSplit,0);assert.equal(r.spectralEnvelopePeriod,null);assert.equal(r.firstEnvelopeSwap,null);vector(r.query.position,[Math.cos(2),0],1e-13);close(r.query.netTransfer,0,1e-13);close(r.query.positionApproximationError!,0,1e-13);
});
test('zero initial amplitude removes observed exchange and energy fractions, without removing the intrinsic spectrum',()=>{
 const r=beating({...equal,amplitude:0},64,12);assert.equal(r.energy0,0);assert.equal(r.assignedEnergyFractions,null);assert.equal(r.observedEnvelopePeriod,null);assert.equal(r.firstEnvelopeSwap,null);assert.ok(r.spectralEnvelopePeriod!>0);finite(r);
});
test('a zero-energy static zero mode is not a zero displacement or a normalized energy distribution',()=>{
 const r=normal({...system,stiffness1:0,stiffness2:0},{position:[1,1],velocity:[0,0]},8,4);assert.equal(r.energy0,0);assert.equal(r.modalEnergyFractions,null);vector(r.query.position,[1,1],1e-13);
});
test('weak-coupling approximation error is nonzero and depends on the observation horizon',()=>{
 const a=beating(equal,8,4),b=beating(equal,64,4);assert.ok(a.maximumPositionApproximationError!>0);assert.ok(b.maximumPositionApproximationError!>a.maximumPositionApproximationError!);vector(a.query.position,b.query.position,1e-13);
});
test('drawing intervals resolve the fastest phase and remain samples of the same analytic flow',()=>{
 const s={mass1:.25,mass2:.25,stiffness1:4,stiffness2:4,coupling:4},r=normal(s,initial,128,64),b=beating({mass:.25,stiffness:4,coupling:4,amplitude:4},128,64);
 for(const model of [r,b]){assert.ok(model.sampling.maximumPhaseAdvance<=.125+1e-14);assert.ok(model.sampling.intervals<=8192);assert.equal(model.states.length,model.sampling.intervals+1);finite(model);}
 assert.ok(r.maximumEnergyDrift<1e-10);assert.ok(b.maximumTransferResidual<1e-10);
});
test('zero duration gives an actual initial state and no invented advance',()=>{
 const r=normal(system,initial,0,0),b=beating(equal,0,0);assert.ok(r.states.every(s=>s.time===0));vector(r.query.position,initial.position,1e-13);close(b.query.netTransfer,0,1e-13);assert.equal(b.firstEnvelopeSwap!.withinWindow,false);
});
test('a commensurate frequency example can exchange bare energy exactly while retaining coupling energy',()=>{
 const r=beat({...equal,coupling:1.5},Math.PI);vector(r.position,[0,-1],1e-13);vector(r.velocity,[0,0],1e-13);vector(r.bareEnergy,[0,.5],1e-13);close(r.couplingEnergy,.75,1e-13);vector(r.assignedEnergy,[.375,.875],1e-13);close(r.netTransfer,.5,1e-13);
});
test('general mass-weighted evolution and the equal-mass envelope solution describe the same physical experiment',()=>{
 for(const mass of [.25,1,4])for(const stiffness of [0,1,4])for(const coupling of [0,1/16,4])for(const t of [0,.8,12]){
  const a=state({mass1:mass,mass2:mass,stiffness1:stiffness,stiffness2:stiffness,coupling},initial,t),b=beat({mass,stiffness,coupling,amplitude:1},t);
  vector(a.position,b.position,1e-11);vector(a.velocity,b.velocity,1e-11);close(a.flux12,b.flux12,1e-11);
 }
});
test('invalid physical parameters, unsupported grid points, initial data and query windows are rejected',()=>{
 for(const key of Object.keys(system))for(const value of [NaN,Infinity,-Infinity,undefined])assert.throws(()=>spectrum({...system,[key]:value}as typeof system));
 assert.throws(()=>spectrum({...system,mass1:0}));assert.throws(()=>spectrum({...system,coupling:-1}));assert.throws(()=>spectrum({...system,coupling:1/32}));assert.throws(()=>spectrum({...system,mass1:1.1}));
 for(const t of [-1,513,NaN])assert.throws(()=>state(system,initial,t));assert.throws(()=>normal(system,initial,1,2));assert.throws(()=>beating(equal,129,1));assert.throws(()=>beating({...equal,amplitude:-1},1,0));assert.throws(()=>state(system,{position:new Array(2)as [number,number],velocity:[0,0]},0));
});
for(const id of ['mass-weighted-coupled-normal-modes','coupled-mode-beating-energy-transfer'])for(const lang of ['ko','en']as const)test(`${id}.${lang}: detailed theory, six explained exercises and strict math render with a bilingual outline`,async()=>{
 const{readFile}=await import('node:fs/promises'),{getLearningContent}=await import('../src/lib/learning-content');
 const raw=await readFile(`content/learning/physics-math-for-ai/${id}.${lang}.md`,'utf8'),rendered=await getLearningContent(id);
 assert.ok(raw.length>6500);assert.ok(rendered.outline[lang].length>=16);assert.equal(rendered.outline.ko.length,rendered.outline.en.length);
 assert.equal(new Set(rendered.outline[lang].map(h=>h.id)).size,rendered.outline[lang].length);assert.equal(rendered.outline[lang].length,(raw.match(/^## /gm)||[]).length);
 assert.ok(rendered[lang].includes('katex-display'));assert.ok(!rendered[lang].includes('katex-error'));
 assert.equal((raw.match(/^### (?:Exercise|문제) [1-6]/gm)||[]).length,6);assert.equal((raw.match(lang==='en'?/\*\*Solution:\*\*/g:/\*\*해설:\*\*/g)||[]).length,6);
 assert.doesNotMatch(raw,/TODO|TBD|coming soon/i);if(lang==='en')assert.doesNotMatch(raw,/[가-힣]/);
});
