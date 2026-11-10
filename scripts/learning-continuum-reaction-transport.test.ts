import test from 'node:test';
import assert from 'node:assert/strict';
import {reversibleReactionState as kinetics,reversibleReactionAt as reaction,reversibleReactionWindow as reactionWindow,fisherFrontAt as fisher,fisherLevelPosition as levelPosition,fisherFrontWindow as frontWindow,type ReversibleTransport,type FisherFrontParameters,type ReactionValue}from '../src/lib/learning-continuum-reaction-transport';
import refs from './fixtures/continuum-reaction-transport-independent.json';
const p:ReversibleTransport={mass:2,area:1.5,initialSigma:.5,initialMean:0,velocity:.4,diffusivity:.1,forwardRate:1,reverseRate:.5,initialFractionA:.8};
const f:FisherFrontParameters={diffusivity:1/6,growthRate:1,carryingConcentration:4,bulkVelocity:0,initialFront:0,area:2};
function near(a:ReactionValue,b:number,tol=3e-10){assert.notEqual(a,null);assert.ok(Math.abs(a!-b)<=tol*Math.max(1,Math.abs(b)),`${a} vs ${b}`);}
function expected(a:ReactionValue,text:string){if(text==='None'){assert.equal(a,null);return;}const x=Number(text);if(x===0&&text!=='0.0')assert.equal(a,null);else if(x===0)assert.equal(a,0);else if(Math.abs(x)<1e-300){assert.notEqual(a,null);assert.ok(Math.abs(a!-x)<=Number.MIN_VALUE);}else{assert.notEqual(a,null);assert.ok(Math.abs(a!/x-1)<3e-11,`${a} vs ${x}`);}}
function finiteTree(v:unknown):void{if(typeof v==='number')assert.ok(Number.isFinite(v));else if(v&&typeof v==='object')Object.values(v).forEach(finiteTree);}
test('24 independently differentiated reversible species match concentrations, time, flux and sources',()=>{
 assert.equal(refs.reactionCases.length,24);for(const r of refs.reactionCases){const q=reaction(r.parameters,r.x,r.time);near(q.state.fractionA,r.expected.fractionA);
  for(const[j,s]of[q.a,q.b].entries()){const e=r.expected.species[j];near(s.concentration,e.concentration);near(s.spatialDerivative,e.dx);near(s.timeDerivative,e.dt);near(s.flux,e.flux);near(s.fluxGradient,e.fluxGradient);near(s.source,e.source);near(s.balanceResidual,0);}
 }
});
test('24 independent physical-window integrals verify species storage, flux, source and total conservation',()=>{
 for(const r of refs.reactionCases){const q=reactionWindow(r.parameters,r.left,r.right,r.time);for(const[j,s]of[q.a,q.b].entries()){const e=r.expected.species[j].window;for(const key of ['massInside','massRate','netOutflow','integratedSource']as const)near(s[key],e[key]);near(s.balanceResidual,0);}
  near(q.a.massInside!+q.b.massInside!,q.total.massInside!);near(q.a.integratedSource!+q.b.integratedSource!,0);near(q.a.massRate!+q.b.massRate!,q.total.massRate!);
 }
});
test('actual species finite differences verify time dependence and the source signs',()=>{
 const x=.3,t=.7,h=1e-5,q=reaction(p,x,t),lo=reaction(p,x,t-h),hi=reaction(p,x,t+h),xm=reaction(p,x-h,t),xp=reaction(p,x+h,t);
 for(const key of ['a','b']as const){near((hi[key].concentration!-lo[key].concentration!)/(2*h),q[key].timeDerivative!,2e-8);near((xp[key].flux!-xm[key].flux!)/(2*h),q[key].fluxGradient!,2e-8);near(q[key].timeDerivative!+q[key].fluxGradient!-q[key].source!,0);}
});
test('global species masses change while the finite total mass remains conserved',()=>{
 const q=reaction(p,.2,.7),initial=reaction(p,.2,0);assert.ok(q.globalMassA!<initial.globalMassA!);assert.ok(q.globalMassB!>initial.globalMassB!);near(q.globalMassA!+q.globalMassB!,p.mass);near(q.globalSourceA!+q.globalSourceB!,0);assert.equal(q.totalSource,0);
});
test('zero rates leave all declared fractions stationary with no uniquely preferred equilibrium',()=>{
 for(const initialFractionA of [0,.3,.5,1]){const q=kinetics({...p,forwardRate:0,reverseRate:0,initialFractionA},3);near(q.fractionA,initialFractionA);near(q.fractionB,1-initialFractionA);assert.equal(q.equilibriumA,null);assert.equal(q.equilibriumStatus,'nonunique-all-fractions-stationary');assert.equal(q.fractionRateA,0);assert.equal(q.stableFractionChangeA,0);}
});
test('stationary equal-rate composition can have nonzero forward and reverse turnover',()=>{
 const q=reaction({...p,forwardRate:1,reverseRate:1,initialFractionA:.5},0,1);near(q.a.source,0);near(q.state.fractionA,.5);assert.ok(q.forwardTurnover!>0);near(q.forwardTurnover,q.reverseTurnover!);near(q.state.stableFractionChangeA,0);
});
test('seven stable kinetic references distinguish tiny relaxation, transient underflow and true stationarity',()=>{
 assert.equal(refs.kineticCases.length,7);for(const r of refs.kineticCases){const q=kinetics(r.parameters,r.time);expected(q.fractionA,r.expected.fractionA);expected(q.fractionB,r.expected.fractionB);expected(q.fractionRateA,r.expected.fractionRateA);expected(q.stableFractionChangeA,r.expected.stableChangeA);
  if(r.expected.logRateMagnitude!=='None')near(q.logFractionRateMagnitude,Number(r.expected.logRateMagnitude));
 }
 const tiny=kinetics({forwardRate:1e-20,reverseRate:0,initialFractionA:1},1);assert.equal(tiny.fractionA,1);assert.equal(tiny.roundedComplementB,0);assert.ok(tiny.fractionB!>0);assert.notEqual(tiny.stableFractionChangeA,0);
 const fast=kinetics({forwardRate:128,reverseRate:128,initialFractionA:1},4);near(fast.fractionA,.5);assert.equal(fast.fractionRateA,null);assert.equal(fast.fractionRateSign,-1);assert.ok(fast.logFractionRateMagnitude!<-1000);
});
test('equilibrium fractions can underflow while turnover combined with rate remains representable',()=>{
 const q=reaction({...p,mass:1,area:1,diffusivity:0,velocity:0,forwardRate:128,reverseRate:Number.MIN_VALUE,initialFractionA:0},0,4);
 assert.equal(q.state.equilibriumA,null);assert.ok(q.state.logEquilibriumA!<-745);assert.equal(q.forwardTurnover,Number.MIN_VALUE);assert.equal(q.reverseTurnover,Number.MIN_VALUE);assert.equal(q.a.source,null);assert.equal(q.roundedTurnoverDifference,0);
});
test('omitting reaction on the same field exposes the source instead of altering the carrier',()=>{
 const q=reaction(p,.3,.7),w=reactionWindow(p,-.8,.9,.7);near(q.a.omittedReactionResidual,q.a.source!);near(q.b.omittedReactionResidual,q.b.source!);assert.ok(Math.abs(q.a.source!)>.01);near(w.a.omittedReactionResidual,w.a.integratedSource!);near(w.b.omittedReactionResidual,w.b.integratedSource!);
});
test('zero total mass preserves declared kinetics but leaves mass-normalized composition unavailable',()=>{
 const q=reaction({...p,mass:0},.3,1),w=reactionWindow({...p,mass:0},-.8,.9,1);assert.equal(q.a.concentration,0);assert.equal(q.b.concentration,0);assert.equal(q.a.source,0);assert.equal(q.massNormalizedFractionA,null);assert.ok(q.state.fractionA!>0);assert.equal(w.a.massInside,0);assert.equal(w.b.integratedSource,0);
});
test('species concentration logs survive far-carrier underflow without claiming zero material',()=>{
 const q=reaction({...p,initialSigma:.125,diffusivity:0,velocity:0},15,0);assert.equal(q.a.concentration,null);assert.equal(q.b.concentration,null);assert.ok(q.a.logConcentration!<-7000);assert.ok(q.b.logConcentration!<-7000);assert.equal(q.roundedSourceSum,null);assert.equal(q.totalSource,0);
});
test('24 independent Fisher fields verify actual derivatives, flux and nonlinear growth',()=>{
 assert.equal(refs.frontCases.length,24);for(const r of refs.frontCases){const q=fisher(r.parameters,r.x,r.time),e=r.expected;near(q.concentration,e.concentration);near(q.spatialDerivative,e.dx);near(q.secondDerivative,e.dxx);near(q.timeDerivative,e.dt);near(q.flux,e.flux);near(q.source,e.source);near(q.balanceResidual,0);}
});
test('24 independent Fisher window budgets balance storage plus boundary flux against actual growth integral',()=>{
 for(const r of refs.frontCases){const q=frontWindow(r.parameters,r.left,r.right,r.time);for(const key of ['massInside','massRate','netOutflow','integratedSource']as const)near(q[key],r.expected.window[key]);near(q.balanceResidual,0);near(q.omittedSourceResidual,q.integratedSource!);}
});
test('worked Fisher quarter level has time rate5/6, diffusion1/12 and source3/4',()=>{
 const q=fisher(f,0,0);near(q.lambda,1);near(q.relativeSpeed,5/6);near(q.concentration,1);near(q.spatialDerivative,-1);near(q.secondDerivative,.5);near(q.timeDerivative,5/6);near(f.diffusivity*q.secondDerivative!,1/12);near(q.source,.75);near(q.omittedSourceResidual,.75);
});
test('relative concentration levels follow a common wave speed but the reference point is not the half level',()=>{
 for(const level of [.01,.25,.5,.9,.99]){const x=levelPosition(f,level,.6),q=fisher(f,x,.6);near(q.concentration!/f.carryingConcentration,level);near(levelPosition(f,level,.9)-x,q.laboratorySpeed*.3);}
 near(levelPosition(f,.25,.6),fisher(f,0,.6).center);assert.ok(levelPosition(f,.5,.6)<levelPosition(f,.25,.6));
});
test('laboratory motion includes advection and can be stationary while relative front propagation stays positive',()=>{
 const q=fisher(f,.3,.7),speed=q.relativeSpeed,stationary=fisher({...f,bulkVelocity:-speed},.3,2);assert.equal(stationary.laboratorySpeed,0);assert.equal(stationary.timeDerivative,0);assert.ok(stationary.relativeSpeed>0);assert.ok(stationary.source!>0);near(stationary.balanceResidual,0);
 const advected=fisher({...f,bulkVelocity:.4},.3+.4*.7,.7);near(advected.concentration,q.concentration!);near(advected.source,q.source!);near(advected.flux,q.flux!+.4*q.concentration!);
});
test('actual Fisher concentration and window-mass differences verify the PDE and finite budget',()=>{
 const x=.3,t=.7,dt=1e-5,dx=2e-4,q=fisher(f,x,t),cm=fisher(f,x-dx,t).concentration!,cp=fisher(f,x+dx,t).concentration!;
 near((fisher(f,x,t+dt).concentration!-fisher(f,x,t-dt).concentration!)/(2*dt),q.timeDerivative!,2e-8);
 near((cp-cm)/(2*dx),q.spatialDerivative!,2e-8);near(q.timeDerivative!-f.diffusivity*(cp-2*q.concentration!+cm)/(dx*dx)-q.source!,0,3e-8);
 const a=frontWindow(f,-1,1,t-dt),b=frontWindow(f,-1,1,t+dt),mid=frontWindow(f,-1,1,t);near((b.massInside!-a.massInside!)/(2*dt),mid.massRate!,2e-8);
});
test('seven rational front-tail references preserve concentration, deficits, derivatives and growth logs',()=>{
 assert.equal(refs.frontTailCases.length,7);for(const r of refs.frontTailCases){const q=fisher(r.parameters,r.x,0),e=r.expected;expected(q.concentration,e.concentration);expected(q.stableDeficit,e.deficit);expected(q.source,e.source);expected(q.spatialDerivative,e.dx);near(q.logConcentration,Number(e.logConcentration),2e-12);near(q.logDeficit,Number(e.logDeficit),2e-12);near(q.logSource,Number(e.logSource),2e-12);}
});
test('rounded plateau growth can vanish while the stable positive deficit and source remain nonzero',()=>{
 const r=refs.frontTailCases.find(r=>r.requestedPhase===-40)!,q=fisher(r.parameters,r.x,0);assert.equal(q.concentration,r.parameters.carryingConcentration);assert.equal(q.roundedDeficit,0);assert.equal(q.roundedLogisticSource,0);assert.ok(q.stableDeficit!>0);assert.ok(q.source!>0);
 const rescued=refs.frontTailCases.find(r=>r.requestedPhase===373.1)!,s=fisher(rescued.parameters,rescued.x,0);assert.equal(s.concentration,null);assert.ok(s.source!>0);
});
test('finite Fisher windows retain positive log observations in both tails without a false finite global mass',()=>{
 const p=refs.frontTailCases[0].parameters,large=frontWindow(p,-12,-11.9,0),small=frontWindow(p,12,12.1,0);assert.ok(large.massInside!>0);assert.equal(large.integratedSource,null);assert.ok(Number.isFinite(large.logIntegratedSource));assert.equal(small.massInside,null);assert.ok(Number.isFinite(small.logMassInside));assert.equal(fisher(f,0,0).globalMassStatus,'diverges-on-entire-line');
});
test('small front displacement survives rounded center equality and sub-range nonzero steps remain missing',()=>{
 const q=fisher({...f,initialFront:1},0,1e-20);assert.equal(q.center,1);assert.ok(q.stableDisplacement!>0);
 const tiny=fisher({...f,diffusivity:.0001,growthRate:.0001},0,Number.MIN_VALUE);assert.equal(tiny.stableDisplacement,null);assert.equal(fisher(f,0,0).stableDisplacement,0);
});
test('invalid domains are rejected and supported extreme observations never leak NaN or infinity',()=>{
 for(const bad of [NaN,Infinity,-1,129])assert.throws(()=>kinetics({...p,forwardRate:bad},0),RangeError);assert.throws(()=>fisher({...f,diffusivity:0},0,0),RangeError);assert.throws(()=>fisher({...f,growthRate:0},0,0),RangeError);assert.throws(()=>frontWindow(f,1,1,0),RangeError);assert.throws(()=>reactionWindow(p,1,-1,0),RangeError);assert.throws(()=>levelPosition(f,0,0),RangeError);
 for(const mass of [0,Number.MIN_VALUE,1e-200,4])for(const forwardRate of [0,Number.MIN_VALUE,128])finiteTree(reaction({...p,mass,forwardRate},16,4));
 for(const diffusivity of [.0001,4])for(const growthRate of [.0001,4])for(const x of [-16,16])finiteTree(fisher({...f,diffusivity,growthRate},x,4));
});
