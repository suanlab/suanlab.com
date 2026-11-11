import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {initialBoundary,waitingSolutions,expIntegral,odeFlow,phaseFlow,pitchforkValue,bifurcationExperiment,principalSymbol,periodicMode,characteristics,riemannValue,riemannMass,boundaryFluxIntegral,riemannExperiment,finiteVolume,evaluateDifferential,differentialKinds} from '../src/lib/learning-differential';
import {differentialLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-9)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const integrate=(f:(x:number)=>number,a:number,b:number,N=4000)=>{let s=f(a)+f(b);for(let i=1;i<N;i++)s+=(i%2?4:2)*f(a+(b-a)*i/N);return s*(b-a)/(3*N);};
test('logistic superposition retains tiny signed nonlinear effects instead of subtracting rounded flows',()=>{
 for(const r of [-1e-20,1e-20,-1e-100,1e-100]){
  const q=odeFlow(r,.2,.3,1,2,0),leading=-2*.2*.3*r;
  assert.ok(q.defect!==0);assert.ok(Math.abs(q.defect/leading-1)<1e-14);
 }
 assert.equal(odeFlow(0,.2,.3,1,2,0).defect,0);
 for(const r of [-2,-.1,.1,2])close(odeFlow(r,.2,.3,1.7,2,0).defect,odeFlow(r,.2,.3,1.7,2,0).expandedDefect,1e-14);
});
test('structured phase isolation is independent of determinant underflow',()=>{
 for(const tiny of [Number.MIN_VALUE,1e-200]){
  assert.equal(phaseFlow(tiny,0,1,0,1,0).isolated,true);
  assert.equal(phaseFlow(0,tiny,1,0,1,0).classification,'center');
  assert.equal(phaseFlow(0,tiny,1,0,1,0).isolated,true);
  const diagonal=phaseFlow(-tiny,tiny,1,0,1,1);
  assert.equal(diagonal.isolated,true);assert.equal(diagonal.stable,true);
  assert.equal(phaseFlow(0,tiny,1,0,1,1).isolated,false);
 }
 assert.equal(evaluateDifferential('phase-portraits-fixed-points',[-1e-200,0,1,0,1,0]).stats['equilibrium class code'],0);
});
test('principal type uses the exact sign of stored coefficients through underflow and rounded cancellation',()=>{
 for(const tiny of [Number.MIN_VALUE,1e-200]){
  assert.equal(principalSymbol(tiny,0,tiny,0,0).type,'elliptic');
  assert.equal(principalSymbol(-tiny,0,-tiny,0,0).type,'elliptic');
  assert.equal(principalSymbol(tiny,0,-tiny,0,0).type,'hyperbolic');
  assert.equal(principalSymbol(0,tiny,0,0,0).type,'hyperbolic');
  assert.equal(principalSymbol(tiny,tiny,tiny,0,0).type,'parabolic rank one');
 }
 // Both products round to 1, but (1+2^-52)(1-2^-52)-1 is strictly negative.
 const a=1+2**-52,c=1-2**-52;
 assert.equal(a*c-1,0);assert.equal(principalSymbol(a,1,c,0,0).type,'hyperbolic');
 assert.equal(principalSymbol(0,-0,0,0,0).type,'zero principal part');
});
test('IVP derivatives and BVP boundary data follow separate equations and independent polynomial coefficients',()=>{
 const r=initialBoundary(1,1,2,0,0,.5);close(r.ivp,Math.exp(-.5));close(r.bvpQuery,.25);for(const force of [-2,0,2])for(const left of [-1,.3,1])for(const right of [-1,.2,1]){const q=initialBoundary(.7,.8,force,left,right,.37);close(q.bvp(0),left);close(q.bvp(1),right);const h=1e-3,x=.37;close((q.bvp(x+h)-2*q.bvp(x)+q.bvp(x-h))/(h*h),-force,1e-8);close((q.ode(x+1e-5)-q.ode(x-1e-5))/2e-5,-.7*q.ode(x),1e-9);}
});
test('waiting-time solutions satisfy the same IVP, including derivative continuity at departure',()=>{
 for(const c of [0,.2,.5,1.5])for(const t of [0,.1,.5,1,2]){const r=waitingSolutions(1,c,t,1,.1);close(r.initialDifference,0);close(r.residual,0);close(r.value,t>c?(t-c)**2:0);assert.equal(r.unique,false);const h=1e-6;close((r.first(c+h)-r.first(Math.max(0,c-h)))/(c===0?h:2*h),0,1e-6);}close(waitingSolutions(1,.5,1,1,.1).value,.25);close(waitingSolutions(1,.5,1,1,.1).other,1);
});
test('linear uniqueness and initial-data sensitivity are not confused with nonuniqueness',()=>{
 const r=waitingSolutions(1,.5,1,0,.1);assert.equal(r.unique,true);close(r.initialDifference,.1);close(r.other,.1*Math.E);assert.ok(2*Math.sqrt(1e-12)/1e-12>1e6);
});
test('linear forced flows match integrating factors, semigroup and the zero-rate limit',()=>{
 close(expIntegral(0,1.5),1.5);close(expIntegral(1e-14,1.5),1.5,1e-12);for(const r of [-2,0,1e-14,1])for(const mode of [0,1,2]){const q=odeFlow(r,.2,.3,.7,mode,.5),h=1e-5,y=q.solve(.2,.7),derivative=(q.solve(.2,.7+h)-q.solve(.2,.7-h))/(2*h);close(derivative,mode===0?r*y:mode===1?r*y+.5:r*y*(1-y),1e-8);close(q.solve(q.solve(.2,.3),.4),q.solve(.2,.7));}close(odeFlow(1,.2,.3,1,1,.5).defect,-.5*(Math.E-1));close(odeFlow(1,.2,.3,1,0,.5).defect,0);assert.ok(Math.abs(odeFlow(1,.2,.3,1,2,0).defect)>.1);
});
test('phase orbits satisfy the actual matrix field, semigroup and equal-radius rotation',()=>{
 for(const mode of [0,1])for(const a of [-.3,0,.5])for(const b of [0,1,2]){const r=phaseFlow(a,b,.6,-.2,.7,mode),h=1e-5,lo=r.solution(.7-h),hi=r.solution(.7+h);vector(hi.map((v,i)=>(v-lo[i])/(2*h)),r.field(r.point),1e-8);const first=r.solution(.3),next=phaseFlow(a,b,...first,.4,mode).point;vector(next,r.point);}close(phaseFlow(0,1,1,0,1,0).radius,1);assert.equal(phaseFlow(.5,1,0,1,1,1).stable,false);close(phaseFlow(.5,1,0,1,1,1).radius,Math.exp(-1));assert.equal(phaseFlow(0,0,1,1,1,0).isolated,false);
});
test('pitchfork exact flow matches derivatives, critical cubic decay and stable branch equilibria',()=>{
 for(const mu of [-1,-1e-12,0,1e-12,.5,1])for(const y0 of [-1,0,.3,1]){const t=.7,h=1e-5,y=pitchforkValue(mu,y0,t),d=(pitchforkValue(mu,y0,t+h)-pitchforkValue(mu,y0,t-h))/(2*h);close(d,mu*y-y**3,1e-8);}close(pitchforkValue(0,.5,2),.5/Math.sqrt(2));close(pitchforkValue(.5,Math.sqrt(.5),2),Math.sqrt(.5));close(pitchforkValue(.5,0,2),0);
});
test('logistic-map iterates and finite diagnostics use the declared actual window, including empty and zero factors',()=>{
 const r=bifurcationExperiment(.5,.5,2,4,.2,120,20);for(let i=0;i<120;i++)close(r.orbit[i+1],4*r.orbit[i]*(1-r.orbit[i]));close(r.lyapunov!,r.orbit.slice(20,120).reduce((s,x)=>s+Math.log(Math.abs(4*(1-2*x))),0)/100);assert.equal(r.window,100);const empty=bifurcationExperiment(0,0,0,3.2,.2,20,100);assert.equal(empty.window,0);assert.equal(empty.lyapunov,null);const zero=bifurcationExperiment(0,0,0,4,.5,20,0);assert.equal(zero.zeroFactors,1);assert.equal(zero.lyapunov,null);
});
test('principal-symbol rotation matches independent matrix multiplication and invariants',()=>{
 for(const A of [-1,0,1])for(const B of [-1,0,.5])for(const C of [-1,0,2])for(const t of [0,.5,Math.PI/2]){const phi=.7,r=principalSymbol(A,B,C,t,phi),R=[[Math.cos(t),-Math.sin(t)],[Math.sin(t),Math.cos(t)]],Q=[[A,B],[B,C]],rot=R.map((_,i)=>R.map((_,j)=>R.reduce((s,row,k)=>s+row[i]*Q[k].reduce((v,a,l)=>v+a*R[l][j],0),0)));vector(r.rotated,[rot[0][0],rot[0][1],rot[1][1]]);close(r.rotated[0]+r.rotated[2],A+C);close(r.rotated[0]*r.rotated[2]-r.rotated[1]**2,A*C-B*B);close(r.rotationResidual,0);close(r.eigen[0]*r.eigen[1],r.det);}
});
test('elliptic, hyperbolic, rank-one and zero principal parts are distinguished without factor-two errors',()=>{
 assert.equal(principalSymbol(1,0,1,0,0).type,'elliptic');assert.equal(principalSymbol(-1,0,-1,0,0).type,'elliptic');assert.equal(principalSymbol(1,0,-1,0,0).type,'hyperbolic');assert.equal(principalSymbol(1,1,1,0,0).type,'parabolic rank one');assert.equal(principalSymbol(0,0,0,0,0).type,'zero principal part');close(principalSymbol(1,.5,1,0,Math.PI/4).query,1.5);
});
test('periodic mode derivatives satisfy heat, wave and advection PDEs through independent finite differences',()=>{
 for(const mode of [0,1,2]){const A=.7,M=.2,k=2,t=.6,nu=.3,c=1.2,B=.4,r=periodicMode(A,M,k,t,nu,c,B,mode),x=.37,h=1e-4,q=r.evaluate(x),u=(z:number,s:number)=>periodicMode(A,M,k,s,nu,c,B,mode).evaluate(z).value;close((u(x+h,t)-u(x-h,t))/(2*h),q.ux,1e-7);close((u(x,t+h)-u(x,t-h))/(2*h),q.ut,1e-7);if(mode===1)close((u(x,t+h)-2*u(x,t)+u(x,t-h))/(h*h),q.utt,1e-7);close(q.residual,0);close(integrate(z=>r.evaluate(z).value,0,2*Math.PI),2*Math.PI*M);close(integrate(z=>(r.evaluate(z).value-M)**2/2,0,2*Math.PI),r.centeredEnergy);}
});
test('heat dissipates, waves exchange energy and advection preserves centered L2',()=>{
 const heat=periodicMode(1,0,2,.5,.2,1,0,0);close(heat.amp,Math.exp(-.4));assert.ok(heat.heatDissipation<0);const wave=periodicMode(1,0,1,Math.PI/2,.2,1,0,1);assert.ok(wave.centeredEnergy<1e-25);close(wave.waveEnergy,Math.PI/2);close(wave.waveEnergy,wave.waveReference);for(const t of [0,.5,2]){const w=periodicMode(.7,.3,3,t,.2,1.4,.4,1);close(integrate(x=>{const q=w.evaluate(x);return (q.ut*q.ut+1.4**2*q.ux*q.ux)/2;},0,2*Math.PI),w.waveReference);close(periodicMode(1,.5,2,t,.2,1,0,2).centeredEnergy,Math.PI/2);}
});
test('characteristics transport initial values and inverse-map gradients only before breakdown',()=>{
 const r=characteristics(.5,1,.5,1,.5,0);close(r.jacobian,.5);close(r.gradient!,-2);close(r.queryPosition,.25);for(const b of [-1,0,1])for(const t of [0,.5,1,2]){const q=characteristics(.3,b,t,1,0,.2);close(q.position(.2,t),.2+t*(.3-b*.2));if(q.valid){const x=q.queryPosition,y=(.3-b*x)/(1-b*t);close(y,q.queryValue);}else assert.equal(q.gradient,null);}assert.equal(characteristics(.5,1,1,1,.5,0).valid,false);assert.equal(characteristics(.5,-1,2,1,.5,0).breakdown,null);close(characteristics(.5,1,2,0,.5,0).jacobian,1);
});
test('Riemann shocks and rarefactions have explicit weak and entropy diagnostics',()=>{
 const shock=riemannExperiment(1,0,1,.25,0),fan=riemannExperiment(0,1,1,.25,0),expansion=riemannExperiment(0,1,1,.25,1);close(shock.speed,.5);close(shock.production,-1/12);close(fan.query,.25);close(expansion.RH,0);close(expansion.production,1/12);assert.equal(expansion.entropy,false);assert.equal(fan.entropy,true);for(const L of [-1,0,1,2])for(const R of [-1,0,1,2])for(const mode of [0,1]){const r=riemannExperiment(L,R,1,.2,mode);close(r.RH,0);close(r.production,r.fan?0:(R-L)**3/12);}
});
test('Riemann interval mass and boundary fluxes satisfy exact open-domain conservation including exiting waves',()=>{
 for(const L of [-1,-.3,0,1,2])for(const R of [-1,0,.4,1,2])for(const mode of [0,1])for(const t of [0,.5,1,2]){const r=riemannExperiment(L,R,t,0,mode);close(r.balance,0,1e-10);close(r.mass,riemannMass(r.data,-2,2,t));}const r=riemannExperiment(2,1,2,0,0);close(r.mass,8);close(r.initialMass,6);close(r.outFlux-r.inFlux,-2);
});
test('boundary-flux integrals match independent midpoint integration through shock and fan events',()=>{
 for(const data of [{left:-1,right:2,mode:0},{left:0,right:2,mode:0},{left:2,right:-1,mode:0},{left:-1,right:2,mode:1}])for(const x of [-2,0,2]){const T=2,N=50000,h=T/N,quad=Array.from({length:N},(_,i)=>{const v=riemannValue(data,x,(i+.5)*h);return v*v*h/2;}).reduce((a,b)=>a+b,0);close(boundaryFluxIntegral(data,x,T),quad,1e-4);}
});
test('shock rendering never inserts a fake ramp at a plot boundary',()=>{
 const r=evaluateDifferential('weak-entropy-riemann-solutions',[2,1,2,0,0]);assert.ok(r.curves.some(line=>line[0][0]===-3&&line.at(-1)![0]===3&&line.every(p=>p[1]===2)));assert.ok(r.dots!.some(([x,y])=>x===3&&y===1));
});
test('finite-volume initial averages integrate the original pulse exactly for nonaligned grids',()=>{
 for(const N of [4,7,12,24]){const r=finiteVolume(N,1,.8,0,0,.3,.7);close(r.initialMass,.175);vector(r.initial,r.exact);close(r.L1,0);}
});
test('upwind updates match independent flux differences and telescope periodic and open boundaries',()=>{
 for(const mode of [0,1])for(const N of [4,7,12])for(const cfl of [0,.5,1,1.2]){const r=finiteVolume(N,1.3,cfl,20,mode,.4,.7);let u=[...r.initial],cumulative=0;for(const h of r.history){close(h.mass,u.reduce((a,b)=>a+b,0)/N);close(h.balance,0,1e-9);close(h.cumulative,cumulative);const flux=[1.3*(mode===0?u.at(-1)!:.4),...u.map(v=>1.3*v)];cumulative+=r.dt*(flux[0]-flux.at(-1)!);u=u.map((v,i)=>v-r.dt/r.dx*(flux[i+1]-flux[i]));}if(cfl<=1){assert.ok(r.last.min>=-1e-12);assert.ok(r.last.max<=.7+1e-12);}}
});
test('CFL-one translations reproduce exact cell averages at matched physical time, including inflow and wraps',()=>{
 for(const N of [4,7,12,24])for(const steps of [0,1,8,25,40])for(const mode of [0,1]){const r=finiteVolume(N,1.3,1,steps,mode,.4,.7);close(r.L1,0,1e-10);close(r.time,steps/(N*1.3));}const unstable=finiteVolume(12,1,1.2,8,0,0,1);assert.ok(unstable.last.min<0||unstable.last.max>1);close(unstable.last.mass,.25);
});
test('differential model domains reject malformed, nonfinite and fractional discrete controls',()=>{
 for(const kind of differentialKinds){const s=differentialLabs[kind],initial=s.controls.map(c=>c.initial);assert.throws(()=>evaluateDifferential(kind,[]));for(const [i,c] of s.controls.entries())for(const v of [NaN,Infinity,c.min-1,c.max+1,...(c.step==='any'?[]:[c.min+.5])]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluateDifferential(kind,bad));}}
});
for(const kind of differentialKinds)test(`${kind}: meaningful guides and finite geometry across parameter boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const s=differentialLabs[kind],initial=s.controls.map(c=>c.initial);for(const p of [...s.presets.map(p=>p.values),...s.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))]){const r=evaluateDifferential(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[]),...(r.holes??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(Number.isFinite(v));if(r.phase)assert.ok(r.phase.trajectory.flat().every(Number.isFinite));if(r.characteristics)assert.ok(r.characteristics.lines.flat(2).every(Number.isFinite));}
});
for(const kind of differentialKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
