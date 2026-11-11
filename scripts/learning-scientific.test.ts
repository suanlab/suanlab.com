import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {scientificKinds,evaluateScientific,integrateDecay,discretizationError,stiffModes,oscillator,tridiagonal,poissonMethods,spectralDerivative,meshBoundary,cflExperiment,twoGrid,matvec,denseSolve,discreteAdjoint,manufacturedConvergence} from '../src/lib/learning-scientific';
import {scientificLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const norm=(x:number[])=>Math.max(0,...x.map(Math.abs));
const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
test('analytic decay error retains a tiny lost state change without erasing large-time tails',()=>{
 for(const rate of [1e-20,1e-100,1e-200])for(const method of [0,1,2,3]){
  const r=integrateDecay(rate,1,4,method);
  assert.equal(r.path.at(-1),1);assert.equal(r.exact,1);
  assert.ok(r.error>0&&Math.abs(r.error/rate-1)<1e-14);
 }
 const tail=integrateDecay(80,2,160,0);
 assert.equal(tail.path.at(-1),0);assert.ok(tail.error>0);close(tail.error/Math.exp(-160),1,1e-14);
});
test('pure Neumann compatibility uses the prescribed source bias rather than a roundoff tolerance',()=>{
 for(const bias of [1e-20,-1e-20,1e-100]){
  const r=meshBoundary(8,1.5,.5,bias,2);
  assert.equal(r.compatibility,bias);assert.equal(r.compatible,false);assert.equal(r.values,null);
 }
 assert.equal(meshBoundary(8,1.5,.5,0,2).compatible,true);
});
test('unit-CFL transport is a bit-exact permutation, including samples near zero',()=>{
 for(const power of [3,4,5])for(const steps of [1,10,50])for(const noise of [0,.001]){
  const r=cflExperiment(power,1,steps,.5,1,noise);
  assert.deepEqual(r.values,r.initial.map((_,i)=>r.initial[(i-steps%r.N+r.N)%r.N]));
 }
});
test('continuous logistic sensitivity retains a small target mismatch independently of rounded states',()=>{
 for(const theta of [-1e-20,1e-20,-1e-100,1e-100]){
  const r=discreteAdjoint(theta,1,8,.2),leading=.0256*theta;
  assert.ok(r.continuousGradient!==0&&Math.abs(r.continuousGradient/leading-1)<1e-14);
  assert.equal(r.expandedContinuousGradient,0);
 }
});
test('actual integrator stages match independent stability polynomials and zero initial states',()=>{
 for(const rate of [0,.3,2,8])for(const T of [.1,1,2])for(const N of [1,7,64])for(let method=0;method<4;method++){const z=-rate*T/N,R=[1+z,1+z+z*z/2,1+z+z*z/2+z**3/6+z**4/24,1/(1-z)][method],r=integrateDecay(rate,T,N,method);close(r.factor,R);close(r.path.at(-1)!,R**N,2e-8);close(r.exact,Math.exp(-rate*T));assert.equal(integrateDecay(rate,T,N,method,0).path.at(-1),0);}
 close(integrateDecay(1,1,2,0).path.at(-1)!,1/4);close(integrateDecay(1,1,2,3).path.at(-1)!,4/9);
});
test('unscaled local defects have p+1 order and matched-time global errors p order',()=>{
 for(const [m,p] of [[0,1],[1,2],[2,4],[3,1]]){const r=discretizationError(1,1,8,m);assert.ok(Math.abs(r.globalOrder!-p)<.2);assert.ok(Math.abs(r.localOrder!-(p+1))<.15);for(const level of r.levels)close(level.N*level.h,1);}const r=discretizationError(1,1,2,0);close(r.levels[0].local,Math.exp(-.5)-.5);close(r.levels[0].global,Math.exp(-1)-.25);
});
test('stiff mode absence hides growth without changing the unstable multiplier',()=>{
 const absent=stiffModes(40,1,10,0,0,0),seeded=stiffModes(40,1,10,0,1e-6,0),implicit=stiffModes(40,1,10,1,0,1);assert.equal(absent.fastStable,false);assert.equal(absent.quick.path.at(-1),0);close(seeded.quick.path.at(-1)!,1e-6*3**10);close(implicit.quick.factor,.2);assert.ok(implicit.slow.error>0);close(stiffModes(40,1,10,1,0,0).quick.path.at(-1)!,59049);
});
test('independent oscillator matrices preserve symplectic form while Euler expands area',()=>{
 for(const w of [.5,1,2])for(const N of [4,40,200])for(const m of [0,1,2]){const r=oscillator(w,2,N,m),h=2/N,z=w*h,M=m===0?[[1,h],[-h*w*w,1]]:m===1?[[1-z*z,h],[-h*w*w,1]]:[[1-z*z/2,h],[-h*w*w*(1-z*z/4),1-z*z/2]],det=M[0][0]*M[1][1]-M[0][1]*M[1][0];close(det,r.determinant);close(r.path[1][0],M[0][0]);close(r.path[1][1],M[1][0]);if(m)close(r.modifiedDrift!,0,1e-8);else assert.ok(r.energyDrift>0);}
 const r=oscillator(1,10,50,2);close(r.path[1][0],.98);close(r.path[1][1],-.198);close(r.energy[1],.499802);close(r.modified[1]!,r.modified[0]!);
});
test('Verlet inverse steps recover state, and area does not imply stable large steps',()=>{
 const h=.2,w=1.3,step=(q:number,p:number,t:number)=>{p-=t*w*w*q/2;q+=t*p;p-=t*w*w*q/2;return [q,p];};let [q,p]=step(.3,-.7,h);[q,p]=step(q,p,-h);close(q,.3);close(p,-.7);const r=oscillator(2,20,4,2);close(r.determinant,1);assert.equal(r.positiveModified,false);assert.ok(Math.abs(r.path.at(-1)![0])>1);
});
test('Thomas and pivoted coarse solvers agree with independent prescribed solutions',()=>{
 const A=[[4,-1,0],[-1,3,-1],[0,-1,2]],x=[2,-1,3],b=matvec(A,x),t=tridiagonal([-1,-1],[4,3,2],[-1,-1],b),d=denseSolve(A,b);t.forEach((v,i)=>close(v,x[i]));d.forEach((v,i)=>close(v,x[i]));const pivot=denseSolve([[0,1],[1,2]],[3,8]);close(pivot[0],2);close(pivot[1],3);
});
test('FD and exact-load FEM reproduce quadratic nodes; FV half-cell boundary has its own error',()=>{
 for(const N of [4,8,19,32])for(const m of [0,1,2]){const r=poissonMethods(N,0,.3,-.2,m);for(let i=0;i<r.values.length;i++)close(r.values[i]-r.exact(r.xs[i]),m===1?r.h*r.h/4:0);assert.ok(r.residual<1e-9);assert.equal(r.values.length,m===1?N:N-1);if(m===1){close(r.diag[0],3/r.h**2);for(let i=0;i<N;i++)close((r.flux[i+1]-r.flux[i])/r.h,2);}}
});
test('FEM sinusoidal loads match an independently integrated hat function',()=>{
 const r=poissonMethods(8,1,0,0,2),N=2000;for(let i=0;i<r.xs.length;i++){const x=r.xs[i],h=r.h,integrand=(t:number)=>r.source(t)*Math.max(0,1-Math.abs(t-x)/h);let integral=0;for(let j=0;j<N;j++){const t=x-h+(j+.5)*2*h/N;integral+=integrand(t)*2*h/N;}close(r.rhs[i],integral,5e-5);}assert.ok(poissonMethods(16,1,0,0,0).maxError<poissonMethods(8,1,0,0,0).maxError);
});
test('DFT differentiation resolves trigonometric polynomials and preserves normalization',()=>{
 for(const power of [3,4,5,6])for(const k of [1,2,3]){const r=spectralDerivative(power,k,.7);assert.ok(r.maxError<1e-10);close(r.sampleEnergy,r.fourierEnergy);assert.ok(r.fdError>r.maxError);}const r=spectralDerivative(4,3,1);close(r.difference[0],Math.sin(3*r.h)/r.h);
});
test('aliasing and real Nyquist convention do not pretend to recover unseen derivatives',()=>{
 const a=spectralDerivative(4,1,1),b=spectralDerivative(4,17,1);a.values.forEach((v,i)=>close(v,b.values[i]));a.spectral.forEach((v,i)=>close(v,b.spectral[i]));assert.ok(b.maxError>15);const ny=spectralDerivative(4,8,1);close(ny.spectral[0],0);close(ny.exact[0],8);assert.equal(ny.resolved,false);
});
test('nonuniform FEM stiffness has the expected nullspace and exact polynomial weak loads',()=>{
 for(const N of [4,9,32])for(const power of [.5,1,2])for(const m of [0,1,2]){const r=meshBoundary(N,power,.5,0,m);for(let i=0;i<=N;i++){close(r.diag[i]+(i?r.off[i-1]:0)+(i<N?r.off[i]:0),0);close(r.values![i],r.reference(r.xs[i]));}assert.equal(r.nullity,m===2?1:0);assert.ok(r.boundaryResidual!<1e-10);assert.ok(r.residual!<1e-10);if(m===2)close(r.compatibility!,0);}const r=meshBoundary(8,2,.5,0,0);close(r.minSpacing,1/64);
});
test('pure Neumann compatibility precedes gauge fixing; source bias changes other boundary problems',()=>{
 for(const delta of [-1,-.2,.2,1]){const r=meshBoundary(8,1.5,.5,delta,2);close(r.compatibility!,delta);assert.equal(r.compatible,false);assert.equal(r.values,null);assert.equal(r.residual,null);for(const mode of [0,1]){const q=meshBoundary(8,1.5,.5,delta,mode);q.values!.forEach((v,i)=>close(v,q.reference(q.xs[i])));}}const r=meshBoundary(8,1,.5,0,0),mid=(r.values![0]+r.values![1])/2;assert.ok(Math.abs(mid-r.reference((r.xs[0]+r.xs[1])/2))>1e-4);
});
test('CFL updates match independent Fourier multipliers and conserve periodic mass',()=>{
 for(const mode of [0,1])for(const r of [0,.4,.5,.75,1,1.5]){const q=cflExperiment(4,r,8,.5,mode,.001),ny=dot(q.values,q.values.map((_,i)=>(-1)**i))/q.N;close(ny,.001*(1-(mode===0?4:2)*r)**8);close(q.mass,q.initialMass,1e-8);assert.equal(q.stable,r<=(mode===0?.5:1));close(q.T,8*q.dt);assert.ok(q.maxAmplification>=1-1e-12);}const exploded=cflExperiment(4,.75,20,.5,0,.001);close(exploded.nyquist,1048.576);
});
test('unit-CFL positive upwind translates all samples by exactly one cell per step',()=>{
 const r=cflExperiment(4,1,10,.5,1,.001);r.values.forEach((v,i)=>close(v,r.initial[(i-10+r.N)%r.N]));assert.ok(r.maxError<1e-12);const init=cflExperiment(3,1.5,0,.5,0,.01);assert.deepEqual(init.initial,init.values);close(init.T,0);
});
test('Galerkin two-grid correction annihilates coarse-space components without smoothing',()=>{
 const r=twoGrid(3,1,2/3,0,1),z=Array.from({length:r.nc},(_,i)=>i-.7),e=matvec(r.P,z),res=matvec(r.A,e).map(v=>-v),coarse=denseSolve(r.coarse,matvec(r.R,res)),corrected=e.map((v,i)=>v+matvec(r.P,coarse)[i]);assert.ok(norm(corrected)<1e-12);r.coarse.forEach((row,i)=>row.forEach((v,j)=>close(v,dot(r.R[i],matvec(r.A,r.P.map(p=>p[j]))))));assert.ok(norm(matvec(r.R,matvec(r.A,r.error)))<1e-10);
});
test('two-grid performs genuine smoothing and repeated residual correction on multiple grids',()=>{
 for(const p of [3,4,5])for(const m of [1,3,7]){const r=twoGrid(p,m,2/3,2,3);assert.ok(r.reduction<.02);assert.equal(r.history.length,4);const h=1/(r.n+1),res=matvec(r.A,r.error);close(r.history[3].residual,Math.sqrt(h*dot(res,res)));}const zero=twoGrid(4,1,2/3,2,0);assert.deepEqual(zero.error,zero.initial);close(zero.reduction,1);
});
test('forward tangent and reverse adjoint agree with independent centered objective differences',()=>{
 for(const theta of [-2,0,1,2])for(const T of [.2,1,2])for(const N of [2,16,64]){const r=discreteAdjoint(theta,T,N,.8);close(r.gradient,r.forwardGradient);close(r.gradient,r.fd,3e-8);}for(const N of [2,16,64])close(discreteAdjoint(0,1,N,.8).gradient,-.096);
});
test('discrete and continuous objectives are different at finite steps and converge under refinement',()=>{
 const coarse=discreteAdjoint(2,2,2,.8),fine=discreteAdjoint(2,2,64,.8);assert.ok(Math.abs(coarse.gradient-coarse.continuousGradient)>Math.abs(fine.gradient-fine.continuousGradient));const r=discreteAdjoint(1,1,8,.8);close(r.adjoint.at(-1)!,r.path.at(-1)!-.8);
});
test('manufactured direct solutions match the independent discrete sine eigenvalue answer',()=>{
 for(const p of [2,3,4])for(const bias of [0,.1])for(const nm of [0,1]){const r=manufacturedConvergence(p,bias,0,0,nm);for(const level of r.levels){const h=level.h,scale=Math.PI**2*h*h/(4*Math.sin(Math.PI*h/2)**2);level.values.forEach((v,i)=>close(v,scale*Math.sin(Math.PI*level.xs[i])+bias*(1-level.xs[i])));assert.ok(level.residual<1e-9);}if(bias===0)assert.ok(r.orders.every(p=>p!>1.9&&p!<2.1));else if(nm===0)assert.ok(r.levels.every(l=>l.error>=.1));}
});
test('fixed Jacobi work leaves algebraic error, while small residual can hide wrong boundaries',()=>{
 const low=manufacturedConvergence(3,0,1,5,0),high=manufacturedConvergence(3,0,1,500,0),wrong=manufacturedConvergence(3,.1,0,100,0);assert.ok(high.levels[2].error<low.levels[2].error);assert.ok(wrong.levels[2].error>=.1);assert.ok(wrong.levels[2].residual<1e-9);const init=manufacturedConvergence(2,.1,1,0,0);assert.ok(init.levels.every(l=>l.values.every(v=>v===0)));
});
for(const id of scientificKinds)test(`scientific presets, parameter boundaries and bilingual guide: ${id}`,()=>{
 const spec=scientificLabs[id],guide=learningGuides[id],unit=learningUnits.find(u=>u.id===id)!;assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);assert.equal(unit.questions.length,2);const vectors=[spec.controls.map(c=>c.initial),...spec.presets.map(p=>p.values)];for(let i=0;i<spec.controls.length;i++)for(const v of [spec.controls[i].min,spec.controls[i].max]){const p=spec.controls.map(c=>c.initial);p[i]=v;vectors.push(p);}for(const p of vectors){const r=evaluateScientific(id,p);assert.ok(r.curves.flat(2).every(Number.isFinite));if(r.bounds){const [lx,hx,ly,hy]=r.bounds;assert.ok(hx>lx&&hy>ly);assert.ok(r.curves.flat().every(([x,y])=>x>=lx&&x<=hx&&y>=ly&&y<=hy));}for(const v of Object.values(r.stats).flat())assert.ok(v===null||Number.isFinite(v));assert.ok(!/[가-힣]/.test(r.statement??''));if(r.phase)assert.ok(r.phase.flat().every(Number.isFinite));if(r.mesh)assert.ok(r.mesh.every((x,i)=>Number.isFinite(x)&&(!i||x>r.mesh![i-1])));}assert.throws(()=>evaluateScientific(id,[]));const invalid=spec.controls.map(c=>c.initial);invalid[0]=NaN;assert.throws(()=>evaluateScientific(id,invalid));const int=spec.controls.findIndex(c=>c.step===1);if(int>=0){const p=spec.controls.map(c=>c.initial);p[int]=spec.controls[int].min+.5;assert.throws(()=>evaluateScientific(id,p));}
});
for(const id of scientificKinds)for(const locale of ['ko','en'] as const)test(`scientific substantive content and rendered equations: ${id}.${locale}`,async()=>{
 const raw=await fs.readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8'),content=await getLearningContent(id);assert.ok((raw.match(/^## /gm)||[]).length>=5);assert.ok(raw.includes('$$'));assert.ok(raw.includes('https://'));assert.ok(raw.includes(locale==='en'?'Exercise:':'직접 풀기:'));if(locale==='en'){assert.ok(raw.split(/\s+/).length>350);assert.ok(!/[가-힣]/.test(raw));}else assert.ok((raw.match(/[가-힣]/g)||[]).length>700);assert.ok(content[locale].includes('katex-display'));assert.ok(!content[locale].includes('katex-error'));assert.ok(content[locale].length>4000);
});
