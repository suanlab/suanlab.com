import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {markovExperiment,continuousTwoState,reverseKernel,poissonExperiment,martingaleExperiment,brownianGrid,stochasticSums,expAverage,oneMinusAverage,bridgeVariance,ouDiscretization,ouDensity,reversedOU,ergodicExperiment,leapfrog,mcmcExperiment,lqgExperiment,evaluateStochastic,stochasticKinds} from '../src/lib/learning-stochastic';
import {stochasticLabs,learningUnits,learningGuides} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<=tol*(1+Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-9)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));};
const integrate=(f:(x:number)=>number,a:number,b:number,N=2000)=>{let s=f(a)+f(b);for(let i=1;i<N;i++)s+=(i%2?4:2)*f(a+(b-a)*i/N);return s*(b-a)/(3*N);};
const multiply=(A:number[][],B:number[][])=>A.map(row=>B[0].map((_,j)=>row.reduce((s,v,k)=>s+v*B[k][j],0)));
test('Markov iteration, horizon kernels and Bayes reversal preserve actual joint probabilities',()=>{
 for(const a of [0,.2,1])for(const b of [0,.4,1])for(const p of [0,.3,1])for(const N of [0,1,5]){const r=markovExperiment(a,b,p,N,1,0);let P=[[1,0],[0,1]];for(let n=0;n<N;n++)P=multiply(P,[[1-a,a],[b,1-b]]);vector(r.matrix.flat(),P.flat());close(r.current,(1-p)*P[0][1]+p*P[1][1]);r.reverse.reverse.forEach((row,j)=>{if(row){close(row[0]+row[1],1);for(let i=0;i<2;i++)close(r.reverse.next[j]*row[i],[1-p,p][i]*P[i][j]);}else close(r.reverse.next[j],0);});}close(markovExperiment(.2,.4,1,2,1,0).current,.44);assert.equal(reverseKernel([[1,0],[0,1]],[0,1]).reverse[0],null);
});
test('continuous transition kernels match independent uniformization and semigroup',()=>{
 for(const a of [0,.3,1])for(const b of [0,.7,1])for(const t of [0,.5,2]){const M=continuousTwoState(a,b,t),q=Math.max(a,b);if(q===0){vector(M.flat(),[1,0,0,1]);continue;}const U=[[1-a/q,a/q],[b/q,1-b/q]];let power=[[1,0],[0,1]],weight=Math.exp(-q*t),answer=[[weight,0],[0,weight]];for(let n=1;n<60;n++){power=multiply(power,U);weight*=q*t/n;answer=answer.map((row,i)=>row.map((v,j)=>v+weight*power[i][j]));}vector(M.flat(),answer.flat());vector(multiply(continuousTwoState(a,b,t/3),continuousTwoState(a,b,2*t/3)).flat(),M.flat());}
});
test('Poisson probabilities have independent factorial answers and a separately normalized tail',()=>{
 for(const rate of [0,.2,2,4])for(const T of [0,1,3])for(const K of [0,2,10,20]){const r=poissonExperiment(rate,T,K,7),m=rate*T;let factorial=1;for(let k=0;k<=K;k++){if(k)factorial*=k;close(r.pmf[k],Math.exp(-m)*m**k/factorial);}close(r.shownMass+r.tail,1);assert.ok(r.tail>=0);if(rate===0||T===0)assert.equal(r.arrivals.length,0);for(let i=0;i<r.arrivals.length;i++){assert.ok(r.arrivals[i]>0&&r.arrivals[i]<=T);if(i)assert.ok(r.arrivals[i]>r.arrivals[i-1]);}}close(poissonExperiment(2,1,2,7).shownMass,5*Math.exp(-2));
});
test('Poisson cutoff leaves the seeded arrival path unchanged instead of conditioning it',()=>{
 const a=poissonExperiment(2,1,2,7),b=poissonExperiment(2,1,10,7);assert.deepEqual(a.arrivals,b.arrivals);assert.ok(a.tail>b.tail);assert.equal(a.capped,false);
});
test('bounded stopped martingale expectations match exhaustive path enumeration',()=>{
 for(const p of [.25,.5,.75])for(const N of [2,5,8])for(const B of [1,2,3]){let expectedS=0,expectedTime=0,expectedM=0;for(let mask=0;mask<2**N;mask++){let prob=1,state=0,tau=N,hit=false;for(let n=0;n<N;n++){const up=Boolean(mask&(1<<n));prob*=up?p:1-p;if(!hit){state+=up?1:-1;if(Math.abs(state)>=B){tau=n+1;hit=true;}}}expectedS+=prob*state;expectedTime+=prob*tau;expectedM+=prob*(state-(2*p-1)*tau);}const r=martingaleExperiment(p,N,B,7);close(r.last.meanS,expectedS);close(r.last.meanTime,expectedTime);close(r.last.meanM,expectedM);close(r.last.mass,1);close(r.last.meanM,0);}const r=martingaleExperiment(.75,4,2,7);close(r.last.meanTime,2.75);close(r.last.meanS,1.375);
});
test('sampled stopped compensation uses the frozen clock and cannot redefine the exact expectation',()=>{
 const a=martingaleExperiment(.75,8,1,7),b=martingaleExperiment(.75,8,1,19);close(a.sampleStop,1);close(a.sampleM.at(-1)![1],a.sampleState-.5);close(a.last.meanM,b.last.meanM);assert.deepEqual(a.history,b.history);
});
test('Brownian grids preserve shared endpoints and rescale the same realization correctly',()=>{
 const a=brownianGrid(4,1,7),b=brownianGrid(6,1,7),c=brownianGrid(4,2,7);assert.equal(a.end,b.end);for(let i=0;i<=a.N;i++)close(a.path[i],b.path[i*4]);vector(c.path,a.path.map(x=>x*Math.SQRT2));close(c.QV,2*a.QV);close(a.increments.reduce((s,x)=>s+x,0),a.end);assert.equal(a.path[0],0);
});
test('Brownian covariance follows independent increment overlap, not one path statistics',()=>{
 for(const N of [4,16,128])for(const i of [0,1,N/2,N])for(const j of [0,N/2,N]){const coeffA:number[]=Array.from({length:N},(_,k)=>k<i?1:0),coeffB=Array.from({length:N},(_,k)=>k<j?1:0),cov=coeffA.reduce((s,v,k)=>s+v*coeffB[k]/N,0);close(cov,Math.min(i,j)/N);}close(2/(16),.125);close(2/64,.03125);
});
test('actual Ito and symmetric sums satisfy telescoping identities and the observed-QV correction',()=>{
 for(const k of [2,4,7])for(const T of [.1,1,2])for(const seed of [1,7,19]){const r=stochasticSums(k,T,seed);close(r.left.at(-1)!,r.leftIdentity);close(r.mid.at(-1)!,r.midIdentity);close(r.mid.at(-1)!-r.left.at(-1)!,r.QV/2);close(r.left.at(-1)!-r.trueIto,(T-r.QV)/2);close(r.theoreticalMSE,T*T/(2*2**k));}assert.ok(Math.abs(stochasticSums(4,1,7).QV-1)>.001);
});
test('OU bridge covariance matches nonnegative independent kernel integrals and small-rate limits',()=>{
 close(expAverage(0),1);close(oneMinusAverage(0),0);close(bridgeVariance(0),0);for(const z of [.001,.01,.1,.5,1]){const mean=integrate(x=>Math.exp(-z*x),0,1),variance=integrate(x=>(Math.exp(-z*x)-mean)**2,0,1);close(expAverage(z),mean);close(bridgeVariance(z),variance,1e-11);assert.ok(bridgeVariance(z)>=0);}close(bridgeVariance(1e-10)/(1e-20),1/12,1e-9);close(oneMinusAverage(1e-14)/1e-14,.5,1e-10);
});
test('coupled OU strong MSE matches direct Brownian-kernel integration rather than a shared-Z shortcut',()=>{
 for(const theta of [0,.5,2])for(const sigma of [0,1])for(const T of [.1,1,2])for(const k of [2,4]){const r=ouDiscretization(theta,sigma,.7,T,k,7),N=2**k,h=T/N,a=1-theta*h,meanError=.7*(a**N-Math.exp(-theta*T));let noise=0;for(let j=0;j<N;j++)noise+=sigma*sigma*integrate(s=>(a**(N-1-j)-Math.exp(-theta*(T-s)))**2,j*h,(j+1)*h,1000);close(r.last.MSE,meanError*meanError+noise,1e-10);if(theta===0){close(r.last.MSE,0);close(r.last.euler,r.last.exact);}}
});
test('OU exact moments, deterministic limits and finite innovations remain valid at tiny theta',()=>{
 const r=ouDiscretization(1,1,1,1,3,7);close(r.mean,Math.exp(-1));close(r.variance,(1-Math.exp(-2))/2);close(ouDiscretization(1,0,1,1,3,7).variance,0);const tiny=ouDiscretization(1e-14,1,1,1,3,7);assert.ok(tiny.last.MSE>0);assert.ok(Number.isFinite(tiny.last.MSE));
});
test('Gaussian Fokker-Planck density matches independent differentiation, normalization and moment equations',()=>{
 for(const theta of [0,1,2]){const r=ouDensity(theta,1,.7,.5,.6),x=.3,h=1e-5,q=r.evaluate(x),lo=ouDensity(theta,1,.7,.5,.6-h),hi=ouDensity(theta,1,.7,.5,.6+h);close((hi.evaluate(x).rho-lo.evaluate(x).rho)/(2*h),q.dt,1e-8);close((r.evaluate(x+h).flux-r.evaluate(x-h).flux)/(2*h),q.dxFlux,1e-8);close(q.residual,0);close(integrate(z=>r.evaluate(z).rho,r.mean-10*Math.sqrt(r.variance),r.mean+10*Math.sqrt(r.variance)),1);close((hi.mean-lo.mean)/(2*h),r.meanPrime);close((hi.variance-lo.variance)/(2*h),r.variancePrime);}
});
test('stationary and diffusive Gaussian laws are distinct exact special cases',()=>{
 const stationary=ouDensity(1,Math.SQRT2,0,1,1);close(stationary.variance,1);close(stationary.evaluate(.5).flux,0);close(ouDensity(0,1,1,.5,1).variance,1.25);close(ouDensity(1,1,1,1,1).mean,Math.exp(-1));
});
test('reverse OU score correction matches reversed physical-time moment derivatives',()=>{
 for(const theta of [0,1,2]){const r=reversedOU(theta,1,.7,.5,1,.4,.3),h=1e-5,lo=reversedOU(theta,1,.7,.5,1,.4-h,.3),hi=reversedOU(theta,1,.7,.5,1,.4+h,.3);close((hi.mean-lo.mean)/(2*h),r.reverseMeanPrime);close((hi.variance-lo.variance)/(2*h),r.reverseVariancePrime);close(r.varianceEquationResidual,0);close(r.drift(.3),theta*.3-(.3-r.mean)/r.variance);}
});
test('stationary reversibility and zero-forward-drift examples refute naive sign reversal',()=>{
 const r=reversedOU(1,Math.SQRT2,0,1,1,.5,1);close(r.queryDrift,-1);close(r.naive(1),1);close(reversedOU(0,1,0,1,1,1,1).queryDrift,-1);close(reversedOU(1,1,1,.5,1,0,1).time,1);
});
test('finite-window chain means and variances match full enumeration including periodicity',()=>{
 for(const a of [0,.2,1])for(const b of [0,.4,1])for(const p0 of [.3,1])for(const burn of [0,2,9]){const N=6,r=ergodicExperiment(a,b,p0,N,burn,7);if(burn>N){assert.equal(r.meanVariance,null);continue;}let E=0,E2=0;for(let mask=0;mask<2**(N+1);mask++){const state=Array.from({length:N+1},(_,i)=>(mask>>i)&1);let prob=state[0]?p0:1-p0;for(let n=0;n<N;n++)prob*=state[n]===0?(state[n+1]?a:1-a):(state[n+1]?1-b:b);const avg=state.slice(burn).reduce((s,v)=>s+v,0)/(N+1-burn);E+=prob*avg;E2+=prob*avg*avg;}close(r.expectedMean!,E);close(r.meanVariance!,E2-E*E);}
});
test('periodic time averages and reducible nonconcentration are not confused with mixing',()=>{
 const periodic=ergodicExperiment(1,1,1,19,0,7);close(periodic.sampleMean!,.5);close(periodic.meanVariance!,0);assert.equal(periodic.irreducible,true);assert.equal(periodic.mixing,false);const reducible=ergodicExperiment(0,0,.3,20,0,7);assert.equal(reducible.stationary,null);close(reducible.meanVariance!,.21);
});
test('harmonic leapfrog is reversible and preserves phase area under an independent Jacobian check',()=>{
 for(const h of [.1,.5,1.5,2.2])for(const L of [1,5,12]){const r=leapfrog(.7,-.3,h,L),back=leapfrog(r.q,-r.p,h,L);close(back.q,.7,1e-6);close(back.p,.3,1e-6);}const h=.4,base=leapfrog(.7,-.3,h,3),eps=1e-5,q=leapfrog(.7+eps,-.3,h,3),p=leapfrog(.7,-.3+eps,h,3);close(((q.q-base.q)*(p.p-base.p)-(p.q-base.q)*(q.p-base.p))/(eps*eps),1,1e-8);
});
test('MCMC rejection counts, empirical windows and ULA invariant variance follow their actual algorithms',()=>{
 for(const mode of [0,1,2]){const r=mcmcExperiment(.5,60,10,7,1,mode,5);assert.equal(r.count,51);const retained=r.path.slice(10),mean=retained.reduce((s,x)=>s+x,0)/51;close(r.mean!,mean);close(r.second!,retained.reduce((s,x)=>s+(x-mean)**2,0)/51);if(mode===0)assert.equal(r.acceptance,null);else {close(r.acceptance!,r.path.slice(1).filter((v,i)=>v!==r.path[i]).length/60);assert.ok(r.acceptance!>=0&&r.acceptance!<=1);}assert.deepEqual(r,mcmcExperiment(.5,60,10,7,1,mode,5));}for(const h of [.05,.5,1.9]){const v=mcmcExperiment(h,10,0,7,1,0,5).ulaStationaryVariance!;close((1-h)**2*v+2*h,v);}assert.equal(mcmcExperiment(2.2,10,100,7,1,0,5).ulaStationaryVariance,null);assert.equal(mcmcExperiment(.5,10,100,7,1,1,5).mean,null);
});
test('LQG Bellman recursion satisfies independent Gaussian quadrature and local minimization',()=>{
 const r=lqgExperiment(1,1,1,1,1,1,0);close(r.gain[0],.5);close(r.value,1.25);close(r.selected.cost,1.25);close(lqgExperiment(1,1,1,1,1,1,1).selected.cost,1.5);close(lqgExperiment(1,1,1,1,0,1,0).value,.75);
 for(const a of [.5,1,1.5])for(const R of [.2,1,2])for(const sigma of [0,.5,1]){const r=lqgExperiment(a,R,1,4,sigma,.7,0);close(r.value,r.optimal.cost);for(let n=0;n<4;n++){const x=.7,u=-r.gain[n]*x,V=(control:number)=>(x*x+R*control*control)/2+[-Math.sqrt(3),0,Math.sqrt(3)].reduce((s,z,i)=>s+[1/6,2/3,1/6][i]*(r.P[n+1]*(a*x+control+sigma*z)**2/2+r.offset[n+1]),0);close(V(u),r.P[n]*x*x/2+r.offset[n]);for(const trial of [-2,-1,0,1,2])assert.ok(V(trial)>=V(u)-1e-10);}assert.ok(lqgExperiment(a,R,1,4,sigma,.7,1).selected.cost>=r.value-1e-10);}
});
test('LQG additive-noise offsets change values while fully observed optimal gains stay fixed',()=>{
 const a=lqgExperiment(1,1,1,4,0,1,0),b=lqgExperiment(1,1,1,4,1,1,0);assert.deepEqual(a.gain,b.gain);assert.ok(b.value>a.value);close(a.offset[0],0);close(lqgExperiment(1,1,0,1,1,1,0).gain[0],0);
});
test('stochastic model domains reject malformed, nonfinite and fractional discrete controls',()=>{
 for(const kind of stochasticKinds){const s=stochasticLabs[kind],initial=s.controls.map(c=>c.initial);assert.throws(()=>evaluateStochastic(kind,[]));for(const [i,c] of s.controls.entries())for(const v of [NaN,Infinity,c.min-1,c.max+1,...(c.step==='any'?[]:[c.min+.5])]){const bad=[...initial];bad[i]=v;assert.throws(()=>evaluateStochastic(kind,bad));}}
});
for(const kind of stochasticKinds)test(`${kind}: meaningful guides and finite geometry across parameter boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const s=stochasticLabs[kind],initial=s.controls.map(c=>c.initial);for(const p of [...s.presets.map(p=>p.values),...s.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))]){const r=evaluateStochastic(kind,p);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.bounds!.every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(Number.isFinite(v));if(r.diagram)assert.ok(r.diagram.curves.flat(2).every(Number.isFinite));}
});
for(const kind of stochasticKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
