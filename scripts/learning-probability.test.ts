import test from 'node:test';
import assert from 'node:assert/strict';
import { probabilityKinds, evaluateProbability as evaluate, binomialMass, binomialDeviationIndices, normalCdf } from '../src/lib/learning-probability';
import { probabilityLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('finite probability measures obey normalization, complement and inclusion exclusion for every event pair',()=>{
 for(let mode=0;mode<3;mode++)for(let a=0;a<16;a++)for(let b=0;b<16;b++){
  const r=evaluate('probability-spaces-events',[mode,a,b]);close(Number(r.stats['P(A∪B)'])+Number(r.stats['P(A∩B)']),Number(r.stats['P(A)'])+Number(r.stats['P(B)']));close(Number(r.stats['P(A)'])+Number(r.stats['P(Aᶜ)']),1);close(r.table.rows.reduce((s,row)=>s+Number(row[2]),0),1);
 }
 assert.equal(evaluate('probability-spaces-events',[2,4,8]).stats['P(A)'],0);
});
test('conditional direction, Bayes and independence agree with direct four-cell counts without defining null ratios',()=>{
 for(let a=0;a<=2;a++)for(let b=0;b<=2;b++)for(let c=0;c<=2;c++)for(let d=0;d<=2;d++){
  const r=evaluate('conditional-probability-independence',[a,b,c,d]),N=a+b+c+d;if(!N){assert.equal(r.stats['P(A)'],null);assert.equal(r.stats['1{independent}'],null);assert.deepEqual(r.curves,[]);continue;}
  if(a+b)close(Number(r.stats['P(A|B)'])*(a+b)/N,a/N);else assert.equal(r.stats['P(A|B)'],null);
  if(a+c)close(Number(r.stats['P(B|A)'])*(a+c)/N,a/N);else assert.equal(r.stats['P(B|A)'],null);
  close(r.grid!.reduce((s,x)=>s+x.p,0),1);
 }
 close(Number(evaluate('conditional-probability-independence',[1,2,1,6]).stats['P(A|B)']),1/3);
});
test('point mass is a CDF jump while a uniform density can exceed one',()=>{
 const u=evaluate('mass-density-cdf',[1,.5,.25]);assert.equal(u.stats['Lebesgue density f(x)'],2);assert.equal(u.stats['P(X=x)'],0);assert.equal(u.stats['F(x)=P(X≤x)'],.5);
 for(const [x,F,mass] of [[-.1,0,0],[0,.75,.75],[.5,.75,0],[1,1,.25]]){const r=evaluate('mass-density-cdf',[0,1,x]);assert.equal(r.stats['F(x)=P(X≤x)'],F);assert.equal(r.stats['P(X=x)'],mass);assert.equal(r.stats['Lebesgue density f(x)'],null);}
});
test('binary joint laws preserve marginals and reproduce direct centered-product covariance',()=>{
 for(let a=0;a<=2;a++)for(let b=0;b<=2;b++)for(let c=0;c<=2;c++)for(let d=0;d<=2;d++){
  const r=evaluate('joint-marginal-conditional',[a,b,c,d]),N=a+b+c+d;if(!N){assert.equal(r.stats['Cov(X,Y)'],null);continue;}
  const ex=(c+d)/N,ey=(b+d)/N;close(Number(r.stats['P(X=1)']),ex);close(Number(r.stats['P(Y=1)']),ey);const cov=r.grid!.reduce((sum,t)=>sum+(t.x-ex)*(t.y-ey)*t.p,0);close(Number(r.stats['Cov(X,Y)']),cov);
  if(ex===0||ex===1||ey===0||ey===1)assert.equal(r.stats['Corr(X,Y)'],null);
 }
});
test('Gaussian geometry preserves Mahalanobis radius and distinguishes singular laws from missing laws',()=>{
 for(const sx of [.5,1,2])for(const sy of [.5,1,2])for(const rho of [-1,-.5,0,.5,1]){
  const r=evaluate('multivariate-gaussians',[sx,sy,rho,1]);close(Number(r.stats['E[Y|X=x]']),rho*sy/sx);assert.ok(Number(r.stats['Var(Y|X=x)'])>=0);
  if(Math.abs(rho)===1){assert.equal(r.stats['2D density at (x,0)'],null);assert.equal(r.stats['Var(Y|X=x)'],0);r.curves[0].forEach(([x,y])=>close(y,rho*sy/sx*x));}
  else r.curves[0].forEach(([x,y])=>close(((x/sx)**2-2*rho*x*y/(sx*sy)+(y/sy)**2)/(1-rho*rho),1));
 }
 close(Number(evaluate('multivariate-gaussians',[1,1,0,0]).stats['mass inside unit Mahalanobis ellipse']),1-Math.exp(-.5));
});
test('signed transformations retain positive densities, include both square branches and represent collapsed atoms',()=>{
 for(const mode of [0,1])for(const a of [-2,-1,1,2])for(const u of [.25,.75]){
  const y=a*u,r=evaluate('random-variable-transformations',[mode,a,y]),h=1e-6,left=evaluate('random-variable-transformations',[mode,a,y-h]),right=evaluate('random-variable-transformations',[mode,a,y+h]);close(Number(r.stats['Lebesgue density at y']),(Number(right.stats['F_Y(y)'])-Number(left.stats['F_Y(y)']))/(2*h),1e-7);
 }
 assert.equal(evaluate('random-variable-transformations',[1,1,.25]).stats['Lebesgue density at y'],1);assert.equal(evaluate('random-variable-transformations',[1,1,0]).stats['1{unbounded density at boundary}'],1);
 for(const y of [-1,0,1]){const r=evaluate('random-variable-transformations',[0,0,y]);assert.equal(r.stats['P(Y=y)'],Number(y===0));assert.equal(r.stats['F_Y(y)'],Number(y>=0));assert.equal(r.stats['Lebesgue density at y'],null);}
});
test('binomial masses equal independent binary-outcome enumeration and concentrate the sample mean',()=>{
 for(let n=1;n<=8;n++)for(const p of [0,.2,.5,1]){
  const expected=Array(n+1).fill(0);for(let mask=0;mask<2**n;mask++){let ones=0;for(let k=0;k<n;k++)ones+=(mask>>k)&1;expected[ones]+=p**ones*(1-p)**(n-ones);}const mass=binomialMass(n,p);mass.forEach((v,k)=>close(v,expected[k]));close(mass.reduce((s,v)=>s+v,0),1);
 }
 for(const n of [1,10,30])for(const p of [0,.1,.5,.9,1]){const r=evaluate('laws-large-numbers-clt',[n,p,.2]);assert.ok(Number(r.stats['P(|mean−p|≥ε)'])<=Number(r.stats['Chebyshev bound'])+1e-12);if(p===0||p===1){assert.equal(r.stats['standardized variance'],null);assert.equal(r.stats['normal mass on displayed bins'],null);}}
});
test('inclusive deviation events use exact decimal boundaries without an arbitrary tolerance',()=>{
 assert.deepEqual(binomialDeviationIndices(10,.5,.2),[0,1,2,3,7,8,9,10]);assert.ok(binomialDeviationIndices(10,.1,.2).includes(3));assert.ok(!binomialDeviationIndices(10,.1,.2000000000000001).includes(3));assert.deepEqual(binomialDeviationIndices(10,0,.2),[2,3,4,5,6,7,8,9,10]);
});
test('normal CDF approximation is monotone, symmetric and agrees with known reference integrals',()=>{
 close(normalCdf(0),.5);close(normalCdf(1),.8413447460685429,1e-6);close(normalCdf(2),.9772498680518208,1e-6);let previous=0;for(let k=-80;k<=80;k++){const x=k/10,v=normalCdf(x);assert.ok(v>=previous);close(v+normalCdf(-x),1,1e-12);previous=v;}
});
test('pathwise gradients agree with independent finite differences and quadratic moment checks',()=>{
 for(const mu of [-1,0,1])for(const sigma of [0,.5,2])for(const e of [-1,0,1]){
  const r=evaluate('reparameterization-gradients',[mu,sigma,e]),f=(m:number,s:number)=>.5*(m+s*e-1)**2,h=1e-5,g=r.stats['pathwise gradient (μ,σ)'] as number[];close(g[0],(f(mu+h,sigma)-f(mu-h,sigma))/(2*h),1e-8);close(g[1],(f(mu,sigma+h)-f(mu,sigma-h))/(2*h),1e-8);assert.deepEqual(r.stats['two-node moment gradient'],r.stats['exact expected gradient']);
 }
 const r=evaluate('reparameterization-gradients',[0,1,0]);assert.deepEqual(r.stats['pathwise gradient (μ,σ)'],[-1,0]);assert.deepEqual(r.stats['exact expected gradient'],[-1,1]);
});
for(const kind of probabilityKinds)test(`probability presets, finite diagrams and input domains: ${kind}`,()=>{
 assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=probabilityLabs[kind];for(const p of spec.presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.table.rows.length>0);}assert.throws(()=>evaluate(kind,[]));const values=spec.controls.map(c=>c.initial);values[0]=NaN;assert.throws(()=>evaluate(kind,values));
});
test('uniform CDF corners use the actual support endpoint rather than a sampling grid',()=>{
 const r=evaluate('mass-density-cdf',[1,.733,.3]);assert.ok(r.cdf);assert.ok(r.curves[0].some(([x,y])=>x===.733&&y===1));
 const neg=evaluate('random-variable-transformations',[0,-2,-1]);assert.ok(neg.curves[0].some(([x,y])=>x===-2&&y===0));assert.ok(neg.curves[0].some(([x,y])=>x===0&&y===1));
});
