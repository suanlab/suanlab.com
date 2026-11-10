import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateFunctional as evaluate,geometricFunction,legendreProjection,spectralExample,triangleMoment,sobolevPower,rankOneKernel,projectionTail,functionalKinds} from '../src/lib/learning-functional';
import {functionalLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
function integrate(f:(x:number)=>number,lo=0,hi=1,n=2400){let s=f(lo)+f(hi);for(let i=1;i<n;i++)s+=(i%2?4:2)*f(lo+(hi-lo)*i/n);return s*(hi-lo)/(3*n);}
const num=(r:ReturnType<typeof evaluate>,key:string)=>r.stats[key] as number;
test('geometric polynomial norms and exact uniform tails agree with independent quadrature',()=>{
 for(let m=0;m<=8;m++){const r=geometricFunction(m),direct=(x:number)=>Array.from({length:m+1},(_,k)=>x**k/2**(k+1)).reduce((s,v)=>s+v,0);close(r.l1,integrate(direct));close(r.l2,Math.sqrt(integrate(x=>direct(x)**2)));close(1-direct(1),r.uniformError);for(let i=0;i<=100;i++)assert.ok(Math.abs(1/(2-i/100)-direct(i/100))<=r.uniformError+1e-14);}
 close(integrate(x=>1/(2-x)),Math.log(2));close(integrate(x=>1/(2-x)**2),.5);
});
test('parallelogram defects distinguish inner-product geometry from mere completeness',()=>{
 const norm=(x:number[],p:number)=>p===Infinity?Math.max(...x.map(Math.abs)):x.reduce((s,v)=>s+Math.abs(v)**p,0)**(1/p);
 for(const [mode,p] of [[0,1],[1,2],[2,Infinity]]){const defect=norm([1,1],p)**2+norm([1,-1],p)**2-2*norm([1,0],p)**2-2*norm([0,1],p)**2;close(num(evaluate('normed-banach-hilbert',[2,mode,1]),'parallelogram defect on two disjoint unit vectors'),defect);}
});
test('diagonal operator tail bounds have explicit basis witnesses and finite unbounded norms grow',()=>{
 for(const alpha of [.25,1,2])for(const N of [1,4,16]){const r=evaluate('bounded-compact-operators',[1,alpha,N,N]);close(num(r,'operator norm of infinite truncation tail'),(N+1)**(-alpha));const tail=Array.from({length:200},(_,i)=>(N+i+1)**(-alpha));close(Math.max(...tail),num(r,'operator norm of infinite truncation tail'));}
 for(const N of [1,4,16]){const r=evaluate('bounded-compact-operators',[2,1,N,N]);assert.equal(r.stats['finite N operator norm'],N);assert.equal(r.stats['infinite operator norm'],Infinity);assert.equal(r.stats['1{bounded map on all ℓ²}'],0);}
});
test('identity remains noncompact whereas compact diagonal basis images vanish',()=>{
 for(const j of [1,4,16]){const identity=evaluate('bounded-compact-operators',[0,1,j,4]);close(num(identity,'distance between neighboring basis images'),Math.SQRT2);assert.equal(identity.stats['1{compact bounded map on all ℓ²}'],0);const decay=evaluate('bounded-compact-operators',[1,1,j,4]);close(num(decay,'selected basis image norm'),1/j);}
});
test('Riesz pairing, dual norm and spike ratios match direct integration',()=>{
 for(const a of [-2,0,1,2])for(const b of [-1,0,1]){const r=evaluate('dual-functionals-evaluation',[a,b,4]);close(num(r,'functional value integral gf'),integrate(t=>(1+a*t)*(1+b*t)));const gNorm=Math.sqrt(integrate(t=>(1+a*t)**2));close(num(r,'dual norm of ell_g'),gNorm);close(integrate(t=>(1+a*t)**2/gNorm),gNorm);assert.ok(Math.abs(num(r,'functional value integral gf'))<=num(r,'Cauchy-Schwarz upper bound')+1e-12);}
 for(const m of [1,4,16])close(Math.sqrt(integrate(t=>(1-m*t)**2,0,1/m)),num(evaluate('dual-functionals-evaluation',[1,1,m]),'spike L² norm'));
});
test('Legendre residual moments, norms and Pythagoras agree with independent integration',()=>{
 const P=[()=>1,(x:number)=>x,(x:number)=>(3*x*x-1)/2];
 for(const a of [-2,0,1,2])for(const degree of [0,1,2]){const r=legendreProjection(a,degree),f=(x:number)=>x*x+a*x,residual=(x:number)=>f(x)-r.value(x);close(integrate(x=>f(x)**2,-1,1),r.normSquared);close(integrate(x=>residual(x)**2,-1,1),r.errorSquared);close(r.projectionSquared+r.errorSquared,r.normSquared);for(let k=0;k<=degree;k++)close(integrate(x=>residual(x)*P[k](x),-1,1),0);}
});
test('orthogonal projection beats explicit competing polynomial approximations',()=>{
 for(const a of [-1,1])for(const degree of [0,1,2]){const r=legendreProjection(a,degree);for(const perturb of [-2,-.1,.3,2]){const competitor=(x:number)=>r.value(x)+perturb*(degree===0?1:x);close(integrate(x=>(x*x+a*x-competitor(x))**2,-1,1),r.errorSquared+perturb**2*(degree===0?2:2/3));}}
});
test('symmetric spectral decompositions satisfy independent characteristic equations and reconstruction',()=>{
 for(const a of [-2,0,2])for(const b of [-1,1])for(const c of [-1,0,.5]){const r=spectralExample(0,a,b,c);assert.ok(r.reconstructionError!<1e-12);r.lambda.forEach(lambda=>close((a-lambda)*(b-lambda)-c*c,0));close(r.Q[0]**2+r.Q[2]**2,1);close(r.Q[0]*r.Q[1]+r.Q[2]*r.Q[3],0);}
 const r=evaluate('spectral-theorem-conditions',[0,2,1,.5]),bounds=r.bounds!;close(250/(bounds[1]-bounds[0]),240/(bounds[3]-bounds[2]));
});
test('Jordan defect and normal complex rotation prevent invalid real spectral claims',()=>{
 const jordan=spectralExample(1,1,0,1);assert.equal(jordan.normal,false);assert.equal(jordan.realOrthogonal,false);assert.equal(jordan.reconstructionError,null);close(jordan.commutator,Math.SQRT2);
 const rotation=spectralExample(2,0,0,1);assert.equal(rotation.normal,true);assert.equal(rotation.realOrthogonal,false);assert.deepEqual(rotation.imaginary,[1,-1]);assert.equal(rotation.reconstructionError,null);
 for(const mode of [0,1,2])assert.equal(spectralExample(mode,1,1,0).realOrthogonal,true);
});
test('triangular test actions and derivative signs match independent piecewise quadrature',()=>{
 for(const a of [-1,0,.5,1])for(const epsilon of [.05,.5,1])for(let degree=0;degree<=3;degree++){
  const density=(x:number)=>(1-Math.abs(x-a)/epsilon)/epsilon;
  const moment=integrate(x=>density(x)*x**degree,a-epsilon,a)+integrate(x=>density(x)*x**degree,a,a+epsilon);close(moment,triangleMoment(a,epsilon,degree));
  const derivativeAction=(integrate(x=>x**degree,a-epsilon,a)-integrate(x=>x**degree,a,a+epsilon))/(epsilon*epsilon),r=evaluate('distributions-test-functions',[a,epsilon,degree]);close(derivativeAction,num(r,'triangle derivative action'),1e-9);
 }
});
test('delta approximation preserves mass while peak diverges and quadratic test error has second order',()=>{
 for(const epsilon of [1,.5,.1,.05]){const r=evaluate('distributions-test-functions',[.5,epsilon,2]);assert.equal(r.stats['triangle total mass'],1);close(num(r,'triangle peak height'),1/epsilon);close(num(r,'action approximation error'),epsilon*epsilon/6);}
});
test('weak integration by parts holds on both sides of a corner and below the H1 threshold',()=>{
 for(const alpha of [.25,.5,1,2]){const r=sobolevPower(alpha),left=integrate(t=>4*t**(4*alpha+3)*(1-3*t**8))*2,right=-8*integrate(t=>{const x=t**(4/alpha);return t**3*x*(1-x*x);});close(left,r.weakIdentity,1e-9);close(right,r.weakIdentity,1e-8);assert.equal(r.inH1,alpha>.5);close(r.normSquared,integrate(t=>4*t**(8*alpha+3))*2);}
 assert.equal(evaluate('weak-derivatives-sobolev',[1,1,0]).stats['classical derivative at query'],null);assert.equal(evaluate('weak-derivatives-sobolev',[2,1,0]).stats['classical derivative at query'],0);
});
test('Sobolev L2 derivative membership has a strict threshold and logarithmic boundary divergence',()=>{
 for(const alpha of [.25,.5])assert.equal(sobolevPower(alpha).derivativeSquared,Infinity);
 for(const alpha of [.75,1,2]){const r=sobolevPower(alpha);close(r.derivativeSquared,2*alpha*alpha/(2*alpha-1));}
 const truncated=(epsilon:number)=>.5*Math.log(1/epsilon);assert.ok(truncated(1e-8)>truncated(1e-4));
});
test('rank-one integral action, Hilbert-Schmidt norm and midpoint bias match direct integrals',()=>{
 for(const a of [-2,0,1,2])for(const b of [-1,0,1])for(const N of [1,2,8,16]){const r=rankOneKernel(a,b,N);close(r.exact,integrate(t=>r.g(t)*r.f(t)));close(r.eigenvalue,integrate(t=>r.g(t)**2));close(r.approx-r.exact,-a*b/(12*N*N));close(r.discreteEigenvalue-r.eigenvalue,-a*a/(12*N*N));const hsSquared=integrate(x=>integrate(t=>(r.g(x)*r.g(t))**2,0,1,40),0,1,40);close(Math.sqrt(hsSquared),r.eigenvalue);}
});
test('rank-one Nyström matrix has its claimed eigenvector with quadrature weights included',()=>{
 const r=rankOneKernel(1,1,4),g=r.nodes.map(r.g);for(let i=0;i<4;i++)close(g.reduce((s,gj,j)=>s+r.g(r.nodes[i])*r.g(r.nodes[j])*gj/4,0),r.discreteEigenvalue*g[i]);
});
test('infinite tails satisfy independent integral-test bounds and moving witnesses preserve norm error',()=>{
 for(let N=1;N<=16;N++){const tail=projectionTail(N);assert.ok(tail*tail>=1/(N+1));assert.ok(tail*tail<=1/N);const r=evaluate('finite-infinite-approximations',[N,N+1]);assert.equal(r.stats['selected basis witness error'],1);assert.equal(r.stats['operator norm of I minus P_N'],1);close(num(r,'norm of projected fixed x')**2+tail**2,Math.PI**2/6);if(N>1)assert.ok(tail<projectionTail(N-1));}
});
test('all functional controls reject invalid parameters while legitimate infinite norms remain explicit',()=>{
 for(const kind of functionalKinds){const v=functionalLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[NaN,...v.slice(1)]));const index=functionalLabs[kind].controls.findIndex(c=>c.step!=='any');if(index>=0){const bad=[...v];bad[index]=functionalLabs[kind].controls[index].min+.5;assert.throws(()=>evaluate(kind,bad));}}
 for(const f of [()=>geometricFunction(-1),()=>triangleMoment(0,0,2),()=>rankOneKernel(0,0,0),()=>projectionTail(17),()=>sobolevPower(0),()=>spectralExample(3,0,0,0)])assert.throws(f);
});
for(const kind of functionalKinds)test(`${kind}: authored guides and presets have finite visual geometry`,()=>{
 const spec=functionalLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);
 for(const e of learningGuides[kind].experiments)assert.ok(spec.presets[e.preset]);for(const p of spec.presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.dots?.flat().every(Number.isFinite)??true);for(const value of Object.values(r.stats))if(typeof value==='number')assert.ok(!Number.isNaN(value));assert.ok(!/[가-힣]/.test(r.statement??''));}
});

test('spectral image markers represent Ae1 even when the transformed circle collapses to zero',()=>{
 for(const values of [[2,0,0,0],[0,2,1,.5],[1,1,0,1]]){const r=evaluate('spectral-theorem-conditions',values);assert.deepEqual(r.dots,[[Number(r.table.rows[0][1]),Number(r.table.rows[1][1])],[1,0]]);}
});
