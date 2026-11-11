import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluateApproximation as evaluate,approximationKinds,quadraticApproximation,quadraticApproximationError,reluQuadratic,cholesky,solveCholesky,fitRbf,rbfKernel,fourierFeatureMap,networkJacobian,trainTangentNetwork,normalizedVanishingPolynomial,finiteRiskErrors} from '../src/lib/learning-approximation';
import {approximationLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const dot=(a:number[],b:number[])=>a.reduce((s,x,i)=>s+x*b[i],0);
const integrate=(f:(x:number)=>number,lo=-1,hi=1,n=4000)=>{let sum=f(lo)+f(hi);for(let i=1;i<n;i++)sum+=(i%2?4:2)*f(lo+(hi-lo)*i/n);return sum*(hi-lo)/(3*n);};
const choose=(n:number,k:number)=>{let out=1;for(let i=1;i<=k;i++)out=out*(n-i+1)/i;return out;};
function inverse3(A:number[][]){const [[a,b,c],[d,e,f],[g,h,i]]=A,det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);return [[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]].map(row=>row.map(x=>x/det));}
test('Bernstein formula equals independently summed binomial weights and has exact quadratic error',()=>{
 for(const n of [1,2,4,16])for(const x of [0,.1,.5,.9,1]){let value=0;for(let k=0;k<=n;k++)value+=(k/n)**2*choose(n,k)*x**k*(1-x)**(n-k);close(quadraticApproximation(0,n,x),value);assert.ok(value-x*x<=1/(4*n)+1e-12);}for(const n of [1,4,16])close(quadraticApproximation(0,n,.5)-.25,1/(4*n));
});
test('linear splines interpolate knots and attain their exact error at every midpoint',()=>{
 for(const n of [1,4,16])for(let j=0;j<n;j++){const lo=j/n,hi=(j+1)/n,mid=(lo+hi)/2;close(quadraticApproximation(1,n,lo),lo*lo);close(quadraticApproximation(1,n,mid)-mid*mid,1/(4*n*n));}close(quadraticApproximation(1,4,1),1);
});
test('ReLU hinge network equals direct piecewise linear interpolation and beats the affine barrier',()=>{
 for(const n of [1,2,4,16])for(let j=0;j<n;j++){const lo=-1+2*j/n,hi=-1+2*(j+1)/n,mid=(lo+hi)/2;close(reluQuadratic(n,mid),(lo*lo+hi*hi)/2);close(reluQuadratic(n,mid)-mid*mid,1/(n*n));}assert.ok(1/(4*4)<.5);
 for(const slope of [-1,0,1])for(const offset of [-1,0,.5,1,2])assert.ok(Math.max(Math.abs(1-(offset-slope)),Math.abs(offset),Math.abs(1-(offset+slope)))>=.5);
});
test('finite RKHS reproduction, evaluation norm and Mercer eigenvalues match direct features and integrals',()=>{
 for(const rho of [.2,1,2])for(const x of [-1,0,.5,1]){const a=1,b=-.5,weights=[a,b/Math.sqrt(rho)],section=[1,Math.sqrt(rho)*x];close(dot(weights,section),a+b*x);const r=evaluate('rkhs-mercer-features',[rho,a,b,x]);close(r.stats['target squared RKHS norm (∞ if outside)'] as number,dot(weights,weights));close(r.stats['point evaluation norm'] as number,Math.sqrt(dot(section,section)));close(integrate(t=>(1+rho*x*t))/2,1);close(integrate(t=>(1+rho*x*t)*Math.sqrt(3)*t)/2,(rho/3)*Math.sqrt(3)*x);}
});
test('degenerate RKHS excludes sloped functions instead of assigning a fictitious finite norm',()=>{
 const outside=evaluate('rkhs-mercer-features',[0,1,1,.5]);assert.equal(outside.stats['1{target in RKHS}'],0);assert.equal(outside.stats['target squared RKHS norm (∞ if outside)'],Infinity);assert.equal(outside.stats['reproducing inner product'],null);assert.equal(evaluate('rkhs-mercer-features',[0,1,0,.5]).stats['target squared RKHS norm (∞ if outside)'],1);
});
test('representer complement leaves observations invariant but increases norm and objective exactly',()=>{
 for(const lambda of [.05,1,2])for(const c of [-2,0,1]){const t=1/(2+lambda),w=[t+c/Math.SQRT2,0,t-c/Math.SQRT2],r=evaluate('representer-span-penalty',[lambda,c,0]);for(const x of [-1,1])close(dot(w,[1,x,x*x]),2*t);close(dot(w,w),2*t*t+c*c);close(r.stats['objective increase'] as number,lambda*c*c);const gradient=[0,1,2].map(j=>[-1,1].reduce((s,x)=>s+(dot([t,0,t],[1,x,x*x])-1)*[1,x,x*x][j],0)+2*lambda*[t,0,t][j]);gradient.forEach(x=>close(x,0));}
});
test('Cholesky solves agree with an independent cofactor inverse and reject nonpositive systems',()=>{
 for(const length of [.2,1,2])for(const lambda of [0,.1,1]){const r=fitRbf(length,lambda,.3),inverse=inverse3(r.A),expected=inverse.map(row=>dot(row,r.labels));r.alpha.forEach((x,i)=>close(x,expected[i],1e-9));r.A.forEach((row,i)=>close(dot(row,r.alpha),r.labels[i]));const rebuilt=r.L.map(row=>r.L.map(other=>dot(row,other)));rebuilt.forEach((row,i)=>row.forEach((x,j)=>close(x,r.A[i][j])));}
 assert.throws(()=>cholesky([[1,1],[1,1]]));assert.throws(()=>cholesky([[1,2],[0,1]]));assert.throws(()=>cholesky([[NaN]]));assert.throws(()=>solveCholesky([[1]],[NaN]));
});
test('kernel ridge interpolates at zero regularization and its coefficients satisfy the primal stationarity equation',()=>{
 for(const length of [.2,1,2]){const r=fitRbf(length,0,.3);r.nodes.forEach((x,i)=>close(r.predict(x),r.labels[i]));close(r.smootherTrace,3);}
 for(const lambda of [.1,1]){const r=fitRbf(1,lambda,.3),residual=r.K.map((row,i)=>dot(row,r.alpha)-r.labels[i]);r.K.forEach((row,j)=>close(dot(row,residual)+lambda*dot(row,r.alpha),0));assert.ok(r.smootherTrace<3&&r.smootherTrace>0);assert.ok(r.normSquared>=0);}
});
test('GP conditional means and variances equal independent block-Gaussian Schur complements',()=>{
 for(const length of [.2,1,2])for(const noise of [0,.1,1])for(const query of [-1,-.25,0,1,1.5]){const r=fitRbf(length,noise,.3),inverse=inverse3(r.A),k=r.nodes.map(x=>rbfKernel(query,x,length)),mu=dot(k,inverse.map(row=>dot(row,r.labels))),variance=1-dot(k,inverse.map(row=>dot(row,k)));close(r.predict(query),mu,1e-8);close(r.variance(query),variance,1e-8);assert.ok(r.variance(query)>=0&&r.variance(query)<=1+1e-12);const gp=evaluate('gaussian-process-posteriors',[length,noise,.3,query]);close((gp.stats['new noisy observation variance'] as number)-(gp.stats['latent posterior variance'] as number),noise);}
});
test('GP covariance is label independent and noiseless observed locations have exactly zero variance',()=>{
 const a=fitRbf(1,.1,-1),b=fitRbf(1,.1,1);for(const x of [-1,.2,1.5])close(a.variance(x),b.variance(x));for(const x of [-1,0,1])assert.equal(fitRbf(1,0,0).variance(x),0);
});
test('paired random features form a PSD Gram matrix with unit diagonal and deterministic seed prefixes',()=>{
 for(const m of [1,4,32]){const r=fourierFeatureMap(m,1,1),nodes=[-1,0,1],Z=nodes.map(r.features),weights=[1,-2,1];for(let i=0;i<3;i++){close(dot(Z[i],Z[i]),1);for(let j=0;j<3;j++)close(dot(Z[i],Z[j]),r.kernel(nodes[i],nodes[j]));}const quadratic=nodes.reduce((sum,x,i)=>sum+nodes.reduce((s,z,j)=>s+weights[i]*weights[j]*r.kernel(x,z),0),0);assert.ok(quadratic>=-1e-12);}
 assert.deepEqual(fourierFeatureMap(4,1,1).frequencies,fourierFeatureMap(32,1,1).frequencies.slice(0,4));assert.notDeepEqual(fourierFeatureMap(4,1,1).frequencies,fourierFeatureMap(4,1,2).frequencies);
});
test('ideal Gaussian frequency integral reproduces the RBF while one seeded error need not be monotone',()=>{
 for(const length of [.5,1,2])for(const gap of [.3,1]){const density=(w:number)=>Math.exp(-w*w*length*length/2)*length/Math.sqrt(2*Math.PI);close(integrate(w=>Math.cos(w*gap)*density(w),-10/length,10/length,12000),rbfKernel(gap,0,length),1e-9);}
 const errors=Array.from({length:32},(_,i)=>Math.abs(fourierFeatureMap(i+1,1,1).kernel(0,.5)-rbfKernel(0,.5,1)));assert.ok(errors.some((e,i)=>i>0&&e>errors[i-1]));
});
test('finite tanh Jacobians agree with central finite differences for every parameter type',()=>{
 for(const width of [1,2,8]){const r=trainTangentNetwork(width,.1,0,1);for(const x of [-1,.2,1]){const analytic=networkJacobian(r.initial,x).gradient;analytic.forEach((value,j)=>{const plus=[...r.initial],minus=[...r.initial],h=1e-6;plus[j]+=h;minus[j]-=h;close(value,(networkJacobian(plus,x).value-networkJacobian(minus,x).value)/(2*h),1e-8);});}}
});
test('frozen Jacobian training obeys exact empirical NTK residual dynamics',()=>{
 for(const width of [1,2,8]){const r=trainTangentNetwork(width,.1,4,1);let predicted=r.nodes.map(x=>networkJacobian(r.initial,x).value);for(let t=0;t<4;t++){const residual=predicted.map((x,i)=>x-r.labels[i]);predicted=predicted.map((x,i)=>x-.1*dot(r.K0[i],residual)/3);}predicted.forEach((x,i)=>close(x,r.linearPredict(r.nodes[i])));}
});
test('nonlinear and tangent training coincide at zero steps and differ at second order for one small step',()=>{
 const initial=trainTangentNetwork(2,.1,0,1);assert.equal(initial.history[0].gap,0);assert.equal(initial.history[0].drift,0);const one=trainTangentNetwork(2,.001,1,1).history.at(-1)!,small=trainTangentNetwork(2,.0001,1,1).history.at(-1)!;assert.ok(one.gap>0);assert.ok(small.gap<.02*one.gap);assert.ok(trainTangentNetwork(2,.1,4,1).history.at(-1)!.drift>0);const noUpdate=trainTangentNetwork(8,0,12,16);assert.deepEqual(noUpdate.theta,noUpdate.initial);
});
test('interpolating alternatives vanish at every sample but retain analytically normalized population error',()=>{
 for(const n of [2,3,5,9]){const r=normalizedVanishingPolynomial(n);r.nodes.forEach(x=>assert.ok(r.value(x)===0));close(integrate(x=>r.value(x)**2)/2,1,1e-8);for(const amplitude of [-2,.5,1]){const out=evaluate('expressiveness-sample-efficiency',[n,amplitude,.25]);assert.equal(out.stats['noiseless training MSE'],0);close(integrate(x=>(amplitude*r.value(x))**2)/2,out.stats['population MSE under dx/2'] as number,1e-8);}}
});
test('finite risk decomposition matches direct population and observed-sample losses including signed differences',()=>{
 for(const ones of [0,8,12,16])for(const steps of [0,4,16]){const r=finiteRiskErrors(ones,0,.5,steps),sample=Array.from({length:16},(_,i)=>i<ones?1:-1);close(r.population(r.current),((r.current-1)**2+(r.current+1)**2)/2);close(r.empirical(r.current),sample.reduce((s,x)=>s+(r.current-x)**2,0)/16);close(r.approximation+r.estimation+r.signedOptimization,r.population(r.current));close(r.empiricalOptimization,r.empirical(r.current)-r.empirical(r.empiricalMean));assert.ok(r.empiricalOptimization>=0);}
 const stopped=finiteRiskErrors(12,0,.5,0);assert.equal(stopped.signedOptimization,-.25);assert.equal(stopped.empiricalOptimization,.25);assert.equal(stopped.population(stopped.current),1);
});
test('expected ERM estimation error equals 1/16 under all 2^16 iid binary training sequences',()=>{
 let expected=0;for(let k=0;k<=16;k++)expected+=choose(16,k)/2**16*(2*k/16-1)**2;close(expected,1/16);
});
test('model guards reject invalid lengths, widths, seeds and nonfinite values',()=>{
 for(const kind of approximationKinds){const values=approximationLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));const j=approximationLabs[kind].controls.findIndex(c=>c.step!=='any');if(j>=0){const bad=[...values];bad[j]=approximationLabs[kind].controls[j].min+.5;assert.throws(()=>evaluate(kind,bad));}}
 for(const f of [()=>fitRbf(0,0,0),()=>fourierFeatureMap(0,1,1),()=>fourierFeatureMap(4,1,0),()=>trainTangentNetwork(2,1,1,1),()=>normalizedVanishingPolynomial(1),()=>finiteRiskErrors(17,0,.5,1)])assert.throws(f);
});
for(const kind of approximationKinds)test(`${kind}: authored presets and guides preserve finite diagram geometry`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const e of learningGuides[kind].experiments)assert.ok(approximationLabs[kind].presets[e.preset]);for(const p of approximationLabs[kind].presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.dots?.flat().every(Number.isFinite)??true);for(const value of Object.values(r.stats))if(typeof value==='number')assert.ok(!Number.isNaN(value));assert.ok(!/[가-힣]/.test(r.statement??''));}
});

test('regularized representer and initial NTK plots include observations outside the predicted range',()=>{
 for(const [kind,values] of [['representer-span-penalty',[1,0,0]],['ntk-linearized-training',[2,.1,0,1]]] as const){const r=evaluate(kind,[...values]);assert.ok(r.bounds![3]>=1);for(const [x,y] of r.dots!){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}}
});


test('GP conditional variances retain positive noise and near-observation uncertainty',()=>{
 for(const anchor of [-1,0,1])for(const length of [.2,1,2]){
  const noisy=fitRbf(length,1e-30,0);assert.ok(Math.abs(noisy.variance(anchor)/1e-30-1)<1e-11);
  const near=fitRbf(length,0,0).variance(anchor===0?1e-100:anchor+1e-8);assert.ok(near>0);
  assert.equal(fitRbf(length,0,0).variance(anchor),0);
 }
 assert.ok(fitRbf(.2,0,0).variance(1e-162)>0);
 const under=evaluate('gaussian-process-posteriors',[1,0,0,1e-200]);assert.equal(under.warning,'numeric-range');assert.equal(under.stats['latent posterior variance'],null);assert.equal(under.stats['latent interval half-width (2 sd)'],null);
});

test('RKHS slope cost scales before squaring and separates membership from finite-number range',()=>{
 const small=evaluate('rkhs-mercer-features',[1e-300,0,1e-200,.5]);assert.ok(Math.abs((small.stats['target squared RKHS norm (∞ if outside)'] as number)/1e-100-1)<1e-12);
 for(const values of [[1e-320,0,1,.5],[1,0,1e-200,.5]]){const r=evaluate('rkhs-mercer-features',values);assert.equal(r.stats['1{target in RKHS}'],1);assert.equal(r.stats['target squared RKHS norm (∞ if outside)'],null);assert.equal(r.warning,'numeric-range');}
 assert.equal(evaluate('rkhs-mercer-features',[0,0,1,.5]).stats['target squared RKHS norm (∞ if outside)'],Infinity);
 assert.equal(evaluate('rkhs-mercer-features',[1,0,0,.5]).stats['target squared RKHS norm (∞ if outside)'],0);
});

test('anchored GP conditioning preserves prior variance bounds throughout the stated length and query range',()=>{
 for(const length of [.2,.25,.5,.8,1,1.5,1.9,2])for(const noise of [0,1e-30,.1,1]){
  const model=fitRbf(length,noise,0);
  for(let i=0;i<=120;i++){const variance=model.variance(-1.5+i/40);assert.ok(variance>=0&&variance<=1+1e-12,`${length}, ${noise}, ${i}: ${variance}`);}
 }
});


test('quadratic query errors remain positive when the interpolated height rounds to the target',()=>{
 const x=1-2**-53;
 for(const mode of [0,1]){
  const expected=mode===0?x*2**-57:(x-15/16)*2**-53;
  const result=evaluate('polynomial-spline-approximation',[mode,16,x]);
  assert.ok(Math.abs((result.stats['error at query'] as number)/expected-1)<1e-12);
  assert.ok(quadraticApproximationError(mode,16,x).positive);
  const tiny=evaluate('polynomial-spline-approximation',[mode,16,Number.MIN_VALUE]);assert.equal(tiny.warning,'numeric-range');assert.equal(tiny.stats['error at query'],null);assert.deepEqual(tiny.dots,[]);
  assert.equal(evaluate('polynomial-spline-approximation',[mode,16,1]).stats['error at query'],0);
 }
 const bits=new DataView(new ArrayBuffer(8));
 for(let n=1;n<=16;n++)for(let j=1;j<n;j++){
  const knot=j/n;bits.setFloat64(0,knot);const encoding=bits.getBigUint64(0);
  for(const offset of [BigInt(-1),BigInt(0),BigInt(1)]){bits.setBigUint64(0,encoding+offset);const q=bits.getFloat64(0);assert.ok(quadraticApproximationError(1,n,q).error>=0);}
  assert.equal(quadraticApproximationError(1,n,knot).error,0);
 }
});

test('finite-risk optimization tracks the analytic displacement before rounding the current constant',()=>{
 const initial=.5+2**-53,r=finiteRiskErrors(12,initial,.5,16);
 assert.equal(r.current,.5);assert.equal(r.empiricalOptimization,2**-138);assert.ok(Math.abs(r.signedOptimization/2**-69-1)<1e-12);
 const tiny=evaluate('approximation-estimation-optimization-errors',[8,Number.MIN_VALUE,.5,16]);assert.equal(tiny.warning,'numeric-range');assert.equal(tiny.stats['empirical optimization excess'],null);assert.equal(tiny.stats['signed population difference to ERM'],null);
 const fixed=evaluate('approximation-estimation-optimization-errors',[8,0,.5,16]);assert.equal(fixed.stats['empirical optimization excess'],0);assert.equal(fixed.warning,undefined);
 assert.equal(finiteRiskErrors(12,-.5,0,0).signedOptimization,0); // Equal population risks at opposite constants.
});
