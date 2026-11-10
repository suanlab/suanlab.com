import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {convexKinds,evaluateConvex as evaluate,jensenExample,subgradientExample,boxOptimality,scalarKKT,dualityExample,programExample,proximalExample,regularizationExample,barrierValue,centralPoint,barrierNewton,type ConvexKind} from '../src/lib/learning-convex';
import {convexLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const dot=(a:number[],b:number[])=>a.reduce((s,x,i)=>s+x*b[i],0);
const vector=(a:number[],b:number[],tol=1e-10)=>a.forEach((x,i)=>close(x,b[i],tol));
test('quadratic Jensen gap equals a t(1-t)(x-y)^2 including affine and concave limits',()=>{
 for(const a of [-2,0,.3,2])for(const x of [-2,-.2,1])for(const y of [-1,.7,2])for(const t of [0,.2,.5,1])close(jensenExample(0,a,x,y,t,0).gap!,a*t*(1-t)*(x-y)**2);
 assert.equal(jensenExample(0,1,-1,1,.5,1).combinationInSet,false);assert.equal(jensenExample(0,1,-1,1,.5,1).endpointsInSet,true);
});
test('absolute value is convex without differentiability while log requires positive endpoints',()=>{
 close(jensenExample(1,0,-1,1,.5,0).gap!,1);assert.equal(jensenExample(2,0,-1,1,.75,0).gap,null);assert.equal(jensenExample(2,0,.5,2,.5,0).functionConvex,true);assert.ok(jensenExample(2,0,.5,2,.5,0).gap!>0);
});
test('global subgradient support is distinct from one positive sampled residual',()=>{
 const r=subgradientExample(0,0,1.5,-2);assert.ok(r.queryResidual>0);assert.equal(r.minimumResidual,-Infinity);assert.equal(r.conjugate,Infinity);close(r.distance,.5);
 for(const v of [-1,-.2,1]){const q=subgradientExample(0,0,v,2);close(q.minimumResidual!,0);close(q.distance,0);}
});
test('Fenchel residual minimum agrees with its separately derived quadratic minimizer',()=>{
 for(const gamma of [.2,1,2])for(const v of [-3,-.5,0,1.5,3])for(const u of [-1,0,.4]){const r=subgradientExample(u,gamma,v,1),xmin=Math.sign(v)*Math.max(Math.abs(v)-1,0)/gamma;close(r.f(xmin)-r.line(xmin),r.minimumResidual!);for(const x of [-10,-2,0,2,10])assert.ok(r.f(x)-r.line(x)>=r.minimumResidual!-1e-10);}
});
test('finite conjugate overflow is hidden rather than changed to mathematical infinity',()=>{
 const r=evaluate('subgradient-support-lines',[0,Number.MIN_VALUE,3,1]);assert.equal(r.numericNote,true);assert.equal(r.stats['Fenchel conjugate at slope'],null);assert.ok(r.curves.flat(2).every(Number.isFinite));
});
test('tiny positive multipliers retain finite dual values without overflowing plot bounds',()=>{
 for(const lambda of [1e-160,1.4e-309,Number.MIN_VALUE]){const r=evaluate('duality-slater-counterexamples',[1,lambda,4]);assert.equal(r.numericNote,true);assert.ok(r.bounds!.every(Number.isFinite));assert.ok(r.curves.flat(2).every(Number.isFinite));assert.equal(r.dots!.length,0);if(lambda!==Number.MIN_VALUE)assert.ok(Number.isFinite(r.stats['dual value at multiplier']));else assert.equal(r.stats['dual value at multiplier'],null);}
});
test('normal cone and variational inequality characterize box optima including singleton',()=>{
 for(const a of [-3,-.2,0,.7,3])for(const lower of [-2,-.5,0])for(const upper of [0,.5,2]){const optimum=Math.max(lower,Math.min(upper,a)),r=boxOptimality(a,lower,upper,optimum);assert.equal(r.normalValid,true);close(r.minimumVI!,0);for(const x of [lower,(lower+upper)/2,upper]){const q=boxOptimality(a,lower,upper,x);assert.equal(q.normalValid,x===optimum);assert.equal(q.minimumVI!>=0,x===optimum);assert.ok(q.value>=r.value-1e-12);}}
 assert.equal(boxOptimality(0,-1,1,2).minimumVI,null);assert.equal(boxOptimality(3,0,0,0).normalValid,true);
});
test('KKT optimal pair gives matching primal and dual values over all active-set regimes',()=>{
 for(const a of [-2,-.3,0,2])for(const b of [-1,0,1]){const x=Math.min(a,b),lambda=Math.max(a-b,0),r=scalarKKT(a,b,x,lambda);close(r.stationarity,0);close(r.complementarity,0);close(r.stableGap,0);close(r.dual,r.f(x));}
});
test('gap decomposition agrees algebraically but certifies only feasible pairs',()=>{
 for(const a of [-2,.2,2])for(const b of [-1,1])for(const x of [-2,0,2])for(const lambda of [-2,0,.5,4]){const r=scalarKKT(a,b,x,lambda);close(r.expandedGap,r.stableGap);if(x<=b&&lambda>=0){assert.ok(r.certifiedGap!>=0);assert.ok(r.suboptimality!<=r.certifiedGap!+1e-12);assert.ok(Math.abs(x-r.optimum)<=r.distanceBound!+1e-12);}else assert.equal(r.certifiedGap,null);}
 close(scalarKKT(0,1,0,-1).stableGap,-.5);assert.equal(scalarKKT(2,1,2,1).distanceBound,null);
});
test('zero optimal-value gap and finite multiplier attainment are independent properties',()=>{
 const r=dualityExample(0,1,4);close(r.primal,r.dual!);assert.equal(r.strict,true);
 for(const l of [.1,1,4]){const q=dualityExample(1,l,4);close(q.dual!,-1/(4*l));close(q.witnessX!+l*q.witnessX!**2,q.dual!);assert.equal(q.attained,false);close(q.primal,q.supremum);}assert.equal(dualityExample(1,0,4).dual,-Infinity);
});
test('perspective constraint has a genuine positive duality gap with explicit infimum witnesses',()=>{
 for(const l of [0,1,4]){let previous=Infinity;for(const M of [1,2,4,8,16]){const r=dualityExample(2,l,M);close(r.witnessConstraint!,r.witnessX!**2/r.witnessY);assert.ok(r.witnessConstraint!>0);close(r.witnessL!,Math.exp(-M)+l/(M*M));assert.ok(r.witnessL!<previous);previous=r.witnessL!;close(r.primal-r.supremum,1);assert.equal(r.attained,true);}}
});
test('LP optima and multipliers agree with independently enumerated rational feasible points',()=>{
 for(const c of [[-1,-1],[1,2],[-.3,.7],[0,0]] as [number,number][]){const r=programExample(0,c,[0,0]);close(r.value,r.dual);assert.ok(r.lambda>=0&&r.mu.every(x=>x>=0));vector(c.map((x,i)=>x+r.lambda-r.mu[i]),[0,0]);for(let i=0;i<=20;i++)for(let j=0;j<=20-i;j++)assert.ok(dot(c,[i/20,j/20])>=r.value-1e-12);}
 assert.equal(programExample(0,[-1,-1],[0,0]).unique,false);assert.equal(programExample(0,[-1,0],[0,0]).unique,true);
});
test('QP projection satisfies independent variational inequalities and primal-dual equalities',()=>{
 for(const c of [[-2,-1],[.2,.3],[2,.5],[2,2],[-1,2],[.1,2]] as [number,number][]){const r=programExample(1,c,[0,0]);assert.ok(r.point.every(x=>x>=0));assert.ok(r.point[0]+r.point[1]<=1+1e-12);close(r.value,r.dual);for(let i=0;i<=12;i++)for(let j=0;j<=12-i;j++){const z=[i/12,j/12];assert.ok(dot(r.point.map((x,k)=>x-c[k]),z.map((x,k)=>x-r.point[k]))>=-1e-12);}}
 vector(programExample(1,[2,.5],[0,0]).point,[1,0]);close(programExample(1,[2,.5],[0,0]).value,.625);
});
test('SDP dual matrix is PSD, matches affine trace equations and annihilates optimal slack',()=>{
 for(const c of [[1.2,1.6],[-1,.3],[0,0]] as [number,number][]){const r=programExample(2,c,[0,0]),Z=r.Z!,[a,b]=r.point,F=[[1,a,b],[a,1,0],[b,0,1]];close(2*Z[0][1],c[0]);close(2*Z[0][2],c[1]);close(-Z.reduce((s,row,i)=>s+row[i],0),r.value);for(const v of [[1,0,0],[0,1,-2],[.3,-.7,1]])assert.ok(dot(v,Z.map(row=>dot(row,v)))>=-1e-12);for(let i=0;i<3;i++)for(let j=0;j<3;j++)close(F[i].reduce((s,x,k)=>s+x*Z[k][j],0),0);}
 close(programExample(2,[1.2,1.6],[1,1]).minEigen!,1-Math.SQRT2);assert.equal(programExample(2,[1,0],[1,1]).feasible,false);
});
test('proximal solutions beat independent objective grid searches and satisfy subgradient conditions',()=>{
 for(const mode of [0,1,2])for(const y of [-3,-.3,0,.7,3])for(const l of [0,.5,2])for(const eta of [.1,.7,2]){const r=proximalExample(mode,y,l,eta);assert.ok(r.violation<1e-12);for(let j=0;j<=120;j++){const x=-3+j*.05;assert.ok(r.objective(x)>=r.objective(r.point)-1e-11);}}
 close(proximalExample(2,2,1,.5).point,4/3);assert.equal(proximalExample(1,2,0,1).lo,-Infinity);assert.equal(proximalExample(1,2,0,1).hi,Infinity);
});
// The envelope gradient is 1/eta-Lipschitz: centered differences at a kink have O(h), not O(h²), error.
test('Moreau envelope gradient agrees with independent finite differences across thresholds',()=>{
 for(const mode of [0,1,2])for(const y of [-2,-.5,0,.5,2]){const eta=.7,l=.5,h=1e-6,r=proximalExample(mode,y,l,eta),val=(z:number)=>{const p=proximalExample(mode,z,l,eta);return p.objective(p.point);};close((val(y+h)-val(y-h))/(2*h),r.subgradient,h/(2*eta)+2e-9);}
});
test('all proximal families are firmly nonexpansive including identity limits',()=>{
 for(const mode of [0,1,2])for(const l of [0,.5,2])for(const a of [-3,-.2,1])for(const b of [-1,0,2]){const pa=proximalExample(mode,a,l,.7).point,pb=proximalExample(mode,b,l,.7).point;assert.ok((pa-pb)**2<=(pa-pb)*(a-b)+1e-12);}
});
test('ridge lasso and elastic net have independently worked coefficients and exact zero behavior',()=>{
 vector(regularizationExample(0,[2,.25],1,.5).point,[1,.125]);vector(regularizationExample(1,[2,.25],.5,.5).point,[1.5,0]);vector(regularizationExample(2,[2,.25],.5,.5).point,[1.4,0]);for(const m of [0,1,2])vector(regularizationExample(m,[2,.25],0,.5).point,[2,.25]);
});
test('elastic endpoints coincide with ridge/lasso and every penalty contour has the achieved level',()=>{
 for(const alpha of [0,.3,1])for(const l of [0,.5,2]){const r=regularizationExample(2,[1.4,-.7],l,alpha);assert.ok(r.stationarity<1e-12);for(const p of r.boundary)close(r.penalty(p),r.penalty(r.point));if(alpha===0||alpha===1)vector(r.point,regularizationExample(alpha===0?0:1,[1.4,-.7],l,.5).point);}
});
test('exact central points satisfy reciprocal slack stationarity and the two-over-t gap',()=>{
 for(const c of [-2,-.3,0,.4,2])for(const t of [1,2,16,256]){const r=centralPoint(c,t);assert.ok(r.x>0&&r.x<1);close(c-r.lower+r.upper,0);close(r.lower*r.x,1/t);close(r.upper*(1-r.x),1/t);close(r.gap,2/t);close(centralPoint(-c,t).x,1-r.x);}
});
test('damped Newton preserves strict interior, Armijo decrease and actual step histories',()=>{
 for(const c of [-2,-.4,0,1,2])for(const exponent of [0,4,8])for(const start of [.05,.5,.95]){const r=barrierNewton(c,exponent,start,12);assert.equal(r.history.length,13);for(let i=1;i<r.history.length;i++){const prev=r.history[i-1],now=r.history[i],g=r.t*c-1/prev.x+1/(1-prev.x),H=1/prev.x**2+1/(1-prev.x)**2,dir=-g/H;assert.ok(now.x>0&&now.x<1);close(now.x,prev.x+now.alpha*dir);assert.ok(now.value<=prev.value+.01*now.alpha*g*dir+1e-12);assert.ok(now.value<=prev.value+1e-12);}assert.ok(r.currentGap>=c*r.x-r.optimumValue-1e-12);}
});
test('finite Newton certificate is not mislabeled as exact central gap',()=>{
 const r=barrierNewton(1,8,.5,0);close(r.x,.5);close(r.central.gap,.0078125);close(r.currentGap,.503921508556239);assert.ok(r.currentGap>60*r.central.gap);close(centralPoint(1,1).x,(3-Math.sqrt(5))/2);assert.equal(barrierValue(1,1,0),Infinity);
});
test('feasible distance bound is conservative and invalid signs never receive a certificate',()=>{
 const r=scalarKKT(2,1,0,0);close(r.certifiedGap!,2);close(r.suboptimality!,1.5);close(r.distanceBound!,2);assert.equal(scalarKKT(0,1,0,-1).certifiedGap,null);assert.equal(scalarKKT(2,1,2,1).certifiedGap,null);
});
test('all model domains reject malformed and nonfinite values and fractional choices',()=>{
 for(const kind of convexKinds){const spec=convexLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let j=0;j<initial.length;j++){for(const v of [NaN,Infinity,spec.controls[j].max+1,spec.controls[j].min-1]){const bad=[...initial];bad[j]=v;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[j].step!=='any'){const bad=[...initial];bad[j]=spec.controls[j].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
});
for(const kind of convexKinds)test(`${kind}: meaningful guided presets and finite geometry across control boundaries`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=convexLabs[kind],initial=spec.controls.map(c=>c.initial),values=[...spec.presets.map(p=>p.values),...spec.controls.flatMap((c,i)=>[c.min,c.max].map(x=>initial.map((v,j)=>i===j?x:v)))];
 for(const p of values){const r=evaluate(kind as ConvexKind,p);assert.ok(r.curves.flat(2).every(Number.isFinite),`${kind} ${p}`);for(const [x,y] of [...r.curves.flat(),...(r.dots??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of convexKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed bilingual theory, valid equations and explained exercises`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
