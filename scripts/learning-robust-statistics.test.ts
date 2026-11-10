import test from 'node:test';
import assert from 'node:assert/strict';
import {huberInterval,robustLocation,contaminatedUniform} from '../src/lib/learning-robust-statistics';
const near=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
// Independent piecewise-affine score construction over every knot, not bisection.
function knotsReference(xs:number[],c:number){const knots=[...new Set(xs.flatMap(x=>[x-c,x+c]))].sort((a,b)=>a-b),roots:number[]=[];for(const p of knots){const s=xs.reduce((a,x)=>a+(x-p<-c?-c:x-p>c?c:x-p),0);if(Math.abs(s)<1e-10)roots.push(p);}for(let j=0;j<knots.length-1;j++){const a=knots[j],b=knots[j+1],mid=(a+b)/2;let slope=0,intercept=0;for(const x of xs){if(x-mid>c)intercept+=c;else if(x-mid<-c)intercept-=c;else{slope++;intercept+=x;}}if(slope){const root=intercept/slope;if(root>=a-1e-12&&root<=b+1e-12)roots.push(root);}else if(intercept===0)roots.push(a,b);}assert.ok(roots.length);return[Math.min(...roots),Math.max(...roots)];}
test('3080 replacement settings agree with an independent knot solver and exact lattice sums',()=>{
 let cases=0;
 for(let m=0;m<=21;m++)for(const z of [-100,-3,-1,0,1,3,100])for(const k of [0,1,5,9])for(const c of [.25,.5,1,2.5,4]){
  const a=robustLocation(m,z,k,c),ref=knotsReference(a.sample,c);
  near(a.mean,a.meanFormula);near(a.interval[0],ref[0]);near(a.interval[1],ref[1]);near(a.scoreResidual,0,1e-9);
  assert.equal(a.replaced.filter(Boolean).length,m);const ordered=a.sample.slice().sort((a,b)=>a-b);assert.equal(a.median,ordered[10]);
  for(const t of [-10,-1,0,1,10]){const b=huberInterval(a.sample.map(x=>x+t),c);near(b[0],a.interval[0]+t);near(b[1],a.interval[1]+t);}
  cases++;
 }assert.equal(cases,3080);
});
test('Huber solver exposes a genuine interval and is equivariant when the threshold scales',()=>{
 assert.deepEqual(huberInterval([-3,3],1),[-2,2]);
 for(const xs of [[-2,0,1],[-3,3],[-1,-1,2,4]])for(const c of [.25,1,3])for(const scale of [.5,2,10]){
  const a=huberInterval(xs,c),b=huberInterval(xs.map(x=>x*scale+7),c*scale);near(b[0],a[0]*scale+7);near(b[1],a[1]*scale+7);
 }
});
test('analytic contamination guides show small-sample resistance and majority breakdown',()=>{
 const one=robustLocation(1,100,1,2.5);assert.equal(one.meanExact.label,'14/3');assert.equal(one.median,0);assert.equal(one.trimmed,0);near(one.huber,1/40);
 const ten=robustLocation(10,100,1,2.5),eleven=robustLocation(11,100,1,2.5);assert.equal(ten.median,0);near(ten.huber,8/5);assert.equal(eleven.median,100);near(eleven.huber,1075/11);
 assert.equal(robustLocation(10,-100,0,1).median,-2);assert.equal(robustLocation(11,-100,0,1).median,-100);
});
test('exact accumulation preserves symmetric zero without erasing small nonzero estimates',()=>{
 for(let k=0;k<=9;k++){const r=robustLocation(0,0,k,1);assert.equal(r.mean,0);assert.equal(r.trimmed,0);assert.equal(r.meanExact.label,'0');assert.equal(r.trimmedExact.label,'0');assert.equal(r.huber,0);assert.deepEqual(r.interval,[0,0]);}
 const r=robustLocation(1,2.25,1,1);assert.equal(r.meanExact.label,'1/84');assert.ok(r.mean>0);
});
test('replacement labels describe the mechanism even when a replaced value is unchanged',()=>{
 const r=robustLocation(1,2,0,1);assert.deepEqual(r.sample,r.clean);assert.equal(r.replaced.filter(Boolean).length,1);assert.equal(r.empiricalFraction,1/21);
 for(let k=0;k<=9;k++)assert.equal(robustLocation(0,0,k,1).replacementBreakdown.trimmed,(k+1)/21);
});
const clip=(x:number,c:number)=>Math.max(-c,Math.min(c,x));
test('140 population mixtures satisfy independently integrated scores and median CDF inequalities;20 derivatives match',()=>{
let cases=0;
// Independent midpoint integration of the unclipped population score; exact median CDF inequalities.
for(const epsilon of [0,.01,.1,.25,.49])for(const z of [-100,-1,-.25,0,.25,1,100])for(const c of [.25,.5,1,2]){const r=contaminatedUniform(epsilon,z,c);near(r.scoreResidual,0);const n=4000;let expected=0;for(let i=0;i<n;i++)expected+=clip(-1+2*(i+.5)/n-r.huber,c)/n;expected=(1-epsilon)*expected+epsilon*clip(z-r.huber,c);near(expected,0,1e-7);const cleanCDF=(x:number)=>Math.max(0,Math.min(1,(x+1)/2)),below=(1-epsilon)*cleanCDF(r.median)+(z<r.median?epsilon:0),at=(1-epsilon)*cleanCDF(r.median)+(z<=r.median?epsilon:0);assert.ok(below<=.5+1e-12&&at>=.5-1e-12);const mirrored=contaminatedUniform(epsilon,-z,c);near(r.huber,-mirrored.huber);near(r.median,-mirrored.median);cases++;}
for(const z of [-10,-.25,0,.25,10])for(const c of [.25,.5,1,2]){const e=1e-7,r=contaminatedUniform(e,z,c);near(r.huber/e,r.influence.huber,3e-6);near(r.median/e,r.influence.median,3e-6);}
assert.equal(cases,140);
});
test('finite contamination bias differs from the first-order influence approximation',()=>{
 const r=contaminatedUniform(.25,100,.5);near(r.mean,25);near(r.median,1/3);near(r.firstOrder.median,.25);assert.notEqual(r.median,r.firstOrder.median);assert.ok(r.huber>r.firstOrder.huber);
 for(const c of [.25,.5,1,2,4]){near(contaminatedUniform(0,100,c).huber,0);near(contaminatedUniform(.25,0,c).huber,0);near(contaminatedUniform(.1,100,c).cleanAsymptoticVariance.huber,c<1?1-2*c/3:1/3);}
});
test('known-scale Huber influence uses its derivative denominator rather than the clipped score alone',()=>{
 const r=contaminatedUniform(.01,100,.25);assert.equal(r.influence.huber,1);assert.equal(r.influence.median,1);assert.equal(r.influence.mean,100);
 assert.equal(contaminatedUniform(.01,100,2).influence.huber,2);
});
test('all control boundaries are validated and output arrays do not alias later calls',()=>{
 for(const bad of [NaN,Infinity,-1,22,.5])assert.throws(()=>robustLocation(bad,0,0,1));
 for(const bad of [NaN,Infinity,-1,10,.5])assert.throws(()=>robustLocation(0,0,bad,1));
 for(const bad of [NaN,Infinity,-100.25,100.25,.1]){assert.throws(()=>robustLocation(0,bad,0,1));assert.throws(()=>contaminatedUniform(.1,bad,1));}
 for(const bad of [NaN,Infinity,0,-1,.1,4.25]){assert.throws(()=>robustLocation(0,0,0,bad));assert.throws(()=>contaminatedUniform(.1,0,bad));}
 for(const bad of [NaN,Infinity,-.1,.5,1])assert.throws(()=>contaminatedUniform(bad,0,1));
 for(const xs of [[],[NaN],[Infinity],[1e7]])assert.throws(()=>huberInterval(xs,1));
 for(const bad of [0,NaN,Infinity,1e-7,1e7])assert.throws(()=>huberInterval([0],bad));
 const a=robustLocation(0,0,0,1);a.sample[0]=999;a.clean[0]=999;a.replaced[0]=true;const b=robustLocation(0,0,0,1);assert.equal(b.sample[0],-2);assert.equal(b.clean[0],-2);assert.equal(b.replaced[0],false);
});

test('sample display adapter decodes quarter controls and preserves the analytic guides',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 for(const [m,wantMedian,wantHuber]of [[1,0,1/40],[10,0,8/5],[11,100,1075/11]]){
  const out=evaluatePrivacy('replacement-contamination-location',[m,400,1,10]);assert.equal(out.stats['replacement location z'],100);assert.equal(out.stats['fixed Huber threshold c'],2.5);assert.equal(out.stats['sample median'],wantMedian);near(out.stats['Huber sample location'] as number,wantHuber);assert.equal(out.table.rows.length,21);assert.ok(out.curves.flat(2).every(Number.isFinite));assert.equal(out.robustSample!.replaced.filter(Boolean).length,m);
 }
});
test('population display adapter separates finite bias, local influence and first-order approximation',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 const a=evaluatePrivacy('influence-and-breakdown',[25,400,2]),b=evaluatePrivacy('influence-and-breakdown',[25,400,8]);near(a.stats['population Huber bias'] as number,1/3);near(a.stats['Huber first-order bias'] as number,1/4);near(b.stats['population Huber bias'] as number,2/3);near(b.stats['Huber first-order bias'] as number,1/2);assert.equal(a.table.rows.length,3);assert.ok(a.curves.flat(2).every(Number.isFinite));assert.notDeepEqual(a.curves,b.curves);
});
test('robust display adapters reject missing vectors and fractional encoded controller values',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 for(const id of ['replacement-contamination-location','influence-and-breakdown'] as const){assert.throws(()=>evaluatePrivacy(id,[]));const values=id==='replacement-contamination-location'?[1,400,1,10]:[1,400,2];for(let i=0;i<values.length;i++){const v=values.slice();v[i]+=.5;assert.throws(()=>evaluatePrivacy(id,v));}}
});
test('every robust plot state stays inside its declared numerical bounds',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 for(const ep of [0,1,25,49])for(const z of [-400,0,400])for(const c of [1,2,8,16]){
  const out=evaluatePrivacy('influence-and-breakdown',[ep,z,c]),[xmin,xmax,ymin,ymax]=out.bounds!;assert.ok([...out.curves.flat(),...out.dots!].every(([x,y])=>x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax));assert.ok(Object.values(out.stats).every(v=>typeof v==='number'&&Number.isFinite(v)));
 }
});
