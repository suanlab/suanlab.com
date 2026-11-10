import test from 'node:test';
import assert from 'node:assert/strict';
import {affineLab,reluLab,type V} from '../src/lib/learning-adversarial';
const near=(a:number,b:number,tol=1e-12)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
test('5832 affine settings attain the dual bound with feasible witnesses and independent feasible directions',()=>{
 let cases=0;for(const w0 of [-2,0,2])for(const w1 of [-2,0,2])for(const x0 of [-4,0,4])for(const x1 of [-4,0,4])for(const b of [-4,0,4])for(const ep of [0,2,4,8])for(const norm of [0,1,2])for(const y of [-1,1]){
  const r=affineLab(w0,w1,x0,x1,b,ep,norm,y),d=r.delta,primal=norm===0?Math.abs(d[0])+Math.abs(d[1]):norm===1?Math.hypot(...d):Math.max(Math.abs(d[0]),Math.abs(d[1]));assert.ok(primal<=r.eps+1e-12);near(y*(w0*(r.x[0]+d[0])+w1*(r.x[1]+d[1])+r.b),r.worstSignedMargin);
  for(let k=0;k<24;k++){const angle=k*Math.PI/12;let a=Math.cos(angle),b=Math.sin(angle);const length=norm===0?Math.abs(a)+Math.abs(b):norm===1?Math.hypot(a,b):Math.max(Math.abs(a),Math.abs(b));a=a*r.eps/length;b=b*r.eps/length;assert.ok(y*(w0*(r.x[0]+a)+w1*(r.x[1]+b)+r.b)>=r.worstSignedMargin-1e-12);}cases++;
 }assert.equal(cases,5832);
});
test('exact radius ties preserve the declared positive tie convention separately from strict certification',()=>{
 const a=affineLab(1,0,4,0,0,4,1,1),b=affineLab(1,0,-4,0,0,4,1,-1);for(const r of [a,b]){assert.equal(r.atWorstTie,true);assert.equal(r.strictMarginCertificate,false);assert.equal(r.algebraicWorstPrediction,1);assert.equal(r.robustRadius,1);}assert.equal(a.predictionRobust,true);assert.equal(b.predictionRobust,false);assert.equal(a.robustRadiusBoundaryIncluded,true);assert.equal(b.robustRadiusBoundaryIncluded,false);
});
test('radical sign decisions use squared integer comparisons across all integer weight pairs',()=>{
 for(let a=-4;a<=4;a++)for(let b=-4;b<=4;b++)for(let radius=0;radius<=8;radius++)for(let bias=-16;bias<=16;bias++){
  const r=affineLab(a,b,0,0,bias,radius,1,1),expected=bias<0?-1:Math.sign(bias*bias-radius*radius*(a*a+b*b));assert.equal(r.exactWorstSign,expected);assert.equal(r.predictionRobust,expected>=0);
 }
});
test('zero weights, already wrong points and optimizer nonuniqueness have explicit meanings',()=>{
 const a=affineLab(0,0,0,0,4,8,1,1),b=affineLab(0,0,0,0,4,8,1,-1);assert.equal(a.radiusInfinite,true);assert.equal(a.robustRadius,null);assert.equal(a.predictionRobust,true);assert.equal(b.radiusUndefined,true);assert.equal(b.predictionRobust,false);assert.equal(b.robustRadius,null);
 const tie=affineLab(1,1,4,4,0,4,0,1);assert.deepEqual(tie.delta,[-1,0]);near(tie.worstSignedMargin,1);assert.equal(affineLab(1,0,-4,0,0,0,2,1).cleanCorrect,false);
});
test('different norm balls of equal numerical radius give different exact worst margins',()=>{
 const r=[0,1,2].map(p=>affineLab(3,4,0,0,16,4,p,1));assert.deepEqual(r.map(x=>x.dual),[4,5,7]);assert.deepEqual(r.map(x=>x.worstSignedMargin),[0,-1,-3]);assert.equal(r[0].predictionRobust,true);
});
// Independent activation-pattern polygon clipping, distinct from pairwise line intersections.
function clippedExtrema(net:ReturnType<typeof reluLab>['net'],center:V,eps:number){
 const value=(p:V)=>net.bias+net.nodes.reduce((s,n)=>s+n.a*Math.max(0,n.w[0]*p[0]+n.w[1]*p[1]-n.h),0),values:number[]=[];
 if(eps===0)return[value(center),value(center)];
 for(let mask=0;mask<2**net.nodes.length;mask++){
  let polygon:V[]=[[center[0]-eps,center[1]-eps],[center[0]+eps,center[1]-eps],[center[0]+eps,center[1]+eps],[center[0]-eps,center[1]+eps]];
  for(const [i,n]of net.nodes.entries()){
   if(!polygon.length)break;const sign=(mask>>i)&1?1:-1,f=(p:V)=>sign*(n.w[0]*p[0]+n.w[1]*p[1]-n.h),out:V[]=[];
   for(let j=0;j<polygon.length;j++){const a=polygon[j],b=polygon[(j+1)%polygon.length],fa=f(a),fb=f(b);if(fa>=0)out.push(a);if((fa<0&&fb>0)||(fb<0&&fa>0)){const t=fa/(fa-fb);out.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}}
   polygon=out;
  }values.push(...polygon.map(value));
 }assert.ok(values.length);return[Math.min(...values),Math.max(...values)];
}
test('675 ReLU settings match independent activation polygons and preserve bound/exact/attack inequalities',()=>{
 let cases=0;for(let mode=0;mode<3;mode++)for(const x of [-1,0,1])for(const y of [-1,0,1])for(const eps of [0,.25,.5,1,2])for(const start of [0,1,2,3,4]){
  const r=reluLab(mode,[x,y],eps,10,start,.25),ref=clippedExtrema(r.net,[x,y],eps);near(r.exactMinimum,ref[0]);near(r.exactMaximum,ref[1]);assert.ok(r.intervalLower<=r.exactMinimum&&r.exactMinimum<=r.attackBest);assert.ok(r.exactMaximum<=r.intervalUpper);assert.ok(r.trajectory.every(p=>Math.abs(p[0]-x)<=eps&&Math.abs(p[1]-y)<=eps));assert.equal(r.trajectory.length,11);cases++;
 }assert.equal(cases,675);
});
test('a center-start zero-gradient attack can fail on an actually nonrobust network',()=>{
 const r=reluLab(0,[0,0],1,10,0,.25);assert.equal(r.attackBest,1);assert.equal(r.exactMinimum,-.5);assert.equal(r.attackFoundMisclassification,false);assert.equal(r.exactPredictionRobust,false);const restart=reluLab(0,[0,0],1,10,2,.25);assert.equal(restart.attackFoundMisclassification,true);assert.equal(restart.attackBest,-.5);
});
test('independent intervals lose shared-activation cancellation, so failed certification need not imply an attack',()=>{
 const r=reluLab(1,[0,0],2,10,0,.25);assert.equal(r.intervalLower,-1);assert.equal(r.exactMinimum,1);assert.equal(r.exactMaximum,1);assert.equal(r.strictCertifiedByInterval,false);assert.equal(r.exactPredictionRobust,true);assert.equal(r.attackFoundMisclassification,false);
});
test('attack values always retain the initial point and best visited point including zero-step and zero-radius cases',()=>{
 for(const steps of [0,1,20])for(const eps of [0,1,2])for(const size of [0,.25,2]){const r=reluLab(2,[0,0],eps,steps,0,size);assert.equal(r.trajectory.length,steps+1);assert.equal(r.attackBest,Math.min(...r.attackMargins));assert.ok(r.attackBest<=r.attackMargins[0]);if(eps===0)assert.ok(r.trajectory.every(p=>p[0]===0&&p[1]===0));}
});
test('all adversarial controls reject unsupported values and returned metadata cannot mutate future results',()=>{
 const vector=[1,1,0,0,4,4,1,1];for(let i=0;i<vector.length;i++)for(const v of [NaN,Infinity,1e6,.5]){const p=vector.slice();p[i]=v;assert.throws(()=>Reflect.apply(affineLab,null,p));}
 for(const v of [-1,3,.5,NaN])assert.throws(()=>reluLab(v,[0,0],1,1,0,.25));for(const c of [[NaN,0],[0,.1],[3,0],[0,0,0]])assert.throws(()=>reluLab(0,c as V,1,1,0,.25));for(const v of [-1,3,.1,NaN,Infinity]){assert.throws(()=>reluLab(0,[0,0],v,1,0,.25));assert.throws(()=>reluLab(0,[0,0],1,1,0,v));}for(const v of [-1,21,.5,NaN])assert.throws(()=>reluLab(0,[0,0],1,v,0,.25));for(const v of [-1,5,.5,NaN])assert.throws(()=>reluLab(0,[0,0],1,1,v,.25));
 const center:V=[0,0],a=reluLab(0,center,1,1,0,.25);center[0]=1;a.net.nodes[0].w[0]=999;a.net.nodes[0].a=999;a.center[0]=999;a.vertices[0][0]=999;const b=reluLab(0,[0,0],1,1,0,.25);assert.equal(b.net.nodes[0].w[0],1);assert.equal(b.net.nodes[0].a,-3);assert.equal(b.center[0],0);assert.equal(b.exactMinimum,-.5);
});

test('affine adapter preserves the three guide results and the exact positive-versus-negative tie distinction',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 const presets=[[3,4,0,0,16,4,2,1],[3,4,0,0,16,4,0,1],[3,4,0,0,-16,4,0,0]];
 presets.forEach((p,i)=>{const r=evaluatePrivacy('linear-norm-adversarial-margins',p);assert.equal(r.stats['worst signed margin'],[-3,0,0][i]);assert.equal(r.stats['1{actual prediction robust including tie rule}'],[0,1,0][i]);assert.equal(r.stats['1{strict positive-margin certificate}'],0);assert.equal(r.table.rows.length,2);assert.ok(r.curves.flat(2).every(Number.isFinite));});
});
test('ReLU adapter separates a failed attack from a failed sufficient certificate',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 const vectors=[[0,0,0,4,10,0,1],[0,0,0,4,10,2,1],[1,0,0,8,10,0,1]];
 vectors.forEach((p,i)=>{const r=evaluatePrivacy('relu-attacks-and-certificates',p);assert.equal(r.stats['exact toy minimum margin'],[-.5,-.5,1][i]);assert.equal(r.stats['best visited attack margin'],[1,-.5,1][i]);assert.equal(r.stats['1{attack finds a negative margin}'],[0,1,0][i]);assert.equal(r.stats['1{exact prediction robust with positive tie}'],[0,0,1][i]);assert.equal(r.table.rows.length,11);});
});
test('attack controls preserve exact model quantities while best-visited plots and bounds remain valid',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 const base=evaluatePrivacy('relu-attacks-and-certificates',[0,0,0,4,0,0,0]).adversarialRelu!;
 for(const steps of [0,1,20])for(const start of [0,1,4])for(const alpha of [0,1,8]){const out=evaluatePrivacy('relu-attacks-and-certificates',[0,0,0,4,steps,start,alpha]),r=out.adversarialRelu!,[xmin,xmax,ymin,ymax]=out.bounds!;assert.equal(r.exactMinimum,base.exactMinimum);assert.equal(r.intervalLower,base.intervalLower);assert.deepEqual(r.vertices,base.vertices);assert.ok(out.curves.flat().every(([x,y])=>x>=xmin&&x<=xmax&&y>=ymin&&y<=ymax));const best=out.curves[0].map(p=>p[1]);assert.ok(best.every((v,i)=>i===0||v<=best[i-1]));}
});
test('adversarial adapters reject incomplete, extra and fractional encoded parameters',async()=>{
 const {evaluatePrivacy}=await import('../src/lib/learning-privacy');
 for(const id of ['linear-norm-adversarial-margins','relu-attacks-and-certificates'] as const){const initial=id==='linear-norm-adversarial-margins'?[3,4,0,0,16,4,2,1]:[0,0,0,4,10,0,1];for(const values of [[],initial.slice(1),[...initial,0]])assert.throws(()=>evaluatePrivacy(id,values));for(let i=0;i<initial.length;i++){const p=initial.slice();p[i]+=.5;assert.throws(()=>evaluatePrivacy(id,p));}}
 assert.throws(()=>evaluatePrivacy('linear-norm-adversarial-margins',[1,1,0,0,0,1,1,2]));
});
