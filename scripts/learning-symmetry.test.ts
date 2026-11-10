import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {symmetryKinds,evaluateSymmetry as evaluate,dihedralMatrix,dihedralProduct,dihedralOrbit,rotation2,equivarianceExample,cyclicDecomposition,hat3,expSO3,rotationCoefficients,expTwist,axisRotation,composePoses,applyPose,inversePose,determinant3,lieCommutator,poseOrder,gaugeExample,reynoldsProjection,gramFeatures,tensorRepresentation,multiplyMatrices,type Vector3,type Matrix} from '../src/lib/learning-symmetry';
import {symmetryLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const vector=(a:number[],b:number[],tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((x,i)=>close(x,b[i],tol));};
const matrix=(a:Matrix,b:Matrix,tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((row,i)=>vector(row,b[i],tol));};
const identity=(n:number)=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>Number(i===j)));
const transpose=(A:Matrix)=>A[0].map((_,j)=>A.map(r=>r[j]));
const apply=(A:Matrix,x:number[])=>A.map(row=>row.reduce((s,v,i)=>s+v*x[i],0));
function product(A:Matrix,B:Matrix){const out=Array.from({length:A.length},()=>Array(B[0].length).fill(0));for(let k=0;k<B.length;k++)for(let j=0;j<B[0].length;j++)for(let i=0;i<A.length;i++)out[i][j]+=A[i][k]*B[k][j];return out;}
function matrixExp(A:Matrix){const n=A.length,sum=identity(n);let term=identity(n);for(let k=1;k<=140;k++){term=product(term,A).map(row=>row.map(x=>x/k));for(let i=0;i<n;i++)for(let j=0;j<n;j++)sum[i][j]+=term[i][j];}return sum;}
const cross=(a:number[],b:number[])=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a:number[],b:number[])=>a.reduce((s,v,i)=>s+v*b[i],0);
test('all D4 products match independent matrices, inverses and associativity',()=>{
 const elements=Array.from({length:8},(_,i)=>[i%4,Math.floor(i/4)] as [number,number]);for(const g of elements){matrix(product(transpose(dihedralMatrix(...g)),dihedralMatrix(...g)),identity(2));assert.ok(elements.some(h=>dihedralProduct(...g,...h).join(',')==='0,0'));for(const h of elements){const gh=dihedralProduct(...g,...h);matrix(product(dihedralMatrix(...g),dihedralMatrix(...h)),dihedralMatrix(...gh));for(const k of elements)assert.deepEqual(dihedralProduct(...gh,...k),dihedralProduct(...g,...dihedralProduct(...h,...k)));}}
 assert.notDeepEqual(dihedralProduct(1,0,0,1),dihedralProduct(0,1,1,0));
});
test('D4 orbit and stabilizer counts agree with independent point images and action composition',()=>{
 for(const mode of [0,1,2,3])for(let k=0;k<4;k++)for(const b of [0,1]){const r=dihedralOrbit(mode,k,b);assert.equal(r.unique.length,[1,4,4,8][mode]);assert.equal(r.stabilizer,[8,2,2,1][mode]);assert.equal(r.unique.length*r.stabilizer,8);close(dot(r.selected,r.selected),dot(r.point,r.point));const g=dihedralProduct(k,b,3,1);vector(apply(dihedralMatrix(k,b),apply(dihedralMatrix(3,1),r.point)),apply(dihedralMatrix(...g),r.point));}
});
test('equivariant SO2 maps commute for all sampled angles while anisotropy has the independently derived defect',()=>{
 for(const theta of [0,.2,Math.PI/2,Math.PI,5.7,2*Math.PI])for(const a of [-2,0,1])for(const b of [-1,0,.5]){close(equivarianceExample(theta,a,b,0).error,0);for(const c of [-1,.5,1]){const r=equivarianceExample(theta,a,b,c);close(r.error,2*Math.abs(c*Math.sin(theta))*Math.sqrt(1.25));close(r.invariantGap,0);assert.equal(r.global,false);}}
 assert.equal(equivarianceExample(Math.PI,1,0,.5).error,0);assert.equal(equivarianceExample(Math.PI,1,0,.5).global,false);
});
test('real cyclic decomposition reconstructs all shifts and preserves orthogonal component energy',()=>{
 for(const x of [[1,0,0,0],[0,1,0,0],[1,1,1,1],[1,-1,1,-1],[.2,-.7,1.3,-2]])for(let k=0;k<4;k++){const r=cyclicDecomposition(x,k);vector(r.reconstructed,x.map((_,n)=>x[(n-k+4)%4]));close(dot(r.coefficients,r.coefficients),dot(x,x));close(dot(r.transformed,r.transformed),dot(x,x));}
 const impulse=cyclicDecomposition([1,0,0,0],1);vector(impulse.coefficients,[.5,.5,Math.SQRT1_2,0]);vector(impulse.transformed,[.5,-.5,0,Math.SQRT1_2]);
});
test('real quarter-turn plane has characteristic polynomial lambda squared plus one and complex eigenlines',()=>{
 const J=rotation2(Math.PI/2);for(const lambda of [-2,-1,0,1,2])close((J[0][0]-lambda)*(J[1][1]-lambda)-J[0][1]*J[1][0],lambda*lambda+1);
 // J(1,i)=(-i,1)=(-i)(1,i); real and imaginary parts are verified separately.
 vector(apply(J,[1,0]),[0,1]);vector(apply(J,[0,1]),[-1,0]);matrix(product(J,J),[[-1,0],[0,-1]]);
});
test('hat operators implement cross products and their commutator equals the hat of the cross product',()=>{
 for(const a of [[1,0,0],[.3,-.7,1.2]] as Vector3[])for(const b of [[0,1,0],[1,.4,-.2]] as Vector3[]){vector(apply(hat3(a),b),cross(a,b));matrix(hat3(a).map((row,i)=>row.map((x,j)=>x+hat3(a)[j][i])),Array.from({length:3},()=>[0,0,0]));const AB=product(hat3(a),hat3(b)),BA=product(hat3(b),hat3(a));matrix(AB.map((row,i)=>row.map((x,j)=>x-BA[i][j])),hat3(cross(a,b) as Vector3));}
});
test('Rodrigues agrees with independently summed matrix exponentials and produces proper rotations',()=>{
 for(const w of [[0,0,0],[1e-10,-2e-10,3e-10],[.4,-.7,1.2],[0,0,Math.PI/2],[0,0,2*Math.PI],[2*Math.PI,-2*Math.PI,2*Math.PI]] as Vector3[]){const R=expSO3(w);matrix(R,matrixExp(hat3(w)),3e-11);matrix(product(transpose(R),R),identity(3),1e-12);close(determinant3(R),1,1e-12);vector(apply(expSO3(w.map(x=>-x) as Vector3),apply(R,[.3,-.2,1])),[.3,-.2,1],1e-12);}
});
test('small-angle coefficients remain regular at zero and retain first nontrivial terms',()=>{
 assert.deepEqual(rotationCoefficients(0),{A:1,B:.5,C:1/6});for(const theta of [1e-12,1e-8,1e-5]){const c=rotationCoefficients(theta);close(c.A,1-theta*theta/6,1e-15);close(c.B,.5-theta*theta/24,1e-15);close(c.C,1/6-theta*theta/120,1e-15);}
 matrix(expSO3([0,0,2*Math.PI]),identity(3));
});
test('scaled group commutator converges to its bracket while zero and precision-limited cases stay distinct',()=>{
 const values=[.1,.05,.025].map(h=>lieCommutator(1,1,h));assert.ok(values[1].error!<.51*values[0].error!);assert.ok(values[2].error!<.51*values[1].error!);close(values[0].bracket[1][0],1);close(values[0].scaled![1][0],.996646232657051);assert.equal(lieCommutator(1,1,0).scaled,null);assert.equal(lieCommutator(1,1,0).numericRange,false);assert.equal(lieCommutator(1,1,1e-8).numericRange,true);assert.equal(evaluate('lie-brackets-commutators',[1,1,1e-8]).precisionNote,true);matrix(lieCommutator(1,0,.1).K,identity(3));
});
test('SE3 composition equals sequential point actions, homogeneous products and inverse roundtrips',()=>{
 const A={R:axisRotation(2,.7),t:[1,-.5,.3] as Vector3},B={R:axisRotation(0,-.4),t:[-.3,.7,1] as Vector3},x:Vector3=[.2,-1,.5],AB=composePoses(A,B);vector(applyPose(AB,x),applyPose(A,applyPose(B,x)));vector(applyPose(inversePose(AB),applyPose(AB,x)),x);const H=(P:typeof A)=>[...P.R.map((row,i)=>[...row,P.t[i]]),[0,0,0,1]];matrix(H(AB),product(H(A),H(B)));vector(composePoses(A,inversePose(A)).t,[0,0,0]);
});
test('rigid quarter-turn order and pure translation have independently worked xyz answers',()=>{
 vector(poseOrder(Math.PI/2,Math.PI/2,[1,0,0],0).result,[1,1,0]);vector(poseOrder(Math.PI/2,Math.PI/2,[1,0,0],1).result,[1,0,1]);close(poseOrder(Math.PI/2,Math.PI/2,[1,0,0],0).gap,Math.SQRT2);vector(poseOrder(0,0,[1,2,-1],0).result,[2,2,-1]);
});
test('twist exponential matches an independent four by four power series for arbitrary 3D generators',()=>{
 for(const w of [[0,0,0],[.4,-.7,1.2],[1e-8,-2e-8,3e-8],[0,0,Math.PI/2]] as Vector3[])for(const time of [0,.3,1]){const v:Vector3=[.7,-.4,.2],H=[...hat3(w).map((row,i)=>[...row,v[i]]),[0,0,0,0]].map(row=>row.map(x=>time*x)),expected=matrixExp(H),actual=expTwist(w,v,time);matrix([...actual.R.map((row,i)=>[...row,actual.t[i]]),[0,0,0,1]],expected,1e-11);}
});
test('twist translations equal a separately integrated rotating velocity and one-parameter subgroup composition',()=>{
 const w:Vector3=[.4,-.7,1.2],v:Vector3=[.7,-.4,.2],time=.8,N=1000,h=time/N,integral=[0,0,0];for(let j=0;j<=N;j++){const s=j*h,R=matrixExp(hat3(w).map(row=>row.map(x=>x*s))),value=apply(R,v),weight=j===0||j===N?1:j%2?4:2;value.forEach((x,i)=>integral[i]+=weight*x*h/3);}vector(expTwist(w,v,time).t,integral,1e-10);const combined=composePoses(expTwist(w,v,.3),expTwist(w,v,.4)),one=expTwist(w,v,.7);matrix(combined.R,one.R);vector(combined.t,one.t);
});
test('spatial-twist derivative matches Omega R and Omega p plus v rather than an uncoupled straight path',()=>{
 const w:Vector3=[.4,-.7,1.2],v:Vector3=[.7,-.4,.2],time=.4,h=1e-6,middle=expTwist(w,v,time),plus=expTwist(w,v,time+h),minus=expTwist(w,v,time-h),expected=apply(hat3(w),middle.t).map((x,i)=>x+v[i]);vector(plus.t.map((x,i)=>(x-minus.t[i])/(2*h)),expected,1e-9);matrix(plus.R.map((row,i)=>row.map((x,j)=>(x-minus.R[i][j])/(2*h))),product(hat3(w),middle.R),1e-9);
});
test('twist quarter-turn, zero-angle and full-turn translation limits have closed-form references',()=>{
 vector(expTwist([0,0,Math.PI/2],[1,0,0],1).t,[2/Math.PI,2/Math.PI,0]);vector(expTwist([0,0,0],[1,-.5,.3],1).t,[1,-.5,.3]);vector(expTwist([0,0,2*Math.PI],[1,0,0],1).t,[0,0,0]);vector(evaluate('generators-twist-exponentials',[0,0,0,1,0,2,1]).stats['exponential translation xyz'] as number[],[1,0,2]);matrix(expTwist([0,0,2*Math.PI],[0,0,0],1).R,identity(3));
});
test('local gauge transport cancels endpoint frames and the wrong fixed-link message fails',()=>{
 for(const phis of [[0,0,0],[Math.PI/2,0,-Math.PI/2],[.2,-.8,1.1]])for(const edge of [0,.4,Math.PI/2]){const r=gaugeExample(phis,edge);vector(r.newMessage,apply(rotation2(phis[0]),r.message));close(r.error,0);close(r.tracePrime,r.trace);close(r.loopError,0);r.gauged.forEach((U,i)=>matrix(U,rotation2(r.angles[i])));close(r.changed.reduce((s,x)=>s+dot(x,x),0),r.features.reduce((s,x)=>s+dot(x,x),0));}
 close(gaugeExample([Math.PI/2,0,0],Math.PI/2).naiveError,Math.sqrt(2.5));
});
test('pure local gauge links can be nonidentity while the Wilson loop remains identity',()=>{
 const r=gaugeExample([Math.PI/2,Math.PI,-Math.PI/2],0);assert.ok(r.angles.some(x=>x!==0));matrix(r.loopPrime,identity(2));close(r.tracePrime,2);close(r.error,0);
});
test('Reynolds averages match independent commutant formulas and fix every allowed map',()=>{
 for(const W of [[[1,2],[0,1]],[[.3,-.7],[1.2,-1]],[[1,0],[0,-1]]]){const a=(W[0][0]+W[1][1])/2,b=(W[1][0]-W[0][1])/2;matrix(reynoldsProjection(W,0),[[a,-b],[b,a]]);matrix(reynoldsProjection(W,1),[[a,0],[0,a]]);for(const group of [0,1]){const P=reynoldsProjection(W,group);matrix(reynoldsProjection(P,group),P);for(let f=0;f<=group;f++)for(let k=0;k<4;k++){const g=dihedralMatrix(k,f);matrix(product(P,g),product(g,P));}}}
});
test('Reynolds residual is Frobenius orthogonal to the full allowed matrix space and satisfies Pythagoras',()=>{
 const W=[[.3,-.7],[1.2,-1]];for(const group of [0,1]){const P=reynoldsProjection(W,group),residual=W.map((row,i)=>row.map((x,j)=>x-P[i][j])),basis=group===0?[identity(2),rotation2(Math.PI/2)]:[identity(2)];for(const Z of basis)close(dot(residual.flat(),Z.flat()),0);const Z=group===0?[[2,1],[-1,2]]:[[2,0],[0,2]],distance=(A:Matrix,B:Matrix)=>A.flat().reduce((s,x,i)=>s+(x-B.flat()[i])**2,0);close(distance(W,Z),distance(W,P)+distance(P,Z));}
});
test('an arbitrary pointwise nonlinearity can break rotational equivariance after an allowed linear map',()=>{
 const x=[1,-1],R=rotation2(Math.PI/2),relu=(v:number[])=>v.map(x=>Math.max(0,x));assert.notDeepEqual(relu(apply(R,x)),apply(R,relu(x)));vector(relu(apply(R,x)),[1,1]);vector(apply(R,relu(x)),[0,1]);
});
test('Gram invariants and signed volume distinguish proper rotations, reflections and near-degenerate orientation',()=>{
 for(const h of [-1,0,1,1e-20])for(const yaw of [.2,Math.PI/2])for(const mirror of [0,1]){const r=gramFeatures(yaw,.7,mirror,h);matrix(r.gramPrime,r.gram,1e-12);close(r.volume,h);close(r.volumePrime,(mirror?-1:1)*h);close(determinant3(r.gram),h*h,1e-12);assert.equal(r.orientation,h===0?null:(mirror?-1:1)*Math.sign(h));assert.equal(r.rank,h===0?2:3);}
 const proper=gramFeatures(.2,.7,0,1),mirror=gramFeatures(.2,.7,1,1);matrix(proper.gramPrime,mirror.gramPrime);assert.equal(proper.volumePrime,-mirror.volumePrime);
});
test('rank-two tensor action obeys the double-angle and reflection laws, preserves quadratic forms and spectrum',()=>{
 for(const [p,q,r] of [[2,0,0],[1,1,-1],[-1,.3,2],[1,0,1]])for(const theta of [0,.2,Math.PI/4,Math.PI/2,4.7])for(const reflection of [0,1]){const out=tensorRepresentation(p,q,r,theta,reflection);vector(out.actual,out.predicted,1e-12);close(out.trace,out.tracePrime);close(out.det,out.detPrime,1e-12);close(dot(out.T.flat(),out.T.flat()),dot(out.transformed.flat(),out.transformed.flat()),1e-12);const x=[.3,-.8],gx=apply(out.g,x);close(dot(gx,apply(out.transformed,gx)),dot(x,apply(out.T,x)),1e-12);close(out.eigenvalues[0]+out.eigenvalues[1],out.trace);close(out.eigenvalues[0]*out.eigenvalues[1],out.det);}
});
test('tensor double-angle representation restricts to minus identity for a C4 quarter-turn generator',()=>{
 for(const [a,b] of [[1,0],[0,1],[.3,-.7]]){const out=tensorRepresentation(a,b,-a,Math.PI/2,0);vector(out.actual,[-a,-b]);}const isotropic=tensorRepresentation(1,0,1,.7,1);matrix(isotropic.transformed,identity(2));close(tensorRepresentation(1,1,-1,0,0).eigenvalues[1],-Math.SQRT2);
});
test('symmetry model domains reject nonfinite values, incorrect lengths and fractional integer choices',()=>{
 for(const kind of symmetryKinds){const spec=symmetryLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let j=0;j<initial.length;j++){for(const x of [NaN,Infinity,spec.controls[j].max+1]){const bad=[...initial];bad[j]=x;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[j].step!=='any'){const bad=[...initial];bad[j]=spec.controls[j].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
 for(const f of [()=>multiplyMatrices([[1,2]],[[1,2]]),()=>multiplyMatrices([[NaN]],[[1]]),()=>expSO3([Infinity,0,0]),()=>gaugeExample([0,0],0),()=>cyclicDecomposition([1,2],0),()=>reynoldsProjection([[1,2]],0)])assert.throws(f);
});
for(const kind of symmetryKinds)test(`${kind}: guided presets keep finite geometry, bounds and true 3D frames`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const g of learningGuides[kind].experiments)assert.ok(symmetryLabs[kind].presets[g.preset]);
 for(const p of symmetryLabs[kind].presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[]),...(r.holes??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}if(r.frame3D)for(const vector of [...r.frame3D.blue,...r.frame3D.orange,...(r.frame3D.orangeOrigin?[r.frame3D.orangeOrigin]:[])]){assert.equal(vector.length,3);assert.ok(vector.every(Number.isFinite));}assert.ok(!/[가-힣]/.test(r.statement??''));for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of symmetryKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: detailed theory, valid equations, explained exercise and primary reading`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
