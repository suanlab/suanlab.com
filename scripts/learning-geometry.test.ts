import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {geometryKinds,evaluateGeometry as evaluate,finiteTopology,topologyMap,circleCharts,circlePoint,circlePairing,circleGeodesic,sphereTransport,formField,circleForm,immersionPoint,ripsFiltration,persistentHomology,bettiAt,gf2Rank,finiteBottleneck,rectanglePoints,stabilityExperiment,coordinateMetric,type FilteredSimplex} from '../src/lib/learning-geometry';
import {geometryLabs,learningGuides,learningUnits} from '../src/data/interactive-book';
import {getLearningContent} from '../src/lib/learning-content';
import type {Vec2} from '../src/lib/learning-simulations';
const close=(a:number,b:number,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const dot=(a:number[],b:number[])=>a.reduce((s,x,i)=>s+x*b[i],0);
function integrate(f:(x:number)=>number,lo:number,hi:number,n=4000){let s=f(lo)+f(hi);for(let i=1;i<n;i++)s+=(i%2?4:2)*f(lo+(hi-lo)*i/n);return s*(hi-lo)/(3*n);}
function graphComponents(adj:boolean[][]){let count=0;const seen=new Set<number>();for(let i=0;i<adj.length;i++)if(!seen.has(i)){count++;const q=[i];seen.add(i);while(q.length){const j=q.pop()!;for(let k=0;k<adj.length;k++)if(adj[j][k]&&!seen.has(k)){seen.add(k);q.push(k);}}}return count;}
test('all finite topology opens satisfy axioms and connectedness agrees with independent neighborhood graphs',()=>{
 for(const mode of [0,1,2]){const opens=finiteTopology(mode);assert.ok(opens.includes(0)&&opens.includes(7));for(const a of opens)for(const b of opens){assert.ok(opens.includes(a|b));assert.ok(opens.includes(a&b));}const neighborhoods=[0,1,2].map(i=>opens.filter(s=>(s>>i)&1).reduce((a,b)=>a&b,7)),adj=neighborhoods.map((s,i)=>neighborhoods.map((t,j)=>Boolean(((s>>j)&1)||((t>>i)&1))));assert.equal(topologyMap(mode,1,0).components,graphComponents(adj));}
});
test('all finite map masks obey connected-image theorem and discrete/indiscrete universal cases',()=>{
 for(const source of [0,1,2])for(const target of [0,1])for(let mask=0;mask<8;mask++){const r=topologyMap(source,target,mask);if(source===1||target===0)assert.equal(r.continuous,true);if(r.components===1&&r.continuous&&target===1)assert.equal(r.imageSize,1);if((source===0||source===2)&&target===1)assert.equal(r.continuous,mask===0||mask===7);}
 assert.deepEqual(topologyMap(0,1,1).preimages,[0,6,1,7]);
});
test('stereographic inverses reconstruct every valid chart point and transitions follow the chain rule',()=>{
 for(const theta of [.1,.7,Math.PI/2,2,3.5,5.9]){const r=circleCharts(theta,0),t=r.t!,u=r.u!;close((1-t*t)/(1+t*t),r.point[0]);close(2*t/(1+t*t),r.point[1]);close((u*u-1)/(u*u+1),r.point[0]);close(2*u/(1+u*u),r.point[1]);close(t*u,1);const h=1e-5*Math.abs(t);close((1/(t+h)-1/(t-h))/(2*h),r.transitionDerivative!,1e-7);}
});
test('chart exclusions and transition domains remain distinct from manifold regularity',()=>{
 assert.equal(circleCharts(Math.PI,0).selected,null);assert.equal(circleCharts(Math.PI,1).selected,0);assert.equal(circleCharts(0,1).selected,null);assert.equal(circleCharts(0,0).selected,0);assert.equal(circleCharts(2*Math.PI,1).overlap,false);assert.equal(circleCharts(0,0).transitionDerivative,null);
 const underflow=evaluate('circle-manifold-charts',[1e-320,1]);assert.equal(underflow.stats['1{selected chart valid}'],1);assert.equal(underflow.warning,'numeric-range');assert.equal(underflow.stats['selected coordinate'],null);
});
test('tangent velocities match derivatives and reciprocal covector scaling preserves pairing and metric norm',()=>{
 for(const theta of [.2,Math.PI/2,3])for(const a of [-2,0,1])for(const b of [-1,.5,2])for(const s of [.5,1,3]){const r=circlePairing(theta,a,b,s),h=1e-6,plus=circlePoint(theta+h),minus=circlePoint(theta-h);r.tangent.forEach((x,j)=>close(x,a*(plus[j]-minus[j])/(2*h),1e-9));close(dot(r.point,r.tangent),0);close(r.alphaU*r.vU,a*b);close(r.metricUU*r.vU*r.vU,a*a);close(dot(r.tangent,r.tangent),a*a);}
});
test('circle distance minimizes all winding lifts and constant-speed arcs attain the energy bound',()=>{
 for(const R of [.5,1,2])for(const sep of [0,.2,Math.PI/2,Math.PI,5,2*Math.PI])for(const winding of [-1,0,1]){const r=circleGeodesic(R,.3,sep,winding),best=Math.min(...Array.from({length:9},(_,i)=>R*Math.abs(sep+2*Math.PI*(i-4))));close(r.distance,best);close(r.energy,r.length*r.length/2);assert.ok(r.chord<=r.distance+1e-12);close(r.path.at(-1)![0],R*Math.cos(.3+sep));close(r.path.at(-1)![1],R*Math.sin(.3+sep));}
 const a=circleGeodesic(2,0,Math.PI,0),b=circleGeodesic(2,0,Math.PI,-1);close(a.length,b.length);assert.equal(a.antipodal,true);close(circleGeodesic(2,0,0,1).distance,0);close(circleGeodesic(2,0,0,1).length,4*Math.PI);
});
function ambientTransport(theta:number,travel:number,alpha:number){const si=Math.sin(theta),co=Math.cos(theta),v=[co*Math.cos(alpha),Math.sin(alpha),-si*Math.cos(alpha)],N=1500,h=travel/N;
 const rhs=(phi:number,w:number[])=>{const n=[si*Math.cos(phi),si*Math.sin(phi),co],np=[-si*Math.sin(phi),si*Math.cos(phi),0],q=-dot(w,np);return n.map(x=>q*x);};
 for(let j=0;j<N;j++){const phi=j*h,k1=rhs(phi,v),k2=rhs(phi+h/2,v.map((x,i)=>x+h*k1[i]/2)),k3=rhs(phi+h/2,v.map((x,i)=>x+h*k2[i]/2)),k4=rhs(phi+h,v.map((x,i)=>x+h*k3[i]));for(let i=0;i<3;i++)v[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;}
 const et=[co*Math.cos(travel),co*Math.sin(travel),-si],ep=[-Math.sin(travel),Math.cos(travel),0],normal=[si*Math.cos(travel),si*Math.sin(travel),co];return {components:[dot(v,et),dot(v,ep)],norm:dot(v,v),normal:dot(v,normal)};}
test('sphere parallel transport agrees with an independent three-dimensional ambient RK4 ODE',()=>{
 for(const theta of [.5,Math.PI/3,Math.PI/2,2.3])for(const travel of [1,2*Math.PI]){const independent=ambientTransport(theta,travel,.4),r=sphereTransport(1,theta,travel,.4);r.vector.forEach((x,i)=>close(x,independent.components[i],1e-9));close(independent.norm,1,1e-9);close(independent.normal,0,1e-9);}
});
test('sphere curvature and cap holonomy agree with independent area integration and scale cancellation',()=>{
 for(const theta of [.4,Math.PI/3,Math.PI/2,2.5]){const a=sphereTransport(1,theta,2*Math.PI,0),b=sphereTransport(2,theta,2*Math.PI,0);close(a.capCurvature,2*Math.PI*integrate(t=>Math.sin(t),0,theta));close(a.curvature/4,b.curvature);close(a.capArea*4,b.capArea);close(Math.cos(a.capCurvature),a.vector[0]);close(Math.sin(a.capCurvature),a.vector[1]);close(b.loopHolonomy!,a.loopHolonomy!);}
 assert.equal(sphereTransport(1,1,1,0).loopHolonomy,null);assert.equal(sphereTransport(1,Math.PI/2,2*Math.PI,0).geodesic,true);
});
test('one-form line integrals match independently integrated translated-circle pullbacks and orientation',()=>{
 for(const mode of [0,1,2])for(const [R,center] of [[1,0],[1,.25],[.5,1.5]])for(const reverse of [0,1]){const r=circleForm(mode,R,center,reverse),numeric=integrate(t=>{const x=center+R*Math.cos(t),y=R*Math.sin(t),field=formField(mode,x,y)!;return field[0]*(-R*Math.sin(t))+field[1]*R*Math.cos(t);},0,2*Math.PI)*r.sign;close(numeric,r.integral!,1e-9);if(r.diskAllowed)close(r.integral!,r.diskIntegral!);}
});
test('angular form is locally closed but not globally exact on the punctured plane; singular boundary stays undefined',()=>{
 for(const [x,y] of [[1,0],[.3,.4],[-1,2]]){const h=1e-6,dxQ=(formField(2,x+h,y)![1]-formField(2,x-h,y)![1])/(2*h),dyP=(formField(2,x,y+h)![0]-formField(2,x,y-h)![0])/(2*h);close(dxQ-dyP,0,1e-8);}
 const around=circleForm(2,1,0,0);close(around.integral!,2*Math.PI);assert.equal(around.diskAllowed,false);assert.equal(around.diskIntegral,null);const hit=circleForm(2,1,1,0);assert.equal(hit.integral,null);assert.equal(hit.onSingularity,true);assert.equal(formField(2,0,0),null);
});
test('immersion derivatives agree with finite differences and distinguish crossing from cusp',()=>{
 for(const mode of [0,1,2])for(const u of [-.8,-.3,0,.4,.8]){const r=immersionPoint(mode,u),h=1e-6,plus=immersionPoint(mode,u+h).point,minus=immersionPoint(mode,u-h).point;r.derivative.forEach((x,j)=>close(x,(plus[j]-minus[j])/(2*h),1e-8));}
 assert.deepEqual(immersionPoint(1,-1).point,immersionPoint(1,0).point);assert.equal(immersionPoint(1,0).immersion,true);assert.equal(immersionPoint(1,0).injective,false);const cusp=immersionPoint(2,0);assert.equal(cusp.rank,0);assert.equal(cusp.topologicalEmbedding,true);assert.equal(cusp.smoothEmbedding,false);assert.equal(immersionPoint(2,.5).rank,1);
});
function bruteBetti(f:FilteredSimplex[],epsilon:number){const active=f.filter(s=>s.value<=epsilon),groups=[0,1,2,3].map(d=>active.filter(s=>s.dimension===d));const boundary=(s:FilteredSimplex)=>s.dimension===0?0:s.vertices.reduce((mask,_,k)=>{const face=s.vertices.filter((_,j)=>j!==k).join(',');return mask^(1<<groups[s.dimension-1].findIndex(t=>t.vertices.join(',')===face));},0);const cycles:number[]=[],images:number[]=[];
 for(let d=0;d<=3;d++){let zero=0;const image=new Set<number>();for(let mask=0;mask<(1<<groups[d].length);mask++){let out=0;groups[d].forEach((s,i)=>{if((mask>>i)&1)out^=boundary(s);});image.add(out);if(out===0)zero++;}cycles[d]=zero;images[d]=image.size;}
 return [0,1,2].map(d=>Math.log2(cycles[d])-Math.log2(images[d+1]));}
test('boundary of boundary is zero for every simplex in the full Rips filtration',()=>{
 const f=ripsFiltration(rectanglePoints(1));assert.equal(f.length,15);for(const s of f.filter(s=>s.dimension>=2)){const parity=new Map<string,number>();for(let i=0;i<s.vertices.length;i++){const face=s.vertices.filter((_,j)=>i!==j);for(let j=0;j<face.length;j++){const key=face.filter((_,k)=>k!==j).join(',');parity.set(key,(parity.get(key)??0)^1);}}assert.ok([...parity.values()].every(x=>x===0));}
});
test('binary boundary ranks and persistence interval counts match exhaustive chain enumeration at every critical value',()=>{
 for(const points of [rectanglePoints(1),rectanglePoints(.5,2),[[0,0],[1,0],[2,0],[3,0]] as Vec2[],[[0,0],[0,0],[1,0],[1,1]] as Vec2[]]){const f=ripsFiltration(points),bars=persistentHomology(f),critical=[...new Set(f.map(s=>s.value))].sort((a,b)=>a-b),queries=[...critical,...critical.slice(1).map((x,i)=>(x+critical[i])/2)];for(const epsilon of queries){const reference=bruteBetti(f,epsilon);assert.deepEqual(bettiAt(f,epsilon).betti,reference);assert.deepEqual([0,1,2].map(d=>bars.filter(b=>b.dimension===d&&b.birth<=epsilon&&epsilon<b.death).length),reference);}}
});
test('square and rectangle persistence have independently known births, deaths and endpoint convention',()=>{
 for(const [scale,aspect] of [[1,1],[1,2],[.5,.5],[2,1]]){const f=ripsFiltration(rectanglePoints(scale,aspect)),bars=persistentHomology(f),h1=bars.filter(b=>b.dimension===1&&b.death>b.birth);assert.equal(h1.length,1);close(h1[0].birth,scale*Math.max(1,aspect));close(h1[0].death,scale*Math.hypot(1,aspect));assert.equal(bars.filter(b=>b.dimension===0&&b.death===Infinity).length,1);assert.equal(bettiAt(f,h1[0].birth).betti[1],1);assert.equal(bettiAt(f,h1[0].death).betti[1],0);}
});
test('omitting the tetrahedron creates a false H2; full complex removes it',()=>{
 const f=ripsFiltration(rectanglePoints(1)),full=bettiAt(f,2),surface=bettiAt(f.filter(s=>s.dimension<3),2);assert.deepEqual(full.betti,[1,0,0]);assert.deepEqual(surface.betti,[1,0,1]);assert.equal(full.counts[2],4);assert.equal(full.counts[3],1);
});
test('persistence positive bars are invariant under vertex relabeling',()=>{
 const P=rectanglePoints(1,2),reference=persistentHomology(ripsFiltration(P)).filter(b=>b.death>b.birth);for(const permutation of [[1,0,2,3],[3,2,1,0],[2,0,3,1]])assert.deepEqual(persistentHomology(ripsFiltration(permutation.map(i=>P[i]))).filter(b=>b.death>b.birth),reference);
});
test('finite bottleneck matching respects diagonal deletion, multiplicity and a nontrivial optimal assignment',()=>{
 close(finiteBottleneck([],[]),0);close(finiteBottleneck([[1,2]],[]),.5);close(finiteBottleneck([[1,2]],[[1.1,2.2]]),.2);close(finiteBottleneck([[1,2],[1,2]],[[1,2]]),.5);close(finiteBottleneck([[0,10],[4,5]],[[0,10]]),.5);close(finiteBottleneck([[0,10],[5,15]],[[5,15],[0,10]]),0);
 for(const delta of [.1,.25]){const r=stabilityExperiment(delta,1,1),p=r.pBars.find(b=>b.dimension===1&&b.death>b.birth)!,q=r.qBars.find(b=>b.dimension===1&&b.death>b.birth)!,direct=Math.max(Math.abs(p.birth-q.birth),Math.abs(p.death-q.death)),diagonal=Math.max((p.death-p.birth)/2,(q.death-q.birth)/2);close(r.bottleneck1,Math.min(direct,diagonal));if(delta===.25)assert.ok(r.bottleneck1<direct);}
});
test('labeled metric distortion bounds every simplex shift, bottleneck distances and displacement bound',()=>{
 for(const delta of [-.25,-.1,0,.1,.25])for(const scale of [.5,1,2]){const r=stabilityExperiment(delta,scale,1.05*scale);for(const simplex of r.f){const other=r.g.find(s=>s.vertices.join(',')===simplex.vertices.join(','))!;assert.ok(Math.abs(simplex.value-other.value)<=r.distortion+1e-12);}assert.ok(r.distortion<=2*r.eta+1e-12);assert.ok(r.bottleneck0<=r.distortion+1e-12);assert.ok(r.bottleneck1<=r.distortion+1e-12);}
});
test('stable diagrams may change fixed-threshold Betti numbers and all distances scale predictably',()=>{
 const a=stabilityExperiment(.1,1,1.05),b=stabilityExperiment(.1,2,2.1);assert.equal(a.pBetti.betti[1],1);assert.equal(a.qBetti.betti[1],0);assert.deepEqual(a.pBetti.betti,b.pBetti.betti);assert.deepEqual(a.qBetti.betti,b.qBetti.betti);close(a.bottleneck0*2,b.bottleneck0);close(a.bottleneck1*2,b.bottleneck1);close(a.distortion*2,b.distortion);close(stabilityExperiment(0,1,1).bottleneck0,0);close(stabilityExperiment(.25,1,1).bottleneck1,(Math.SQRT2-1)/2);
});
test('pullback metric, covector and raised gradient satisfy independent physical-coordinate calculations',()=>{
 for(const s of [.5,1,3])for(const k of [-2,0,1])for(const a of [-2,0,1])for(const b of [-1,0,2]){const r=coordinateMetric(s,k,a,b),apply=(v:number[])=>[s*v[0]+k*v[1],v[1]/s];close(apply(r.vu)[0],a);close(apply(r.vu)[1],b);close(r.normSquared,a*a+b*b);close(dot(r.alpha,r.vu),a+2*b);close(r.g[0][0]*r.g[1][1]-r.g[0][1]*r.g[1][0],1,1e-12);r.g.forEach((row,i)=>close(dot(row,r.gradient),r.alpha[i],1e-12));close(apply(r.gradient)[0],1);close(apply(r.gradient)[1],2);}
});
test('all geometry controls reject invalid ranges and fractional integer choices',()=>{
 for(const kind of geometryKinds){const spec=geometryLabs[kind],initial=spec.controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[]));for(let j=0;j<initial.length;j++){for(const x of [NaN,Infinity,spec.controls[j].max+1]){const bad=[...initial];bad[j]=x;assert.throws(()=>evaluate(kind,bad));}if(spec.controls[j].step!=='any'){const bad=[...initial];bad[j]=spec.controls[j].min+.5;assert.throws(()=>evaluate(kind,bad));}}}
 for(const f of [()=>ripsFiltration([]),()=>ripsFiltration([[NaN,0]]),()=>finiteBottleneck([[0,Infinity]],[]),()=>finiteBottleneck([[2,1]],[]),()=>gf2Rank([[-1]]),()=>persistentHomology([{vertices:[0,1],dimension:1,value:1}])])assert.throws(f);
});
for(const kind of geometryKinds)test(`${kind}: presets preserve finite geometry and guided targets`,()=>{
 assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const g of learningGuides[kind].experiments)assert.ok(geometryLabs[kind].presets[g.preset]);
 for(const p of geometryLabs[kind].presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));for(const [x,y] of [...r.curves.flat(),...(r.dots??[]),...(r.holes??[])]){assert.ok(x>=r.bounds![0]&&x<=r.bounds![1]);assert.ok(y>=r.bounds![2]&&y<=r.bounds![3]);}assert.ok(!/[가-힣]/.test(r.statement??''));for(const v of Object.values(r.stats))if(typeof v==='number')assert.ok(!Number.isNaN(v));}
});
for(const kind of geometryKinds)for(const lang of ['ko','en'] as const)test(`${kind}.${lang}: substantive theory, valid math, explained exercise and further reading`,async()=>{
 const source=await fs.readFile(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),rendered=await getLearningContent(kind);assert.ok(rendered.outline[lang].length>=4);assert.ok(rendered[lang].length>4000);assert.ok(rendered[lang].includes('katex'));assert.ok(!rendered[lang].includes('katex-error'));assert.ok(source.includes(lang==='ko'?'직접 풀기':'Exercise'));assert.ok(source.includes('https://'));if(lang==='en'){assert.ok(!/[가-힣]/.test(source));assert.ok(source.split(/\s+/).length>=350);}
});
