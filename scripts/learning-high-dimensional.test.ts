import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateHighDimensional as evaluate, highDimensionalKinds, seededNormals, symmetricEigenvalues, finiteCircleNets, chiMean, gaussianAxisWidth, bernoulliDeviationTail } from '../src/lib/learning-high-dimensional';
import { highDimensionalLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const num=(r:ReturnType<typeof evaluate>,key:string)=>{assert.equal(typeof r.stats[key],'number');return r.stats[key] as number;};
const integrate=(f:(x:number)=>number,lo:number,hi:number,N=6000)=>{const h=(hi-lo)/N;let sum=f(lo)+f(hi);for(let i=1;i<N;i++)sum+=(i%2?4:2)*f(lo+i*h);return sum*h/3;};
test('concentration tails agree with exhaustive independent sign realizations and inclusive boundaries',()=>{
 for(let n=1;n<=10;n++)for(const t of [0,.2,.5,1]){let count=0;for(let mask=0;mask<2**n;mask++){let k=0;for(let j=0;j<n;j++)k+=(mask>>j)&1;if(Math.abs(2*k-n)>=t*n)count++;}const r=evaluate('concentration-finite-averages',[n,t]);close(num(r,'exact inclusive tail'),count/2**n);assert.ok(num(r,'exact inclusive tail')<=num(r,'Hoeffding upper bound')+1e-12);assert.ok(num(r,'exact inclusive tail')<=num(r,'Chebyshev upper bound')+1e-12);}
 close(num(evaluate('concentration-finite-averages',[16,.5]),'exact inclusive tail'),5034/65536);
});
test('sub-Gaussian MGFs match two-point enumeration and Gaussian tails match independent integration',()=>{
 for(const sigma of [.25,1,2])for(const lambda of [-3,-1,0,1,3]){const r=evaluate('sub-gaussian-tail-bounds',[sigma,2,lambda]),log=Math.log((Math.exp(sigma*lambda)+Math.exp(-sigma*lambda))/2);close(num(r,'Rademacher log MGF at λ'),log);assert.ok(log<=num(r,'Gaussian log MGF at λ')+1e-12);}
 for(const t of [0,1,2]){const r=evaluate('sub-gaussian-tail-bounds',[1,t,1]),density=(x:number)=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI),tail=2*integrate(density,t,10);close(num(r,'Gaussian two-sided tail (approx.)'),tail,2e-7);assert.equal(r.stats['Rademacher inclusive tail'],Number(t<=1));}
});
test('Hoeffding and Bernstein dominate exact Bernoulli tails across variance and endpoint regimes',()=>{
 for(const n of [1,4,10,30])for(const p of [0,.1,.5,.9,1])for(const t of [0,.05,.2,.4,1]){const r=evaluate('hoeffding-bernstein-bounds',[n,p,t]),actual=num(r,'exact inclusive tail');assert.ok(actual<=num(r,'Hoeffding upper bound')+1e-12);assert.ok(actual<=num(r,'Bernstein upper bound')+1e-12);if(t===0)assert.equal(actual,1);if((p===0||p===1)&&t>0){assert.equal(actual,0);assert.equal(r.stats['Bernstein upper bound'],0);}}
 close(bernoulliDeviationTail(10,.5,.2),.34375);close(num(evaluate('hoeffding-bernstein-bounds',[30,.1,.2]),'Bernstein upper bound'),2*Math.exp(-4));
});
test('finite circle covers and packings are optimal against exhaustive subsets of small polygons',()=>{
 for(let n=4;n<=9;n++)for(const epsilon of [.1,.8,1.4,2]){const points=Array.from({length:n},(_,i)=>[Math.cos(2*Math.PI*i/n),Math.sin(2*Math.PI*i/n)]),distance=(i:number,j:number)=>Math.hypot(points[i][0]-points[j][0],points[i][1]-points[j][1]);let minCover=n,maxPacking=1;for(let mask=1;mask<2**n;mask++){const selected=Array.from({length:n},(_,i)=>i).filter(i=>(mask>>i)&1);if(selected.length<minCover&&points.every((_,i)=>selected.some(j=>distance(i,j)<=epsilon+1e-12)))minCover=selected.length;if(selected.length>maxPacking&&selected.every((i,k)=>selected.slice(k+1).every(j=>distance(i,j)>epsilon+1e-12)))maxPacking=selected.length;}const result=finiteCircleNets(n,epsilon);assert.equal(result.cover.length,minCover);assert.equal(result.packing.length,maxPacking);assert.ok(points.every((_,i)=>result.cover.some(j=>distance(i,j)<=epsilon+1e-12)));}
 const r=evaluate('covering-packing-nets',[12,2]);assert.equal(r.stats['minimum vertex-center cover count'],1);assert.equal(r.stats['maximum strict packing count'],1);assert.equal(r.stats['minimum packing separation'],null);
});
test('ball radius moments and shells agree with independent radial-density integration',()=>{
 for(const d of [1,2,8,32,128]){const density=(r:number)=>d*r**(d-1),mean=integrate(r=>r*density(r),0,1),variance=integrate(r=>(r-mean)**2*density(r),0,1),r=evaluate('high-dimensional-geometry',[d,.1]);close(num(r,'ball mean radius'),mean,1e-7);close(num(r,'ball radius variance'),variance,1e-7);close(num(r,'outer-shell probability'),integrate(density,.9,1),1e-7);}
 assert.equal(evaluate('high-dimensional-geometry',[128,0]).stats['outer-shell probability'],0);assert.equal(evaluate('high-dimensional-geometry',[128,1]).stats['outer-shell probability'],1);
});
test('sphere squared-distance moments follow from independent isotropic finite directions',()=>{
 for(const d of [1,2,4,16]){const points:number[][]=Array.from({length:2*d},(_,i)=>Array.from({length:d},(_,j)=>i%d===j?(i<d?1:-1):0)),distances=points.flatMap(x=>points.map(y=>x.reduce((s,v,i)=>s+(v-y[i])**2,0))),mean=distances.reduce((s,v)=>s+v,0)/distances.length,variance=distances.reduce((s,v)=>s+(v-mean)**2,0)/distances.length,r=evaluate('high-dimensional-geometry',[d,.1]);close(num(r,'sphere squared-distance mean'),mean);close(num(r,'sphere squared-distance variance'),variance);}
 // These discrete directions share the sphere's first two isotropic moments, not its full law.
});
test('pseudo-Gaussian generator is finite, reproducible and prefix-consistent with Box–Muller radius',()=>{
 const a=seededNormals(1,9),b=seededNormals(1,13);assert.deepEqual(a,b.slice(0,9));assert.notDeepEqual(a,seededNormals(2,9));assert.ok(seededNormals(32,16384).every(Number.isFinite));const u=(1015568748+.5)/4294967296;close(a[0]**2+a[1]**2,-2*Math.log(u));
});
test('JL table matches independent row-wise projection norms and original fixed-point distances',()=>{
 for(const m of [1,8,96]){const r=evaluate('johnson-lindenstrauss-projections',[m,.5,7]),normals=seededNormals(7,128*m),pair01=Array.from({length:m},(_,i)=>normals[128*i]**2/m).reduce((s,v)=>s+v,0);close(Number(r.table.rows[0][3]),pair01);const original=[1,1,1,2,2-2/Math.sqrt(128),2-2/Math.sqrt(128)];r.table.rows.forEach((row,i)=>close(Number(row[2]),original[i]));close(num(r,'maximum squared-distance relative error'),Math.max(...r.table.rows.map(row=>Math.abs(Number(row[4])-1))));}
});
test('JL sufficient dimensions include six-pair union factor and satisfy the declared failure level',()=>{
 for(const epsilon of [.1,.5,.8,.9]){const r=evaluate('johnson-lindenstrauss-projections',[8,epsilon,7]),required=num(r,'sufficient m for ideal failure ≤ .05'),coefficient=epsilon*epsilon/4-epsilon**3/6;assert.ok(12*Math.exp(-required*coefficient)<=.05);assert.ok(12*Math.exp(-(required-1)*coefficient)>.05);assert.ok((epsilon-Math.log1p(epsilon))/2>=coefficient-1e-12);assert.ok((-epsilon-Math.log1p(-epsilon))/2>=coefficient-1e-12);}
 assert.ok(num(evaluate('johnson-lindenstrauss-projections',[96,.8,7]),'ideal Gaussian failure upper bound')<.05);
});
test('Jacobi eigenvalues match analytic tridiagonal, indefinite and rank-one spectra',()=>{
 for(let n=2;n<=8;n++){const matrix=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?2:Math.abs(i-j)===1?-1:0)),actual=symmetricEigenvalues(matrix),expected=Array.from({length:n},(_,i)=>2-2*Math.cos((i+1)*Math.PI/(n+1)));actual.forEach((v,i)=>close(v,expected[i],1e-10));}
 const indefinite=symmetricEigenvalues([[1,2],[2,-1]]);close(indefinite[0],-Math.sqrt(5));close(indefinite[1],Math.sqrt(5));const rankOne=symmetricEigenvalues([[1,2,3],[2,4,6],[3,6,9]]);close(rankOne[0],0);close(rankOne[1],0);close(rankOne[2],14);
});
test('sample spectra preserve Gram trace and Frobenius identities and reveal rank deficiency',()=>{
 for(const d of [2,6,8])for(const m of [1,3,12])for(const seed of [1,7,32]){const r=evaluate('random-matrix-spectra',[d,m,seed]),normal=seededNormals(seed,d*m),gram=Array.from({length:d},(_,i)=>Array.from({length:d},(_,j)=>Array.from({length:m},(_,k)=>normal[k*d+i]*normal[k*d+j]/m).reduce((s,v)=>s+v,0))),eigen=r.table.rows.map(row=>Number(row[1]));close(eigen.reduce((s,v)=>s+v,0),normal.reduce((s,v)=>s+v*v/m,0),1e-8);close(eigen.reduce((s,v)=>s+v*v,0),gram.flat().reduce((s,v)=>s+v*v,0),1e-7);assert.ok(eigen.every(v=>v>=0));assert.equal(r.stats['numerical rank'],Math.min(d,m));close(num(r,'MP limiting zero-atom mass'),Math.max(0,(d-m)/d));}
});
test('Gaussian widths match one- and two-dimensional closed forms and respect geometric bounds',()=>{
 close(chiMean(1),Math.sqrt(2/Math.PI));close(chiMean(2),Math.sqrt(Math.PI/2));close(chiMean(3),2*Math.sqrt(2/Math.PI));close(gaussianAxisWidth(1),Math.sqrt(2/Math.PI),2e-6);close(gaussianAxisWidth(2),2/Math.sqrt(Math.PI),2e-6);
 for(const d of [1,2,16,64]){assert.ok(gaussianAxisWidth(d)<=chiMean(d)+2e-6);assert.ok(chiMean(d)<=Math.sqrt(d)+1e-12);assert.ok(gaussianAxisWidth(d)<=Math.sqrt(2*Math.log(2*d))+1e-12);const a=evaluate('gaussian-width-geometry',[d,1,0]),b=evaluate('gaussian-width-geometry',[d,2,0]);close(num(b,'selected Gaussian width'),2*num(a,'selected Gaussian width'));}
});
test('paired random-feature expectation and variance match independent Gaussian quadrature',()=>{
 const density=(g:number)=>Math.exp(-g*g/2)/Math.sqrt(2*Math.PI);for(const length of [.25,1,2])for(const separation of [0,.5,1,3]){const r=evaluate('random-fourier-features',[32,length,7,separation]),mean=integrate(g=>Math.cos(g*separation/length)*density(g),-10,10),second=integrate(g=>Math.cos(g*separation/length)**2*density(g),-10,10);close(num(r,'exact kernel at separation r'),mean,1e-8);close(num(r,'ideal pointwise variance'),(second-mean*mean)/32,1e-9);}
});
test('paired features have unit norm, PSD Gram and the displayed inner product',()=>{
 for(const m of [1,4,32]){const omega=seededNormals(7,m),points=[0,.5,1],features=points.map(x=>omega.flatMap(w=>[Math.cos(w*x)/Math.sqrt(m),Math.sin(w*x)/Math.sqrt(m)])),dot=(x:number[],y:number[])=>x.reduce((s,v,i)=>s+v*y[i],0);features.forEach(z=>close(dot(z,z),1));const r=evaluate('random-fourier-features',[m,1,7,1]);close(num(r,'feature estimate at r'),dot(features[0],features[2]));const coefficients=[1,-2,1],quadratic=features.reduce((s,x,i)=>s+features.reduce((t,y,j)=>t+coefficients[i]*coefficients[j]*dot(x,y),0),0);assert.ok(quadratic>=-1e-12);assert.equal(r.stats['feature coordinate count'],2*m);}
 const same=evaluate('random-fourier-features',[32,1,7,0]);assert.equal(same.stats['feature estimate at r'],1);assert.equal(same.stats['ideal pointwise variance'],0);
});
test('effective ranks use distinct conventions and exact tensor-grid strings preserve large integers',()=>{
 for(const d of [1,16,32]){const iso=evaluate('curse-effective-dimension',[d,10,1,.1]),single=evaluate('curse-effective-dimension',[d,10,0,.1]);assert.equal(iso.stats['participation rank'],d);assert.equal(iso.stats['covariance effective rank (trace/max)'],d);close(num(iso,'ridge effective dimension'),d/1.1);assert.equal(single.stats['participation rank'],1);assert.equal(single.stats['covariance effective rank (trace/max)'],1);close(num(single,'ridge effective dimension'),1/1.1);assert.deepEqual(iso.table.rows.map(row=>row[2]),single.table.rows.map(row=>row[2]));}
 assert.equal(evaluate('curse-effective-dimension',[32,10,0,.1]).table.rows[31][2],'1'+'0'.repeat(32));const small=evaluate('curse-effective-dimension',[16,4,.5,.01]),large=evaluate('curse-effective-dimension',[16,4,.5,1]);assert.ok(num(small,'ridge effective dimension')>num(large,'ridge effective dimension'));
});
test('high-dimensional helpers and controls reject invalid seeds, matrices and domains',()=>{
 for(const f of [()=>seededNormals(0,10),()=>seededNormals(1,16385),()=>seededNormals(1.5,10),()=>symmetricEigenvalues([[1,2],[1,1]]),()=>symmetricEigenvalues([[NaN]]),()=>finiteCircleNets(3,1),()=>finiteCircleNets(4,0),()=>chiMean(0),()=>gaussianAxisWidth(65),()=>bernoulliDeviationTail(0,.5,.2)])assert.throws(f);
 for(const kind of highDimensionalKinds){const values=highDimensionalLabs[kind].controls.map(c=>c.initial);assert.throws(()=>evaluate(kind,[NaN,...values.slice(1)]));assert.throws(()=>evaluate(kind,[]));assert.throws(()=>evaluate(kind,[Infinity,...values.slice(1)]));}
});
for(const kind of highDimensionalKinds)test(`${kind} has four concept steps and three finite guided experiments`,()=>{
 const spec=highDimensionalLabs[kind],guide=learningGuides[kind];assert.equal(guide.steps.length,4);assert.equal(guide.experiments.length,3);for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat().every(v=>v.every(Number.isFinite)));assert.ok(!/[가-힣]/.test(r.statement??''));for(const value of Object.values(r.stats)){if(typeof value==='number')assert.ok(Number.isFinite(value));if(Array.isArray(value))assert.ok(value.every(Number.isFinite));}}
});

test('small Rademacher log MGFs retain their positive quadratic term and even symmetry',()=>{
 for(const sigma of [.25,1,2])for(const lambda of [1e-8,1e-10,1e-100,1e-150]){
  const r=evaluate('sub-gaussian-tail-bounds',[sigma,2,lambda]);
  const actual=num(r,'Rademacher log MGF at λ'),leading=(sigma*lambda)**2/2;
  assert.ok(actual>0);close(actual/leading,1,1e-12);
  assert.equal(evaluate('sub-gaussian-tail-bounds',[sigma,2,-lambda]).stats['Rademacher log MGF at λ'],actual);
 }
 for(const lambda of [1-1e-12,1,1+1e-12]){
  close(num(evaluate('sub-gaussian-tail-bounds',[1,2,lambda]),'Rademacher log MGF at λ'),Math.log(Math.cosh(lambda)),1e-12);
 }
 assert.equal(evaluate('sub-gaussian-tail-bounds',[1,2,0]).stats['Rademacher log MGF at λ'],0);
});

test('subnormal positive log MGFs are retained and unrepresentable ones are withheld',()=>{
 // Exact binary64 input 2.6e-162 has a quadratic half-square rounding to one subnormal unit.
 const tiny=evaluate('sub-gaussian-tail-bounds',[1,2,2.6e-162]);
 assert.equal(tiny.stats['Rademacher log MGF at λ'],Number.MIN_VALUE);
 assert.equal(tiny.stats['Gaussian log MGF at λ'],Number.MIN_VALUE);
 const under=evaluate('sub-gaussian-tail-bounds',[1,2,1e-162]);
 assert.equal(under.stats['Rademacher log MGF at λ'],null);
 assert.equal(under.stats['Gaussian log MGF at λ'],null);
 assert.equal(under.warning,'numeric-range');assert.deepEqual(under.dots,[]);
});
