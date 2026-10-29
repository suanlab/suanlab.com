import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';
export const analysisKinds=['limits-continuity-completeness','differentiability-mean-value','integration-fundamental-theorem','taylor-remainders','series-convergence-tests','pointwise-uniform-convergence','compactness-extreme-values','convexity-first-order','limits-integration-interchange','implicit-inverse-functions','matrix-trace-logdet-derivatives','vector-fields-divergence-curl','line-surface-volume-integrals','gauss-stokes-theorems','higher-order-autodiff'] as const;
export type AnalysisKind=typeof analysisKinds[number];
export function isAnalysisKind(kind:string):kind is AnalysisKind{return (analysisKinds as readonly string[]).includes(kind);}
const samples=(count:number,fn:(i:number)=>Vec2)=>Array.from({length:count},(_,i)=>fn(i));
function integer(n:number,min:number,max:number){if(!Number.isInteger(n)||n<min||n>max)throw new Error('Integer parameter outside the example domain.');}
function interval(x:number,min:number,max:number){if(x<min||x>max)throw new Error('Parameter outside the example domain.');}
export function expTaylor(x:number,degree:number){integer(degree,0,8);let term=1,sum=1;for(let k=1;k<=degree;k++){term*=x/k;sum+=term;}return sum;}
export function evaluateAnalysis(kind:AnalysisKind,p:number[]):FoundationResult{
 const counts:Record<AnalysisKind,number>={'limits-continuity-completeness':3,'differentiability-mean-value':3,'integration-fundamental-theorem':2,'taylor-remainders':2,'series-convergence-tests':2,'pointwise-uniform-convergence':3,'compactness-extreme-values':2,'convexity-first-order':3,'limits-integration-interchange':2,'implicit-inverse-functions':2,'matrix-trace-logdet-derivatives':4,'vector-fields-divergence-curl':4,'line-surface-volume-integrals':2,'gauss-stokes-theorems':3,'higher-order-autodiff':4};
 if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Use finite parameters of the expected shape.');const [a,b,c,d]=p;
 if(kind==='limits-continuity-completeness'){
  integer(a,0,1);interval(b,0,3);interval(c,.01,1);const left=a===0?2:0,right=a===0?2:1;
  return {statement:a===0?`f(x)=x+1 for x≠1; f(1)=${b}`:`f(x)=0 for x<1; f(x)=1 for x>1; f(1)=${b}`,stats:{'left limit':left,'right limit':right,'two-sided limit':a===0?2:null,'f(1)':b,'1{continuous at 1}':a===0&&b===2?1:0,'δ':c},curves:[samples(51,i=>{const x=1-c+i*(c-Math.min(.01,c/10))/50;return [x,a===0?x+1:0];}),samples(51,i=>{const gap=Math.min(.01,c/10);const x=1+gap+i*(c-gap)/50;return [x,a===0?x+1:1];}),[[1,b]]],holes:(left===right?[[1,left]]:[[1,left],[1,right]]).filter(point=>point[1]!==b) as Vec2[],curveLabels:['x<1','x>1','f(1)']};
 }
 if(kind==='differentiability-mean-value'){
  integer(a,0,1);interval(b,-1,1);interval(c,0,1);const f=(x:number)=>a===0?x*x:Math.abs(x),derivative=a===0?2*b:b===0?null:Math.sign(b),value=f(b);
  return {statement:`f(x)=${a===0?'x²':'|x|'}; x₀=${b}; h=${c}; MVT interval=[−1,1]`,stats:{'f(x₀)':value,'f′(x₀)':derivative,'forward secant':c===0?null:(f(b+c)-value)/c,'mean slope [−1,1]':0,'MVT witness c':a===0?0:null,'1{MVT hypotheses hold}':a===0?1:0},curves:[samples(101,i=>{const x=-2+i*.04;return [x,f(x)];}),...(derivative===null?[]:[[[ -2,value+derivative*(-2-b)],[2,value+derivative*(2-b)]] as Vec2[]])]};
 }
 if(kind==='integration-fundamental-theorem'){
  interval(a,0,2);integer(b,1,64);const h=a/b,left=Array.from({length:b},(_,k)=>(k*h)**2*h).reduce((s,v)=>s+v,0),exact=a**3/3;
  return {statement:`F(x)=∫₀ˣt²dt; x=${a}; left rectangles N=${b}`,stats:{'F(x)=x³/3':exact,'left sum':left,'absolute error':Math.abs(exact-left),'error bound x³/N':a**3/b,'F′(x)=f(x)':a*a},curves:[samples(81,i=>{const x=i/40;return [x,x*x];}),samples(81,i=>{const x=i/40;return [x,x**3/3];})]};
 }
 if(kind==='taylor-remainders'){
  interval(a,-2,2);integer(b,0,8);let factorial=1;for(let k=2;k<=b+1;k++)factorial*=k;const approx=expTaylor(a,b),error=Math.abs(Math.exp(a)-approx),bound=Math.exp(Math.abs(a))*Math.abs(a)**(b+1)/factorial;
  return {statement:`f(x)=exp(x); T_${b}(x) at 0; probe x=${a}`,stats:{'exp(x)':Math.exp(a),'T_n(x)':approx,'absolute error':error,'Lagrange bound':bound,'degree n':b},curves:[samples(101,i=>{const x=-2+i*.04;return [x,Math.exp(x)];}),samples(101,i=>{const x=-2+i*.04;return [x,expTaylor(x,b)];})]};
 }
 if(kind==='series-convergence-tests'){
  integer(a,0,2);integer(b,1,128);let sum=0;const curve:Vec2[]=[[0,0]];for(let n=1;n<=b;n++){sum+=a===0?1/n:a===1?(n%2?1:-1)/n:1/(n*n);curve.push([n,sum]);}const limit=a===0?null:a===1?Math.log(2):Math.PI**2/6;
  return {statement:`a_n=${a===0?'1/n':a===1?'(−1)^(n+1)/n':'1/n²'}; N=${b}`,stats:{'S_N':sum,'a_N':a===0?1/b:a===1?(b%2?1:-1)/b:1/(b*b),'series sum':limit,'tail bound':a===0?null:a===1?1/(b+1):1/b,'1{absolutely convergent}':a===2?1:0},curves:[curve,...(limit===null?[]:[[[0,limit],[b,limit]] as Vec2[]])]};
 }
 if(kind==='pointwise-uniform-convergence'){
  integer(a,0,1);integer(b,1,128);interval(c,0,1);const fn=(x:number)=>a===0?x**b:x/b,limit=a===0&&c===1?1:0;
  return {statement:`f_n(x)=${a===0?'xⁿ':'x/n'} on [0,1]; n=${b}; x₀=${c}`,stats:{'f_n(x₀)':fn(c),'pointwise limit at x₀':limit,'pointwise error':Math.abs(fn(c)-limit),'sup error on [0,1]':a===0?1:1/b,'1{uniform convergence}':a===0?0:1},curves:[samples(101,i=>[i/100,fn(i/100)]),[[0,0],[a===0?.99:1,0]],...(a===0?[[[1,1]] as Vec2[]]:[])],holes:a===0?[[1,0]]:undefined,curveLabels:a===0?['f_n','limit for x<1','limit at x=1']:['f_n','limit']};
 }
 if(kind==='compactness-extreme-values'){
  integer(a,0,1);interval(b,0,1);const defined=a===0||(b>0&&b<1);
  return {statement:`f(x)=x; domain=${a===0?'[0,1]':'(0,1)'}; x₀=${b}`,stats:{'f(x₀)':defined?b:null,'inf f':0,'sup f':1,'minimum':a===0?0:null,'maximum':a===0?1:null,'1{domain compact}':a===0?1:0},curves:[a===0?[[0,0],[1,1]]:[[.01,.01],[.99,.99]]],holes:a===0?undefined:[[0,0],[1,1]],warning:defined?undefined:'function-domain'};
 }
 if(kind==='convexity-first-order'){
  interval(a,-2,2);interval(b,-1,1);interval(c,-1,1);const f=(x:number)=>a*x*x,tangent=(x:number)=>a*b*b+2*a*b*(x-b);
  return {statement:`f(x)=(${a})x²; base x=${b}; comparison y=${c}`,stats:{'f(y)':f(c),'f(x)+f′(x)(y−x)':tangent(c),'exact tangent gap a(y−x)²':a*(c-b)**2,'1{convex on R}':a>=0?1:0,'strong convexity m (a>0)':a>0?2*a:null},curves:[samples(81,i=>{const x=-1+i/40;return [x,f(x)];}),[[-1,tangent(-1)],[1,tangent(1)]]]};
 }
 if(kind==='limits-integration-interchange'){
  integer(a,0,1);integer(b,2,64);const spike=a===0;
  return {statement:spike?`f_n(x)=n·max(0,1−|nx−1|); n=${b}; x∈[0,1]`:`f_n(x)=x/n; n=${b}; x∈[0,1]`,stats:{'∫ f_n':spike?1:1/(2*b),'lim ∫ f_n':spike?1:0,'∫ lim f_n':0,'peak height':spike?b:1/b,'1{limit and integral agree}':spike?0:1},curves:[spike?[[0,0],[1/b,b],[2/b,0],[1,0]]:[[0,0],[1,1/b]],[[0,0],[1,0]]]};
 }
 if(kind==='implicit-inverse-functions'){
  integer(a,0,1);interval(b,-1,1);
  if(a===0){const y=Math.sqrt(Math.max(0,1-b*b));return {statement:`F(x,y)=x²+y²−1=0; upper branch; x₀=${b}`,stats:{'y(x₀)':y,'F_y=2y':2*y,'dy/dx':y===0?null:-b/y,'1{implicit theorem condition}':y>0?1:0},curves:[samples(101,i=>{const x=-1+i*.02;return [x,Math.sqrt(Math.max(0,1-x*x))];})]};}
  return {statement:`f(x)=x³; inverse real cube root; derivative at y=f(${b})`,stats:{'f(x₀)':b**3,'f′(x₀)':3*b*b,'inverse derivative at f(x₀)':b===0?null:1/(3*b*b),'1{locally invertible}':1},curves:[samples(101,i=>{const x=-1+i*.02;return [x,x**3];}),samples(101,i=>{const x=-1+i*.02;return [x,Math.cbrt(x)];})]};
 }
 if(kind==='matrix-trace-logdet-derivatives'){
  interval(a,1,3);interval(b,-2,2);interval(c,1,3);interval(d,0,.1);const det=a*c-b*b,pd=det>0,logdet=(t:number)=>{const value=(a+t)*c-b*b;return value>0?Math.log(value):null;},left=logdet(-d),right=logdet(d);
  const curve=samples(101,i=>{const t=-.5+i*.01;return [t,logdet(t)??NaN];}).filter(v=>Number.isFinite(v[1]));
  return {statement:`A=[${a},${b};${b},${c}]; perturb A₀₀ by t; h=${d}`,stats:{'det A':det,'log det A (SPD domain)':pd?Math.log(det):null,'d logdet/da':pd?c/det:null,'central difference':d>0&&left!==null&&right!==null?(right-left)/(2*d):null,'∇_A tr(AᵀA) row-major':[2*a,2*b,2*b,2*c],'d tr(AᵀA)/db (shared b)':4*b},curves:curve.length?[curve]:[]};
 }
 if(kind==='vector-fields-divergence-curl'){
  p.forEach(x=>interval(x,-2,2));return {statement:`F(x,y)=(${a}x+${b}y, ${c}x+${d}y); arrow scale=0.3`,stats:{'div F':a+d,'curl_z F':c-b,'Jacobian row-major':[a,b,c,d],'1{curl-free on R²}':b===c?1:0},curves:[-1,0,1].flatMap(x=>[-1,0,1].map(y=>[[x,y],[x+.3*(a*x+b*y),y+.3*(c*x+d*y)]] as Vec2[]))};
 }
 if(kind==='line-surface-volume-integrals'){
  integer(a,0,1);interval(b,0,2);const field=(x:number,y:number)=>a===0?[-y,x]:[x,y];
  return {statement:`F=${a===0?'(−y,x)':'(x,y)'}; circle radius R=${b}; CCW tangent, outward normal`,stats:{'∮ F·dr':a===0?2*Math.PI*b*b:0,'∮ F·n ds':a===1?2*Math.PI*b*b:0,'∫_disk 1 dA':Math.PI*b*b,'circle length':2*Math.PI*b},curves:[samples(81,i=>{const t=i/80*2*Math.PI;return [b*Math.cos(t),b*Math.sin(t)];}),...samples(8,i=>{const t=i/8*2*Math.PI;return [b*Math.cos(t),b*Math.sin(t)];}).map(([x,y])=>{const f=field(x,y);return [[x,y],[x+.25*f[0],y+.25*f[1]]] as Vec2[];})]};
 }
 if(kind==='gauss-stokes-theorems'){
  interval(a,0,2);interval(b,-2,2);interval(c,-2,2);let sphere=0;const n=64,h=Math.PI/n;for(let k=0;k<=n;k++)sphere+=(k===0||k===n?1:k%2?4:2)*Math.sin(k*h);sphere*=h/3*2*Math.PI*b*a**3;
  let circulation=0;const step=2*Math.PI/128;for(let k=0;k<128;k++){const t=(k+.5)*step,x=a*Math.cos(t),y=a*Math.sin(t),fx=b*x-c*y,fy=b*y+c*x;circulation+=(fx*(-a*Math.sin(t))+fy*a*Math.cos(t))*step;}
  return {statement:`F=a(x,y,z)+b(−y,x,0); a=${b}, b=${c}, R=${a}; picture: equatorial cross-section`,stats:{'div F':3*b,'curl_z F':2*c,'sphere flux (Simpson)':sphere,'∫_ball div F':4*Math.PI*b*a**3,'CCW circulation (midpoint)':circulation,'∫_disk curl·n dA':2*Math.PI*c*a*a},curves:[samples(81,i=>{const t=i/80*2*Math.PI;return [a*Math.cos(t),a*Math.sin(t)];})]};
 }
 interval(a,-2,2);interval(b,-2,2);interval(c,-1,1);interval(d,-1,1);const f=(x:number,y:number)=>.5*x*x*y+Math.sin(y),value=f(a,b),gradient=[a*b,.5*a*a+Math.cos(b)],hv=[b*c+a*d,a*c-Math.sin(b)*d],curvature=c*hv[0]+d*hv[1],slope=gradient[0]*c+gradient[1]*d;
 return {statement:`f(x,y)=½x²y+sin y; base=(${a},${b}); v=(${c},${d})`,stats:{'f':value,'gradient':gradient,'H row-major':[b,a,a,-Math.sin(b)],'Hv':hv,'vᵀHv':curvature},curves:[samples(101,i=>{const t=-.5+i*.01;return [t,f(a+t*c,b+t*d)];}),samples(101,i=>{const t=-.5+i*.01;return [t,value+slope*t+.5*curvature*t*t];})]};
}
