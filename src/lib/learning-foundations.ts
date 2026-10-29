import { symmetricEigen2 } from './learning-linear-algebra';
import type { Vec2 } from './learning-simulations';
export const foundationKinds = ['svd','probability-distributions','expectation-variance','statistical-estimation','bayesian-inference','information-theory','regularization','stochastic-optimization','differential-equations','mechanics-energy','thermodynamics','stochastic-processes','inverse-problems','research-methods'] as const;
export type FoundationKind = typeof foundationKinds[number];
export function isFoundationKind(kind:string):kind is FoundationKind {return (foundationKinds as readonly string[]).includes(kind);}
export interface FoundationResult { stats:Record<string,number|number[]|null>; curves:Vec2[][]; statement?:string;holes?:Vec2[];curveLabels?:string[]; warning?:'support'|'rank'|'stationary'|'stability'|'uncertainty'|'zero-divisor'|'undefined-angle'|'degenerate-equation'|'function-domain'|'infinite-sum'|'asymptotic-bound'|'numeric-range'; }
const dot=(a:number[],b:number[])=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const range=(n:number,f:(i:number)=>Vec2)=>Array.from({length:n},(_,i)=>f(i));
function probability(p:number){if(p<0||p>1)throw new Error('Probability outside [0,1].');}
function logFactorial(n:number){let sum=0;for(let i=2;i<=n;i++)sum+=Math.log(i);return sum;}
function entropy(p:number){return p===0||p===1?0:-p*Math.log(p)-(1-p)*Math.log1p(-p);}
function logLikelihood(theta:number,k:number,n:number){return (k===0?0:k*Math.log(theta))+(k===n?0:(n-k)*Math.log1p(-theta));}
export function svd2(a:number,b:number,c:number,d:number) {
 const e=symmetricEigen2(a*a+c*c,a*b+c*d,b*b+d*d);const s=e.values.map(x=>Math.sqrt(Math.max(0,x)));const vs=[e.first,e.second];
 const mapped=vs.map(v=>[a*v[0]+b*v[1],c*v[0]+d*v[1]] as Vec2);
 // Compute the small singular value from det/sigma1 to avoid cancellation in AᵀA.
 if(s[0]>0)s[1]=Math.abs(a*d-b*c)/s[0];
 const u1:Vec2=s[0]>0?[mapped[0][0]/s[0],mapped[0][1]/s[0]]:[1,0];
 const u2:Vec2=s[1]>0?[mapped[1][0]/s[1],mapped[1][1]/s[1]]:[-u1[1],u1[0]];
 const rankOne=[s[0]*u1[0]*vs[0][0],s[0]*u1[0]*vs[0][1],s[0]*u1[1]*vs[0][0],s[0]*u1[1]*vs[0][1]];
 const reconstructed=rankOne.map((x,i)=>x+s[1]*u2[Math.floor(i/2)]*vs[1][i%2]);
 return {singular:s,right:vs,left:[u1,u2],rankOne,reconstructed,error:Math.hypot(...[a,b,c,d].map((x,i)=>x-rankOne[i])),condition:s[1]===0?null:s[0]/s[1]};
}
/** All examples are deliberately small, deterministic and evaluated independently of rendering. */
export function evaluateFoundation(kind:FoundationKind,p:number[]):FoundationResult {
 const counts:Record<FoundationKind,number>={'svd':4,'probability-distributions':2,'expectation-variance':3,'statistical-estimation':3,'bayesian-inference':2,'information-theory':2,'regularization':3,'stochastic-optimization':3,'differential-equations':3,'mechanics-energy':3,'thermodynamics':3,'stochastic-processes':3,'inverse-problems':3,'research-methods':2};
 if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Inputs must be finite.');
 const [a,b,c,d]=p;
 if(kind==='svd'){
  const s=svd2(a,b,c,d);const circle=range(81,i=>{const t=i/80*2*Math.PI;return [Math.cos(t),Math.sin(t)];});
  return {stats:{'σ₁, σ₂':s.singular,'A₁ (row-major)':s.rankOne,'‖A−A₁‖F':s.error,'κ₂(A)':s.condition,'‖A−UΣVᵀ‖F':Math.hypot(...[a,b,c,d].map((x,i)=>x-s.reconstructed[i]))},curves:[circle.map(v=>[a*v[0]+b*v[1],c*v[0]+d*v[1]]),circle.map(v=>[s.rankOne[0]*v[0]+s.rankOne[1]*v[1],s.rankOne[2]*v[0]+s.rankOne[3]*v[1]])],warning:s.condition===null?'rank':undefined};
 }
 if(kind==='probability-distributions'){
  probability(a);if(!Number.isInteger(b)||b<1||b>30)throw new Error('Use 1–30 trials.');
  const pmf=range(b+1,k=>[k,Math.exp(logFactorial(b)-logFactorial(k)-logFactorial(b-k))*a**k*(1-a)**(b-k)]);
  return {stats:{'Σ P(K=k)':pmf.reduce((s,v)=>s+v[1],0),'E[K]':b*a,'Var(K)':b*a*(1-a),'P(K=0)':pmf[0][1],'P(K=n)':pmf[b][1]},curves:[pmf]};
 }
 if(kind==='expectation-variance'){
  if(p.some(x=>x<0)||a+b+c<=0)throw new Error('Weights must have positive total.');const weights=p.map(x=>x/(a+b+c));const xs=[-1,0,2];const mean=dot(xs,weights);const second=dot(xs.map(x=>x*x),weights);const variance=second-mean*mean;
  return {stats:{'P(X=−1,0,2)':weights,'E[X]':mean,'E[X²]':second,'Var(X)':variance,'SD(X)':Math.sqrt(Math.max(0,variance))},curves:[xs.map((x,i)=>[x,weights[i]]),[[mean,0],[mean,1]]]};
 }
 if(kind==='statistical-estimation'){
  probability(a);if(!Number.isInteger(b)||!Number.isInteger(c)||b<1||c<0||c>b)throw new Error('Use integer 0≤k≤n.');const mle=c/b;const logL=logLikelihood(a,c,b);
  return {stats:{'n, k':[b,c],'p̂MLE':mle,'ℓ(p)':logL,'ℓ(p̂)':logLikelihood(mle,c,b),'SE plug-in':Math.sqrt(mle*(1-mle)/b)},curves:[range(99,i=>{const theta=(i+1)/100;return [theta,Math.exp(logLikelihood(theta,c,b)-logLikelihood(mle,c,b))];})],warning:!Number.isFinite(logL)?'support':undefined};
 }
 if(kind==='bayesian-inference'){
  if(!Number.isInteger(a)||!Number.isInteger(b)||a<1||b<1||a>20||b>20)throw new Error('Use positive integer prior shapes.');
  const alpha=a+7,beta=b+3;const density=(x:number,aa:number,bb:number)=>Math.exp((aa-1)*Math.log(x)+(bb-1)*Math.log1p(-x)+logFactorial(aa+bb-1)-logFactorial(aa-1)-logFactorial(bb-1));
  return {stats:{'αpost, βpost':[alpha,beta],'E[p|D]':alpha/(alpha+beta),'Var(p|D)':alpha*beta/((alpha+beta)**2*(alpha+beta+1)),'P(next=1|D)':alpha/(alpha+beta)},curves:[range(99,i=>[(i+1)/100,density((i+1)/100,a,b)]),range(99,i=>[(i+1)/100,density((i+1)/100,alpha,beta)])]};
 }
 if(kind==='information-theory'){
  probability(a);probability(b);const ce=(a===0?0:-a*Math.log(b))+(a===1?0:-(1-a)*Math.log1p(-b));const h=entropy(a);const kl=ce-h;
  return {stats:{'H(P) [nat]':h,'H(P,Q) [nat]':ce,'D_KL(P‖Q) [nat]':Math.abs(kl)<1e-14?0:kl},curves:[range(101,i=>[i/100,entropy(i/100)]),[[a,0],[a,h]]],warning:!Number.isFinite(ce)?'support':undefined};
 }
 if(kind==='regularization'){
  if(a<0||b<0)throw new Error('Use nonnegative penalty and weak scale.');const weights=(lambda:number)=>[9/(9+lambda),b*b+lambda===0?0:b*c/(b*b+lambda)];const w=weights(a);const residual=[3-3*w[0],c-b*w[1]];
  return {stats:{'wλ':w,'½‖y−Xw‖²':.5*dot(residual,residual),'λ‖w‖²/2':a/2*dot(w,w),'κ₂(XᵀX+λI)':b*b+a===0?null:(9+a)/(b*b+a)},curves:[range(81,i=>[i/20,weights(i/20)[0]]),range(81,i=>[i/20,weights(i/20)[1]])],warning:a===0&&b===0?'rank':undefined};
 }
 if(kind==='stochastic-optimization'){
  if(a<=0||!Number.isInteger(c)||c<1)throw new Error('Use positive rate and integer seed.');let state=c>>>0;const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};const data=[-1,1,3];let gd=b,sgd=b;const g:Vec2[]=[[0,gd]],s:Vec2[]=[[0,sgd]];
  for(let i=1;i<=30;i++){gd-=a*(gd-1);sgd-=a*(sgd-data[Math.floor(random()*3)]);g.push([i,gd]);s.push([i,sgd]);}
  const loss=(x:number)=>data.reduce((sum,z)=>sum+.5*(x-z)**2,0)/3;
  return {stats:{'E[gᵢ(w₀)]':b-1,'Var(gᵢ)':8/3,'wGD, wSGD':[gd,sgd],'L(wGD), L(wSGD)':[loss(gd),loss(sgd)]},curves:[g,s],warning:a>=2?'stability':undefined};
 }
 if(kind==='differential-equations'){
  if(a<=0||b<=0)throw new Error('Use positive decay and step.');const exact=range(21,i=>[i*b,c*Math.exp(-a*i*b)]);const numeric=range(21,i=>[i*b,c*(1-a*b)**i]);
  return {stats:{'kΔt':a*b,'xexact(T)':exact[20][1],'xEuler(T)':numeric[20][1],'|error(T)|':Math.abs(exact[20][1]-numeric[20][1])},curves:[exact,numeric],warning:a*b>=2?'stability':undefined};
 }
 if(kind==='mechanics-energy'){
  if(a<=0||b<=0)throw new Error('Use positive step and stiffness.');let q=c,v=0;const exact:Vec2[]=[[0,c]],numeric:Vec2[]=[[0,c]];const initial=.5*b*c*c;let maxError=0;
  for(let i=1;i<=100;i++){v-=a*b*q;q+=a*v;maxError=Math.max(maxError,Math.abs(.5*v*v+.5*b*q*q-initial));numeric.push([i*a,q]);exact.push([i*a,c*Math.cos(Math.sqrt(b)*i*a)]);}
  return {stats:{'m [kg]':1,'q(T), v(T)':[q,v],'E₀ [J]':initial,'E(T) [J]':.5*v*v+.5*b*q*q,'max |E−E₀| [J]':maxError,'ωΔt':Math.sqrt(b)*a},curves:[exact,numeric],warning:Math.sqrt(b)*a>=2?'stability':undefined};
 }
 if(kind==='thermodynamics'){
  if(a<=0||b<=0||!Number.isInteger(c)||c<1)throw new Error('Use positive gap, temperature and integer degeneracy.');const at=(temp:number)=>{const weight=c*Math.exp(-a/temp);const z=1+weight;const p1=weight/z;const free=-temp*Math.log1p(weight);const u=a*p1;return {z,p1,free,u,s:(u-free)/temp};};const r=at(b);
  return {stats:{'Z':r.z,'P(E=Δ)':r.p1,'U':r.u,'F':r.free,'S/kB':r.s,'H(binary occupancy)':entropy(r.p1)},curves:[range(100,i=>{const temp=.05+(i/99)*4.95;return [temp,at(temp).p1];}),range(100,i=>{const temp=.05+(i/99)*4.95;return [temp,at(temp).s];})]};
 }
 if(kind==='stochastic-processes'){
  probability(a);probability(b);probability(c);let current=c;const path:Vec2[]=[[0,c]];for(let i=1;i<=30;i++){current=a+(1-a-b)*current;path.push([i,current]);}const stationary=a+b===0?null:a/(a+b);
  return {stats:{'p₀':c,'p₃₀':current,'π₁':stationary,'λ₂':1-a-b},curves:[path,...(stationary===null?[]:[[[0,stationary],[30,stationary]] as Vec2[]])],warning:a+b===0?'stationary':a===1&&b===1?'stability':undefined};
 }
 if(kind==='inverse-problems'){
  if(a<0||c<0)throw new Error('Use nonnegative observation scale and regularizer.');const sum=3/(1+c/2);const contrast=2*a*a+c===0?0:2*a*(-a+b)/(2*a*a+c);const estimate=[(sum+contrast)/2,(sum-contrast)/2];
  return {stats:{'y':[3,-a+b],'θtrue':[1,2],'θ̂':estimate,'‖θ̂−θtrue‖₂':Math.hypot(estimate[0]-1,estimate[1]-2),'κ₂(X)':a===0?null:1/a},curves:[range(81,i=>{const noise=-.2+i*.005;const dif=2*a*a+c===0?0:2*a*(-a+noise)/(2*a*a+c);return [noise,(sum+dif)/2];}),[[-.2,1],[.2,1]]],warning:a===0&&c===0?'rank':undefined};
 }
 // A paired, four-seed educational example; t interval assumes independent normal paired differences.
 if(b<0)throw new Error('Use nonnegative variation.');const effects=[a,a+b,a-b,a];const se=b/Math.sqrt(6);const critical=3.182446305;const ci=[a-critical*se,a+critical*se];
 return {stats:{'paired effects':effects,'mean Δ':a,'sample SD':b*Math.sqrt(2/3),'SE(mean)':se,'95% t CI (df=3)':ci},curves:[effects.map((x,i)=>[i+1,x]),[[1,a],[4,a]]],warning:ci[0]<=0&&ci[1]>=0?'uncertainty':undefined};
}
