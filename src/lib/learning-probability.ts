import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';
export const probabilityKinds=['probability-spaces-events','conditional-probability-independence','mass-density-cdf','joint-marginal-conditional','multivariate-gaussians','random-variable-transformations','laws-large-numbers-clt','reparameterization-gradients'] as const;
export type ProbabilityKind=typeof probabilityKinds[number];
export const isProbabilityKind=(k:string):k is ProbabilityKind=>(probabilityKinds as readonly string[]).includes(k);
export interface ProbabilityResult extends FoundationResult {plot:'discrete'|'continuous'|'joint'|'ellipse';cdf?:boolean;table:{headers:string[];rows:(number|string)[][]};grid?:{x:number;y:number;p:number}[];gridLabels?:[string,string];jumps?:{x:number;from:number;to:number}[];}
const indexed=(v:number[]):Vec2[]=>v.map((x,i)=>[i,x]);
const clip=(x:number)=>Math.min(1,Math.max(0,x));
const ratio=(a:number,b:number)=>b>0?a/b:null;
export function normalCdf(x:number){if(!Number.isFinite(x))throw new Error('Use a finite normal query.');if(x===0)return .5;const t=1/(1+.2316419*Math.abs(x)),poly=t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));const tail=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI)*poly;return x>0?1-tail:tail;}
export function binomialMass(n:number,p:number):number[]{if(!Number.isInteger(n)||n<1||n>30||!Number.isFinite(p)||p<0||p>1)throw new Error('Invalid binomial parameters.');let choose=1;return Array.from({length:n+1},(_,k)=>{if(k>0)choose=choose*(n-k+1)/k;return choose*p**k*(1-p)**(n-k);});}
// Interpret the displayed decimal controls exactly when deciding inclusive event boundaries.
function decimalParts(value:number):[bigint,bigint]{const [coefficient,exponent='0']=value.toString().split('e'),fraction=coefficient.split('.')[1]?.length??0,digits=BigInt(coefficient.replace('.','')),power=Number(exponent)-fraction;return power>=0?[digits*BigInt(10)**BigInt(power),BigInt(1)]:[digits,BigInt(10)**BigInt(-power)];}
export function binomialDeviationIndices(n:number,p:number,epsilon:number):number[]{if(!Number.isInteger(n)||n<1||n>30||!Number.isFinite(p)||p<0||p>1||!Number.isFinite(epsilon)||epsilon<=0||epsilon>1)throw new Error('Invalid deviation event.');const [pn,pd]=decimalParts(p),[en,ed]=decimalParts(epsilon),N=BigInt(n);return Array.from({length:n+1},(_,k)=>k).filter(k=>{const difference=BigInt(k)*pd-N*pn;return (difference<BigInt(0)?-difference:difference)*ed>=N*en*pd;});}
export function evaluateProbability(kind:ProbabilityKind,p:number[]):ProbabilityResult{
 const specs:Record<ProbabilityKind,[number,number,boolean][]>={'probability-spaces-events':[[0,2,true],[0,15,true],[0,15,true]],'conditional-probability-independence':Array(4).fill([0,10,true]),'mass-density-cdf':[[0,1,true],[.5,3,false],[-1,4,false]],'joint-marginal-conditional':Array(4).fill([0,8,true]),'multivariate-gaussians':[[.5,2,false],[.5,2,false],[-1,1,false],[-2,2,false]],'random-variable-transformations':[[0,1,true],[-2,2,true],[-2,2,false]],'laws-large-numbers-clt':[[1,30,true],[0,1,false],[.1,1,false]],'reparameterization-gradients':[[-2,2,false],[0,2,false],[-2,2,false]]};
 if(p.length!==specs[kind].length||p.some((x,i)=>!Number.isFinite(x)||x<specs[kind][i][0]||x>specs[kind][i][1]||(specs[kind][i][2]&&!Number.isInteger(x))))throw new Error('Use probability parameters inside their stated domains.');const [a,b,c,d]=p;
 if(kind==='probability-spaces-events'){
  const weights=a===0?[1,1,1,1]:a===1?[1,2,3,4]:[1,1,0,0],total=weights.reduce((s,x)=>s+x,0),mass=weights.map(x=>x/total),prob=(mask:number)=>mass.reduce((s,x,i)=>s+((mask>>i)&1)*x,0);
  return {plot:'discrete',statement:`Ω={0,1,2,3}; mode=${a}; A mask=${b}, B mask=${c}; bit i selects outcome i`,stats:{'P(A)':prob(b),'P(B)':prob(c),'P(A∩B)':prob(b&c),'P(A∪B)':prob(b|c),'P(Aᶜ)':prob(15^b),'total probability':1},curves:[indexed(mass)],table:{headers:['ω','weight','P({ω})','1_A','1_B'],rows:mass.map((v,i)=>[i,weights[i],v,(b>>i)&1,(c>>i)&1])}};
 }
 if(kind==='conditional-probability-independence'){
  const N=a+b+c+d,prior=ratio(a+c,N),positive=ratio(a+b,N),posterior=ratio(a,a+b),sens=ratio(a,a+c),falsePositive=ratio(b,b+d),valid=N>0;
  return {plot:'joint',gridLabels:['1_A','1_B'],statement:`Counts: A∩B=${a}, Aᶜ∩B=${b}, A∩Bᶜ=${c}, Aᶜ∩Bᶜ=${d}; N=${N}`,stats:{'P(A)':prior,'P(B)':positive,'P(A|B)':posterior,'P(B|A)':sens,'P(B|Aᶜ)':falsePositive,'1{independent}':valid?Number(a*d===b*c):null},curves:valid?[indexed([d,c,b,a].map(v=>v/N))]:[],grid:valid?[{x:0,y:0,p:d/N},{x:1,y:0,p:c/N},{x:0,y:1,p:b/N},{x:1,y:1,p:a/N}]:[],warning:valid?undefined:'function-domain',table:{headers:['1_A','1_B','count','joint probability'],rows:[[1,1,a,ratio(a,N)??'—'],[0,1,b,ratio(b,N)??'—'],[1,0,c,ratio(c,N)??'—'],[0,0,d,ratio(d,N)??'—']]}};
 }
 if(kind==='mass-density-cdf'){
  const bernoulli=a===0,cdf=(x:number)=>bernoulli?(x<0?0:x<1?.75:1):clip(x/b),density=bernoulli?null:(c>=0&&c<=b?1/b:0),mass=bernoulli?(c===0?.75:c===1?.25:0):0;
  const curves:Vec2[][]=bernoulli?[[[-1,0],[0,0]],[[0,.75],[1,.75]],[[1,1],[4,1]]]:[[[-1,0],[0,0],[b,1],[4,1]]];
  return {plot:'continuous',cdf:true,statement:bernoulli?'X~Bernoulli(1/4); width control inactive; plot: right-continuous CDF':`X~Uniform[0,${b}]; density endpoints included by convention; plot: CDF`,stats:{'F(x)=P(X≤x)':cdf(c),'P(X=x)':mass,'Lebesgue density f(x)':density,'support width':bernoulli?null:b},curves,jumps:bernoulli?[{x:0,from:0,to:.75},{x:1,from:.75,to:1}]:undefined,table:{headers:['query x','CDF','point mass','density'],rows:[[c,cdf(c),mass,density??'—']]}};
 }
 if(kind==='joint-marginal-conditional'){
  const N=a+b+c+d,m=[a,b,c,d].map(x=>N>0?x/N:0),px=N>0?m[2]+m[3]:null,py=N>0?m[1]+m[3]:null,cov=N>0?m[3]-px!*py!:null,varx=px===null?null:px*(1-px),vary=py===null?null:py*(1-py),corr=varx&&vary?cov!/Math.sqrt(varx*vary):null;
  return {plot:'joint',statement:`Counts (X,Y)=(0,0),(0,1),(1,0),(1,1): (${p}); N=${N}`,stats:{'P(X=1)':px,'P(Y=1)':py,'P(Y=1|X=1)':ratio(d,c+d),'Cov(X,Y)':cov,'Corr(X,Y)':corr,'1{independent}':N>0?Number(a*d===b*c):null},curves:N>0?[indexed(m)]:[],grid:N>0?m.map((v,i)=>({x:Math.floor(i/2),y:i%2,p:v})):[],warning:N>0?undefined:'function-domain',table:{headers:['X','Y','count','joint mass','P(X)','P(Y)'],rows:p.map((v,i)=>[Math.floor(i/2),i%2,v,N>0?m[i]:'—',N>0?(i<2?m[0]+m[1]:m[2]+m[3]):'—',N>0?(i%2===0?m[0]+m[2]:m[1]+m[3]):'—'])}};
 }
 if(kind==='multivariate-gaussians'){
  const residual=Math.max(0,(1-c)*(1+c)),det=a*a*b*b*residual,ellipse:Vec2[]=Array.from({length:97},(_,i)=>{const t=2*Math.PI*i/96;return [a*Math.cos(t),b*(c*Math.cos(t)+Math.sqrt(residual)*Math.sin(t))];}),mean=c*b/a*d,variance=b*b*residual,density=residual>0?Math.exp(-.5*(d/a)**2/residual)/(2*Math.PI*a*b*Math.sqrt(residual)):null;
  return {plot:'ellipse',statement:`(X,Y) centered Gaussian; σx=${a}, σy=${b}, ρ=${c}; curve=L(unit circle), not samples`,stats:{'covariance row-major':[a*a,c*a*b,c*a*b,b*b],'det Σ':det,'E[Y|X=x]':mean,'Var(Y|X=x)':variance,'2D density at (x,0)':density,'1{nonsingular}':Number(residual>0),'mass inside unit Mahalanobis ellipse':residual>0?1-Math.exp(-.5):null},curves:[ellipse],table:{headers:['query x','conditional mean','conditional variance'],rows:[[d,mean,variance]]}};
 }
 if(kind==='random-variable-transformations'){
  const atom=b===0,cdf=(y:number)=>{if(atom)return y<0?0:1;const u=y/b,z=a===0?clip(u):Math.sqrt(clip(u));return b>0?z:1-z;};
  const u=atom?0:c/b,inside=u>0&&u<1,density=atom?null:a===0?(inside?1/Math.abs(b):0):(inside?1/(2*Math.abs(b)*Math.sqrt(u)):c===0?null:0);
  const curves:Vec2[][]=atom?[[[-2,0],[0,0]],[[0,1],[2,1]]]:a===0?[[[-2,0],[Math.min(0,b),0],[Math.max(0,b),1],[2,1]]]:[Array.from({length:161},(_,i)=>{const y=-2+i/40;return [y,cdf(y)];})];
  return {plot:'continuous',cdf:true,statement:atom?'Y=0 almost surely; point mass, no Lebesgue density':a===0?`X~Uniform[0,1], Y=${b}X; plot: F_Y`:`X~Uniform[-1,1], Y=${b}X²; both preimages contribute; plot: F_Y`,stats:{'F_Y(y)':cdf(c),'Lebesgue density at y':density,'P(Y=y)':atom?Number(c===0):0,'E[Y]':a===0?b/2:b/3,'1{unbounded density at boundary}':Number(!atom&&a===1&&c===0)},curves,jumps:atom?[{x:0,from:0,to:1}]:undefined,table:{headers:['y','CDF','density','point mass'],rows:[[c,cdf(c),density??'—',atom?Number(c===0):0]]}};
 }
 if(kind==='laws-large-numbers-clt'){
  const mass=binomialMass(a,b),varMean=b*(1-b)/a,sd=Math.sqrt(a*b*(1-b)),approx=sd>0?mass.map((_,k)=>normalCdf((k+.5-a*b)/sd)-normalCdf((k-.5-a*b)/sd)):null,tail=binomialDeviationIndices(a,b,c).reduce((s,k)=>s+mass[k],0);
  return {plot:'discrete',statement:`S~Binomial(${a},${b}); sample mean=S/n; ε=${c}; orange=normal bin masses with continuity correction, not density`,stats:{'E[sample mean]':b,'Var(sample mean)':varMean,'P(|mean−p|≥ε)':tail,'Chebyshev bound':Math.min(1,varMean/(c*c)),'normal mass on displayed bins':approx?approx.reduce((s,v)=>s+v,0):null,'standardized variance':sd>0?1:null},curves:[mass.map((v,k)=>[k/a,v]),...(approx?[approx.map((v,k)=>[k/a,v] as Vec2)]:[])],table:{headers:['k','sample mean','exact mass','normal bin approximation'],rows:mass.map((v,k)=>[k,k/a,v,approx?.[k]??'—'])}};
 }
 const z=a+b*c,loss=.5*(z-1)**2,gradient=[z-1,(z-1)*c].map(v=>v===0?0:v),rows=[-1,0,1].map(e=>{const zz=a+b*e;return [e,zz,.5*(zz-1)**2,zz-1,(zz-1)*e];});
 return {plot:'continuous',statement:`ε fixed at ${c}; z=μ+σε; ℓ=½(z−1)²; noise law N(0,1) for expected readouts`,stats:{'z':z,'single-noise loss':loss,'pathwise gradient (μ,σ)':gradient,'exact expected loss':.5*((a-1)**2+b*b),'exact expected gradient':[a-1,b],'two-node moment gradient':[(rows[0][3]+rows[2][3])/2,(rows[0][4]+rows[2][4])/2]},curves:[Array.from({length:81},(_,i)=>{const e=-2+i/20;return [e,.5*(a+b*e-1)**2];})],table:{headers:['fixed ε','z','loss','∂loss/∂μ','∂loss/∂σ'],rows}};
}
