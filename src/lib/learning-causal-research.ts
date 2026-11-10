import type { CausalResult } from './learning-causality';
import type { CausalNetwork, CausalEdge } from './learning-causal-graphs';
import { causalOrder, causalSeparation } from './learning-causal-graphs';
import type { Vec2 } from './learning-simulations';
export const causalResearchKinds=['instrumental-moment-assumptions','natural-controlled-mediation','gaussian-markov-discovery','invariant-mechanism-transport','hidden-confounding-off-policy'] as const;
export type CausalResearchKind=typeof causalResearchKinds[number];
export const isCausalResearchKind=(id:string):id is CausalResearchKind=>(causalResearchKinds as readonly string[]).includes(id);
const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated causal-model domain.');};
const integer=(v:number,lo:number,hi:number)=>{range(v,lo,hi);if(!Number.isInteger(v))throw new Error('Use an integer causal-model choice.');};
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0);
const average=(a:number[])=>sum(a)/a.length;
const ratio=(a:number,b:number)=>{const r=a/b;return b!==0&&Number.isFinite(r)?r:null;};
const bern=(p:number,v:number)=>v?p:1-p;

export function instrumentalMoments(gamma:number,alpha:number,beta:number,delta:number,kappa:number,rho:number,sigma:number){
 [gamma,alpha,delta,sigma].forEach(v=>range(v,0,1));range(beta,-1,1);range(kappa,-.5,.5);range(rho,-.8,.8);
 const covZX=gamma+alpha*rho,varX=gamma*gamma+alpha*alpha+2*gamma*alpha*rho+sigma*sigma,covUX=gamma*rho+alpha;
 const covZY=beta*covZX+delta*rho+kappa,covXY=beta*varX+delta*covUX+kappa*covZX;
 return {covZX,varX,covUX,covZY,covXY,iv:ratio(covZY,covZX),ols:ratio(covXY,varX),truth:beta};
}
export function mediationEffects(a:number,b:number,c:number,d:number,m:number){
 [a,b,c,d].forEach(v=>range(v,-1,1));range(m,-2,2);
 const atoms=[];for(const em of [-.5,.5])for(const ey of [-.25,.25]){const M0=em,M1=a+em,y=(x:number,z:number)=>c*x+b*z+d*x*z+ey;
  atoms.push({em,ey,M0,M1,Y00:y(0,M0),Y10:y(1,M0),Y01:y(0,M1),Y11:y(1,M1),CDE:y(1,m)-y(0,m)});}
 const TE=average(atoms.map(v=>v.Y11-v.Y00)),PNDE=average(atoms.map(v=>v.Y10-v.Y00)),TNIE=average(atoms.map(v=>v.Y11-v.Y10)),PNIE=average(atoms.map(v=>v.Y01-v.Y00)),TNDE=average(atoms.map(v=>v.Y11-v.Y01)),CDE=average(atoms.map(v=>v.CDE));
 return {atoms,TE,PNDE,TNIE,PNIE,TNDE,CDE};
}
const discoveryEdges:CausalEdge[][]=[[[0,1],[1,2]],[[1,0],[1,2]],[[0,1],[2,1]],[[0,1],[1,2],[0,2]]];
export const threeNodeDAGs:CausalEdge[][]=[];
const possible:CausalEdge[]=[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]];
for(let mask=0;mask<64;mask++){const edges=possible.filter((_,i)=>(mask>>i)&1);try{causalOrder(3,edges);threeNodeDAGs.push(edges);}catch{/* Cyclic graphs are not DAG candidates. */}}
export function gaussianDiscovery(mode:number,a:number,b:number,c:number,given:number){
 integer(mode,0,3);range(a,.25,1);range(b,.25,1);range(c,-1,1);integer(given,0,1);
 const T=mode===0?[[1,0,0],[a,1,0],[a*b,b,1]]:mode===1?[[1,a,0],[0,1,0],[0,b,1]]:mode===2?[[1,0,0],[a,1,b],[0,0,1]]:[[1,0,0],[a,1,0],[c+a*b,b,1]];
 const covariance=T.map(row=>T.map(other=>sum(row.map((v,i)=>v*other[i]))));
 const correlation=(i:number,j:number,k:number|null)=>{const s=covariance,cross=s[i][j]-(k===null?0:s[i][k]*s[j][k]/s[k][k]),vi=s[i][i]-(k===null?0:s[i][k]**2/s[k][k]),vj=s[j][j]-(k===null?0:s[j][k]**2/s[k][k]);return cross/Math.sqrt(vi*vj);};
 const queries:{i:number;j:number;k:number|null;rho:number;independent:boolean}[]=[];for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)for(const k of [null,[0,1,2].find(v=>v!==i&&v!==j)!])queries.push({i,j,k,rho:correlation(i,j,k),independent:Math.abs(correlation(i,j,k))<=1e-9});
 const patterns=(edges:CausalEdge[])=>queries.map(q=>causalSeparation(3,edges,q.i,q.j,q.k===null?[]:[q.k]).separated);
 const candidates=threeNodeDAGs.map(edges=>{const implied=patterns(edges),markov=implied.every((v,i)=>!v||queries[i].independent),faithful=markov&&implied.every((v,i)=>v===queries[i].independent);return {edges,implied,markov,faithful};});
 const truePattern=patterns(discoveryEdges[mode]);
 return {T,covariance,queries,candidates,edges:discoveryEdges[mode],marginal:correlation(0,2,null),partial:correlation(0,2,1),selected:correlation(0,2,given?1:null),trueFaithful:truePattern.every((v,i)=>v===queries[i].independent)};
}
export function mechanismTransport(source:number,target:number,e0:number,e1:number,shift:number){
 [source,target,e0,e1].forEach(v=>range(v,0,1));range(shift,-.1,.1);
 const P=[1-source,source],Q=[1-target,target],e=[e0,e1],q=(z:number,x:number)=>.1+.4*z+(.1+.2*z)*x;
 const cells:{z:number;x:number;y:number;mass:number}[]=[];for(const z of [0,1])for(const x of [0,1])for(const y of [0,1])cells.push({z,x,y,mass:P[z]*bern(e[z],x)*bern(q(z,x),y)});
 const rates=[0,1].map(z=>[0,1].map(x=>{const group=cells.filter(v=>v.z===z&&v.x===x),mass=sum(group.map(v=>v.mass));return mass>0?sum(group.map(v=>v.mass*v.y))/mass:null;}));
 let transported:number|null=0;for(const z of [0,1])if(Q[z]>0){if(rates[z].some(v=>v===null)){transported=null;break;}transported+=Q[z]*(rates[z][1]!-rates[z][0]!);}
 return {P,Q,e,q,cells,rates,transported,sourceATE:.1+.2*source,targetATE:.1+.2*target+shift,invariant:shift===0};
}
export function offPolicyConfounding(pi:number,e0:number,e1:number,policy:number){
 [pi,e0,e1,policy].forEach(v=>range(v,0,1));const P=[1-pi,pi],e=[e0,e1],target=[1-policy,policy],q=(u:number,a:number)=>.1+.6*u+.2*a;
 const masses=P.map((p,u)=>[p*(1-e[u]),p*e[u]]),marginals=[sum(masses.map(v=>v[0])),sum(masses.map(v=>v[1]))];
 const marginalSupport=target.every((p,a)=>p===0||marginals[a]>0),conditionalSupport=P.every((p,u)=>p===0||target.every((t,a)=>t===0||bern(e[u],a)>0));
 // Integrate weighted outcomes algebraically, before forming potentially overflowing raw weights.
 const naive=marginalSupport?sum([0,1].map(a=>target[a]===0?0:sum([0,1].map(u=>target[a]*(masses[u][a]/marginals[a])*q(u,a))))):null;
 const conditional=conditionalSupport?sum([0,1].map(u=>sum([0,1].map(a=>P[u]*target[a]*q(u,a))))):null;
 const truth=.1+.6*pi+.2*policy,logging=sum([0,1].map(u=>sum([0,1].map(a=>masses[u][a]*q(u,a)))));
 return {P,e,target,q,masses,marginals,marginalSupport,conditionalSupport,naive,conditional,truth,logging};
}
const triangle=(names=['X','M','Y']):CausalNetwork=>({nodes:[{name:names[0],x:75,y:170},{name:names[1],x:175,y:45},{name:names[2],x:275,y:170}],edges:[{from:0,to:1},{from:1,to:2},{from:0,to:2}]});
export function evaluateCausalResearch(kind:CausalResearchKind,p:number[]):CausalResult{
 const dimensions:Record<CausalResearchKind,number>={'instrumental-moment-assumptions':7,'natural-controlled-mediation':5,'gaussian-markov-discovery':5,'invariant-mechanism-transport':5,'hidden-confounding-off-policy':4};if(p.length!==dimensions[kind])throw new Error('Use the complete causal research vector.');
 const [a,b,c,d,e,f,g]=p;
 if(kind==='instrumental-moment-assumptions'){
  const r=instrumentalMoments(a,b,c,d,e,f,g),series=Array.from({length:41},(_,i)=>{const k=-.5+i/40,v=instrumentalMoments(a,b,c,d,k,f,g);return {k,v};});
  const segments:Vec2[][]=[];let current:Vec2[]=[];for(const row of series){if(row.v.iv!==null&&Math.abs(row.v.iv)<=5)current.push([row.k,row.v.iv]);else if(current.length){segments.push(current);current=[];}}if(current.length)segments.push(current);
  return {plot:'continuous',graph:'instrument',intervention:false,network:{nodes:[{name:'H',x:175,y:25,hidden:true},{name:'Z',x:65,y:65},{name:'U',x:285,y:65,hidden:true},{name:'X',x:65,y:185},{name:'Y',x:285,y:185}],edges:[{from:0,to:1},...(f===0?[]:[{from:0,to:2}]),...(a===0?[]:[{from:1,to:3}]),...(b===0?[]:[{from:2,to:3}]),...(d===0?[]:[{from:2,to:4}]),...(c===0?[]:[{from:3,to:4}]),...(e===0?[]:[{from:1,to:4}])]},
   statement:`Centered unit-variance independent H,V,E_X,E_Y; Z=H; U=rho H+sqrt(1-rho^2)V; X=gamma Z+alpha U+sigma E_X; Y=beta X+delta U+kappa Z+E_Y. Population IV and OLS moments, not finite-sample unbiasedness. Plot only IV values within [-5,5]; out-of-window values are omitted. A zero denominator or nonfinite machine ratio is unavailable.`,
   stats:{'structural treatment effect beta':r.truth,'population IV ratio':r.iv,'population OLS slope':r.ols,'instrument first-stage covariance':r.covZX,'instrument outcome covariance':r.covZY,'treatment variance':r.varX,'1{exclusion kappa=0}':Number(e===0),'1{instrument independent of U}':Number(f===0),'plot IV points omitted':series.filter(row=>row.v.iv===null||Math.abs(row.v.iv)>5).length},
   curves:[...segments,[[-.5,c],[.5,c]]],curveColors:[...segments.map(()=>'blue' as const),'orange'],dots:[...segments.filter(v=>v.length===1).flat(),...(r.iv!==null&&Math.abs(r.iv)<=5?[[e,r.iv] as Vec2]:[])],dotColors:Array(segments.filter(v=>v.length===1).length+Number(r.iv!==null&&Math.abs(r.iv)<=5)).fill('blue'),bounds:[-.5,.5,-5.05,5.05],table:{headers:['moment','value'],rows:[['Cov(Z,X)',r.covZX],['Cov(Z,Y)',r.covZY],['Cov(X,Y)',r.covXY],['Var(X)',r.varX],['IV',r.iv??'—'],['OLS',r.ols??'—']]}};
 }
 if(kind==='natural-controlled-mediation'){
  const r=mediationEffects(a,b,c,d,e);return {plot:'continuous',graph:'mediation',intervention:true,network:triangle(),statement:`Independent mean-zero mediator noise ±0.5 and outcome noise ±0.25, each pair probability 1/4. M_x=a x+E_M; Y(x,m)=c x+b m+d x m+E_Y. Natural effects compare 0 to 1 under this cross-world coupling. They are model-defined, not automatically identified from arbitrary observational data.`,
   stats:{'total effect':r.TE,'pure natural direct effect':r.PNDE,'total natural indirect effect':r.TNIE,'pure natural indirect effect':r.PNIE,'total natural direct effect':r.TNDE,'controlled direct effect at m':r.CDE,'first decomposition residual':r.TE-r.PNDE-r.TNIE,'second decomposition residual':r.TE-r.PNIE-r.TNDE},
   curves:[Array.from({length:41},(_,i)=>{const m=-2+i/10;return [m,c+d*m] as Vec2;}),[[-2,r.PNDE],[2,r.PNDE]]],curveColors:['blue','orange'],dots:[[e,r.CDE]],dotColors:['blue'],bounds:[-2,2,-3.1,3.1],table:{headers:['E_M','E_Y','M0','M1','Y(0,M0)','Y(1,M0)','Y(0,M1)','Y(1,M1)'],rows:r.atoms.map(v=>[v.em,v.ey,v.M0,v.M1,v.Y00,v.Y10,v.Y01,v.Y11])}};
 }
 if(kind==='gaussian-markov-discovery'){
  const r=gaussianDiscovery(a,b,c,d,e),names=['A','B','C'],label=(q:typeof r.queries[number])=>`${names[q.i]} _|_ ${names[q.j]}${q.k===null?'':` | ${names[q.k]}`}`;
  return {plot:'discrete',graph:'discovery',intervention:false,network:{nodes:[{name:'A',x:75,y:170},{name:'B',x:175,y:45},{name:'C',x:275,y:170}],edges:r.edges.map(([from,to])=>({from,to}))},statement:`Three-node linear Gaussian SCM with independent unit-variance innovations. Modes: 0 A->B->C, 1 A<-B->C, 2 A->B<-C, 3 A->B->C plus c*A in C. All 25 three-node DAGs are checked against exact Gaussian CI correlations with tolerance 1e-9; this is not a finite-sample CI test. Declared graph, Markov compatibility and faithfulness are separate.`,
   stats:{'marginal A C correlation':r.marginal,'A C partial correlation given B':r.partial,'selected correlation':r.selected,'Markov CI-compatible DAG count':r.candidates.filter(v=>v.markov).length,'faithful CI-compatible DAG count':r.candidates.filter(v=>v.faithful).length,'1{declared graph faithful}':Number(r.trueFaithful),'candidate DAG count':25},
   curves:[[[0,r.marginal],[1,r.partial]],[[0,0],[1,0]]],curveColors:['blue','orange'],dots:[[e,r.selected]],dotColors:['blue'],bounds:[0,1,-1.05,1.05],table:{headers:['DAG','implied independences','Markov compatible','faithful compatible'],rows:r.candidates.map(v=>[v.edges.length?v.edges.map(([i,j])=>`${names[i]}->${names[j]}`).join('; '):'empty',v.implied.flatMap((x,i)=>x?[label(r.queries[i])]:[]).join('; ')||'none',Number(v.markov),Number(v.faithful)])}};
 }
 if(kind==='invariant-mechanism-transport'){
  const r=mechanismTransport(a,b,c,d,e),levels=Array.from({length:21},(_,i)=>({t:i/20,r:mechanismTransport(a,i/20,c,d,e)})),orange=levels.flatMap(v=>v.r.transported===null?[]:[[v.t,v.r.transported] as Vec2]);
  return {plot:'continuous',graph:'transport',intervention:false,network:{nodes:[{name:'E',x:175,y:30},{name:'Z',x:65,y:100},{name:'X',x:65,y:195},{name:'Y',x:285,y:195}],edges:[{from:0,to:1},{from:1,to:2},{from:1,to:3},{from:2,to:3},...(e===0?[]:[{from:0,to:3}])]},statement:`Source and target group probabilities differ. Source q(z,x)=0.1+0.4z+(0.1+0.2z)x; target q*=q+shift*x. Source treatment propensities e0,e1; independent outcome noise given Z,X. Transport uses actual source cell ratios and target P(Z), without parametric extrapolation into missing cells. E is environment, not a selected collider.`,
   stats:{'source population ATE':r.sourceATE,'target oracle ATE':r.targetATE,'transported source-cell ATE':r.transported,'transport minus target':r.transported===null?null:r.transported-r.targetATE,'1{target strata and treatments supported}':Number(r.transported!==null),'1{outcome mechanism invariant}':Number(r.invariant)},
   curves:[levels.map(v=>[v.t,v.r.targetATE] as Vec2),orange],curveColors:['blue','orange'],dots:[[b,r.targetATE],...(r.transported===null?[]:[[b,r.transported] as Vec2]),...(orange.length===1?orange:[])],dotColors:['blue',...(r.transported===null?[]:['orange' as const]),...(orange.length===1?['orange' as const]:[])],bounds:[0,1,-.05,.45],
   table:{headers:['Z','X','source P(Z,X)','source conditional mean','target oracle conditional mean','target P(Z)'],rows:[0,1].flatMap(z=>[0,1].map(x=>[z,x,r.P[z]*bern(r.e[z],x),r.rates[z][x]??'—',r.q(z,x)+e*x,r.Q[z]]))}};
 }
 const r=offPolicyConfounding(a,b,c,d),levels=Array.from({length:21},(_,i)=>({t:i/20,r:offPolicyConfounding(a,b,c,i/20)})),orange=levels.flatMap(v=>v.r.naive===null?[]:[[v.t,v.r.naive] as Vec2]);
 const network=triangle(['X','U','Y']);network.nodes[1].hidden=true;
 // triangle() has X->U; the logging graph instead has latent U->X.
 network.edges=[...(b===c?[]:[{from:1,to:0}]),{from:1,to:2},{from:0,to:2}];
 return {plot:'continuous',graph:'policy',intervention:false,network,statement:`Hidden U~Bern(pi); logging P(X=1|U)=e_U; q(U,X)=0.1+0.6U+0.2X. Target assigns X~Bern(r) independently of U. Marginal IPS uses logging P(X); conditional IPS requires observed/known U and P(X|U), and is only an oracle here. Target-zero actions need no overlap. Exact weighted population integration, not finite-sample estimation.`,
  stats:{'target oracle policy value':r.truth,'marginal propensity IPS value':r.naive,'conditional propensity oracle IPS value':r.conditional,'marginal IPS bias':r.naive===null?null:r.naive-r.truth,'logging policy value':r.logging,'1{marginal action support}':Number(r.marginalSupport),'1{conditional action support}':Number(r.conditionalSupport)},
  curves:[levels.map(v=>[v.t,v.r.truth] as Vec2),orange],curveColors:['blue','orange'],dots:[[d,r.truth],...(r.naive===null?[]:[[d,r.naive] as Vec2]),...(orange.length===1?orange:[])],dotColors:['blue',...(r.naive===null?[]:['orange' as const]),...(orange.length===1?['orange' as const]:[])],bounds:[0,1,0,1.05],table:{headers:['U','X','logging P(U,X)','outcome mean','target action probability','marginal weight','conditional oracle weight'],rows:[0,1].flatMap(u=>[0,1].map(x=>[u,x,r.masses[u][x],r.q(u,x),r.target[x],r.target[x]===0?0:ratio(r.target[x],r.marginals[x])??'—',r.target[x]===0?0:ratio(r.target[x],bern(r.e[u],x))??'—']))}};
}
