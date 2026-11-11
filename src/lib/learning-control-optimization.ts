/** Small scalar, discrete LQR / input-box MPC teaching model; no hidden state constraints. */
export interface ControlPlan {states:number[];inputs:number[];cost:number;gradient:number[];kktResidual:number;}
const check=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated LQR/MPC parameter domain.');};
function solve(A:number[][],b:number[]){const M=A.map((r,i)=>[...r,b[i]]),n=b.length;for(let k=0;k<n;k++){let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[pivot][k]))pivot=i;[M[k],M[pivot]]=[M[pivot],M[k]];const scale=M[k][k];if(!(Math.abs(scale)>1e-14))throw new Error('Positive-definite control Hessian required.');for(let j=k;j<=n;j++)M[k][j]/=scale;for(let i=0;i<n;i++)if(i!==k){const f=M[i][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}}return M.map(r=>r[n]);}
export function scalarRiccati(a:number,q:number,r:number,p:number,H:number){
 const P=Array(H+1).fill(0) as number[],K=Array(H).fill(0) as number[],expandedP=Array(H+1).fill(0) as number[],expandedK=Array(H).fill(0) as number[];
 P[H]=expandedP[H]=p;
 for(let t=H-1;t>=0;t--){
  expandedK[t]=a*expandedP[t+1]/(r+expandedP[t+1]);
  expandedP[t]=q+a*a*r*expandedP[t+1]/(r+expandedP[t+1]);
  // Divide before multiplying subnormal terminal penalties. No numerical floor.
  K[t]=a*(P[t+1]/(r+P[t+1]));
  P[t]=q+a*a*(r/(r+P[t+1]))*P[t+1];
 }
 return {P,K,expandedP,expandedK};
}
export function controlPlanCost(a:number,q:number,r:number,p:number,x0:number,inputs:number[]){const states=[x0];let cost=0;for(const u of inputs){const x=states.at(-1)!;cost+=q*x*x+r*u*u;states.push(a*x+u);}cost+=p*states.at(-1)!**2;return {states,cost};}
export function scalarBoxPlan(a:number,q:number,r:number,p:number,H:number,x0:number,bound:number):ControlPlan{
 const Hmat=Array.from({length:H},(_,i)=>Array.from({length:H},(_,j)=>i===j?r:0)),f=Array(H).fill(0) as number[];
 for(let t=1;t<=H;t++){const w=t===H?p:q,base=a**t*x0;for(let i=0;i<t;i++){const vi=a**(t-1-i);f[i]+=w*vi*base;for(let j=0;j<t;j++)Hmat[i][j]+=w*vi*a**(t-1-j);}}
 let best:number[]|null=null,bestCost=Infinity;
 // Each face fixes inputs to -bound/+bound or solves its free positive-definite block.
 for(let mask=0;mask<3**H;mask++){let code=mask;const u=Array(H).fill(0) as number[],free:number[]=[],fixed:number[]=[];for(let i=0;i<H;i++){const digit=code%3;code=Math.floor(code/3);if(digit===0)free.push(i);else{fixed.push(i);u[i]=digit===1?-bound:bound;}}
  const values=solve(free.map(i=>free.map(j=>Hmat[i][j])),free.map(i=>-f[i]-fixed.reduce((s,j)=>s+Hmat[i][j]*u[j],0)));if(values.some(v=>Math.abs(v)>bound+1e-10))continue;free.forEach((i,k)=>u[i]=Math.max(-bound,Math.min(bound,values[k])));const cost=controlPlanCost(a,q,r,p,x0,u).cost;if(cost<bestCost){bestCost=cost;best=u;}
 }
 if(!best)throw new Error('The input box must be feasible.');const inputs=best,gradient=inputs.map((_,i)=>2*(f[i]+Hmat[i].reduce((s,v,j)=>s+v*inputs[j],0))),kktResidual=Math.max(0,...gradient.map((g,i)=>Math.abs(inputs[i]-Math.max(-bound,Math.min(bound,inputs[i]-g)))));
 return {...controlPlanCost(a,q,r,p,x0,inputs),inputs,gradient,kktResidual};
}
export function lqrMpc(a:number,q:number,r:number,p:number,H:number,x0:number,bound:number,steps:number){
 check(a,.5,1.2);check(q,0,2);check(r,.1,2);check(p,0,3);check(H,1,5);check(x0,-2,2);check(bound,.1,2);check(steps,1,12);if(!Number.isInteger(H)||!Number.isInteger(steps))throw new Error('Use integer planning and simulation horizons.');
 const riccati=scalarRiccati(a,q,r,p,H),clip=(u:number)=>Math.max(-bound,Math.min(bound,u)),firstPlan=scalarBoxPlan(a,q,r,p,H,x0,bound),lqrInputs:number[]=[],clippedInputs:number[]=[];let z=x0;
 for(let t=0;t<H;t++){const u=-riccati.K[t]*z;lqrInputs.push(u);z=a*z+u;}z=x0;for(let t=0;t<H;t++){const u=clip(-riccati.K[t]*z);clippedInputs.push(u);z=a*z+u;}
 const clippedPlan={...controlPlanCost(a,q,r,p,x0,clippedInputs),inputs:clippedInputs},lqrPlan={...controlPlanCost(a,q,r,p,x0,lqrInputs),inputs:lqrInputs};
 let m=x0,c=x0,l=x0;const rows=[];for(let t=0;t<steps;t++){const plan=scalarBoxPlan(a,q,r,p,H,m,bound),um=plan.inputs[0],uc=clip(-riccati.K[0]*c),ul=-riccati.K[0]*l;rows.push({t,mpc:m,clipped:c,unbounded:l,mpcInput:um,clippedInput:uc,unboundedInput:ul,plannedCost:plan.cost,kktResidual:plan.kktResidual});m=a*m+um;c=a*c+uc;l=a*l+ul;}
 return {riccati,firstPlan,clippedPlan,lqrPlan,rows,terminal:{mpc:m,clipped:c,unbounded:l},maxResidual:Math.max(...rows.map(v=>v.kktResidual)),initialPlanGap:clippedPlan.cost-firstPlan.cost};
}
