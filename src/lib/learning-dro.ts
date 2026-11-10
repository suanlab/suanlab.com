/** Finite KL ambiguity sets. Logarithms and primal/dual gaps are numerical, not interval-arithmetic proofs. */
const validateLoss=(loss:readonly number[])=>{if(loss.length!==3||loss.some(v=>!Number.isInteger(v)||v<0||v>10))throw new Error('Use three integer losses in[0,10].');};
function code(value:number,max:number){if(!Number.isInteger(value)||value<0||value>max)throw new Error('Use the stated integer ambiguity-radius control.');}
function fraction(numerator:number,denominator:number){let a=Math.abs(numerator),b=denominator;while(b)[a,b]=[b,a%b];return{numerator,denominator,value:numerator/denominator,label:denominator/a===1?String(numerator/a):`${numerator/a}/${denominator/a}`};}
export function klRisk(loss:readonly number[],radiusCode:number){
 validateLoss(loss);code(radiusCode,2000);loss=loss.slice();const radius=radiusCode/1000;
 const p=[.2,.3,.5],maximum=Math.max(...loss),nominal=p.reduce((s,v,i)=>s+v*loss[i],0),top=p.reduce((s,v,i)=>s+(loss[i]===maximum?v:0),0),critical=-Math.log(top),kl=(q:number[])=>q.reduce((s,v,i)=>s+(v===0?0:v*Math.log(v/p[i])),0);
 if(loss.every(v=>v===maximum))return{loss:loss.slice(),radiusCode,saturationRadius:critical,radius,p,q:p.slice(),nominal,risk:maximum,kl:0,dual:maximum,rawGap:0,eta:0,phase:'constant-loss'};
 if(radius===0)return{loss:loss.slice(),radiusCode,saturationRadius:critical,radius,p,q:p.slice(),nominal,risk:nominal,kl:0,dual:nominal,rawGap:0,eta:null,phase:'zero-radius-limit'};
 if(radius>=critical){const q=p.map((v,i)=>loss[i]===maximum?v/top:0);return{loss:loss.slice(),radiusCode,saturationRadius:critical,radius,p,q,nominal,risk:maximum,kl:kl(q),dual:maximum,rawGap:0,eta:0,phase:'saturation'};}
 const tilt=(tau:number)=>{const u=p.map((v,i)=>v*Math.exp(tau*(loss[i]-maximum))),total=u.reduce((a,b)=>a+b,0),q=u.map(v=>v/total);return{q,kl:kl(q),logCenteredZ:Math.log(total)};};
 let lo=0,hi=1;while(tilt(hi).kl<radius)hi*=2;
 for(let i=0;i<100;i++){const mid=(lo+hi)/2;if(mid===lo||mid===hi)break;if(tilt(mid).kl<=radius)lo=mid;else hi=mid;}
 const q=tilt(lo).q,eta=1/lo,risk=q.reduce((s,v,i)=>s+v*loss[i],0),dual=eta*radius+maximum+eta*tilt(lo).logCenteredZ;
 return{loss:loss.slice(),radiusCode,saturationRadius:critical,radius,p,q,nominal,risk,kl:kl(q),dual,rawGap:dual-risk,eta,phase:'finite-tilt'};
}

/** Exact finite transport-risk and dual certificates on a fixed three-point support. */
export function transportRisk(loss:readonly number[],budgetCode:number){
 validateLoss(loss);code(budgetCode,40);loss=loss.slice();
 const mass=[8,12,20],cost=[[0,1,2],[1,0,1],[2,1,0]],budget=2*budgetCode,plans:number[][][]=[];
 const add=(p:number[][])=>{const used=p.reduce((s,row,i)=>s+row.reduce((t,v,j)=>t+v*cost[i][j],0),0);if(used<=budget&&p.every((row,i)=>row.every(v=>Number.isInteger(v)&&v>=0)&&row.reduce((s,x)=>s+x,0)===mass[i]))plans.push(p);};
 for(let a=0;a<3;a++)for(let b=0;b<3;b++)for(let c=0;c<3;c++){
  const assignment=[a,b,c],p=mass.map((n,i)=>Array.from({length:3},(_,j)=>j===assignment[i]?n:0));add(p);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(j!==assignment[i]){const k=assignment[i],d=cost[i][j]-cost[i][k];if(d===0)continue;const used=p.reduce((s,row,r)=>s+row.reduce((t,v,q)=>t+v*cost[r][q],0),0),move=(budget-used)/d;if(move<0||move>mass[i])continue;const q=p.map(row=>row.slice());q[i][k]-=move;q[i][j]+=move;add(q);}
 }
 const risks=plans.map(p=>p.reduce((s,row)=>s+row.reduce((t,v,j)=>t+v*loss[j],0),0)),primal=Math.max(...risks),plan=plans[risks.indexOf(primal)],lambda2=new Set([0]);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=j+1;k<3;k++){const d=cost[i][j]-cost[i][k];if(d){const lam2=2*(loss[j]-loss[k])/d;if(lam2>=0)lambda2.add(lam2);}}
 const duals=[...lambda2].map(lam2=>({lam2,numerator:lam2*budgetCode+mass.reduce((s,p,i)=>s+p*Math.max(...loss.map((v,j)=>2*v-lam2*cost[i][j]))/2,0)})),dual=Math.min(...duals.map(v=>v.numerator));
 return{loss:loss.slice(),budgetCode,radius:budgetCode/20,primalNumerator:primal,dualNumerator:dual,denominator:40,planUnits:plan,destinationUnits:loss.map((_,j)=>plan.reduce((s,row)=>s+row[j],0)),costNumerator:plan.reduce((s,row,i)=>s+row.reduce((t,v,j)=>t+v*cost[i][j],0),0),optimalLambda2:duals.find(v=>v.numerator===dual)!.lam2,dualCandidates:duals,nominalNumerator:mass.reduce((s,v,i)=>s+v*loss[i],0),riskExact:fraction(primal,40),dualExact:fraction(dual,40),radiusExact:fraction(budgetCode,20),exactRiskEquality:primal===dual};
}
