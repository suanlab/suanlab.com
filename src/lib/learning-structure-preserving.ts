import type {Vec2} from './learning-simulations';
type Matrix=number[][];
type Jet={v:number;d:Vec2};
export type QuarticMethod='verlet'|'rk4'|'energy-projected';
export interface QuarticSystem {mass:number;stiffness:number;quartic:number;}
const j=(v:number,d:Vec2=[0,0]):Jet=>({v,d}),plus=(a:Jet,b:Jet):Jet=>j(a.v+b.v,[a.d[0]+b.d[0],a.d[1]+b.d[1]]),scale=(a:Jet,c:number):Jet=>j(a.v*c,[a.d[0]*c,a.d[1]*c]),times=(a:Jet,b:Jet):Jet=>j(a.v*b.v,[a.d[0]*b.v+a.v*b.d[0],a.d[1]*b.v+a.v*b.d[1]]),minus=(a:Jet,b:Jet)=>plus(a,scale(b,-1));
const eye=():Matrix=>[[1,0],[0,1]],mm=(a:Matrix,b:Matrix):Matrix=>a.map(row=>b[0].map((_,k)=>row.reduce((s,v,i)=>s+v*b[i][k],0))),det=(a:Matrix)=>a[0][0]*a[1][1]-a[0][1]*a[1][0];
const validate=(p:number[],ranges:readonly (readonly[number,number])[])=>{if(!Array.isArray(p)||p.length!==ranges.length)throw new Error('Wrong control count.');p.forEach((v,i)=>{if(!Number.isInteger(v)||v<ranges[i][0]||v>ranges[i][1])throw new Error(`Invalid control ${i}.`);});for(let i=0;i<p.length;i++)if(!Object.prototype.hasOwnProperty.call(p,i))throw new Error('Sparse controls.');};
const potential=(q:number,s:QuarticSystem)=>s.stiffness*q*q/2+s.quartic*q**4/4,gradient=(q:number,s:QuarticSystem)=>s.stiffness*q+s.quartic*q**3,energy=(q:number,p:number,s:QuarticSystem)=>p*p/(2*s.mass)+potential(q,s),shadow=(q:number,p:number,h:number,s:QuarticSystem)=>energy(q,p,s)+h*h*((s.stiffness+3*s.quartic*q*q)*p*p/(12*s.mass*s.mass)-gradient(q,s)**2/(24*s.mass));
export function quarticDiscreteActionStep(method:QuarticMethod,q:number,p:number,h:number,s:QuarticSystem){
 if(!['verlet','rk4','energy-projected'].includes(method)||![q,p,h,s.mass,s.stiffness,s.quartic].every(Number.isFinite)||s.mass<=0||s.stiffness<0||s.quartic<0)throw new Error('Invalid quartic step.');
 const U=(x:Jet)=>plus(scale(times(x,x),s.stiffness/2),scale(times(times(x,x),times(x,x)),s.quartic/4)),G=(x:Jet)=>plus(scale(x,s.stiffness),scale(times(times(x,x),x),s.quartic));
 let Q=j(q,[1,0]),P=j(p,[0,1]),budget:number|null=null,boundary=false,status='regular';
 if(h!==0){
  if(method==='rk4'){
   const rhs=(a:Jet,b:Jet)=>[scale(b,1/s.mass),scale(G(a),-1)],k1=rhs(Q,P),k2=rhs(plus(Q,scale(k1[0],h/2)),plus(P,scale(k1[1],h/2))),k3=rhs(plus(Q,scale(k2[0],h/2)),plus(P,scale(k2[1],h/2))),k4=rhs(plus(Q,scale(k3[0],h)),plus(P,scale(k3[1],h)));
   Q=plus(Q,scale(plus(plus(k1[0],scale(k2[0],2)),plus(scale(k3[0],2),k4[0])),h/6));P=plus(P,scale(plus(plus(k1[1],scale(k2[1],2)),plus(scale(k3[1],2),k4[1])),h/6));
  }else{
   const oldQ=Q,oldP=P,half=minus(P,scale(G(Q),h/2));Q=plus(Q,scale(half,h/s.mass));P=minus(half,scale(G(Q),h/2));
   if(method==='energy-projected'&&(s.stiffness!==0||s.quartic!==0)){
    const W=minus(plus(scale(times(oldP,oldP),1/(2*s.mass)),U(oldQ)),U(Q));budget=Number.isFinite(W.v)?W.v:null;
    if(!Number.isFinite(W.v))status='nonfinite-update';else if(W.v<0)status='energy-target-unreachable';else if(W.v===0){P=j(0);boundary=true;status='projection-boundary';}else if(P.v===0)status='projection-sign-undetermined';else{const root=Math.sign(P.v)*Math.sqrt(2*s.mass*W.v);P=j(root,[s.mass*W.d[0]/root,s.mass*W.d[1]/root]);}
   }
  }
 }
 if(!['regular','projection-boundary'].includes(status)||![Q.v,P.v].every(Number.isFinite))return{state:null,jacobian:null,determinant:null,energyBudget:budget,discreteAction:null,status:status==='regular'?'nonfinite-update':status};
 if(Math.abs(Q.v)>1e6||Math.abs(P.v)>1e9)return{state:null,jacobian:null,determinant:null,energyBudget:budget,discreteAction:null,status:'display-limit'};
 const J=boundary?null:[Q.d,P.d],jacobian=J&&J.flat().every(Number.isFinite)?J:null;
 const discreteAction=h===0?null:{value:s.mass*(Q.v-q)**2/(2*h)-h*(potential(q,s)+potential(Q.v,s))/2,initialMomentum:s.mass*(Q.v-q)/h+h*gradient(q,s)/2,finalMomentum:s.mass*(Q.v-q)/h-h*gradient(Q.v,s)/2,initialLegendreResidual:s.mass*(Q.v-q)/h+h*gradient(q,s)/2-p,finalLegendreResidual:s.mass*(Q.v-q)/h-h*gradient(Q.v,s)/2-P.v};
 return{state:{position:Q.v,momentum:P.v},jacobian,determinant:jacobian?det(jacobian):null,energyBudget:budget,discreteAction,status:jacobian?status:boundary?status:'nonfinite-derivative'};
}
function rk4Value(q:number,p:number,h:number,s:QuarticSystem):Vec2{
 const f=(a:number,b:number):Vec2=>[b/s.mass,-gradient(a,s)],a=f(q,p),b=f(q+h*a[0]/2,p+h*a[1]/2),c=f(q+h*b[0]/2,p+h*b[1]/2),d=f(q+h*c[0],p+h*c[1]);return[q+h*(a[0]+2*b[0]+2*c[0]+d[0])/6,p+h*(a[1]+2*b[1]+2*c[1]+d[1])/6];
}
function referenceTrajectory(q0:number,p0:number,T:number,N:number,s:QuarticSystem){
 const states=[{time:0,position:q0,momentum:p0}],atol=1e-12,rtol=1e-11;let q=q0,p=p0,t=0,dt=Math.min(.02,T/N),accepted=0,rejected=0,maxAcceptedRatio=0,status='completed';
 for(let n=1;n<=N;n++){
  const target=n*T/N;
  while(t<target){
   if(accepted+rejected>=200000||t+dt===t){status='reference-budget-exhausted';return{states,status,accepted,rejected,maxAcceptedRatio,absolutePositionTolerance:atol,absoluteMomentumTolerance:atol,relativeTolerance:rtol};}
   const step=Math.min(dt,target-t),full=rk4Value(q,p,step,s),half=rk4Value(q,p,step/2,s),twice=rk4Value(half[0],half[1],step/2,s),dq=(twice[0]-full[0])/15,dp=(twice[1]-full[1])/15,ratio=Math.max(Math.abs(dq)/(atol+rtol*Math.max(Math.abs(q),Math.abs(twice[0]))),Math.abs(dp)/(atol+rtol*Math.max(Math.abs(p),Math.abs(twice[1]))));
   if(!Number.isFinite(ratio)){status='reference-nonfinite';return{states,status,accepted,rejected,maxAcceptedRatio,absolutePositionTolerance:atol,absoluteMomentumTolerance:atol,relativeTolerance:rtol};}
   if(ratio<=1){q=twice[0]+dq;p=twice[1]+dp;t=step===target-t?target:t+step;accepted++;maxAcceptedRatio=Math.max(maxAcceptedRatio,ratio);}else rejected++;
   dt=step*Math.max(.2,Math.min(2,ratio===0?2:.9*ratio**(-.2)));
  }
  states.push({time:target,position:q,momentum:p});
 }
 return{states,status,accepted,rejected,maxAcceptedRatio,absolutePositionTolerance:atol,absoluteMomentumTolerance:atol,relativeTolerance:rtol};
}
/** [m,k,a,2q0,2p0,2T,level,queryIndex]. N=4*2^level; h=T/N.
 * Reference tolerances control local estimates,not a rigorous global error bound.
 */
export function quarticDiscreteActionSplitting(p:number[]){
 validate(p,[[1,4],[0,4],[0,4],[-4,4],[-4,4],[0,8],[0,4],[0,64]]);
 const system={mass:p[0],stiffness:p[1],quartic:p[2]},q0=p[3]/2,p0=p[4]/2,T=p[5]/2,N=4*2**p[6],h=T/N,E0=energy(q0,p0,system),qScale=Math.max(1,Math.abs(q0)),pScale=Math.max(1,Math.abs(p0),Math.sqrt(2*system.mass*E0)),reference=referenceTrajectory(q0,p0,T,N,system);
 const run=(method:QuarticMethod,count:number)=>{
  const step=T/count,initialShadow=shadow(q0,p0,step,system),states=[{index:0,time:0,position:q0,momentum:p0,energy:E0,energyDrift:0,shadowEnergy:initialShadow,shadowDrift:0,localDeterminant:1 as number|null,cumulativeDeterminant:1 as number|null,positionError:0 as number|null,momentumError:0 as number|null}],start={position:q0,momentum:p0};let current=start,J:Matrix|null=eye(),stopReason:string|null=null;
  for(let n=1;n<=count;n++){
   const update=quarticDiscreteActionStep(method,current.position,current.momentum,step,system);
   if(!update.state){stopReason=update.status;break;}current=update.state;J=J&&update.jacobian?mm(update.jacobian,J):null;if(J&&(!J.flat().every(Number.isFinite)||!Number.isFinite(det(J))))J=null;
   const energyNow=energy(current.position,current.momentum,system),shadowNow=shadow(current.position,current.momentum,step,system),ref=count===N?reference.states[n]:null;
   states.push({index:n,time:n*step,...current,energy:energyNow,energyDrift:energyNow-E0,shadowEnergy:shadowNow,shadowDrift:shadowNow-initialShadow,localDeterminant:update.determinant,cumulativeDeterminant:J?det(J):null,positionError:ref?current.position-ref.position:null,momentumError:ref?current.momentum-ref.momentum:null});
  }
  const completed=states.length===count+1,terminal=completed?states[states.length-1]:null,ref=reference.states.length===N+1?reference.states[N]:null,error=terminal&&ref?Math.hypot((terminal.position-ref.position)/qScale,(terminal.momentum-ref.momentum)/pScale):null;
  return{method,stepSize:step,requestedSteps:count,completedSteps:states.length-1,completed,stopReason,states,terminal,normalizedTerminalError:error,cumulativeJacobian:J};
 };
 const methods=(['verlet','rk4','energy-projected']as const).map(method=>{
  const coarse=run(method,N),medium=run(method,2*N),fine=run(method,4*N),errors=[coarse.normalizedTerminalError,medium.normalizedTerminalError,fine.normalizedTerminalError],comparisonFloor=1e-10,order=(a:number|null,b:number|null)=>a!==null&&b!==null&&a>comparisonFloor&&b>comparisonFloor?Math.log2(a/b):null;
  let reversePositionError:number|null=null,reverseMomentumError:number|null=null,reverseStopReason:string|null=coarse.completed?null:'forward-run-incomplete';
  if(coarse.terminal){let state={position:coarse.terminal.position,momentum:coarse.terminal.momentum};for(let i=0;i<N;i++){const u=quarticDiscreteActionStep(method,state.position,state.momentum,-h,system);if(!u.state){reverseStopReason=u.status;break;}state=u.state;}if(!reverseStopReason){reversePositionError=state.position-q0;reverseMomentumError=state.momentum-p0;}}
  return{...coarse,firstStep:quarticDiscreteActionStep(method,q0,p0,h,system),query:p[7]<=N?coarse.states[p[7]]??null:null,queryStatus:p[7]>N?'outside-run':p[7]>=coarse.states.length?'run-stopped':'available',structurallySymplectic:method==='verlet'||h===0||(system.stiffness===0&&system.quartic===0),structurallySymmetric:method==='verlet'||h===0||(system.stiffness===0&&system.quartic===0),refinements:[{steps:N,error:errors[0],completed:coarse.completed},{steps:2*N,error:errors[1],completed:medium.completed},{steps:4*N,error:errors[2],completed:fine.completed}],observedOrders:[order(errors[0],errors[1]),order(errors[1],errors[2])],comparisonFloor,reversePositionError,reverseMomentumError,reverseStopReason};
 });
 return{system,initialPosition:q0,initialMomentum:p0,initialEnergy:E0,duration:T,steps:N,stepSize:h,queryIndex:p[7],positionScale:qScale,momentumScale:pScale,reference,methods,positionCurves:methods.map(r=>r.states.map(s=>[s.time,s.position]as Vec2)),momentumCurves:methods.map(r=>r.states.map(s=>[s.time,s.momentum]as Vec2)),phaseCurves:methods.map(r=>r.states.map(s=>[s.position,s.momentum]as Vec2)),energyDriftCurves:methods.map(r=>r.states.map(s=>[s.time,s.energyDrift]as Vec2)),shadowDriftCurve:methods[0].states.map(s=>[s.time,s.shadowDrift]as Vec2),positionErrorCurves:methods.map(r=>r.states.filter(s=>s.positionError!==null).map(s=>[s.time,s.positionError!]as Vec2)),momentumErrorCurves:methods.map(r=>r.states.filter(s=>s.momentumError!==null).map(s=>[s.time,s.momentumError!]as Vec2)),determinantCurves:methods.map(r=>r.states.filter(s=>s.cumulativeDeterminant!==null).map(s=>[s.time,s.cumulativeDeterminant!]as Vec2)),referencePhase:reference.states.map(s=>[s.position,s.momentum]as Vec2)};
}

function exactFraction(n:number,d:number){let a=Math.abs(n),b=d;while(b){const c=a%b;a=b;b=c;}const g=a||1;return{numerator:String(n/g),denominator:String(d/g),label:d/g===1?String(n/g):`${n/g}/${d/g}`,value:n/d};}
/** [m,2R,2p_t,20h,N,quadrant,4u,queryIndex]. All queries are actual step indices.
 * Near-identity RATTLE exists at |nu|<=1 but its derivative requires |nu|<1.
 * Cartesian updates are executed; recorded constraint residuals include roundoff.
 */
export function circleRattleProjectionPhase(p:number[]){
 validate(p,[[1,4],[1,4],[-4,4],[0,40],[1,64],[0,3],[0,4],[0,64]]);
 const m=p[0],radius=p[1]/2,pt=p[2]/2,h=p[3]/20,N=p[4],u=p[6]/4,c=(1-u*u)/(1+u*u),s=2*u/(1+u*u),cs=[[c,s],[-s,c],[-c,-s],[s,-c]][p[5]],q0:Vec2=[radius*cs[0],radius*cs[1]],p0:Vec2=[-pt*cs[1],pt*cs[0]],L0=radius*pt,I=m*radius*radius,E0=pt*pt/(2*m),nuNumerator=p[3]*p[2],nuDenominator=20*m*p[1],nu=nuNumerator/nuDenominator,discNumerator=nuDenominator**2-nuNumerator**2,discriminant=exactFraction(discNumerator,nuDenominator**2),admissibility=discNumerator>0?'regular':discNumerator===0?'boundary':'unreachable',root=discNumerator>=0?Math.sqrt(discriminant.value):null;
 const initialAngle=p[5]*Math.PI/2+2*Math.atan(u),dot=(a:Vec2,b:Vec2)=>a[0]*b[0]+a[1]*b[1],cross=(a:Vec2,b:Vec2)=>a[0]*b[1]-a[1]*b[0],make=(index:number,q:Vec2,momentum:Vec2,phase:number,jacobian:Matrix|null,method:'rattle'|'projected')=>{
  const L=cross(q,momentum),E=dot(momentum,momentum)/(2*m),idealL=method==='rattle'?L0:L0/Math.sqrt(1+index*nu*nu),idealE=method==='rattle'?E0:E0/(1+index*nu*nu);
  return{index,time:index*h,position:q,momentum,unwrappedAngle:initialAngle+phase,phaseAdvance:phase,phaseError:phase-index*nu,angularMomentum:L,angularMomentumDrift:L-L0,energy:E,energyDrift:E-E0,positionConstraint:(dot(q,q)-radius*radius)/2,momentumConstraint:dot(q,momentum),idealAngularMomentum:idealL,idealEnergy:idealE,idealMomentumResidual:L-idealL,idealEnergyResidual:E-idealE,jacobian,determinant:jacobian?det(jacobian):null};
 };
 const run=(method:'rattle'|'projected')=>{
  let q:Vec2=[...q0],mom:Vec2=[...p0],phase=0,J:Matrix|null=eye(),stopReason:string|null=null;const states=[make(0,q,mom,phase,J,method)];
  for(let n=1;n<=N;n++){
   if(method==='rattle'&&root===null){stopReason='rattle-position-root-unreachable';break;}
   if(h!==0){
    if(method==='rattle'){
     const nextQ:Vec2=[root!*q[0]+h*mom[0]/m,root!*q[1]+h*mom[1]/m],nextP:Vec2=[root!*mom[0]-nu*pt*q[0]/radius,root!*mom[1]-nu*pt*q[1]/radius];q=nextQ;mom=nextP;phase+=Math.asin(nu);J=discNumerator>0?[[1,n*h/(I*root!)],[0,1]]:null;
    }else{
     const L=cross(q,mom),v=h*L/I,temp:Vec2=[q[0]+h*mom[0]/m,q[1]+h*mom[1]/m],length=Math.hypot(...temp),nextQ:Vec2=[radius*temp[0]/length,radius*temp[1]/length],normal=dot(nextQ,mom)/(radius*radius),nextP:Vec2=[mom[0]-normal*nextQ[0],mom[1]-normal*nextQ[1]],local=[[1,h/(I*(1+v*v))],[0,(1+v*v)**(-1.5)]];J=J?mm(local,J):null;q=nextQ;mom=nextP;phase+=Math.atan(v);
    }
   }
   states.push(make(n,q,mom,phase,J,method));
  }
  const completed=states.length===N+1,query=p[7]<=N?states[p[7]]??null:null;
  let reversePositionError:number|null=null,reverseMomentumError:number|null=null;
  if(completed){let Q:Vec2=[...q],P:Vec2=[...mom];for(let i=0;i<N;i++){if(h===0)continue;if(method==='rattle'){const nextQ:Vec2=[root!*Q[0]-h*P[0]/m,root!*Q[1]-h*P[1]/m],nextP:Vec2=[root!*P[0]+nu*pt*Q[0]/radius,root!*P[1]+nu*pt*Q[1]/radius];Q=nextQ;P=nextP;}else{const temp:Vec2=[Q[0]-h*P[0]/m,Q[1]-h*P[1]/m],length=Math.hypot(...temp),nextQ:Vec2=[radius*temp[0]/length,radius*temp[1]/length],normal=dot(nextQ,P)/(radius*radius);P=[P[0]-normal*nextQ[0],P[1]-normal*nextQ[1]];Q=nextQ;}}reversePositionError=Math.hypot(Q[0]-q0[0],Q[1]-q0[1]);reverseMomentumError=Math.hypot(P[0]-p0[0],P[1]-p0[1]);}
  return{method,states,completed,completedSteps:states.length-1,stopReason,query,reversePositionError,reverseMomentumError,queryStatus:p[7]>N?'outside-run':p[7]>=states.length?'run-stopped':'available',terminal:completed?states[states.length-1]:null};
 };
 const rattle=run('rattle'),projected=run('projected'),reference=Array.from({length:N+1},(_,i)=>{const a=i*nu,C=Math.cos(a),S=Math.sin(a);return{index:i,time:i*h,position:[C*q0[0]-S*q0[1],S*q0[0]+C*q0[1]]as Vec2,momentum:[C*p0[0]-S*p0[1],S*p0[0]+C*p0[1]]as Vec2,phaseAdvance:a,energy:E0,angularMomentum:L0};});
 const refinements=[1,2,4].map(factor=>{
  const v=nu/factor,count=N*factor,rattleError=Math.abs(v)<=1?count*(Math.asin(v)-v):null;let projectedPhase=0;for(let n=0;n<count;n++)projectedPhase+=Math.atan(v/Math.sqrt(1+n*v*v));return{steps:count,stepSize:h/factor,rattlePhaseError:rattleError,projectedPhaseError:projectedPhase-N*nu,projectedMomentumError:L0/Math.sqrt(1+count*v*v)-L0};
 });
 return{mass:m,radius,tangentialMomentum:pt,stepSize:h,steps:N,queryIndex:p[7],initialPosition:q0,initialMomentum:p0,initialAngle,initialAngularMomentum:L0,initialEnergy:E0,inertia:I,nu:exactFraction(nuNumerator,nuDenominator),rattleDiscriminant:discriminant,rattleAdmissibility:admissibility,rattleRoot:root,rattleSmoothAtInitial:discNumerator>0,rattleSymplecticOnRegularDomain:true,projectedSymplecticOnWholeReducedDomain:h===0,rattleMomentumCoefficient:h===0?0:root===null?null:h*pt*pt/(m*radius*radius*(1+root)),rattle,projected,reference,refinements,phaseCurves:[rattle.states.map(s=>s.position),projected.states.map(s=>s.position),reference.map(s=>s.position)],phaseErrorCurves:[rattle,projected].map(run=>run.states.map(s=>[s.time,s.phaseError]as Vec2)),energyDriftCurves:[rattle,projected].map(run=>run.states.map(s=>[s.time,s.energyDrift]as Vec2)),angularMomentumCurves:[rattle,projected].map(run=>run.states.map(s=>[s.time,s.angularMomentum]as Vec2)),positionConstraintCurves:[rattle,projected].map(run=>run.states.map(s=>[s.time,s.positionConstraint]as Vec2)),momentumConstraintCurves:[rattle,projected].map(run=>run.states.map(s=>[s.time,s.momentumConstraint]as Vec2)),determinantCurves:[rattle,projected].map(run=>run.states.filter(s=>s.determinant!==null).map(s=>[s.time,s.determinant!]as Vec2))};
}
