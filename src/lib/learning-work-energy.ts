import type {Vec2} from './learning-simulations';
const B=BigInt,Z=B(0),ONE=B(1);type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat{if(d===Z)throw new Error('Zero rational denominator.');if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};}
const R=(n:number)=>rat(B(n)),D=(n:number)=>rat(B(n),B(10)),add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n);
function expose(a:Rat){return{numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const numeric=(a:Rat)=>expose(a).value;
function controls(p:number[],ranges:readonly(readonly[number,number])[]){if(p.length!==ranges.length)throw new Error('Use the stated classical-mechanics controls.');p.forEach((v,j)=>{if(!Number.isInteger(v)||v<ranges[j][0]||v>ranges[j][1])throw new Error('Use bounded integer control codes.');});}


/** Quartic conservative force. Allowed states are not time-parametrized trajectories. */
export function quarticWorkEnergy(p:number[]) {
 controls(p,[[1,5],[1,5],[-4,4],[-40,60],[-30,30],[-30,30],[-20,20]]);
 const [mass,a,b,ec,ac,bc,cc]=p,E=D(ec),xA=D(ac),xB=D(bc),offset=D(cc);
 const potential=(x:Rat)=>add(div(mul(R(a),mul(mul(x,x),mul(x,x))),R(4)),div(mul(R(b),mul(x,x)),R(2)));
 const force=(x:Rat)=>neg(add(mul(R(a),mul(mul(x,x),x)),mul(R(b),x)));
 const va=potential(xA),vb=potential(xB),available=sub(E,vb),work=sub(va,vb),discriminantCode=10*b*b+4*a*ec,intervals:Vec2[]=[];
 if(discriminantCode>=0){
  const root=Math.sqrt(discriminantCode/10),lower=Math.max(0,(-b-root)/a),upper=(-b+root)/a;
  if(upper>=lower){const lo=Math.sqrt(lower),hi=Math.sqrt(upper);if(lo===0)intervals.push([-hi,hi]);else intervals.push([-hi,-lo],[lo,hi]);}
 }
 const allowed=available.n>=Z,queryForce=force(xB),isTurning=available.n===Z,stationaryQuery=isTurning&&queryForce.n===Z;
 const extent=Math.max(2,Math.abs(ac/10),Math.abs(bc/10),...intervals.flat().map(v=>Math.abs(v)*1.1)),energyCurve:Vec2[]=[],forceCurve:Vec2[]=[];
 for(let j=0;j<=200;j++){const x=-extent+2*extent*j/200;energyCurve.push([x,a*x**4/4+b*x*x/2+cc/10]);forceCurve.push([x,-a*x**3-b*x]);}
 return{mass,quarticCoefficient:a,quadraticCoefficient:b,energy:expose(E),offset:expose(offset),displayEnergy:expose(add(E,offset)),positionA:expose(xA),positionB:expose(xB),potentialA:expose(add(va,offset)),potentialB:expose(add(vb,offset)),work:expose(work),availableKinetic:expose(available),queryForce:expose(queryForce),queryAllowed:allowed,speed:allowed?Math.sqrt(2*numeric(available)/mass):null,isTurning,stationaryQuery,separatrixLevel:b<0&&ec===0,discriminant:expose(D(discriminantCode)),intervals,extent,curves:[energyCurve,forceCurve],workPotentialResidual:expose(add(work,sub(add(vb,offset),add(va,offset))))};
}

/** Integral of exp(-lambda*s)*s^n. Series avoids short-time cancellation. */
function exponentialMoment(lambda:number,t:number,n:number) {
 const z=lambda*t;
 if(z<.5){let term=1,sum=1/(n+1);for(let j=1;j<40;j++){term*=-z/j;sum+=term/(n+j+1);}return t**(n+1)*sum;}
 const e=Math.exp(-z),i0=-Math.expm1(-z)/lambda;
 if(n===0)return i0;
 const i1=(i0-t*e)/lambda;
 return n===1?i1:(2*i1-t*t*e)/lambda;
}
const exponentialIntegral=(rate:number,t:number)=>rate===0?t:Math.expm1(rate*t)/rate;

/** Analytic constant-drive oscillator; heat is independently integrated from v². */
export function dampedWorkPower(p:number[]) {
 controls(p,[[1,4],[1,4],[0,10],[-20,20],[-20,20],[-20,20],[0,100]]);
 const [mass,stiffness,damping,fc,xc,vc,tc]=p,drive=fc/10,x0=xc/10,v0=vc/10,time=tc/10,equilibrium=drive/stiffness,y0=x0-equilibrium,alpha=damping/(2*mass),omegaSquared=stiffness/mass,disc=damping*damping-4*mass*stiffness;
 const regime=damping===0?'undamped':disc<0?'underdamped':disc===0?'critical':'overdamped';
 const initialEnergy=mass*v0*v0/2+stiffness*x0*x0/2;
 const state=(t:number)=>{
  let y:number,v:number,heatIntegral=0;
  if(disc<0){
   const w=Math.sqrt(omegaSquared-alpha*alpha),c=Math.cos(w*t),s=Math.sin(w*t),e=Math.exp(-alpha*t),A=v0,B=-(alpha*v0+omegaSquared*y0)/w;
   y=e*(y0*c+(v0+alpha*y0)*s/w);v=e*(A*c+B*s);
   if(damping!==0){const lambda=2*alpha,b=2*w,et=Math.exp(-lambda*t),den=lambda*lambda+b*b,i0=-Math.expm1(-lambda*t)/lambda,ic=(lambda+et*(-lambda*Math.cos(b*t)+b*Math.sin(b*t)))/den,is=(b+et*(-lambda*Math.sin(b*t)-b*Math.cos(b*t)))/den;heatIntegral=(A*A+B*B)*i0/2+(A*A-B*B)*ic/2+A*B*is;}
  }else if(disc===0){
   const h=v0+alpha*y0,e=Math.exp(-alpha*t),A=v0,B=-alpha*h;
   y=e*(y0+h*t);v=e*(v0-alpha*h*t);
   heatIntegral=A*A*exponentialMoment(2*alpha,t,0)+2*A*B*exponentialMoment(2*alpha,t,1)+B*B*exponentialMoment(2*alpha,t,2);
  }else{
   const beta=Math.sqrt(alpha*alpha-omegaSquared),slow=-omegaSquared/(alpha+beta),fast=-(alpha+beta),cs=(v0-fast*y0)/(slow-fast),cf=(slow*y0-v0)/(slow-fast),ds=slow*cs,df=fast*cf;
   y=cs*Math.exp(slow*t)+cf*Math.exp(fast*t);v=ds*Math.exp(slow*t)+df*Math.exp(fast*t);
   heatIntegral=ds*ds*exponentialIntegral(2*slow,t)+df*df*exponentialIntegral(2*fast,t)+2*ds*df*exponentialIntegral(slow+fast,t);
  }
  // Preserve specified initial data, without introducing a branch-dependent roundoff offset.
  const x=t===0?x0:y+equilibrium;if(t===0)v=v0;if(v===0)v=0;
  const heat=t===0?0:Math.max(0,damping*heatIntegral),kinetic=mass*v*v/2,potential=stiffness*x*x/2,energy=kinetic+potential,driveWork=drive===0||x===x0?0:drive*(x-x0),drivePower=drive===0||v===0?0:drive*v,dampingPower=damping===0||v===0?0:-damping*v*v,energyRate=drivePower+dampingPower,acceleration=(drive-stiffness*x-damping*v)/mass;
  return{time:t,position:x,velocity:v,acceleration,kinetic,potential,energy,heat,driveWork,drivePower,dampingPower,energyRate,balanceResidual:energy-initialEnergy+heat-driveWork};
 };
 const query=state(time),curves:Vec2[][]=Array.from({length:7},()=>[]);
 for(let j=0;j<=200;j++){const s=state(j/20);[s.position,s.energy,s.heat,initialEnergy+s.driveWork,s.drivePower,s.dampingPower,s.energyRate].forEach((v,i)=>curves[i].push([j/20,v]));}
 return{mass,stiffness,damping,drive,initialPosition:x0,initialVelocity:v0,initialEnergy,equilibrium,discriminant:disc,regime,query,curves};
}
