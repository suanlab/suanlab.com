import type {Vec2} from './learning-simulations';

// Exact arithmetic is confined to rational polynomial data and coefficients of pi.
// Trigonometric evaluations and plots are floating point; this is not an ODE solver.
const B=BigInt,Z=B(0),ONE=B(1);
type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat {
  if(d===Z)throw new Error('Zero denominator.');
  if(d<Z){n=-n;d=-d;}
  let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}
  return {n:n/a,d:d/a};
}
const R=(n:number,d=1)=>rat(B(n),B(d));
const add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d);
const neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b));
const mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n),sq=(a:Rat)=>mul(a,a);
const sum=(a:Rat[])=>a.reduce(add,R(0));
function expose(a:Rat){return {numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:Number(a.n)/Number(a.d)};}
const num=(a:Rat)=>expose(a).value;
function piFactor(a:Rat,power:2|4=2){return {coefficient:expose(a),piPower:power,label:a.n===Z?'0':`pi^${power} * (${expose(a).label})`,value:num(a)*Math.PI**power};}
function controls(p:number[],ranges:readonly(readonly[number,number])[]){
  if(p.length!==ranges.length)throw new Error('Use the stated action controls.');
  for(let j=0;j<ranges.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
const derivative=(p:Rat[])=>p.slice(1).map((a,j)=>mul(R(j+1),a));
function product(a:Rat[],b:Rat[]){const p=Array.from({length:a.length+b.length-1},()=>R(0));a.forEach((x,i)=>b.forEach((y,j)=>{p[i+j]=add(p[i+j],mul(x,y));}));return p;}
function evaluate(p:Rat[],s:Rat){return p.reduceRight((acc,a)=>add(a,mul(s,acc)),R(0));}
const integral=(p:Rat[],s:Rat)=>evaluate([R(0),...p.map((a,j)=>div(a,R(j+1)))],s);
const polynomial=(a:Rat,b:Rat,A:Rat,C:Rat)=>[a,sum([b,neg(a),A,neg(C)]),add(neg(A),mul(R(3),C)),mul(R(-2),C)];

/** [m,10T,10a,10b,10A,10B,10gamma,10u0,10u1,10C,10D,100epsilon,100s].
 * L_gamma=m*v²/2+gamma*q*v. q is a prescribed cubic candidate.
 * Endpoint variations u0/u1 are distinguished from fixed-endpoint stationarity.
 */
export function freeParticleActionBoundaryGauge(p:number[]){
  controls(p,[[1,4],[5,40],[-20,20],[-20,20],[-20,20],[-20,20],[-30,30],[-10,10],[-10,10],[-10,10],[-10,10],[-100,100],[0,100]]);
  const [mc,tc,ac,bc,Ac,Bc,gc,u0c,u1c,Cc,Dc,ec,sc]=p;
  const m=R(mc),T=R(tc,10),a=R(ac,10),b=R(bc,10),A=R(Ac,10),Bv=R(Bc,10),gamma=R(gc,10),u0=R(u0c,10),u1=R(u1c,10),C=R(Cc,10),D=R(Dc,10),epsilon=R(ec,100),s=R(sc,100);
  const q=polynomial(a,b,A,Bv),qs=derivative(q),qss=derivative(qs),xi=polynomial(u0,u1,C,D),d=sub(b,a),du=sub(u1,u0),mT=div(m,T);
  const baseAction=mul(div(m,R(2)),div(sum([sq(d),div(sq(A),R(3)),div(sq(Bv),R(5))]),T));
  const gaugeShift=mul(div(gamma,R(2)),sub(sq(b),sq(a))),action=add(baseAction,gaugeShift),classicalAction=add(mul(div(mT,R(2)),sq(d)),gaugeShift);
  const gradientA=div(mul(mT,A),R(3)),gradientB=div(mul(mT,Bv),R(5));
  const first=sum([mul(mT,mul(d,du)),mul(gamma,sub(mul(b,u1),mul(a,u0))),mul(gradientA,C),mul(gradientB,D)]);
  const second=add(mul(mT,sum([sq(du),div(sq(C),R(3)),div(sq(D),R(5))])),mul(gamma,sub(sq(u1),sq(u0))));
  const actionAt=(e:Rat)=>sum([action,mul(e,first),mul(div(sq(e),R(2)),second)]);
  const state=(z:Rat)=>{
    const position=evaluate(q,z),velocity=div(evaluate(qs,z),T),acceleration=div(evaluate(qss,z),sq(T)),kinetic=mul(div(m,R(2)),sq(velocity));
    const mechanicalMomentum=mul(m,velocity),canonicalMomentum=add(mechanicalMomentum,mul(gamma,position)),gaugeTerm=mul(gamma,mul(position,velocity));
    const cumulativeBase=mul(div(mT,R(2)),integral(product(qs,qs),z)),cumulativeGauge=mul(div(gamma,R(2)),sub(sq(position),sq(a)));
    return {position:expose(position),velocity:expose(velocity),acceleration:expose(acceleration),kinetic:expose(kinetic),mechanicalMomentum:expose(mechanicalMomentum),canonicalMomentum:expose(canonicalMomentum),gaugeTerm:expose(gaugeTerm),lagrangian:expose(add(kinetic,gaugeTerm)),eulerLagrangeResidual:expose(mul(m,acceleration)),cumulativeAction:expose(add(cumulativeBase,cumulativeGauge)),hamiltonian:expose(sub(mul(canonicalMomentum,velocity),add(kinetic,gaugeTerm)))};
  };
  const start=state(R(0)),end=state(R(1)),boundaryTerm=sub(mul(rat(B(end.canonicalMomentum.numerator),B(end.canonicalMomentum.denominator)),u1),mul(rat(B(start.canonicalMomentum.numerator),B(start.canonicalMomentum.denominator)),u0));
  const interiorTerm=mul(mT,integral(product(qss,xi),R(1))),integrationByPartsFirst=sub(boundaryTerm,interiorTerm);
  const residualMeanSquare=mul(sq(div(m,sq(T))),integral(product(qss,qss),R(1)));
  // Physical-time series: q, q_classical, q_varied, v, v_classical, EL residual,
  // L_gamma, K, cumulative S_gamma, cumulative S_classical, p_gamma, m*v.
  const curves:Vec2[][]=Array.from({length:12},()=>[]);
  for(let j=0;j<=100;j++){
    const z=R(j,100),st=state(z),cl=add(a,mul(d,z)),clv=div(d,T),clS=add(mul(mul(div(mT,R(2)),sq(d)),z),mul(div(gamma,R(2)),sub(sq(cl),sq(a))));
    [st.position.value,num(cl),num(add(evaluate(q,z),mul(epsilon,evaluate(xi,z)))),st.velocity.value,num(clv),st.eulerLagrangeResidual.value,st.lagrangian.value,st.kinetic.value,st.cumulativeAction.value,num(clS),st.canonicalMomentum.value,st.mechanicalMomentum.value].forEach((v,i)=>curves[i].push([num(mul(T,z)),v]));
  }
  const variationCurves:Vec2[][]=[[],[],[]];
  for(let j=-100;j<=100;j++){const e=R(j,100),x=num(e);[actionAt(e),add(action,mul(e,first)),sum([action,mul(e,first),mul(div(sq(e),R(2)),second)])].forEach((v,i)=>variationCurves[i].push([x,num(v)]));}
  return {mass:mc,duration:num(T),time:num(mul(T,s)),epsilon:num(epsilon),gamma:num(gamma),coefficients:q.map(expose),variationCoefficients:xi.map(expose),baseAction:expose(baseAction),gaugeShift:expose(gaugeShift),action:expose(action),classicalAction:expose(classicalAction),excessAction:expose(sub(action,classicalAction)),gradientA:expose(gradientA),gradientB:expose(gradientB),hessianAA:expose(div(mT,R(3))),hessianBB:expose(div(mT,R(5))),firstVariation:expose(first),secondVariation:expose(second),variedAction:expose(actionAt(epsilon)),boundaryTerm:expose(boundaryTerm),interiorTerm:expose(interiorTerm),integrationByPartsFirst:expose(integrationByPartsFirst),integrationByPartsResidual:expose(sub(first,integrationByPartsFirst)),residualMeanSquare:expose(residualMeanSquare),stationaryForFixedEndpoints:Ac===0&&Bc===0,directionFixesEndpoints:u0c===0&&u1c===0,firstVariationAlongDirectionZero:first.n===Z,stationaryForBothFreeEndpoints:Ac===0&&Bc===0&&start.canonicalMomentum.numerator==='0'&&end.canonicalMomentum.numerator==='0',query:state(s),start,end,curves,variationCurves};
}

/** [m,f,10T,10A,10B,10C,10D,100epsilon,100s]. omega=pi*f/4,
 * k=m*omega²; q=A*sin(pi*s)+B*sin(3*pi*s). Fixed zero endpoints.
 * Hessian classification describes the stationary ZERO reference, not any trial.
 */
export function harmonicActionStationarySaddle(p:number[]){
  controls(p,[[1,4],[0,4],[5,25],[-20,20],[-20,20],[-10,10],[-10,10],[-100,100],[0,100]]);
  const [mass,f,tc,ac,bc,cc,dc,ec,sc]=p,T=R(tc,10),A=R(ac,10),Bv=R(bc,10),C=R(cc,10),D=R(dc,10),epsilon=R(ec,100),duration=num(T),omega=Math.PI*f/4,stiffness=mass*omega**2;
  const d1=1600-(f*tc)**2,d3=14400-(f*tc)**2,scale=div(R(mass),mul(R(6400),T));
  const actionCoefficient=(a:Rat,b:Rat)=>mul(scale,add(mul(R(d1),sq(a)),mul(R(d3),sq(b))));
  const action=actionCoefficient(A,Bv),gradientA=mul(R(2*d1),mul(scale,A)),gradientB=mul(R(2*d3),mul(scale,Bv));
  const first=add(mul(gradientA,C),mul(gradientB,D)),second=mul(R(2),mul(scale,add(mul(R(d1),sq(C)),mul(R(d3),sq(D)))));
  const H1=mul(R(2*d1),scale),H3=mul(R(2*d3),scale);
  const residualMeanSquare=div(mul(R(mass*mass),add(mul(R(d1*d1),sq(A)),mul(R(d3*d3),sq(Bv)))),mul(R(2*1600*1600),sq(sq(T))));
  const state=(s:number)=>{
    const edge=s===0||s===1,sn=(n:number)=>edge?0:Math.sin(n*Math.PI*s),cs=(n:number)=>edge?(s===0?1:-1):Math.cos(n*Math.PI*s);
    const a=num(A),b=num(Bv),position=edge?0:a*sn(1)+b*sn(3),velocity=Math.PI/duration*(a*cs(1)+3*b*cs(3)),acceleration=-((Math.PI/duration)**2)*(a*sn(1)+9*b*sn(3));
    const kinetic=mass*velocity**2/2,potential=stiffness*position**2/2,energy=kinetic+potential,initialEnergy=mass/2*(Math.PI/duration)**2*(a+3*b)**2;
    const el=edge?0:-mass*Math.PI**2/(1600*duration**2)*(d1*a*sn(1)+d3*b*sn(3)),direct=mass*acceleration+stiffness*position;
    const cross=(n:number,k:number,cosine:boolean)=>{
      if(edge)return n===k?s/2:0;
      if(n===k)return s/2+(cosine?1:-1)*Math.sin(2*n*Math.PI*s)/(4*n*Math.PI);
      const low=Math.sin((n-k)*Math.PI*s)/(2*(n-k)*Math.PI),high=Math.sin((n+k)*Math.PI*s)/(2*(n+k)*Math.PI);return low+(cosine?high:-high);
    };
    const cumulativeAction=mass*Math.PI**2/(2*duration)*(a*a*cross(1,1,true)+6*a*b*cross(1,3,true)+9*b*b*cross(3,3,true))-stiffness*duration/2*(a*a*cross(1,1,false)+2*a*b*cross(1,3,false)+b*b*cross(3,3,false));
    return {position,velocity,acceleration,force:-stiffness*position,kinetic,potential,energy,initialEnergy,energyChangeFromInitial:energy-initialEnergy,lagrangian:kinetic-potential,eulerLagrangeResidual:el,directEulerLagrangeResidual:direct,representationResidual:direct-el,cumulativeAction};
  };
  // Physical-time series: q, q_varied, velocity, force, EL residual, K,U,E,L,S.
  const curves:Vec2[][]=Array.from({length:10},()=>[]);
  for(let j=0;j<=100;j++){const s=j/100,st=state(s),varied=st.position+num(epsilon)*(num(C)*Math.sin(Math.PI*s)+num(D)*Math.sin(3*Math.PI*s));[st.position,j===0||j===100?0:varied,st.velocity,st.force,st.eulerLagrangeResidual,st.kinetic,st.potential,st.energy,st.lagrangian,st.cumulativeAction].forEach((v,i)=>curves[i].push([s*duration,v]));}
  const variationCurves:Vec2[][]=[[],[],[]];
  for(let j=-100;j<=100;j++){const e=R(j,100);[actionCoefficient(add(A,mul(e,C)),add(Bv,mul(e,D))),add(action,mul(e,first)),sum([action,mul(e,first),mul(div(sq(e),R(2)),second)])].forEach((v,i)=>variationCurves[i].push([num(e),piFactor(v).value]));}
  return {mass,frequencyCode:f,duration,omega,stiffness,time:duration*sc/100,epsilon:num(epsilon),amplitudes:[num(A),num(Bv)],variationAmplitudes:[num(C),num(D)],d1,d3,zeroReferenceNature:d1>0?'strict-minimum' as const:d1===0?'flat-minimum' as const:'saddle' as const,selectedPathStationary:d1*ac===0&&bc===0,action:piFactor(action),gradientA:piFactor(gradientA),gradientB:piFactor(gradientB),hessianAA:piFactor(H1),hessianBB:piFactor(H3),firstVariation:piFactor(first),secondVariation:piFactor(second),variedAction:piFactor(actionCoefficient(add(A,mul(epsilon,C)),add(Bv,mul(epsilon,D)))),residualMeanSquare:piFactor(residualMeanSquare,4),residualRms:Math.sqrt(piFactor(residualMeanSquare,4).value),query:state(sc/100),start:state(0),end:state(1),fullIntegralRepresentationResidual:state(1).cumulativeAction-piFactor(action).value,curves,variationCurves};
}
