import type {Vec2} from './learning-simulations';
// Exact SI-coordinate values. Only clock labels/caustic times use atan2 or pi.
const B=BigInt,Z=B(0),ONE=B(1);
type Rat={n:bigint;d:bigint};
function rat(n:bigint,d=ONE):Rat{if(d===Z)throw new Error('Zero denominator.');if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};}
const R=(n:number,d=1)=>rat(B(n),B(d));
const add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n),sq=(a:Rat)=>mul(a,a);
const num=(a:Rat)=>Number(a.n)/Number(a.d),expose=(a:Rat)=>({numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:num(a)}),em=(a:Rat[][])=>a.map(row=>row.map(expose));
function validate(p:number[],ranges:readonly (readonly [number,number])[]){if(!Array.isArray(p)||p.length!==ranges.length)throw new Error('Wrong control count.');for(let i=0;i<p.length;i++)if(!Object.prototype.hasOwnProperty.call(p,i)||!Number.isInteger(p[i])||p[i]<ranges[i][0]||p[i]>ranges[i][1])throw new Error(`Invalid control ${i}.`);}
const forceRanges=[[1,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[0,40],[-4,4]]as const;
const oscillatorRanges=[[1,4],[1,4],[-4,4],[-4,4],[0,3],[0,4],[-4,4],[-4,4]]as const;

/** [m,2F,2alpha,2beta,2c,2lambda,20t,2queryOffset].
 * H=p²/(2m)-Fq; S=(alpha+Ft)q-alpha²t/(2m)-alphaFt²/(2m)
 * -F²t³/(6m)+c alpha²/2+lambda t q²/2.
 * alpha:kg m/s,beta:m,c:s/kg,lambda:N/m. The lambda term is a candidate defect,
 * not a changed mechanical force. A canonical map alone need not freeze dynamics.
 */
export function uniformForceHamiltonJacobiFamily(p:number[]){
 validate(p,forceRanges);
 const m=R(p[0]),F=R(p[1],2),alpha=R(p[2],2),beta=R(p[3],2),c=R(p[4],2),lambda=R(p[5],2),time=R(p[6],20),offset=R(p[7],2),q0=sub(beta,mul(c,alpha));
 const initialS=add(mul(alpha,q0),div(mul(c,sq(alpha)),R(2)));
 const state=(t:Rat,qOffset=R(0))=>{
  const t2=sq(t),t3=mul(t2,t),refP=add(alpha,mul(F,t)),refQ=add(add(q0,div(mul(alpha,t),m)),div(mul(F,t2),mul(R(2),m))),q=add(refQ,qOffset),v=div(refP,m),gradientP=add(refP,mul(lambda,mul(t,q))),action=add(sub(mul(refP,q),add(add(div(mul(sq(alpha),t),mul(R(2),m)),div(mul(mul(alpha,F),t2),mul(R(2),m))),div(mul(sq(F),t3),mul(R(6),m)))),add(div(mul(c,sq(alpha)),R(2)),div(mul(lambda,mul(t,sq(q))),R(2)))),partialTime=add(sub(mul(F,q),div(sq(refP),mul(R(2),m))),div(mul(lambda,sq(q)),R(2))),H=sub(div(sq(gradientP),mul(R(2),m)),mul(F,q)),residual=add(partialTime,H),canonicalQ=add(sub(sub(q,div(mul(alpha,t),m)),div(mul(F,t2),mul(R(2),m))),mul(c,alpha));
  const residualQ=add(mul(lambda,q),div(mul(lambda,mul(t,add(refP,mul(lambda,mul(t,q))))),m)),residualAlpha=div(mul(lambda,mul(t,q)),m),newQRate=add(residualAlpha,mul(residualQ,sub(div(t,m),c))),newPRate=neg(residualQ);
  const frozenP=add(refP,mul(lambda,mul(t,refQ))),frozenPDot=add(F,mul(lambda,add(refQ,mul(t,v)))),velocityResidual=sub(v,div(frozenP,m)),forceResidual=sub(frozenPDot,F),referenceEnergy=sub(div(sq(refP),mul(R(2),m)),mul(F,refQ)),frozenEnergy=sub(div(sq(frozenP),mul(R(2),m)),mul(F,refQ));
  const mechanicalAction=add(add(mul(add(div(sq(alpha),mul(R(2),m)),mul(F,q0)),t),div(mul(mul(alpha,F),t2),m)),div(mul(sq(F),t3),mul(R(3),m))),pathAction=add(add(initialS,mechanicalAction),div(mul(lambda,mul(t,sq(refQ))),R(2))),pathActionChange=sub(pathAction,initialS),pathActionDefect=sub(pathActionChange,mechanicalAction),Salphaalpha=sub(c,div(t,m)),lambdaT=mul(lambda,t),jacobian=[[R(1),neg(Salphaalpha)],[lambdaT,sub(R(1),mul(lambdaT,Salphaalpha))]],det=sub(mul(jacobian[0][0],jacobian[1][1]),mul(jacobian[0][1],jacobian[1][0]));
  return{time:num(t),position:expose(q),referencePosition:expose(refQ),queryOffset:expose(qOffset),referenceMomentum:expose(refP),gradientMomentum:expose(gradientP),action:expose(action),partialTime:expose(partialTime),hamiltonian:expose(H),hamiltonJacobiResidual:expose(residual),canonicalPosition:expose(canonicalQ),canonicalMomentum:expose(alpha),mixedHessian:expose(R(1)),momentumHessian:expose(Salphaalpha),positionHessian:expose(lambdaT),residualPositionDerivative:expose(residualQ),transformedPositionRate:expose(newQRate),transformedMomentumRate:expose(newPRate),frozenPositionVelocity:expose(v),frozenMomentum:expose(frozenP),frozenMomentumRate:expose(frozenPDot),frozenVelocityResidual:expose(velocityResidual),frozenForceResidual:expose(forceResidual),referenceEnergy:expose(referenceEnergy),frozenEnergy:expose(frozenEnergy),mechanicalAction:expose(mechanicalAction),pathAction:expose(pathAction),pathActionChange:expose(pathActionChange),pathActionDefect:expose(pathActionDefect),canonicalMapJacobian:em(jacobian),canonicalMapDeterminant:expose(det),canonicalMapResidual:expose(sub(det,R(1))),queryOnSelectedReference:qOffset.n===Z};
 };
 const query=state(time,offset),samples=Array.from({length:41},(_,i)=>state(R(i,20))),curve=(f:(s:typeof query)=>number):Vec2[]=>samples.map(s=>[s.time,f(s)]),qSamples=Array.from({length:81},(_,i)=>state(time,R(i-40,10)));
 // qSamples use offsets here; the public field abscissa is the actual q, not the offset.
 const field=(f:(s:typeof query)=>number):Vec2[]=>qSamples.map(s=>[s.position.value,f(s)]);
 const coefficients=[div(lambda,R(2)),div(sq(lambda),mul(R(2),m)),div(mul(lambda,alpha),m),div(mul(lambda,F),m)];
 return {mass:p[0],force:num(F),alpha:num(alpha),beta:num(beta),labelShear:num(c),candidateDefect:num(lambda),initialPosition:expose(q0),initialAction:expose(initialS),globalHamiltonJacobiSolution:p[5]===0,mixedCanonicalMapRegular:true,wholeFrozenPathPhysical:p[5]===0||(q0.n===Z&&alpha.n===Z&&F.n===Z),globalResidualBasis:['q²','t² q²','t q','t² q'],globalResidualCoefficientUnits:['N/m','kg/s⁴','N/s','N/s²'],globalResidualCoefficients:coefficients.map(expose),query,referenceStart:state(R(0)),samples,positionCurves:[curve(s=>s.referencePosition.value)],momentumCurves:[curve(s=>s.referenceMomentum.value),curve(s=>s.frozenMomentum.value)],velocityResidualCurve:curve(s=>s.frozenVelocityResidual.value),forceResidualCurve:curve(s=>s.frozenForceResidual.value),energyCurves:[curve(s=>s.referenceEnergy.value),curve(s=>s.frozenEnergy.value)],actionCurves:[curve(s=>s.pathActionChange.value),curve(s=>s.mechanicalAction.value)],fieldActionCurve:field(s=>s.action.value),fieldMomentumCurve:field(s=>s.gradientMomentum.value),fieldResidualCurve:field(s=>s.hamiltonJacobiResidual.value)};
}

/** [m,omega,2q0,2p0,quadrant,uCode,2qEndpoint,4sigma].
 * theta=quadrant*pi/2+2atan(uCode/4), physical T=theta/omega.
 * cos/sin and every query rank/zero are exact rational values. This rational-circle
 * clock parametrizes the exact flow; it is not a Cayley numerical time step.
 */
export function oscillatorHamiltonJacobiCausticAtlas(p:number[]){
 validate(p,oscillatorRanges);
 const m=R(p[0]),w=R(p[1]),mw=mul(m,w),mw2=mul(mw,w),q0=R(p[2],2),p0=R(p[3],2),endpoint=R(p[6],2),sigma=R(p[7],4);
 const clock=(quadrant:number,u:Rat)=>{
  const den=add(R(1),sq(u)),a=div(sub(R(1),sq(u)),den),b=div(mul(R(2),u),den),pairs=[[a,b],[neg(b),a],[neg(a),neg(b)],[b,neg(a)]];
  return{cos:pairs[quadrant][0],sin:pairs[quadrant][1],phase:quadrant*Math.PI/2+2*Math.atan(num(u))};
 };
 const state=(quadrant:number,u:Rat)=>{
  const {cos:C,sin:S,phase}=clock(quadrant,u),M=[[C,div(S,mw)],[neg(mul(mw,S)),C]],q=add(mul(C,q0),div(mul(S,p0),mw)),mom=add(neg(mul(mw,mul(S,q0))),mul(C,p0)),energy=add(div(sq(mom),mul(R(2),m)),div(mul(mw2,sq(q)),R(2))),initialEnergy=add(div(sq(p0),mul(R(2),m)),div(mul(mw2,sq(q0)),R(2))),pathAction=div(sub(mul(q,mom),mul(q0,p0)),R(2)),A=add(C,div(mul(sigma,S),mw)),graphB=add(neg(mul(mw,S)),mul(sigma,C)),det=sub(mul(M[0][0],M[1][1]),mul(M[0][1],M[1][0]));
  const endpointChart=(()=>{
   if(S.n===Z)return null;
   const requiredInitialMomentum=div(mul(mw,sub(endpoint,mul(C,q0))),S),finalMomentum=div(mul(mw,sub(mul(C,endpoint),q0)),S),action=div(mul(mw,sub(mul(C,add(sq(endpoint),sq(q0))),mul(R(2),mul(endpoint,q0)))),mul(R(2),S)),partialTime=div(mul(mw2,add(neg(add(sq(endpoint),sq(q0))),mul(R(2),mul(C,mul(endpoint,q0))))),mul(R(2),sq(S))),H=add(div(sq(finalMomentum),mul(R(2),m)),div(mul(mw2,sq(endpoint)),R(2))),boundaryAction=div(sub(mul(endpoint,finalMomentum),mul(q0,requiredInitialMomentum)),R(2));
   return{action:expose(action),initialMomentum:expose(requiredInitialMomentum),finalMomentum:expose(finalMomentum),partialTime:expose(partialTime),hamiltonian:expose(H),hamiltonJacobiResidual:expose(add(partialTime,H)),mixedHessian:expose(neg(div(mw,S))),boundaryAction:expose(boundaryAction),boundaryActionResidual:expose(sub(action,boundaryAction)),selectedPathMatchesEndpoint:sub(endpoint,q).n===Z};
  })();
  const mixedChart=(()=>{
   if(C.n===Z)return null;
   const numerator=sub(mul(endpoint,p0),mul(S,add(div(mul(mw,sq(endpoint)),R(2)),div(sq(p0),mul(R(2),mw))))),action=div(numerator,C),gradientP=div(sub(p0,mul(mw,mul(S,endpoint))),C),initialQ=div(sub(endpoint,div(mul(S,p0),mw)),C),partialTime=add(neg(add(div(mul(mw2,sq(endpoint)),R(2)),div(sq(p0),mul(R(2),m)))),div(mul(w,mul(S,numerator)),sq(C))),H=add(div(sq(gradientP),mul(R(2),m)),div(mul(mw2,sq(endpoint)),R(2))),onPathNumerator=sub(mul(q,p0),mul(S,add(div(mul(mw,sq(q)),R(2)),div(sq(p0),mul(R(2),mw))))),pathActionFromMixed=sub(div(onPathNumerator,C),mul(q0,p0));
   return{action:expose(action),finalMomentum:expose(gradientP),initialPosition:expose(initialQ),partialTime:expose(partialTime),hamiltonian:expose(H),hamiltonJacobiResidual:expose(add(partialTime,H)),mixedHessian:expose(div(R(1),C)),pathActionFromMixed:expose(pathActionFromMixed),pathActionResidual:expose(sub(pathActionFromMixed,pathAction))};
  })();
  const graphChart=(()=>{
   if(A.n===Z)return null;
   const hessian=div(graphB,A),action=div(mul(hessian,sq(endpoint)),R(2)),gradientP=mul(hessian,endpoint),partialTime=neg(div(mul(add(mw2,div(sq(hessian),m)),sq(endpoint)),R(2))),H=add(div(sq(gradientP),mul(R(2),m)),div(mul(mw2,sq(endpoint)),R(2))),initialLabel=div(endpoint,A),graphPathAction=div(sub(mul(endpoint,gradientP),mul(sigma,sq(initialLabel))),R(2)),initialAction=div(mul(sigma,sq(initialLabel)),R(2));
   return{action:expose(action),momentum:expose(gradientP),hessian:expose(hessian),initialLabel:expose(initialLabel),partialTime:expose(partialTime),hamiltonian:expose(H),hamiltonJacobiResidual:expose(add(partialTime,H)),initialAction:expose(initialAction),pathAction:expose(graphPathAction),transportedActionResidual:expose(sub(action,add(initialAction,graphPathAction)))};
  })();
  return{phase,time:phase/p[1],cosine:expose(C),sine:expose(S),flowMatrix:em(M),flowDeterminant:expose(det),symplecticResidual:expose(sub(det,R(1))),position:expose(q),momentum:expose(mom),velocity:expose(div(mom,m)),momentumRate:expose(neg(mul(mw2,q))),energy:expose(energy),energyResidual:expose(sub(energy,initialEnergy)),physicalAction:expose(pathAction),endpointSensitivity:expose(div(S,mw)),mixedSensitivity:expose(C),graphProjection:expose(A),graphMomentumCoefficient:expose(graphB),initialGraphMember:sub(p0,mul(sigma,q0)).n===Z,endpointChart,mixedChart,graphChart,endpointChartRegular:S.n!==Z,mixedChartRegular:C.n!==Z,graphChartRegular:A.n!==Z,endpointCompatible:S.n!==Z||sub(endpoint,mul(C,q0)).n===Z,boundaryInitialMomentumClass:S.n!==Z?'unique':sub(endpoint,mul(C,q0)).n===Z?'undetermined':'none',zeroDuration:quadrant===0&&u.n===Z,atlasCoversFlow:S.n!==Z||C.n!==Z,graphQueryClass:A.n!==Z?'unique':endpoint.n===Z?'undetermined':'none'};
 };
 const query=state(p[4],R(p[5],4)),samples=Array.from({length:81},(_,i)=>state(i===80?3:Math.floor(i/20),i===80?R(1):R(i%20,20))),curve=(f:(s:typeof query)=>number):Vec2[]=>samples.map(s=>[s.time,f(s)]);
 const segmented=(fn:(s:typeof query)=>number|null,sign:(s:typeof query)=>number):Vec2[][]=>{const segments:Vec2[][]=[];let active:Vec2[]=[],previous=0;for(const s of samples){const value=fn(s),current=sign(s);if(value===null||previous*current<0){if(active.length)segments.push(active);active=[];}if(value!==null)active.push([s.time,value]);previous=current;}if(active.length)segments.push(active);return segments;};
 const phaseSamples=Array.from({length:41},(_,i)=>R(i-20,10));
 return{mass:p[0],frequency:p[1],initialPosition:expose(q0),initialMomentum:expose(p0),endpointPosition:expose(endpoint),initialGraphSlope:expose(sigma),query,samples,phaseOrdering:['q','p'],flowCoefficientUnits:[['1','s/kg'],['kg/s','1']],positionCurve:curve(s=>s.position.value),momentumCurve:curve(s=>s.momentum.value),phaseCurve:samples.map(s=>[s.position.value,s.momentum.value]as Vec2),energyCurve:curve(s=>s.energy.value),physicalActionCurve:curve(s=>s.physicalAction.value),endpointSensitivityCurve:curve(s=>s.endpointSensitivity.value),mixedSensitivityCurve:curve(s=>s.mixedSensitivity.value),graphProjectionCurve:curve(s=>s.graphProjection.value),graphHessianSegments:segmented(s=>s.graphChart?.hessian.value??null,s=>s.graphProjection.value),endpointActionSegments:segmented(s=>s.endpointChart?.action.value??null,s=>s.sine.value),mixedActionSegments:segmented(s=>s.mixedChart?.action.value??null,s=>s.cosine.value),initialGraph:phaseSamples.map(x=>[num(x),num(mul(sigma,x))]as Vec2),transportedGraph:phaseSamples.map(x=>[query.graphProjection.value*num(x),query.graphMomentumCoefficient.value*num(x)]as Vec2),graphLabelMap:phaseSamples.map(x=>[num(x),query.graphProjection.value*num(x)]as Vec2),graphCausticTimes:[Math.atan2(num(mw),-num(sigma))/p[1],(Math.atan2(num(mw),-num(sigma))+Math.PI)/p[1]]};
}
