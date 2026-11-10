/** Exact three-state Hastings correction and an explicitly uncorrected comparator.
 * A target-prepared ledger is not a stationary ledger when the target drifts.
 */
import {markovFraction as fraction,exactMarkovObservation as observe,type MarkovFraction as Q} from './learning-finite-markov-exact';
const Z=BigInt(0),O=BigInt(1),q0=fraction(Z),q1=fraction(O);
const add=(a:Q,b:Q)=>fraction(a.n*b.d+b.n*a.d,a.d*b.d);
const sub=(a:Q,b:Q)=>fraction(a.n*b.d-b.n*a.d,a.d*b.d);
const mul=(a:Q,b:Q)=>fraction(a.n*b.n,a.d*b.d);
const div=(a:Q,b:Q)=>fraction(a.n*b.d,a.d*b.n);
const cmp=(a:Q,b:Q)=>a.n*b.d-b.n*a.d;
const abs=(a:Q)=>fraction(a.n<Z?-a.n:a.n,a.d);
const min=(a:Q,b:Q)=>cmp(a,b)<Z?a:b;
const sum=(v:Q[])=>v.reduce(add,q0);
const output=(v:Q)=>observe(v.n,v.d);
const matrixOutput=(a:Q[][])=>a.map(row=>row.map(output));
const advance=(v:Q[],p:Q[][])=>p.map((_,j)=>sum(v.map((x,i)=>mul(x,p[i][j]))));
const weight=(n:unknown):n is number=>typeof n==='number'&&Number.isInteger(n)&&n>=0&&n<=1000000;

export interface HastingsProposalParameters {
 targetWeights:readonly number[];
 forwardWeight:number;
 backwardWeight:number;
 stayWeight:number;
}

/** Exact elimination of all three balance equations plus normalization. */
function stationarySolution(p:Q[][]) {
 const rows=p.map((_,i)=>[...p.map((row,j)=>sub(row[i],i===j?q1:q0)),q0]);
 rows.push([q1,q1,q1,q1]);
 let rank=0;
 for(let col=0;col<3;col++){
  const pivot=rows.findIndex((row,i)=>i>=rank&&row[col].n!==Z);
  if(pivot<0)continue;
  [rows[rank],rows[pivot]]=[rows[pivot],rows[rank]];
  const scale=rows[rank][col];rows[rank]=rows[rank].map(v=>div(v,scale));
  for(let i=0;i<rows.length;i++)if(i!==rank){const factor=rows[i][col];rows[i]=rows[i].map((v,j)=>sub(v,mul(factor,rows[rank][j])));}
  rank++;
 }
 return {stationaryRank:rank-1,familyDimension:3-rank,distribution:rank===3?rows.slice(0,3).map(row=>row[3]):null};
}

/** Reachability and return periods use exact positivity, never projected sizes. */
function graph(p:Q[][]) {
 const support=p.map(row=>row.map(v=>v.n>Z)),reach=support.map((row,i)=>row.map((v,j)=>v||i===j));
 for(let k=0;k<3;k++)for(let i=0;i<3;i++)for(let j=0;j<3;j++)reach[i][j]=reach[i][j]||reach[i][k]&&reach[k][j];
 const classes:number[][]=[];
 for(let i=0;i<3;i++)if(!classes.some(group=>group.includes(i)))classes.push(reach[i].flatMap((v,j)=>v&&reach[j][i]?[j]:[]));
 const gcd=(a:number,b:number):number=>b===0?a:gcd(b,a%b),periods=[0,0,0];
 let power=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>i===j));
 for(let length=1;length<=9;length++){
  power=power.map(row=>row.map((_,j)=>row.some((v,k)=>v&&support[k][j])));
  periods.forEach((v,i)=>{if(power[i][i])periods[i]=gcd(v,length);});
 }
 return {irreducible:classes.length===1,communicatingClasses:classes,returnPeriods:periods,
  uniqueClassPeriod:classes.length===1?periods[0]:null};
}

export function metropolisHastingsProposalModel(input:HastingsProposalParameters) {
 if(!input||![input.forwardWeight,input.backwardWeight,input.stayWeight].every(weight)||
  input.forwardWeight+input.backwardWeight+input.stayWeight===0||!Array.isArray(input.targetWeights)||input.targetWeights.length!==3||
  !Array.from(input.targetWeights).every(n=>weight(n)&&n>0))
  throw new RangeError('Expected three positive integer target weights and nonzero-sum proposal weights, each in 0..1000000');
 const parameters=Object.freeze({targetWeights:Object.freeze(Array.from(input.targetWeights)),
  forwardWeight:input.forwardWeight+0,backwardWeight:input.backwardWeight+0,stayWeight:input.stayWeight+0});
 const w=parameters.targetWeights.map(BigInt),W=w.reduce((a,b)=>a+b,Z),target=w.map(n=>fraction(n,W));
 const f=BigInt(input.forwardWeight),b=BigInt(input.backwardWeight),s=BigInt(input.stayWeight),D=f+b+s;
 const proposal=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>fraction(i===j?s:j===(i+1)%3?f:b,D)));

 function prepare(corrected:boolean) {
  const acceptance=proposal.map((row,i)=>row.map((q,j)=>q.n===Z?null:
   min(q1,corrected?div(mul(target[j],proposal[j][i]),mul(target[i],q)):div(target[j],target[i]))));
  const accepted=proposal.map((row,i)=>row.map((q,j)=>mul(q,acceptance[i][j]??q0)));
  const rejected=proposal.map((row,i)=>row.map((q,j)=>sub(q,accepted[i][j]))),rejection=rejected.map(sum);
  const transition=accepted.map((row,i)=>row.map((q,j)=>i===j?add(q,rejection[i]):q));
  const targetFlux=transition.map((row,i)=>row.map(q=>mul(target[i],q)));
  const targetCurrent=targetFlux.map((row,i)=>row.map((q,j)=>sub(q,targetFlux[j][i])));
  const targetOneStep=advance(target,transition),targetResidual=targetOneStep.map((q,i)=>sub(q,target[i]));
  const targetInvariant=targetResidual.every(q=>q.n===Z),targetDetailedBalance=targetCurrent.every(row=>row.every(q=>q.n===Z));
  const solved=stationarySolution(transition),structure=graph(transition),stationary=solved.distribution;
  const stationaryFlux=stationary?transition.map((row,i)=>row.map(q=>mul(stationary[i],q))):null;
  const stationaryCurrent=stationaryFlux?stationaryFlux.map((row,i)=>row.map((q,j)=>sub(q,stationaryFlux[j][i]))):null;
  const stationaryResidual=stationary?advance(stationary,transition).map((q,i)=>sub(q,stationary[i])):null;
  const movement=transition.map((row,i)=>sub(q1,row[i]));
  const jumpTransition=movement.every(q=>q.n>Z)?transition.map((row,i)=>row.map((q,j)=>i===j?q0:div(q,movement[i]))):null;
  const stationaryMovement=stationary?sum(stationary.map((q,i)=>mul(q,movement[i]))):null;
  const jumpStationary=stationary&&stationaryMovement&&stationaryMovement.n>Z?
   stationary.map((q,i)=>div(mul(q,movement[i]),stationaryMovement)):null;
  const jumpResidual=jumpStationary&&jumpTransition?advance(jumpStationary,jumpTransition).map((q,i)=>sub(q,jumpStationary[i])):null;
  const rowDistances=transition.flatMap(row=>transition.map(other=>div(sum(row.map((q,j)=>abs(sub(q,other[j])))),fraction(BigInt(2)))));
  const coefficient=rowDistances.reduce((a,b)=>cmp(a,b)>Z?a:b,q0);
  const meanAccepted=sum(accepted.map((row,i)=>mul(target[i],sum(row)))),meanRejected=sum(rejection.map((q,i)=>mul(target[i],q)));
  const meanMoves=sum(accepted.map((row,i)=>mul(target[i],sum(row.filter((_,j)=>i!==j)))));
  return {
   acceptance:acceptance.map(row=>row.map(q=>q?output(q):null)),acceptedProposal:matrixOutput(accepted),rejectedProposal:matrixOutput(rejected),
   rejectionByState:rejection.map(output),selfProposalByState:proposal.map((row,i)=>output(row[i])),
   totalHoldingByState:transition.map((row,i)=>output(row[i])),transition:matrixOutput(transition),
   targetFlux:matrixOutput(targetFlux),targetCurrent:matrixOutput(targetCurrent),targetOneStep:targetOneStep.map(output),
   targetResidual:targetResidual.map(output),targetInvariant,targetDetailedBalance,
   stationaryRank:solved.stationaryRank,stationaryFamilyDimension:solved.familyDimension,
   stationaryDistribution:stationary?stationary.map(output):null,
   stationaryFlux:stationaryFlux?matrixOutput(stationaryFlux):null,stationaryCurrent:stationaryCurrent?matrixOutput(stationaryCurrent):null,
   stationaryResidual:stationaryResidual?stationaryResidual.map(output):null,
   stationaryTargetDistance:stationary?output(div(sum(stationary.map((q,i)=>abs(sub(q,target[i])))),fraction(BigInt(2)))):null,
   movementByState:movement.map(output),jumpTransition:jumpTransition?matrixOutput(jumpTransition):null,
   jumpStationaryDistribution:jumpStationary?jumpStationary.map(output):null,
   jumpStationaryResidual:jumpResidual?jumpResidual.map(output):null,
   jumpTargetDistance:jumpStationary?output(div(sum(jumpStationary.map((q,i)=>abs(sub(q,target[i])))),fraction(BigInt(2)))):null,
   meanMovementAtStationary:stationaryMovement?output(stationaryMovement):null,
   contractionCoefficient:output(coefficient),meanProposalAcceptanceAtTarget:output(meanAccepted),
   meanProposalRejectionAtTarget:output(meanRejected),meanMovementAtTarget:output(meanMoves),
   classification:{...structure,stationaryIsUnique:stationary!==null,targetIsUniqueStationary:targetInvariant&&stationary!==null,
    convergesToTargetFromEveryInitial:targetInvariant&&structure.irreducible&&structure.uniqueClassPeriod===1},
  };
 }
 const corrected=prepare(true),uncorrected=prepare(false);
 return {parameters,arithmetic:'exact-bigint-rational' as const,fullyResolved:true,target:target.map(output),proposal:matrixOutput(proposal),
  // With full target support, every uncorrected edge current is proportional
  // to (f-b) times a strictly positive minimum target weight.
  corrected,uncorrected,correctionChangesTransition:f!==b,
  approximationContract:'Three named states with full-support target integer weights in 1..1000000. Cyclic proposal forward/backward/holding integer weights lie in 0..1000000 with a positive sum. Exact BigInt rational Hastings acceptance is defined only for present proposals; an absent proposal has null conditional acceptance, a present proposal without a reverse has zero acceptance, and present self proposals are accepted by convention. Rejections remain at the departure state. A separately labeled uncorrected target-only Metropolis comparison omits the proposal ratio. Target-prepared fluxes and drift are distinct from the computed unique stationary law and its fluxes. A nonunique stationary family is not replaced by an arbitrary selected equilibrium. All bounds are exact rational points; Number values and magnitude logs are display projections. Acceptance includes self proposals, movement excludes holding, and neither statistic certifies mixing efficiency. The state-change jump chain removes all holding, including accepted self proposals and rejection repeats; its stationary law uses the actual unique stationary preparation and movement weights. Without movement its jump law is not defined. This is a fixed finite-kernel analysis, not a sampled trajectory or a physical-time process. Full-support restrictions, discrete reference measure and fixed proposal assumptions do not establish correctness for zero-target states, continuous densities, adaptive proposals, learned kernels or general LLM samplers. No physical heat or thermodynamic entropy production is inferred.'};
}
