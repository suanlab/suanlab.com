/** Exact local balances at a selected state. No time stepping or continuum claim. */
import {markovFraction as f,type MarkovFraction as F} from './learning-finite-markov-exact';
import {calculateScalarField} from './learning-scalar-field';
import {readFieldBalanceInput,type FieldBalanceInput} from './learning-field-balance-input';
const Z=f(BigInt(0)),I=f(BigInt(1)),T=f(BigInt(2)),three=f(BigInt(3));
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),sq=(x:F)=>mul(x,x),sum=(xs:F[])=>xs.reduce(add,Z);
const read=(s:string)=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));},wire=(x:F)=>x.d===BigInt(1)?String(x.n):`${x.n}/${x.d}`;
export interface FieldBalanceEdge {left:number|null;right:number|null;energy:string;energyDerivative:string;energyCurrent:string;shiftCurrent:string;}
export interface FieldBalanceCell {site:number;energy:string;energyDerivative:string;energyInflow:string;energyResidual:string;shiftCharge:string;shiftDerivative:string;shiftInflow:string;shiftSource:string;shiftResidual:string;}
export interface FieldBalancePrefix {lastSite:number;energy:string;energyDerivative:string;energyInflow:string;energyResidual:string;shiftCharge:string;shiftDerivative:string;shiftInflow:string;shiftSource:string;shiftResidual:string;}
export interface FieldBalanceResult {
 input:FieldBalanceInput;sites:number;edges:FieldBalanceEdge[];cells:FieldBalanceCell[];prefixes:FieldBalancePrefix[];
 totalEnergy:string;energyPartitionResidual:string;energyDerivative:string;
 shiftCharge:string;shiftDerivative:string;shiftSecondDerivative:string;shiftSource:string;shiftBoundaryInflow:string;
 uniformFieldShiftSymmetry:boolean;instantaneousShiftDerivativeZero:boolean;
 boundaryEnergyAllocation:'full-to-adjacent-active-site';timeDiscretized:false;
 spatialGenerator:null|{value:string;derivative:string;velocityTerm:string;nonlinearDefect:string;balanceResidual:string;quadraticSkewGeneratorSymmetry:boolean;cyclicPermutationSymmetry:true;literalContinuousSpatialTranslationClaimed:false;convention:'delta-phi=+epsilon*Dphi'};
}
export function calculateFieldBalance(value:unknown):FieldBalanceResult{
 const input=readFieldBalanceInput(value);if(!input)throw new RangeError('Invalid local-balance input');
 const base=calculateScalarField(input.field),q=input.field.position.map(read),v=input.field.velocity.map(read),acc=base.acceleration.map(read),n=q.length,h=read(input.field.spacing),k=div(sq(read(input.field.speed)),h),a=read(input.field.quadratic),b=read(input.field.quartic),alpha=read(input.share),other=sub(I,alpha);
 const source=q.map(x=>neg(mul(h,add(mul(a,x),mul(b,mul(sq(x),x))))));
 const energy=base.kineticBySite.map((x,i)=>add(read(x),read(base.localPotentialBySite[i]))),energyDot=v.map((x,i)=>sub(mul(h,mul(x,acc[i])),mul(source[i],x))),incoming=Array.from({length:n},()=>Z),shiftIncoming=Array.from({length:n},()=>Z);
 const energyFlows:F[]=[],shiftFlows:F[]=[];
 const edges=base.edges.map(edge=>{
  const{left,right}=edge,l=left===null?Z:v[left],r=right===null?Z:v[right],difference=read(edge.difference),E=read(edge.energy),Edot=mul(k,mul(difference,sub(r,l))),shiftCurrent=neg(mul(k,difference));
  const internal=left!==null&&right!==null,energyCurrent=internal?neg(mul(k,mul(difference,add(mul(other,l),mul(alpha,r))))):Z;
  const assign=(site:number,share:F)=>{energy[site]=add(energy[site],mul(share,E));energyDot[site]=add(energyDot[site],mul(share,Edot));};
  if(internal){assign(left!,alpha);assign(right!,other);}else assign(left??right!,I);
  if(left!==null){incoming[left]=sub(incoming[left],energyCurrent);shiftIncoming[left]=sub(shiftIncoming[left],shiftCurrent);}
  if(right!==null){incoming[right]=add(incoming[right],energyCurrent);shiftIncoming[right]=add(shiftIncoming[right],shiftCurrent);}
  energyFlows.push(energyCurrent);shiftFlows.push(shiftCurrent);
  return{left,right,energy:wire(E),energyDerivative:wire(Edot),energyCurrent:wire(energyCurrent),shiftCurrent:wire(shiftCurrent)};
 });
 const charge=v.map(x=>mul(h,x)),chargeDot=acc.map(x=>mul(h,x));
 const cells=q.map((_,i)=>({site:i,energy:wire(energy[i]),energyDerivative:wire(energyDot[i]),energyInflow:wire(incoming[i]),energyResidual:wire(sub(energyDot[i],incoming[i])),shiftCharge:wire(charge[i]),shiftDerivative:wire(chargeDot[i]),shiftInflow:wire(shiftIncoming[i]),shiftSource:wire(source[i]),shiftResidual:wire(sub(chargeDot[i],add(shiftIncoming[i],source[i])))}));
 const cut=(last:number,flows:F[])=>sum(base.edges.map((edge,i)=>{
  const inside=(site:number|null)=>site!==null&&site<=last;
  return inside(edge.left)===inside(edge.right)?Z:inside(edge.right)?flows[i]:neg(flows[i]);
 }));
 const prefixes=q.map((_,last)=>{const take=(xs:F[])=>sum(xs.slice(0,last+1)),Ei=take(energy),dot=take(energyDot),flow=cut(last,energyFlows),Qi=take(charge),Qdot=take(chargeDot),Qflow=cut(last,shiftFlows),Qs=take(source);return{lastSite:last,energy:wire(Ei),energyDerivative:wire(dot),energyInflow:wire(flow),energyResidual:wire(sub(dot,flow)),shiftCharge:wire(Qi),shiftDerivative:wire(Qdot),shiftInflow:wire(Qflow),shiftSource:wire(Qs),shiftResidual:wire(sub(Qdot,add(Qflow,Qs)))};});
 const sourceDerivative=neg(mul(h,sum(q.map((x,i)=>mul(add(a,mul(three,mul(b,sq(x)))),v[i]))))),boundaryDerivative=input.field.boundary==='dirichlet-zero'?neg(mul(k,add(v[0],v[n-1]))):Z;
 let spatialGenerator:FieldBalanceResult['spatialGenerator']=null;
 if(input.field.boundary==='periodic'){
  const D=(xs:F[])=>xs.map((_,i)=>div(sub(xs[(i+1)%n],xs[(i+n-1)%n]),mul(T,h))),Dq=D(q),Dv=D(v),velocityTerm=mul(h,sum(v.map((x,i)=>mul(x,Dv[i])))),value=mul(h,sum(v.map((x,i)=>mul(x,Dq[i])))),derivative=add(mul(h,sum(acc.map((x,i)=>mul(x,Dq[i])))),velocityTerm),defect=neg(mul(h,mul(b,sum(q.map((x,i)=>mul(mul(sq(x),x),Dq[i]))))));
  spatialGenerator={value:wire(value),derivative:wire(derivative),velocityTerm:wire(velocityTerm),nonlinearDefect:wire(defect),balanceResidual:wire(sub(derivative,defect)),quadraticSkewGeneratorSymmetry:b.n===BigInt(0),cyclicPermutationSymmetry:true,literalContinuousSpatialTranslationClaimed:false,convention:'delta-phi=+epsilon*Dphi'};
 }
 return{input,sites:n,edges,cells,prefixes,totalEnergy:base.energy,energyPartitionResidual:wire(sub(sum(energy),read(base.energy))),energyDerivative:wire(sum(energyDot)),shiftCharge:wire(sum(charge)),shiftDerivative:wire(sum(chargeDot)),shiftSecondDerivative:wire(add(sourceDerivative,boundaryDerivative)),shiftSource:wire(sum(source)),shiftBoundaryInflow:wire(sum(shiftIncoming)),uniformFieldShiftSymmetry:a.n===BigInt(0)&&b.n===BigInt(0)&&input.field.boundary!=='dirichlet-zero',instantaneousShiftDerivativeZero:sum(chargeDot).n===BigInt(0),boundaryEnergyAllocation:'full-to-adjacent-active-site',timeDiscretized:false,spatialGenerator};
}
