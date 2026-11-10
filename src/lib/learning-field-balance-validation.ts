/** Independent forward differentiation of allocated energies; Worker-side only. */
import {readFieldBalanceInput,fieldBalanceInputKey,type FieldBalanceInput} from './learning-field-balance-input';
import {snapshotQubitWire} from './learning-qubit-wire';
import type {FieldBalanceResult} from './learning-field-balance';
class R {
 readonly n:bigint;readonly d:bigint;
 constructor(n:bigint,d=BigInt(1)){if(!d)throw Error('Zero denominator');if(d<BigInt(0)){n=-n;d=-d;}let x=n<BigInt(0)?-n:n,y=d;while(y)[x,y]=[y,x%y];this.n=n/x;this.d=d/x;}
 static read(text:string){const[n,d='1']=text.split('/');return new R(BigInt(n),BigInt(d));}
 add(y:R){return new R(this.n*y.d+y.n*this.d,this.d*y.d);}neg(){return new R(-this.n,this.d);}sub(y:R){return this.add(y.neg());}mul(y:R){return new R(this.n*y.n,this.d*y.d);}div(y:R){return new R(this.n*y.d,this.d*y.n);}text(){return this.d===BigInt(1)?String(this.n):`${this.n}/${this.d}`;}
}
const Z=new R(BigInt(0)),I=new R(BigInt(1)),two=new R(BigInt(2)),four=new R(BigInt(4));
const sum=(xs:R[])=>xs.reduce((x,y)=>x.add(y),Z);
type Gradient=Map<number,R>;
const scale=(g:Gradient,c:R)=>new Map([...g].map(([i,x])=>[i,x.mul(c)]));
function combine(...maps:Gradient[]):Gradient{const out:Gradient=new Map();for(const map of maps)for(const[i,x]of map){const next=(out.get(i)??Z).add(x);if(next.n===BigInt(0))out.delete(i);else out.set(i,next);}return out;}
/** g is a gradient; gt its directional derivative along fixed dq/ds=v. */
class J {
 constructor(readonly value:R,readonly g:Gradient=new Map(),readonly tangent:R=Z,readonly gt:Gradient=new Map()){}
 add(y:J){return new J(this.value.add(y.value),combine(this.g,y.g),this.tangent.add(y.tangent),combine(this.gt,y.gt));}
 neg(){return this.times(I.neg());}sub(y:J){return this.add(y.neg());}
 times(c:R){return new J(this.value.mul(c),scale(this.g,c),this.tangent.mul(c),scale(this.gt,c));}
 mul(y:J){return new J(this.value.mul(y.value),combine(scale(this.g,y.value),scale(y.g,this.value)),this.tangent.mul(y.value).add(this.value.mul(y.tangent)),combine(scale(this.gt,y.value),scale(this.g,y.tangent),scale(y.g,this.tangent),scale(y.gt,this.value)));}
 sq(){return this.mul(this);}
}
const total=(xs:J[])=>xs.reduce((x,y)=>x.add(y),new J(Z));
function record(value:unknown,keys:readonly string[]):Record<string,unknown>{if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)throw Error('Exact plain record required');const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))throw Error('Owned field required');out[key]=d.value;}return out;}
const same=(actual:unknown,expected:R|boolean|number|string|null)=>{if(actual!==(expected instanceof R?expected.text():expected))throw Error('Incorrect response claim');};
const list=(value:unknown,n:number)=>{if(!Array.isArray(value)||value.length!==n)throw Error('Incorrect array size');return value;};
const keys=['input','sites','edges','cells','prefixes','totalEnergy','energyPartitionResidual','energyDerivative','shiftCharge','shiftDerivative','shiftSecondDerivative','shiftSource','shiftBoundaryInflow','uniformFieldShiftSymmetry','instantaneousShiftDerivativeZero','boundaryEnergyAllocation','timeDiscretized','spatialGenerator'];
export function validateFieldBalanceResult(source:FieldBalanceInput,value:unknown):FieldBalanceResult|null{
 try{
  const input=readFieldBalanceInput(source);if(!input)return null;const owned=snapshotQubitWire(value),r=record(owned,keys),attributed=readFieldBalanceInput(r.input);if(!attributed||fieldBalanceInputKey(attributed)!==fieldBalanceInputKey(input))return null;
  const raw=input.field,n=raw.position.length,h=R.read(raw.spacing),c=R.read(raw.speed),a=R.read(raw.quadratic),b=R.read(raw.quartic),alpha=R.read(input.share),vel=raw.velocity.map(R.read);
  const q=raw.position.map((x,i)=>new J(R.read(x),new Map([[i,I]]),vel[i])),v=vel.map((x,i)=>new J(x,new Map([[n+i,I]])));
  const count=raw.boundary==='periodic'?n:raw.boundary==='dirichlet-zero'?n+1:n-1;
  const pairs=Array.from({length:count},(_,i):[number|null,number|null]=>raw.boundary==='periodic'?[i,(i+1)%n]:raw.boundary==='dirichlet-zero'?[i===0?null:i-1,i===n?null:i]:[i,i+1]);
  const edge=pairs.map(([l,rr])=>(rr===null?new J(Z):q[rr]).sub(l===null?new J(Z):q[l]).sq().times(c.mul(c).div(two.mul(h))));
  const onsite=q.map(x=>x.sq().times(a.div(two)).add(x.sq().sq().times(b.div(four))).times(h)),kinetic=v.map(x=>x.sq().times(h.div(two))),U=total([...edge,...onsite]),H=U.add(total(kinetic));
  const acceleration=q.map((_,i)=>(U.g.get(i)??Z).neg().div(h));
  const rate=(expression:J)=>sum([...expression.g].map(([i,partial])=>partial.mul(i<n?vel[i]:acceleration[i-n])));
  const energies=q.map((_,i)=>onsite[i].add(kinetic[i])),energyFlows:R[]=[],chargeFlows:R[]=[];
  pairs.forEach(([l,rr],i)=>{
   const E=edge[i];
   if(l!==null&&rr!==null){energies[l]=energies[l].add(E.times(alpha));energies[rr]=energies[rr].add(E.times(I.sub(alpha)));const force=(E.g.get(l)??Z).neg();energyFlows.push(force.mul(vel[l]).add(rate(E).mul(alpha)).neg());}
   else{const site=l??rr!;energies[site]=energies[site].add(E);energyFlows.push(Z);}
   chargeFlows.push(l===null?(E.g.get(rr!)??Z).neg():E.g.get(l)??Z);
  });
  const cut=(sites:Set<number>,flows:R[])=>sum(pairs.map(([l,rr],i)=>{const left=l!==null&&sites.has(l),right=rr!==null&&sites.has(rr);return left===right?Z:right?flows[i]:flows[i].neg();}));
  const charge=v.map(x=>x.times(h)),sourceTerms=onsite.map((V,i)=>(V.g.get(i)??Z).neg());
  list(r.edges,count).forEach((row,i)=>{const d=record(row,['left','right','energy','energyDerivative','energyCurrent','shiftCurrent']);same(d.left,pairs[i][0]);same(d.right,pairs[i][1]);same(d.energy,edge[i].value);same(d.energyDerivative,rate(edge[i]));same(d.energyCurrent,energyFlows[i]);same(d.shiftCurrent,chargeFlows[i]);});
  const cellRates=energies.map(rate),chargeRates=charge.map(rate);
  const prefix=energies.map((_,i)=>({energy:Z,rate:Z,charge:Z,qrate:Z,source:Z,index:i}));
  prefix.forEach((entry,i)=>{const prior=i?prefix[i-1]:{energy:Z,rate:Z,charge:Z,qrate:Z,source:Z};entry.energy=prior.energy.add(energies[i].value);entry.rate=prior.rate.add(cellRates[i]);entry.charge=prior.charge.add(charge[i].value);entry.qrate=prior.qrate.add(chargeRates[i]);entry.source=prior.source.add(sourceTerms[i]);});
  const cellKeys=['site','energy','energyDerivative','energyInflow','energyResidual','shiftCharge','shiftDerivative','shiftInflow','shiftSource','shiftResidual'];
  const prefixKeys=['lastSite',...cellKeys.slice(1)];
  const check=(row:unknown,sites:Set<number>,index:number,isPrefix:boolean)=>{
   const d=record(row,isPrefix?prefixKeys:cellKeys);same(d[isPrefix?'lastSite':'site'],index);const summary=isPrefix?prefix[index]:{energy:energies[index].value,rate:cellRates[index],charge:charge[index].value,qrate:chargeRates[index],source:sourceTerms[index]},Edot=summary.rate,Qdot=summary.qrate,Jin=cut(sites,energyFlows),Cin=cut(sites,chargeFlows),S=summary.source;
   same(d.energy,summary.energy);same(d.energyDerivative,Edot);same(d.energyInflow,Jin);same(d.energyResidual,Edot.sub(Jin));same(d.shiftCharge,summary.charge);same(d.shiftDerivative,Qdot);same(d.shiftInflow,Cin);same(d.shiftSource,S);same(d.shiftResidual,Qdot.sub(Cin).sub(S));
  };
  list(r.cells,n).forEach((row,i)=>check(row,new Set([i]),i,false));list(r.prefixes,n).forEach((row,i)=>check(row,new Set(Array.from({length:i+1},(_,j)=>j)),i,true));
  const Q=total(charge),Qdot=rate(Q),all=new Set(Array.from({length:n},(_,i)=>i));
  same(r.sites,n);same(r.totalEnergy,H.value);same(r.energyPartitionResidual,total(energies).value.sub(H.value));same(r.energyDerivative,rate(total(energies)));same(r.shiftCharge,Q.value);same(r.shiftDerivative,Qdot);same(r.shiftSecondDerivative,sum([...U.gt.values()]).neg());same(r.shiftSource,sum(sourceTerms));same(r.shiftBoundaryInflow,cut(all,chargeFlows));
  same(r.uniformFieldShiftSymmetry,a.n===BigInt(0)&&b.n===BigInt(0)&&raw.boundary!=='dirichlet-zero');same(r.instantaneousShiftDerivativeZero,Qdot.n===BigInt(0));same(r.boundaryEnergyAllocation,'full-to-adjacent-active-site');same(r.timeDiscretized,false);
  if(raw.boundary!=='periodic')same(r.spatialGenerator,null);
  else{
   const d=record(r.spatialGenerator,['value','derivative','velocityTerm','nonlinearDefect','balanceResidual','quadraticSkewGeneratorSymmetry','cyclicPermutationSymmetry','literalContinuousSpatialTranslationClaimed','convention']);
   const Dq=q.map((_,i)=>q[(i+1)%n].sub(q[(i+n-1)%n]).times(I.div(two.mul(h)))),Dv=v.map((_,i)=>v[(i+1)%n].sub(v[(i+n-1)%n]).times(I.div(two.mul(h))));
   const P=total(v.map((x,i)=>x.mul(Dq[i]))).times(h),vv=total(v.map((x,i)=>x.mul(Dv[i]))).times(h).value;
   // Variation of the onsite action along Dq independently yields the nonlinear defect.
   const defect=sum(onsite.map((V,i)=>(V.g.get(i)??Z).mul(Dq[i].value))).neg(),Pdot=rate(P);
   same(d.value,P.value);same(d.derivative,Pdot);same(d.velocityTerm,vv);same(d.nonlinearDefect,defect);same(d.balanceResidual,Pdot.sub(defect));same(d.quadraticSkewGeneratorSymmetry,b.n===BigInt(0));same(d.cyclicPermutationSymmetry,true);same(d.literalContinuousSpatialTranslationClaimed,false);same(d.convention,'delta-phi=+epsilon*Dphi');
  }
  return owned as FieldBalanceResult;
 }catch{return null;}
}
