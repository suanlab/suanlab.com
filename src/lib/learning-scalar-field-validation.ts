/** Worker-only independent energy-difference checks. Never calls the producer. */
import {readScalarFieldInput,scalarFieldInputKey,type ScalarFieldInput} from './learning-scalar-field-input';
import {snapshotQubitWire} from './learning-qubit-wire';
import type {ScalarFieldResult} from './learning-scalar-field';
class R {
 readonly n:bigint;readonly d:bigint;
 constructor(n:bigint,d=BigInt(1)){if(d===BigInt(0))throw Error('Zero denominator');if(d<BigInt(0)){n=-n;d=-d;}let a=n<BigInt(0)?-n:n,b=d;while(b){[a,b]=[b,a%b];}this.n=n/a;this.d=d/a;}
 static read(s:unknown){if(typeof s!=='string'||s.length>8192||!/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(s))throw Error('Bounded rational required');const[n,d='1']=s.split('/');return new R(BigInt(n),BigInt(d));}
 add(y:R){return new R(this.n*y.d+y.n*this.d,this.d*y.d);}neg(){return new R(-this.n,this.d);}sub(y:R){return this.add(y.neg());}mul(y:R){return new R(this.n*y.n,this.d*y.d);}div(y:R){return new R(this.n*y.d,this.d*y.n);}sq(){return this.mul(this);}text(){return this.d===BigInt(1)?String(this.n):`${this.n}/${this.d}`;}
}
const Z=new R(BigInt(0)),I=new R(BigInt(1)),T=new R(BigInt(2)),F=new R(BigInt(4));
const total=(values:R[])=>values.reduce((a,b)=>a.add(b),Z);
function record(value:unknown,keys:readonly string[]):Record<string,unknown>{if(!value||typeof value!=='object'||Array.isArray(value)||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)throw Error('Exact record required');const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))throw Error('Owned field required');out[key]=d.value;}return out;}
const resultKeys=['input','sites','length','massPerSite','positions','edges','kineticBySite','localPotentialBySite','potentialGradient','acceleration','potentialHessian','kineticEnergy','gradientEnergy','localPotentialEnergy','energy','potentialPower','kineticPower','energyDerivative','totalMomentum','momentumDerivative','localPotentialForce','boundaryForce','localPotentialLowerBound','timeDiscretized'];
/** Canonical exact strings matter: no rounded display value is trusted. */
function same(value:unknown,expected:R){if(value!==expected.text())throw Error('Incorrect exact value');}
function vector(value:unknown,expected:R[]){if(!Array.isArray(value)||value.length!==expected.length)throw Error('Incorrect vector size');expected.forEach((x,i)=>same(value[i],x));}
export function validateScalarFieldResult(source:ScalarFieldInput,value:unknown):ScalarFieldResult|null{
 try{
  const input=readScalarFieldInput(source);if(!input)return null;
  const owned=snapshotQubitWire(value),r=record(owned,resultKeys),attributed=readScalarFieldInput(r.input);
  if(!attributed||scalarFieldInputKey(attributed)!==scalarFieldInputKey(input))return null;
  const q=input.position.map(R.read),v=input.velocity.map(R.read),n=q.length,h=R.read(input.spacing),speed=R.read(input.speed),a=R.read(input.quadratic),b=R.read(input.quartic),coupling=speed.sq().div(T.mul(h));
  if(r.sites!==n||r.timeDiscretized!==false)return null;
  const count=input.boundary==='periodic'?n:input.boundary==='dirichlet-zero'?n+1:n-1;
  const pairs=Array.from({length:count},(_,i):[number|null,number|null]=>input.boundary==='periodic'?[i,(i+1)%n]:input.boundary==='dirichlet-zero'?[i===0?null:i-1,i===n?null:i]:[i,i+1]);
  const potential=(x:R)=>h.mul(a.mul(x.sq()).div(T).add(b.mul(x.sq().sq()).div(F)));
  const edgeEnergy=(values:R[],pair:[number|null,number|null])=>coupling.mul((pair[0]===null?Z:values[pair[0]]).sub(pair[1]===null?Z:values[pair[1]]).sq());
  // Only terms depending on selected coordinates enter a directional difference.
  const localEnergy=(indices:number[],changes:Map<number,R>)=>{
   const values=q.map((x,i)=>changes.get(i)??x);
   return total(indices.map(i=>potential(values[i]))).add(total(pairs.filter(([l,rr])=>indices.some(i=>i===l||i===rr)).map(pair=>edgeEnergy(values,pair))));
  };
  const gradient:R[]=[],diagonal:R[]=[];
  for(let i=0;i<n;i++){
   const plus=localEnergy([i],new Map([[i,q[i].add(I)]])),minus=localEnergy([i],new Map([[i,q[i].sub(I)]])),center=localEnergy([i],new Map());
   gradient.push(plus.sub(minus).div(T).sub(h.mul(b).mul(q[i])));
   diagonal.push(plus.add(minus).sub(T.mul(center)).sub(h.mul(b).div(T)));
  }
  const hessian=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>{
   if(i===j)return diagonal[i];
   if(!pairs.some(([l,rr])=>(l===i&&rr===j)||(l===j&&rr===i)))return Z;
   const e=(si:R,sj:R)=>localEnergy([i,j],new Map([[i,q[i].add(si)],[j,q[j].add(sj)]]));
   return e(I,I).sub(e(I,I.neg())).sub(e(I.neg(),I)).add(e(I.neg(),I.neg())).div(F);
  }));
  const acceleration=gradient.map(x=>x.neg().div(h)),kinetic=v.map(x=>h.mul(x.sq()).div(T)),onsite=q.map(potential),edges=pairs.map(pair=>edgeEnergy(q,pair));
  const pp=total(gradient.map((x,i)=>x.mul(v[i]))),kp=h.mul(total(acceleration.map((x,i)=>x.mul(v[i])))),momentumRate=h.mul(total(acceleration)),localForce=total(q.map(x=>h.mul(a.mul(x).add(b.mul(x.sq().mul(x)))))).neg();
  same(r.length,h.mul(new R(BigInt(input.boundary==='dirichlet-zero'?n+1:n))));same(r.massPerSite,h);
  const offset=input.boundary==='neumann-zero'?I.div(T):input.boundary==='dirichlet-zero'?I:Z;
  vector(r.positions,q.map((_,i)=>h.mul(new R(BigInt(i)).add(offset))));vector(r.kineticBySite,kinetic);vector(r.localPotentialBySite,onsite);vector(r.potentialGradient,gradient);vector(r.acceleration,acceleration);
  if(!Array.isArray(r.potentialHessian)||r.potentialHessian.length!==n)return null;r.potentialHessian.forEach((row,i)=>vector(row,hessian[i]));
  if(!Array.isArray(r.edges)||r.edges.length!==count)return null;
  r.edges.forEach((edge,i)=>{const d=record(edge,['left','right','difference','energy']),[left,right]=pairs[i];if(d.left!==left||d.right!==right)throw Error('Incorrect edge attribution');same(d.difference,(right===null?Z:q[right]).sub(left===null?Z:q[left]));same(d.energy,edges[i]);});
  same(r.kineticEnergy,total(kinetic));same(r.gradientEnergy,total(edges));same(r.localPotentialEnergy,total(onsite));same(r.energy,total(kinetic).add(total(edges)).add(total(onsite)));
  same(r.potentialPower,pp);same(r.kineticPower,kp);same(r.energyDerivative,pp.add(kp));same(r.totalMomentum,h.mul(total(v)));same(r.momentumDerivative,momentumRate);same(r.localPotentialForce,localForce);same(r.boundaryForce,momentumRate.sub(localForce));
  const bound=a.n>=BigInt(0)?Z:b.n>BigInt(0)?a.sq().div(F.mul(b)).neg():null;
  if(bound===null){if(r.localPotentialLowerBound!==null)return null;}else same(r.localPotentialLowerBound,bound);
  return owned as ScalarFieldResult;
 }catch{return null;}
}
