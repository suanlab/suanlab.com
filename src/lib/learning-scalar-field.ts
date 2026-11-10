/** Exact spatial discretization; time remains continuous. No time integrator. */
import {markovFraction as f,type MarkovFraction as F} from './learning-finite-markov-exact';
import {readScalarFieldInput,type ScalarFieldInput} from './learning-scalar-field-input';
const z=f(BigInt(0)),one=f(BigInt(1)),two=f(BigInt(2)),four=f(BigInt(4));
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),square=(a:F)=>mul(a,a),sum=(a:F[])=>a.reduce(add,z);
const read=(s:string)=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const wire=(x:F)=>x.d===one.n?x.n.toString():`${x.n}/${x.d}`;
export interface ScalarFieldEdge {left:number|null;right:number|null;difference:string;energy:string;}
export interface ScalarFieldResult {
 input:ScalarFieldInput;sites:number;length:string;massPerSite:string;positions:string[];
 edges:ScalarFieldEdge[];kineticBySite:string[];localPotentialBySite:string[];
 potentialGradient:string[];acceleration:string[];potentialHessian:string[][];
 kineticEnergy:string;gradientEnergy:string;localPotentialEnergy:string;energy:string;
 potentialPower:string;kineticPower:string;energyDerivative:string;
 totalMomentum:string;momentumDerivative:string;localPotentialForce:string;boundaryForce:string;
 localPotentialLowerBound:string|null;timeDiscretized:false;
}
export function calculateScalarField(value:unknown):ScalarFieldResult{
 const input=readScalarFieldInput(value);if(!input)throw new RangeError('Invalid scalar-field input');
 const q=input.position.map(read),v=input.velocity.map(read),n=q.length,h=read(input.spacing),c2=square(read(input.speed)),a=read(input.quadratic),b=read(input.quartic),stiffness=div(c2,h);
 const gradient=q.map(x=>mul(h,add(mul(a,x),mul(b,mul(square(x),x)))));
 const curvature=q.map(x=>mul(h,add(a,mul(f(BigInt(3)),mul(b,square(x))))));
 const hessian=q.map((_,i)=>q.map((_,j)=>i===j?curvature[i]:z));
 const edges:ScalarFieldEdge[]=[],edgeEnergies:F[]=[];
 const edge=(left:number|null,right:number|null)=>{
  const difference=sub(right===null?z:q[right],left===null?z:q[left]),force=mul(stiffness,difference),energy=div(mul(force,difference),two);
  if(left!==null){gradient[left]=sub(gradient[left],force);hessian[left][left]=add(hessian[left][left],stiffness);}
  if(right!==null){gradient[right]=add(gradient[right],force);hessian[right][right]=add(hessian[right][right],stiffness);}
  if(left!==null&&right!==null){hessian[left][right]=sub(hessian[left][right],stiffness);hessian[right][left]=sub(hessian[right][left],stiffness);}
  edges.push({left,right,difference:wire(difference),energy:wire(energy)});edgeEnergies.push(energy);
 };
 if(input.boundary==='dirichlet-zero')edge(null,0);
 for(let i=0;i<n-1;i++)edge(i,i+1);
 if(input.boundary==='periodic')edge(n-1,0);else if(input.boundary==='dirichlet-zero')edge(n-1,null);
 const acceleration=gradient.map(x=>neg(div(x,h))),kinetic=q.map((_,i)=>div(mul(h,square(v[i])),two));
 const onsite=q.map(x=>mul(h,add(div(mul(a,square(x)),two),div(mul(b,square(square(x))),four))));
 const potentialPower=sum(gradient.map((x,i)=>mul(x,v[i]))),kineticPower=mul(h,sum(v.map((x,i)=>mul(x,acceleration[i]))));
 const momentumDerivative=mul(h,sum(acceleration)),localPotentialForce=neg(mul(h,sum(q.map(x=>add(mul(a,x),mul(b,mul(square(x),x)))))));
 const localBound=a.n>=BigInt(0)?z:b.n>BigInt(0)?neg(div(square(a),mul(four,b))):null;
 const offset=input.boundary==='neumann-zero'?div(one,two):input.boundary==='dirichlet-zero'?one:z;
 return {input,sites:n,length:wire(mul(h,f(BigInt(input.boundary==='dirichlet-zero'?n+1:n)))),massPerSite:wire(h),positions:q.map((_,i)=>wire(mul(h,add(f(BigInt(i)),offset)))),edges,
  kineticBySite:kinetic.map(wire),localPotentialBySite:onsite.map(wire),potentialGradient:gradient.map(wire),acceleration:acceleration.map(wire),potentialHessian:hessian.map(row=>row.map(wire)),
  kineticEnergy:wire(sum(kinetic)),gradientEnergy:wire(sum(edgeEnergies)),localPotentialEnergy:wire(sum(onsite)),energy:wire(add(add(sum(kinetic),sum(edgeEnergies)),sum(onsite))),potentialPower:wire(potentialPower),kineticPower:wire(kineticPower),energyDerivative:wire(add(potentialPower,kineticPower)),
  totalMomentum:wire(mul(h,sum(v))),momentumDerivative:wire(momentumDerivative),localPotentialForce:wire(localPotentialForce),boundaryForce:wire(sub(momentumDerivative,localPotentialForce)),localPotentialLowerBound:localBound===null?null:wire(localBound),timeDiscretized:false};
}
