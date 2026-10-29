import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';
export type Complex=Vec2;
export const complexKinds=['complex-polar-maps','cauchy-riemann-holomorphic','complex-contour-integrals','residues-winding','analytic-functions-singularities','analytic-continuation-branches','complex-linear-algebra','wirtinger-gradients','complex-signals-phase'] as const;
export type ComplexKind=typeof complexKinds[number];
export function isComplexKind(kind:string):kind is ComplexKind{return (complexKinds as readonly string[]).includes(kind);}
export const complexMultiply=([a,b]:Complex,[c,d]:Complex):Complex=>[a*c-b*d,a*d+b*c];
export function complexReciprocal([x,y]:Complex):Complex|null {const scale=Math.max(Math.abs(x),Math.abs(y));if(scale===0)return null;const a=x/scale,b=y/scale,denom=a*a+b*b;return [a/denom/scale,-b/denom/scale];}
export function complexPower(z:Complex,n:number):Complex{if(!Number.isInteger(n)||n<0||n>4||!z.every(Number.isFinite))throw new Error('Use a finite complex value and power 0–4.');let result:Complex=[1,0];for(let i=0;i<n;i++)result=complexMultiply(result,z);return result;}
const polar=(r:number,t:number):Complex=>[r*Math.cos(t),r*Math.sin(t)];
const sample=(n:number,f:(i:number)=>Vec2)=>Array.from({length:n},(_,i)=>f(i));
function bound(x:number,lo:number,hi:number){if(x<lo||x>hi)throw new Error('Parameter outside the complex example domain.');}
function mode(x:number,lo:number,hi:number){bound(x,lo,hi);if(!Number.isInteger(x))throw new Error('Use an integer mode or power.');}
/** power=0 integrates z; power>0 integrates (z−pole)^(-power). */
export function circleQuadrature(radius:number,pole:number,power:number,direction=1,panels=512):Complex|null{
 if(!Number.isFinite(radius)||!Number.isFinite(pole)||radius<0||![1,-1].includes(direction)||!Number.isInteger(power)||power<0||power>3||!Number.isInteger(panels)||panels<8||panels>65536)throw new Error('Invalid contour quadrature.');
 if(power>0&&Math.abs(pole)===radius)return null;let re=0,im=0;const dt=2*Math.PI/panels;
 for(let k=0;k<panels;k++){const t=direction*(k+.5)*dt,z=polar(radius,t),dz:Complex=[-direction*radius*Math.sin(t),direction*radius*Math.cos(t)];const f=power===0?z:complexPower(complexReciprocal([z[0]-pole,z[1]])!,power);const term=complexMultiply(f,dz);re+=term[0]*dt;im+=term[1]*dt;}
 return [re,im];
}
function sinc(z:Complex):Complex{let result:Complex=[1,0],term:Complex=[1,0];const z2=complexMultiply(z,z);for(let k=1;k<=20;k++){term=complexMultiply(term,[-z2[0]/((2*k)*(2*k+1)),-z2[1]/((2*k)*(2*k+1))]);result=[result[0]+term[0],result[1]+term[1]];}return result;}
function singularValue(which:number,z:Complex):Complex|null{if(which===0)return sinc(z);const inverse=complexReciprocal(z);if(inverse===null)return null;if(which===1)return inverse;const mag=Math.exp(inverse[0]);return [mag*Math.cos(inverse[1]),mag*Math.sin(inverse[1])];}
function evaluateRaw(kind:ComplexKind,p:number[]):FoundationResult{
 const counts:Record<ComplexKind,number>={'complex-polar-maps':3,'cauchy-riemann-holomorphic':3,'complex-contour-integrals':3,'residues-winding':3,'analytic-functions-singularities':3,'analytic-continuation-branches':3,'complex-linear-algebra':2,'wirtinger-gradients':4,'complex-signals-phase':3};
 if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Use finite complex parameters of the expected shape.');const [a,b,c,d]=p;
 if(kind==='complex-polar-maps'){
  bound(a,0,2);bound(b,-180,180);mode(c,1,4);const theta=b*Math.PI/180,z=polar(a,theta),w=complexPower(z,c),underflow=a>0&&Math.hypot(...w)===0;
  return {statement:`z=r(cosθ+i sinθ); r=${a}; θ=${b}°; w=z^${c}`,stats:{'z (Re,Im)':z,'w (Re,Im)':underflow?null:w,'|w|':underflow?null:Math.hypot(...w),'arg(w) [degrees]':a===0?null:Math.atan2(Math.sin(c*theta),Math.cos(c*theta))*180/Math.PI,'unwrapped kθ [degrees]':c*b},curves:underflow?[[[0,0],z]]:[[[0,0],z],[[0,0],w]],warning:underflow?'numeric-range':undefined};
 }
 if(kind==='cauchy-riemann-holomorphic'){
  mode(a,0,2);bound(b,-1,1);bound(c,-1,1);const f=(x:number,y:number):Complex=>a===0?[x*x-y*y,2*x*y]:a===1?[x,-y]:[x*x+y*y,0];const ux=a===0?2*b:a===1?1:2*b,uy=a===0?-2*c:a===1?0:2*c,vx=a===0?2*c:0,vy=a===0?2*b:a===1?-1:0;const derivative=a===0?[2*b,2*c]:a===2&&b===0&&c===0?[0,0]:null;
  return {statement:`f(z)=${a===0?'z²':a===1?'conj(z)':'|z|²'}; z₀=${b}+(${c})i`,stats:{'Cauchy–Riemann residuals':[ux-vy,uy+vx],'x-direction quotient limit':[ux,vx],'iy-direction quotient limit':[vy,-uy],'complex derivative':derivative,'1{holomorphic near z₀}':a===0?1:0},curves:[sample(81,i=>f(b-.5+i/80,c)),sample(81,i=>f(b,c-.5+i/80))]};
 }
 if(kind==='complex-contour-integrals'){
  mode(a,0,1);bound(b,0,2);mode(c,0,1);const direction=c===0?1:-1,numeric=circleQuadrature(b,0,a,direction),analytic:Complex|null=a===1&&b===0?null:[0,a===0?0:direction*2*Math.PI];
  return {statement:`∮ f(z)dz; f=${a===0?'z':'1/z'}; radius=${b}; direction=${direction===1?'CCW':'CW'}`,stats:{'numeric integral (Re,Im)':numeric,'analytic integral (Re,Im)':analytic,'error norm':numeric&&analytic?Math.hypot(numeric[0]-analytic[0],numeric[1]-analytic[1]):null},curves:[sample(81,i=>polar(b,direction*i/80*2*Math.PI)),...[0,1,2,3].map(i=>{const t=i*Math.PI/2,z=polar(b,t);return [z,[z[0]-.2*direction*b*Math.sin(t),z[1]+.2*direction*b*Math.cos(t)]] as Vec2[];})],warning:analytic===null?'function-domain':undefined};
 }
 if(kind==='residues-winding'){
  bound(a,-2,2);bound(b,.25,2);mode(c,1,3);const on=Math.abs(a)===b,inside=Math.abs(a)<b,numeric=circleQuadrature(b,a,c),analytic:Complex|null=on?null:[0,inside&&c===1?2*Math.PI:0];
  return {statement:`f(z)=1/(z−(${a}))^${c}; CCW circle |z|=${b}`,stats:{'residue at pole':c===1?1:0,'winding about pole':on?null:inside?1:0,'numeric integral (Re,Im)':numeric,'analytic integral (Re,Im)':analytic,'distance pole to contour':Math.abs(Math.abs(a)-b),'error norm':numeric&&analytic?Math.hypot(numeric[0]-analytic[0],numeric[1]-analytic[1]):null},curves:[sample(81,i=>polar(b,i/80*2*Math.PI)),[[a,0]]],holes:[[a,0]],warning:on?'function-domain':undefined};
 }
 if(kind==='analytic-functions-singularities'){
  mode(a,0,2);bound(b,0,1);bound(c,-180,180);const theta=c*Math.PI/180,z=polar(b,theta),value=singularValue(a,z),curve=(part:number)=>sample(91,i=>{const r=.1+i*.01;return [r,singularValue(a,polar(r,theta))![part]];});
  return {statement:`f(z)=${a===0?'sin(z)/z, extended at 0':a===1?'1/z':'exp(1/z)'}; z=r·exp(iθ), r=${b}, θ=${c}°`,stats:{'f(z) (Re,Im)':value,'value at zero after removable extension':a===0?1:null,'Laurent residue':a===0?0:1,'singularity type code':a},curves:[curve(0),curve(1)],warning:value===null?'function-domain':undefined};
 }
 if(kind==='analytic-continuation-branches'){
  bound(a,-170,170);mode(b,-2,2);bound(c,.25,2);const principal=a*Math.PI/180,lift=principal+2*Math.PI*b,z=polar(c,principal),root=polar(Math.sqrt(c),lift/2),path=sample(161,i=>polar(c,lift*i/160)),rootPath=sample(161,i=>polar(Math.sqrt(c),lift*i/320));
  return {statement:`z=r·exp(iθ), r=${c}; θ=${a}°; lifted angle Θ=θ+2π(${b})`,stats:{'principal log (Re,Im)':[Math.log(c),principal],'continued log (Re,Im)':[Math.log(c),lift],'continued sqrt (Re,Im)':root,'log increment (Im)':2*Math.PI*b,'exp(log) residual':Math.hypot(c*Math.cos(lift)-z[0],c*Math.sin(lift)-z[1])},curves:[path,rootPath]};
 }
 if(kind==='complex-linear-algebra'){
  bound(a,-180,180);bound(b,0,2);const t=a*Math.PI/180,w0:Complex=[b,0],w1=polar(b,t),hermitian:Complex=[b+w1[1],-w1[0]],bilinear:Complex=[b-w1[1],w1[0]];
  return {statement:`v=(1,i); w=${b}·(1,exp(iθ)); θ=${a}°; inner v* w`,stats:{'Hermitian v* w':hermitian,'bilinear vᵀw':bilinear,'v* v':2,'vᵀv':0,'‖v‖₂‖w‖₂':2*b,'normalized |v* w|':b===0?null:Math.hypot(...hermitian)/(2*b)},curves:[[[0,0],w0],[[0,0],w1]]};
 }
 if(kind==='wirtinger-gradients'){
  bound(a,-2,2);bound(b,-2,2);bound(c,0,1.5);mode(d,0,1);const center:Complex=d===0?[0,0]:[1,1],offset:Complex=[a-center[0],b-center[1]],next:Complex=[a-2*c*offset[0],b-2*c*offset[1]],trajectory:Vec2[]=[[a,b]];for(let i=0;i<6;i++){const last=trajectory[trajectory.length-1];trajectory.push([last[0]-2*c*(last[0]-center[0]),last[1]-2*c*(last[1]-center[1])]);}
  return {statement:`L(z)=|z−c₀|²; c₀=${center[0]}+${center[1]}i; η=${c}; z_next=z−2η·∂L/∂conj(z)`,stats:{'∂L/∂z':[offset[0],-offset[1]],'∂L/∂conj(z)':offset,'real gradient (x,y)':[2*offset[0],2*offset[1]],'next z':next,'L before':offset[0]**2+offset[1]**2,'L after':(next[0]-center[0])**2+(next[1]-center[1])**2},curves:[trajectory,[center]]};
 }
 bound(a,-180,180);bound(b,-180,180);bound(c,0,2);const t1=a*Math.PI/180,t2=b*Math.PI/180,z1=polar(1,t1),z2=polar(c,t2),opposite=c===1&&(b-a===180||b-a===-180),sum:Complex=opposite?[0,0]:[z1[0]+z2[0],z1[1]+z2[1]],magnitude=Math.hypot(...sum);
 return {statement:`s(t)=Re[(exp(iφ₁)+A₂exp(iφ₂))exp(it)]; φ₁=${a}°, φ₂=${b}°, A₂=${c}`,stats:{'combined phasor (Re,Im)':sum,'combined amplitude':magnitude,'combined phase [degrees]':magnitude===0?null:Math.atan2(sum[1],sum[0])*180/Math.PI,'power amplitude²':magnitude*magnitude},curves:[sample(129,i=>{const t=i/128*2*Math.PI;return [t,Math.cos(t+t1)];}),sample(129,i=>{const t=i/128*2*Math.PI;return [t,sum[0]*Math.cos(t)-sum[1]*Math.sin(t)];})]};
}
export function evaluateComplex(kind:ComplexKind,p:number[]):FoundationResult{
 const result=evaluateRaw(kind,p);let overflow=false;result.stats=Object.fromEntries(Object.entries(result.stats).map(([key,value])=>{if(value!==null&&(Array.isArray(value)?value.some(x=>!Number.isFinite(x)):!Number.isFinite(value))){overflow=true;return [key,null];}return [key,value];}));if(overflow)result.warning='numeric-range';return result;
}
