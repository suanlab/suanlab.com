import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';
export const elementaryKinds=['numbers-ratios-units','real-complex-numbers','equations-inequalities','polynomial-rational-functions','exponentials-logarithms','trigonometry-coordinates','composition-inverse-functions','sequences-series','asymptotic-growth'] as const;
export type ElementaryKind=typeof elementaryKinds[number];
export function isElementaryKind(kind:string):kind is ElementaryKind{return (elementaryKinds as readonly string[]).includes(kind);}
const points=(count:number,fn:(i:number)=>Vec2)=>Array.from({length:count},(_,i)=>fn(i));
export function gcd(a:number,b:number){if(!Number.isSafeInteger(a)||!Number.isSafeInteger(b))throw new Error('Use safe integer fractions.');a=Math.abs(a);b=Math.abs(b);while(b){const next=a%b;a=b;b=next;}return a;}
function evaluateElementaryRaw(kind:ElementaryKind,p:number[]):FoundationResult{
 const counts:Record<ElementaryKind,number>={'numbers-ratios-units':4,'real-complex-numbers':3,'equations-inequalities':4,'polynomial-rational-functions':3,'exponentials-logarithms':3,'trigonometry-coordinates':2,'composition-inverse-functions':3,'sequences-series':3,'asymptotic-growth':4};
 if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Use finite parameters of the expected shape.');const [a,b,c,d]=p;
 if(kind==='numbers-ratios-units'){
  if(!Number.isSafeInteger(a)||!Number.isSafeInteger(b)||b<0||c<0||d<0)throw new Error('Invalid fraction or physical input.');const divisor=gcd(a,b);const ratio=b===0?null:a/b;
  return {statement:`a/b=(${a})/(${b}); L=${c} cm; t=${d} s`,stats:{'a/b (≈)':ratio,'reduced (numerator,denominator)':b===0?null:[a/divisor,b/divisor],'L [m]':c/100,'v=L/t [m/s]':d===0?null:c/100/d},curves:ratio===null?[]:[[[0,0],[ratio,0]],[[ratio,-.15],[ratio,.15]]],warning:b===0||d===0?'zero-divisor':undefined};
 }
 if(kind==='real-complex-numbers'){
  const theta=c*Math.PI/180;const rotated:Vec2=[a*Math.cos(theta)-b*Math.sin(theta),a*Math.sin(theta)+b*Math.cos(theta)];const magnitude=Math.hypot(a,b);
  return {statement:`z=(${a})+(${b})i; w=z·(cosθ+i sinθ), θ=${c}°`,stats:{'z (Re,Im)':[a,b],'w (Re,Im)':rotated,'|z|':magnitude,'|w|':Math.hypot(...rotated),'arg(z) [°]':magnitude===0?null:Math.atan2(b,a)*180/Math.PI,'arg(w) [°]':magnitude===0?null:Math.atan2(rotated[1],rotated[0])*180/Math.PI},curves:[[[0,0],[a,b]],[[0,0],rotated]],warning:magnitude===0?'undefined-angle':undefined};
 }
 if(kind==='equations-inequalities'){
  if(!Number.isInteger(a)||a<0||a>1)throw new Error('Use equation mode 0 or 1.');
  if(a===0){const roots=b===0?(c===0?null:[]):[c/b];return {statement:`(${b})x=(${c}); x₀=${d}`,stats:{'real solutions':roots,'real solution count':b===0?(c===0?Infinity:0):1,'a·x₀−b':b*d-c},curves:[points(81,i=>{const x=-4+i*.1;return [x,b*x];}),[[-4,c],[4,c]]],warning:b===0?'degenerate-equation':undefined};}
  const discriminant=b*b-4*c;const roots=discriminant<0?[]:discriminant===0?[-b/2]:[(-b-Math.sqrt(discriminant))/2,(-b+Math.sqrt(discriminant))/2];
  return {statement:`x²+(${b})x+(${c})≤0; x₀=${d}`,stats:{'D=a²−4b':discriminant,'real roots':roots,'solution interval':roots.length===0?[]:[roots[0],roots[roots.length-1]],'f(x₀)':d*d+b*d+c},curves:[points(81,i=>{const x=-4+i*.1;return [x,x*x+b*x+c];}),[[-4,0],[4,0]]]};
 }
 if(kind==='polynomial-rational-functions'){
  if(!Number.isInteger(a)||a<0||a>2)throw new Error('Use function mode 0–2.');
  if(a===2)return {statement:`f(x)=x²−(${b})`,stats:{'f(x₀)':c*c-b},curves:[points(101,i=>{const x=-3+i*.06;return [x,x*x-b];})],curveLabels:['f(x)']};
  const excluded=a===0?1:b,gap=a===0?.02:.08;const f=(x:number)=>a===0?x+1:1/(x-excluded);const left=points(81,i=>{const x=-3+(excluded-gap+3)*i/80;return [x,f(x)];}),right=points(81,i=>{const x=excluded+gap+(3-excluded-gap)*i/80;return [x,f(x)];});
  return {statement:a===0?'f(x)=(x²−1)/(x−1), x≠1':`f(x)=1/(x−(${b})), x≠${b}`,stats:{'f(x₀)':c===excluded?null:f(c),'excluded x':excluded,'sampling gap':gap,...(a===0?{'lim x→1 f(x)':2}:{})},curves:[left,right],curveLabels:[`x<${excluded}`,`x>${excluded}`],holes:a===0?[[1,2]]:undefined,warning:c===excluded?'function-domain':undefined};
 }
 if(kind==='exponentials-logarithms'){
  if(a<=0||c<0)throw new Error('Use positive base and nonnegative log probe.');const exponential=a**b;const logarithm=a===1||c===0?null:Math.log(c)/Math.log(a);
  return {statement:`f(x)=(${a})ˣ; log base=${a}; y₀=${c}`,stats:{'aˣ₀':exponential,'logₐ(y₀)':logarithm,'a^(logₐ y₀)':logarithm===null?null:a**logarithm,'d(aˣ)/dx at x₀':Math.log(a)*exponential},curves:[points(81,i=>{const x=-2+i*5/80;return [x,a**x];})],warning:logarithm===null?'function-domain':undefined};
 }
 if(kind==='trigonometry-coordinates'){
  if(b<0)throw new Error('Use nonnegative radius.');const radians=a*Math.PI/180;const cosine=Math.cos(radians),sine=Math.sin(radians);const point:Vec2=[b*cosine,b*sine];
  return {statement:`θ=${a}°; r=${b}; q=(1,0)`,stats:{'θ [rad]':radians,'cos θ':cosine,'sin θ':sine,'p=(r cosθ,r sinθ)':point,'‖p−q‖₂':Math.hypot(point[0]-1,point[1]),'arg(p) [°]':b===0?null:Math.atan2(point[1],point[0])*180/Math.PI},curves:[points(81,i=>[Math.cos(i/80*2*Math.PI),Math.sin(i/80*2*Math.PI)]),[[0,0],point]],warning:b===0?'undefined-angle':undefined};
 }
 if(kind==='composition-inverse-functions'){
  const u=a*c+b,target=u*u;const positiveRoot=Math.sqrt(target);const inverse=a===0?null:(positiveRoot-b)/a;const roots=a===0?null:(positiveRoot===0?[-b/a]:[(-positiveRoot-b)/a,(positiveRoot-b)/a].sort((x,y)=>x-y));
  return {statement:`g(x)=(${a})x+(${b}); f(u)=u²; h=f∘g`,stats:{'g(x₀)':u,'h(x₀)':target,'g(f(x₀))':a*c*c+b,'h⁻¹({h(x₀)})':roots,'preimage count':a===0?Infinity:roots!.length,'inverse with g(x)≥0':inverse,'1{g(x₀)≥0}':u>=0?1:0,'g⁻¹(g(x₀))':a===0?null:(u-b)/a},curves:[points(101,i=>{const x=-3+i*.06;return [x,(a*x+b)**2];})],warning:a===0?'zero-divisor':undefined};
 }
 if(kind==='sequences-series'){
  if(!Number.isInteger(c)||c<0||c>30)throw new Error('Use 0–30 terms.');let sum=0;const partials:Vec2[]=[[0,0]];for(let k=0;k<c;k++){sum+=b*a**k;partials.push([k+1,sum]);}const limit=b===0?0:Math.abs(a)<1?b/(1-a):null;
  return {statement:`uₙ=(${b})·(${a})ⁿ; Sₖ=Σₙ₌₀ᵏ⁻¹uₙ`,stats:{'N':c,'u_N':b*a**c,'S_N':sum,'lim S_k':limit,'1{u_n→0}':b===0||Math.abs(a)<1?1:0},curves:[partials,...(limit===null?[]:[[[0,limit],[Math.max(1,c),limit]] as Vec2[]])],warning:limit===null?'infinite-sum':undefined};
 }
 if([a,b,c,d].some(x=>!Number.isSafeInteger(x))||a<=0||b<0||c<=0||d<1)throw new Error('Use positive growth constants and integer threshold.');const required=c>a?Math.max(1,Math.ceil(b/(c-a))):c===a&&b===0?1:null;
 return {statement:`f(n)=(${a})n+(${b}); candidate f(n)≤(${c})n for n≥${d}`,stats:{'f(n₀)':a*d+b,'f(n₀)/n₀':a+b/d,'lim f(n)/n':a,'required n₀ for this C':required,'1{bound holds ∀n≥n₀}':required!==null&&d>=required?1:0},curves:[points(128,i=>[i+1,a+b/(i+1)]),[[1,c],[128,c]]],warning:required===null?'asymptotic-bound':undefined};
}

/** Distinguish representational overflow from mathematical division by zero. */
export function evaluateElementary(kind:ElementaryKind,p:number[]):FoundationResult {
 const result=evaluateElementaryRaw(kind,p);let overflow=false;const allowedInfinity=new Set(['real solution count','preimage count']);
 result.stats=Object.fromEntries(Object.entries(result.stats).map(([key,value])=>{if(value!==null&&!allowedInfinity.has(key)&&(Array.isArray(value)?value.some(x=>!Number.isFinite(x)):!Number.isFinite(value))){overflow=true;return [key,null];}return [key,value];}));
 if(overflow)result.warning='numeric-range';return result;
}
