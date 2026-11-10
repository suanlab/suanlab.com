/** Independent checker arithmetic. No producer modules or constants imported.
 * Bounds are integers in units of 2^-bits. Transcendental remainders are
 * included analytically; numerical fixtures are diagnostic only.
 */
export interface CheckerInterval {lower:bigint;upper:bigint;}
type Fraction={n:bigint;d:bigint};
const Z=BigInt(0),O=BigInt(1),T=BigInt(2);
export function checkerFloor(n:bigint,d:bigint):bigint {
  if(d===Z)throw new RangeError('Checker division by zero');
  if(d<Z){n=-n;d=-d;}const q=n/d;return n<Z&&n%d!==Z?q-O:q;
}
const ceil=(n:bigint,d:bigint)=>-checkerFloor(-n,d);
const abs=(n:bigint)=>n<Z?-n:n;
function integerRoot(n:bigint,degree:number):bigint {
  if(n<Z)throw new RangeError('Negative checker radicand');if(n<T)return n;
  const k=BigInt(degree);let x=O<<BigInt(Math.ceil(n.toString(2).length/degree));
  for(;;){const next=((k-O)*x+n/x**(k-O))/k;if(next>=x)break;x=next;}
  if(x**k>n||(x+O)**k<=n)throw Error('Checker integer root certificate');return x;
}
export function createRealTimeCheckerInterval(bits=1536) {
  if(!Number.isSafeInteger(bits)||bits<256||bits>32768)throw new RangeError('Checker precision256..32768');
  const S=O<<BigInt(bits),point=(n:bigint):CheckerInterval=>({lower:n,upper:n});
  const integer=(n:number|bigint)=>point(BigInt(n)*S);
  const from=(q:Fraction):CheckerInterval=>{
    if(q.d<=Z)throw new RangeError('Positive checker denominator');
    return{lower:checkerFloor(q.n*S,q.d),upper:ceil(q.n*S,q.d)};
  };
  const neg=(x:CheckerInterval)=>({lower:-x.upper,upper:-x.lower});
  const add=(x:CheckerInterval,y:CheckerInterval)=>({lower:x.lower+y.lower,upper:x.upper+y.upper});
  const sub=(x:CheckerInterval,y:CheckerInterval)=>add(x,neg(y));
  const widen=(x:CheckerInterval,n:bigint)=>({lower:x.lower-n,upper:x.upper+n});
  const multiply=(x:CheckerInterval,y:CheckerInterval):CheckerInterval=>{
    const p=[x.lower*y.lower,x.lower*y.upper,x.upper*y.lower,x.upper*y.upper];
    return{lower:checkerFloor(p.reduce((a,b)=>a<b?a:b),S),upper:ceil(p.reduce((a,b)=>a>b?a:b),S)};
  };
  const scale=(x:CheckerInterval,n:number|bigint):CheckerInterval=>{
    const k=BigInt(n);return k>=Z?{lower:x.lower*k,upper:x.upper*k}:{lower:x.upper*k,upper:x.lower*k};
  };
  const divideInteger=(x:CheckerInterval,n:number|bigint):CheckerInterval=>{
    const d=BigInt(n);if(d<=Z)throw new RangeError('Positive checker divisor');
    return{lower:checkerFloor(x.lower,d),upper:ceil(x.upper,d)};
  };
  const divide=(x:CheckerInterval,y:CheckerInterval):CheckerInterval=>{
    if(y.lower<=Z&&y.upper>=Z)throw new RangeError('Checker interval crosses zero');
    const pairs=[[x.lower,y.lower],[x.lower,y.upper],[x.upper,y.lower],[x.upper,y.upper]];
    return{lower:pairs.map(([n,d])=>checkerFloor(n*S,d)).reduce((a,b)=>a<b?a:b),
      upper:pairs.map(([n,d])=>ceil(n*S,d)).reduce((a,b)=>a>b?a:b)};
  };
  const square=(x:CheckerInterval):CheckerInterval=>{
    const a=x.lower*x.lower,b=x.upper*x.upper;
    return{lower:x.lower<=Z&&x.upper>=Z?Z:checkerFloor(a<b?a:b,S),upper:ceil(a>b?a:b,S)};
  };
  const root=(x:CheckerInterval,degree:2|3|4):CheckerInterval=>{
    if(x.lower<Z)throw new RangeError('Nonnegative checker root interval');
    const factor=S**BigInt(degree-1),lo=integerRoot(x.lower*factor,degree),hi=integerRoot(x.upper*factor,degree);
    return{lower:lo,upper:hi**BigInt(degree)===x.upper*factor?hi:hi+O};
  };
  /** Positive atanh series. z<=1/3 gives remaining tail <=
   * 3*z^(2j+1)/(2j+1), including the factor2 in log((1+z)/(1-z)). */
  function logSeries(z:CheckerInterval):CheckerInterval {
    if(z.lower<Z||z.upper*T>=S)throw new RangeError('Checker logarithm reduction');
    if(z.upper===Z)return point(Z);
    const z2=square(z);let power=z,total=point(Z);
    for(let j=0;j<bits+64;j++) {
      total=add(total,scale(divideInteger(power,2*j+1),2));
      power=multiply(power,z2);
      if(power.upper<=BigInt(4))return{lower:total.lower,upper:total.upper+ceil(BigInt(3)*power.upper,BigInt(2*j+3))};
    }
    throw Error('Checker logarithm series budget');
  }
  let ln2:CheckerInterval|undefined;
  const logTwo=()=>ln2??=logSeries(from({n:O,d:BigInt(3)}));
  function logPoint(n:bigint):CheckerInterval {
    if(n<=Z)throw new RangeError('Positive checker logarithm');
    const exponent=n.toString(2).length-1-bits;
    const reduced=exponent>=0?{n,d:S<<BigInt(exponent)}:{n:n<<BigInt(-exponent),d:S};
    const z=from({n:reduced.n-reduced.d,d:reduced.n+reduced.d});
    return add(logSeries(z),scale(logTwo(),exponent));
  }
  const log=(x:CheckerInterval):CheckerInterval=>({lower:logPoint(x.lower).lower,upper:logPoint(x.upper).upper});
  /** Taylor for exp(-r),0<=r<=1/8, then repeated squaring.
   * If x>=bits, e^-x <2^-bits, so [0,1] is a certified enclosure. */
  function expPoint(n:bigint):CheckerInterval {
    if(n<Z)throw new RangeError('Nonnegative checker decay');if(n===Z)return point(S);
    if(n>=BigInt(bits)*S)return{lower:Z,upper:O};
    const k=Math.max(0,n.toString(2).length-bits+4),r=from({n,d:S<<BigInt(k)});
    let term=point(S),total=point(S),result:CheckerInterval|undefined;
    for(let j=1;j<bits+64;j++) {
      term=neg(divideInteger(multiply(term,r),j));total=add(total,term);
      const next=divideInteger(multiply({lower:Z,upper:abs(term.lower)>abs(term.upper)?abs(term.lower):abs(term.upper)},r),j+1);
      if(next.upper<=BigInt(4)){result=widen(total,next.upper);break;}
    }
    if(!result)throw Error('Checker exponential series budget');
    result={lower:result.lower<Z?Z:result.lower,upper:result.upper>S?S:result.upper};
    for(let j=0;j<k;j++)result=square(result);
    return{lower:result.lower<Z?Z:result.lower,upper:result.upper>S?S:result.upper};
  }
  const expNegative=(x:CheckerInterval):CheckerInterval=>({lower:expPoint(x.upper).lower,upper:expPoint(x.lower).upper});
  /** Small-angle Taylor with first omitted term; no pi or modulo reduction. */
  function sinCosSmall(x:CheckerInterval) {
    if(x.lower<-(S/BigInt(8))||x.upper>S/BigInt(8))throw new RangeError('Checker small angle');
    if(x.lower===Z&&x.upper===Z)return{sin:point(Z),cos:point(S)};
    const x2=square(x);let st=x,ct=point(S),sin=x,cos=ct;
    for(let j=1;j<bits+64;j++) {
      st=neg(divideInteger(multiply(st,x2),BigInt(2*j)*BigInt(2*j+1)));
      ct=neg(divideInteger(multiply(ct,x2),BigInt(2*j-1)*BigInt(2*j)));
      const tail=([st.lower,st.upper,ct.lower,ct.upper].map(abs)).reduce((a,b)=>a>b?a:b);
      if(tail<=BigInt(4))return{sin:widen(sin,tail),cos:widen(cos,tail)};
      sin=add(sin,st);cos=add(cos,ct);
    }
    throw Error('Checker trigonometry series budget');
  }
  let piValue:CheckerInterval|undefined;
  function pi() {
    if(piValue)return piValue;
    function atan(q:bigint) {
      let denominator=q,total=point(Z);
      for(let j=0;j<bits+64;j++) {
        const term=from({n:O,d:denominator*BigInt(2*j+1)});
        if(term.upper<=O)return j%2?{lower:total.lower-term.upper,upper:total.upper}:{lower:total.lower,upper:total.upper+term.upper};
        total=j%2?sub(total,term):add(total,term);denominator*=q*q;
      }
      throw Error('Checker pi series budget');
    }
    // tan(atan(1/3)+atan(1/7))=1/2; adding atan(1/3)
    // gives tan=1, with sum in the first quadrant. Hence pi=8atan(1/3)+4atan(1/7).
    piValue=add(scale(atan(BigInt(3)),8),scale(atan(BigInt(7)),4));return piValue;
  }
  return{bits,S,point,zero:point(Z),one:point(S),from,integer,neg,add,sub,multiply,divide,divideInteger,scale,square,root,log,expNegative,sinCosSmall,pi};
}

/** Independent large-phase evaluation by halving and angle doubling.
 * The supported magnitude covers the physical contract's |S/h|<2^4568.
 * Extra2k guard bits cover dependency growth in k doubling operations.
 */
export function checkerRealTimeRotation(phase:Fraction,outputBits=1024) {
  if(typeof phase?.n!=='bigint'||typeof phase.d!=='bigint'||phase.d<=Z||
    abs(phase.n).toString(2).length>65536||phase.d.toString(2).length>65536||
    !Number.isSafeInteger(outputBits)||outputBits<64||outputBits>2048)throw new RangeError('Checker phase bounds');
  const magnitude=Math.max(0,abs(phase.n).toString(2).length-phase.d.toString(2).length+1);
  if(magnitude>8192)throw new RangeError('Checker physical phase magnitude');
  const halvings=magnitude+5,bits=outputBits+2*halvings+128,ctx=createRealTimeCheckerInterval(bits);
  let {sin,cos}=ctx.sinCosSmall(ctx.from({n:phase.n,d:phase.d<<BigInt(halvings)}));
  const clamp=(x:CheckerInterval)=>({lower:x.lower< -ctx.S?-ctx.S:x.lower,upper:x.upper>ctx.S?ctx.S:x.upper});
  for(let k=0;k<halvings;k++) {
    const nextSin=clamp(ctx.scale(ctx.multiply(sin,cos),2));
    cos=clamp(ctx.sub(ctx.one,ctx.scale(ctx.square(sin),2)));sin=nextSin;
  }
  const divisor=O<<BigInt(bits-outputBits);
  const out=(x:CheckerInterval)=>({lower:checkerFloor(x.lower,divisor),upper:ceil(x.upper,divisor)});
  return{sin:out(sin),cos:out(cos),bits:outputBits,workBits:bits,halvings};
}
