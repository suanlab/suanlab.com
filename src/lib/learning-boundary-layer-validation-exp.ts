/** Independent exp(-r): alternating small-negative Taylor series, followed by
 * repeated squaring. The producer instead inverts a positive Taylor series.
 */
import {Rational as Q} from './learning-graph-gauge-validation-math';
const B=BigInt,targetBits=192,target=B(1)<<B(targetBits);
type I=readonly[bigint,bigint];
const ceil=(n:bigint,d:bigint)=>(n+d-B(1))/d;
export function boundaryLayerIndependentDecay(n:bigint,d:bigint):readonly[Q,Q]{
 if(n===B(0))return[new Q(B(1)),new Q(B(1))];
 if(n<B(0)||d<=B(0))throw Error('Positive rational magnitude required');
 let squarings=0;while(B(8)*n>(d<<B(squarings)))squarings++;
 for(let bits=256;bits<=2048;bits+=64){
  const scale=B(1)<<B(bits),den=d<<B(squarings),x:I=[n*scale/den,ceil(n*scale,den)];
  const times=(a:I,b:I):I=>[a[0]*b[0]/scale,ceil(a[1]*b[1],scale)];
  const even=2*Math.ceil((bits+12)/6);let term:I=[scale,scale],sum:I=term,upper=scale;
  for(let j=1;j<=even+1;j++){const p=times(term,x);term=[p[0]/B(j),ceil(p[1],B(j))];sum=j%2?[sum[0]-term[1],sum[1]-term[0]]:[sum[0]+term[0],sum[1]+term[1]];if(j===even)upper=sum[1];}
  let r:I=[sum[0],upper];if(r[0]<B(0))throw Error('Invalid alternating remainder');
  for(let j=0;j<squarings;j++)r=times(r,r);
  const lo=r[0]>>B(bits-targetBits),hi=r[1]>>B(bits-targetBits);
  if(lo===hi)return[new Q(lo,target),new Q(lo+B(1),target)];
 }
 throw Error('Independent decay cell unresolved');
}
