/** One-step min-max control with observed state, then input, then bounded disturbance. */
export function robustSafeControl(a:number,x:number,d:number,U:number,L:number,q:number){
 const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated one-step robust-control domain.');};range(a,-2,2);range(x,-2,2);[d,U,L].forEach(v=>range(v,0,2));range(q,0,4);
 const center=a*x,worstCost=(u:number)=>u*u+q*(Math.abs(center+u)+d)**2,nominalCost=(u:number)=>u*u+q*(center+u)**2;
 const solve=(radius:number)=>{const lo=Math.max(-U,-L+radius-center),hi=Math.min(U,L-radius-center),feasible=radius<=L&&lo<=hi,z=Math.sign(center)*Math.max((Math.abs(center)-q*radius)/(1+q),0),expandedUnconstrained=z-center,unconstrained=q===0||center===0?0:Math.abs(center)<=q*radius?-center:-q*(center+Math.sign(center)*radius)/(1+q),u=feasible?Math.min(hi,Math.max(lo,unconstrained)):null;
  if(u===null)return {lo,hi,feasible,u,unconstrained,expandedUnconstrained,cost:null,worstCost:null,successors:null,safetyMargin:null,optimalityResidual:null};
  const z0=center+u,derivativeLo=2*u+2*q*(z0+(z0>0?radius:-radius)),derivativeHi=2*u+2*q*(z0+(z0>=0?radius:-radius)),residual=lo===hi?0:u===lo?Math.max(0,-derivativeHi):u===hi?Math.max(0,derivativeLo):Math.max(0,derivativeLo,-derivativeHi);
  return {lo,hi,feasible,u,unconstrained,expandedUnconstrained,cost:u*u+q*(Math.abs(z0)+radius)**2,worstCost:worstCost(u),successors:[z0-d,z0+d] as [number,number],safetyMargin:L-(Math.abs(z0)+d),optimalityResidual:residual};};
 const robust=solve(d),nominal=solve(0),curves=Array.from({length:121},(_,i)=>{const u=U===0?0:-U+2*U*i/120;return {u,worst:worstCost(u),nominal:nominalCost(u),robustSafe:d<=L&&u>=robust.lo&&u<=robust.hi};});
 return {center,d,U,L,q,robust,nominal,curves,worstCost,nominalCost};
}
