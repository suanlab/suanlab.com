/** Smooth fixed-time, free-endpoint scalar control problem. No half factors in cost. */
const range=(v:number,lo:number,hi:number)=>{if(!Number.isFinite(v)||v<lo||v>hi)throw new Error('Use the stated continuous HJB/PMP parameter domain.');};
export function hjbPoint(q:number,r:number,p:number,T:number,x0:number,t:number){
 const k=Math.sqrt(q/r),FG=(tau:number)=>{const z=k*tau,S=tau*(z===0?1:Math.sinh(z)/z),C=Math.cosh(z);return {F:C+(p/r)*S,G:q*S+p*C};},initial=FG(T),{F,G}=FG(T-t),P=G/F,Pt=(G*(G/r)-q*F*F)/(F*F),x=x0*F/initial.F,u=-x0*G/(r*initial.F),lambda=2*x0*G/initial.F,V=P*x*x,Vx=2*P*x,Vt=Pt*x*x;
 return {t,P,Pt,x,u,lambda,V,Vx,Vt,hjbResidual:Vt+q*x*x+r*u*u+Vx*u,stationarity:2*r*u+lambda};
}
export function continuousHjb(q:number,r:number,p:number,T:number,x0:number,fraction:number){
 range(q,0,2);range(r,.2,2);range(p,0,3);range(T,.25,3);range(x0,-1,1);range(fraction,0,1);
 const rows=Array.from({length:101},(_,i)=>hjbPoint(q,r,p,T,x0,T*i/100)),selected=hjbPoint(q,r,p,T,x0,T*fraction),first=rows[0],last=rows[100],value=first.P*x0*x0;
 const N=400,integrand=(t:number)=>{const z=hjbPoint(q,r,p,T,x0,t);return q*z.x*z.x+r*z.u*z.u;};let quadrature=integrand(0)+integrand(T);for(let i=1;i<N;i++)quadrature+=(i%2?4:2)*integrand(T*i/N);quadrature=quadrature*T/(3*N)+p*last.x*last.x;
 const bowl=Array.from({length:81},(_,i)=>{const x=-2+4*i/80;return [x,selected.P*x*x] as [number,number];}),tangent=[-2,2].map(x=>[x,selected.V+selected.Vx*(x-selected.x)] as [number,number]);
 return {rows,selected,first,last,value,quadrature,bowl,tangent,maxHjbResidual:Math.max(...rows.map(v=>Math.abs(v.hjbResidual))),maxGradientMismatch:Math.max(...rows.map(v=>Math.abs(v.lambda-v.Vx))),terminalResidual:last.lambda-2*p*last.x};
}
