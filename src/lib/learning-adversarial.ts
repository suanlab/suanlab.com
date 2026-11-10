/** Affine classifiers on an unconstrained norm ball. Integer comparisons certify ties; plotted L2 coordinates are approximate. */
function integer(v:number,lo:number,hi:number){if(!Number.isInteger(v)||v<lo||v>hi)throw new Error('Use the stated integer adversarial controls.');}
export function affineLab(w0:number,w1:number,x0q:number,x1q:number,bq:number,epq:number,norm:number,y:number){
 integer(w0,-4,4);integer(w1,-4,4);integer(x0q,-8,8);integer(x1q,-8,8);integer(bq,-16,16);integer(epq,0,8);integer(norm,0,2);if(y!==-1&&y!==1)throw new Error('Ground-truth label must be-1 or+1.');
 const M=y*(w0*x0q+w1*x1q+bq),n2=w0*w0+w1*w1,dual=norm===0?Math.max(Math.abs(w0),Math.abs(w1)):norm===1?Math.sqrt(n2):Math.abs(w0)+Math.abs(w1),compare=norm===1?(M<0?-1:Math.sign(M*M-epq*epq*n2)):Math.sign(M-epq*dual),eps=epq/4;
 let delta:[number,number]=[0,0];if(dual>0){if(norm===0){const j=Math.abs(w0)>=Math.abs(w1)?0:1;delta[j]=-eps*y*Math.sign([w0,w1][j]);}else if(norm===1)delta=[-eps*y*w0/dual,-eps*y*w1/dual];else delta=[-eps*y*Math.sign(w0),-eps*y*Math.sign(w1)];}
 const cleanCorrect=y===1?M>=0:M>0,predictionRobust=y===1?compare>=0:compare>0;
 return{w:[w0,w1],x:[x0q/4,x1q/4],b:bq/4,y,norm,eps,signedMargin:M/4,dual,worstSignedMargin:(M-epq*dual)/4,exactWorstSign:compare,algebraicWorstPrediction:compare===0?1:compare>0?y:-y,delta,strictMarginCertificate:compare>0,predictionRobust,atWorstTie:compare===0,cleanCorrect,robustRadius:cleanCorrect&&dual>0?M/(4*dual):null,radiusInfinite:cleanCorrect&&dual===0,radiusUndefined:!cleanCorrect,robustRadiusBoundaryIncluded:cleanCorrect&&dual>0&&y===1};
}

/** Three fixed one-hidden-layer ReLU examples. All supported inputs and arrangement intersections have exact binary coordinates. */
export type V=[number,number];
const networks=[{bias:1,nodes:[{w:[1,0] as V,h:.25,a:-3},{w:[1,0] as V,h:.75,a:3}]},{bias:1,nodes:[{w:[1,0] as V,h:0,a:1},{w:[1,0] as V,h:0,a:-1}]},{bias:.25,nodes:[{w:[1,1] as V,h:0,a:1},{w:[1,-1] as V,h:.25,a:-2},{w:[-1,-1] as V,h:.5,a:1}]}];
const dot=(a:V,b:V)=>a[0]*b[0]+a[1]*b[1];
export function reluLab(mode:number,center:V,eps:number,steps:number,start:number,stepSize:number){
 integer(mode,0,2);if(center.length!==2||center.some(x=>!Number.isInteger(x*4)||x< -2||x>2))throw new Error('Use two quarter-grid center coordinates in[-2,2].');
 if(!Number.isInteger(eps*4)||eps<0||eps>2||!Number.isInteger(stepSize*4)||stepSize<0||stepSize>2)throw new Error('Use quarter-grid radius and step size in[0,2].');integer(steps,0,20);integer(start,0,4);
 const net={bias:networks[mode].bias,nodes:networks[mode].nodes.map(n=>({...n,w:n.w.slice() as V}))},value=(x:V)=>net.bias+net.nodes.reduce((s,n)=>s+n.a*Math.max(0,dot(n.w,x)-n.h),0),lo:V=[center[0]-eps,center[1]-eps],hi:V=[center[0]+eps,center[1]+eps],corners:V[]=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],lines=[...net.nodes.map(n=>({w:n.w,h:n.h})),{w:[1,0] as V,h:lo[0]},{w:[1,0] as V,h:hi[0]},{w:[0,1] as V,h:lo[1]},{w:[0,1] as V,h:hi[1]}],vertices:V[]=[];
 const add=(p:V)=>{if(p.every((v,i)=>v>=lo[i]&&v<=hi[i])&&!vertices.some(q=>q[0]===p[0]&&q[1]===p[1]))vertices.push(p);};corners.forEach(add);
 for(let i=0;i<lines.length;i++)for(let j=i+1;j<lines.length;j++){const a=lines[i],b=lines[j],det=a.w[0]*b.w[1]-a.w[1]*b.w[0];if(det)add([(a.h*b.w[1]-a.w[1]*b.h)/det,(a.w[0]*b.h-a.h*b.w[0])/det]);}
 const exactMinimum=Math.min(...vertices.map(value)),exactMaximum=Math.max(...vertices.map(value));let intervalLower=net.bias,intervalUpper=net.bias;
 for(const n of net.nodes){const c=dot(n.w,center)-n.h,r=eps*(Math.abs(n.w[0])+Math.abs(n.w[1])),a=Math.max(0,c-r),b=Math.max(0,c+r);intervalLower+=n.a*(n.a>=0?a:b);intervalUpper+=n.a*(n.a>=0?b:a);}
 let point:V=start===0?center.slice() as V:corners[start-1].slice() as V;const trajectory=[point.slice() as V];
 for(let k=0;k<steps;k++){const gradient:V=[0,0];for(const n of net.nodes)if(dot(n.w,point)>n.h){gradient[0]+=n.a*n.w[0];gradient[1]+=n.a*n.w[1];}point=point.map((v,i)=>Math.max(lo[i],Math.min(hi[i],v-stepSize*Math.sign(gradient[i])))) as V;trajectory.push(point.slice() as V);}
 const attackMargins=trajectory.map(value),attackBest=Math.min(...attackMargins);
 return{mode,center:center.slice() as V,eps,steps,start,stepSize,net,vertices,vertexMargins:vertices.map(value),exactMinimum,exactMaximum,intervalLower,intervalUpper,trajectory,attackMargins,attackBest,strictCertifiedByInterval:intervalLower>0,exactStrictMargin:exactMinimum>0,exactPredictionRobust:exactMinimum>=0,attackFoundMisclassification:attackBest<0};
}
