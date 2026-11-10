import type {Vec2} from './learning-simulations';

// Exact rational structure tests; no floating tolerance decides geometric properties.
const B=BigInt,Z=B(0),ONE=B(1);
type Rat={n:bigint;d:bigint};
type Matrix=Rat[][];
function rat(n:bigint,d=ONE):Rat{
 if(d===Z)throw new Error('Zero rational denominator.');
 if(d<Z){n=-n;d=-d;}let a=n<Z?-n:n,b=d;while(b!==Z){const c=a%b;a=b;b=c;}return{n:n/a,d:d/a};
}
const R=(n:number,d=1)=>rat(B(n),B(d)),add=(a:Rat,b:Rat)=>rat(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:Rat)=>rat(-a.n,a.d),sub=(a:Rat,b:Rat)=>add(a,neg(b)),mul=(a:Rat,b:Rat)=>rat(a.n*b.n,a.d*b.d),div=(a:Rat,b:Rat)=>rat(a.n*b.d,a.d*b.n),sum=(a:Rat[])=>a.reduce(add,R(0));
const num=(a:Rat)=>Number(a.n)/Number(a.d);
const expose=(a:Rat)=>({numerator:a.n.toString(),denominator:a.d.toString(),label:a.d===ONE?a.n.toString():`${a.n}/${a.d}`,value:num(a)});
const matrix=(a:number[][])=>a.map(row=>row.map(v=>R(v)));
const identity=(n:number):Matrix=>Array.from({length:n},(_,i)=>Array.from({length:n},(__,j)=>R(Number(i===j))));
const transpose=(a:Matrix):Matrix=>a[0].map((_,j)=>a.map(row=>row[j]));
const mm=(a:Matrix,b:Matrix):Matrix=>a.map(row=>b[0].map((_,j)=>sum(row.map((v,k)=>mul(v,b[k][j])))));
const mv=(a:Matrix,v:Rat[])=>a.map(row=>sum(row.map((x,j)=>mul(x,v[j]))));
const msub=(a:Matrix,b:Matrix)=>a.map((row,i)=>row.map((x,j)=>sub(x,b[i][j])));
const exposedMatrix=(a:Matrix)=>a.map(row=>row.map(expose));
const numericMatrix=(a:Matrix)=>a.map(row=>row.map(num));
const zeroMatrix=(a:Matrix)=>a.every(row=>row.every(v=>v.n===Z));
function rank(a:Matrix){
 const c=a.map(row=>row.slice());let r=0;
 for(let j=0;j<c[0].length&&r<c.length;j++){
  const pivot=c.findIndex((row,i)=>i>=r&&row[j].n!==Z);if(pivot<0)continue;
  [c[r],c[pivot]]=[c[pivot],c[r]];const d=c[r][j];c[r]=c[r].map(v=>div(v,d));
  for(let i=0;i<c.length;i++)if(i!==r){const x=c[i][j];c[i]=c[i].map((v,k)=>sub(v,mul(x,c[r][k])));}r++;
 }
 return r;
}
function determinant(a:Matrix){
 const c=a.map(row=>row.slice());let d=R(1);
 for(let j=0;j<c.length;j++){
  const pivot=c.findIndex((row,i)=>i>=j&&row[j].n!==Z);if(pivot<0)return R(0);
  if(pivot!==j){[c[j],c[pivot]]=[c[pivot],c[j]];d=neg(d);}const x=c[j][j];d=mul(d,x);
  for(let i=j+1;i<c.length;i++){const f=div(c[i][j],x);c[i]=c[i].map((v,k)=>sub(v,mul(f,c[j][k])));}
 }
 return d;
}
function inverse(a:Matrix):Matrix{
 const n=a.length,c=a.map((row,i)=>[...row,...identity(n)[i]]);
 for(let j=0;j<n;j++){
  const pivot=c.findIndex((row,i)=>i>=j&&row[j].n!==Z);if(pivot<0)throw new Error('Singular matrix.');
  [c[j],c[pivot]]=[c[pivot],c[j]];const x=c[j][j];c[j]=c[j].map(v=>div(v,x));
  for(let i=0;i<n;i++)if(i!==j){const f=c[i][j];c[i]=c[i].map((v,k)=>sub(v,mul(f,c[j][k])));}
 }
 return c.map(row=>row.slice(n));
}
const J=matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]]);
const pairAreas=(u:Rat[],v:Rat[])=>[sub(mul(u[0],v[2]),mul(u[2],v[0])),sub(mul(u[1],v[3]),mul(u[3],v[1]))];
const pairing=(u:Rat[],v:Rat[])=>sum(pairAreas(u,v));
const pfaffian=(a:Matrix)=>add(sub(mul(a[0][1],a[2][3]),mul(a[0][2],a[1][3])),mul(a[0][3],a[1][2]));
function controls(p:number[],ranges:number[][]){
 if(p.length!==ranges.length)throw new Error('Use the complete symplectic-geometry control vector.');
 for(let j=0;j<p.length;j++)if(!Number.isInteger(p[j])||p[j]<ranges[j][0]||p[j]>ranges[j][1])throw new Error('Use bounded integer control codes.');
}
function boundary(base:Rat[],u:Rat[],v:Rat[]){
 const vertices=[base,base.map((x,j)=>add(x,u[j])),base.map((x,j)=>add(add(x,u[j]),v[j])),base.map((x,j)=>add(x,v[j])),base];
 return Array.from({length:81},(_,j)=>{const edge=Math.min(3,Math.floor(j/20)),s=R(j-20*edge,20);return vertices[edge].map((x,k)=>add(x,mul(s,sub(vertices[edge+1][k],x))));});
}
function actions(path:Rat[][],pairs:number){
 const out=[R(0)];for(let j=1;j<path.length;j++)out.push(add(out[j-1],sum(Array.from({length:pairs},(_,i)=>mul(div(add(path[j][pairs+i],path[j-1][pairs+i]),R(2)),sub(path[j][i],path[j-1][i]))))));return out;
}
function numericalAction(path:number[][],pairs:number){let value=0;for(let j=1;j<path.length;j++)for(let i=0;i<pairs;i++)value+=(path[j][pairs+i]+path[j-1][pairs+i])*(path[j][i]-path[j-1][i])/2;return value;}
const probes=[[[1,0,0,0],[0,0,1,0]],[[0,1,0,0],[0,0,0,1]],[[1,0,0,0],[0,1,0,0]],[[1,1,0,0],[0,0,1,-1]],[[1,0,0,1],[0,1,-1,0]],[[1,1,1,1],[2,2,2,2]]];
const bases=[[0,0,0,0],[1,1,1,0],[1,-1,0,1],[-1,.5,1,-.5]];
const halfVector=(v:number[])=>v.map(x=>R(x*2,2));
/** [kind0..3,a1..4,b1..4,2h=-4..4,2c=-4..4,probe0..5,base0..3,20boundary0..80].
 * Phase ordering(q1,q2,p1,p2);h,c are dimensionless multipliers of stated block units.
 * kind0:D,1:K D,2:T K D,3:T K Dbad. Boundary parameter is NOT physical time.
 */
export function fourDimensionalSymplecticMaps(p:number[]){
 controls(p,[[0,3],[1,4],[1,4],[-4,4],[-4,4],[0,5],[0,3],[0,80]]);
 const[kind,ac,bc,hc,cc,pc,baseCode,sc]=p,a=R(ac),b=R(bc),h=R(hc,2),c=R(cc,2),D=identity(4),K=identity(4),T=identity(4);
 D[0][0]=a;D[1][1]=b;D[2][2]=div(R(1),kind===3?b:a);D[3][3]=div(R(1),kind===3?a:b);
 const kick:Matrix=[[h,c],[c,neg(h)]],drift:Matrix=[[h,c],[c,R(0)]];
 if(kind!==0)for(let i=0;i<2;i++)for(let j=0;j<2;j++)K[2+i][j]=kick[i][j];
 if(kind>=2)for(let i=0;i<2;i++)for(let j=0;j<2;j++)T[i][2+j]=drift[i][j];
 const M=mm(T,mm(K,D)),inv=inverse(M),pullback=mm(transpose(M),mm(J,M)),pushforward=mm(M,mm(J,transpose(M))),formDefect=msub(pullback,J),poissonDefect=msub(pushforward,J),det=determinant(M);
 const [u,v]=probes[pc].map(halfVector),base=halfVector(bases[baseCode]),Mu=mv(M,u),Mv=mv(M,v),originalAreas=pairAreas(u,v),transformedAreas=pairAreas(Mu,Mv),area=sum(originalAreas),mappedArea=sum(transformedAreas),areaChange=sub(mappedArea,area);
 const path=boundary(base,u,v),mapped=path.map(z=>mv(M,z)),originalActions=actions(path,2),mappedActions=actions(mapped,2),numericPath=path.map(z=>z.map(num)),numericMapped=mapped.map(z=>z.map(num));
 const formUnits=Array.from({length:4},(_,i)=>Array.from({length:4},(__,j)=>i<2&&j<2?'kg/s':i>=2&&j>=2?'s/kg':'1')),poissonUnits=Array.from({length:4},(_,i)=>Array.from({length:4},(__,j)=>i<2&&j<2?'s/kg':i>=2&&j>=2?'kg/s':'1'));
 const query={boundaryParameter:sc/20,original:path[sc].map(expose),transformed:mapped[sc].map(expose),originalCumulativeAction:expose(originalActions[sc]),transformedCumulativeAction:expose(mappedActions[sc]),cumulativeActionChange:expose(sub(mappedActions[sc],originalActions[sc]))};
 return {kind,scaleA:ac,scaleB:bc,shearH:num(h),shearC:num(c),probeCode:pc,baseCode,phaseOrdering:['q1','q2','p1','p2'],matrix:exposedMatrix(M),numericMatrix:numericMatrix(M),mapCoefficientUnits:Array.from({length:4},(_,i)=>Array.from({length:4},(__,j)=>i<2&&j>=2?'s/kg':i>=2&&j<2?'kg/s':'1')),inverseMatrix:exposedMatrix(inv),mapRank:rank(M),determinant:expose(det),isVolumePreserving:det.n===det.d,isSymplectic:zeroMatrix(formDefect),preservesCanonicalBrackets:zeroMatrix(poissonDefect),pullbackForm:exposedMatrix(pullback),formDefect:exposedMatrix(formDefect),formCoefficientUnits:formUnits,pushforwardPoisson:exposedMatrix(pushforward),poissonDefect:exposedMatrix(poissonDefect),poissonCoefficientUnits:poissonUnits,inverseIdentityResidual:exposedMatrix(msub(mm(inv,M),identity(4))),inverseIsSymplectic:zeroMatrix(msub(mm(transpose(inv),mm(J,inv)),J)),pfaffianCanonical:expose(pfaffian(J)),pfaffianPullback:expose(pfaffian(pullback)),symplecticVolumeScale:expose(div(pfaffian(pullback),pfaffian(J))),probeRank:rank(transpose([u,v])),originalProbeIsLagrangian:rank(transpose([u,v]))===2&&area.n===Z,transformedProbeIsLagrangian:rank(transpose([Mu,Mv]))===2&&mappedArea.n===Z,probeDefectIdentityResidual:expose(sub(areaChange,sum(u.map((x,j)=>mul(x,mv(formDefect,v)[j]))))),probeU:u.map(expose),probeV:v.map(expose),transformedProbeU:Mu.map(expose),transformedProbeV:Mv.map(expose),originalPairAreas:originalAreas.map(expose),transformedPairAreas:transformedAreas.map(expose),originalSymplecticArea:expose(area),transformedSymplecticArea:expose(mappedArea),symplecticAreaChange:expose(areaChange),selectedProbePreservesForm:areaChange.n===Z,areaRatio:area.n===Z?null:expose(div(mappedArea,area)),originalClosedAction:expose(originalActions[80]),transformedClosedAction:expose(mappedActions[80]),originalActionAreaResidual:expose(add(originalActions[80],area)),transformedActionAreaResidual:expose(add(mappedActions[80],mappedArea)),originalFloatingActionResidual:numericalAction(numericPath,2)-num(originalActions[80]),transformedFloatingActionResidual:numericalAction(numericMapped,2)-num(mappedActions[80]),query,originalLoopProjections:[0,1].map(i=>numericPath.map(z=>[z[i],z[i+2]]as Vec2)),transformedLoopProjections:[0,1].map(i=>numericMapped.map(z=>[z[i],z[i+2]]as Vec2)),cumulativeActionCurves:[originalActions,mappedActions,mappedActions.map((x,j)=>sub(x,originalActions[j]))].map(curve=>curve.map((x,j)=>[j/20,num(x)]as Vec2)),activeKick:kind!==0,activeDrift:kind>=2};
}

/** [2a,2d,2b,2c,2q10,2q20,2width,2height,lineDirection0..3,20boundary0..80].
 * p=Aq,A=[[a,b+c],[b-c,d]]kg/s. This is a graph,not a time trajectory.
 * theta=p dotdq,omega=-dtheta;positive q1q2 rectangle is counterclockwise.
 */
export function lagrangianMomentumGraph(p:number[]){
 controls(p,[[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[-4,4],[1,4],[1,4],[0,3],[0,80]]);
 const[ac,dc,bc,cc,qc1,qc2,wc,hc,lineCode,sc]=p,a=R(ac,2),d=R(dc,2),b=R(bc,2),c=R(cc,2),width=R(wc,2),height=R(hc,2),A:Matrix=[[a,add(b,c)],[sub(b,c),d]],symmetric:Matrix=[[a,b],[b,d]],base=[R(qc1,2),R(qc2,2)],embedding:Matrix=[[R(1),R(0)],[R(0),R(1)],...A],pullback=mm(transpose(embedding),mm(J,embedding));
 const potential=(q:Rat[])=>div(sum(q.map((x,i)=>mul(x,mv(symmetric,q)[i]))),R(2));
 const state=(q:Rat[])=>{const momentum=mv(A,q),gradient=mv(symmetric,q);return{position:q.map(expose),momentum:momentum.map(expose),symmetricAction:expose(potential(q)),gradientSymmetricAction:gradient.map(expose),momentumMinusGradient:momentum.map((x,i)=>expose(sub(x,gradient[i])))};};
 const qPath=boundary(base,[width,R(0)],[R(0),height]),phasePath=qPath.map(q=>[...q,...mv(A,q)]),action=actions(phasePath,2),potentialChange=qPath.map(q=>sub(potential(q),potential(base))),loopArea=mul(width,height),flux=mul(sub(A[0][1],A[1][0]),loopArea),predictedAction=neg(flux);
 const detSym=determinant(symmetric),trace=add(a,d),sign=(x:Rat)=>x.n<Z?-1:x.n>Z?1:0;
 const signature=sign(detSym)<0?'indefinite':sign(detSym)>0?(sign(trace)>0?'positive-definite':'negative-definite'):sign(trace)>0?'positive-semidefinite':sign(trace)<0?'negative-semidefinite':'zero';
 const eigGap=Math.sqrt(num(add(mul(sub(a,d),sub(a,d)),mul(R(4),mul(b,b))))),eigenvalues=[(num(trace)-eigGap)/2,(num(trace)+eigGap)/2];
 const direction=matrix([[1,0],[0,1],[1,1],[1,-1]])[lineCode],tangent=mv(embedding,direction),linePullback=pairing(tangent,tangent),linePath=Array.from({length:41},(_,j)=>{const q=base.map((x,i)=>add(x,mul(R(j-20,20),direction[i])));return[...q,...mv(A,q)].map(num);});
 const isLagrangian=cc===0,query={boundaryParameter:sc/20,...state(qPath[sc]),cumulativeAction:expose(action[sc]),potentialChange:expose(potentialChange[sc]),actionMinusPotential:expose(sub(action[sc],potentialChange[sc]))};
 return {coefficientA:num(a),coefficientD:num(d),coefficientB:num(b),coefficientC:num(c),matrix:exposedMatrix(A),numericMatrix:numericMatrix(A),symmetricPart:exposedMatrix(symmetric),antisymmetricPart:exposedMatrix(msub(A,symmetric)),momentumMapRank:rank(A),momentumMapDeterminant:expose(determinant(A)),graphEmbedding:exposedMatrix(embedding),graphDimension:rank(embedding),ambientDimension:4,ambientFormRank:rank(J),ambientFormNondegenerate:rank(J)===4,pullbackForm:exposedMatrix(pullback),pullbackRank:rank(pullback),isIsotropicGraph:zeroMatrix(pullback),isLagrangianGraph:isLagrangian,hasGeneratingAction:isLagrangian,symmetricActionSignature:signature,symmetricActionIsConvex:signature==='positive-definite'||signature==='positive-semidefinite'||signature==='zero',symmetricActionIsStrictlyConvex:signature==='positive-definite',symmetricActionEigenvalues:eigenvalues,symmetricActionDeterminant:expose(detSym),curlCoefficient:expose(sub(A[1][0],A[0][1])),rectangleArea:expose(loopArea),pullbackFlux:expose(flux),closedLoopAction:expose(action[80]),predictedClosedLoopAction:expose(predictedAction),stokesResidual:expose(add(action[80],flux)),closedActionFormulaResidual:expose(sub(action[80],predictedAction)),floatingClosedActionResidual:numericalAction(phasePath.map(q=>q.map(num)),2)-num(action[80]),width:num(width),height:num(height),base:state(base),query,lineDirection:direction.map(expose),lineTangent:tangent.map(expose),lineDimension:rank(transpose([tangent])),lineIsIsotropic:linePullback.n===Z,lineIsLagrangian:false,linePullback:expose(linePullback),linePhaseProjections:[0,1].map(i=>linePath.map(z=>[z[i],z[i+2]]as Vec2)),configurationBoundary:qPath.map(q=>q.map(num)as Vec2),phaseProjections:[0,1].map(i=>phasePath.map(q=>[num(q[i]),num(q[i+2])]as Vec2)),cumulativeActionCurves:[action,potentialChange,action.map((x,j)=>sub(x,potentialChange[j]))].map(curve=>curve.map((x,j)=>[j/20,num(x)]as Vec2)),momentumCurves:[0,1].map(i=>phasePath.map((q,j)=>[j/20,num(q[i+2])]as Vec2)),gradientResidualCurves:[0,1].map(i=>qPath.map((q,j)=>[j/20,num(sub(mv(A,q)[i],mv(symmetric,q)[i]))]as Vec2))};
}
