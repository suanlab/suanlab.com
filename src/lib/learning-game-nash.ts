export type NashFraction=[number,number];
export type NashRectangle=[NashFraction,NashFraction,NashFraction,NashFraction];
const gcd=(a:number,b:number):number=>{a=Math.abs(a);b=Math.abs(b);while(b){const r=a%b;a=b;b=r;}return a;};
const fraction=(n:number,d=1):NashFraction=>{if(d<0){n=-n;d=-d;}const k=gcd(n,d);return [n===0?0:n/k,d/k];};
const compare=(a:NashFraction,b:NashFraction)=>a[0]*b[1]-b[0]*a[1],zero=fraction(0),one=fraction(1),midpoint=(a:NashFraction,b:NashFraction)=>fraction(a[0]*b[1]+b[0]*a[1],2*a[1]*b[1]);
export const nashNumber=(q:NashFraction)=>q[0]/q[1];
export const nashFractionLabel=(q:NashFraction)=>q[1]===1?String(q[0]):`${q[0]}/${q[1]}`;
function validateMatrix(a:readonly number[]){if(a.length!==4||a.some(v=>!Number.isInteger(v)||Math.abs(v)>9))throw new Error('Use four integer payoffs from-9 through9.');}
function atoms(offset:number,slope:number){const cuts=[zero,one];if(slope){const root=fraction(-offset,slope);if(compare(root,zero)>0&&compare(root,one)<0)cuts.push(root);}cuts.sort(compare);return [...cuts.map(v=>({lo:v,hi:v,sample:v})),...cuts.slice(0,-1).map((lo,i)=>({lo,hi:cuts[i+1],sample:midpoint(lo,cuts[i+1])}))];}
function response(sign:number,mix:NashFraction){return sign===0||(sign>0?compare(mix,one)===0:compare(mix,zero)===0);}
export function nashEquilibriumSet(A:readonly number[],B:readonly number[]):NashRectangle[]{
 validateMatrix(A);validateMatrix(B);const a0=A[1]-A[3],as=A[0]-A[2]-a0,b0=B[2]-B[3],bs=B[0]-B[1]-b0,ps=atoms(b0,bs),qs=atoms(a0,as),raw:NashRectangle[]=[];
 for(const p of ps)for(const q of qs)if(response(a0*q.sample[1]+as*q.sample[0],p.sample)&&response(b0*p.sample[1]+bs*p.sample[0],q.sample))raw.push([p.lo,p.hi,q.lo,q.hi]);
 const unique=[...new Map(raw.map(r=>[r.map(nashFractionLabel).join(','),r])).values()],contains=(a:NashRectangle,b:NashRectangle)=>compare(a[0],b[0])<=0&&compare(a[1],b[1])>=0&&compare(a[2],b[2])<=0&&compare(a[3],b[3])>=0;
 return unique.filter(r=>!unique.some(s=>s!==r&&contains(s,r))).sort((a,b)=>{for(let i=0;i<4;i++){const v=compare(a[i],b[i]);if(v)return v;}return 0;}).map(r=>r.map(v=>[...v]) as NashRectangle);
}
export const nashGames=[
 {name:'coordination',row:[2,0,0,1],column:[2,0,0,1]},
 {name:'opposed payoffs',row:[1,-1,-1,1],column:[-1,1,1,-1]},
 {name:'social dilemma',row:[3,0,5,1],column:[3,5,0,1]},
 {name:'P1 indifferent',row:[0,0,0,0],column:[2,0,0,1]},
 {name:'both indifferent',row:[0,0,0,0],column:[0,0,0,0]},
] as const;
export function nashLab(family:number,pNum:number,qNum:number,player:number,proposedNum:number){
 for(const [x,lo,hi] of [[family,0,4],[pNum,0,12],[qNum,0,12],[player,0,1],[proposedNum,0,12]])if(!Number.isInteger(x)||x<lo||x>hi)throw new Error('Use the stated game, probabilities in twelfths, player0/1 and proposed probability in twelfths.');
 const g=nashGames[family],D=144,mass=[pNum*qNum,pNum*(12-qNum),(12-pNum)*qNum,(12-pNum)*(12-qNum)],utilityNumerators=[mass.reduce((s,v,i)=>s+v*g.row[i],0),mass.reduce((s,v,i)=>s+v*g.column[i],0)],pureNumerators=[[qNum*g.row[0]+(12-qNum)*g.row[1],qNum*g.row[2]+(12-qNum)*g.row[3]],[pNum*g.column[0]+(12-pNum)*g.column[2],pNum*g.column[1]+(12-pNum)*g.column[3]]],bestMasks=pureNumerators.map(v=>Number(v[0]>=v[1])+2*Number(v[1]>=v[0])),gapNumerators=pureNumerators.map((v,i)=>12*Math.max(...v)-utilityNumerators[i]),proposedUtilityNumerator=proposedNum*pureNumerators[player][0]+(12-proposedNum)*pureNumerators[player][1],proposedChangeNumerator=proposedUtilityNumerator-utilityNumerators[player],set=nashEquilibriumSet(g.row,g.column),pointCount=set.filter(r=>compare(r[0],r[1])===0&&compare(r[2],r[3])===0).length;
 return {family,pNum,qNum,player,proposedNum,g,D,mass,utilityNumerators,pureNumerators,bestMasks,gapNumerators,proposedUtilityNumerator,proposedChangeNumerator,set,responses:[nashEquilibriumSet(g.row,[0,0,0,0]),nashEquilibriumSet([0,0,0,0],g.column)],pointCount,hasContinuum:set.some(r=>compare(r[0],r[1])!==0||compare(r[2],r[3])!==0),isNash:gapNumerators.every(n=>n===0),queryInSet:set.some(r=>compare(r[0],[pNum,12])<=0&&compare(r[1],[pNum,12])>=0&&compare(r[2],[qNum,12])<=0&&compare(r[3],[qNum,12])>=0)};
}
