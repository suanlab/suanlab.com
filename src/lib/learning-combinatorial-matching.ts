export type MatchingPair=[number,number];
export interface MatchingState {step:number;flow:number[][];pairs:MatchingPair[];reachable:number[];nextPath:number[]|null;terminal:boolean;cutCapacity:number|null;coverMask:number|null;}
const bit=(mask:number,i:number)=>(mask>>i)&1;
export const matchingNodeName=(node:number)=>node===0?'s':node===7?'t':node<4?`L${node-1}`:`R${node-4}`;
/** Deterministic integral residual BFS on a three-by-three bipartite reduction. */
export function bipartiteMatchingCover(m0:number,m1:number,m2:number,requested:number,queryMask:number){
 for(const [v,max] of [[m0,7],[m1,7],[m2,7],[requested,3],[queryMask,63]])if(!Number.isInteger(v)||v<0||v>max)throw new Error('Use integer neighbor masks, prefix and queried vertex mask in their stated ranges.');
 const masks=[m0,m1,m2],capacity=Array.from({length:8},()=>Array(8).fill(0) as number[]),edges:MatchingPair[]=[];
 for(let i=0;i<3;i++){capacity[0][i+1]=1;capacity[i+4][7]=1;for(let j=0;j<3;j++)if(bit(masks[i],j)){capacity[i+1][j+4]=4;edges.push([i,j]);}}
 const cut=(sourceSide:number[])=>capacity.reduce((sum,row,u)=>sum+(sourceSide.includes(u)?row.reduce((n,c,v)=>n+(!sourceSide.includes(v)?c:0),0):0),0),history:MatchingState[]=[];let flow=capacity.map(row=>row.map(()=>0)),step=0;
 while(true){const parent=Array(8).fill(-1) as number[],queue=[0];parent[0]=0;for(let head=0;head<queue.length;head++){const u=queue[head];for(let v=0;v<8;v++)if(parent[v]===-1&&capacity[u][v]-flow[u][v]>0){parent[v]=u;queue.push(v);}}
  const reachable=parent.flatMap((p,i)=>p<0?[]:[i]),terminal=parent[7]<0,pairs=edges.filter(([i,j])=>flow[i+1][j+4]===1),path:number[]=[];if(!terminal){for(let v=7;v!==0;v=parent[v])path.push(v);path.push(0);path.reverse();}
  const coverMask=terminal?Array.from({length:3},(_,i)=>Number(!reachable.includes(i+1))*(1<<i)+Number(reachable.includes(i+4))*(1<<(i+3))).reduce((a,b)=>a+b,0):null;
  history.push({step,flow:flow.map(row=>row.slice()),pairs:pairs.map(p=>[...p] as MatchingPair),reachable,nextPath:terminal?null:path,terminal,cutCapacity:terminal?cut(reachable):null,coverMask});if(terminal)break;
  flow=flow.map(row=>row.slice());for(let i=0;i<path.length-1;i++){const u=path[i],v=path[i+1];flow[u][v]++;flow[v][u]--;}step++;
 }
 const current=history[Math.min(requested,history.length-1)],last=history.at(-1)!,queryVertices=Array.from({length:6},(_,i)=>i).filter(i=>bit(queryMask,i)),uncovered=edges.filter(([i,j])=>!bit(queryMask,i)&&!bit(queryMask,j+3)),querySourceSide=[0,...Array.from({length:3},(_,i)=>i+1).filter(i=>!bit(queryMask,i-1)),...Array.from({length:3},(_,j)=>j+4).filter(j=>bit(queryMask,j-1))],queryCutCapacity=cut(querySourceSide),nextArcs=current.nextPath?.slice(0,-1).map((u,i)=>{const v=current.nextPath![i+1];return {u,v,residual:capacity[u][v]-current.flow[u][v],kind:u>=1&&u<=3&&v>=4&&v<=6?'add':u>=4&&u<=6&&v>=1&&v<=3?'cancel':'unit'};})??[];
 return {masks,edges,capacity,history,current,last,requested,queryMask,queryVertices,uncovered,queryIsCover:uncovered.length===0,querySourceSide,queryCutCapacity,queryGap:queryCutCapacity-current.step,nextArcs};
}
