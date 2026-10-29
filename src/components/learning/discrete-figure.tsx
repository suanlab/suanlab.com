'use client';
import { useId } from 'react';
import { useLanguage } from '@/components/language-provider';
import type { FoundationResult } from '@/lib/learning-foundations';
import type { DiscreteNetwork, DiscreteResult } from '@/lib/learning-discrete';
import { formatNumber } from '@/lib/learning-simulations';

export function DiscreteNetworkDrawing({network,label}:{network:DiscreteNetwork;label:string}) {
 const arrow=useId().replace(/:/g,'');
 return <svg viewBox="0 0 350 300" role="img" aria-label={label} data-discrete-network="true">
 <defs><marker id={arrow} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="var(--learning-neutral)"/></marker></defs>
 {network.edges.map((edge,i)=>{
  const u=network.nodes.find(n=>n.id===edge.from)!,v=network.nodes.find(n=>n.id===edge.to)!;const dx=v.x-u.x,dy=v.y-u.y,len=Math.hypot(dx,dy),self=u.id===v.id;
  // Curves separate opposite directions; symbol 1 uses the other side of a self-loop.
  const sign=edge.label==='1'?-1:1;
  const sx=self?u.x-17:u.x+dx/len*24,sy=self?u.y-16:u.y+dy/len*24,tx=self?u.x+17:v.x-dx/len*25,ty=self?u.y-16:v.y-dy/len*25;
  const cx=self?u.x:(u.x+v.x)/2-dy/len*28,cy=self?u.y-80*sign:(u.y+v.y)/2+dx/len*28;
  const path=self?`M${sx} ${sy} C${u.x-55} ${u.y-90*sign} ${u.x+55} ${u.y-90*sign} ${tx} ${ty}`:network.directed?`M${sx} ${sy} Q${cx} ${cy} ${tx} ${ty}`:`M${sx} ${sy}L${tx} ${ty}`;
  return <g key={i}><path d={path} fill="none" stroke={edge.active?'var(--learning-orange)':'var(--learning-neutral)'} strokeWidth={edge.active?3:1.8} markerEnd={network.directed?`url(#${arrow})`:undefined}/>{edge.label!==undefined&&<text x={cx} y={self?u.y-65*sign:cy-5} textAnchor="middle" fill="var(--learning-neutral)" fontSize="13">{edge.label}</text>}</g>;
 })}
 {network.nodes.map(node=><g key={node.id}><circle cx={node.x} cy={node.y} r="23" fill="hsl(var(--card))" stroke={node.active?'var(--learning-orange)':'var(--learning-blue)'} strokeWidth={node.active?3:2}/>{node.accepting&&<circle cx={node.x} cy={node.y} r="18" fill="none" stroke="var(--learning-blue)" strokeWidth="1.5"/>}<text x={node.x} y={node.y+4} textAnchor="middle" fill="var(--learning-neutral)" fontSize={node.label.length>5?10:13}>{node.label}</text></g>)}
 </svg>;
}
export default function DiscreteFigure({result,label,axis,multicolor=false}:{result:FoundationResult;label:string;axis:string;multicolor?:boolean}) {
 const {language}=useLanguage(),ko=language==='ko',r=result as DiscreteResult;
 const pts=r.curves.flat(),maxX=Math.max(1,...pts.map(v=>v[0])),maxY=Math.max(1,...pts.map(v=>Math.abs(v[1])));
 const xy=([x,y]:number[])=>[45+260*x/maxX,145-100*y/maxY];
 return <figure className="ai-foundation-figure ai-discrete-figure">
 {(r.network||r.curves.length>0)&&<div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'이산 구조 그림 · 작은 화면에서 가로로 이동':'Discrete structure · Scroll horizontally on narrow screens'}>
 {r.network?<DiscreteNetworkDrawing network={r.network} label={label}/>:<svg viewBox="0 0 350 280" role="img" aria-label={label} data-discrete-plot="true"><path d="M35 145H320M45 30V250" fill="none" stroke="var(--learning-neutral)"/>{r.curves.flatMap((curve,i)=>curve.map((point,j)=><g key={`${i}-${j}`}><path d={`M${xy([point[0],0])}L${xy(point)}`} stroke={multicolor&&i?"var(--learning-orange)":"var(--learning-blue)"} strokeWidth="2"/><circle cx={xy(point)[0]} cy={xy(point)[1]} r={multicolor&&i?6:4} fill={multicolor&&i?"none":"var(--learning-blue)"} stroke={multicolor&&i?"var(--learning-orange)":"var(--learning-blue)"} strokeWidth={multicolor?2:0}/></g>))}<g fontSize="12" fill="var(--learning-neutral)"><text x="45" y="265">0</text><text x="305" y="265" textAnchor="end">{maxX}</text><text x="45" y="25">±{formatNumber(maxY)}</text></g></svg>}
 </div>}
 <figcaption>{axis} {r.network?(ko?'노드 위치와 선 길이는 수치 크기를 나타내지 않습니다.':'Node positions and edge lengths do not encode magnitude.'):(ko?'정수 위치의 값입니다. 점 사이를 연속 함수로 보간하지 않습니다.':'Values at integer indices; no continuous interpolation is implied.')}</figcaption>
 {r.table&&<div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'계산 표 · 좌우 화살표로 이동':'Calculation table · Use arrow keys to scroll'}><table><thead><tr>{r.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{r.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?formatNumber(v):v}</td>)}</tr>)}</tbody></table></div>}
 </figure>;
}
