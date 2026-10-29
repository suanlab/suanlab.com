'use client';
import { useLanguage } from '@/components/language-provider';
import { formatNumber } from '@/lib/learning-simulations';
import type { FoundationResult } from '@/lib/learning-foundations';
import type { TensorResult } from '@/lib/learning-tensors';
import type { BookText } from '@/data/interactive-book';
export default function TensorFigure({result,label,axis,legend}:{result:FoundationResult;label:string;axis:string;legend:BookText[]}){
 const {language}=useLanguage(),ko=language==='ko',r=result as TensorResult,network=r.network,pts=r.curves.flat();
 const maxX=Math.max(1,...pts.map(p=>p[0])),minY=Math.min(0,...pts.map(p=>p[1])),maxY=Math.max(1,...pts.map(p=>p[1])),xy=([x,y]:number[])=>[45+260*x/maxX,220-170*(y-minY)/(maxY-minY)];
 return <figure className="ai-foundation-figure ai-tensor-figure">
 {(network||r.curves.length>0)&&<div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'텐서 그림 · 작은 화면에서 가로로 이동':'Tensor diagram · Scroll horizontally on narrow screens'}>
 {network?<svg viewBox="0 0 350 260" role="img" aria-label={label} data-tensor-network="true">
 {network.edges.map((edge,i)=>{const u=network.nodes.find(n=>n.id===edge.from)!,v=network.nodes.find(n=>n.id===edge.to)!;return <g key={i}><path d={`M${u.x} ${u.y}L${v.x} ${v.y}`} stroke="var(--learning-blue)" strokeWidth="2"/><text x={(u.x+v.x)/2+(u.x===v.x?25:0)} y={(u.y+v.y)/2+(u.x===v.x?4:-8)} fill="var(--learning-neutral)" textAnchor="middle" fontSize="11">{edge.label}</text></g>;})}
 {network.legs.map((leg,i)=>{const n=network.nodes.find(n=>n.id===leg.node)!;return <g key={i}><path d={`M${n.x} ${n.y}L${leg.x} ${leg.y}`} stroke="var(--learning-orange)" strokeWidth="2"/><text x={leg.x} y={leg.y+17} fill="var(--learning-neutral)" textAnchor={leg.x<25?'start':leg.x>325?'end':'middle'} fontSize="12">{leg.label}</text></g>;})}
 {network.nodes.map(n=><g key={n.id}><rect x={n.x-37} y={n.y-18} width="74" height="36" rx="7" fill="hsl(var(--card))" stroke="var(--learning-blue)" strokeWidth="2"/><text x={n.x} y={n.y+4} textAnchor="middle" fill="var(--learning-neutral)" fontSize="11">{n.label}</text></g>)}
 <text x="175" y="250" textAnchor="middle" fill="var(--learning-neutral)" fontSize="11">{ko?'파랑: 합산 지표 · 주황: 자유 지표':'Blue: summed indices · Orange: free indices'}</text>
 </svg>:<svg viewBox="0 0 350 280" role="img" aria-label={label} data-tensor-entries="true">
 <path d={`M35 ${xy([0,0])[1]}H320M45 35V230`} fill="none" stroke="var(--learning-neutral)"/>
 {r.curves.flatMap((curve,i)=>curve.map((p,j)=><g key={`${i}-${j}`}><path d={`M${xy([p[0],0])}L${xy(p)}`} stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth={i?2:3}/><circle cx={xy(p)[0]} cy={xy(p)[1]} r={i?5:3} fill={i?'none':'var(--learning-blue)'} stroke={`var(--learning-${i?'orange':'blue'})`} strokeWidth="2"/></g>))}
 <g fill="var(--learning-neutral)" fontSize="12"><text x="45" y="25">{formatNumber(maxY)}</text><text x="45" y="245">0</text><text x="305" y="245" textAnchor="end">{formatNumber(maxX)}</text><text x="45" y="265">{ko?'이산 지표의 값 · 선형 보간하지 않습니다.':'Discrete indices; no interpolation is implied.'}</text></g>
 </svg>}
 </div>}
 {r.curves.length===0&&!network&&<p className="ai-model-note" role="status">{ko?'입력 shape가 호환되지 않아 출력 텐서가 없습니다.':'Incompatible input shapes: no output tensor.'}</p>}
 <figcaption>{axis} {network&&(ko?'상자의 위치·선 길이는 수치 크기나 실행 시간을 나타내지 않습니다.':'Box positions and edge lengths encode neither magnitude nor runtime.')}</figcaption>
 {!network&&<div className="ai-legend">{legend.slice(0,r.curves.length).map((text,i)=><span key={i} className={i?'ai-orange':'ai-blue'}>{text[language]}</span>)}</div>}
 <div className="ai-discrete-table" tabIndex={0} role="region" aria-label={ko?'텐서 계산 표 · 좌우 화살표로 이동':'Tensor calculation table · Use arrow keys to scroll'}><table><thead><tr>{r.table.headers.map(h=><th key={h} scope="col">{h}</th>)}</tr></thead><tbody>{r.table.rows.map((row,i)=><tr key={i}>{row.map((v,j)=><td key={j}>{typeof v==='number'?formatNumber(v):v}</td>)}</tr>)}</tbody></table></div>
 </figure>;
}
