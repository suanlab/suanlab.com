'use client';
import { useId } from 'react';
import { useLanguage } from '@/components/language-provider';
import type { GraphicalNetwork, GraphicalResult } from '@/lib/learning-graphical';
import MeasureFigure from './measure-figure';
export function GraphicalDrawing({graph,label}:{graph:GraphicalNetwork;label:string}){
 const arrow=useId().replace(/:/g,'');
 return <svg viewBox="0 0 350 185" role="img" aria-label={label} data-graphical-structure="true"><defs><marker id={arrow} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="var(--learning-neutral)"/></marker></defs>{graph.edges.map((edge,i)=>{const u=graph.nodes.find(v=>v.id===edge.from)!,v=graph.nodes.find(v=>v.id===edge.to)!,dx=v.x-u.x,dy=v.y-u.y,len=Math.hypot(dx,dy),offsetU=u.factor?25:20,offsetV=v.factor?25:22;return <path key={i} d={`M${u.x+dx/len*offsetU} ${u.y+dy/len*offsetU}L${v.x-dx/len*offsetV} ${v.y-dy/len*offsetV}`} stroke="var(--learning-neutral)" strokeWidth="1.8" fill="none" markerEnd={edge.directed?`url(#${arrow})`:undefined}/>;})}{graph.nodes.map(node=><g key={node.id}>{node.factor?<rect x={node.x-23} y={node.y-16} width="46" height="32" rx="3" fill="hsl(var(--muted))" stroke={node.observed?'var(--learning-orange)':'var(--learning-blue)'} strokeWidth="2"/>:<circle cx={node.x} cy={node.y} r="18" fill={node.observed?'hsl(var(--muted))':'hsl(var(--card))'} stroke={node.observed?'var(--learning-orange)':'var(--learning-blue)'} strokeWidth={node.observed?3:2}/>}<text x={node.x} y={node.y+4} textAnchor="middle" fill="var(--learning-neutral)" fontSize={node.label.length>3?10:12}>{node.label}</text></g>)}</svg>;
}
export default function GraphicalFigure({result,label,axis}:{result:GraphicalResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko';
 return <><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={ko?'확률 그래프 구조 · 작은 화면에서 가로로 이동':'Probabilistic graph structure · Scroll horizontally on narrow screens'}><GraphicalDrawing graph={result.graph} label={(ko?'확률 모형의 구조. ':'Probabilistic model structure. ')+label}/></div><figcaption>{ko?'구조 그림입니다. 간선 길이는 의존 강도를 나타내지 않습니다. 주황 테두리는 관측 또는 조건화 표시입니다.':'Structural diagram. Edge lengths do not encode dependence strength. Orange outlines mark observation or conditioning.'}</figcaption></figure><MeasureFigure result={result} label={label} axis={axis} variant="graphical"/></>;
}
