'use client';
import {useLanguage} from '@/components/language-provider';
import type {GeometryResult} from '@/lib/learning-geometry';
import MeasureFigure from './measure-figure';
export function GeometryComplexDrawing({complex,label}:{complex:NonNullable<GeometryResult['complex']>;label:string}){
 const points=complex.points,xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),loX=Math.min(0,...xs),hiX=Math.max(.1,...xs),loY=Math.min(0,...ys),hiY=Math.max(.1,...ys),scale=Math.min(240/(hiX-loX),170/(hiY-loY)),cx=(hiX+loX)/2,cy=(hiY+loY)/2;
 const xy=(i:number)=>[175+(points[i][0]-cx)*scale,115-(points[i][1]-cy)*scale];
 return <svg viewBox="0 0 350 235" role="img" aria-label={label} data-geometry-complex="true">{complex.simplices.filter(s=>s.length===3).map((s,i)=><polygon key={'f'+i} points={s.map(j=>xy(j).join(',')).join(' ')} fill="var(--learning-blue)" opacity=".12"/>)}{complex.simplices.filter(s=>s.length===2).map(([a,b],i)=><path key={'e'+i} d={`M${xy(a)}L${xy(b)}`} fill="none" stroke="var(--learning-blue)" strokeWidth="2"/>)}{points.map((_,i)=><g key={i}><circle cx={xy(i)[0]} cy={xy(i)[1]} r="5" fill="var(--learning-orange)"/><text x={xy(i)[0]+8} y={xy(i)[1]-8} fill="var(--learning-neutral)" fontSize="12">{i}</text></g>)}</svg>;
}
export default function GeometryFigure({result,label,axis}:{result:GeometryResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko',complex=result.complex;
 return <>{complex&&<figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'현재 임계값의 Rips complex · 작은 화면에서 가로로 이동':'Rips complex at current threshold · Scroll horizontally on narrow screens'}><GeometryComplexDrawing complex={complex} label={ko?'현재 거리 임계값에서 포함된 꼭짓점·간선·삼각형':'Vertices, edges and triangles included at the current distance threshold'}/></div><figcaption>{ko?'현재 threshold의 복합체를 동일 축척으로 그립니다. 색칠한 삼각형은 포함된 2-simplex입니다. 사면체는 평면에 투영되므로 별도 수치로 확인하세요.':'Equal-scale drawing of the complex at the current threshold. Shaded triangles are included 2-simplices. The tetrahedron is projected into the plane; inspect its count separately.'}</figcaption></figure>}<MeasureFigure result={result} label={label} axis={axis} variant="geometry" equalAspect={Boolean(result.spatial)}/></>;
}
