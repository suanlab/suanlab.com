'use client';
import React,{useId,useMemo,useState} from 'react';
import type {RealTimePhaseValidatedSuccess} from '@/lib/learning-real-time-phase-protocol';
import {defaultRealTimeMatrixDisplay,realTimeMatrixKinds,realTimePhaseMatrixView,type RealTimeMatrixDisplay,type RealTimeMatrixKind} from '@/lib/learning-real-time-phase-matrix-view';
import Plot from './real-time-phase-matrix-plot';
const names:Record<RealTimeMatrixKind,{ko:string;en:string}>={
  hessian:{ko:'작용 Hessian',en:'Action Hessian'},inverse:{ko:'역곡률 H⁻¹',en:'Inverse curvature H⁻¹'},
  'formal-connected':{ko:'형식적 연결 응답 iℏH⁻¹',en:'Formal connected response i hbar H⁻¹'},
  'regulated-connected':{ko:'조절된 연결 응답 ℏCε⁻¹',en:'Regulated connected response hbar Cε⁻¹'},
  map:{ko:'고전 map M',en:'Classical map M'},'initial-covariance':{ko:'초기 대칭화 공분산',en:'Initial symmetrized covariance'},'final-covariance':{ko:'최종 대칭화 공분산',en:'Final symmetrized covariance'},
};
/** Caller supplies only the owned response from useRealTimePhaseCalculation. */
export default function RealTimePhaseMatrix({response,language}:{response:RealTimePhaseValidatedSuccess;language:'ko'|'en'}){
  const id=useId(),t=(ko:string,en:string)=>language==='ko'?ko:en;
  const [display,setDisplay]=useState<RealTimeMatrixDisplay>(defaultRealTimeMatrixDisplay),[table,setTable]=useState(false);
  const kinds=realTimeMatrixKinds(response),kind=kinds.includes(display.kind)?display.kind:kinds[0];
  const model=useMemo(()=>kind?realTimePhaseMatrixView(response,{...display,kind}):null,[response,display,kind]);
  if(!model)return <p data-real-time-matrix="not-applicable">{t('canonical fold는 별도의 한 좌표 적분이며 이 사슬 행렬을 갖지 않습니다.','The canonical fold is a separate one-coordinate integral and has no such chain matrix.')}</p>;
  const m=model,selected=m.selectedEntry,range=selected[display.component];
  return <section aria-labelledby={id} data-real-time-matrix={m.kind}>
    <h2 id={id}>{t('검산된 행렬을 비교하세요','Compare the verified matrices')}</h2>
    <p>{t('행과 열의 전체 성분을 유지합니다. 내부 격자 행렬은 고정끝을 제외한1…N−1, map과 공분산은 q,p 순서입니다. 복소 연결 응답은 위치 확률 공분산이 아닙니다. q,p 행렬에서는 단위가 다른 성분의 색을 같은 물리량으로 해석하지 마세요.','Every row and column is retained. Interior lattice matrices use 1…N−1, excluding pinned endpoints; maps and covariances use q,p. Complex connected responses are not position probability covariances. Entries of a q,p matrix can have different units; their colors do not compare one common physical quantity.')}</p>
    <label className="qubit-field">{t('행렬','Matrix')}<select aria-label={t('행렬','Matrix')} value={m.kind} onChange={e=>setDisplay(d=>({...d,kind:e.target.value as RealTimeMatrixKind,selectedCell:0}))}>{kinds.map(k=><option key={k} value={k}>{names[k][language]}</option>)}</select></label>
    <label className="qubit-field">{t('복소 성분','Complex component')}<select aria-label={t('복소 성분','Complex component')} value={display.component} onChange={e=>setDisplay(d=>({...d,component:e.target.value as 'real'|'imaginary'}))}><option value="real">{t('실수부','Real part')}</option><option value="imaginary">{t('허수부','Imaginary part')}</option></select></label>
    <label className="qubit-field">{t('계수 표시 척도','Coefficient display scale')}<select aria-label={t('계수 표시 척도','Coefficient display scale')} value={display.scale} onChange={e=>setDisplay(d=>({...d,scale:e.target.value as 'linear'|'signed-log'}))}><option value="linear">{t('선형 계수','Linear coefficients')}</option><option value="signed-log">{t('부호 로그 계수','Signed-log coefficients')}</option></select></label>
    <div className="qubit-actions">{(['row','column'] as const).map(axis=><label key={axis} className="qubit-field">{axis==='row'?t('선택 행','Selected row'):t('선택 열','Selected column')}<select aria-label={axis==='row'?t('선택 행','Selected row'):t('선택 열','Selected column')} value={m[axis]} onChange={e=>setDisplay(d=>({...d,selectedCell:axis==='row'?Number(e.target.value)*m.n+m.column:m.row*m.n+Number(e.target.value)}))}>{m.labels.map((label,i)=><option key={i} value={i}>{label}</option>)}</select></label>)}</div>
    <p style={{overflowWrap:'anywhere'}} data-real-time-scale>{t('성분의 원래 값','Original component value')} = {m.multiplier} × 2^{m.scalePowerOfTwo} × f. {t('전송 계수 구간은','Wire coefficient intervals represent')} f = [lower, upper] × 2^−{m.coefficientBits}. {t('소수 그림의 구간 중점은 표시용 근사입니다.','Decimal midpoint drawings are display approximations.')} {t('표시 변환','Display transform')}: {m.displayTransform}.</p>
    {m.projectionAvailable?<>
      <figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('행렬 열지도 · 좌우 이동 가능','Matrix heat map; scroll horizontally')}><svg viewBox={`0 0 ${70+m.n*20} ${70+m.n*20}`} role="img" aria-label={names[m.kind][language]} style={{width:'100%',minWidth:Math.max(320,70+m.n*20)}}>
        {m.labels.map((label,i)=><g key={i} fill="currentColor" fontSize={13}><text x={50+i*20} y={28} textAnchor="middle">{label}</text><text x={26} y={54+i*20} textAnchor="end">{label}</text></g>)}
        {m.heights.flatMap((row,i)=>row.map((value,j)=>{const cell=m.source[i][j],chosen=i*m.n+j===m.selected;return <g key={`${i}-${j}`}><rect x={40+j*20} y={40+i*20} width={19} height={19} fill={value>0?'var(--learning-blue)':value<0?'var(--learning-orange)':'var(--learning-neutral)'} fillOpacity={cell.exactZero?.12:Math.max(.08,Math.abs(value)/m.maximum)} stroke={chosen?'var(--learning-green)':'none'} strokeWidth={chosen?3:0}/>{(cell.unresolved||!cell.exactZero&&Math.abs(value)/m.maximum<1e-6)&&<path d={`M${44+j*20},${44+i*20}l10,10m0,-10l-10,10`} stroke="currentColor" strokeWidth={1}/>}</g>;}))}
      </svg></div><figcaption>{t('열지도: 중점 양수 파랑·음수 주황, 선택 성분 초록 테두리. ×는 부호 미해결 또는 이 척도에서 작은 성분입니다. 행·열 선택기로 어느 성분이든 확인할 수 있습니다.','Heat map: positive midpoint blue, negative midpoint orange; green outlines the selection. A cross marks an unresolved or tiny component at this scale. Row and column selectors reach every entry.')}</figcaption></figure>
      <Plot model={m} language={language}/>
    </>:<p role="status">{t('소수 그림으로 안전하게 투영할 수 없습니다. 원래 구간 표시는 유지합니다.','A safe decimal drawing is unavailable. The original interval representation is retained.')}</p>}
    <h3>{t('선택 성분의 원래 구간','Original interval for the selected entry')} ({m.labels[m.row]}, {m.labels[m.column]})</h3>
    <p>{selected.exactZero?t('선택한 성분은 정확한0입니다.','The selected component is exactly zero.'):selected.unresolved?t('이 전송 구간은0을 포함합니다. 부호나 정확한0을 단정하지 않습니다.','This wire interval contains zero. Its sign and exact zero are not asserted.'):t('선택한 성분의 구간은0을 제외합니다.','The selected component interval excludes zero.')}</p>
    <pre data-real-time-selected style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{JSON.stringify({real:selected.real,imaginary:selected.imaginary,coefficientBits:selected.coefficientBits,scalePowerOfTwo:selected.scalePowerOfTwo,multiplier:selected.multiplier,component:display.component,lower:range.lower,upper:range.upper},null,2)}</pre>
    <details onToggle={e=>setTable(e.currentTarget.open)}><summary>{t('전체 성분의 원래 전송 구간','Original wire intervals for every entry')}</summary>{table&&<div className="qubit-table" style={{maxHeight:440,overflowY:'auto'}} role="region" tabIndex={0} aria-label={t('전체 행렬 구간 표','Full matrix interval table')}><table><caption>{names[m.kind][language]} · {m.n}×{m.n}</caption><thead><tr><th scope="col">i,j</th><th scope="col">Re f</th><th scope="col">Im f</th></tr></thead><tbody>{m.source.flatMap((row,i)=>row.map((cell,j)=><tr key={`${i}-${j}`}><th scope="row">{m.labels[i]}, {m.labels[j]}</th><td style={{overflowWrap:'anywhere'}}>[{cell.real.lower}, {cell.real.upper}]</td><td style={{overflowWrap:'anywhere'}}>[{cell.imaginary.lower}, {cell.imaginary.upper}]</td></tr>))}</tbody></table></div>}</details>
  </section>;
}
