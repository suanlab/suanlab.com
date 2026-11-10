'use client';
import { useId } from 'react';
import { useLanguage } from '@/components/language-provider';
import type { CausalResult } from '@/lib/learning-causality';
import MeasureFigure from './measure-figure';

const researchDiagramText=(graph:string,ko:boolean,short=false)=>{
 const labels:Record<string,[string,string,string,string]>={
 instrument:['도구변수의 경로와 가정','Instrument paths and assumptions','숨은 H는 Z와 U의 연관을 만들 수 있습니다. 현재 0이 아닌 계수의 경로를 표시합니다. 그래프의 가정과 IV 비율의 수치 일치를 구분하세요.','Hidden H can correlate Z and U. Paths with current nonzero coefficients are shown. Distinguish graphical assumptions from a numerical IV equality.'],
 mediation:['직접 경로와 매개 경로','Direct and mediated paths','X→M→Y와 X→Y를 분리합니다. 자연 효과는 같은 잡음을 공유하는 잠재 매개값을 사용하며, 관측자료의 식별에는 추가 조건이 필요합니다.','Separate X→M→Y from X→Y. Natural effects use potential mediators sharing the same noise; observational identification requires additional conditions.'],
 discovery:['선언한 생성 그래프','Declared generating graph','그림은 선언한 생성 구조입니다. 표는 관측 공분산의 독립성과 양립하는 DAG 후보입니다. 후보 한 개를 곧 실제 인과 구조라고 단정하지 마세요.','The diagram is the declared generating structure. The table lists DAGs compatible with observed covariance independences. A candidate is not automatically the true causal structure.'],
 transport:['E는 환경 차이를 표시','E denotes environment differences','환경 E는 모집단 Z의 분포를 바꿉니다. E→Y가 있으면 결과 기제도 달라집니다. E는 선택 collider S=1과 구분합니다.','Environment E changes the distribution of Z. An E→Y edge additionally changes the outcome mechanism. Distinguish E from selection on a collider S=1.'],
 policy:['숨은 U를 가진 로그 생성 그래프','Logging graph with hidden U','그림은 로그를 만든 배정 기제입니다. 목표 정책은 U에 독립인 새 동전으로 X를 배정합니다. 조건부 propensity는 숨은 U를 안다는 oracle 계산입니다.','This is the logging assignment graph. The target policy assigns X with a new coin independent of U. Conditional propensity weighting is an oracle calculation requiring knowledge of hidden U.']};
 const row=labels[graph];return row?.[(short?0:2)+(ko?0:1)];
};

export function CausalGraph({ result, label }: { result: CausalResult; label: string }) {
  const marker = useId().replace(/:/g, '');
  const { language } = useLanguage(), ko = language === 'ko';
  const both = result.graph === 'confounding-selection';
  const node = (x: number, y: number, name: string, selected = false) => <g key={name + x}>{selected ? <rect x={x - 22} y={y - 22} width="44" height="44" rx="6" fill="hsl(var(--card))" stroke="var(--learning-orange)" strokeWidth="2"/> : <circle cx={x} cy={y} r="22" fill="hsl(var(--card))" stroke="var(--learning-blue)" strokeWidth="2"/>}<text x={x} y={y + 6} textAnchor="middle" fill="currentColor" fontSize="18">{name}</text></g>;
  const edge = (x1: number, y1: number, x2: number, y2: number, cut = false) => {
    const dx = x2 - x1, dy = y2 - y1, length = Math.hypot(dx, dy), ux = dx / length, uy = dy / length;
    return <g key={`${x1},${y1},${x2},${y2}`}><path d={`M${x1 + 25 * ux} ${y1 + 25 * uy}L${x2 - 29 * ux} ${y2 - 29 * uy}`} fill="none" stroke={cut ? 'var(--learning-orange)' : 'var(--learning-blue)'} strokeWidth="2.5" strokeDasharray={cut ? '5 4' : undefined} markerEnd={cut ? undefined : `url(#${marker})`}/>{cut && <text x={(x1 + x2) / 2 - 8} y={(y1 + y2) / 2} fontSize="22" fill="var(--learning-orange)">×</text>}</g>;
  };
  if(result.network){
    const network=result.network;
    return <svg data-causal-graph data-causal-kind={result.graph} viewBox="0 0 350 255" role="img" aria-label={label}>
      <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="var(--learning-blue)"/></marker></defs>
      {network.edges.map(e=>edge(network.nodes[e.from].x,network.nodes[e.from].y,network.nodes[e.to].x,network.nodes[e.to].y,e.removed))}
      {network.nodes.map(n=><g key={n.name}>{n.conditioned?<rect x={n.x-22} y={n.y-22} width="44" height="44" rx="6" fill="hsl(var(--card))" stroke="var(--learning-orange)" strokeWidth="2"/>:<circle cx={n.x} cy={n.y} r="22" fill="hsl(var(--card))" stroke={n.hidden||n.intervened?'var(--learning-orange)':'var(--learning-blue)'} strokeWidth="2" strokeDasharray={n.hidden?'5 4':undefined}/>}<text x={n.x} y={n.y+5} textAnchor="middle" fill="currentColor" fontSize="16">{n.name}</text></g>)}
      <text x="175" y="248" textAnchor="middle" fill="currentColor" fontSize="12">{researchDiagramText(result.graph,ko,true)??(result.graph==='criterion'?(ko?'규칙 조건을 검사하는 변환 그래프':'Transformed graph for the rule criterion'):result.graph==='twin'?(ko?'f: 관측 세계 · a: 행동 세계 · 공유 U':'f: factual · a: action · shared U'):(ko?'점선 U: 관측되지 않은 공통 원인':'Dashed U: unobserved common cause'))}</text>
    </svg>;
  }
  return <svg data-causal-graph viewBox={both ? '0 0 600 225' : '0 0 350 225'} role="img" aria-label={label}>
    <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="var(--learning-blue)"/></marker></defs>
    {both ? <>{edge(145,55,65,165)}{edge(145,55,225,165)}{edge(65,165,225,165)}{node(145,55,'Z')}{node(65,165,'X')}{node(225,165,'Y')}{edge(375,55,455,165)}{edge(535,55,455,165)}{node(375,55,'X')}{node(535,55,'Y')}{node(455,165,'S=1',true)}<text x="145" y="215" textAnchor="middle" fill="currentColor" fontSize="13">{ko ? 'A: 공통 원인' : 'A: common cause'}</text><text x="455" y="215" textAnchor="middle" fill="currentColor" fontSize="13">{ko ? 'B: 선택된 collider' : 'B: selected collider'}</text></> : <>{edge(175,45,75,155,result.intervention)}{result.graph !== 'chain' && edge(175,45,275,155)}{result.graph !== 'fork' && edge(75,155,275,155)}{node(175,45,'Z')}{node(75,155,'X')}{node(275,155,'Y')}<text x="175" y="215" textAnchor="middle" fill="currentColor" fontSize="13">{result.intervention ? (ko ? '× 처치 배정 기제 교체' : '× assignment mechanism replaced') : (ko ? '화살표: 선언한 구조적 의존성' : 'Arrows: declared structural dependencies')}</text></>}
  </svg>;
}
export default function CausalFigure({ result, label, axis }: { result: CausalResult; label: string; axis: string }) {
  const { language } = useLanguage(), ko = language === 'ko';
  const display = { ...result, dots: result.dots ?? result.curves.filter(c => c.length === 1).flat() };
  return <div className="ai-causal-figure"><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko ? '인과 그래프 · 작은 화면에서 가로로 이동' : 'Causal graph · Scroll horizontally on narrow screens'}><CausalGraph result={result} label={label}/></div><figcaption>{researchDiagramText(result.graph,ko)??(result.network?(result.graph==='criterion'?(ko?'주황 ×는 규칙 조건을 검사하기 위해 삭제한 변입니다. 이 그래프를 확률 계산의 개입 모형과 구별하세요. 사각형은 조건 변수입니다.':'Orange × marks edges removed for the rule criterion. Distinguish this graph from the intervention model used in probability calculations. Squares denote conditioning.'):result.graph==='twin'?(ko?'두 세계는 같은 외생 상태 U를 공유합니다. 사각형은 관측 증거, 주황 원은 행동으로 고정한 변수, 점선 원은 잠재 상태입니다.':'Both worlds share the same exogenous state U. Squares mark evidence, the orange circle marks the assigned action, and the dashed circle marks latent state.'):(ko?'숨은 U가 처치 배정과 결과 기제를 함께 결정합니다. 관측표가 같아도 미관측 잠재 결과의 완성은 여러 가지입니다.':'Hidden U determines assignment and outcome mechanisms. The same observed table admits multiple completions of unobserved potential outcomes.')):ko ? '화살표는 모형의 인과 가정입니다. 크기나 통계적 유의성을 나타내지 않습니다. 주황 ×는 개입으로 교체한 입력 기제, 사각형 S=1은 선택 조건입니다.' : 'Arrows encode model assumptions, not effect sizes or statistical significance. Orange × marks a replaced input mechanism; square S=1 marks selection.')}</figcaption></figure><MeasureFigure result={display} label={label} axis={axis} variant="causal" wideLabels/></div>;
}
