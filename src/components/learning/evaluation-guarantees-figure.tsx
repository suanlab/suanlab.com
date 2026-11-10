'use client';
import {useLanguage} from '@/components/language-provider';
import type {PrivacyResult} from '@/lib/learning-privacy';
import MeasureFigure from './measure-figure';
const fmt=(v:number)=>Number(v.toPrecision(4)).toString();
export default function EvaluationGuaranteesFigure({result,label,axis}:{result:PrivacyResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko',s=result.evaluationStratified,r=result.evaluationSelection;
 const rows=s?[
  ko?'파랑: 신뢰 상한 · 주황: 예시의 참 위험':'Blue: upper bound · orange: oracle risk',
  `n = (${s.n.join(', ')}) · k = (${s.k.join(', ')})`,
  `${ko?'요청한 k':'Requested k'} = (${s.requested.join(', ')})`,
  `${ko?'목표 가중 추정':'Target empirical'}: ${s.weightedPoint===null?'—':fmt(s.weightedPoint)}`,
  `${ko?'관측 표본 통합 비율':'Pooled observed rate'}: ${s.pooledObservedError===null?'—':fmt(s.pooledObservedError)}`,
  `q₀ = ${fmt(s.q0)} · ${ko?'예시 참 위험':'Oracle risk'} = ${fmt(s.targetRisk)}`,
  `${ko?'알려진 혼합 상한':'Known mixture upper'}: ${fmt(s.knownMixtureUpper)}`,
  `${ko?'가중치 구간':'Weight interval'}: [${s.weightRange.map(fmt).join(', ')}]`,
  `${ko?'구간 내 최악 상한':'Worst upper in interval'}: ${fmt(s.ambiguousMixtureUpper)}`,
  ko?'참 위험은 관측으로 추정한 값이 아닙니다.':'Oracle risk is not estimated from observations.',
 ]:r?[
  ko?'파랑: 단일 개수 · 주황: 선택된 최소 개수':'Blue: single count · orange: selected minimum',
  `n=${r.n} · p=${fmt(r.p)} · M=${r.models}`,
  `${ko?'요청→실제 조회':'Requested → effective query'}: ${r.requestedCount} → ${r.k}`,
  `${ko?'가능한 조회 사건':'Possible query event'}: ${Number(r.queryPossible)}`,
  `${ko?'조회 상한':'Query upper'}: ${fmt(r.naiveUpper)} / ${fmt(r.adjustedUpper)}`,
  `${ko?'선택 후 포함률':'Selected coverage'}: ${fmt(r.naiveSelectedCoverage)}`,
  `${ko?'보정 후 포함률':'Adjusted coverage'}: ${fmt(r.adjustedSelectedCoverage)}`,
  `${ko?'새 홀드아웃 포함률':'Fresh holdout coverage'}: ${fmt(r.freshHoldoutCoverage)}`,
  `${ko?'정확한 보정 기준 통과':'Exact adjusted coverage passes'}: ${Number(r.adjustedMeetsAdvertised)}`,
  ko?'포함률은 반복 실험의 절차 성질입니다.':'Coverage is a repeated-experiment property.',
 ]:[];
 return <div className="ai-game-figure"><figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'평가 표본과 신뢰 보장 · 가로 이동 가능':'Evaluation samples and confidence guarantees · Horizontal scrolling available'}><svg data-evaluation-distribution viewBox="0 0 350 560" role="img" aria-label={label}>
 <path d="M45 55V205H315" fill="none" stroke="var(--learning-neutral)"/>
 {s?s.n.map((n,i)=><g key={i}><rect x={95+130*i} y={205-140*s.upper[i]} width="22" height={140*s.upper[i]} fill="var(--learning-blue)"/><rect x={120+130*i} y={205-140*s.conditionalTruth[i]} width="22" height={140*s.conditionalTruth[i]} fill="var(--learning-orange)"/><text x={118+130*i} y="225" fill="currentColor" textAnchor="middle" fontSize="11">{`G${i}: ${n===0?'—':fmt(s.k[i]/n)}`}</text></g>):r?r.pmf.map((v,k)=>{const cell=260/(r.n+1),x=50+cell*k;return <g key={k}><rect x={x} y={205-140*v} width={cell*.42} height={140*v} fill="var(--learning-blue)"/><rect x={x+cell*.45} y={205-140*r.minPMF[k]} width={cell*.42} height={140*r.minPMF[k]} fill="var(--learning-orange)"/>{k===r.k&&<path d={`M${x+cell/2} 50V210`} stroke="currentColor" strokeDasharray="3 3"/>}</g>;}):null}
 <g fill="currentColor" textAnchor="end" fontSize="10"><text x="38" y="69">1</text><text x="38" y="208">0</text></g>
 {r&&<g fill="currentColor" fontSize="10"><text x="50" y="225">0</text><text x="310" y="225" textAnchor="end">{r.n}</text></g>}
 {rows.map((row,i)=><text key={i} x="175" y={255+29*i} fill="currentColor" textAnchor="middle" fontSize="10">{row}</text>)}
 </svg></div><figcaption>{s?(ko?'집단별 막대는 수치 신뢰 상한과 설정한 참 위험을 구분합니다. 표본 없는 집단의 경험 오차는 정의되지 않습니다. 요청 개수는 표본 수로 제한됩니다.':'Group bars separate numerical upper bounds from illustrative true risks. An unsampled group has no empirical error rate. Requested counts clamp to sample sizes.'):(ko?'동일한 참 위험이어도 가장 낮은 관측 오차로 모델을 선택하면 개수 분포가 달라집니다. 점선은 조회한 개수이며 무작위 추출 결과가 아닙니다.':'Selecting the smallest observed error changes the count distribution even at identical true risk. The dashed line marks a queried count, not a simulated draw.')}</figcaption></figure>
 <figure className="ai-foundation-figure"><div className="ai-foundation-scroll" tabIndex={0} role="region" aria-label={ko?'평균 오차와 전체 입력 보장의 차이':'Average error versus correctness on every input'}><svg data-evaluation-null-set viewBox="0 0 350 350" role="img" aria-label={ko?'한 점에서만 틀리는 고정 반례':'Fixed counterexample with one incorrect point'}><path d="M45 90H305M175 50V220" fill="none" stroke="var(--learning-neutral)"/><path d="M55 210L175 90L295 210" fill="none" stroke="var(--learning-blue)" strokeWidth="3"/><circle cx="175" cy="90" r="6" fill="var(--learning-orange)"/>{[ko?'고정 반례: 점수 s(x) = −|x|':'Fixed counterexample: score s(x) = −|x|',ko?'정답 −1 · 점수가 0이면 +1':'Target −1 · predict +1 at score 0',ko?'오직 x=0에서 오분류':'Misclassification only at x=0',ko?'연속 균등 분포에서 평균 오차는 0':'Error probability under continuous uniform is 0',ko?'그래도 모든 입력에 옳다는 주장은 거짓':'Correctness on every input is still false'].map((v,i)=><text key={i} x="175" y={240+23*i} textAnchor="middle" fontSize="10" fill="currentColor">{v}</text>)}</svg></div><figcaption>{ko?'이 그림은 조작값과 별개의 고정 반례입니다. 유한 표본의 평가나 평균 위험 보장은 모든 입력에 대한 논리적 보장과 구별해야 합니다.':'This fixed counterexample is separate from the controls. Finite-sample evaluation and average-risk bounds differ from a logical guarantee for every input.'}</figcaption></figure>
 <MeasureFigure result={result} label={label} axis={axis} variant="privacy" wideLabels/>
 </div>;
}
