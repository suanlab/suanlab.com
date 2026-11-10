'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { BlockOrderInput } from '@/lib/learning-block-order-input';
import type { BlockOrderResult } from '@/lib/learning-block-order-protocol';
import { blockOrderApprox, blockOrderSpatial, type BlockOrderMode } from '@/lib/learning-block-order-view';
import type { BlockOrderDisplay } from '@/lib/learning-block-order-state';
import BlockOrderSpatial from './block-order-spatial';

export default function BlockOrderResultView({ result,input,language,display,onDisplay,onProfile }: {
  result: BlockOrderResult; input: BlockOrderInput; language: 'ko' | 'en'; display: BlockOrderDisplay;
  onDisplay: (change: BlockOrderDisplay | ((p: BlockOrderDisplay) => BlockOrderDisplay)) => void; onProfile: (upCounts: number[]) => void;
}) {
  const id = useId(), r = result.exact, obs = result.observations, profiles = r.profileChunks.flat(), selected = profiles[r.selected.profileIndex];
  const t = (ko: string,en: string) => language === 'ko' ? ko : en;
  const [playing,setPlaying] = useState(false), [reduced,setReduced] = useState(false), latest = useRef(onDisplay); latest.current = onDisplay;
  const model = blockOrderSpatial(input,r,display.mode,display.example), pages = Math.ceil(profiles.length/25), page = Math.min(display.page,pages-1);
  const update = (change: Partial<BlockOrderDisplay>) => onDisplay(p => ({ ...p,...change }));
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)'), change = () => { setReduced(media.matches); if (media.matches) setPlaying(false); };
    const hide = () => { if (document.visibilityState === 'hidden') setPlaying(false); };
    change(); media.addEventListener('change',change); document.addEventListener('visibilitychange',hide);
    return () => { media.removeEventListener('change',change); document.removeEventListener('visibilitychange',hide); };
  },[]);
  useEffect(() => {
    if (!playing || reduced || display.mode !== 'configuration') return;
    const timer = setInterval(() => { if (document.visibilityState === 'hidden') { setPlaying(false); return; } latest.current(p => ({ ...p,example: (Math.min(p.example,r.selected.examples.length-1)+1)%r.selected.examples.length })); },1100);
    return () => clearInterval(timer);
  },[playing,reduced,display.mode,r.selected.examples.length]);
  const table = (caption: string,rows: (string | number)[][]) => <div className="qubit-scroll" role="region" tabIndex={0} aria-label={caption}><table className="scalar-table"><caption>{caption}</caption><thead><tr><th scope="col">{t('양','Quantity')}</th><th scope="col">{t('값','Value')}</th></tr></thead><tbody>{rows.map((row,i) => <tr key={i}><th scope="row">{row[0]}</th><td className="block-order-exact">{row[1]}</td></tr>)}</tbody></table></div>;
  const quantities = [
    ['log Zrel',obs.logRelativePartition],['log Z',obs.logPartition],['F/T',obs.freeEnergyOverTemperature],
    ['S(σ)',obs.microscopicEntropy],['S(k)',obs.profileEntropy],['E_k S(σ|k)',obs.meanHiddenEntropy],
    ['S(σ|k selected)',obs.selectedHiddenEntropy],['F_k/T',obs.selectedFreeEnergyOverTemperature],['E[E|k]/T',obs.selectedMeanEnergyOverTemperature],
    ['log(w/wref)',obs.meanProductLogWeightResidual],['log OR',obs.localityLogCrossRatio],
  ] as const;
  const totalUp = selected.upCounts.reduce((a,b) => a+b,0), witness = r.localityWitness;
  return <section data-block-result="verified">
    <h3>{t('현재 적용한 입력의 검증 결과','Verified result for the applied input')}</h3>
    <p className="block-order-exact" data-block-applied>{t('격자','Grid')}={input.shape.join('×')}; {t('블록','Block')}={input.blockShape.join('×')}; x={input.disagreementActivity}; y=({input.blockUpActivities.join(', ')}); k=({selected.upCounts.join(', ')})</p>
    <p>{t('경계','Boundary')}: {input.boundary === 'open' ? t('열린 경계','Open') : t('주기 경계','Periodic')}; N={r.sites}, B={r.blocks}, v={r.blockVolume}, {t('edge 수','edge count')}={r.edgeCount}.</p>
    {table(t('선택 프로필의 정확한 값','Exact values for the selected profile'),[
      ['m_b',selected.magnetizations.join(', ')],['Ω_k',selected.multiplicity],['Q_k(x)',selected.bondPolynomial],['P(k)',selected.probability],
      ['E[d|k]',selected.conditionalMeanDisagreements],['Var(d|k)',selected.conditionalVarianceDisagreements],
      ['K total',totalUp],['P(K total)',r.globalUpProbabilities[totalUp]],
    ])}
    <p>{t('P(k)는 공간 프로필 하나의 확률이며 P(K total)은 같은 전체 위쪽 수의 모든 프로필을 합한 값입니다. 조건부 프로필 확률은 P(k)/P(K total)입니다.','P(k) describes one spatial profile. P(K total) sums every profile with the same global up-count. The conditional profile probability is P(k)/P(K total).')}</p>
    <label className="qubit-field" htmlFor={`${id}-mode`}>{t('공간에 표시할 값','Quantity shown in space')}<select id={`${id}-mode`} aria-label={t('공간에 표시할 값','Quantity shown in space')} value={display.mode} onChange={e => { update({ mode: e.target.value as BlockOrderMode }); setPlaying(false); }}>
      <option value="configuration">{t('선택한 미시 배치 σᵢ','Selected microscopic arrangement σᵢ')}</option><option value="conditional">{t('고정 프로필의 조건부 평균 E[σᵢ|k]','Conditional means E[σᵢ|k] at fixed profile')}</option><option value="profile">{t('블록 평균 m_b','Block profile means m_b')}</option><option value="ensemble">{t('전체 앙상블의 블록 평균 E[m_b]','Block means E[m_b] in the full ensemble')}</option>
    </select></label>
    <p>{display.mode === 'configuration' ? t('아래는 현재 선택한 예제 배치 하나입니다. 앙상블 전체의 전형적인 표본이라는 뜻은 아닙니다.','Below is one selected example arrangement. It is not a representative random sample of the ensemble.')
      : display.mode === 'conditional' ? t('프로필을 고정한 채 모든 호환 미시 배치를 가중 평균합니다. 경계 위치와 내부 위치가 다를 수 있습니다.','Average every compatible microscopic arrangement at fixed profile. Boundary and interior positions can differ.')
      : display.mode === 'profile' ? t('각 블록의 평균을 내부 모든 위치에 반복해서 표시합니다. 실제 조건부 경계 스핀 평균과 같다는 보장은 없습니다.','Repeat each block average over its sites. This need not equal a conditional boundary-spin mean.')
      : t('선택한 프로필에 조건을 주지 않은 전체 분포의 평균입니다. 외부장0이면 스핀 반전 대칭으로 평균이0일 수 있습니다.','These means use the unrestricted distribution, not conditioning on the selected profile. Spin-flip symmetry can make them zero at zero field.')}</p>
    <label className="qubit-field" htmlFor={`${id}-example`}>{t('살펴볼 미시 예제','Microscopic example to inspect')}<select id={`${id}-example`} aria-label={t('살펴볼 미시 예제','Microscopic example to inspect')} value={model.exampleIndex} onChange={e => { update({ example: Number(e.target.value) }); setPlaying(false); }}>{r.selected.examples.map((e,i) => <option key={e.mask} value={i}>{i+1}. mask {e.mask}; d={e.disagreements}</option>)}</select></label>
    <div className="qubit-actions"><Button variant="outline" disabled={r.selected.examples.length<2} onClick={() => { update({ example: (model.exampleIndex+1)%r.selected.examples.length }); setPlaying(false); }}>{t('다음 예제 배치','Next example arrangement')}</Button><Button variant="outline" disabled={reduced || display.mode !== 'configuration' || r.selected.examples.length<2} onClick={() => setPlaying(p => !p)}>{playing ? t('배치 재생 멈춤','Pause arrangements') : t('예제 배치 순서대로 보기','Play example arrangements')}</Button></div>
    {reduced && <p>{t('동작 줄이기 설정으로 자동 재생을 끕니다. 다음 배치 버튼은 사용할 수 있습니다.','Reduced-motion preference disables autoplay. The next-arrangement button remains available.')}</p>}
    <p className="block-order-exact" data-block-example>mask={model.example.mask}; d={model.example.disagreements}; P(σ|k)={model.example.conditionalProbability}</p>
    <p>{t('배치는 가장 작은 numeric mask 최대8개입니다. 시간 진화·MCMC·무작위 추출이 아니며 이 목록으로 평균을 추정하지 않습니다. 전체 조건부 상태 수는 Ω_k입니다.','These are up to eight smallest numeric masks. Playback is not time evolution, MCMC or random sampling; do not estimate averages from this list. The complete conditional state count is Ω_k.')}</p>
    <BlockOrderSpatial model={model} input={input} language={language} flat={display.flat} onFlat={flat => update({ flat })}/>
    {table(t('그림에 표시한 위치별 정확한 값','Exact site values shown in the drawing'),model.points.map(p => [`i=${p.site}, B${p.block}`,p.exact]))}
    {table(t('블록 프로필과 앙상블 평균','Block profile and ensemble means'),selected.magnetizations.map((m,b) => [`B${b}`,`m_b=${m}; E[m_b]=${r.meanMagnetizations[b]}`]))}
    <h4>{t('최근접 pair 분해의 조건부 검사','Conditional test of nearest-neighbor pair factorization')}</h4>
    {!witness ? <p>{t('블록이 하나여서 쌍 검사를 정의하지 않습니다.','With one block no pair test is defined.')}</p> : <>
      {table(t('선택한 블록 쌍의 검사','Test of the selected block pair'),[[t('블록 쌍','Block pair'),witness.pair.join(', ')],[t('실제 이웃인가','Actually neighbors'),witness.pairIsNeighbor ? t('예','Yes') : t('아니요','No')],['OR',witness.crossRatio]])}
      <p data-block-locality>{witness.rulesOutNearestNeighborPairFactorization ? t('다른 블록을 고정해도 비이웃 쌍의 OR가1이 아닙니다. 이 거친 그래프의 양의 최근접 pair 분해는 불가능합니다.','With other blocks fixed, the nonneighbor pair has nonunit OR. Positive nearest-neighbor pair factorization on this coarse graph is impossible.')
        : witness.pairIsNeighbor ? t('검사한 두 블록은 실제 이웃입니다. OR가1이 아니어도 이 검사로 최근접 pair 분해를 배제할 수 없습니다.','The tested blocks are actual neighbors. A nonunit OR here does not rule out nearest-neighbor pair factorization.')
        : t('이 선택의 OR는1입니다. 한 검사만으로 모든 프로필과 모든 쌍의 독립성을 증명하지는 않습니다.','This selection has unit OR. One test does not establish independence for every profile and pair.')}</p>
      <p>{t('나머지 모든 블록은 적용한 프로필에 고정합니다. 전체 자화는 추가 고정하지 않습니다. 장거리·다체항을 허용한 모형 전체를 배제하지 않습니다.','Every other block is fixed to the applied profile; no additional global magnetization constraint is imposed. Models with longer-range or multiblock terms are not ruled out.')}</p>
    </>}
    <h4>{t('자유에너지·숨긴 엔트로피·국소 기준','Free energy, hidden entropy and the local reference')}</h4>
    {table(t('초월함수 구간의 근사 요약','Approximate summaries of transcendental enclosures'),quantities.flatMap(([name,v]) => v ? [[name,`${blockOrderApprox(v.lower)} … ${blockOrderApprox(v.upper)}`]] : []))}
    <p>{t('소수는 포함 구간의 근사 표시입니다. 같은 숫자로 보여도 정확한 폭0을 뜻하지 않습니다. 매우 작은 엔트로피의 구간이0을 가로지를 수도 있습니다. F/T는 무차원 값이며 에너지의 영점과 counting measure를 유지합니다.','Decimals approximate enclosure endpoints. Matching digits do not imply exactly zero width; an enclosure for a very small entropy may cross zero. F/T is dimensionless and retains the energy reference and counting measure.')}</p>
    <p className="block-order-formula">S(σ)=S(k)+E_k S(σ|k); log(w/wref)=log(Rpower)/b</p>
    <details><summary>{t('정확한 피연산자와 로그 포함 구간','Exact operands and logarithm enclosures')}</summary>
      {table(t('정확한 정규화·국소 기준 피연산자','Exact normalization and local-reference operands'),[['Zrel',r.relativePartition],['Z²',r.absolutePartitionSquared],['Q_b(k_b)',r.selected.isolatedBlockPolynomials.join(', ')],['dbar cross',r.selected.meanProductCrossDisagreements],['Rpower',r.selected.meanProductResidualPower],['b',r.selected.meanProductResidualDivisor]])}
      {table(t('정확한 유리수 양끝점','Exact rational endpoints'),quantities.flatMap(([name,v]) => v ? [[`${name}: lower`,v.lower],[`${name}: upper`,v.upper]] : []))}
      <p>{t('국소 기준은 블록 내부 합을 유지하고 cross edge의 스핀곱을 블록 평균곱으로 대체합니다. 잔차는 정규화한 KL이 아니고 일반적인 부호 제한도 주장하지 않습니다.','The local reference retains internal sums and replaces cross-edge spin products by block-mean products. Its residual is not a normalized KL divergence, and no general sign restriction is asserted.')}</p>
    </details>
    <details><summary>{t('전체 프로필 분포와 미시 상태 수','Full profile distribution and microscopic counts')}</summary>
      <p data-block-profile-page>{t('페이지','Page')} {page+1}/{pages}; {t('프로필 전체','all profiles')}: {profiles.length}. {t('한 페이지25행이며 작은 확률도 생략하지 않습니다.','25 rows per page; small probabilities are retained.')}</p>
      <div className="qubit-actions"><Button variant="outline" disabled={page===0} onClick={() => update({ page: page-1 })}>{t('이전 프로필 페이지','Previous profile page')}</Button><Button variant="outline" disabled={page===pages-1} onClick={() => update({ page: page+1 })}>{t('다음 프로필 페이지','Next profile page')}</Button></div>
      <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('전체 프로필 표 · 좌우 이동 가능','All-profile table; scroll horizontally')}><table className="scalar-table block-order-profiles"><caption>{t('전체 프로필의 정확한 확률','Exact probabilities of all profiles')}</caption><thead><tr><th scope="col">k</th><th scope="col">Ω_k</th><th scope="col">Q_k</th><th scope="col">P(k)</th></tr></thead><tbody>{profiles.slice(page*25,(page+1)*25).map((row,i) => <tr key={page*25+i}><th scope="row"><Button variant="outline" aria-pressed={page*25+i===r.selected.profileIndex} aria-label={`${t('프로필 적용','Apply profile')} k=(${row.upCounts.join(',')})`} data-block-profile-index={page*25+i} onClick={() => onProfile(row.upCounts)}>({row.upCounts.join(', ')})</Button></th><td>{row.multiplicity}</td><td className="block-order-exact">{row.bondPolynomial}</td><td className="block-order-exact">{row.probability}</td></tr>)}</tbody></table></div>
      {table(t('선택 프로필의 불일치 수별 미시 상태 수','Microscopic counts by disagreement number in the selected profile'),selected.disagreementHistogram.map(([d,count]) => [`d=${d}`,count]))}
      {table(t('전체 위쪽 수의 정확한 분포','Exact global up-count distribution'),r.globalUpProbabilities.map((p,k) => [`K=${k}, M=${2*k-r.sites}`,p]))}
      {table(t('블록 자화의 정확한 공분산','Exact block-magnetization covariance'),r.magnetizationCovariance.flatMap((row,a) => row.map((value,b) => [`B${a}, B${b}`,value])))}
    </details>
    <p>{t('이 결과는 유한 이산 모형입니다. 무한 부피 임계점·연속체 극한·재규격화 흐름을 계산한 것은 아닙니다.','This is a finite discrete model. It does not compute an infinite-volume critical point, continuum limit or renormalization flow.')}</p>
  </section>;
}
