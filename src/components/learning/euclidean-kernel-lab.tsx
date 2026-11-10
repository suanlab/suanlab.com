'use client';
import React, { useEffect, useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { euclideanKernelExperiments, euclideanKernelQuestions, euclideanKernelLabels } from '@/data/euclidean-kernel-teaching';
import { euclideanKernelDefault, euclideanKernelExamples, type EuclideanKernelInput } from '@/lib/learning-euclidean-kernel-input';
import { euclideanKernelInputKey } from '@/lib/learning-euclidean-kernel-protocol';
import { euclideanKernelDraft, parseEuclideanKernelDraft } from '@/lib/learning-euclidean-kernel-view';
import { useEuclideanKernelCalculation } from './use-euclidean-kernel-calculation';
import { useSavedEuclideanKernel } from './use-saved-euclidean-kernel';
import EuclideanKernelResultView from './euclidean-kernel-result';
import EuclideanKernelConcept from './euclidean-kernel-concept';
export default function EuclideanKernelLab({ language, embedded = false }: { language: 'ko' | 'en'; embedded?: boolean }) {
  const id = useId(), t = (ko: string, en: string) => language === 'ko' ? ko : en, saved = useSavedEuclideanKernel(euclideanKernelDefault);
  const [running, setRunning] = useState(true), calculation = useEuclideanKernelCalculation(saved.input, saved.ready && running);
  const [draft, setDraft] = useState(() => euclideanKernelDraft(saved.input)), [invalid, setInvalid] = useState(false);
  const { learning } = saved, experiment = learning.experiment, explain = learning.explanationOpen, answers = learning.answers;
  useEffect(() => { setDraft(euclideanKernelDraft(saved.input)); setInvalid(false); }, [saved.input]);
  const choose = (input: EuclideanKernelInput) => { saved.setInput(input); setRunning(true); calculation.retry(); };
  const key = euclideanKernelInputKey(saved.input), example = euclideanKernelExamples.find(e => euclideanKernelInputKey(e.input) === key), guide = euclideanKernelExperiments[experiment];
  const labels = { time: t('총시간 τ', 'Total time τ'), mass: t('질량 m', 'Mass m'), hbar: t('작용 상수 ℏ', 'Action constant hbar'), omegaSquared: t('진동수 제곱 ω²', 'Squared frequency ω²'), start: t('시작값 a', 'Start value a'), end: t('마지막 값 b', 'End value b') };
  return <section id={embedded ? undefined : 'experiment'} className="ai-qubit-lab information-lab scalar-lab" lang={language} aria-labelledby={`${id}-title`} data-euclidean-phase={calculation.status}>
    <h2 id={`${id}-title`}>{t('시간 분할과 Euclidean kernel 실험', 'Explore time slicing and Euclidean kernels')}</h2>
    <p>{t('고정끝 평균·요동·커널 정규화·열적 trace를 비교하세요. 정확한 유한 규칙과 근사 Mehler 기준해를 구별하고 시간 분할 오차를 살펴봅니다.', 'Compare pinned means, fluctuations, kernel normalization and thermal traces. Distinguish the exact finite rule from an approximate Mehler reference to inspect time-slicing error.')}</p>
    {!embedded && <EuclideanKernelConcept language={language} selectedStep={learning.step} onStep={step => saved.setLearning({ ...learning, step })} />}
    <h3>{t('먼저 예측하고 비교하기', 'Predict first, then compare')}</h3>
    <label className="qubit-field" htmlFor={`${id}-guide`}>{t('안내 실험', 'Guided experiment')}<select id={`${id}-guide`} aria-label={t('안내 실험', 'Guided experiment')} value={experiment} onChange={e => { saved.setLearning({ ...learning, experiment: Number(e.target.value), explanationOpen: false }); }}>{euclideanKernelExperiments.map((g, i) => <option value={i} key={i}>{g.title[language]}</option>)}</select></label>
    <p>{guide.predict[language]}</p><div className="qubit-actions" role="group" aria-label={t('안내 예제', 'Guided examples')}>{guide.examples.map(name => <Button key={name} variant="outline" disabled={!saved.ready} aria-pressed={example?.id === name} onClick={() => choose(euclideanKernelExamples.find(e => e.id === name)!.input)}>{euclideanKernelLabels[name][language]}</Button>)}</div>
    <Button variant="outline" aria-expanded={explain} onClick={() => saved.setLearning({ ...learning, explanationOpen: !explain })}>{t('해설 보기', 'Show explanation')}</Button>{explain && <p data-euclidean-guide-explanation>{guide.explanation[language]}</p>}
    <label className="qubit-field" htmlFor={`${id}-example`}>{t('전체 예제', 'All examples')}<select id={`${id}-example`} aria-label={t('전체 예제', 'All examples')} value={example?.id ?? 'custom'} disabled={!saved.ready} onChange={e => { const value = euclideanKernelExamples.find(v => v.id === e.target.value); if (value) choose(value.input); }}><option value="custom" disabled>{t('직접 입력', 'Custom input')}</option>{euclideanKernelExamples.map(e => <option key={e.id} value={e.id}>{euclideanKernelLabels[e.id][language]}</option>)}</select></label>
    {saved.storage !== 'available' && <p role="status" data-euclidean-storage={saved.storage}>{saved.storage === 'corrupt' ? t('저장된 입력 또는 표시 선택이 잘못되어 첫 예제로 복원했습니다.', 'Saved input or display choices were invalid; the first example was restored.') : t('저장할 수 없지만 현재 화면에서는 계속 실험할 수 있습니다.', 'Storage is unavailable; you can continue experimenting in this session.')}</p>}
    <form onSubmit={e => { e.preventDefault(); const input = parseEuclideanKernelDraft(draft); if (input) { choose(input); setInvalid(false); } else setInvalid(true); }}>
      <h3>{t('직접 입력', 'Edit parameters')}</h3>
      <p>{t('구간 수 N은2…32, 관측 첨자는0…N의 정수입니다. τ, m, ℏ는0보다 크고16이하, ω²는0이상16이하이며 양끝 절댓값은16이하입니다. 분자·분모는 각각12자리 이내의 정수·분수로 입력하며 소수·지수 표기는 받지 않습니다.', 'Use integer N=2…32 and probe index0…N. τ, m, hbar must be positive and at most16; ω² must be nonnegative and at most16; endpoint magnitudes at most16. Numerators/denominators have at most12 digits; decimals and exponents are rejected.')}</p>
      <div className="scalar-inputs">
        <label className="qubit-field" htmlFor={`${id}-intervals`}>{t('구간 수 N', 'Interval count N')}<input id={`${id}-intervals`} inputMode="numeric" value={draft.intervals} maxLength={32} onChange={e => setDraft(d => ({ ...d, intervals: e.target.value }))} /></label>
        <label className="qubit-field" htmlFor={`${id}-probe`}>{t('관측 시각 첨자', 'Probe time index')}<input id={`${id}-probe`} inputMode="numeric" value={draft.probe} maxLength={32} onChange={e => setDraft(d => ({ ...d, probe: e.target.value }))} /></label>
        {(['time', 'mass', 'hbar', 'omegaSquared', 'start', 'end'] as const).map(field => <label className="qubit-field" key={field} htmlFor={`${id}-${field}`}>{labels[field]}<input id={`${id}-${field}`} value={draft[field]} maxLength={26} spellCheck={false} autoComplete="off" onChange={e => setDraft(d => ({ ...d, [field]: e.target.value }))} /></label>)}
      </div>
      <p>{t('N을 바꾸면 관측 첨자도 범위에 맞추세요. 이 입력은 정련 비교의 같은 물리 시각을 정합니다. 결과의 표시 슬라이더는 새 계산을 만들지 않습니다. 적용 전 결과는 마지막 적용 입력의 값입니다.', 'Keep the probe in range when changing N; it selects a fixed physical time for refinement comparisons. The result display slider creates no new calculation. Until applying, results belong to the last applied input.')}</p>
      {invalid && <p role="alert">{t('구간 수·관측 첨자·유리수 형식과 범위를 확인하세요. 아래 결과는 이전에 적용한 입력의 값입니다.', 'Check interval count, probe index, rational syntax and bounds. Results below belong to the previously applied input.')}</p>}
      <div className="qubit-actions"><Button type="submit" disabled={!saved.ready}>{t('입력 적용', 'Apply input')}</Button><Button type="button" variant="outline" onClick={() => choose(euclideanKernelDefault)}>{t('첫 예제로 초기화', 'Reset to first example')}</Button></div>
    </form>
    <div className="qubit-actions">{calculation.status === 'pending' && <Button variant="outline" onClick={() => { setRunning(false); calculation.cancel(); }}>{t('계산 취소', 'Cancel calculation')}</Button>}{(calculation.error || !running) && <Button onClick={() => { setRunning(true); calculation.retry(); }}>{t('다시 계산', 'Recalculate')}</Button>}</div>
    <p role={calculation.error ? 'alert' : 'status'} aria-live="polite">{calculation.error ? t('계산을 마치지 못했습니다. 다시 계산하거나 예제를 선택하세요.', 'Calculation could not finish. Recalculate or select an example.') : !running ? t('계산을 취소했습니다.', 'Calculation cancelled.') : calculation.status === 'ready' ? t('현재 입력의 계산과 독립 검산을 완료했습니다.', 'Calculation and independent checks for the current input are complete.') : calculation.phase === 'validating' ? t('결과를 독립 검산하고 있습니다.', 'Independently checking the result.') : t('현재 입력을 계산하고 있습니다.', 'Calculating the current input.')}</p>
    {calculation.response && <EuclideanKernelResultView key={key} result={calculation.response.result} language={language} display={saved.display} onDisplay={saved.setDisplay} />}
    {!embedded && <><h3>{t('이해 확인', 'Check your understanding')}</h3>{euclideanKernelQuestions.map((q, i) => <fieldset key={i} className="information-question"><legend>{i + 1}. {q.prompt[language]}</legend>{q.options.map((option, j) => <label key={j}><input type="radio" name={`${id}-question-${i}`} checked={answers[i] === j} onChange={() => saved.setLearning({ ...learning, answers: answers.map((answer, index) => index === i ? j : answer) })} />{option[language]}</label>)}{answers[i] >= 0 && <p role="status"><strong>{answers[i] === q.correct ? t('맞습니다. ', 'Correct. ') : t('다시 생각해보세요. ', 'Think again. ')}</strong>{q.explanation[language]}</p>}</fieldset>)}</>}
  </section>;
}
