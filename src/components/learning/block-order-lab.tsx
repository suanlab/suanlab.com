'use client';
import React, { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { blockOrderLabels, blockOrderExperiments, blockOrderQuestions } from '@/data/block-order-teaching';
import { blockOrderDefault, blockOrderExamples, type BlockOrderInput } from '@/lib/learning-block-order-input';
import { blockOrderInputKey } from '@/lib/learning-block-order-protocol';
import { blockOrderDraft, parseBlockOrderDraft } from '@/lib/learning-block-order-view';
import { useBlockOrderCalculation } from './use-block-order-calculation';
import { useSavedBlockOrder } from './use-saved-block-order';
import BlockOrderConcept from './block-order-concept';
import BlockOrderResultView from './block-order-result';
import '@/styles/block-order-lab.css';

export default function BlockOrderLab({ language,embedded=false }: { language: 'ko' | 'en'; embedded?: boolean }) {
  const id = useId(), t = (ko: string,en: string) => language === 'ko' ? ko : en, saved = useSavedBlockOrder(), { model,ready } = saved;
  const { input,draft,guide,explain,answers } = model, [running,setRunning] = useState(true), [invalid,setInvalid] = useState(false);
  const calculation = useBlockOrderCalculation(input,ready && running), key = blockOrderInputKey(input), currentGuide = blockOrderExperiments[guide];
  const selectedExample = blockOrderExamples.find(e => blockOrderInputKey(e.input) === key);
  const choose = (value: BlockOrderInput,keepPage=false) => {
    saved.update(p => ({ ...p,input: value,draft: blockOrderDraft(value),display: { ...p.display,example: 0,page: keepPage ? p.display.page : 0 } }));
    setInvalid(false); setRunning(true); calculation.retry();
  };
  const fields = [
    ['shape',t('격자 축 길이','Grid axis lengths'),12],['blockShape',t('블록 축 길이','Block axis lengths'),12],
    ['disagreementActivity',t('불일치 활동도 x','Disagreement activity x'),13],['blockUpActivities',t('블록별 위쪽 활동도 y','Blockwise up activities y'),64],
    ['selectedUpCounts',t('선택 프로필의 위쪽 수 k','Selected profile up-counts k'),20],['witnessPair',t('검사할 블록 쌍','Block pair to test'),12],
  ] as const;
  return <section className="ai-qubit-lab scalar-lab information-lab block-order-lab" lang={language} aria-labelledby={`${id}-title`} data-block-status={calculation.status} data-block-storage-ready={ready ? 'true' : 'false'} data-block-input-key={key}>
    <h2 id={`${id}-title`}>{t('미시상태에서 국소 질서변수로','From microscopic states to local order parameters')}</h2>
    <p>{t('같은 평균 안에 어떤 정보가 숨어 있을까요? 실제 공간 결합의 가중치를 정확하게 더하고, 남긴 블록 변수가 어떤 확률법을 따르는지 비교합니다.','What information hides behind an average? Sum the weights of spatially interacting spins exactly and compare the statistical law of the retained block variables.')}</p>
    {saved.storage !== 'available' && <p role="status" data-block-storage={saved.storage}>{saved.storage === 'corrupt' ? t('저장 형식이 잘못되어 기본값으로 복원했습니다.','Saved state was invalid; defaults were restored.') : t('브라우저에 저장할 수 없습니다. 현재 화면에서 계속 실험할 수 있습니다.','Storage is unavailable. Continue experimenting in this session.')}</p>}
    <fieldset disabled={!ready} className="block-order-controls"><legend className="sr-only">{t('모형 입력과 학습 활동','Model inputs and learning activities')}</legend>
      {!embedded && <BlockOrderConcept language={language} step={model.step} onStep={step => saved.update(p => ({ ...p,step }))} motion={model.motion} onMotion={motion => saved.update(p => ({ ...p,motion }))}/>}
      <h3>{t('먼저 예측하고 비교하기','Predict first, then compare')}</h3>
      <label className="qubit-field" htmlFor={`${id}-guide`}>{t('안내 실험','Guided experiment')}<select id={`${id}-guide`} aria-label={t('안내 실험','Guided experiment')} value={guide} onChange={e => saved.update(p => ({ ...p,guide: Number(e.target.value),explain: false }))}>{blockOrderExperiments.map((g,i) => <option key={i} value={i}>{g.title[language]}</option>)}</select></label>
      <p>{currentGuide.predict[language]}</p>
      <div className="qubit-actions" role="group" aria-label={t('안내 예제','Guided examples')}>{currentGuide.examples.map(example => <Button key={example} variant="outline" aria-pressed={selectedExample?.id === example} onClick={() => choose(blockOrderExamples.find(e => e.id === example)!.input)}>{blockOrderLabels[example][language]}</Button>)}</div>
      <Button variant="outline" aria-expanded={explain} onClick={() => saved.update(p => ({ ...p,explain: !p.explain }))}>{t('해설 보기','Show explanation')}</Button>
      {explain && <p data-block-guide-explanation>{currentGuide.explanation[language]}</p>}
      <label className="qubit-field" htmlFor={`${id}-all`}>{t('전체 예제','All examples')}<select id={`${id}-all`} aria-label={t('전체 예제','All examples')} value={selectedExample?.id ?? 'custom'} onChange={e => { const value = blockOrderExamples.find(ex => ex.id === e.target.value); if (value) choose(value.input); }}><option value="custom" disabled>{t('직접 입력','Custom input')}</option>{blockOrderExamples.map(e => <option key={e.id} value={e.id}>{blockOrderLabels[e.id][language]}</option>)}</select></label>
      <form onSubmit={e => { e.preventDefault(); const parsed = parseBlockOrderDraft(draft); if (parsed) choose(parsed); else setInvalid(true); }}>
        <h3>{t('공간·가중치·프로필 직접 입력','Edit geometry, weights and profile')}</h3>
        <p>{t('1/2차원·전체16스핀 이하·동일 직사각 블록4개 이하입니다. 각 축은2 이상, 블록 변은 해당 격자 축을 나누어야 합니다. 벡터는 쉼표로 구분하세요.','Use one or two dimensions, at most16 spins and four equal rectangular blocks. Each axis has at least2 sites; block sides must divide grid sides. Separate vector entries with commas.')}</p>
        <p>{t('x=exp(−2J/T), y_b=exp(2H_b/T)는 양의 정수 또는 분수이며 분자·분모 각각6자리 이하입니다.0·소수·지수 표기는 받지 않습니다. x<1은 강자성, x>1은 반강자성입니다. x는 온도 자체가 아닙니다.','Activities x=exp(−2J/T), y_b=exp(2H_b/T) are positive integers or fractions with at most six digits per numerator and denominator. Zero, decimals and exponent notation are rejected. x<1 is ferromagnetic and x>1 antiferromagnetic. x is not temperature itself.')}</p>
        <div className="scalar-inputs">{fields.map(([field,label,length]) => <label className="qubit-field" key={field} htmlFor={`${id}-${field}`}>{label}<input id={`${id}-${field}`} value={draft[field]} maxLength={length} autoComplete="off" spellCheck={false} onChange={e => saved.update(p => ({ ...p,draft: { ...p.draft,[field]: e.target.value } }))}/></label>)}
          <label className="qubit-field" htmlFor={`${id}-boundary`}>{t('경계조건','Boundary condition')}<select id={`${id}-boundary`} aria-label={t('경계조건','Boundary condition')} value={draft.boundary} onChange={e => saved.update(p => ({ ...p,draft: { ...p.draft,boundary: e.target.value as BlockOrderInput['boundary'] } }))}><option value="open">{t('열린 경계','Open')}</option><option value="periodic">{t('주기 경계','Periodic')}</option></select></label>
        </div>
        <p>{t('예: 격자4, 블록2, y는1,1, k는1,1, 검사 쌍은0,1입니다. 한 블록이면 검사 쌍을 비우세요. k는 블록당0…v이며 쌍은 서로 다른 오름차순 블록 번호입니다. 길이2 주기 축의 평행 edge는 두 개를 유지합니다.','Example: grid4, block2, y=1,1, k=1,1, pair=0,1. Leave the pair empty for one block. Each k is0…v; pairs use distinct ascending block indices. Parallel edges on a periodic axis of length2 are both retained.')}</p>
        <p>{t('적용 전에는 마지막으로 적용한 모형의 결과를 보여줍니다. 적용 입력·미완성 편집·학습 선택은 브라우저에 저장하고 계산 결과는 다시 검산합니다.','Until application, results belong to the last applied model. Applied input, unfinished drafts and learning choices are saved in this browser; results are recomputed and checked.')}</p>
        {invalid && <p role="alert">{t('분수·격자와 블록 크기·벡터 길이·k 범위·검사 쌍을 확인하세요. 결과는 이전 적용 입력의 값입니다.','Check fractions, grid and block sizes, vector lengths, k ranges and pair indices. Results still belong to the previous applied input.')}</p>}
        <div className="qubit-actions"><Button type="submit">{t('입력 적용','Apply input')}</Button><Button type="button" variant="outline" onClick={() => choose(blockOrderDefault)}>{t('첫 예제로 초기화','Reset to first example')}</Button></div>
      </form>
      <div className="qubit-actions">{calculation.status === 'pending' && <Button variant="outline" onClick={() => { setRunning(false); calculation.cancel(); }}>{t('계산 취소','Cancel calculation')}</Button>}{(calculation.error || !running) && <Button onClick={() => { setRunning(true); calculation.retry(); }}>{t('다시 계산','Recalculate')}</Button>}</div>
      <p role={calculation.error ? 'alert' : 'status'} aria-live="polite">{!ready ? t('저장한 상태를 복원합니다.','Restoring saved state.') : calculation.error ? t('계산을 완료하지 못했습니다. 다시 계산하거나 다른 예제를 선택하세요.','Calculation could not finish. Recalculate or choose another example.') : !running ? t('계산을 취소했습니다.','Calculation cancelled.') : calculation.status === 'ready' ? t('현재 입력의 계산과 독립 검산을 완료했습니다.','Calculation and independent checks for the current input are complete.') : calculation.phase === 'validating' ? t('결과를 독립 검산하고 있습니다.','Independently checking the result.') : t('현재 입력을 계산하고 있습니다.','Calculating the current input.')}</p>
      {calculation.response && <BlockOrderResultView key={key} result={calculation.response.result} input={input} language={language} display={model.display} onDisplay={change => saved.update(p => ({ ...p,display: typeof change === 'function' ? change(p.display) : change }))} onProfile={upCounts => choose({ ...input,selectedUpCounts: upCounts },true)}/>}
      {!embedded && <><h3>{t('이해 확인과 해설','Check your understanding')}</h3>{blockOrderQuestions.map((q,i) => <fieldset className="information-question" key={i}><legend>{i+1}. {q.prompt[language]}</legend>{q.options.map((option,j) => <label key={j}><input type="radio" name={`${id}-quiz-${i}`} checked={answers[i] === j} onChange={() => saved.update(p => ({ ...p,answers: p.answers.map((a,k) => k === i ? j : a) }))}/>{option[language]}</label>)}{answers[i] !== null && <p role="status" data-block-answer={answers[i] === q.correct ? 'correct' : 'incorrect'}><strong>{answers[i] === q.correct ? t('맞습니다. ','Correct. ') : t('다시 생각해보세요. ','Think again. ')}</strong>{q.explanation[language]}</p>}</fieldset>)}</>}
    </fieldset>
  </section>;
}
