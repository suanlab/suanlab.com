'use client';
import { useId } from 'react';
import { Button } from '@/components/ui/button';
import { saddleFluctuationSteps } from '@/data/saddle-fluctuation-teaching';
import SaddleFluctuationGraphic from './saddle-fluctuation-graphic';
import { useSavedState } from './use-saved-state';
const validStep = (value: unknown): value is number => typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < 10;
export default function SaddleFluctuationConcept({ language, selectedStep, onStep }: { language: 'ko' | 'en'; selectedStep?: number; onStep?: (step: number) => void }) {
  const id = useId(), [localStep, setLocalStep] = useSavedState('suanlab-learning:saddle-fluctuation-and-nongaussian-approximation:concept', 0, validStep), step = Math.max(0, Math.min(9, selectedStep ?? localStep)), setStep = onStep ?? setLocalStep, current = saddleFluctuationSteps[step], t = (ko: string, en: string) => language === 'ko' ? ko : en;
  return <section id="visual-intuition" className="ai-intuition" aria-labelledby={id} data-saddle-concept={step}>
    <h2 id={id}>{t('그림으로 안장점·요동·비가우스 근사 이해하기', 'Understand saddles, fluctuations and non-Gaussian approximations visually')}</h2>
    <p>{t('정상점·잔차 인증·Gaussian 폭과 보정·실제 경계의 조건·여러 우물과 퇴화를 차례로 연결합니다. 기준 개념도와 현재 입력의 계산 결과를 구별하세요.', 'Connect stationary points, residual certificates, Gaussian widths and corrections, conditional ratio bounds, multiple wells and degeneracy. Distinguish these reference diagrams from results for your current input.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계', 'Concept steps')}>{saddleFluctuationSteps.map((s, i) => <Button key={i} variant="outline" aria-pressed={step === i} onClick={() => setStep(i)}>{i + 1}. {s.title[language]}</Button>)}</div>
    <article className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.body[language]}</p><p className="ai-concept-equation" style={{ overflowWrap: 'anywhere' }}>{current.formula}</p><figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('기준 개념도 · 좌우 이동 가능', 'Reference diagram; scroll horizontally')}><SaddleFluctuationGraphic step={step} language={language} /></div><figcaption>{current.diagram[language]}</figcaption></figure></article>
    <div className="qubit-actions"><Button variant="outline" disabled={step === 0} onClick={() => setStep(step - 1)}>{t('이전 단계', 'Previous step')}</Button><Button variant="outline" disabled={step === 9} onClick={() => setStep(step + 1)}>{t('다음 단계', 'Next step')}</Button><a href="#experiment">{t('실험에서 비교하기 →', 'Compare in the experiment →')}</a></div>
  </section>;
}
