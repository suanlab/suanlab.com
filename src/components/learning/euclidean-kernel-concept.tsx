'use client';
import { useId } from 'react';
import { Button } from '@/components/ui/button';
import { euclideanKernelSteps } from '@/data/euclidean-kernel-teaching';
import EuclideanKernelGraphic from './euclidean-kernel-graphic';
import { useSavedState } from './use-saved-state';
const validStep = (value: unknown): value is number => typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < 8;
export default function EuclideanKernelConcept({ language, selectedStep, onStep }: { language: 'ko' | 'en'; selectedStep?: number; onStep?: (step: number) => void }) {
  const id = useId(), [localStep, setLocalStep] = useSavedState('suanlab-learning:euclidean-quantum-kernels-and-time-slicing:concept', 0, validStep), step = Math.max(0, Math.min(7, selectedStep ?? localStep)), setStep = onStep ?? setLocalStep, current = euclideanKernelSteps[step], t = (ko: string, en: string) => language === 'ko' ? ko : en;
  return <section id="visual-intuition" className="ai-intuition" aria-labelledby={id} data-euclidean-concept={step}>
    <h2 id={id}>{t('그림으로 Euclidean 커널 이해하기', 'Understand Euclidean kernels visually')}</h2>
    <p>{t('유한 Gaussian 적분에서 커널·조건부 법칙·열적 trace를 차례로 연결합니다. 정확한 분할 규칙과 연속 기준해의 근사 그림을 구별하세요.', 'Connect finite Gaussian integration to kernels, conditioned laws and thermal traces. Distinguish the exact slicing rule from approximate drawings of the continuous reference.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계', 'Concept steps')}>{euclideanKernelSteps.map((s, i) => <Button key={i} variant="outline" aria-pressed={step === i} onClick={() => setStep(i)}>{i + 1}. {s.title[language]}</Button>)}</div>
    <article className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.body[language]}</p><p className="ai-concept-equation" style={{ overflowWrap: 'anywhere' }}>{current.formula}</p><figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('기준 개념도 · 좌우 이동 가능', 'Reference diagram; scroll horizontally')}><EuclideanKernelGraphic step={step} language={language} /></div><figcaption>{current.diagram[language]}</figcaption></figure></article>
    <div className="qubit-actions"><Button variant="outline" disabled={step === 0} onClick={() => setStep(step - 1)}>{t('이전 단계', 'Previous step')}</Button><Button variant="outline" disabled={step === 7} onClick={() => setStep(step + 1)}>{t('다음 단계', 'Next step')}</Button><a href="#experiment">{t('실험에서 비교하기 →', 'Compare in the experiment →')}</a></div>
  </section>;
}
