'use client';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { brownianBridgeSteps } from '@/data/brownian-bridge-teaching';
import BrownianBridgeGraphic from './brownian-bridge-graphic';
export default function BrownianBridgeConcept({ language, selectedStep, onStep }: { language: 'ko' | 'en'; selectedStep?: number; onStep?: (step: number) => void }) {
  const id = useId(), [localStep, setLocalStep] = useState(0), step = Math.max(0, Math.min(7, selectedStep ?? localStep)), setStep = onStep ?? setLocalStep, current = brownianBridgeSteps[step], t = (ko: string, en: string) => language === 'ko' ? ko : en;
  return <section id="visual-intuition" className="ai-intuition" aria-labelledby={id} data-brownian-concept={step}>
    <h2 id={id}>{t('그림으로 시간 분할과 bridge 이해하기', 'Understand time slicing and bridges visually')}</h2>
    <p>{t('고정된 양끝 사이의 불확실성을 유한 적분에서 시작해 살펴보세요. 같은 시각의 법칙과 격자에 의존하는 관측량을 구분합니다.', 'Start with finite integrals for uncertainty between fixed endpoints. Separate laws at the same times from observables that depend on the grid.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계', 'Concept steps')}>{brownianBridgeSteps.map((s, i) => <Button key={i} variant="outline" aria-pressed={step === i} onClick={() => setStep(i)}>{i + 1}. {s.title[language]}</Button>)}</div>
    <article className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.body[language]}</p><p className="ai-concept-equation" style={{ overflowWrap: 'anywhere' }}>{current.formula}</p><figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('기준 개념도 · 좌우 이동 가능', 'Reference diagram; scroll horizontally')}><BrownianBridgeGraphic step={step} language={language} /></div><figcaption>{current.diagram[language]}</figcaption></figure></article>
    <div className="qubit-actions"><Button variant="outline" disabled={step === 0} onClick={() => setStep(step - 1)}>{t('이전 단계', 'Previous step')}</Button><Button variant="outline" disabled={step === 7} onClick={() => setStep(step + 1)}>{t('다음 단계', 'Next step')}</Button><a href="#experiment">{t('실험에서 비교하기 →', 'Compare in the experiment →')}</a></div>
  </section>;
}
