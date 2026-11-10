'use client';
import React, { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { freeEnergyTeaching } from '@/data/interactive-book';
import type { FreeEnergyKind } from '@/lib/learning-free-energy';
import { FreeEnergyConceptView } from './free-energy-lab';
import LearningQuiz from './learning-quiz';
export function FreeEnergyConcepts({ kind }: { kind: FreeEnergyKind }) {
  const { language } = useLanguage(), [step, setStep] = useState(0), id = useId();
  const guide = freeEnergyTeaching[kind].guide, current = guide.steps[step];
  return <section className="ai-concept" aria-labelledby={`${id}-title`}><h2 id={`${id}-title`}>{language === 'ko' ? '개념을 단계별로 살펴보기' : 'Explore the concept step by step'}</h2>
    <p>{guide.intuition[language]}</p><div className="ai-concept-steps" role="group" aria-label={language === 'ko' ? '개념 단계 선택' : 'Choose a concept step'}>{guide.steps.map((item, i) => <Button key={i} type="button" variant="outline" aria-pressed={step === i} onClick={() => setStep(i)}>{i + 1}. {item.title[language]}</Button>)}</div>
    <div className="ai-concept-explanation"><h3>{current.title[language]}</h3><p>{current.explanation[language]}</p><p className="ai-concept-equation">{current.formula}</p></div>
    <FreeEnergyConceptView kind={kind} step={step} language={language} />
  </section>;
}
export function FreeEnergyQuiz({ kind }: { kind: FreeEnergyKind }) { return <LearningQuiz unit={freeEnergyTeaching[kind].unit} />; }
