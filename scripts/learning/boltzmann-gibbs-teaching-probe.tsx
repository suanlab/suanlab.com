/** Isolated teaching UI: real providers, controls and worker calculations. */
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LanguageProvider, useLanguage } from '../../src/components/language-provider';
import { SiteMotionProvider, MotionControls } from '../../src/components/motion/motion-provider';
import BoltzmannGibbsLab from '../../src/components/learning/boltzmann-gibbs-lab';
import { BoltzmannConceptView } from '../../src/components/learning/boltzmann-gibbs-concept';
import LearningQuiz from '../../src/components/learning/learning-quiz';
import { boltzmannConceptSteps, boltzmannConceptQuestions } from '../../src/lib/learning-boltzmann-gibbs-concepts';
import { boltzmannGibbsKinds as kinds, type BoltzmannGibbsKind } from '../../src/lib/learning-boltzmann-gibbs';

const metrics = { started: 0, terminated: 0, live: 0, blocked: false };
const NativeWorker = window.Worker;
window.Worker = class extends NativeWorker {
  private stopped = false;
  constructor(_url: string | URL, options?: WorkerOptions) {
    if (metrics.blocked) throw new Error('Test-only worker construction failure');
    super(new URL('./worker.js', location.href), options); metrics.started++; metrics.live++;
  }
  terminate() { if (!this.stopped) { this.stopped = true; metrics.terminated++; metrics.live--; } super.terminate(); }
};
function Shell() {
  const { language, setLanguage } = useLanguage();
  const [kind, setKind] = useState<BoltzmannGibbsKind>(kinds[0]), [step, setStep] = useState(0), [mounted, setMounted] = useState(true);
  Object.assign(window, { boltzmannTeaching: { metrics, kinds, setKind, setStep, setLanguage, setMounted,
    setTheme: (theme: string) => document.documentElement.classList.toggle('dark', theme === 'dark') } });
  const concept = boltzmannConceptSteps[kind][step];
  return <div className="ai-book"><main id="main-content" style={{ maxWidth: 1100, margin: 'auto', padding: 16 }}>
    <h1>{language === 'ko' ? 'Boltzmann·Gibbs 교재 검증' : 'Boltzmann–Gibbs teaching verification'}</h1>
    <MotionControls />
    {mounted && <>
      <section aria-labelledby="concept-title"><h2 id="concept-title">{concept.title[language]}</h2><p>{concept.explanation[language]}</p>
        <BoltzmannConceptView kind={kind} step={step} language={language} />
      </section>
      <BoltzmannGibbsLab key={`lab:${kind}`} kind={kind} />
      <LearningQuiz key={`quiz:${kind}`} unit={{ id: kind, questions: [...boltzmannConceptQuestions[kind]] }} />
    </>}
  </main></div>;
}
createRoot(document.getElementById('root')!).render(<LanguageProvider><SiteMotionProvider><Shell /></SiteMotionProvider></LanguageProvider>);
