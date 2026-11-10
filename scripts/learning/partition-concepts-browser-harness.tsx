import React, { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LanguageProvider, useLanguage } from '../../src/components/language-provider';
import { PartitionConceptView } from '../../src/components/learning/partition-concept';
import PartitionCover from '../../src/components/learning/partition-cover';
import { partitionKinds, type PartitionKind } from '../../src/lib/learning-partition';
Object.assign(window, { partitionConceptKinds: partitionKinds });
function Harness() {
  const [kind, setKind] = useState<PartitionKind>(partitionKinds[0]), [step, setStep] = useState(0), { language, setLanguage } = useLanguage();
  return <div className="ai-book"><nav aria-label="Verification controls" style={{ padding: '1rem', display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
    {partitionKinds.map((k, i) => <button type="button" data-testid={`kind-${i}`} key={k} onClick={() => { setKind(k); setStep(0); }}>Family {i + 1}</button>)}
    {[0, 1, 2, 3].map(s => <button type="button" data-testid={`step-${s}`} key={s} onClick={() => setStep(s)}>Step {s + 1}</button>)}
    <button type="button" data-testid="language-ko" onClick={() => setLanguage('ko')}>KO</button><button type="button" data-testid="language-en" onClick={() => setLanguage('en')}>EN</button></nav>
    <main id="main-content" style={{ width: 'min(100%, 800px)', padding: '.5rem', margin: 'auto' }}><h1>{language === 'ko' ? '고정 개념 검증' : 'Fixed concept verification'}</h1><PartitionCover kind={kind} /><PartitionConceptView kind={kind} step={step} language={language} /></main></div>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><LanguageProvider><Harness /></LanguageProvider></StrictMode>);
