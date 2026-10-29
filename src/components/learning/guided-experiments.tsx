'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { learningGuides, type GuidedExperiment, type LearningUnitId } from '@/data/interactive-book';
export default function GuidedExperiments({ kind, ready, onLoad }: { kind: LearningUnitId; ready: boolean; onLoad: (guide: GuidedExperiment) => void }) {
  const { language } = useLanguage(); const ko = language === 'ko'; const [selected, setSelected] = useState(0);
  const experiments = learningGuides[kind].experiments; const guide = experiments[selected];
  return <details className="ai-guided-lab" open><summary>{ko ? '안내를 따라 실험하기' : 'Follow a guided experiment'}</summary>
    <div className="ai-guided-choices" role="group" aria-label={ko ? '안내 실험 선택' : 'Choose a guided experiment'}>{experiments.map((item, index) => <Button type="button" variant="outline" size="sm" key={index} aria-pressed={selected === index} onClick={() => setSelected(index)}>{index + 1}. {item.title[language]}</Button>)}</div>
    <h3>{guide.title[language]}</h3><p><strong>{ko ? '먼저 예측' : 'Predict first'} · </strong>{guide.predict[language]}</p><p><strong>{ko ? '조작 방법' : 'Try it'} · </strong>{guide.action[language]}</p>
    <Button type="button" variant="outline" size="sm" disabled={!ready} onClick={() => onLoad(guide)}>{ko ? '이 예제 열기' : 'Load this example'}</Button>
    <details key={selected} className="ai-guided-answer"><summary>{ko ? '결과 해석 보기' : 'Reveal the explanation'}</summary><p>{guide.explain[language]}</p></details>
    <p className="ai-small">{ko ? '예제를 열면 현재 실험 설정을 해당 값으로 바꿉니다. 아래에서 자유롭게 수정할 수 있습니다.' : 'Loading replaces the current experiment settings with this example. You can edit them freely below.'}</p>
  </details>;
}
