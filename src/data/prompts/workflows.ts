import type { PromptWorkflow } from './index';

// 연구 작업을 빌더 체인으로 엮은 워크플로우 템플릿. 단계의 builderId는 builders.ts의 id와 일치.
export const promptWorkflows: PromptWorkflow[] = [
  {
    id: 'wf-paper',
    title: { ko: '논문 투고 파이프라인', en: 'Paper Submission Pipeline' },
    description: {
      ko: '아이디어 → 실험 설계 → 작성 → 자가 리뷰 → 학회 전략의 전 주기.',
      en: 'Idea → experiment → writing → self-review → venue strategy.',
    },
    icon: 'target',
    steps: [
      { builderId: 'idea', note: { ko: '방향과 문헌으로 연구 기회를 좁힌다', en: 'Narrow the research opportunity' } },
      { builderId: 'experiment', note: { ko: '가설을 검증할 실험 플랜 설계', en: 'Design experiments for the hypothesis' } },
      { builderId: 'writing', note: { ko: '결과를 학회 규격 LaTeX으로 작성', en: 'Write up in venue LaTeX style' } },
      { builderId: 'paper-review', note: { ko: '제출 전 다중 페르소나로 자가 리뷰', en: 'Self-review with multi-persona before submission' } },
      { builderId: 'conference', note: { ko: '투고처·타이밍·포지셔닝 전략 수립', en: 'Decide venue, timing, positioning' } },
    ],
  },
  {
    id: 'wf-debug',
    title: { ko: '디버그 & 재현', en: 'Debug & Reproduce' },
    description: {
      ko: '에러 진단 → 최소 재현 → 실험으로 검증.',
      en: 'Diagnose → minimal repro → verify with an experiment.',
    },
    icon: 'shield-check',
    steps: [
      { builderId: 'error-analysis', note: { ko: '에러/로그로 근본 원인 가설', en: 'Form root-cause hypotheses' } },
      { builderId: 'code-gen', note: { ko: '최소 재현 예제(MCVE) 구현', en: 'Build a minimal repro' } },
      { builderId: 'experiment', note: { ko: '수정 전후를 비교하는 검증 실험', en: 'A/B experiment to confirm the fix' } },
    ],
  },
  {
    id: 'wf-data',
    title: { ko: '데이터 프로젝트', en: 'Data Project' },
    description: {
      ko: '데이터셋 설계 → 정제 → EDA → 모델링 실험.',
      en: 'Dataset design → cleaning → EDA → modeling.',
    },
    icon: 'database',
    steps: [
      { builderId: 'dataset-design', note: { ko: '스키마·품질·평가 프로토콜 설계', en: 'Design schema, QA, eval protocol' } },
      { builderId: 'data-cleaning', note: { ko: '결측/이상치/누수 방지 정제', en: 'Clean with leakage prevention' } },
      { builderId: 'experiment', note: { ko: 'EDA 결과로 가설 세우고 실험', en: 'Turn EDA into hypotheses & experiments' } },
    ],
  },
  {
    id: 'wf-learn',
    title: { ko: '개념 학습 → 정착', en: 'Learn & Solidify' },
    description: {
      ko: '개념 설명 → 문헌 조사 → 자가 점검 퀴즈.',
      en: 'Concept explanation → literature → self-quiz.',
    },
    icon: 'graduation-cap',
    steps: [
      { builderId: 'explain', note: { ko: '수준에 맞춰 개념 학습', en: 'Learn at your level' } },
      { builderId: 'literature-review', note: { ko: '주변 문헌으로 맥락 파악', en: 'Context via related literature' } },
      { builderId: 'idea', note: { ko: '학습을 연구 질문으로 발전', en: 'Grow learning into research questions' } },
    ],
  },
  {
    id: 'wf-rebuttal',
    title: { ko: '리뷰 대응 / Rebuttal', en: 'Review Response' },
    description: {
      ko: '논문 자가 리뷰로 약점 파악 → 리뷰어 관점 Rebuttal 작성.',
      en: 'Find weaknesses via self-review → write the rebuttal.',
    },
    icon: 'message-square-reply',
    steps: [
      { builderId: 'paper-review', note: { ko: '리뷰어가 공격할 약점을 미리 파악', en: 'Anticipate reviewer attacks' } },
      { builderId: 'rebuttal', note: { ko: '실제 리뷰에 대한 반박 전략 수립', en: 'Build the rebuttal strategy' } },
    ],
  },
];
