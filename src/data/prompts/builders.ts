import type { PromptBuilder } from './index';

// ───────────────────────────────────────────────────────────
// 조회 맵 (lookup maps) — 원본 AI-Research-Prompt-Toolkit에서 이식
// ───────────────────────────────────────────────────────────

const RI: Record<string, string> = {
  researcher: '당신은 동일 분야의 경험 많은 연구자 동료로서 솔직하고 실질적인 관점에서 조언해 주세요.',
  critic: '당신은 냉정한 peer reviewer입니다. 현재 접근법의 약점과 결함을 먼저 지적한 뒤 개선 방향을 제시해 주세요.',
  inventor: '당신은 기존 방법에 얽매이지 않는 창의적 발명가 사고로 비관습적 해결책을 우선 탐색해 주세요.',
  phd: '당신은 엄격한 PhD 어드바이저로서 이 문제를 학술 논문으로 풀어낼 수 있는 연구 방향을 제시해 주세요.',
};

const DC: Record<string, string> = {
  ML: 'ML(머신러닝)', DL: '딥러닝(DL)', NLP: 'NLP(자연어처리)', CV: '컴퓨터 비전(CV)',
  RL: '강화학습(RL)', GNN: '그래프 신경망(GNN)', GenAI: '생성형 AI',
  AutoML: 'AutoML/하이퍼파라미터 최적화', Theory: '이론 ML',
};

const BC: Record<string, string> = {
  perf: '성능이 특정 지점에서 정체되는 성능 천장(performance ceiling) 문제',
  theory: '수렴 보장이나 표현력에 관한 이론적 한계',
  compute: '현실적인 계산 자원 내에서 실행 불가능한 계산 비용 문제',
  data: '학습 데이터 부족 또는 데이터 품질 문제',
  overfit: '훈련 셋에만 특화되어 일반화가 실패하는 과적합 문제',
  unstable: '학습 과정에서의 불안정성(발산, NaN, 폭발적 기울기 등)',
  arch: '적합한 모델 아키텍처를 선택하거나 설계하는 어려움',
  repro: '실험 결과의 재현 불가 또는 일관성 없는 결과',
};

const SL: Record<string, string> = {
  analogy: '타 분야(물리학, 생물학, 통계학 등)와의 유추를 통해 새로운 시각을 제시하고',
  first_principles: '제1원리(first principles)로 문제를 분해하여 근본 원인을 파악하고',
  literature: '관련 최신 논문(2022-2025)에서 유사 문제의 해결 사례를 찾아 적용 가능성을 검토하고',
  ablation: '체계적인 ablation study 설계를 통해 핵심 변수를 격리하고',
  math: '수학적으로 문제를 정식화하여 이론적 하한/상한을 분석하고',
  experiment: '대조 실험 계획을 수립하여 가설을 검증할 수 있는 최소 실험 단위를 제안하고',
  critique: '현재 접근법의 가정(assumption)들을 비판적으로 검토하여 숨겨진 편향을 찾아내고',
  alternative: '완전히 다른 아키텍처 또는 알고리즘 패러다임을 제안하고',
  hypothesis: '반증 가능한(falsifiable) 구체적 가설들을 생성하고',
};

const DI: Record<string, string> = {
  deep: '각 분석은 표면적 해결책이 아니라 근본 메커니즘 수준에서 심층적으로 다루어 주세요.',
  quick: '핵심 아이디어만 빠르게 10개 이내로 정리해 주세요. 각 아이디어는 1-2문장으로.',
  structured: '로드맵 형태로 체계적으로 정리해 주세요. 단기(1주), 중기(1개월), 장기(3개월) 액션으로 구분해 주세요.',
};

const FI: Record<string, string> = {
  steps: '실행 가능한 구체적 단계(Step 1, Step 2...)로 제시해 주세요.',
  hypotheses: '검증 가능한 가설 형태(H1: ~라면 ~일 것이다)로 5개 이상 제시하고 최소 실험을 함께 제안해 주세요.',
  experiments: '실험 설계서 형태로: 독립변수, 종속변수, 통제변수, 예상 결과, 실패 시 대안을 포함해 주세요.',
  code: '핵심 아이디어는 Python/PyTorch 슈도코드 또는 실제 코드로 함께 보여 주세요.',
};

const CL: Record<string, string> = {
  c1: '환각 함수/라이브러리 확인 — 존재하지 않는 API 사용 여부',
  c2: '버전 호환성 검증 — deprecated 문법, 패키지 버전 명시 여부',
  c3: '보안 취약점 점검 — 하드코딩된 키, SQL injection, 미검증 입력',
  c4: '엣지 케이스 처리 — 예외, 경계값, null/undefined 핸들링',
  c5: '로직 오류 검증 — 겉으론 실행되지만 틀린 결과를 내는 코드',
  c6: '테스트 코드 실효성',
  c7: '코드 가독성/일관성',
  c8: '과잉 엔지니어링 여부',
  e1: '데이터 누수(Data Leakage) — train/test 분리, 전처리 순서 오류',
  e2: '통계 설계 — 샘플 크기, 검정력, 유의수준 적절성',
  e3: '재현성 확보 — 랜덤 시드 고정, 환경/버전 명시 여부',
  e4: '베이스라인 적절성',
  e5: '결과 해석 정확성 — 지표 의미, 인과/상관 혼동 여부',
  e6: '확증 편향',
  e7: '하이퍼파라미터 튜닝 — 테스트셋 간접 과적합 여부',
  e8: '도메인 프로토콜',
  p1: '인용 검증 — 실제 존재하는 논문인지, DOI/링크 확인',
  p2: '방법론 일치성 — 실제 구현과 논문 Methods 섹션이 일치하는지',
  p3: '연구 윤리 — AI 사용 여부 공시, 저자 기여 명시',
  p4: '결과 미화 — 부정적 결과 축소, 과도한 일반화 표현',
  p5: '논리 일관성 — Introduction과 Conclusion의 주장이 정합하는지',
  p6: 'Limitation 섹션',
  p7: '문헌 최신성',
  p8: '저널 투고 규정',
};

const SEV: Record<string, string> = {
  strict: '의심되는 모든 항목을 표시해 주세요. 확실하지 않더라도 ⚠️로 표기하고 이유를 밝혀 주세요.',
  balanced: '중요도가 높은 문제를 중심으로 판정해 주세요.',
  light: '치명적 결함(❌)만 표시해 주세요.',
};

const AOUT: Record<string, string> = {
  table: '영역(코드/실험/논문)별로 구분하여 각 항목에 ✅ / ⚠️ / ❌ 판정과 한 줄 근거를 작성해 주세요.',
  priority: 'High / Medium / Low 우선순위별 수정 Action List를 작성해 주세요.',
  both: '영역별 ✅ / ⚠️ / ❌ 판정표를 작성한 후, High / Medium / Low 우선순위별 Action List를 이어서 작성해 주세요.',
};

const PERSONA_DESC: Record<string, { name: string; role: string; focus: string; tone: string }> = {
  senior: { name: 'Reviewer A', role: 'Senior Researcher (10+ yrs)', focus: '이론적 엄밀함, 수학적 정확성, 기존 문헌과의 연결성', tone: 'formal and probing. Challenges assumptions directly.' },
  industry: { name: 'Reviewer B', role: 'Industry Practitioner', focus: '실제 벤치마크 결과, 계산 효율, 실용적 적용 가능성', tone: 'direct and results-focused. Asks "does this actually work at scale?"' },
  phd: { name: 'Reviewer C', role: 'PhD Student (3rd year)', focus: '관련 연구 포괄성, 기존 논문과의 차별성, 실험 디테일', tone: 'thorough and detail-oriented. May be pedantic about citations.' },
  interdisciplinary: { name: 'Reviewer D', role: 'Interdisciplinary Researcher', focus: 'novelty의 진정성, 더 넓은 AI 커뮤니티에 대한 기여', tone: 'big-picture thinker. Critical of incremental or over-hyped work.' },
  stats: { name: 'Reviewer E', role: 'Methods & Statistics Expert', focus: '통계적 유의성, 실험 설계의 엄밀함, 재현 가능성', tone: 'precise and quantitative. Will flag any statistical irregularities.' },
};

const RIGOR_NOTE: Record<string, string> = {
  top: 'You are an exceptionally rigorous reviewer at a top-tier venue. Your bar is high — only papers that make a clear, significant contribution should pass. Be direct and unsparing.',
  typical: 'You are a fair and balanced reviewer. You aim to give constructive, realistic feedback that reflects the typical standards of this conference.',
  lenient: 'You are a constructive reviewer who looks for the strengths in a paper. You give authors reasonable benefit of the doubt, while still flagging genuine issues.',
};

const CONF_P: Record<string, { scoreRange: string; acceptance: string; style: string }> = {
  NeurIPS: { scoreRange: '1-10', acceptance: '~26%', style: 'Calibrated, theoretically rigorous. Values novelty and empirical soundness equally. Checks statistical significance carefully.' },
  ICML: { scoreRange: '1-6', acceptance: '~23%', style: 'Strong emphasis on theoretical grounding and algorithmic novelty. Expects rigorous proofs or proof sketches.' },
  ICLR: { scoreRange: '1-10', acceptance: '~32%', style: 'Open Review style. Values reproducibility, ablation studies, and clear writing.' },
  CVPR: { scoreRange: '1-6', acceptance: '~25%', style: 'Cares deeply about benchmark performance and visual quality. Wants clear SOTA comparisons and strong ablations.' },
  ICCV: { scoreRange: '1-6', acceptance: '~26%', style: 'Biennial, very competitive. Similar to CVPR but slightly more theory-friendly.' },
  ECCV: { scoreRange: '1-6', acceptance: '~28%', style: 'Values elegant methodology. Slightly more tolerant of incremental but solid work.' },
  ACL: { scoreRange: '1-5', acceptance: '~23%', style: 'Linguistics-aware. Expects careful error analysis, human evaluation for generation tasks.' },
  EMNLP: { scoreRange: '1-5', acceptance: '~24%', style: 'More empirical than ACL. Values large-scale experiments and practical NLP applications.' },
  AAAI: { scoreRange: '1-6', acceptance: '~24%', style: 'Broad AI scope. Values clarity and applicability.' },
  KDD: { scoreRange: '1-5', acceptance: '~19%', style: 'Data-centric. Expects scalability analysis, real-world dataset experiments, and runtime benchmarks.' },
  ICRA: { scoreRange: '1-5', acceptance: '~44%', style: 'Robotics focus. Values hardware experiments, safety considerations, and real-world applicability.' },
  UAI: { scoreRange: '1-5', acceptance: '~28%', style: 'Probabilistic ML and uncertainty. Expects rigorous Bayesian analysis and theoretical proofs.' },
};

const DOMAIN_FULL: Record<string, string> = {
  NLP: 'NLP / 언어모델', CV: '컴퓨터 비전', RL: '강화학습', ML: '머신러닝 일반',
  GenAI: '생성 AI', GNN: '그래프 신경망', MultiModal: '멀티모달', Theory: '이론 ML',
  Bio: 'AI × 생물/의료', Robotics: '로보틱스',
};

const TIER: Record<string, string> = {
  top: 'Top-tier (NeurIPS/ICML/ICLR/ACL/CVPR)',
  a: 'A급 (AAAI/EMNLP/ICCV/KDD 등)',
  workshop: 'Workshop / 탐색적 연구',
};

const DURATION: Record<string, string> = {
  '3mo': '3개월 이내', '6mo': '6개월', '1yr': '1년 이상',
};

const DIRECTION: Record<string, string> = {
  novel_method: '새로운 방법론 제안',
  analysis: '심층 분석 / 이해',
  application: '새로운 적용 영역',
  benchmark: '벤치마크 / 데이터셋 구축',
  survey: 'Survey / 정리',
};

const EX_TYPE: Record<string, string> = {
  method: '새로운 방법론 제안 논문',
  comparison: '기존 방법 비교 분석 논문',
  ablation: 'Ablation study 중심',
  scaling: 'Scaling 실험',
  analysis: '분석 / 이해 논문',
};

const EX_METRIC: Record<string, string> = {
  accuracy: '정확도/성능 지표 (Accuracy, F1, BLEU 등)',
  efficiency: '효율성 지표 (속도, 메모리, FLOPs)',
  both: '성능 + 효율성 동시',
  human: 'Human evaluation',
  custom: '커스텀 지표',
};

const EX_CONCERN: Record<string, string> = {
  repro: '재현성 / 랜덤 시드 고정 전략',
  stat: '통계적 유의성 검증 방법',
  ablation: 'Ablation study 설계 순서',
  compute: '계산 비용 사전 추정',
  leakage: '데이터 누수 방지 체크리스트',
  fair: '공정한 비교 조건 설계',
  negative: '부정적 결과 처리 방법',
  checklist: '완전한 재현성 체크리스트 생성',
};

const W_SECTIONS: Record<string, string> = {
  abstract: 'Abstract', intro: 'Introduction', related: 'Related Work',
  method: 'Method (제안 방법)', experiments: 'Experiments',
  conclusion: 'Conclusion', limitation: 'Limitations', appendix: 'Appendix',
};

const STYLE_PKG: Record<string, string> = {
  neurips: '\\usepackage{neurips_2024}', icml: '\\usepackage{icml2024}',
  iclr: '\\usepackage{iclr2024_conference}', cvpr: '\\usepackage{cvpr}',
  iccv: '\\usepackage{iccv}', eccv: '\\usepackage{eccv}', acl: '\\usepackage{acl}',
  aaai: '\\usepackage{aaai25}', ijcai: '\\usepackage{ijcai24}',
  kdd: 'ACM \\documentclass{acmart}', www: 'ACM \\documentclass{acmart}',
  sigir: 'ACM \\documentclass{acmart}', icra: 'IEEE \\usepackage{ieeeconf}',
  uai: '\\usepackage{uai2024}',
};

const ST_LEVEL: Record<string, string> = {
  sota: 'SOTA 달성 / 확실한 성능 개선',
  competitive: '경쟁력 있는 수준의 결과',
  modest: '소폭 개선 또는 아이디어·분석 중심',
  negative: '부정적 결과 또는 분석 논문',
  theoretical: '이론적 기여 중심 (실험 최소화)',
};

const ST_FACTORS: Record<string, string> = {
  fit: '각 후보 학회에 대한 논문 적합도 분석 (토픽 매칭, 최근 accept 트렌드 고려)',
  timing: '다음 6개월 내 주요 학회 데드라인 기반 제출 타이밍 최적화',
  positioning: '이 논문의 핵심 contribution을 학회별로 어떻게 다르게 포지셔닝할지 전략',
  rejection: '각 학회별 리젝 확률 예측 및 주요 리젝 사유 예방 전략',
  cover: '커버레터 전략 및 에디터에게 어필할 포인트',
  concurrent: '병렬 투고 가능성 및 순서 전략 (workshop → main conf → 저널)',
  arxiv: 'arXiv 공개 최적 타이밍 및 커뮤니티 파급 효과 극대화 방법',
};

const RB_STRATEGY: Record<string, string> = {
  triage: '각 리뷰어 코멘트를 "반드시 반박해야 할 것 / 수용해야 할 것 / 추가 실험으로 답해야 할 것"으로 분류해 주세요.',
  draft: '리뷰어별로 각 약점에 대한 구체적인 반박 초안을 작성해 주세요.',
  experiment: '반박에 필요한 추가 실험 계획을 우선순위 순으로 제시해 주세요. 제약 시간 내에 실행 가능한 실험만 포함해 주세요.',
  tone: '리뷰어를 설득하는 톤과 어조 가이드를 제시해 주세요. 과도하게 방어적이거나 공격적이지 않으면서 자신감 있는 rebuttal 작성법을 알려 주세요.',
  common: '여러 리뷰어가 공통으로 제기한 문제를 묶어 한 번에 효율적으로 답변하는 방법을 제시해 주세요.',
  ac: 'Area Chair를 설득하기 위한 전략을 제시해 주세요. 리뷰어 의견 중 AC가 중요하게 볼 포인트와 그에 대한 대응 방법을 포함해 주세요.',
};

// ───────────────────────────────────────────────────────────
// 빌더 정의
// ───────────────────────────────────────────────────────────

export const promptBuilders: PromptBuilder[] = [
  // 1. 돌파구 전략
  {
    id: 'breakthrough',
    title: { ko: '돌파구 전략', en: 'Breakthrough Strategy' },
    description: {
      ko: '연구가 막혔을 때 다각도 접근으로 돌파구를 찾는 프롬프트를 생성합니다.',
      en: 'Generate a prompt to find a research breakthrough from multiple angles.',
    },
    category: 'ideation',
    icon: 'lightbulb',
    tags: ['연구 기획', '문제 해결', '아이디어'],
    fields: [
      {
        id: 'role', type: 'select', required: true,
        label: { ko: '역할', en: 'Role' },
        options: [
          { value: 'researcher', label: { ko: '동료 연구자', en: 'Peer researcher' } },
          { value: 'critic', label: { ko: '냉정한 리뷰어', en: 'Critical reviewer' } },
          { value: 'inventor', label: { ko: '창의적 발명가', en: 'Creative inventor' } },
          { value: 'phd', label: { ko: '엄격한 어드바이저', en: 'Strict advisor' } },
        ],
      },
      {
        id: 'domain', type: 'select', required: true, default: 'ML',
        label: { ko: '분야', en: 'Domain' },
        options: Object.keys(DC).map((k) => ({ value: k, label: { ko: DC[k], en: DC[k] } })),
      },
      {
        id: 'blockType', type: 'select', required: true, default: 'perf',
        label: { ko: '현재 문제 유형', en: 'Block type' },
        options: Object.keys(BC).map((k) => ({ value: k, label: { ko: BC[k], en: BC[k] } })),
      },
      {
        id: 'situation', type: 'textarea', required: true,
        label: { ko: '지금 상황 (구체적으로)', en: 'Current situation (specific)' },
        placeholder: { ko: '예: Transformer 기반 모델이 특정 데이터셋에서 85% 정확도에 머물고 있음...', en: 'e.g. Our Transformer plateaus at 85% accuracy on ...' },
      },
      {
        id: 'tried', type: 'textarea',
        label: { ko: '이미 시도한 것들', en: 'What you already tried' },
      },
      {
        id: 'strategies', type: 'multiselect',
        label: { ko: '접근법 (복수 선택)', en: 'Strategies (multi)' },
        help: { ko: '미선택 시 다각도 접근으로 자동 구성됩니다.', en: 'If none selected, a multi-angle approach is used.' },
        options: Object.keys(SL).map((k) => ({ value: k, label: { ko: SL[k].replace(/하고$/, ''), en: k } })),
      },
      {
        id: 'depth', type: 'select', default: 'deep',
        label: { ko: '분석 깊이', en: 'Depth' },
        options: [
          { value: 'deep', label: { ko: '심층 분석', en: 'Deep' } },
          { value: 'quick', label: { ko: '빠른 요약', en: 'Quick' } },
          { value: 'structured', label: { ko: '로드맵', en: 'Roadmap' } },
        ],
      },
      {
        id: 'format', type: 'select', default: 'hypotheses',
        label: { ko: '출력 형식', en: 'Output format' },
        options: [
          { value: 'steps', label: { ko: '실행 단계', en: 'Steps' } },
          { value: 'hypotheses', label: { ko: '검증 가능한 가설', en: 'Hypotheses' } },
          { value: 'experiments', label: { ko: '실험 설계서', en: 'Experiment design' } },
          { value: 'code', label: { ko: '슈도코드', en: 'Pseudocode' } },
        ],
      },
    ],
    generate: (v) => {
      const sit = String(v.situation ?? '').trim();
      const strats = (v.strategies as string[]) ?? [];
      const tried = String(v.tried ?? '').trim();
      const lines: (string | false)[] = [
        RI[String(v.role ?? 'researcher')],
        '',
        '## 연구 맥락',
        `- 분야: ${DC[String(v.domain ?? 'ML')]}`,
        `- 현재 문제: ${BC[String(v.blockType ?? 'perf')]}`,
        `- 상황: ${sit}`,
        tried ? `\n## 이미 시도한 방법들\n${tried}\n→ 위 방법들이 왜 충분하지 않았는지도 함께 분석해 주세요.` : false,
        '',
        '## 요청',
        '다음 접근법들을 활용하여 이 연구의 돌파구를 제시해 주세요:',
        strats.length ? strats.map((s) => `  - ${SL[s]}`).join('\n') : '  - 다각도로 접근하여',
        '',
        '## 출력 방식',
        DI[String(v.depth ?? 'deep')],
        FI[String(v.format ?? 'hypotheses')],
        '',
        '마지막으로, 이 문제가 해결된다면 어떤 더 큰 연구 기회가 열릴 수 있는지 한 문단으로 제시해 주세요.',
      ];
      return lines.filter((l) => l !== false).join('\n');
    },
  },

  // 2. 종합 품질 감사
  {
    id: 'audit',
    title: { ko: '종합 품질 감사', en: 'Quality Audit' },
    description: {
      ko: '코드·실험·논문 전반의 품질을 체계적으로 점검하는 감사 프롬프트를 생성합니다.',
      en: 'Generate an audit prompt to check code, experiment, and paper quality.',
    },
    category: 'quality',
    icon: 'shield-check',
    tags: ['검증', '체크리스트', '디버깅'],
    fields: [
      {
        id: 'target', type: 'textarea', required: true,
        label: { ko: '점검 대상 설명', en: 'Target to audit' },
        placeholder: { ko: '예: MNIST 분류 모델 학습 코드와 실험 스크립트, 결과 표', en: 'e.g. MNIST training code, experiment scripts, result tables' },
      },
      {
        id: 'checks', type: 'multiselect', required: true,
        label: { ko: '점검 항목 (복수 선택)', en: 'Check items (multi)' },
        options: [
          ...['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'].map((k) => ({ value: k, label: { ko: `[코드] ${CL[k]}`, en: `[Code] ${CL[k]}` } })),
          ...['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'].map((k) => ({ value: k, label: { ko: `[실험] ${CL[k]}`, en: `[Exp] ${CL[k]}` } })),
          ...['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'].map((k) => ({ value: k, label: { ko: `[논문] ${CL[k]}`, en: `[Paper] ${CL[k]}` } })),
        ],
      },
      {
        id: 'severity', type: 'select', default: 'balanced',
        label: { ko: '판정 엄격도', en: 'Severity' },
        options: [
          { value: 'strict', label: { ko: '엄격 (모두 표시)', en: 'Strict' } },
          { value: 'balanced', label: { ko: '균형', en: 'Balanced' } },
          { value: 'light', label: { ko: '치명적만', en: 'Critical only' } },
        ],
      },
      {
        id: 'output', type: 'select', default: 'both',
        label: { ko: '출력 형식', en: 'Output format' },
        options: [
          { value: 'table', label: { ko: '영역별 판정표', en: 'Verdict table' } },
          { value: 'priority', label: { ko: '우선순위 액션', en: 'Priority actions' } },
          { value: 'both', label: { ko: '판정표 + 액션', en: 'Table + actions' } },
        ],
      },
      {
        id: 'extra', type: 'textarea',
        label: { ko: '추가 지시사항 (선택)', en: 'Extra instructions (optional)' },
      },
      {
        id: 'lang', type: 'select', default: 'ko',
        label: { ko: '응답 언어', en: 'Response language' },
        options: [
          { value: 'ko', label: { ko: '한국어', en: 'Korean' } },
          { value: 'en', label: { ko: 'English', en: 'English' } },
        ],
      },
    ],
    generate: (v) => {
      const tgt = String(v.target ?? '').trim();
      const sel = (v.checks as string[]) ?? [];
      const ci = sel.filter((e) => e.startsWith('c'));
      const ei = sel.filter((e) => e.startsWith('e'));
      const pi = sel.filter((e) => e.startsWith('p'));
      let n = 1;
      const nl = (items: string[]) => items.map((e) => `${n++}. ${CL[e]}`).join('\n');
      let cb = '';
      if (ci.length) cb += `\n[코드]\n${nl(ci)}\n`;
      if (ei.length) cb += `\n[실험]\n${nl(ei)}\n`;
      if (pi.length) cb += `\n[논문]\n${nl(pi)}\n`;
      const extra = String(v.extra ?? '').trim();
      const p = `당신은 AI 보조 연구의 종합 품질 감사자입니다.\n\n【점검 대상】\n${tgt}\n\n【선택된 점검 항목】${cb}\n【점검 지시사항】\n- ${SEV[String(v.severity ?? 'balanced')]}\n- ${AOUT[String(v.output ?? 'both')]}\n- 서로 다른 영역 간 불일치를 특별히 확인해 주세요.\n- AI 생성 흔적(과도한 일반화, 모호한 표현, 구조적 반복)을 표시해 주세요.${extra ? `\n\n【추가 지시사항】\n${extra}` : ''}${v.lang === 'en' ? '\n\nPlease respond in English.' : ''}`;
      return p;
    },
  },

  // 3. 논문 리뷰 생성
  {
    id: 'paper-review',
    title: { ko: '논문 리뷰 생성', en: 'Paper Review' },
    description: {
      ko: '여러 리뷰어 페르소나로 논문을 리뷰하는 프롬프트를 생성합니다.',
      en: 'Generate a multi-persona paper review prompt.',
    },
    category: 'quality',
    icon: 'file-search',
    tags: ['리뷰', '논문', 'peer review'],
    fields: [
      {
        id: 'filepath', type: 'text', required: true,
        label: { ko: '논문 PDF 경로/파일명', en: 'Paper PDF path/filename' },
        placeholder: { ko: '예: ./papers/my_paper.pdf', en: 'e.g. ./papers/my_paper.pdf' },
      },
      {
        id: 'conf', type: 'select', default: 'NeurIPS',
        label: { ko: '투고(예정) 학회', en: 'Target venue' },
        options: Object.keys(CONF_P).map((k) => ({ value: k, label: { ko: k, en: k } })),
      },
      {
        id: 'count', type: 'select', default: '3',
        label: { ko: '리뷰어 수', en: 'Number of reviewers' },
        options: ['1', '2', '3', '4', '5'].map((n) => ({ value: n, label: { ko: `${n}명`, en: `${n}` } })),
      },
      {
        id: 'personas', type: 'multiselect',
        label: { ko: '리뷰어 페르소나 (복수 선택)', en: 'Reviewer personas (multi)' },
        options: Object.keys(PERSONA_DESC).map((k) => ({
          value: k,
          label: { ko: `${PERSONA_DESC[k].role}`, en: PERSONA_DESC[k].role },
        })),
      },
      {
        id: 'rigor', type: 'select', default: 'typical',
        label: { ko: '리뷰 엄격도', en: 'Review rigor' },
        options: [
          { value: 'top', label: { ko: '최상위 학회급 (엄격)', en: 'Top-tier (strict)' } },
          { value: 'typical', label: { ko: '일반적 기준', en: 'Typical' } },
          { value: 'lenient', label: { ko: '관대/건설적', en: 'Lenient' } },
        ],
      },
      {
        id: 'lang', type: 'select', default: 'ko',
        label: { ko: '응답 언어', en: 'Response language' },
        options: [
          { value: 'ko', label: { ko: '한국어', en: 'Korean' } },
          { value: 'en', label: { ko: 'English', en: 'English' } },
        ],
      },
      {
        id: 'context', type: 'textarea',
        label: { ko: '추가 컨텍스트 (선택)', en: 'Extra context (optional)' },
      },
    ],
    generate: (v) => {
      const conf = String(v.conf ?? 'NeurIPS');
      const cp2 = CONF_P[conf] ?? CONF_P.NeurIPS;
      const count = parseInt(String(v.count ?? '3'), 10);
      const lang = String(v.lang ?? 'ko');
      const ctx = String(v.context ?? '').trim();
      const selected = (v.personas as string[]) ?? [];
      const all = ['senior', 'industry', 'phd', 'interdisciplinary', 'stats'];
      const active = all.filter((p) => selected.includes(p));
      const remaining = all.filter((p) => !selected.includes(p));
      while (active.length < count && remaining.length) active.push(remaining.shift()!);
      const reviewers = active.slice(0, count);
      const langNote = lang === 'ko' ? '모든 리뷰는 한국어로 작성해 주세요.' : 'Write all reviews in English.';
      const scoreMax = cp2.scoreRange.split('-')[1];
      const reviewerBlock = reviewers.map((pid) => {
        const p = PERSONA_DESC[pid];
        return `### ${p.name} — ${p.role}\n- **초점**: ${p.focus}\n- **톤**: ${p.tone}\n- **점수(1-${scoreMax})**: \n- **강점**: \n- **약점**: \n- **질문**: \n- **개선 제안**: `;
      }).join('\n\n');
      return `당신은 top-tier AI/ML 학회의 리뷰 패널입니다. 아래 논문을 읽고 ${count}명의 서로 다른 리뷰어 관점에서 리뷰를 작성해 주세요.\n\n## 논문 파일\n아래 논문 PDF를 읽어 주세요: ${String(v.filepath ?? '').trim()}\n\n## 학회 컨텍스트\n- 투고 학회: ${conf} (점수 ${cp2.scoreRange}, 채택률 ${cp2.acceptance})\n- 학회 스타일: ${cp2.style}\n\n## 리뷰 지침\n${RIGOR_NOTE[String(v.rigor ?? 'typical')]}\n${langNote}\n${ctx ? `\n## 추가 컨텍스트\n${ctx}` : ''}\n\n## 리뷰 형식\n각 리뷰어는 아래 템플릿을 채워 주세요:\n\n${reviewerBlock}\n\n마지막에 **메타리뷰**를 추가해 주세요: 전반적인 강점·약점 종합, 채택 권유 여부(Strong Accept ~ Strong Reject), 가장 결정적인 리뷰 포인트.`;
    },
  },

  // 4. 연구 방향 & 문헌 조사
  {
    id: 'idea',
    title: { ko: '연구 방향 & 문헌 조사', en: 'Research Direction' },
    description: {
      ko: '연구 방향을 설계하고 최신 문헌을 조사하는 프롬프트를 생성합니다.',
      en: 'Generate a research direction and literature survey prompt.',
    },
    category: 'ideation',
    icon: 'compass',
    tags: ['문헌 조사', 'SOTA', '연구 기획'],
    fields: [
      {
        id: 'domain', type: 'select', required: true, default: 'NLP',
        label: { ko: '분야', en: 'Domain' },
        options: Object.keys(DOMAIN_FULL).map((k) => ({ value: k, label: { ko: DOMAIN_FULL[k], en: DOMAIN_FULL[k] } })),
      },
      {
        id: 'tier', type: 'select', default: 'top',
        label: { ko: '목표 학회 티어', en: 'Target tier' },
        options: Object.keys(TIER).map((k) => ({ value: k, label: { ko: TIER[k], en: TIER[k] } })),
      },
      {
        id: 'keywords', type: 'text', required: true,
        label: { ko: '핵심 키워드 / 관심 영역', en: 'Keywords / interests' },
        placeholder: { ko: '예: efficient attention, long-context LLM', en: 'e.g. efficient attention, long-context LLM' },
      },
      {
        id: 'resources', type: 'multiselect',
        label: { ko: '보유 자원', en: 'Resources' },
        options: ['GPU ×1~4', 'GPU 클러스터', '공개 데이터셋', '자체 데이터', '사전학습 모델', '산학 협력'].map((r) => ({ value: r, label: { ko: r, en: r } })),
      },
      {
        id: 'duration', type: 'select', default: '6mo',
        label: { ko: '연구 기간', en: 'Duration' },
        options: Object.keys(DURATION).map((k) => ({ value: k, label: { ko: DURATION[k], en: DURATION[k] } })),
      },
      {
        id: 'direction', type: 'select', default: 'novel_method',
        label: { ko: '선호 방향', en: 'Direction' },
        options: Object.keys(DIRECTION).map((k) => ({ value: k, label: { ko: DIRECTION[k], en: DIRECTION[k] } })),
      },
      {
        id: 'avoid', type: 'textarea',
        label: { ko: '피하고 싶은 방향 / 포화 분야 (선택)', en: 'Directions to avoid (optional)' },
      },
    ],
    generate: (v) => {
      const kw = String(v.keywords ?? '').trim();
      const res = (v.resources as string[]) ?? [];
      const avoid = String(v.avoid ?? '').trim();
      const lines: (string | boolean)[] = [
        '당신은 AI/ML 분야의 연구 방향을 설계하고 문헌을 조사하는 전문 어드바이저입니다.',
        '아래 연구자 프로파일을 바탕으로 다음 세 가지를 순서대로 수행해 주세요.',
        '',
        '## 연구자 프로파일',
        `- 분야: ${DOMAIN_FULL[String(v.domain ?? 'NLP')]}`,
        `- 목표 티어: ${TIER[String(v.tier ?? 'top')]}`,
        `- 핵심 관심 키워드: ${kw}`,
        `- 보유 자원: ${res.length ? res.join(', ') : '미입력'}`,
        `- 연구 기간: ${DURATION[String(v.duration ?? '6mo')]}`,
        `- 선호 방향: ${DIRECTION[String(v.direction ?? 'novel_method')]}`,
        avoid ? `- 피하고 싶은 방향: ${avoid}` : false,
        '',
        '---',
        '## Task 1. 최신 관련 연구 및 SOTA 조사',
        '',
        '위 키워드와 분야를 기준으로 다음을 정리해 주세요.',
        '',
        '### 1-1. 핵심 논문 목록 (최근 3년)',
        '각 논문에 대해 (최소 10편):',
        '- **제목** (저자, 학회/연도)',
        '  - 핵심 아이디어: 한 문장',
        '  - 주요 결과: 대표 수치 또는 기여',
        '  - 한계점: 한 문장',
        '',
        '### 1-2. 현재 SOTA 모델/방법론 정리',
        '각 태스크별 SOTA, 사용 데이터셋, 핵심 지표, 공개 코드 여부',
        '',
        '---',
        '## Task 2. 연구 기회(Gap) 식별',
        '- 현재 연구 동향에서 아직 해결되지 않은 문제(Gap) 5~7개를 도출',
        '- 각 Gap에 대해 왜 중요한지, 해결 시 파급 효과를 설명',
        '',
        '---',
        '## Task 3. 구체적 연구 아이디어 제안',
        '- 프로파일(자원·기간·티어)에 현실적으로 부합하는 연구 아이디어 5개 제안',
        '- 각 아이디어: 핵심 가설, 차별성, 필요 데이터/자원, 예상 소요 시간, 성공 시 기여',
        '- 아이디어별로 위험도(낮음/중간/높음)와 실패 시 대안을 함께 표시',
      ];
      return lines.filter((l) => l !== false).join('\n');
    },
  },

  // 5. 실험 설계
  {
    id: 'experiment',
    title: { ko: '실험 설계', en: 'Experiment Design' },
    description: {
      ko: '연구 가설을 검증할 체계적 실험 플랜을 설계하는 프롬프트를 생성합니다.',
      en: 'Generate a systematic experiment design prompt.',
    },
    category: 'ideation',
    icon: 'flask-conical',
    tags: ['실험', 'Ablation', '재현성'],
    fields: [
      {
        id: 'type', type: 'select', default: 'method',
        label: { ko: '연구 유형', en: 'Research type' },
        options: Object.keys(EX_TYPE).map((k) => ({ value: k, label: { ko: EX_TYPE[k], en: EX_TYPE[k] } })),
      },
      {
        id: 'hypothesis', type: 'textarea', required: true,
        label: { ko: '검증할 핵심 가설', en: 'Core hypothesis' },
        placeholder: { ko: '예: 제안한 attention이 긴 시퀀스에서 더 나은 성능을 낸다', en: 'e.g. Our attention outperforms on long sequences' },
      },
      {
        id: 'metric', type: 'select', default: 'accuracy',
        label: { ko: '평가 지표', en: 'Metric' },
        options: Object.keys(EX_METRIC).map((k) => ({ value: k, label: { ko: EX_METRIC[k], en: EX_METRIC[k] } })),
      },
      {
        id: 'concerns', type: 'multiselect', required: true,
        label: { ko: '챙길 요소 (복수 선택)', en: 'Concerns (multi)' },
        options: Object.keys(EX_CONCERN).map((k) => ({ value: k, label: { ko: EX_CONCERN[k], en: EX_CONCERN[k] } })),
      },
      { id: 'datasets', type: 'text', label: { ko: '데이터셋 / 벤치마크', en: 'Datasets / benchmarks' } },
      { id: 'baselines', type: 'text', label: { ko: '비교 대상 (베이스라인)', en: 'Baselines' } },
      { id: 'constraints', type: 'textarea', label: { ko: '실험 환경 / 제약 (선택)', en: 'Constraints (optional)' } },
    ],
    generate: (v) => {
      const hyp = String(v.hypothesis ?? '').trim();
      const concerns = (v.concerns as string[]) ?? [];
      const ds = String(v.datasets ?? '').trim();
      const bl = String(v.baselines ?? '').trim();
      const cons = String(v.constraints ?? '').trim();
      const p = `당신은 ML/AI 실험 설계 전문가입니다. 다음 연구의 최적 실험 플랜을 설계해 주세요.\n\n## 실험 개요\n- 연구 유형: ${EX_TYPE[String(v.type ?? 'method')]}\n- 핵심 가설: ${hyp}\n- 평가 지표: ${EX_METRIC[String(v.metric ?? 'accuracy')]}${ds ? `\n- 데이터셋/벤치마크: ${ds}` : ''}${bl ? `\n- 베이스라인: ${bl}` : ''}${cons ? `\n- 실험 환경/제약: ${cons}` : ''}\n\n## 요청 사항\n다음 항목들을 포함한 실험 설계를 제시해 주세요:\n${(concerns.length ? concerns : Object.keys(EX_CONCERN)).map((c) => `- ${EX_CONCERN[c]}`).join('\n')}\n\n## 출력 형식\n\n### 실험 로드맵\n우선순위 순으로 실험을 나열하되, 각 실험에 대해:\n- 실험 목적 (어떤 질문에 답하는가)\n- 독립변수 / 종속변수 / 통제변수\n- 예상 결과 및 그 의미\n- 실패했을 때 대안\n- 예상 소요 시간/비용\n\n### 최소 실험 세트 (MVP)\n논문 제출에 반드시 필요한 최소 실험 3-5개를 꼽아 주세요.\n\n### 잠재적 함정\n이 실험 설계에서 흔히 저지르는 실수와 예방법을 제시해 주세요.\n${concerns.includes('checklist') ? `\n### 재현성 체크리스트\n단계별 재현성 확보를 위한 완전한 체크리스트를 작성해 주세요.\n` : ''}`;
      return p;
    },
  },

  // 6. 논문 작성
  {
    id: 'writing',
    title: { ko: '논문 작성', en: 'Paper Writing' },
    description: {
      ko: '학회 규격에 맞는 LaTeX 논문 작성을 지시하는 프롬프트를 생성합니다.',
      en: 'Generate a LaTeX paper writing prompt matching venue style.',
    },
    category: 'writing',
    icon: 'pen-line',
    tags: ['논문', 'LaTeX', '작성'],
    fields: [
      {
        id: 'conf', type: 'select', default: 'neurips',
        label: { ko: '투고 학회 (스타일)', en: 'Venue (style)' },
        options: Object.keys(STYLE_PKG).map((k) => ({ value: k, label: { ko: k.toUpperCase(), en: k.toUpperCase() } })),
      },
      {
        id: 'domain', type: 'text',
        label: { ko: '연구 분야', en: 'Research domain' },
        placeholder: { ko: '예: Long-context language modeling', en: 'e.g. Long-context language modeling' },
      },
      {
        id: 'title', type: 'text',
        label: { ko: '논문 제목 (초안)', en: 'Title (draft)' },
      },
      {
        id: 'contrib', type: 'textarea', required: true,
        label: { ko: '핵심 Contribution (1~3가지)', en: 'Key contributions' },
      },
      { id: 'results', type: 'textarea', label: { ko: '실험 결과 요약', en: 'Results summary' } },
      { id: 'baselines', type: 'text', label: { ko: '주요 베이스라인', en: 'Baselines' } },
      {
        id: 'sections', type: 'multiselect',
        label: { ko: '포함할 섹션', en: 'Sections' },
        options: Object.keys(W_SECTIONS).map((k) => ({ value: k, label: { ko: W_SECTIONS[k], en: W_SECTIONS[k] } })),
      },
      { id: 'extra', type: 'textarea', label: { ko: '추가 정보 / 특이사항 (선택)', en: 'Extra notes (optional)' } },
    ],
    generate: (v) => {
      const conf = String(v.conf ?? 'neurips');
      const pkg = STYLE_PKG[conf] ?? `\\usepackage{${conf}}`;
      const title = String(v.title ?? '').trim();
      const contrib = String(v.contrib ?? '').trim();
      const results = String(v.results ?? '').trim();
      const baselines = String(v.baselines ?? '').trim();
      const extra = String(v.extra ?? '').trim();
      const domain = String(v.domain ?? '').trim();
      const secs = (v.sections as string[]) ?? ['abstract', 'intro', 'related', 'method', 'experiments', 'conclusion'];
      const secList = secs.map((s) => `- ${W_SECTIONS[s] ?? s}`).join('\n');
      return `당신은 top-tier AI/ML 논문 작성 전문가입니다.\n아래 정보를 바탕으로 ${conf.toUpperCase()} 제출 규격에 맞는 완전한 LaTeX 논문을 작성해 주세요.\n\n## LaTeX 스타일 요구사항\n- 공식 스타일 패키지: ${pkg}\n- 최신 ${conf.toUpperCase()} 스타일 파일 규격을 정확히 따라 주세요.\n  익명 제출(blind review) 여부, 줄 번호(line numbers), 컬럼 수 등 학회 규정을 반영해 주세요.\n\n## 논문 정보\n- 제목: ${title || '(작성 중 — 적절한 제목을 제안해 주세요)'}${domain ? `\n- 연구 분야: ${domain}` : ''}\n\n### 핵심 Contribution\n${contrib}${results ? `\n\n### 주요 실험 결과\n${results}` : ''}${baselines ? `\n\n### 베이스라인 / 비교 대상\n${baselines}` : ''}${extra ? `\n\n### 추가 사항\n${extra}` : ''}\n\n## 작성할 섹션\n${secList}\n\n## 작성 지시사항\n\n1. **완전한 .tex 파일**을 출력해 주세요.\n   - \\documentclass 부터 \\end{document} 까지 전체 포함\n   - pdflatex 또는 xelatex으로 바로 컴파일 가능한 수준\n   - 필요한 모든 \\usepackage 선언 포함\n\n2. **섹션별 작성 기준**:\n   - Abstract: 4~6문장, problem / gap / solution / key result 구조\n   - Introduction: motivation → gap → contribution 흐름, \\itemize로 contribution 명시\n   - Related Work: 3~4개 그룹으로 분류, 각 그룹 마지막에 우리 연구와의 차별점 1문장\n   - Method: 핵심 아이디어를 수식(\\equation)과 함께 설명, \\algorithm 환경 포함 권장\n   - Experiments: 메인 결과 table(\\booktabs 사용), ablation table 포함\n   - Conclusion: 기여 요약 + limitation + future work\n\n3. **표와 수식**:\n   - \\usepackage{booktabs} 사용, \\toprule/\\midrule/\\bottomrule\n   - 수식은 \\begin{equation}으로 번호 부여\n\n4. **참고문헌**: \\bibitem 형식으로 예시 5개 포함 (실제 관련 논문으로 채워 주세요)`;
    },
  },

  // 7. 학회 투고 전략
  {
    id: 'conference',
    title: { ko: '학회 투고 전략', en: 'Conference Strategy' },
    description: {
      ko: '논문을 어디에, 언제, 어떻게 투고할지 전략을 수립하는 프롬프트를 생성합니다.',
      en: 'Generate a venue/timing submission strategy prompt.',
    },
    category: 'strategy',
    icon: 'target',
    tags: ['투고', '학회 선택', '포지셔닝'],
    fields: [
      {
        id: 'filepath', type: 'text', required: true,
        label: { ko: '논문 PDF 경로/파일명', en: 'Paper PDF path/filename' },
      },
      {
        id: 'level', type: 'select', default: 'competitive',
        label: { ko: '결과 수준', en: 'Result level' },
        options: Object.keys(ST_LEVEL).map((k) => ({ value: k, label: { ko: ST_LEVEL[k], en: ST_LEVEL[k] } })),
      },
      {
        id: 'domain', type: 'text', default: 'NLP',
        label: { ko: '분야', en: 'Domain' },
      },
      {
        id: 'timing', type: 'text',
        label: { ko: '제출 희망 시기', en: 'Target submission timing' },
        placeholder: { ko: '예: 올해 안에 / 6개월 내', en: 'e.g. within this year' },
      },
      {
        id: 'author', type: 'text',
        label: { ko: '저자 현황', en: 'Author situation' },
        placeholder: { ko: '예: 학부생 주도 / 박사후研究员 포함', en: 'e.g. undergrad-led' },
      },
      {
        id: 'rejected', type: 'textarea',
        label: { ko: '이미 리젝된 학회 / 리뷰 요약 (선택)', en: 'Prior rejections (optional)' },
      },
      {
        id: 'factors', type: 'multiselect',
        label: { ko: '챙길 전략 요소 (복수 선택)', en: 'Strategy factors (multi)' },
        options: Object.keys(ST_FACTORS).map((k) => ({ value: k, label: { ko: ST_FACTORS[k], en: ST_FACTORS[k] } })),
      },
    ],
    generate: (v) => {
      const fileRef = String(v.filepath ?? '').trim();
      const factors = (v.factors as string[]) ?? Object.keys(ST_FACTORS);
      const rejected = String(v.rejected ?? '').trim();
      const level = String(v.level ?? 'competitive');
      const lines: (string | boolean)[] = [
        '당신은 top-tier AI/ML 학회 투고 전략을 전문으로 하는 어드바이저입니다.',
        '',
        '## 논문 파일',
        `아래 논문 PDF를 읽어 주세요: ${fileRef}`,
        '',
        '논문을 읽은 후 아래 정보를 참고하여 투고 전략을 수립해 주세요.',
        '',
        '## 논문 현황',
        `- 결과 수준: ${ST_LEVEL[level] ?? level}`,
        `- 분야: ${String(v.domain ?? '')}`,
        `- 제출 시기: ${String(v.timing ?? '')}`,
        `- 저자 현황: ${String(v.author ?? '')}`,
        rejected ? `- 이전 리젝 히스토리:\n${rejected}` : false,
        '',
        '## 요청 사항',
        factors.map((f) => `- ${ST_FACTORS[f] ?? f}`).join('\n'),
        '',
        '## 출력 형식',
        '',
        '### 추천 학회 순위 (Top 5)',
        '각 학회에 대해: 추천 이유, 예상 합격 가능성(높음/중간/낮음), 강조해야 할 포인트, 데드라인',
        '',
        '### 포지셔닝 전략',
        '이 논문의 핵심 메시지를 학회 성격에 맞게 어떻게 프레이밍할지 구체적으로 제시해 주세요.',
        '',
        '### 리스크 & 대비책',
        '가장 가능성 높은 리젝 시나리오 2가지와 각각에 대한 대비 전략을 제시해 주세요.',
      ];
      return lines.filter((l) => l !== false).join('\n');
    },
  },

  // 8. Rebuttal 작성
  {
    id: 'rebuttal',
    title: { ko: 'Rebuttal 작성', en: 'Rebuttal' },
    description: {
      ko: '리뷰어 코멘트에 대한 설득력 있는 rebuttal 전략을 수립하는 프롬프트를 생성합니다.',
      en: 'Generate a rebuttal strategy prompt for reviewer comments.',
    },
    category: 'writing',
    icon: 'message-square-reply',
    tags: ['rebuttal', '리뷰 대응', '학회'],
    fields: [
      {
        id: 'reviews', type: 'textarea', required: true,
        label: { ko: '리뷰어 코멘트 (전체 붙여넣기)', en: 'Reviewer comments (paste all)' },
      },
      {
        id: 'filepath', type: 'text',
        label: { ko: '논문 PDF 경로/파일명 (선택)', en: 'Paper PDF path (optional)' },
      },
      {
        id: 'venue', type: 'text', default: 'NeurIPS',
        label: { ko: '투고 학회', en: 'Venue' },
      },
      {
        id: 'scores', type: 'text',
        label: { ko: '현재 점수 분포', en: 'Current score distribution' },
        placeholder: { ko: '예: 5, 3, 3, 6', en: 'e.g. 5, 3, 3, 6' },
      },
      {
        id: 'strategies', type: 'multiselect',
        label: { ko: 'Rebuttal 전략 (복수 선택)', en: 'Rebuttal strategies (multi)' },
        options: Object.keys(RB_STRATEGY).map((k) => ({ value: k, label: { ko: RB_STRATEGY[k].replace(/해 주세요\.$/, ''), en: k } })),
      },
      {
        id: 'stance', type: 'textarea',
        label: { ko: '우리 입장 / 미리 준비된 반박 (선택)', en: 'Our stance (optional)' },
      },
      {
        id: 'limits', type: 'textarea',
        label: { ko: '제한 사항 (선택)', en: 'Constraints (optional)' },
        placeholder: { ko: '예: 추가 실험 1주일 가능, GPU 2대', en: 'e.g. 1 week for extra experiments, 2 GPUs' },
      },
    ],
    generate: (v) => {
      const reviews = String(v.reviews ?? '').trim();
      const fileRef = String(v.filepath ?? '').trim();
      const fileInstruction = fileRef ? `아래 논문 PDF를 읽어 주세요: ${fileRef}` : '(논문 파일 미입력 — 아래 리뷰 내용만으로 분석해 주세요)';
      const strats = (v.strategies as string[]) ?? Object.keys(RB_STRATEGY);
      const stance = String(v.stance ?? '').trim();
      const limits = String(v.limits ?? '').trim();
      const lines: (string | boolean)[] = [
        '당신은 top-tier AI/ML 학회 rebuttal 전략 전문가입니다.',
        '',
        '## 논문 파일',
        fileInstruction,
        '',
        '## 투고 정보',
        `- 투고 학회: ${String(v.venue ?? 'NeurIPS')}`,
        `- 현재 점수 분포: ${String(v.scores ?? '미입력')}`,
        stance ? `- 저자 사전 입장:\n${stance}` : false,
        limits ? `- 제약 사항: ${limits}` : false,
        '',
        '## 리뷰어 코멘트',
        reviews,
        '',
        '## 요청 사항',
        strats.map((s) => `- ${RB_STRATEGY[s] ?? s}`).join('\n'),
        '',
        '## 출력 시 주의',
        '- 리뷰어를 존중하면서도 사실에 기반해 단호하게 작성',
        '- 추가 실험 결과는 구체적 수치와 함께 제시',
        '- 분량 제한이 있는 경우 우선순위가 높은 코멘트부터 답변',
      ];
      return lines.filter((l) => l !== false).join('\n');
    },
  },

  // 9. (신규) 체계적 문헌 조사
  {
    id: 'literature-review',
    title: { ko: '체계적 문헌 조사', en: 'Systematic Literature Review' },
    description: {
      ko: '주제별로 체계적으로 문헌을 수집·분류·비교하는 survey 프롬프트를 생성합니다.',
      en: 'Generate a systematic literature review / survey prompt.',
    },
    category: 'learning',
    icon: 'library',
    tags: ['survey', '문헌', '비교'],
    fields: [
      {
        id: 'topic', type: 'textarea', required: true,
        label: { ko: '조사 주제 / 질문', en: 'Review topic / question' },
        placeholder: { ko: '예: LLM의 hallucination 평가 방법론의 발전', en: 'e.g. Evolution of LLM hallucination evaluation' },
      },
      {
        id: 'queries', type: 'text',
        label: { ko: '검색어 (쉼표 구분)', en: 'Search keywords' },
        placeholder: { ko: '예: hallucination, factuality, grounding', en: 'e.g. hallucination, factuality, grounding' },
      },
      {
        id: 'period', type: 'select', default: '5y',
        label: { ko: '조사 기간', en: 'Period' },
        options: [
          { value: '2y', label: { ko: '최근 2년', en: 'Last 2 years' } },
          { value: '5y', label: { ko: '최근 5년', en: 'Last 5 years' } },
          { value: 'all', label: { ko: '전 기간', en: 'All time' } },
        ],
      },
      {
        id: 'angles', type: 'multiselect',
        label: { ko: '다룰 관점 (복수 선택)', en: 'Angles (multi)' },
        options: [
          { value: 'taxonomy', label: { ko: '분류 체계(taxonomy) 구축', en: 'Build taxonomy' } },
          { value: 'timeline', label: { ko: '시간 흐름(발전사)', en: 'Timeline / evolution' } },
          { value: 'compare', label: { ko: '방법론 비교표', en: 'Method comparison table' } },
          { value: 'benchmark', label: { ko: '벤치마크/지표 비교', en: 'Benchmark/metric comparison' } },
          { value: 'gap', label: { ko: '미해결 과제(Gap)', en: 'Open challenges' } },
          { value: 'future', label: { ko: '향후 연구 방향', en: 'Future directions' } },
        ],
      },
      {
        id: 'depth', type: 'select', default: 'full',
        label: { ko: '분량', en: 'Depth' },
        options: [
          { value: 'brief', label: { ko: '간단 요약 (10편 내외)', en: 'Brief (~10)' } },
          { value: 'full', label: { ko: '본격 survey (20~30편)', en: 'Full survey (20-30)' } },
          { value: 'exhaustive', label: { ko: '포괄적 (40편+)', en: 'Exhaustive (40+)' } },
        ],
      },
    ],
    generate: (v) => {
      const topic = String(v.topic ?? '').trim();
      const queries = String(v.queries ?? '').trim();
      const angles = (v.angles as string[]) ?? ['taxonomy', 'compare', 'gap', 'future'];
      const angleMap: Record<string, string> = {
        taxonomy: '- 연구들을 체계적으로 분류하는 taxonomy(트리)를 제시',
        timeline: '- 주요 접근법의 시간적 발전 흐름을 타임라인으로 정리',
        compare: '- 핵심 방법론 비교표 (접근법 / 데이터 / 지표 / 장단점)',
        benchmark: '- 사용된 벤치마크·지표의 발전과 한계 비교',
        gap: '- 현재 연구 동향에서 해결되지 않은 Gap 정리',
        future: '- 향후 연구 방향과 유망한 접근 제안',
      };
      const periodMap: Record<string, string> = { '2y': '최근 2년(2024-2025) 중심', '5y': '최근 5년(2020-2025)', all: '전 기간' };
      return `당신은 AI/ML 분야의 체계적 문헌 조사(systematic review) 전문가입니다.\n\n## 조사 주제\n${topic}\n${queries ? `\n## 검색 키워드\n${queries}` : ''}\n\n## 조사 범위\n- 기간: ${periodMap[String(v.period ?? '5y')]}\n- 분량: ${String(v.depth ?? 'full')}\n\n## 요청 사항\n1. 먼저 해당 주제의 핵심 논문들을 ${v.depth === 'exhaustive' ? '40편 이상' : v.depth === 'brief' ? '10편 내외' : '20~30편'} 선정. 각 논문: 제목(저자, 연도), 핵심 기여 1문장, 한계 1문장.\n2. 아래 관점들을 포함해 survey를 구성:\n${angles.map((a) => angleMap[a] ?? a).join('\n')}\n3. 인용은 실제 존재하는 논문만 사용. 불확실하면 "[verification needed]" 표시. 절대 허위 인용을 만들지 마세요.\n\n## 출력 형식 (마크다운)\n- # 조사 개요\n- # 핵심 논문 목록\n- # 분류 체계(taxonomy)\n- # 비교표\n- # 미해결 과제(Gap)\n- # 향후 연구 방향`;
    },
  },

  // 10. (신규) 연구과제 제안서
  {
    id: 'grant-proposal',
    title: { ko: '연구과제 제안서', en: 'Research Proposal' },
    description: {
      ko: '연구과제/지원금 제안서 초안을 작성하는 프롬프트를 생성합니다.',
      en: 'Generate a research grant proposal drafting prompt.',
    },
    category: 'writing',
    icon: 'file-text',
    tags: ['제안서', '과제', '기획'],
    fields: [
      { id: 'title', type: 'text', required: true, label: { ko: '과제명', en: 'Project title' } },
      {
        id: 'type', type: 'select', default: 'gov',
        label: { ko: '과제 유형', en: 'Proposal type' },
        options: [
          { value: 'gov', label: { ko: '정부 R&D (국과패/연재단 등)', en: 'Government R&D' } },
          { value: 'industry', label: { ko: '산학 협력', en: 'Industry collaboration' } },
          { value: 'foundation', label: { ko: '재단/장학', en: 'Foundation grant' } },
          { value: 'internal', label: { ko: '교내 신진연구', en: 'Internal seed' } },
        ],
      },
      { id: 'necessity', type: 'textarea', required: true, label: { ko: '연구 필요성 / 배경', en: 'Background & necessity' } },
      { id: 'goal', type: 'textarea', required: true, label: { ko: '연구 목표', en: 'Objectives' } },
      { id: 'approach', type: 'textarea', label: { ko: '접근법 / 방법론', en: 'Approach / methodology' } },
      { id: 'outcomes', type: 'textarea', label: { ko: '예상 성과 (논문·특허·SW 등)', en: 'Expected outcomes' } },
      {
        id: 'budget', type: 'text',
        label: { ko: '예산 / 기간', en: 'Budget / duration' },
        placeholder: { ko: '예: 2억원 / 2년', en: 'e.g. 200M KRW / 2 years' },
      },
    ],
    generate: (v) => {
      const typeMap: Record<string, string> = {
        gov: '정부 R&D 과제 (심사 기준: 기술성·파급효과·수행능력·예산 적정성)',
        industry: '산학 협력 과제 (심사 기준: 실용성·기술 이전 가능성·예산 타당성)',
        foundation: '재단 지원 과제 (심사 기준: 학술 기여·독창성·실현 가능성)',
        internal: '교내 신진/기초 연구 (심사 기준: 파급력·후속 연구 가능성)',
      };
      const budget = String(v.budget ?? '').trim();
      const lines: (string | boolean)[] = [
        `당신은 ${typeMap[String(v.type ?? 'gov')]} 제안서 작성 전문가입니다. 아래 정보로 제안서 초안을 작성해 주세요.`,
        '',
        `## 과제명\n${String(v.title ?? '').trim()}`,
        '',
        '## 작성 지시사항',
        '1. **연구 개발 필요성**: 사회적·기술적 배경, 기존 한계, 본 과제의 필요성을 설득력 있게 서술 (1.5~2쪽 분량)',
        `   - 입력 배경: ${String(v.necessity ?? '').trim()}`,
        '2. **연구 개발 목표 및 내용**: 최종 목표, 단계별 목표, 핵심 연구 내용',
        `   - 입력 목표: ${String(v.goal ?? '').trim()}`,
        `3. **연구 방법 / 추진 전략**:${String(v.approach ?? '').trim() ? `\n   ${String(v.approach ?? '').trim()}` : ' (입력된 접근법을 바탕으로 구체적 실험·개발 단계를 설계)'}`,
        `4. **기대 효과**: ${String(v.outcomes ?? '').trim() || '논문·특허·SW·사회·경제적 파급효과를 정량·정성으로 제시'}`,
        budget ? `5. **예산·일정**: ${budget}에 맞춰 연차별 추진 일정(gantt 형식 텍스트)과 예산 항목을 구성` : false,
        '',
        '## 출력 포맷',
        '- 마크다운 헤더로 각 장을 구분',
        '- 각 장 끝에 "[작성자 확인 필요]" 표시로 사용자가 보강할 부분 표시',
        '- 전문적이고 설득력 있는 어조 (정부/재단 제안서 스타일)',
      ];
      return lines.filter((l) => l !== false).join('\n');
    },
  },

  // 11. (신규) 데이터셋/벤치마크 설계
  {
    id: 'dataset-design',
    title: { ko: '데이터셋/벤치마크 설계', en: 'Dataset & Benchmark Design' },
    description: {
      ko: '연구용 데이터셋 또는 벤치마크를 설계하는 프롬프트를 생성합니다.',
      en: 'Generate a dataset/benchmark design prompt.',
    },
    category: 'ideation',
    icon: 'database',
    tags: ['데이터셋', '벤치마크', '평가'],
    fields: [
      { id: 'purpose', type: 'textarea', required: true, label: { ko: '데이터셋 목적 / 평가 대상', en: 'Purpose / what to evaluate' } },
      { id: 'scale', type: 'text', label: { ko: '목표 규모', en: 'Target scale' }, placeholder: { ko: '예: 10K 샘플', en: 'e.g. 10K samples' } },
      { id: 'source', type: 'textarea', label: { ko: '데이터 출처 / 수집 방법', en: 'Source / collection method' } },
      {
        id: 'labeling', type: 'select', default: 'human',
        label: { ko: '라벨링 방식', en: 'Labeling' },
        options: [
          { value: 'human', label: { ko: '사람 라벨링', en: 'Human annotation' } },
          { value: 'auto', label: { ko: '자동/규칙 기반', en: 'Automatic / rule-based' } },
          { value: 'weak', label: { ko: '약한 지도(weak)', en: 'Weak supervision' } },
          { value: 'llm', label: { ko: 'LLM 보조 + 사람 검수', en: 'LLM-assisted + human' } },
        ],
      },
      {
        id: 'metrics', type: 'multiselect',
        label: { ko: '평가 지표 후보 (복수 선택)', en: 'Candidate metrics (multi)' },
        options: ['Accuracy/F1', 'BLEU/ROUGE', 'BERTScore', 'Human eval', 'Pass@k', 'Latency/Throughput'].map((m) => ({ value: m, label: { ko: m, en: m } })),
      },
      {
        id: 'ethics', type: 'textarea',
        label: { ko: '윤리/라이선스/개인정보 고려 (선택)', en: 'Ethics / license / privacy (optional)' },
      },
    ],
    generate: (v) => {
      const metrics = (v.metrics as string[]) ?? [];
      const ethics = String(v.ethics ?? '').trim();
      const labelMap: Record<string, string> = {
        human: '사람 전문가 annotation ( annotator 간 합의/IAG 측정 포함)',
        auto: '자동/규칙 기반 라벨링 (검증 샘플로 정확도 측정)',
        weak: '약한 지도 학습 (다수 라벨러 헹김·합성)',
        llm: 'LLM 보조 라벨링 + 사람 검수(human-in-the-loop)',
      };
      return `당신은 AI/ML 연구용 데이터셋·벤치마크 설계 전문가입니다.\n\n## 설계 목적\n${String(v.purpose ?? '').trim()}\n\n## 기본 설계\n- 목표 규모: ${String(v.scale ?? '미정')}\n- 출처/수집: ${String(v.source ?? '미입력')}\n- 라벨링: ${labelMap[String(v.labeling ?? 'human')]}\n${metrics.length ? `- 평가 지표 후보: ${metrics.join(', ')}` : ''}\n\n## 요청 사항\n1. **스키마 설계**: 필드별 정의, 데이터 타입, 예시 3~5행\n2. **구성 전략**: train/val/test 분할 원칙, 난이도·도메인 분포, 편향 방지 설계\n3. **품질 관리**: annotator 가이드라인 초안, IAG(annotator agreement) 목표, 데이터 누수 방지\n4. **평가 프로토콜**: 리더보드 구성, 제출 형식, 순위 산정 방식${metrics.length ? '' : '\n   (지표 후보가 미입력이면 이 작업에 적합한 지표를 추천)'}\n5. **윤리·라이선스**: ${ethics || '수집 출처의 라이선스, PII 마스킹, 공개 범위, 사용 약관을 설계'}\n\n## 출력\n- 마크다운 형식, 각 항목별 구체적 산출물 제시\n- Datasheets for Datasets 규격을 참고해 "데이터 시트" 초안도 포함`;
    },
  },

  // 12. (신규) 발표 자료 구조 설계
  {
    id: 'presentation',
    title: { ko: '발표 자료 구조 설계', en: 'Presentation Structure' },
    description: {
      ko: '학회 발표/강연 슬라이드의 스토리와 구조를 설계하는 프롬프트를 생성합니다.',
      en: 'Generate a presentation structure/storyline prompt.',
    },
    category: 'writing',
    icon: 'presentation',
    tags: ['발표', '슬라이드', '스토리'],
    fields: [
      { id: 'topic', type: 'textarea', required: true, label: { ko: '발표 주제 / 핵심 내용', en: 'Topic / core content' } },
      {
        id: 'audience', type: 'select', default: 'peer',
        label: { ko: '청중', en: 'Audience' },
        options: [
          { value: 'peer', label: { ko: '동료 연구자 (학회)', en: 'Peer researchers (venue)' } },
          { value: 'general', label: { ko: '일반 청중', en: 'General audience' } },
          { value: 'student', label: { ko: '학생/교육', en: 'Students' } },
          { value: 'exec', label: { ko: '경영진/비기술', en: 'Executives / non-technical' } },
        ],
      },
      {
        id: 'duration', type: 'select', default: '15',
        label: { ko: '발표 시간', en: 'Duration' },
        options: [
          { value: '5', label: { ko: '5분 (Lightning)', en: '5 min' } },
          { value: '15', label: { ko: '15분', en: '15 min' } },
          { value: '30', label: { ko: '30분', en: '30 min' } },
          { value: '60', label: { ko: '60분 (강연)', en: '60 min' } },
        ],
      },
      { id: 'message', type: 'textarea', label: { ko: '전달하고 싶은 핵심 메시지 (1~3)', en: 'Key messages' } },
      {
        id: 'visuals', type: 'multiselect',
        label: { ko: '포함할 시각 자료 (복수 선택)', en: 'Visuals (multi)' },
        options: [
          { value: 'diagram', label: { ko: '개념도/아키텍처', en: 'Concept/architecture diagram' } },
          { value: 'table', label: { ko: '결과 표', en: 'Result tables' } },
          { value: 'chart', label: { ko: '성능 그래프', en: 'Performance charts' } },
          { value: 'demo', label: { ko: '데모/코드', en: 'Demo/code' } },
          { value: 'example', label: { ko: '사례/예시', en: 'Examples' } },
        ],
      },
    ],
    generate: (v) => {
      const audMap: Record<string, string> = {
        peer: '동료 연구자 (기술적 깊이, 수식/벤치마크 허용)',
        general: '일반 청중 (전문 용어 최소화, 직관적 비유)',
        student: '학생 (개념 중심, 단계적 빌드업)',
        exec: '경영진 (의사결정 관점, 비즈니스 임팩트)',
      };
      const durMap: Record<string, number> = { '5': 6, '15': 14, '30': 25, '60': 45 };
      const slides = durMap[String(v.duration ?? '15')] ?? 14;
      const message = String(v.message ?? '').trim();
      const visuals = (v.visuals as string[]) ?? [];
      const visMap: Record<string, string> = {
        diagram: '핵심 개념/아키텍처 다이어그램',
        table: '결과 비교 표',
        chart: '성능 그래프',
        demo: '데모/코드 스크린',
        example: '구체 사례',
      };
      return `당신은 연구 발표 스토리텔링 전문가입니다.\n\n## 발표 정보\n- 주제: ${String(v.topic ?? '').trim()}\n- 청중: ${audMap[String(v.audience ?? 'peer')]}\n- 시간: ${String(v.duration ?? '15')}분 (약 ${slides}장 슬라이드 권장)${message ? `\n- 핵심 메시지:\n${message}` : ''}\n${visuals.length ? `\n## 포함할 시각 자료\n${visuals.map((x) => `- ${visMap[x]}`).join('\n')}` : ''}\n\n## 요청 사항\n1. **전체 스토리라인**: hook → problem → 기존 한계 → 우리 접근 → 결과 → 의의/활용의 흐름으로 설계\n2. **슬라이드별 구성표**: 총 ${slides}장에 대해 각 슬라이드의 (제목 / 핵심 메시지 1문장 / 시각 자료 / 발화 스크립트 요지)를 표로 제시\n3. **첫 3장 집중 설계**: 청중의 주의를 끄는 오프닝과 문제 정의를 구체적으로\n4. **마지막 2장**: take-away와 명확한 CTA(다음 단계/Q&A 유도)\n5. **발화 스크립트**: 발표 시간에 맞춘 분량으로, 슬라이드별 2~4문장 발화 가이드\n\n## 출력\n- 마크다운, 슬라이드별 헤더 사용\n- 청중 수준에 맞는 난이도 조절 명시`;
    },
  },

  // 13. (신규) 개념 설명 / 튜터
  {
    id: 'explain',
    title: { ko: '개념 설명 / 튜터', en: 'Concept Explanation / Tutor' },
    description: {
      ko: '어려운 개념을 학습자 수준에 맞춰 설명하는 튜터 프롬프트를 생성합니다.',
      en: 'Generate a personalized concept-explanation tutor prompt.',
    },
    category: 'learning',
    icon: 'graduation-cap',
    tags: ['학습', '설명', '튜터'],
    fields: [
      { id: 'concept', type: 'text', required: true, label: { ko: '설명할 개념', en: 'Concept to explain' }, placeholder: { ko: '예: Attention mechanism', en: 'e.g. Attention mechanism' } },
      {
        id: 'level', type: 'select', default: 'intermediate',
        label: { ko: '학습자 수준', en: 'Learner level' },
        options: [
          { value: 'beginner', label: { ko: '초급 (처음 접함)', en: 'Beginner' } },
          { value: 'intermediate', label: { ko: '중급 (기초 지식 있음)', en: 'Intermediate' } },
          { value: 'advanced', label: { ko: '고급 (전공자/연구자)', en: 'Advanced' } },
        ],
      },
      { id: 'prereq', type: 'text', label: { ko: '선수 지식', en: 'Prerequisites' }, placeholder: { ko: '예: 선형대수학 기초', en: 'e.g. basic linear algebra' } },
      {
        id: 'depth', type: 'select', default: 'intuition',
        label: { ko: '원하는 깊이', en: 'Depth' },
        options: [
          { value: 'intuition', label: { ko: '직관 위주', en: 'Intuition-first' } },
          { value: 'balanced', label: { ko: '직관+수식', en: 'Intuition + math' } },
          { value: 'rigorous', label: { ko: '엄밀한 유도', en: 'Rigorous derivation' } },
        ],
      },
      {
        id: 'analogy', type: 'select', default: 'yes',
        label: { ko: '비유/예시 허용', en: 'Analogy allowed' },
        options: [
          { value: 'yes', label: { ko: '적극 활용', en: 'Yes, use analogies' } },
          { value: 'no', label: { ko: '비유 자제', en: 'No, keep literal' } },
        ],
      },
      { id: 'goal', type: 'textarea', label: { ko: '학습 목표 (선택)', en: 'Learning goal (optional)' } },
    ],
    generate: (v) => {
      const levelMap: Record<string, string> = {
        beginner: '초급 — 처음 접하는 학습자. 전문 용어를 처음 사용할 때마다 쉽게 정의.',
        intermediate: '중급 — 기초는 알지만 심화가 필요한 학습자.',
        advanced: '고급 — 전공자/연구자. 최신 논문·수식·한계까지 다룸.',
      };
      const depthMap: Record<string, string> = {
        intuition: '직관과 그림·비유로 먼저 설명한 뒤 간결히 정리',
        balanced: '직관 → 수식 → 코드 순으로 점진적 빌드업',
        rigorous: '정의·정리·증명을 포함한 엄밀한 전개',
      };
      const prereq = String(v.prereq ?? '').trim();
      const goal = String(v.goal ?? '').trim();
      return `당신은 뛰어난 1:1 튜터입니다. 아래 학습자에게 맞춰 개념을 설명해 주세요.\n\n## 학습 정보\n- 개념: ${String(v.concept ?? '').trim()}\n- 수준: ${levelMap[String(v.level ?? 'intermediate')]}\n${prereq ? `- 선수 지식: ${prereq}` : '- 선수 지식: 없음 (처음부터 설명)'}\n- 설명 깊이: ${depthMap[String(v.depth ?? 'intuition')]}\n- 비유/예시: ${v.analogy === 'no' ? '자제 (정확한 정의 중심)' : '적극 활용'}${goal ? `\n- 학습 목표: ${goal}` : ''}\n\n## 요청 사항\n1. **한 줄 요약**: 이 개념이 무엇인지 한 문장으로\n2. **왜 필요한가**: 해결하는 문제 / 등장 배경\n3. **직관적 설명**: 일상 비유 또는 시각화로 핵심 아이디어 전달\n4. **정식 설명**: 정의, 핵심 구성 요소, 작동 원리${v.depth !== 'intuition' ? '\n5. **수식/코드**: 핵심 수식과 (가능하면) Python/PyTorch 최소 코드' : ''}\n6. **예제**: 단계별로 풀어보는 작은 예 1~2개\n7. **흔한 오해 & 팁**: 학습자가 자주 헷갈리는 점과 주의점\n8. **점검 질문**: 학습을 확인할 수 있는 질문 3개\n\n## 톤\n- 친절하고 격려하는 어조, 대화형(질문을 던지며 점검)\n- 어려운 부분은 더 천천히, 여러 각도에서 재설명`;
    },
  },
];
