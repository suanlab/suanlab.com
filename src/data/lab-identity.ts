export interface LabIdentity {
  unifying: string;
  approach: string;
  titleKo: string;
  descriptionKo: string;
  descriptionEn: string;
}

// SUANLAB interpretations supplied for the homepage's research identity.
export const labIdentities: LabIdentity[] = [
  {
    unifying: 'Unifying', approach: 'Architectural',
    titleKo: '초지능 통합 구조적 신경망 연구실',
    descriptionKo: '대규모 언어 모델(LLM)과 검색 증강 생성(RAG) 등 다양한 AI 기술을 하나로 엮어내어, 산업 현장에 바로 적용 가능한 거시적이고 안정적인 시스템 아키텍처를 설계합니다.',
    descriptionEn: 'We connect large language models, retrieval-augmented generation, and other AI technologies to design robust system architectures ready for industrial applications.',
  },
  {
    unifying: 'Understanding', approach: 'Analytical',
    titleKo: '초지능 이해 기반 분석적 신경망 연구실',
    descriptionKo: '복잡한 자연어의 맥락을 깊이 이해하고 분석하여, 비정형 텍스트를 데이터베이스 질의로 변환하거나 정교한 금융·산업 데이터를 다루는 특화된 연구를 수행합니다.',
    descriptionEn: 'We study the context of complex natural language, transforming unstructured text into database queries and analyzing specialized financial and industrial data.',
  },
  {
    unifying: 'Uncovering', approach: 'Advanced',
    titleKo: '초지능 발굴형 진보 신경망 연구실',
    descriptionKo: '음성, 텍스트, 비전 등 멀티모달 데이터 속에서 핵심적인 인사이트를 찾아내어, 다면적인 역량 평가나 피드백 시스템을 구축하는 고도화된 기술력을 강조합니다.',
    descriptionEn: 'We uncover insights in speech, text, and vision to build advanced multimodal assessment and feedback systems.',
  },
  {
    unifying: 'Unifying', approach: 'Applied',
    titleKo: '초지능 통합 응용 신경망 연구실',
    descriptionKo: '다양한 인공지능 요소 기술을 통합하여, 실제 공공서비스나 비즈니스 환경의 복잡한 요구사항을 즉각적으로 해결하는 실무 중심의 응용 AI 솔루션을 개발합니다.',
    descriptionEn: 'We integrate AI technologies into practical solutions for complex requirements in public services and business environments.',
  },
  {
    unifying: 'Unleashing', approach: 'Autonomous',
    titleKo: '초지능 자율형 신경망 촉발 연구실',
    descriptionKo: '스스로 판단하고 행동하는 자율형 AI 에이전트들의 잠재력을 폭발적으로 끌어내어, 사람의 개입을 최소화한 차세대 자동화 프레임워크를 연구합니다.',
    descriptionEn: 'We explore the potential of autonomous AI agents that reason and act independently, researching next-generation automation with less manual intervention.',
  },
  {
    unifying: 'User-aligned', approach: 'Adaptive',
    titleKo: '초지능 사용자 정렬 적응형 신경망 연구실',
    descriptionKo: 'AI가 사용자의 의도나 학습자의 수준에 완벽히 부합하도록 유연하게 적응하는 맞춤형 인공지능 지원 플랫폼 방법론을 탐구합니다.',
    descriptionEn: 'We explore personalized AI support platforms that adapt to user intent and each learner’s level.',
  },
  {
    unifying: 'Unbounded', approach: 'Agentic',
    titleKo: '초지능 경계 없는 에이전트 신경망 연구실',
    descriptionKo: '단일 모델의 한계를 넘어, 여러 에이전트가 상호작용하며 복합적인 문제를 해결하는 최신 에이전트 기반 시스템을 설계합니다.',
    descriptionEn: 'We design agentic systems in which multiple agents interact to solve complex problems beyond the capabilities of a single model.',
  },
  {
    unifying: 'Universal', approach: 'Agile',
    titleKo: '초지능 범용 민첩형 신경망 연구실',
    descriptionKo: '특정 도메인에 국한되지 않는 범용적인 데이터 사이언스 역량을 바탕으로, 급변하는 기술 트렌드와 새로운 프로젝트에 매우 빠르고 유연하게 대응합니다.',
    descriptionEn: 'We apply broad data science expertise across domains, responding quickly and flexibly to changing technologies and new projects.',
  },
  {
    unifying: 'Utilizing', approach: 'Augmented',
    titleKo: '초지능 증강 신경망 활용 연구실',
    descriptionKo: '인공지능을 통해 인간의 인지와 의사결정 능력을 실질적으로 증강시키는 데 초점을 맞추며, 실제 유틸리티를 만들어내는 실용주의를 표방합니다.',
    descriptionEn: 'We focus on practical AI that augments human cognition and decision-making, creating tools with real utility.',
  },
  {
    unifying: 'Unlocking', approach: 'Aligned',
    titleKo: '초지능 정렬된 신경망 잠재력 해제 연구실',
    descriptionKo: '신뢰할 수 있고 목적에 잘 정렬된 AI 기술의 진정한 가치를 산업 전반에 풀어내는 선도적인 역할을 의미합니다.',
    descriptionEn: 'We aim to unlock the value of trustworthy, purpose-aligned AI across industries.',
  },
  {
    unifying: 'Unraveling', approach: 'Algorithmic',
    titleKo: '초지능 알고리즘 해명 신경망 연구실',
    descriptionKo: '복잡하게 얽힌 데이터와 알고리즘의 원리를 명확히 풀어내어, 투명하고 해석 가능한 고성능 AI 모델을 연구하는 학술적 깊이를 강조합니다.',
    descriptionEn: 'We investigate the principles of complex data and algorithms to develop transparent, interpretable, high-performance AI models.',
  },
  {
    unifying: 'Ultimate', approach: 'Adaptive',
    titleKo: '초지능 궁극의 적응형 신경망 연구실',
    descriptionKo: '어떠한 엣지 케이스나 새로운 데이터 환경에서도 성능 저하 없이 스스로 최적화하는 궁극적 수준의 적응형 모델을 지향합니다.',
    descriptionEn: 'We pursue adaptive models that optimize themselves for edge cases and new data environments while aiming to preserve performance.',
  },
  {
    unifying: 'Unveiling', approach: 'Actionable',
    titleKo: '초지능 실행 가능 신경망 규명 연구실',
    descriptionKo: '방대한 데이터 속에서 단순히 패턴을 찾는 것을 넘어, 실제 비즈니스나 서비스 기획에 즉시 실행 가능한 인사이트를 밝혀냅니다.',
    descriptionEn: 'We go beyond finding patterns in large datasets to reveal actionable insights for business and service design.',
  },
  {
    unifying: 'User-centric', approach: 'Autonomous',
    titleKo: '초지능 사용자 중심 자율형 신경망 연구실',
    descriptionKo: '고도의 자율성을 가진 AI 시스템을 설계하되, 그 중심에는 항상 사용자의 편의와 경험을 최우선으로 두는 휴먼 인 더 루프(HITL) 철학을 담습니다.',
    descriptionEn: 'We design highly autonomous AI around user experience, keeping people involved through a human-in-the-loop approach.',
  },
  {
    unifying: 'Understanding', approach: 'Agentic',
    titleKo: '초지능 이해 기반 에이전트 신경망 연구실',
    descriptionKo: '사람의 언어와 의도를 깊이 이해하고, 이를 바탕으로 스스로 작업 계획을 수립하고 실행하는 지능형 에이전트 시스템을 집중적으로 구현합니다.',
    descriptionEn: 'We build intelligent agents that understand human language and intent, then plan and execute tasks independently.',
  },
  {
    unifying: 'Unleashing', approach: 'Accelerating',
    titleKo: '초지능 가속형 신경망 촉발 연구실',
    descriptionKo: '데이터 처리 속도, 모델의 추론 성능, 혹은 사용자의 업무 효율성을 비약적으로 가속시키는 혁신적인 프레임워크 연구에 집중합니다.',
    descriptionEn: 'We research frameworks that accelerate data processing, model inference, and the efficiency of people’s work.',
  },
  {
    unifying: 'Uncovering', approach: 'Adaptable',
    titleKo: '초지능 유연한 적응형 신경망 발굴 연구실',
    descriptionKo: '다양한 산업군의 변화하는 요구사항에 맞춰 유연하게 변형하고 적용할 수 있는 새로운 인공지능 방법론과 구조를 끊임없이 발굴합니다.',
    descriptionEn: 'We discover AI methods and architectures that can be adapted to the changing needs of different industries.',
  },
  {
    unifying: 'Utilizing', approach: 'Advanced',
    titleKo: '초지능 진보 신경망 활용 연구실',
    descriptionKo: '최첨단 생성형 AI 기술과 프롬프트 엔지니어링 기법을 가장 빠르게 도입하고 활용하여 독창적인 서비스를 만들어내는 랩의 정체성을 보여줍니다.',
    descriptionEn: 'We put advanced generative AI and prompt engineering into practice to create original services.',
  },
  {
    unifying: 'Unlocking', approach: 'Architectural',
    titleKo: '초지능 구조적 잠재력 해제 신경망 연구실',
    descriptionKo: '기존 AI 시스템이 가진 구조적 한계를 돌파하고, 새로운 시스템 아키텍처 설계를 통해 이전에는 불가능했던 성능과 확장성의 잠재력을 열어젖힙니다.',
    descriptionEn: 'We explore new system architectures to overcome structural limits and unlock greater performance and scalability in AI.',
  },
];

export function labIdentityName(identity: LabIdentity): string {
  return `Superintelligence ${identity.unifying} ${identity.approach} Neural-networks LAB`;
}
