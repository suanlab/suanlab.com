// Editorial order is intentional. Keep IDs stable when adding newer records.
export const featuredPublications = [
  { id: 652, reasonKo: '신경망 방법론: 물리 기반 모델의 구조와 학습 문제를 정리한 리뷰', reasonEn: 'Neural methods: a review of physics-informed architectures and training challenges' },
  { id: 653, reasonKo: '생성 모델: 환경 변화에 따른 Wi-Fi 센싱과 이미지 생성 평가', reasonEn: 'Generative models: evaluating Wi-Fi sensing across environments' },
  { id: 654, reasonKo: '응용 AI: Wi-Fi 신호를 이용한 비접촉 호흡 상태 분류', reasonEn: 'Applied AI: contactless respiratory-state classification with Wi-Fi signals' },
];

// Use declared project status; dates alone do not establish completion.
export const featuredProjectIds = [50, 2];

export interface ResearchLinks {
  publications: number[];
  projects: number[];
  lectures: string[];
}

// Topic relationships, not claims that a publication was funded by a project.
export const researchLinks: Record<string, ResearchLinks> = {
  ds: { publications: [205, 212], projects: [5, 18], lectures: ['bd', 'db', 'ml'] },
  dl: { publications: [652, 203], projects: [3, 11], lectures: ['dl', 'ml', 'ai'] },
  nlp: { publications: [214, 215, 217], projects: [50, 6, 7], lectures: ['nlp', 'dl'] },
  cv: { publications: [653, 201, 208], projects: [2, 14, 19], lectures: ['cv', 'dl'] },
  graphs: { publications: [200, 211, 205], projects: [1], lectures: ['ai', 'ml'] },
  st: { publications: [654, 207, 219], projects: [8, 10, 11], lectures: ['ml', 'dl'] },
  asp: { publications: [218], projects: [15, 16], lectures: ['asp', 'dl'] },
};

export interface LearningPath {
  id: string;
  titleKo: string;
  titleEn: string;
  prerequisiteKo: string;
  prerequisiteEn: string;
  steps: { levelKo: string; levelEn: string; labelKo: string; labelEn: string; href: string }[];
}

export const learningPaths: LearningPath[] = [
  {
    id: 'foundations', titleKo: 'Python에서 AI 기초까지', titleEn: 'Python to AI foundations',
    prerequisiteKo: '처음 시작하는 학습자', prerequisiteEn: 'For first-time learners',
    steps: [
      { levelKo: '입문', levelEn: 'Beginner', labelKo: 'Python 프로그래밍 영상', labelEn: 'Python programming videos', href: '/youtube/pp/' },
      { levelKo: '기초', levelEn: 'Foundation', labelKo: 'AI 입문 강의 자료', labelEn: 'AI introductory lecture notes', href: '/lecture/ai/notes/' },
      { levelKo: '응용', levelEn: 'Applied', labelKo: '머신러닝 강의', labelEn: 'Machine learning course', href: '/lecture/ml/' },
      { levelKo: '읽기', levelEn: 'Reading', labelKo: '머신러닝 온라인 도서 목차', labelEn: 'Machine learning book outline', href: '/book/online/machine-learning-fundamentals/' },
    ],
  },
  {
    id: 'language', titleKo: '딥러닝에서 언어 모델까지', titleEn: 'Deep learning to language models',
    prerequisiteKo: 'Python과 머신러닝 기초', prerequisiteEn: 'Python and machine learning basics',
    steps: [
      { levelKo: '기초', levelEn: 'Foundation', labelKo: '딥러닝 강의', labelEn: 'Deep learning course', href: '/lecture/dl/' },
      { levelKo: '응용', levelEn: 'Applied', labelKo: '자연어처리 영상', labelEn: 'NLP videos', href: '/youtube/nlp/' },
      { levelKo: '읽기', levelEn: 'Reading', labelKo: '자연어처리 온라인 도서 목차', labelEn: 'NLP book outline', href: '/book/online/natural-language-processing/' },
      { levelKo: '연구', levelEn: 'Research', labelKo: '자연어처리 연구와 논문', labelEn: 'NLP research and publications', href: '/research/nlp/' },
    ],
  },
  {
    id: 'vision', titleKo: '시각 데이터에서 컴퓨터 비전까지', titleEn: 'Visual data to computer vision',
    prerequisiteKo: 'Python과 신경망 기초', prerequisiteEn: 'Python and neural network basics',
    steps: [
      { levelKo: '기초', levelEn: 'Foundation', labelKo: '딥러닝 영상', labelEn: 'Deep learning videos', href: '/youtube/dl/' },
      { levelKo: '응용', levelEn: 'Applied', labelKo: '컴퓨터 비전 강의', labelEn: 'Computer vision course', href: '/lecture/cv/' },
      { levelKo: '읽기', levelEn: 'Reading', labelKo: '컴퓨터 비전 온라인 도서 목차', labelEn: 'Computer vision book outline', href: '/book/online/computer-vision/' },
      { levelKo: '연구', levelEn: 'Research', labelKo: '컴퓨터 비전 연구와 논문', labelEn: 'Computer vision research and publications', href: '/research/cv/' },
    ],
  },
];
