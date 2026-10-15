import type { ConferenceVenue } from '@/lib/publication-groups';

// Homepage Top Conf selection; editorial order breaks ties within the same month.
// Add venues explicitly. Publication dates alone do not establish eligibility.
export const featuredConferenceVenues: ConferenceVenue[] = [
  { id: 'neurips', acronym: 'NeurIPS', name: 'Conference on Neural Information Processing Systems', homepageUrl: 'https://neurips.cc/Conferences/2026', aliases: ['NIPS'] },
  { id: 'accv', acronym: 'ACCV', name: 'Asian Conference on Computer Vision', homepageUrl: 'https://accv2026.org/' },
  { id: 'aacl-ijcnlp', acronym: 'AACL-IJCNLP', name: 'Asia Pacific Chapter of ACL and IJCNLP Joint Conference', homepageUrl: 'https://2026.aaclnet.org/' },
  { id: 'icdm', acronym: 'ICDM', name: 'IEEE International Conference on Data Mining', homepageUrl: 'https://www3.cs.stonybrook.edu/~icdm2026/index.html' },
  { id: 'emnlp', acronym: 'EMNLP', name: 'Conference on Empirical Methods in Natural Language Processing', homepageUrl: 'https://2026.emnlp.org/' },
  { id: 'ecml-pkdd', acronym: 'ECML-PKDD', name: 'European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases', homepageUrl: 'https://ecmlpkdd.org/', aliases: ['ECML PKDD'] },
  { id: 'uai', acronym: 'UAI', name: 'Conference on Uncertainty in Artificial Intelligence', homepageUrl: 'https://auai.org/uai2026/' },
  { id: 'kdd', acronym: 'KDD', name: 'ACM SIGKDD Conference on Knowledge Discovery and Data Mining', homepageUrl: 'https://kdd2026.kdd.org/' },
  { id: 'cvpr', acronym: 'CVPR', name: 'IEEE/CVF Conference on Computer Vision and Pattern Recognition', homepageUrl: 'https://cvpr.thecvf.com/Conferences/2026' },
  { id: 'aaai', acronym: 'AAAI', name: 'AAAI Conference on Artificial Intelligence', homepageUrl: 'https://aaai.org/conference/aaai/aaai-26/' },
];
// Use declared project status; dates alone do not establish completion.
export const featuredProjectIds = [50, 2];

// Summaries paraphrase the registered project scope. No benchmark results are inferred.
export const projectSummaries: Record<number, { problem: string; approach: string; outcome: string }> = {
  50: {
    problem: '모델을 재학습하지 않고 AI 생성 텍스트를 식별하는 워터마킹과 탐지 기술 개발',
    approach: '선형 LM Head 구조와 은닉 상태의 미세한 섭동을 활용하는 추론 단계의 워터마킹',
    outcome: '진행 중인 연구입니다. 공개 평가 수치와 산출물 링크는 아직 등록되지 않았습니다.',
  },
  2: {
    problem: 'IR 발표의 전달력과 논리 구조를 분석하고 피드백을 제공하는 평가 시스템 개발',
    approach: '피치덱 문서와 발표 영상의 텍스트·음성·표정·행동 데이터를 통합 분석',
    outcome: '진행 중인 연구입니다. 자동 리포트 제공을 목표로 하며, 공개 평가 수치는 아직 등록되지 않았습니다.',
  },
  3: {
    problem: '설비 전류 데이터로 이상을 탐지하고 잔여수명을 예측하는 AI 모델 개발',
    approach: '이상 탐지·RUL 예측·Health Index 산출 알고리즘과 통합 예측 운영 모델',
    outcome: '종료된 과제입니다. 공개 평가 수치와 산출물 링크는 등록되지 않았습니다.',
  },
};

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
