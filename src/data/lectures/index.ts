export interface LectureSemester {
  semester: string;
  courses: string[];
  institution: string;
}

export interface Lecture {
  slug: string;
  titleKo: string;
  titleEn: string;
  image: string;
  description: string;
  descriptionKo: string;
  icon: string;
  topics: string[];
  relatedYoutube?: string[];
  history: LectureSemester[];
}

export const lectures: Lecture[] = [
  {
    slug: 'ai',
    titleKo: '인공지능',
    titleEn: 'Artificial Intelligence',
    image: '/assets/images/lecture/ai.jpg',
    description: 'Comprehensive course covering fundamental concepts of artificial intelligence, including search algorithms, knowledge representation, reasoning, and introduction to machine learning.',
    descriptionKo: '4차 산업혁명의 핵심 기술인 인공지능의 기본 개념과 응용을 다루는 과목입니다. 탐색 알고리즘, 지식 표현, 추론, 머신러닝 기초를 학습합니다.',
    icon: 'fa fa-brain',
    topics: [
      '인공지능 개요 및 역사',
      '탐색 알고리즘 (BFS, DFS, A*)',
      '지식 표현과 추론',
      '게임 이론과 의사결정',
      '전문가 시스템',
      '머신러닝 기초',
      '신경망 기초',
    ],
    relatedYoutube: ['/youtube/ml', '/youtube/dl'],
    history: [
      { semester: '2025년 2학기', courses: ['(외국인) 인공지능'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2024년 2학기', courses: ['인공지능 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2023년 2학기', courses: ['인공지능 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2022년 2학기', courses: ['인공지능 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2021년 1학기', courses: ['인공지능 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2017년 2학기', courses: ['정보처리특강(1) - 인공지능'], institution: '강원대학교 정보과학대학원' },
    ],
  },
  {
    slug: 'dl',
    titleKo: '딥러닝',
    titleEn: 'Deep Learning',
    image: '/assets/images/lecture/dl.jpg',
    description: 'Advanced course on deep learning covering neural network architectures, CNN, RNN, Transformer, and practical implementations using PyTorch and TensorFlow.',
    descriptionKo: '딥러닝의 핵심 이론과 실습을 다루는 과목입니다. CNN, RNN, Transformer 등 다양한 신경망 구조와 PyTorch, TensorFlow를 활용한 모델 구현을 학습합니다.',
    icon: 'fa fa-layer-group',
    topics: [
      '딥러닝 개요',
      '다층 퍼셉트론 (MLP)',
      '합성곱 신경망 (CNN)',
      '순환 신경망 (RNN, LSTM, GRU)',
      '어텐션 메커니즘',
      'Transformer 아키텍처',
      '생성 모델 (GAN, VAE, Diffusion)',
      'PyTorch / TensorFlow 실습',
    ],
    relatedYoutube: ['/youtube/dl', '/youtube/dlf'],
    history: [
      { semester: '2025년 1학기', courses: ['딥러닝'], institution: '세명대학교 컴퓨터학부' },
    ],
  },
  {
    slug: 'ml',
    titleKo: '머신러닝',
    titleEn: 'Machine Learning',
    image: '/assets/images/lecture/ml.jpg',
    description: 'Core machine learning course covering supervised/unsupervised learning, classification, regression, clustering, and model evaluation techniques.',
    descriptionKo: '머신러닝의 핵심 알고리즘과 이론을 다루는 과목입니다. 지도학습, 비지도학습, 분류, 회귀, 클러스터링, 모델 평가 기법을 학습합니다.',
    icon: 'fa fa-cogs',
    topics: [
      '머신러닝 개요',
      '지도학습 vs 비지도학습',
      '선형/로지스틱 회귀',
      '결정 트리와 랜덤 포레스트',
      'SVM (Support Vector Machine)',
      'K-평균 클러스터링',
      '차원 축소 (PCA)',
      '앙상블 학습',
      '모델 평가 및 하이퍼파라미터 튜닝',
    ],
    relatedYoutube: ['/youtube/ml'],
    history: [
      { semester: '2025년 2학기', courses: ['머신러닝', '(외국인) 머신러닝', '기계학습 (연계전공)'], institution: '세명대학교 컴퓨터학부' },
    ],
  },
  {
    slug: 'nlp',
    titleKo: '자연어처리',
    titleEn: 'Natural Language Processing',
    image: '/assets/images/lecture/nlp.jpg',
    description: 'Natural language processing course covering text preprocessing, word embeddings, language models, and modern NLP applications using deep learning.',
    descriptionKo: '자연어처리의 이론과 실습을 다루는 과목입니다. 텍스트 전처리, 워드 임베딩, 언어 모델, 딥러닝 기반 NLP 응용을 학습합니다.',
    icon: 'fa fa-comments',
    topics: [
      '자연어처리 개요',
      '텍스트 전처리 (토큰화, 정규화)',
      '워드 임베딩 (Word2Vec, GloVe)',
      '순환 신경망 기반 NLP',
      '어텐션과 Transformer',
      'BERT, GPT 등 사전학습 모델',
      '텍스트 분류 및 감성 분석',
      '질의응답 시스템',
      '기계 번역',
    ],
    relatedYoutube: ['/youtube/nlp'],
    history: [
      { semester: '2025년 2학기', courses: ['자연어처리', '(외국인) 자연어처리', '고급자연어처리 (대학원)'], institution: '세명대학교' },
      { semester: '2024년 1학기', courses: ['자연어음성처리 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2023년 1학기', courses: ['자연어음성처리 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2022년 1학기', courses: ['자연어음성처리 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
    ],
  },
  {
    slug: 'cv',
    titleKo: '컴퓨터비전',
    titleEn: 'Computer Vision',
    image: '/assets/images/lecture/cv.jpg',
    description: 'Computer vision course covering image processing, feature extraction, object detection, image segmentation, and deep learning for visual recognition.',
    descriptionKo: '컴퓨터비전의 이론과 실습을 다루는 과목입니다. 이미지 처리, 특징 추출, 객체 탐지, 이미지 분할, 딥러닝 기반 시각 인식을 학습합니다.',
    icon: 'fa fa-eye',
    topics: [
      '컴퓨터비전 개요',
      '이미지 처리 기초',
      '에지 검출 및 필터링',
      '특징 추출 (SIFT, HOG)',
      '객체 탐지 (YOLO, Faster R-CNN)',
      '이미지 분류 (CNN)',
      '시맨틱 세그멘테이션',
      '얼굴 인식',
      '비디오 분석',
    ],
    relatedYoutube: ['/youtube/cv'],
    history: [
      { semester: '2025년 2학기', courses: ['컴퓨터비전', '고급컴퓨터비전 (대학원)'], institution: '세명대학교' },
      { semester: '2021년 2학기', courses: ['이미지프로세싱 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
    ],
  },
  {
    slug: 'asp',
    titleKo: '오디오신호처리',
    titleEn: 'Audio Signal Processing',
    image: '/assets/images/lecture/audio.jpg',
    description: 'Audio signal processing course covering digital audio fundamentals, spectral analysis, speech recognition, and audio deep learning applications.',
    descriptionKo: '오디오 신호처리의 이론과 실습을 다루는 과목입니다. 디지털 오디오 기초, 스펙트럼 분석, 음성 인식, 오디오 딥러닝 응용을 학습합니다.',
    icon: 'fa fa-music',
    topics: [
      '오디오 신호처리 개요',
      '디지털 오디오 기초',
      '푸리에 변환 (FFT)',
      'STFT와 스펙트로그램',
      'MFCC 특징 추출',
      '음성 인식 기초',
      '음성 합성 (TTS)',
      '오디오 분류',
      '음악 정보 검색',
    ],
    relatedYoutube: ['/youtube/asp'],
    history: [
      { semester: '2025년 1학기', courses: ['오디오신호처리'], institution: '세명대학교 컴퓨터학부' },
    ],
  },
  {
    slug: 'bd',
    titleKo: '빅데이터분석',
    titleEn: 'Big Data Analysis',
    image: '/assets/images/lecture/bd.jpg',
    description: 'Big data analysis course covering data collection, preprocessing, visualization, statistical analysis, and machine learning for large-scale data.',
    descriptionKo: '빅데이터 분석의 이론과 실습을 다루는 과목입니다. 데이터 수집, 전처리, 시각화, 통계 분석, 대규모 데이터 머신러닝을 학습합니다.',
    icon: 'fa fa-chart-bar',
    topics: [
      '빅데이터 개요',
      '데이터 수집 및 크롤링',
      '데이터 전처리 (Pandas)',
      '탐색적 데이터 분석 (EDA)',
      '데이터 시각화 (Matplotlib, Seaborn)',
      '통계 분석',
      '분산 처리 (Spark)',
      '빅데이터 머신러닝',
    ],
    relatedYoutube: ['/youtube/ds', '/youtube/da', '/youtube/dv', '/youtube/bd'],
    history: [
      { semester: '2025년 1학기', courses: ['빅데이터분석', '빅데이터프로그래밍 (연계전공)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2024년 1학기', courses: ['빅데이터분석 (2개 분반)', '빅데이터분석 (바이오헬스혁신센터)'], institution: '세명대학교' },
      { semester: '2023년 1학기', courses: ['빅데이터분석 (2개 분반)', '빅데이터분석 (바이오헬스혁신센터)'], institution: '세명대학교' },
      { semester: '2022년 1학기', courses: ['빅데이터분석 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2017년 2학기', courses: ['정보처리특강(2) - 빅데이터'], institution: '강원대학교 정보과학대학원' },
    ],
  },
  {
    slug: 'db',
    titleKo: '데이터베이스',
    titleEn: 'Database',
    image: '/assets/images/lecture/db.jpg',
    description: 'Database course covering relational database design, SQL, normalization, transaction management, and NoSQL databases.',
    descriptionKo: '데이터베이스의 이론과 실습을 다루는 과목입니다. 관계형 데이터베이스 설계, SQL, 정규화, 트랜잭션 관리, NoSQL 데이터베이스를 학습합니다.',
    icon: 'fa fa-database',
    topics: [
      '데이터베이스 개요',
      'ER 모델링',
      '관계형 데이터베이스',
      'SQL (DDL, DML, DCL)',
      '정규화',
      '인덱싱',
      '트랜잭션과 동시성 제어',
      'NoSQL 데이터베이스',
    ],
    relatedYoutube: ['/youtube/db'],
    history: [
      { semester: '2025년 2학기', courses: ['데이터베이스'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2024년 2학기', courses: ['데이터베이스 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2023년 2학기', courses: ['데이터베이스 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2023년 1학기', courses: ['데이터베이스특론 (대학원)'], institution: '세명대학교 대학원' },
      { semester: '2022년 2학기', courses: ['데이터베이스 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2021년 2학기', courses: ['데이터베이스 (2개 분반)'], institution: '세명대학교 컴퓨터학부' },
      { semester: '2016년 2학기', courses: ['데이터베이스 (대학원)'], institution: '강원대학교 정보과학대학원' },
      { semester: '2016년 1학기', courses: ['데이터베이스 (원어강의)'], institution: '인하대학교 산업공학과' },
    ],
  },
  {
    slug: 'web',
    titleKo: '웹프로그래밍',
    titleEn: 'Web Programming',
    image: '/assets/images/lecture/webprogramming.jpg',
    description: 'Web programming course covering HTML, CSS, JavaScript fundamentals, and practical web application development.',
    descriptionKo: '웹 프로그래밍의 기초와 실습을 다루는 과목입니다. HTML, CSS, JavaScript 기초와 웹 애플리케이션 개발을 학습합니다.',
    icon: 'fa fa-globe',
    topics: [
      '웹과 인터넷 개요',
      'HTML5 기초 및 구조',
      'CSS3 스타일링',
      'JavaScript 기초',
      'DOM 조작',
      '반응형 웹 디자인',
      '웹 표준과 접근성',
    ],
    relatedYoutube: ['/youtube/web'],
    history: [
      { semester: '2018년 1학기', courses: ['웹과인터넷활용및실습'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2017년 2학기', courses: ['웹과인터넷활용및실습'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2017년 1학기', courses: ['웹과인터넷활용및실습'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2016년 2학기', courses: ['웹과인터넷활용및실습'], institution: '강원대학교 IT대학 컴퓨터학부' },
      { semester: '2016년 1학기', courses: ['웹과인터넷활용및실습'], institution: '강원대학교 IT특성화학부 컴퓨터과학전공' },
      { semester: '2012년 2학기', courses: ['웹과인터넷활용및실습 (2개 분반)'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
    ],
  },
  {
    slug: 'cpp',
    titleKo: '컴퓨터프로그래밍',
    titleEn: 'Computer Programming',
    image: '/assets/images/lecture/programming.jpg',
    description: 'Introductory computer programming course covering fundamental programming concepts, algorithms, and problem-solving using C/C++ or Python.',
    descriptionKo: '컴퓨터 프로그래밍의 기초를 다루는 과목입니다. 프로그래밍 기본 개념, 알고리즘, 문제 해결 능력을 학습합니다.',
    icon: 'fa fa-code',
    topics: [
      '프로그래밍 개요',
      '변수와 자료형',
      '연산자와 표현식',
      '제어문 (조건문, 반복문)',
      '함수',
      '배열과 포인터',
      '구조체',
      '파일 입출력',
    ],
    relatedYoutube: ['/youtube/pp'],
    history: [
      { semester: '2018년 2학기', courses: ['컴퓨터프로그래밍기초'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2017년 겨울', courses: ['컴퓨터프로그래밍'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2011년 2학기', courses: ['컴퓨터프로그래밍기초'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2011년 1학기', courses: ['컴퓨터프로그래밍기초'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
      { semester: '2010년 2학기', courses: ['컴퓨터프로그래밍응용'], institution: '강원대학교 IT대학 컴퓨터과학전공' },
    ],
  },
  {
    slug: 'cloud',
    titleKo: '클라우드컴퓨팅',
    titleEn: 'Cloud Computing',
    image: '/assets/images/lecture/cloud.jpg',
    description: 'Cloud computing course covering virtualization, cloud service models (IaaS, PaaS, SaaS), and practical cloud platform usage.',
    descriptionKo: '클라우드 컴퓨팅의 이론과 실습을 다루는 과목입니다. 가상화 기술, 클라우드 서비스 모델, 클라우드 플랫폼 활용을 학습합니다.',
    icon: 'fa fa-cloud',
    topics: [
      '클라우드 컴퓨팅 개요',
      '가상화 기술',
      '클라우드 서비스 모델 (IaaS, PaaS, SaaS)',
      '클라우드 배포 모델',
      'AWS / GCP / Azure 기초',
      '컨테이너와 Docker',
      '분산 시스템',
      '클라우드 보안',
    ],
    relatedYoutube: ['/youtube/bd'],
    history: [
      { semester: '2017년 2학기', courses: ['클라우드컴퓨팅 (2개 분반)'], institution: '한국공학대학교 (전 한국산업기술대학교) 컴퓨터공학과' },
      { semester: '2016년 2학기', courses: ['클라우드컴퓨팅 (2개 분반)'], institution: '한국공학대학교 (전 한국산업기술대학교) 컴퓨터공학과' },
    ],
  },
];

// 대학원 과목 목록
export const graduateCourses = [
  { name: '인공지능특론', semesters: ['2024년 2학기', '2022년 1학기'] },
  { name: '고급자연어처리', semesters: ['2025년 2학기'] },
  { name: '고급컴퓨터비전', semesters: ['2025년 2학기'] },
  { name: '강화학습', semesters: ['2024년 1학기'] },
  { name: '병렬처리특론', semesters: ['2023년 2학기'] },
  { name: '데이터베이스특론', semesters: ['2023년 1학기'] },
  { name: '고급데이터마이닝', semesters: ['2022년 2학기'] },
  { name: '고급가상현실', semesters: ['2021년 2학기'] },
];

// 강의 이력 요약
export const teachingHistory = {
  semyungUniv: {
    name: '세명대학교',
    period: '2021.03 - 현재',
    role: '조교수',
    department: 'IT엔지니어링대학 컴퓨터학부',
  },
  kangwonUniv: {
    name: '강원대학교',
    period: '2010 - 2019',
    role: '연구교수/객원교수/시간강사',
    department: 'IT대학, SW중심대학, 정보과학대학원',
  },
  koreatech: {
    name: '한국공학대학교',
    period: '2016 - 2017',
    role: '시간강사',
    department: '컴퓨터공학과',
  },
  inhaUniv: {
    name: '인하대학교',
    period: '2016 - 2017',
    role: '시간강사',
    department: '산업공학과',
  },
};

export const getLectureBySlug = (slug: string): Lecture | undefined => {
  return lectures.find((l) => l.slug === slug);
};
