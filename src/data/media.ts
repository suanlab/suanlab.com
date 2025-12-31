export interface MediaArticle {
  id: string;
  title: string;
  source: string;
  date: string;
  url: string;
  excerpt: string;
  thumbnail?: string;
}

// Sorted by date (newest first)
export const mediaArticles: MediaArticle[] = [
  {
    id: '1',
    title: "세명대-지오비전-강원대 공동연구팀 세계 최고 권위 'AAAI' 논문 채택",
    source: '베리타스알파',
    date: '2025-11-17',
    url: 'https://www.veritas-a.com/news/articleView.html?idxno=583277',
    excerpt: '이수안 교수(세명대 컴퓨터학부 학부장)가 2저자로 참여한 투명 신경망 모델 GATSM 논문이 세계 최고 권위의 AI 학술대회 AAAI 2026에 채택되었다.',
  },
  {
    id: '2',
    title: '세명대 데이터지능연구실, KCC 2025서 우수상 영예',
    source: '충북인뉴스',
    date: '2025-09-11',
    url: 'https://www.inews365.com/news/article.html?no=885169',
    excerpt: '이수안 교수가 학부생 김병학과 공동 연구한 "대규모 언어 모델을 활용한 진로 성숙도 검사 응답 자동 분석 및 피드백 생성" 논문으로 KCC 2025에서 우수상을 수상했다.',
  },
  {
    id: '3',
    title: '세명대 컴퓨터학부 데이터지능연구실, 2025 한국컴퓨터종합학술대회 대상 수상',
    source: '세명대학교',
    date: '2025-09-05',
    url: 'https://www.semyung.ac.kr/prog/vwBoard/bbs08/kor/sub08_03/view.do?board_num=187362',
    excerpt: '이수안 교수가 지도한 데이터지능연구실이 K-GovExam 데이터셋 개발로 KCC 2025에서 대상을 수상했다. 학부생 김재성, 길상현이 sLLM 및 벡터 학습 연구에 참여했다.',
  },
  {
    id: '4',
    title: '인하대 이우기 교수 연구팀, 장비 고장진단에 인공지능 적용 논문 발표',
    source: '컨슈머스토리',
    date: '2025-01-15',
    url: 'https://www.cstimes.com/news/articleView.html?idxno=629037',
    excerpt: '이수안 교수가 인하대 음성AI연구소 객원연구원 및 세명대 겸임교수로서 인하대 이우기 교수팀과 함께 장비 고장진단 AI 연구에 참여했다.',
  },
  {
    id: '5',
    title: '제천제일고, 지역 문화와 AI 융합 스마트팜 프로그램 운영',
    source: '충북인뉴스',
    date: '2024-05-12',
    url: 'https://www.inews365.com/news/article.html?no=813896',
    excerpt: '이수안 교수가 제천제일고등학교 H-스마트팜 교육 프로그램에서 "인공지능과 스마트팜" 주제로 특강을 진행했다.',
  },
  {
    id: '6',
    title: '미래 고급인력 소멸…초거대AI 등 디지털 역량 강화해 생산성 극대화해야',
    source: '전자신문',
    date: '2023-09-19',
    url: 'https://www.etnews.com/20230913000134',
    excerpt: '이수안 교수가 생성형AI의 데이터 분석 민주화에 대해 "비전문가도 AI 대화형 모델을 활용해 더 직관적인 UI로 쉽게 데이터를 분석할 수 있을 것"이라고 전망했다.',
  },
  {
    id: '7',
    title: '한국통계정보원, 통계정보발전포럼 개최',
    source: '세계일보',
    date: '2023-09-12',
    url: 'https://www.segye.com/newsView/20230912515233',
    excerpt: '이수안 교수가 2023 통계정보발전포럼 2세션에서 "인공지능을 통한 통계데이터 분석 혁신" 주제로 발표하며 AI 기반 데이터 분석의 대중화를 전망했다.',
  },
  {
    id: '8',
    title: '세종시 빅데이터 활용 행정서비스 대전환',
    source: '충북인뉴스',
    date: '2023-07-03',
    url: 'https://www.inews365.com/news/article.html?no=771210',
    excerpt: '이수안 교수가 세종시 공무원 약 150명을 대상으로 "공공데이터 이해와 챗GPT 활용 방안" 주제로 디지털 역량 강화 교육을 진행했다.',
  },
  {
    id: '9',
    title: '세명대 컴퓨터학부 학부생, SCI급 국제저널 논문 게재',
    source: '충북인뉴스',
    date: '2023-01-31',
    url: 'https://www.inews365.com/news/article.html?no=750046',
    excerpt: '이수안 교수가 지도한 신재광 학생이 산업용 기계 소리를 통한 고장 탐지 딥러닝 모델 연구로 SCI급 국제저널 Electronics에 교신저자로 논문을 게재했다.',
  },
];
