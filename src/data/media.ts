export interface MediaArticle {
  id: string;
  title: string;
  source: string;
  date: string;
  url: string;
  excerpt: string;
  thumbnail?: string;
}

export const mediaArticles: MediaArticle[] = [
  {
    id: '1',
    title: '세명대 데이터지능연구실, KCC 2025서 우수상 영예',
    source: '충북인뉴스',
    date: '2025-06-25',
    url: 'https://www.inews365.com/news/article.html?no=885169',
    excerpt: '세명대학교 컴퓨터학부 데이터지능연구실이 인공지능을 활용한 진로 상담 연구로 2025 한국컴퓨터종합학술대회(KCC 2025)에서 우수상을 수상했다.',
  },
  {
    id: '2',
    title: '제천제일고, 지역 문화와 AI 융합 스마트팜 프로그램 운영',
    source: '충북인뉴스',
    date: '2024-05-12',
    url: 'https://www.inews365.com/news/article.html?no=813896',
    excerpt: '제천제일고등학교가 의림지박물관 견학과 농업기술센터의 아열대 스마트 농장 견학으로 인공지능을 활용한 스마트 농법을 직접 체험하게 했다.',
  },
  {
    id: '3',
    title: '미래 고급인력 소멸…초거대AI 등 디지털 역량 강화해 생산성 극대화해야',
    source: '전자신문',
    date: '2023-09-19',
    url: 'https://www.etnews.com/20230913000134',
    excerpt: '초거대AI 시대가 도래하면서 전문가들은 생성형AI 등 디지털 활용 역량을 강화해 산업 전반의 생산성을 극대화해야 한다고 제안했다.',
  },
  {
    id: '4',
    title: '한국통계정보원, 통계정보발전포럼 개최',
    source: '세계일보',
    date: '2023-09-12',
    url: 'https://www.segye.com/newsView/20230912515233',
    excerpt: '한국통계정보원이 통계데이터 및 AI 중심의 통계정보플랫폼 발전 전략을 주제로 2023 통계정보발전포럼을 개최했다.',
  },
  {
    id: '5',
    title: '세종시 빅데이터 활용 행정서비스 대전환',
    source: '충북인뉴스',
    date: '2023-07-03',
    url: 'https://www.inews365.com/news/article.html?no=771210',
    excerpt: '세종시가 인공지능 챗GPT를 행정업무에 적용하기로 결정하고 공직자 역량강화 교육을 추진 중이다.',
  },
  {
    id: '6',
    title: '세명대 컴퓨터학부 학부생, SCI급 국제저널 논문 게재',
    source: '충북인뉴스',
    date: '2023-01-31',
    url: 'https://www.inews365.com/news/article.html?no=750046',
    excerpt: '세명대학교 컴퓨터학부 신재광 학생이 이수안 교수 지도 하에 산업용 기계의 소리를 통해 고장을 탐지하는 딥러닝 모델 논문을 SCI급 국제저널에 게재했다.',
  },
  {
    id: '7',
    title: '인하대 이우기 교수 연구팀, 장비 고장진단에 인공지능 적용 논문 발표',
    source: '컨슈머스토리',
    date: '2025-01-15',
    url: 'https://www.cstimes.com/news/articleView.html?idxno=629037',
    excerpt: '인하대학교 이우기 교수 연구팀이 빅데이터 분석과 인공지능을 활용해 제조업체의 설비고장 관리능력을 향상시키는 논문을 발표했다.',
  },
  {
    id: '8',
    title: '이뉴스투데이 AI 빅데이터 관련 보도',
    source: '이뉴스투데이',
    date: '2024-01-01',
    url: 'https://www.enewstoday.co.kr/news/articleView.html?idxno=2356015',
    excerpt: 'AI와 빅데이터 기술의 최신 동향과 연구 성과에 대한 보도.',
  },
  {
    id: '9',
    title: '베리타스알파 교육 AI 관련 보도',
    source: '베리타스알파',
    date: '2024-01-01',
    url: 'https://www.veritas-a.com/news/articleView.html?idxno=571866',
    excerpt: '교육 분야에서의 AI 활용과 혁신에 대한 보도.',
  },
  {
    id: '10',
    title: 'TNT뉴스 데이터 사이언스 관련 보도',
    source: 'TNT뉴스',
    date: '2024-01-01',
    url: 'https://www.newstnt.com/news/articleView.html?idxno=464916',
    excerpt: '데이터 사이언스와 머신러닝 기술 발전에 대한 보도.',
  },
  {
    id: '11',
    title: '농업인신문 스마트팜 AI 관련 보도',
    source: '농업인신문',
    date: '2024-01-01',
    url: 'https://www.agrinet.co.kr/news/articleView.html?idxno=330393',
    excerpt: '스마트팜과 농업 분야 AI 기술 적용에 대한 보도.',
  },
];
