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
    title: '숙명여대 이수안 교수, AI 교육 혁신 선도',
    source: '전자신문',
    date: '2024-12-15',
    url: '#',
    excerpt: 'YouTube 채널을 통해 15만 구독자에게 데이터 과학과 인공지능 교육 콘텐츠를 제공하며 AI 교육의 대중화에 기여하고 있다.',
    thumbnail: '/assets/images/media/news1.jpg',
  },
  {
    id: '2',
    title: 'SuanLab, 빅데이터 분석 오픈소스 프로젝트 공개',
    source: 'AI Times',
    date: '2024-11-20',
    url: '#',
    excerpt: '숙명여자대학교 SuanLab에서 빅데이터 분석을 위한 새로운 오픈소스 도구를 공개하여 연구자들의 큰 관심을 받고 있다.',
    thumbnail: '/assets/images/media/news2.jpg',
  },
  {
    id: '3',
    title: '딥러닝 기반 자연어 처리 연구 성과 발표',
    source: '한국경제',
    date: '2024-10-08',
    url: '#',
    excerpt: 'SuanLab 연구팀이 한국어 자연어 처리 분야에서 새로운 벤치마크를 수립하는 연구 결과를 국제 학술지에 게재했다.',
    thumbnail: '/assets/images/media/news3.jpg',
  },
  {
    id: '4',
    title: '데이터 사이언스 입문자를 위한 무료 강의 시리즈',
    source: '조선일보',
    date: '2024-09-15',
    url: '#',
    excerpt: '이수안 교수가 데이터 사이언스 입문자를 위한 체계적인 무료 강의 시리즈를 YouTube에 공개하여 호평을 받고 있다.',
    thumbnail: '/assets/images/media/news4.jpg',
  },
];
