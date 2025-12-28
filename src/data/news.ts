export interface NewsItem {
  date: string;
  title: string;
  content: string;
  link?: string;
  linkText?: string;
}

export const newsItems: NewsItem[] = [
  {
    date: '2022-12-02',
    title: 'KJDB2022 Workshop',
    content: `한국과 일본의 데이터베이스 연구자 분들이 정례적으로 진행하는
Korea-Japan Database (KJDB) 워크숍이 12월 2일(금)~3일(토)에 개최됩니다.
코로나-19로 인하여 작년과 마찬가지로 온라인으로 진행됩니다.
많은 데이터베이스 연구자 분들의 관심과 참여를 부탁 드립니다.`,
    link: 'https://sites.google.com/view/kjdb2022',
    linkText: 'KJDB2022: https://sites.google.com/view/kjdb2022',
  },
  {
    date: '2020-01-20',
    title: '"바이러스 연구부터 뷰티·배달 AI 결합한 비즈니스..."',
    content: '"[테크 인사이드] 바이러스 연구부터 뷰티·배달 AI 결합한 비즈니스 모델 주목" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?s=401&aid=221011900000',
    linkText: '칼럼 보기',
  },
  {
    date: '2020-10-07',
    title: '"이력서 작성·레시피 제공 다양하게 활용되는 GPT3"',
    content: '"[테크 인사이드] 이력서 작성·레시피 제공 다양하게 활용되는 GPT3" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?s=401&aid=220100600056',
    linkText: '칼럼 보기',
  },
  {
    date: '2020-05-20',
    title: '"인공지능의 보안 위협"',
    content: '"[테크 인사이드] 인공지능의 보안 위협" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?aid=220051900034',
    linkText: '칼럼 보기',
  },
  {
    date: '2020-03-04',
    title: '"데이터 경제 시대"',
    content: '"[테크 인사이드] 데이터 경제 시대" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?aid=220030300001',
    linkText: '칼럼 보기',
  },
  {
    date: '2019-12-25',
    title: '"마이데이터 시대의 도래 데이터 주권과 새로운 가치"',
    content: '"[테크 인사이드] 마이데이터 시대의 도래 데이터 주권과 새로운 가치" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?aid=219122400022',
    linkText: '칼럼 보기',
  },
  {
    date: '2019-09-04',
    title: '"농업으로 들어간 인공지능"',
    content: '"[테크 인사이드] 농업으로 들어간 인공지능" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?aid=219090300125',
    linkText: '칼럼 보기',
  },
  {
    date: '2019-08-07',
    title: '"AI시대 지배할 것인가 지배당하며 살 것인가"',
    content: '"[테크 인사이드] AI시대 지배할 것인가 지배당하며 살 것인가" 강원일보 칼럼 기고',
    link: 'http://www.kwnews.co.kr/nview.asp?aid=219080600011',
    linkText: '칼럼 보기',
  },
];

export const photoImages = [
  '/assets/images/home/bigdata.jpg',
  '/assets/images/home/cloud.jpg',
  '/assets/images/home/dl.jpg',
  '/assets/images/home/market.jpg',
  '/assets/images/home/sn.jpg',
  '/assets/images/home/snvis.jpg',
  '/assets/images/home/network.jpg',
];
