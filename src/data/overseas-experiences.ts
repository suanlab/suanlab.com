export interface OverseasExperience {
  id: number;
  purpose: string;
  countries: string[];
  cities: string;
  period: string;
  year: number;
}

export const overseasExperiences: OverseasExperience[] = [
  { id: 33, purpose: '캐나다 횡단 여행 (예정)', countries: ['캐나다'], cities: '캐나다 횡단', period: '2026.07 (예정)', year: 2026 },
  { id: 34, purpose: 'KCC 2026', countries: ['한국'], cities: '제주도', period: '2026.06', year: 2026 },
  { id: 35, purpose: 'CVPR 2026', countries: ['미국'], cities: '덴버', period: '2026.06', year: 2026 },
  { id: 36, purpose: 'DASFAA 2026', countries: ['한국'], cities: '제주도', period: '2026.04', year: 2026 },
  { id: 37, purpose: 'AAAI 2026', countries: ['싱가포르'], cities: '싱가포르', period: '2026.01', year: 2026 },
  { id: 1, purpose: 'KJDB 2025', countries: ['일본'], cities: '나고야, 이세', period: '2025.12.12 ~ 2025.12.15', year: 2025 },
  { id: 2, purpose: 'ACL 2025', countries: ['오스트리아'], cities: '비엔나', period: '2025.07.23 ~ 2025.08.07', year: 2025 },
  { id: 3, purpose: '성지순례', countries: ['이집트'], cities: '이집트', period: '2025.02.02 ~ 2025.02.11', year: 2025 },
  { id: 4, purpose: 'NeurIPS 2024', countries: ['캐나다'], cities: '벤쿠버, 빅토리아', period: '2024.12.07 ~ 2024.12.17', year: 2024 },
  { id: 6, purpose: 'Trip', countries: ['베트남'], cities: '다낭', period: '2024.08.02 ~ 2024.08.05', year: 2024 },
  { id: 7, purpose: 'DATA 2024', countries: ['프랑스', '독일', '스위스'], cities: '디종, 스트라스부르, 프랑크푸르트, 베른 등', period: '2024.06.30 ~ 2024.07.14', year: 2024 },
  { id: 8, purpose: 'BigComp 2024', countries: ['태국'], cities: '방콕', period: '2024.02.17 ~ 2024.02.22', year: 2024 },
  { id: 9, purpose: 'Trip', countries: ['인도네시아'], cities: '발리', period: '2024.01.21 ~ 2024.01.27', year: 2024 },
  { id: 10, purpose: 'NeurIPS 2023', countries: ['미국'], cities: '뉴올리언스', period: '2023.12.10 ~ 2023.12.17', year: 2023 },
  { id: 11, purpose: 'KJDB 2023', countries: ['일본'], cities: '후쿠오카, 야마구치', period: '2023.10.27 ~ 2023.10.29', year: 2023 },
  { id: 12, purpose: 'Trip', countries: ['독일', '프랑스', '스위스', '이탈리아', '오스트리아'], cities: '프랑크푸르트, 루체른, 밀라노, 베네치아 등', period: '2023.01.26 ~ 2023.02.10', year: 2023 },
  { id: 13, purpose: 'CCTA 2022', countries: ['이탈리아', '오스트리아', '슬로베니아', '크로아티아'], cities: '밀라노, 베네치아, 플리트비체 등', period: '2022.08.14 ~ 2022.08.27', year: 2022 },
  { id: 14, purpose: 'Trip', countries: ['독일', '스위스', '프랑스', '오스트리아', '체코'], cities: '프랑크푸르트, 뮌헨, 취리히, 프라하 등', period: '2020.01.17 ~ 2020.01.28', year: 2020 },
  { id: 15, purpose: '성지순례', countries: ['튀르키예', '그리스', '이탈리아'], cities: '이스탄불, 아테네, 로마, 바티칸 등', period: '2019.01.14 ~ 2019.01.25', year: 2019 },
  { id: 16, purpose: '의료선교', countries: ['인도네시아'], cities: '자카르타, 람뿡', period: '2018.09.23 ~ 2018.09.29', year: 2018 },
  { id: 17, purpose: 'DEXA 2018', countries: ['독일'], cities: '프랑크푸르트, 베를린, 뮌헨 등', period: '2018.08.22 ~ 2018.09.07', year: 2018 },
  { id: 18, purpose: 'Trip', countries: ['미국'], cities: '사이판', period: '2018.07.23 ~ 2018.07.27', year: 2018 },
  { id: 19, purpose: '성지순례', countries: ['이스라엘'], cities: '예루살렘, 텔아비브, 갈릴리, 베들레헴', period: '2018.01.23 ~ 2018.01.31', year: 2018 },
  { id: 20, purpose: 'Lab. Workshop', countries: ['미국'], cities: '시애틀, 옐로우스톤, 샌프란시스코, LA 등', period: '2016.06.19 ~ 2016.07.07', year: 2016 },
  { id: 21, purpose: 'Praise Trip', countries: ['일본'], cities: '오키나와', period: '2016.02.15 ~ 2016.02.18', year: 2016 },
  { id: 22, purpose: 'Vision Trip', countries: ['중국'], cities: '대련, 단동, 심양, 장춘, 하얼빈', period: '2015.01.12 ~ 2015.01.19', year: 2015 },
  { id: 23, purpose: 'Vision Trip', countries: ['인도네시아'], cities: '자카르타, 아체', period: '2014.07.07 ~ 2014.07.16', year: 2014 },
  { id: 24, purpose: 'Vision Trip', countries: ['중국'], cities: '상하이', period: '2013.12.30 ~ 2014.01.05', year: 2013 },
  { id: 25, purpose: 'DaWaK 2012', countries: ['오스트리아', '체코', '스위스', '이탈리아'], cities: '비엔나, 프라하, 취리히, 로마 등', period: '2012.08.17 ~ 2012.09.08', year: 2012 },
  { id: 26, purpose: 'Vision Trip', countries: ['중국'], cities: '베이징', period: '2012.02.20 ~ 2012.02.25', year: 2012 },
  { id: 27, purpose: 'CSN 2011', countries: ['호주'], cities: '시드니', period: '2011.12.09 ~ 2011.12.20', year: 2011 },
  { id: 28, purpose: 'Vision Trip', countries: ['인도네시아'], cities: '자카르타, 또라자', period: '2011.07.11 ~ 2011.07.19', year: 2011 },
  { id: 29, purpose: 'Vision Trip', countries: ['중국'], cities: '베이징', period: '2011.02.14 ~ 2011.02.18', year: 2011 },
  { id: 30, purpose: 'Vision Trip', countries: ['인도네시아', '싱가포르', '말레이시아'], cities: '자카르타, 칼리만탄, 싱가포르, 멜라카', period: '2010.07.05 ~ 2010.07.16', year: 2010 },
  { id: 31, purpose: 'ER 2008', countries: ['스페인', '프랑스'], cities: '마드리드, 바르셀로나, 파리', period: '2008.10.16 ~ 2008.10.31', year: 2008 },
  { id: 32, purpose: 'WORLDCOMP 2008', countries: ['미국'], cities: '라스베이거스', period: '2008.07.14 ~ 2008.07.20', year: 2008 },
];

// 방문한 국가 목록과 국기 이모지
// iso: ISO 3166-1 numeric (world-atlas topojson id). lat/lng: 도시국가 등 지도에 없는 경우 수동 핀 좌표.
export const visitedCountries = [
  { name: '미국', code: 'US', iso: '840', flag: '🇺🇸', continent: 'America' },
  { name: '캐나다', code: 'CA', iso: '124', flag: '🇨🇦', continent: 'America' },
  { name: '일본', code: 'JP', iso: '392', flag: '🇯🇵', continent: 'Asia' },
  { name: '중국', code: 'CN', iso: '156', flag: '🇨🇳', continent: 'Asia' },
  { name: '태국', code: 'TH', iso: '764', flag: '🇹🇭', continent: 'Asia' },
  { name: '베트남', code: 'VN', iso: '704', flag: '🇻🇳', continent: 'Asia' },
  { name: '인도네시아', code: 'ID', iso: '360', flag: '🇮🇩', continent: 'Asia' },
  { name: '싱가포르', code: 'SG', iso: '702', flag: '🇸🇬', continent: 'Asia', lat: 1.3521, lng: 103.8198 },
  { name: '말레이시아', code: 'MY', iso: '458', flag: '🇲🇾', continent: 'Asia' },
  { name: '이스라엘', code: 'IL', iso: '376', flag: '🇮🇱', continent: 'Asia' },
  { name: '튀르키예', code: 'TR', iso: '792', flag: '🇹🇷', continent: 'Asia' },
  { name: '호주', code: 'AU', iso: '036', flag: '🇦🇺', continent: 'Oceania' },
  { name: '독일', code: 'DE', iso: '276', flag: '🇩🇪', continent: 'Europe' },
  { name: '프랑스', code: 'FR', iso: '250', flag: '🇫🇷', continent: 'Europe' },
  { name: '스위스', code: 'CH', iso: '756', flag: '🇨🇭', continent: 'Europe' },
  { name: '이탈리아', code: 'IT', iso: '380', flag: '🇮🇹', continent: 'Europe' },
  { name: '오스트리아', code: 'AT', iso: '040', flag: '🇦🇹', continent: 'Europe' },
  { name: '스페인', code: 'ES', iso: '724', flag: '🇪🇸', continent: 'Europe' },
  { name: '체코', code: 'CZ', iso: '203', flag: '🇨🇿', continent: 'Europe' },
  { name: '그리스', code: 'GR', iso: '300', flag: '🇬🇷', continent: 'Europe' },
  { name: '슬로베니아', code: 'SI', iso: '705', flag: '🇸🇮', continent: 'Europe' },
  { name: '크로아티아', code: 'HR', iso: '191', flag: '🇭🇷', continent: 'Europe' },
  { name: '이집트', code: 'EG', iso: '818', flag: '🇪🇬', continent: 'Africa' },
];

// 대륙별 색상
export const continentColors: Record<string, string> = {
  'Asia': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  'Europe': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  'America': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  'Oceania': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
  'Africa': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
};

// 통계
export const overseasStats = {
  totalCountries: 23,
  totalTrips: 36,
  totalYears: 18, // 2008 ~ 2026
};
