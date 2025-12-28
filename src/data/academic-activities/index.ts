export interface AcademicActivity {
  id: number;
  period: string;
  organization: string;
  role: string;
  category: 'conference' | 'workshop' | 'forum' | 'journal' | 'membership';
}

export interface AdvisoryActivity {
  id: number;
  period: string;
  topic: string;
  organization: string;
  role: string;
}

// 국내외 학회, 워크샵, 포럼 활동
export const academicActivities: AcademicActivity[] = [
  // 2026
  { id: 1, period: '2026.08.09-13', organization: 'KDD 2026', role: 'Web Chair', category: 'conference' },
  { id: 2, period: '2026.04.27-30', organization: 'DASFAA 2026', role: 'Web Chair', category: 'conference' },
  { id: 3, period: '2026.02.02-05', organization: 'IEEE BigComp 2026', role: 'Demo Chair', category: 'conference' },
  // 2025
  { id: 4, period: '2025.12.16', organization: '데이터소사이어티 워크샵 2025', role: '조직위원장', category: 'workshop' },
  { id: 5, period: '2025.12.12-14', organization: 'KJDB 2025: Korea-Japan Database Workshop 2025', role: 'Session Chair', category: 'workshop' },
  { id: 6, period: '2025.06.29-07.02', organization: '데이터지능워크샵 2025', role: '조직위원', category: 'workshop' },
  { id: 7, period: '2025.02.09-12', organization: 'IEEE BigComp 2025', role: 'Poster Session Chair', category: 'conference' },
  // 2024
  { id: 8, period: '2024.12.18-20', organization: '2024 한국소프트웨어종합학술대회 (KSC 2024)', role: '평가위원', category: 'conference' },
  { id: 9, period: '2024.12.10-15', organization: 'NeurIPS 2024', role: 'Member', category: 'conference' },
  { id: 10, period: '2024.11.21-22', organization: '2024 인공지능학회 추계학술대회', role: 'Member', category: 'conference' },
  { id: 11, period: '2024.11.15-17', organization: 'KJDB 2024: Korea-Japan Database Workshop 2024', role: 'Web Chair', category: 'workshop' },
  { id: 12, period: '2024.11.01-02', organization: '제26회 Korean Database Conference (KDBC) 2024', role: '공동조직위원장', category: 'conference' },
  { id: 13, period: '2024.09.27-28', organization: '제42회 미래 정보 기술·산업 전망 (iTIP 2024)', role: 'Member', category: 'forum' },
  { id: 14, period: '2024.06', organization: 'Korea Summer Workshop on Causal Inference 2024', role: 'Member', category: 'workshop' },
  { id: 15, period: '2024.02.18-21', organization: 'IEEE BigComp 2024', role: 'Poster Session Chair', category: 'conference' },
  // 2023
  { id: 16, period: '2023.12.20-22', organization: '2023 한국소프트웨어종합학술대회 (KSC 2023)', role: '평가위원', category: 'conference' },
  { id: 17, period: '2023.12.10-16', organization: 'NeurIPS 2023', role: 'Member', category: 'conference' },
  { id: 18, period: '2023.11.03-04', organization: 'KDBC 2023', role: 'PC, 좌장', category: 'conference' },
  { id: 19, period: '2023.11.01', organization: '제4회 충북 K-한방 정밀의료 국제포럼', role: '좌장', category: 'forum' },
  { id: 20, period: '2023.10.27-29', organization: 'KJDB 2023: Korea-Japan Database Workshop 2023', role: 'Member', category: 'workshop' },
  { id: 21, period: '2023.06.19', organization: '2023 한국컴퓨터종합학술대회 (KCC2023)', role: '평가위원', category: 'conference' },
  { id: 22, period: '2023.06.18', organization: 'KCC 2023 데이터베이스 소사이어티 분과 워크샵', role: '프로그램 위원장', category: 'workshop' },
  { id: 23, period: '2023.03.02', organization: '2023 충북 K-한방 정밀의료 국제포럼', role: '좌장', category: 'forum' },
  { id: 24, period: '2023.02.13-16', organization: 'BigComp 2023: IEEE International Conference on Big Data and Smart Computing', role: 'Poster Session Chair', category: 'conference' },
  // 2022
  { id: 25, period: '2022.12.02-03', organization: 'KJDB 2022: Korea-Japan Database Workshop 2022', role: 'Organization Co-chairs', category: 'workshop' },
  { id: 26, period: '2022.07.18-19', organization: '2022년 전자공학회 영상이해/영상처리 연구회', role: 'Member', category: 'workshop' },
  { id: 27, period: '2022.06.30-07.01', organization: 'KCC 2022 한국컴퓨터종합학술대회', role: '평가위원', category: 'conference' },
  { id: 28, period: '2022.01.17', organization: 'VOICE AI 2022 Workshop', role: 'Chair', category: 'workshop' },
  { id: 29, period: '2022.01.17-20', organization: 'BigComp 2022: IEEE International Conference on Big Data and Smart Computing', role: 'PC member, On-line Presentation Co-Chairs', category: 'conference' },
  // 2021
  { id: 30, period: '2021.10.21', organization: '정보통신기획평가원 국제 인공지능 토론회 (Global AI Forum)', role: 'Member', category: 'forum' },
  { id: 31, period: '2021.04.28', organization: '2021 충북 K-한방 정밀의료 국제포럼', role: 'Speaker', category: 'forum' },
  { id: 32, period: '2021.01.17', organization: 'VOICE AI 2021 Workshop', role: 'Chair', category: 'workshop' },
  { id: 33, period: '2021.01.17-20', organization: 'BigComp2021: IEEE International Conference on Big Data and Smart Computing', role: 'PC member, Culture & Activity Co-Chairs', category: 'conference' },
  // 2020
  { id: 34, period: '2020.09.26-27', organization: 'PyCon Korea 2020', role: 'Member', category: 'conference' },
  { id: 35, period: '2020.08.10-12', organization: '서울대학교 AI 여름 학교', role: 'Member', category: 'workshop' },
  { id: 36, period: '2020.04.14-15', organization: '전자공학회 딥러닝 기술과 자율이동체 응용', role: 'Member', category: 'workshop' },
  { id: 37, period: '2020.02.19-22', organization: 'IEEE BigComp 2020', role: 'Culture & Activity Chair, Session Chair', category: 'conference' },
  // 2019
  { id: 38, period: '2019.12.18', organization: '2019 ETRI AI Practice Tech Day', role: 'Member', category: 'conference' },
  { id: 39, period: '2019.12.17', organization: '글로벌 AI 컨퍼런스 AICON 2019', role: 'Member', category: 'conference' },
  { id: 40, period: '2019.12.07', organization: 'MODUCON 2019', role: 'Member', category: 'conference' },
  { id: 41, period: '2019.12.03', organization: 'AI 기반 사회문제해결형 R/D 추진전략 컨퍼런스', role: 'Member', category: 'conference' },
  { id: 42, period: '2019.11.29', organization: '2019 마이데이터 컨퍼런스', role: 'Member', category: 'conference' },
  { id: 43, period: '2019.09.04', organization: 'AI HIDDEN STAR 2019 DEMO DAY & TALK', role: 'Member', category: 'conference' },
  { id: 44, period: '2019.08.28', organization: '해외전문가 튜토리얼 – eXplainable AI', role: 'Member', category: 'workshop' },
  { id: 45, period: '2019.07.25-26', organization: 'THE AI KOREA 2019 컨퍼런스', role: 'Member', category: 'conference' },
];

// 저널 리뷰어 인터페이스
export interface JournalReview {
  id: number;
  journal: string;
  publisher: string;
  reviewCount: number;
  role: string;
}

// 저널 리뷰어 활동 (총 204건, 50개 저널)
export const journalReviews: JournalReview[] = [
  { id: 1, journal: 'Applied Sciences', publisher: 'MDPI', reviewCount: 31, role: 'Reviewer' },
  { id: 2, journal: 'Electronics', publisher: 'MDPI', reviewCount: 19, role: 'Topical Advisory Panel, Guest Editor, Reviewer' },
  { id: 3, journal: 'The Journal of Supercomputing', publisher: 'Springer', reviewCount: 13, role: 'Reviewer' },
  { id: 4, journal: 'IEEE Access', publisher: 'IEEE', reviewCount: 12, role: 'Reviewer' },
  { id: 5, journal: 'PLoS ONE', publisher: 'PLOS', reviewCount: 12, role: 'Reviewer' },
  { id: 6, journal: 'Sensors', publisher: 'MDPI', reviewCount: 12, role: 'Reviewer' },
  { id: 7, journal: 'ACM Transactions on Multimedia Computing, Communications and Applications', publisher: 'ACM', reviewCount: 8, role: 'Reviewer' },
  { id: 8, journal: 'Mathematics', publisher: 'MDPI', reviewCount: 7, role: 'Reviewer' },
  { id: 9, journal: 'Advanced Intelligent Systems', publisher: 'Wiley', reviewCount: 6, role: 'Reviewer' },
  { id: 10, journal: 'Information', publisher: 'MDPI', reviewCount: 6, role: 'Reviewer' },
  { id: 11, journal: 'IEEE Sensors Journal', publisher: 'IEEE', reviewCount: 4, role: 'Reviewer' },
  { id: 12, journal: 'Remote Sensing', publisher: 'MDPI', reviewCount: 4, role: 'Reviewer' },
  { id: 13, journal: 'Symmetry', publisher: 'MDPI', reviewCount: 4, role: 'Reviewer' },
  { id: 14, journal: 'ACM Computing Surveys', publisher: 'ACM', reviewCount: 3, role: 'Reviewer' },
  { id: 15, journal: 'AI', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 16, journal: 'Algorithms', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 17, journal: 'Big Data and Cognitive Computing', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 18, journal: 'Brain Sciences', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 19, journal: 'Data', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 20, journal: 'Expert Systems with Applications', publisher: 'Elsevier', reviewCount: 3, role: 'Reviewer' },
  { id: 21, journal: 'Future Internet', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 22, journal: 'IEEE Transactions on Image Processing', publisher: 'IEEE', reviewCount: 3, role: 'Reviewer' },
  { id: 23, journal: 'Journal of Imaging', publisher: 'MDPI', reviewCount: 3, role: 'Reviewer' },
  { id: 24, journal: 'Aerospace', publisher: 'MDPI', reviewCount: 2, role: 'Reviewer' },
  { id: 25, journal: 'Cluster Computing', publisher: 'Springer', reviewCount: 2, role: 'Reviewer' },
  { id: 26, journal: 'Computers & Electrical Engineering', publisher: 'Elsevier', reviewCount: 2, role: 'Reviewer' },
  { id: 27, journal: 'Diagnostics', publisher: 'MDPI', reviewCount: 2, role: 'Reviewer' },
  { id: 28, journal: 'Education Sciences', publisher: 'MDPI', reviewCount: 2, role: 'Reviewer' },
  { id: 29, journal: 'IEEE Transactions on Computational Social Systems', publisher: 'IEEE', reviewCount: 2, role: 'Reviewer' },
  { id: 30, journal: 'IEEE Transactions on Industrial Informatics', publisher: 'IEEE', reviewCount: 2, role: 'Reviewer' },
  { id: 31, journal: 'Machine Learning and Knowledge Extraction', publisher: 'MDPI', reviewCount: 2, role: 'Reviewer' },
  { id: 32, journal: 'Pattern Recognition', publisher: 'Elsevier', reviewCount: 2, role: 'Reviewer' },
  { id: 33, journal: 'Technologies', publisher: 'MDPI', reviewCount: 2, role: 'Reviewer' },
  { id: 34, journal: 'Artificial Intelligence Review', publisher: 'Springer', reviewCount: 2, role: 'Reviewer' },
  { id: 35, journal: 'Big Data Research', publisher: 'Elsevier', reviewCount: 1, role: 'Reviewer' },
  { id: 36, journal: 'Computers', publisher: 'MDPI', reviewCount: 1, role: 'Reviewer' },
  { id: 37, journal: 'Concurrency and Computation', publisher: 'Wiley', reviewCount: 1, role: 'Reviewer' },
  { id: 38, journal: 'Connection Science', publisher: 'Taylor & Francis', reviewCount: 1, role: 'Reviewer' },
  { id: 39, journal: 'Engineering Applications of Artificial Intelligence', publisher: 'Elsevier', reviewCount: 1, role: 'Reviewer' },
  { id: 40, journal: 'IEEE Transactions on Engineering Management', publisher: 'IEEE', reviewCount: 1, role: 'Reviewer' },
  { id: 41, journal: 'IEEE Transactions on Services Computing', publisher: 'IEEE', reviewCount: 1, role: 'Reviewer' },
  { id: 42, journal: 'IET Generation, Transmission & Distribution', publisher: 'IET', reviewCount: 1, role: 'Reviewer' },
  { id: 43, journal: 'IET Image Processing', publisher: 'IET', reviewCount: 1, role: 'Reviewer' },
  { id: 44, journal: 'Information Sciences', publisher: 'Elsevier', reviewCount: 1, role: 'Reviewer' },
  { id: 45, journal: 'International Journal for Numerical Methods in Engineering', publisher: 'Wiley', reviewCount: 1, role: 'Reviewer' },
  { id: 46, journal: 'Journal of Experimental & Theoretical Artificial Intelligence', publisher: 'Taylor & Francis', reviewCount: 1, role: 'Reviewer' },
  { id: 47, journal: 'Knowledge', publisher: 'MDPI', reviewCount: 1, role: 'Reviewer' },
  { id: 48, journal: 'Proceedings of the Institution of Mechanical Engineers', publisher: 'SAGE', reviewCount: 1, role: 'Reviewer' },
];

// 저널 리뷰 통계
export const journalReviewStats = {
  totalReviews: 204,
  totalJournals: 48,
};

// 학회 멤버십
export const journalMemberships: AcademicActivity[] = [
  { id: 100, period: '2024 ~ 현재', organization: 'ACM', role: 'Member', category: 'membership' },
  { id: 101, period: '2022 ~ 현재', organization: 'IEEE', role: 'Member', category: 'membership' },
  { id: 102, period: '2022 ~ 현재', organization: '한국정보과학회 데이터소사이어티', role: '이사', category: 'membership' },
  { id: 103, period: '2020 ~ 현재', organization: '한국인공지능학회', role: '정회원', category: 'membership' },
  { id: 104, period: '2010 ~ 현재', organization: '한국정보과학회', role: '정회원', category: 'membership' },
  { id: 105, period: '2009 ~ 현재', organization: '대한전자공학회', role: '정회원', category: 'membership' },
];

// 자문, 심사, 평가위원 활동
export const advisoryActivities: AdvisoryActivity[] = [
  // 2025
  { id: 1, period: '2025.12.03', topic: '제천시 데이터 분석 아이디어 공모전', organization: '제천시청', role: '평가위원' },
  { id: 2, period: '2025.08.29', topic: '초록우산 어린이재단 DT&IT 자문', organization: '초록우산 어린이재단', role: '자문위원' },
  { id: 3, period: '2025.05.08', topic: '일자리평가시스템 과업심의회', organization: '중소벤처기업연구원', role: '자문위원' },
  { id: 4, period: '2025.02.20', topic: '2025~2026년 e나라도움 운영·유지보수사업', organization: '서울지방조달청 (한국재정정보원)', role: '평가위원' },
  // 2024
  { id: 5, period: '2024.11.18', topic: '지역혁신선도기업육성(R&D)', organization: '충북테크노파크', role: '기획위원' },
  { id: 6, period: '2024.11.14', topic: '정신의료기관 응급병상정보 공유시스템 구축', organization: '서울지방조달청 (한국보건의료정보원)', role: '평가위원' },
  { id: 7, period: '2024.11.07', topic: 'On-premise LLM 솔루션 개발', organization: '렉스코드(주)', role: '자문위원' },
  { id: 8, period: '2024.10.30', topic: '충북 반도체산업 기획보고서', organization: '충북테크노파크', role: '집필위원' },
  { id: 9, period: '2024.10.11', topic: '경기도교육청 대표홈페이지 재구축 사업 감리 용역', organization: '인천지방조달청 (경기도교육청)', role: '평가위원' },
  { id: 10, period: '2024.09.12', topic: '산재보험 요양정보 및 산재 고용보험 가입 완납 증명서 진위확인 개방체계 구축', organization: '대구지방조달청 (한국지능정보사회진흥원)', role: '평가위원' },
  { id: 11, period: '2024.08.29', topic: '2024년 무형유산 전승지원 통합플랫폼 고도화 사업 용역', organization: '조달청 (국가유산청)', role: '평가위원' },
  { id: 12, period: '2024.08.21', topic: '초록우산 아동지원플랫폼 구축 용역', organization: '초록우산 어린이재단', role: '평가위원' },
  { id: 13, period: '2024.08.20', topic: '초록우산 어린이재단 DT&IT 자문', organization: '초록우산 어린이재단', role: '자문위원' },
  { id: 14, period: '2024.08-10', topic: '지역리빙랩 - 데이터 기반 생활폐기물 효율적 수거를 위한 최적 경로 도출', organization: '충남대학교', role: 'Member' },
  { id: 15, period: '2024.07.19 ~ 현재', topic: '연구중심병원 육성 R&D', organization: '충북대학교병원', role: '자문위원' },
  { id: 16, period: '2024.06.19', topic: '보안강화를 위한 검찰네트워크 분리 9차 사업', organization: '서울지방조달청 (대검찰청)', role: '평가위원' },
  { id: 17, period: '2024.03.22', topic: '춘천 원도심 지하도상가 위치기반 홍보안내용 로봇구매 평가', organization: '강원정보문화산업진흥원', role: '평가위원' },
  { id: 18, period: '2024.01.17', topic: '미래신산업 분야 기업육성 R&D사업 기획', organization: '충북테크노파크', role: '기획위원' },
  { id: 19, period: '2024.01.09', topic: '춘천 원도심 스마트상점 육성 운영 용역 심사', organization: '강원정보문화산업진흥원', role: '평가위원' },
  // 2023
  { id: 20, period: '2023.11.24', topic: '강원랜드 IT 분야 채용 토론면접', organization: '강원랜드', role: '면접위원' },
  { id: 21, period: '2023.09.21', topic: 'CDC 주요성능 BMT 결과검토위원회', organization: 'TTA 한국정보통신기술협회', role: '평가위원' },
  { id: 22, period: '2023.08.30', topic: '춘천시 원도심 스마트상권통합시스템 합동점검', organization: '강원정보문화산업진흥원', role: '평가위원' },
  { id: 23, period: '2023.07.26', topic: 'AI 기반 추천 시스템 자문', organization: '㈜시즈소프트', role: '자문위원' },
  { id: 24, period: '2023.07.17', topic: '인공지능 탐구활동 멘토링', organization: '국립광주과학관', role: '멘토' },
  { id: 25, period: '2023.07.14', topic: '초록우산 어린이재단 DT&IT 자문', organization: '초록우산 어린이재단', role: '자문위원' },
  { id: 26, period: '2023.06.16 ~', topic: '중소기업 일자리평가시스템 개선 자문', organization: '중소벤처기업연구원', role: '자문위원' },
  { id: 27, period: '2023.04.06 ~', topic: 'CDC(Change Data Capture) BMT(성능)', organization: 'TTA 한국정보통신기술협회', role: '평가위원' },
  { id: 28, period: '2023.05.08 ~', topic: '언어모델 기반 SQL 생성 기술 자문', organization: '㈜범익', role: '자문위원' },
  { id: 29, period: '2023.05.01 ~', topic: 'AI 기반 사람 행위 분석 자문', organization: '㈜페스티벌온', role: '자문위원' },
  { id: 30, period: '2023.01.10 ~', topic: '충북지역주력산업육성사업', organization: '충북테크노파크', role: '자문위원, 집필위원 PM' },
  // 2022
  { id: 31, period: '2022.11.01-2024.12.31', topic: '공정 상황별 운영 최적화를 위한 AI 기반의 제조공정물류 최적화 기술개발', organization: '한국산업지능화협회', role: '전문위원' },
  { id: 32, period: '2022.10.13 ~', topic: '데이터 역량진단 및 강화방안 수립', organization: '한국수자원공사', role: '자문위원' },
  { id: 33, period: '2022.09.23 ~', topic: '강원지역혁신플랫폼', organization: '강원지역혁신플랫폼 총괄운영센터', role: '평가위원' },
  { id: 34, period: '2022.09.01-2023.06.30', topic: '자연어 처리 모델 개발 자문', organization: '㈜렉스코드', role: '자문위원' },
  { id: 35, period: '2022.08.09 ~', topic: 'IT인프라 환경 개선 및 DB이관', organization: '초록우산 어린이재단', role: '자문위원' },
  { id: 36, period: '2022.08.01 ~', topic: '데이터마이닝 기반 기업 운영, DB 구현, SQL 활용 온라인 콘텐츠 감수', organization: '한국기술교육대학교', role: '감수위원' },
  { id: 37, period: '2022.06.01-2022.12.31', topic: '실내 자율주행 데이터 구축 사업', organization: '강원대학교 산학협력단', role: '자문위원' },
  { id: 38, period: '2022.02.10 ~', topic: '중소기업 일자리 평가시스템', organization: '중소벤처기업연구원', role: '자문위원' },
  { id: 39, period: '2022.01-05', topic: '스마트 관광도시 비즈니스 모델 구축', organization: '가평군', role: '자문위원' },
  // 2021
  { id: 40, period: '2021.09.25-2021.10.16', topic: '한국코드페어 SW해커톤 멘토 활동', organization: 'NIA 한국정보화진흥원', role: '멘토' },
  { id: 41, period: '2021.07-2022.01', topic: '강원대학교 정보화사업, SW개발사업, RPA 구축 사업, 메타버스 플랫폼 사업 등', organization: '강원대학교', role: '평가위원' },
  { id: 42, period: '2021.04.30', topic: '방송통신분야 공무원 심사', organization: '강원도청', role: '면접위원' },
  { id: 43, period: '2021.03.15', topic: '이러닝 콘텐츠 개발 사업 (데이터분석을 위한 SQL 입문)', organization: '통계청 통계교육원', role: '검수위원' },
  // 2020
  { id: 44, period: '2020.10.31', topic: '한국코드페어 SW공모전 최종 본선 심사', organization: 'NIA 한국정보화진흥원', role: '심사위원' },
  { id: 45, period: '2020.10.24', topic: '한국코드페어 SW해커톤 심사', organization: 'NIA 한국정보화진흥원', role: '심사위원' },
  { id: 46, period: '2020.08.22-23', topic: '한국코드페어 SW를 통한 착한상상 심사', organization: 'NIA 한국정보화진흥원', role: '심사위원' },
  { id: 47, period: '2020.03.20', topic: '인공지능 중심 산업융합 집적단지 조성 사업 적정성 검토위원회', organization: 'NIPA 정보통신산업진흥원', role: '심사위원' },
  { id: 48, period: '2020.02.18', topic: '데이터 가공 자문 회의', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  // 2019
  { id: 49, period: '2019.12.09', topic: '마이데이터 서비스 안내서 수립 의견수렴회', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 50, period: '2019.11.28', topic: '마이데이터 자문 회의', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 51, period: '2019.10-12', topic: '데이터 기반 성과관리 및 정책결정을 위한 서비스 추진 방향 수립', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 52, period: '2019.10-11', topic: '데이터 안심구역 사업 활성화', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 53, period: '2019.10.12', topic: '한국코드페어 SW빌더스 챌린지', organization: 'NIA 한국정보화진흥원', role: '심사위원' },
  { id: 54, period: '2019.09.03', topic: '2019년 본인정보 활용지원(MyData) 실증서비스 중간보고회', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 55, period: '2019.07.19', topic: '데이터 통합거래 체계구축 및 세부계획(ISP) 수립', organization: 'KDATA 한국데이터산업진흥원', role: '외부전문가' },
  { id: 56, period: '2019.04.22', topic: 'K-CLOUD PARK 특화전략 수립 용역 최종보고회', organization: '강원도청 데이터시티추진단', role: '외부전문가' },
  // 2018
  { id: 57, period: '2018.11.19', topic: 'DOUZONE World Class 300 평가', organization: '더존비즈온', role: '평가위원' },
];

export const activityCategories = {
  conference: { label: '학술대회', color: 'blue' },
  workshop: { label: '워크샵', color: 'purple' },
  forum: { label: '포럼', color: 'green' },
  journal: { label: '저널 리뷰', color: 'orange' },
  membership: { label: '학회 멤버십', color: 'gray' },
};

export const advisoryRoleTypes = {
  '평가위원': 'blue',
  '자문위원': 'green',
  '심사위원': 'purple',
  '면접위원': 'orange',
  '멘토': 'pink',
  '외부전문가': 'gray',
  '기획위원': 'cyan',
  '집필위원': 'yellow',
  '감수위원': 'indigo',
  '검수위원': 'teal',
  '전문위원': 'rose',
};

// 연구과제 경력
export interface ResearchProject {
  id: number;
  fundingAgency: string;
  program: string;
  title: string;
  period: string;
  budget: string;
  description: string[];
  category: 'government' | 'industry' | 'university' | 'foundation';
}

export const researchProjects: ResearchProject[] = [
  // 2025-2026
  {
    id: 1,
    fundingAgency: '㈜네비웍스',
    program: '연구용역과제',
    title: '지휘결심 지원 및 가상 시뮬레이션을 위한 온톨로지 및 에이전트 개발',
    period: '2025.10-2026.03',
    budget: '55,000,000원',
    description: ['(대외비)'],
    category: 'industry',
  },
  {
    id: 2,
    fundingAgency: '중소기업기술정보진흥원',
    program: '창업성장기술개발(R&D)',
    title: '멀티모달 AI 기반 IR 피칭 평가 및 피드백 시스템 "피치스코어"',
    period: '2025.07-2026.12',
    budget: '200,000,000원',
    description: [
      '피치덱 문서와 발표 영상에 포함된 텍스트·음성·표정·행동 등 멀티모달 데이터를 통합 분석',
      '발표자의 전달력과 논리 구조를 수치화하고 자동 리포트를 제공하는 멀티모달기반 IR 평가 시스템',
    ],
    category: 'government',
  },
  {
    id: 3,
    fundingAgency: '㈜아이티공간',
    program: '연구용역과제',
    title: '전류 기반 이상 탐지 및 잔여수명 예측 AI 모델 개발',
    period: '2025.07-2025.12',
    budget: '20,000,000원',
    description: [
      '이상 탐지 모델 개발',
      '잔여수명(RUL) 예측 모델 개발',
      'Health Index 산출 알고리즘 개발',
    ],
    category: 'industry',
  },
  {
    id: 4,
    fundingAgency: '중소기업기술정보진흥원',
    program: '산학연Collabo R&D',
    title: '인종별 피부타입 맞춤형 미용 고주파 조사 AI 알고리즘 개발',
    period: '2025.05-2025.12',
    budget: '50,000,000원',
    description: [
      '인종별 피부 특성 분석 알고리즘 개발',
      '실시간 인종별 피부 반응 예측 모델 구현',
      '온디바이스 AI 기반 피드백 시스템 개발',
    ],
    category: 'government',
  },
  // 2024
  {
    id: 5,
    fundingAgency: '서울신용보증재단',
    program: '연구용역과제',
    title: '상권분석서비스 매출추정 방법론 개선 및 적용 연구',
    period: '2024.11-2024.12',
    budget: '88,000,000원',
    description: ['상권 데이터 매출 추정을 위한 Imputation 모델 설계 및 개발'],
    category: 'foundation',
  },
  {
    id: 6,
    fundingAgency: '중소기업기술정보진흥원',
    program: '중소기업기술혁신개발사업(R&D)',
    title: '금융분야 환경에 대응 가능한 생성형 언어모델 기반 Text-to-SQL 솔루션 서비스',
    period: '2024.10-2026.09',
    budget: '92,000,000원',
    description: ['그래프 데이터베이스 기반 지식베이스로 금융 특화 Text-to-SQL 솔루션 개발'],
    category: 'government',
  },
  {
    id: 7,
    fundingAgency: '중소기업기술정보진흥원',
    program: '창업성장기술개발사업(R&D)',
    title: '대규모 언어 모델 기반 공무원 수험생의 AI 학습지원 플랫폼',
    period: '2024.08-2025.07',
    budget: '40,000,000원',
    description: ['소형 언어 모델(sLLM)을 활용한 AI 기반 실시간 공무원 시험 질의응답과 개인 맞춤형 학습 자료 제공 서비스'],
    category: 'government',
  },
  {
    id: 8,
    fundingAgency: '교육부',
    program: '지방대학활성화사업',
    title: 'Triangle Research Program - WiFi CSI를 이용한 실내외 인간 활동 감지 및 분석',
    period: '2024.05-2024.12',
    budget: '12,000,000원',
    description: ['WiFi CSI 기반 인간 활동 감지 및 분석', '이상 행동 및 낙상 사고 실시간 탐지'],
    category: 'government',
  },
  {
    id: 9,
    fundingAgency: '교육부',
    program: '지자체-대학 협력기반 지역혁신 사업',
    title: '행동 분석 모델 개발 프로젝트랩',
    period: '2024.04-2024.12',
    budget: '30,000,000원',
    description: ['행동 패턴 데이터를 수집할 수 있는 모듈 개발', 'AI 모델 개발'],
    category: 'government',
  },
  // 2023
  {
    id: 10,
    fundingAgency: '교육부',
    program: '지방대학활성화사업',
    title: 'Triangle Research Program - 사용자 위치 기반 행동 패턴 분석 AI 모델',
    period: '2023.11-2024.01',
    budget: '10,000,000원',
    description: ['AI 기반 인파 관리를 위한 모델 개발', '군중 동선 및 밀집 예측 모델'],
    category: 'government',
  },
  {
    id: 11,
    fundingAgency: '교육부',
    program: '대학혁신지원사업',
    title: '시계열 데이터를 이용한 인공지능 모델',
    period: '2023.05-2023.12',
    budget: '8,000,000원',
    description: ['딥러닝을 이용한 시계열 데이터 분류 및 이상치 탐지', '딥러닝을 이용한 시계열, 시공간 데이터 예측'],
    category: 'government',
  },
  {
    id: 12,
    fundingAgency: '교육부',
    program: '지자체-대학 협력기반 지역혁신 사업',
    title: '인공지능 모델 개발 프로젝트랩',
    period: '2023.04-2023.12',
    budget: '25,000,000원',
    description: ['시계열 데이터에서 발생하는 이상치에 대한 분석 및 시각화', '설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발'],
    category: 'government',
  },
  // 2022
  {
    id: 13,
    fundingAgency: '교육부',
    program: '지자체-대학 협력기반 지역혁신 사업',
    title: '인공지능 모델 개발 프로젝트랩',
    period: '2022.10-2023.02',
    budget: '30,000,000원',
    description: ['시계열 데이터에서 발생하는 이상치에 대한 분석 및 시각화', '설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발'],
    category: 'government',
  },
  {
    id: 14,
    fundingAgency: '교육부',
    program: '대학혁신지원사업',
    title: '메가캡스톤디자인 - 자율주행모형차 구현을 위한 환경 구축 및 AI 모델 개발 (SMU Racer)',
    period: '2022.10-2023.01',
    budget: '50,000,000원',
    description: ['자율주행모형차 구현을 위한 환경 구축 및 AI 모델 개발', '자율주행 국제대회 참여 및 관련 작품 및 상품 개발'],
    category: 'government',
  },
  {
    id: 15,
    fundingAgency: '㈜베러마인드',
    program: '연구 용역',
    title: '수면사운드 분석을 위한 인공지능 모델 구축',
    period: '2022.06-2022.12',
    budget: '30,000,000원',
    description: ['수면 사운드 데이터 수집 및 분석', '수면 사운드 기반 상태 분류 모델 개발', '수면 사운드 분석 API 개발'],
    category: 'industry',
  },
  {
    id: 16,
    fundingAgency: '과학기술정보통신부',
    program: '사람중심인공지능핵심원천기술개발(R&D)',
    title: 'XVoice: 멀티모달 음성 메타학습',
    period: '2022.04-2026.12',
    budget: '4,750,000,000원',
    description: [
      '멀티 모달 생체신호 데이터 및 음성에 최적화된 메타 학습 기술 연구 개발',
      '멀티 모달 생체신호로부터 음성을 자연스럽게 합성하는 기술 연구 개발',
      '다양한 생체신호 기반 효율적인 모델 학습을 위한 다차원 텐서 기반의 특징 선택 기법 개발',
    ],
    category: 'government',
  },
  // 2021
  {
    id: 17,
    fundingAgency: '교육부',
    program: '지자체-대학 협력기반 지역혁신 사업',
    title: '인공지능 모델 설계 및 개발 프로젝트랩',
    period: '2021.10-2022.03',
    budget: '30,000,000원',
    description: ['제조 공장 설비에서 발생하는 이상 데이터에 대한 분석 및 시각화', '설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발'],
    category: 'government',
  },
  {
    id: 18,
    fundingAgency: '서울신용보증재단',
    program: '연구용역',
    title: '상권정보 기반 점포수 추정 알고리즘 개발 및 데이터 구축',
    period: '2021.08-2022.01',
    budget: '50,000,000원',
    description: ['점포수 추정을 위해 필요한 데이터 정의 및 확보 방안 제시', '상권정보 현황 기반 미래 운영점포수 추정 알고리즘 개발'],
    category: 'foundation',
  },
  {
    id: 19,
    fundingAgency: '교육부',
    program: '대학혁신지원사업',
    title: '메가캡스톤디자인 - 인공지능 패션 추천 기반 스타일 생성 및 아바타 가상피팅 모델 개발 (AI Fashion)',
    period: '2021.08-2022.01',
    budget: '50,000,000원',
    description: ['멀티모달 딥러닝 기반 패션 스타일 추천 모델 설계 및 학습', '딥러닝 기반 추천 및 아바타 가상 피팅 서빙을 위한 시스템 구축'],
    category: 'government',
  },
  {
    id: 20,
    fundingAgency: '서울산업진흥원',
    program: 'DMC 적재적소 프로그램',
    title: 'CCTV기반 건설산업현장 출입관리 인공지능 서비스 개발',
    period: '2021.05-2021.11',
    budget: '20,000,000원',
    description: ['CCTV기반 위험지역 내 작업 탐지 인공지능 서비스 구축', '작업근로자의 안전모 착용여부 탐지 서비스'],
    category: 'foundation',
  },
  // 2020
  {
    id: 22,
    fundingAgency: '정보통신기획평가원',
    program: '인공지능 융합선도 프로젝트 사업',
    title: '스마트시티 산업 생산성 혁신을 위한 AI융합 기술 개발',
    period: '2019.04-2021.12',
    budget: '4,689,636,000원',
    description: ['스마트시티 안전관련 개인정보 비식별화 정의 및 관련 모델과 기법 분석', '영상 내 개인정보보호를 위한 사람과 객체의 비식별화 딥러닝 모델 개발'],
    category: 'government',
  },
  {
    id: 23,
    fundingAgency: '강원도청',
    program: '민간보조사업',
    title: '데이터산업 창업 활성화 및 인재양성 프로그램 추진사업',
    period: '2019.10-2019.12',
    budget: '200,000,000원',
    description: ['빅데이터 비즈니스 모델 개발 및 안전한 데이터 활용 연구', '빅데이터 인력양성 프로그램 설계 및 운영 방안 연구'],
    category: 'government',
  },
  // 2019
  {
    id: 24,
    fundingAgency: '정보통신기술진흥센터',
    program: 'SW중심대학',
    title: 'SW중심대학（강원대학교）',
    period: '2019.02-2019.11',
    budget: '2,088,000,000원',
    description: ['전공 교육 및 학생 지도', '데이터사이언스학과 전공 강의', 'SW 교육 콘텐츠 및 교재 개발'],
    category: 'government',
  },
  {
    id: 25,
    fundingAgency: '한국데이터산업진흥원',
    program: '용역과제',
    title: '데이터 이동권 도입 방안 연구',
    period: '2019.08-2019.11',
    budget: '49,000,000원',
    description: ['데이터 이동을 위한 기술 및 체계 방안 연구'],
    category: 'foundation',
  },
  {
    id: 26,
    fundingAgency: '정보통신기획평가원',
    program: '방송통신정책연구사업',
    title: '공급망(Supply-Chain) 보안체계 수립을 위한 인증프레임워크 구축 방안 연구',
    period: '2019.04-2019.11',
    budget: '72,312,000원',
    description: ['공급망 보안체계 인증프레임워크 연구'],
    category: 'government',
  },
  {
    id: 27,
    fundingAgency: '한국생산기술연구원',
    program: '용역과제',
    title: '로봇 조작 및 판단기능 기술을 위한 인간 작업 데이터 생성 및 저장',
    period: '2019.03-2019.05',
    budget: '49,500,000원',
    description: ['로봇 빅데이터의 질의/검색 처리를 위한 수집 및 저장 연구', '로봇 빅데이터의 분류를 위한 머신러닝 및 분석 연구'],
    category: 'foundation',
  },
  // 2018
  {
    id: 28,
    fundingAgency: '국립축산과학원',
    program: '초지사료 생산성 향상 및 이용기술 개발',
    title: '풀사료의 생산성 조사 및 예측식에 대한 소프트웨어 개발',
    period: '2018.06-2018.12',
    budget: '120,000,000원',
    description: ['풀사료의 생산성 예측을 위한 빅데이터 활용 방안 연구', '풀사료 생산성 예측 앱 설계 및 개발'],
    category: 'government',
  },
  {
    id: 29,
    fundingAgency: '한국전력공사 전력연구원',
    program: '에너지 거점대학 클러스터 사업',
    title: 'XAI (eXplainable AI) 기반 스마트 에너지 플랫폼 기술 개발',
    period: '2018.03-2019.01',
    budget: '383,260,381원',
    description: ['스마트 에너지 데이터 분산 저장 및 분석', '스마트 에너지 보안 플랫폼 개발'],
    category: 'foundation',
  },
  {
    id: 30,
    fundingAgency: '정보통신기술진흥센터',
    program: 'SW중심대학',
    title: 'SW중심대학（강원대학교）',
    period: '2018.03-2018.12',
    budget: '1,072,000,000원',
    description: ['SW 교육 연구', 'SW 교육 콘텐츠 및 교재 개발'],
    category: 'government',
  },
  // 2017
  {
    id: 31,
    fundingAgency: '산업통산자원부',
    program: '지식서비스산업핵심기술개발사업',
    title: '실 사례 기반 경영 게임을 통한 기업 시뮬레이션 플랫폼 개발',
    period: '2017.06-2018.05',
    budget: '54,600,000원',
    description: ['R을 이용한 상권 데이터 분석 도구 개발', 'TensorFlow를 이용한 상권 추천 딥러닝 모델 개발'],
    category: 'government',
  },
  {
    id: 32,
    fundingAgency: '교육부',
    program: '정책연구개발사업',
    title: '빅데이터 시대에 대응한 교육정보·통계 정책과제 개발 연구',
    period: '2017.04-2017.09',
    budget: '28,000,000원',
    description: ['교육정보·통계데이터의 빅데이터 활용을 위한 정책 연구 및 제언'],
    category: 'government',
  },
  // 2016
  {
    id: 33,
    fundingAgency: '산업통산자원부',
    program: '로봇산업핵심기술개발(R&D)',
    title: '제조로봇용 실시간 지원 SW 플랫폼 기술 개발',
    period: '2016.07-2016.12',
    budget: '181,000,000원',
    description: ['제조로봇의 다양한 데이터 수집 및 저장을 위한 빅데이터 플랫폼 기술 개발'],
    category: 'government',
  },
  {
    id: 34,
    fundingAgency: '산업통산자원부',
    program: '지식서비스산업핵심기술개발사업',
    title: '실 사례 기반 경영 게임을 통한 기업 시뮬레이션 플랫폼 개발',
    period: '2016.06-2017.05',
    budget: '41,280,000원',
    description: ['Python을 이용한 데이터 수집 가공 및 크롤링 개발', 'R을 이용한 경영 빅데이터 시각화 도구 개발'],
    category: 'government',
  },
  {
    id: 35,
    fundingAgency: '한국연구재단',
    program: '지역신산업선도인력양성사업',
    title: '지능형 해부학적 조직 인식 빅데이터를 활용한 줄기세포 분화용 3차원 조직공학 지지체 제조',
    period: '2016.06-2017.12',
    budget: '160,057,060원',
    description: ['줄기세포의 형광 염색 이미지 처리 및 분석', '바이오 빅데이터 분석을 위한 플랫폼 구축 및 연구', '머신 러닝을 이용한 바이오 데이터 학습'],
    category: 'government',
  },
  // 2013
  {
    id: 36,
    fundingAgency: '한국연구재단',
    program: '지역대학우수과학자지원사업',
    title: '맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP',
    period: '2013.05-2014.04',
    budget: '44,518,432원',
    description: ['모바일에서 사진 데이터의 효과적인 관리와 브라우징을 위한 데이터 큐브 저장 연구'],
    category: 'government',
  },
  {
    id: 37,
    fundingAgency: '미래창조과학부',
    program: 'SW컴퓨팅산업원천기술개발',
    title: '대용량 센서 스트림 데이터를 실시간으로 처리하는 개방형 센서 DBMS 개발',
    period: '2011.12-2013.11',
    budget: '1,000,000,000원',
    description: ['실시간 스트림 데이터 처리를 위한 CEP 엔진 연구', '센서 데이터의 효율적인 처리 및 저장을 위한 센서 DBMS 개발'],
    category: 'government',
  },
  // 2011-2012
  {
    id: 38,
    fundingAgency: '한국연구재단',
    program: '지역대학우수과학자지원사업',
    title: '맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP',
    period: '2012.05-2012.10',
    budget: '44,980,619원',
    description: ['맵리듀스를 이용한 대규모 데이터 큐브의 분산 병렬 계산 연구'],
    category: 'government',
  },
  {
    id: 39,
    fundingAgency: '한국연구재단',
    program: '지역대학우수과학자지원사업',
    title: '맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP',
    period: '2011.05-2012.04',
    budget: '44,450,000원',
    description: ['비정형 텍스트 데이터의 큐브 계산 및 다차원 분석 연구'],
    category: 'government',
  },
  {
    id: 40,
    fundingAgency: '한국연구재단',
    program: '중견연구자지원사업',
    title: '분산 컴퓨팅 환경에서 프라이버시 보호 시계열 데이터 마이닝',
    period: '2011.05-2012.04',
    budget: '100,000,000원',
    description: ['분산 컴퓨팅 환경에서 데이터 마이닝 알고리즘 연구'],
    category: 'government',
  },
  {
    id: 41,
    fundingAgency: '한국연구재단',
    program: '신진연구지원사업',
    title: '프로비넌스에 기반한 위치 및 스트림 데이터의 신뢰도 평가',
    period: '2011.05-2012.04',
    budget: '50,000,000원',
    description: ['스트림 데이터의 다차원 분석 연구'],
    category: 'government',
  },
  // 2010
  {
    id: 42,
    fundingAgency: '중소기업청',
    program: '앱(App)창작터 지정운영사업',
    title: '앱(App)창작터 사업',
    period: '2010.07-2010.12',
    budget: '118,000,000원',
    description: ['앱 개발 교육 지원', '안드로이드 개발 및 실습 지원'],
    category: 'government',
  },
  // 2007-2008
  {
    id: 43,
    fundingAgency: '강원대학교',
    program: '캠퍼스간 공동연구비',
    title: '유비쿼터스 센서 네트워크에서 스트림 데이터에 대한 효율적인 다 차원 분석 기술',
    period: '2007.12-2008.12',
    budget: '16,250,000원',
    description: ['스트림 데이터의 효과적인 처리 및 저장 기법 연구', '데이터 스트림 관리 시스템 (DSMS) 엔진 관련 연구'],
    category: 'university',
  },
  // 2006-2007
  {
    id: 44,
    fundingAgency: 'KAIST',
    program: '우수연구센터지원사업',
    title: '정보 및 지식관리 연구',
    period: '2006.03-2007.02',
    budget: '50,032,480원',
    description: ['스프레드시트를 이용한 OLAP 도구 엔진 설계 및 구현', 'OLAP 엔진의 Pivot 알고리즘 개발'],
    category: 'university',
  },
];

export const projectCategories = {
  government: { label: '정부과제', color: 'blue' },
  industry: { label: '산업체과제', color: 'green' },
  university: { label: '대학과제', color: 'purple' },
  foundation: { label: '재단/기관과제', color: 'orange' },
};

// 연구과제 통계
export const projectStats = {
  totalProjects: 45,
  totalYears: 19,
};
