export interface Project {
  id: number;
  title: string;
  organization: string;
  program: string;
  period: string;
  budget: string;
  completed: boolean;
  items: string[];
  url?: string;
}

export const projects: Project[] = [
  {
    "id": 1,
    "title": "지휘결심 지원 및 가상 시뮬레이션을 위한 온톨로지 및 에이전트 개발",
    "organization": "㈜네비웍스",
    "program": "연구용역과제",
    "period": "2025.10.01-2026.03.31",
    "budget": "55,000,000",
    "completed": false,
    "items": [
      "지휘결심 지원 온톨로지 개발",
      "가상 시뮬레이션 에이전트 개발"
    ]
  },
  {
    "id": 2,
    "title": "멀티모달 AI 기반 IR 피칭 평가 및 피드백 시스템 \"피치스코어\"",
    "organization": "중소기업기술정보진흥원",
    "program": "창업성장기술개발(R&D)",
    "period": "2025.07.01-2026.12.31",
    "budget": "200,000,000",
    "completed": false,
    "items": [
      "피치덱 문서와 발표 영상에 포함된 텍스트·음성·표정·행동 등 멀티모달 데이터를 통합 분석",
      "발표자의 전달력과 논리 구조를 수치화하고 자동 리포트를 제공하는 멀티모달기반 IR 평가 시스템"
    ]
  },
  {
    "id": 3,
    "title": "전류 기반 이상 탐지 및 잔여수명 예측 AI 모델 개발",
    "organization": "㈜아이티공간",
    "program": "연구용역과제",
    "period": "2025.07.01-2025.12.31",
    "budget": "20,000,000",
    "completed": true,
    "items": [
      "이상 탐지 모델 개발",
      "잔여수명(RUL) 예측 모델 개발",
      "Health Index 산출 알고리즘 개발",
      "AI 통합 예측 운영 모델 구축"
    ]
  },
  {
    "id": 4,
    "title": "인종별 피부타입 맞춤형 미용 고주파 조사 AI 알고리즘 개발",
    "organization": "중소기업기술정보진흥원",
    "program": "산학연CollaboR&D(R&D)",
    "period": "2025.05.01-2025.12.31",
    "budget": "50,000,000",
    "completed": true,
    "items": [
      "인종별 피부 특성 분석 알고리즘 개발",
      "실시간 인종별 피부 반응 예측 모델 구현",
      "온디바이스 AI 기반 피드백 시스템 개발"
    ]
  },
  {
    "id": 5,
    "title": "상권분석서비스 매출추정 방법론 개선 및 적용 연구",
    "organization": "서울신용보증재단",
    "program": "연구용역과제",
    "period": "2024.11.15-2024.12.13",
    "budget": "88,000,000",
    "completed": true,
    "items": [
      "상권 데이터 매출 추정을 위한 Imputation 모델 설계 및 개발"
    ]
  },
  {
    "id": 6,
    "title": "금융분야 환경에 대응 가능한 생성형 언어모델 기반 Text-to-SQL 솔루션 서비스",
    "organization": "중소기업기술정보진흥원",
    "program": "중소기업기술혁신개발사업(R&D)",
    "period": "2024.10.01-2026.09.30",
    "budget": "92,000,000",
    "completed": false,
    "items": [
      "금융 산업의 다양한 사용자 네트워크 환경을 지원",
      "그래프 데이터베이스 기반 지식베이스로 금융 특화 단어 이해",
      "사용자가 생성 결과를 보정하여 정확도를 향상시키는 생성형 언어모델 기반 금융 특성화 Text-to-SQL 솔루션 개발"
    ]
  },
  {
    "id": 7,
    "title": "대규모 언어 모델 기반 공무원 수험생의 AI 학습지원 플랫폼",
    "organization": "중소기업기술정보진흥원",
    "program": "창업성장기술개발사업(R&D)",
    "period": "2024.08.01-2025.07.31",
    "budget": "40,000,000",
    "completed": true,
    "items": [
      "소형 언어 모델(sLLM)을 활용한 AI 기반 실시간 공무원 시험 질의응답",
      "개인 맞춤형 학습 자료 제공 서비스"
    ]
  },
  {
    "id": 8,
    "title": "WiFi CSI를 이용한 실내외 인간 활동 감지 및 분석",
    "organization": "교육부",
    "program": "지방대학활성화사업 Triangle Research Program",
    "period": "2024.05.01-2024.12.31",
    "budget": "12,000,000",
    "completed": true,
    "items": [
      "WiFi CSI 기반 인간 활동 감지 및 분석",
      "실내 환경에서 재실자 수 정확 파악",
      "이상 행동 및 낙상 사고 실시간 탐지"
    ]
  },
  {
    "id": 9,
    "title": "행동 분석 모델 개발 프로젝트랩",
    "organization": "교육부",
    "program": "지자체-대학 협력기반 지역혁신 사업",
    "period": "2024.04.01-2024.12.31",
    "budget": "30,000,000",
    "completed": true,
    "items": [
      "다양한 실내외 환경에서 효과적으로 행동 패턴 데이터를 수집할 수 있는 모듈 개발",
      "고도화된 머신러닝 및 딥러닝 알고리즘을 활용하여 수집된 행동 패턴 데이터를 분석하는 AI 모델 개발"
    ]
  },
  {
    "id": 10,
    "title": "사용자 위치 기반 행동 패턴 분석 AI 모델",
    "organization": "교육부",
    "program": "지방대학활성화사업 Triangle Research Program",
    "period": "2023.11.01-2024.01.31",
    "budget": "10,000,000",
    "completed": true,
    "items": [
      "AI 기반 인파 관리를 위한 모델 개발",
      "군중 동선 및 밀집 예측 모델"
    ]
  },
  {
    "id": 11,
    "title": "시계열 데이터를 이용한 인공지능 모델",
    "organization": "교육부",
    "program": "대학혁신지원사업 - 전공역량 학술연구 프로그램",
    "period": "2023.05.01-2023.12.31",
    "budget": "8,000,000",
    "completed": true,
    "items": [
      "딥러닝을 이용한 시계열 데이터 분류 및 이상치 탐지",
      "딥러닝을 이용한 시계열, 시공간 데이터 예측"
    ]
  },
  {
    "id": 12,
    "title": "인공지능 모델 개발 프로젝트랩 (2023)",
    "organization": "교육부",
    "program": "지자체-대학 협력기반 지역혁신 사업",
    "period": "2023.04.01-2023.12.31",
    "budget": "25,000,000",
    "completed": true,
    "items": [
      "시계열 데이터에서 발생하는 이상치에 대한 분석 및 시각화",
      "설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발"
    ]
  },
  {
    "id": 13,
    "title": "인공지능 모델 개발 프로젝트랩 (2022)",
    "organization": "교육부",
    "program": "지자체-대학 협력기반 지역혁신 사업",
    "period": "2022.10.01-2023.02.28",
    "budget": "30,000,000",
    "completed": true,
    "items": [
      "시계열 데이터에서 발생하는 이상치에 대한 분석 및 시각화",
      "설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발"
    ]
  },
  {
    "id": 14,
    "title": "자율주행모형차 구현을 위한 환경 구축 및 AI 모델 개발 (SMU Racer)",
    "organization": "교육부",
    "program": "대학혁신지원사업 메가캡스톤디자인",
    "period": "2022.10.01-2023.01.31",
    "budget": "50,000,000",
    "completed": true,
    "items": [
      "자율주행모형차 구현을 위한 환경 구축 및 AI 모델 개발",
      "자율주행 기반 기술인 객체 탐지 및 세그멘테이션 모델 개발",
      "자율주행 국제대회 참여 및 관련 작품 및 상품 개발"
    ]
  },
  {
    "id": 15,
    "title": "수면사운드 분석을 위한 인공지능 모델 구축",
    "organization": "㈜베러마인드",
    "program": "연구 용역",
    "period": "2022.06.20-2022.12.31",
    "budget": "30,000,000",
    "completed": true,
    "items": [
      "수면 사운드 데이터 수집 및 분석",
      "수면 사운드 기반 상태 분류 모델 개발",
      "수면 사운드 분석 API 개발"
    ]
  },
  {
    "id": 16,
    "title": "XVoice: 멀티모달 음성 메타학습",
    "organization": "과학기술정보통신부",
    "program": "사람중심인공지능핵심원천기술개발(R&D)",
    "period": "2022.04.01-2026.12.31",
    "budget": "4,750,000,000",
    "completed": false,
    "items": [
      "멀티 모달 생체신호 데이터 및 음성에 최적화된 메타 학습 기술 연구 개발",
      "멀티 모달 생체신호로부터 음성을 자연스럽게 합성하는 기술 연구 개발",
      "다양한 생체신호 기반 효율적인 모델 학습을 위한 다차원 텐서 기반의 특징 선택 기법 개발",
      "다양한 생체신호 기반의 멀티모달 시스템 구축 및 데이터 처리 기술 개발",
      "반지도 학습 기반 텍스트 메타학습 기법 개발"
    ]
  },
  {
    "id": 17,
    "title": "인공지능 모델 설계 및 개발 프로젝트랩",
    "organization": "교육부",
    "program": "지자체-대학 협력기반 지역혁신 사업",
    "period": "2021.10.01-2022.03.31",
    "budget": "30,000,000",
    "completed": true,
    "items": [
      "제조 공장 설비에서 발생하는 이상 데이터에 대한 분석 및 시각화",
      "설비 상태와 결함 및 고장 발생 패턴 분석 및 예측 모델 개발"
    ]
  },
  {
    "id": 18,
    "title": "상권정보 기반 점포수 추정 알고리즘 개발 및 데이터 구축",
    "organization": "서울신용보증재단",
    "program": "점포수 추정을 통한 자영업 자가진단 알고리즘 개발적용 연구",
    "period": "2021.08.01-2022.01.05",
    "budget": "50,000,000",
    "completed": true,
    "items": [
      "점포수 추정을 위해 필요한 데이터 정의 및 확보방안 제시",
      "상권정보 현황 기반 미래 운영점포수 추정 알고리즘 개발",
      "추정 점포수 데이터의 집계 및 적재 기능 개발"
    ],
    "url": "https://www.seoulshinbo.co.kr/"
  },
  {
    "id": 19,
    "title": "인공지능 패션 추천 기반 스타일 생성 및 아바타 가상피팅 모델 개발 (AI Fashion)",
    "organization": "교육부",
    "program": "대학혁신지원사업 메가캡스톤디자인",
    "period": "2021.08.01-2022.01.31",
    "budget": "50,000,000",
    "completed": true,
    "items": [
      "멀티모달 딥러닝 기반 패션 스타일 추천 모델 설계 및 학습",
      "사용자 맞춤형 패션 스타일 가상 피팅 모델 설계 및 학습",
      "딥러닝 기반 추천 및 아바타 가상 피팅 서빙을 위한 시스템 구축"
    ]
  },
  {
    "id": 20,
    "title": "광범위한 건설 현장에서 스마트 안전을 위한 영상 인식 기술 개발",
    "organization": "교육부",
    "program": "대학혁신지원사업 사제동행형 산학협력 프로젝트",
    "period": "2021.04.01-2021.07.31",
    "budget": "3,000,000",
    "completed": true,
    "items": [
      "딥러닝 기반 사람 인식을 이용한 위험 작업 위치 탐지",
      "딥러닝 기반 객체 인식을 이용한 안전 장비 착용 유무"
    ]
  },
  {
    "id": 21,
    "title": "충청북도 바이오헬스산업 지역혁신 플랫폼",
    "organization": "교육부",
    "program": "지자체-대학 협력기반 지역혁신 사업",
    "period": "2020.08.01-2024.05.31",
    "budget": "213,000,000,000",
    "completed": true,
    "items": [
      "충북형 Bologna Process 도입(Osong Bio Tech. 설립)을 통한 글로벌 통합 인재 양성",
      "학제간 융합(바이오헬스와 인공지능, 빅데이터, IoT, BIT)을 통한 융·복합 인재 양성",
      "Micro-Degree 프로그램 운영을 통한 산업 맞춤형 인재 양성"
    ],
    "url": "http://www.biopride.or.kr/"
  },
  {
    "id": 22,
    "title": "CCTV기반 건설산업현장 출입관리 인공지능 서비스 개발",
    "organization": "서울산업진흥원",
    "program": "DMC 적재적소 프로그램(적재지원형)",
    "period": "2021.05.01-2021.11.30",
    "budget": "20,000,000",
    "completed": true,
    "items": [
      "CCTV기반 위험지역 내 작업 탐지 인공지능 서비스 구축",
      "작업근로자의 위험지역 내 작업자 탐지 서비스",
      "작업근로자의 안전모 착용여부 탐지 서비스",
      "영상 Visual Marker 기반 디코딩 기술기반 객체정보 서비스를 제공하여 건설산업현장 안전관리 고도화와 재해예방 혁신화"
    ],
    "url": "https://new.sba.kr/"
  },
  {
    "id": 22,
    "title": "강릉 IP융복합 콘텐츠 클러스터 조성사업 계획 용역",
    "organization": "강릉과학산업진흥원",
    "program": "기술이전",
    "period": "2020.11.26-2020.12.21",
    "budget": "17,000,000",
    "completed": true,
    "items": [
      "강릉 IP융복합 콘텐츠 클러스터 조성방안 수립",
      "콘텐츠 산업 등 4차산업에 대한 정부정책 지원동향",
      "지역의 산업, 기술, 문화가 융합된 관광명소 IP융복합 콘텐츠 클러스터 조성"
    ],
    "url": "https://www.gsif.or.kr/"
  },
  {
    "id": 23,
    "title": "인공지능기반 스마트 의정 분석 및 구축 기술",
    "organization": "재단법인 여의도연구원",
    "program": "기술이전",
    "period": "2020.08.01-2020.09.18",
    "budget": "15,000,000",
    "completed": true,
    "items": [
      "정치인 정보, 회의록, 인용문 분석 및 시각화",
      "키워드 분석을 통한 최근 트렌드 시각화",
      "뉴스 기사, 댓글에 대한 자연어 처리, 시각화, KcBERT기반 감정 분석 모델 개발"
    ],
    "url": "http://www.ydi.or.kr/"
  },
  {
    "id": 24,
    "title": "스마트시티 산업 생산성 혁신을 위한 AI융합 기술 개발",
    "organization": "세종대학교 산학협력단",
    "program": "인공지능 융합선도 프로젝트 사업",
    "period": "2020.01.01-2020.12.31",
    "budget": "4,689,636,000",
    "completed": true,
    "items": [
      "스마트시티 안전관련 개인정보 비식별화 정의 및 관련 모델과 기법 분석",
      "영상 내 개인정보보호를 위한 사람과 객체의 비식별화 딥러닝 모델 개발"
    ],
    "url": "https://ezone.iitp.kr/common/co_0701/view?PMS_TSK_DGR_ID=2019-0-00136-003&cPage=33&PMS_SRCHCHOICE1=&PMS_SRCHTEXT1=&PMS_SRCHCHOICE2=&PMS_SRCHTEXT2=&PMS_BEGDT=&PMS_ENDDT=&PMS_CHECK1=&PMS_CHECK2=&PMS_CHECK3="
  },
  {
    "id": 25,
    "title": "언어장애 · 언어소외 계층을 위한 인공지능 토킹 시스템 개발 사전기획연구",
    "organization": "과학기술정보통신부",
    "program": "다부처공동기획연구지원사업",
    "period": "2019.10.01-2019.12.31",
    "budget": "50,000,000",
    "completed": true,
    "items": [
      "VOICE AI 기술 기반 언어평등 사회 구현",
      "(VOICE 트윈) AI 토킹시스템, 조음DB, 오픈 라이브러리, 딥러닝 허브, 3D 언어 디지털 트윈 구축",
      "(VOICE 키트) 사용자 친화적 IoT 조음센서, 시스템 반도체, 저전력 무선통신, 웨어러블 디바이스 기술 개발",
      "(VOICE 에듀) 언어치료 및 언어교육 프로그램, 조음 및 언어 빅데이터, 지능형 피드백 및 사용자 인터페이스 개발"
    ],
    "url": "https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=TRKO202000005790"
  },
  {
    "id": 26,
    "title": "데이터산업 창업 활성화 및 인재양성 프로그램 추진사업",
    "organization": "강원도청",
    "program": "민간보조사업",
    "period": "2019.10.01-2019.12.31",
    "budget": "200,000,000",
    "completed": true,
    "items": [
      "빅데이터 비즈니스 모델 개발 및 안전한 데이터 활용 연구",
      "빅데이터 인력양성 프로그램 설계 및 운영 방안 연구",
      "데이터 혁신 생태계 조성 기반 마련 연구"
    ],
    "url": "http://www.provin.gangwon.kr/gw/portal/sub05_02?articleSeq=159655&mode=readForm&curPage=2&boardCode=BDAADD02"
  },
  {
    "id": 27,
    "title": "지능형 CCTV 영상 탐지 인공지능 서비스",
    "organization": "모두의연구소",
    "program": "용역과제",
    "period": "2019.08.01-2019.12.31",
    "budget": "6,600,000",
    "completed": true,
    "items": [
      "객체 정보를 포함한 객체 탐지 및 이상 탐지 연구"
    ],
    "url": "https://modulabs.co.kr/"
  },
  {
    "id": 28,
    "title": "데이터 이동권 도입 방안 연구",
    "organization": "한국데이터산업진흥원",
    "program": "용역과제",
    "period": "2019.08.01-2019.11.30",
    "budget": "49,000,000",
    "completed": true,
    "items": [
      "데이터 이동을 위한 기술 및 체계 방안 연구"
    ],
    "url": "https://kdata.or.kr/board/notice_01_view.html?field=&keyword=&type=notice&page=1&dbnum=1986&mode=detail&type=notice"
  },
  {
    "id": 29,
    "title": "공급망（Supply-Chain）보안체계 수립을 위한 인증프레임워크 구축 방안 연구",
    "organization": "정보통신기획평가원",
    "program": "방송통신정책연구사업",
    "period": "2019.04.01-2019.11.30",
    "budget": "72,312,000",
    "completed": true,
    "items": [
      "공급망 보안체계 인증프레임워크 연구"
    ],
    "url": "https://ezone.iitp.kr/common/co_0701/view?PMS_TSK_DGR_ID=2019-0-01525-001&cPage=2&PMS_SRCHCHOICE1=&PMS_SRCHTEXT1=&PMS_SRCHCHOICE2=&PMS_SRCHTEXT2=&PMS_BEGDT=&PMS_ENDDT=&PMS_CHECK1=&PMS_CHECK2=&PMS_CHECK3="
  },
  {
    "id": 30,
    "title": "로봇 조작 및 판단기능 기술을 위한 인간 작업 데이터 생성 및 저장",
    "organization": "한국생산기술연구원",
    "program": "용역과제",
    "period": "2019.03.01-2019.05-30",
    "budget": "49,500,000",
    "completed": true,
    "items": [
      "로봇 빅데이터의 질의/검색 처리를 위한 수집 및 저장 연구",
      "로봇 빅데이터의 분류를 위한 머신러닝 및 분석 연구"
    ]
  },
  {
    "id": 31,
    "title": "SW중심대학(강원대학교)",
    "organization": "정보통신기술진흥센터",
    "program": "SW중심대학",
    "period": "2019.02.01-2019.11.30",
    "budget": "2,088,000,000",
    "completed": true,
    "items": [
      "4차 산업 혁명을 선도할 4C(Challenge, Creative, Collaborative, Convergent)형 지능정보기술 SW인재 양성"
    ],
    "url": "https://ezone.iitp.kr/common/co_0701/view?PMS_TSK_DGR_ID=2018-0-00191-004&cPage=2&PMS_SRCHCHOICE1=ALL&PMS_SRCHTEXT1=SW%EC%A4%91%EC%8B%AC%EB%8C%80%ED%95%99&PMS_SRCHCHOICE2=&PMS_SRCHTEXT2=&PMS_BEGDT=&PMS_ENDDT=&PMS_CHECK1=&PMS_CHECK2=&PMS_CHECK3="
  },
  {
    "id": 32,
    "title": "풀사료의 생산성 조사 및 예측식에 대한 소프트웨어 개발",
    "organization": "국립축산과학원",
    "program": "초지사료 생산성 향상 및 이용기술 개발",
    "period": "2018.06.01-2018.12.31",
    "budget": "120,000,000",
    "completed": true,
    "items": [
      "풀사료의 생산성 예측을 위한 빅데이터 활용 방안 연구",
      "풀사료 생산성 예측 시각화 도구 개발",
      "풀사료 생산성 예측 앱 설계 및 개발"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1395055017&pageCode=TH_TOTAL_PJT_DTL"
  },
  {
    "id": 33,
    "title": "로봇 조작 및 판단기능 기술을 위한 인간 작업 데이터 생성 및 저장",
    "organization": "한국생산기술연구원",
    "program": "용역과제",
    "period": "2018.04.01-2018.06.14",
    "budget": "40,000,000",
    "completed": true,
    "items": [
      "인간 작업 영상 시퀀스 annotation 및 분류",
      "로봇 빅데이터의 질의/검색 처리를 위한 수집 및 저장 연구"
    ]
  },
  {
    "id": 34,
    "title": "SW중심대학(강원대학교)",
    "organization": "정보통신기술진흥센터",
    "program": "SW중심대학",
    "period": "2018.03.01-2018.12.31",
    "budget": "1,072,000,000",
    "completed": true,
    "items": [
      "4차 산업 혁명을 선도할 4C(Challenge, Creative, Collaborative, Convergent)형 지능정보기술 SW인재 양성"
    ],
    "url": "https://ezone.iitp.kr/common/co_0701/view?PMS_TSK_DGR_ID=2018-0-00191-004&cPage=2&PMS_SRCHCHOICE1=ALL&PMS_SRCHTEXT1=SW%EC%A4%91%EC%8B%AC%EB%8C%80%ED%95%99&PMS_SRCHCHOICE2=&PMS_SRCHTEXT2=&PMS_BEGDT=&PMS_ENDDT=&PMS_CHECK1=&PMS_CHECK2=&PMS_CHECK3="
  },
  {
    "id": 35,
    "title": "XAI (eXplainable AI) 기반 스마트 에너지 플랫폼 기술 개발",
    "organization": "한국전력공사 전력연구원",
    "program": "에너지 거점대학 클러스터 사업",
    "period": "2018.03.01-2019.01.31",
    "budget": "383,260,381",
    "completed": true,
    "items": [
      "스마트 에너지 데이터 분산 저장 및 분석",
      "스마트 에너지 보안 플랫폼 개발"
    ]
  },
  {
    "id": 36,
    "title": "실 사례 기반 경영 게임을 통한 기업 시뮬레이션 플랫폼 개발",
    "organization": "산업통산자원부",
    "program": "지식서비스산업핵심기술개발사업",
    "period": "2017.06.01-2018.05.31",
    "budget": "54,600,000",
    "completed": true,
    "items": [
      "R을 이용한 상권 데이터 분석 도구 개발",
      "TensorFlow를 이용한 상권 추천 딥러닝 모델 개발"
    ],
    "url": "https://ezone.iitp.kr/common/co_0701/view?PMS_TSK_DGR_ID=2018-0-00191-004&cPage=2&PMS_SRCHCHOICE1=ALL&PMS_SRCHTEXT1=SW%EC%A4%91%EC%8B%AC%EB%8C%80%ED%95%99&PMS_SRCHCHOICE2=&PMS_SRCHTEXT2=&PMS_BEGDT=&PMS_ENDDT=&PMS_CHECK1=&PMS_CHECK2=&PMS_CHECK3="
  },
  {
    "id": 37,
    "title": "빅데이터 시대에 대응한 교육정보.통계 정책과제 개발 연구",
    "organization": "교육부",
    "program": "정책연구개발사업(교육부)",
    "period": "2017.04.25-2017.09.25",
    "budget": "28,000,000",
    "completed": true,
    "items": [
      "교육정보·통계데이터의 빅데이터 활용을 위한 정책 연구 및 제언",
      "정책연구보고서 작성 및 제출"
    ],
    "url": "https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=TRKO201900000151"
  },
  {
    "id": 38,
    "title": "제조로봇용 실시간 지원 SW 플랫폼 기술 개발",
    "organization": "산업통산자원부",
    "program": "로봇산업핵심기술개발(R&D)",
    "period": "2016.07.01-2016.12.31",
    "budget": "181,000,000",
    "completed": true,
    "items": [
      "제조로봇의 다양한 데이터 수집 및 저장을 위한 빅데이터 플랫폼 기술 개발"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1415147487&pageCode=TH_PJT_PJT_DTL"
  },
  {
    "id": 39,
    "title": "실 사례 기반 경영 게임을 통한 기업 시뮬레이션 플랫폼 개발",
    "organization": "산업통산자원부",
    "program": "지식서비스산업핵심기술개발사업",
    "period": "2016.06.01-2017.05.31",
    "budget": "41,280,000",
    "completed": true,
    "items": [
      "Python을 이용한 데이터 수집 가공 및 크롤링 개발",
      "R을 이용한 경영 빅데이터 시각화 도구 개발"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1415146442&pageCode=TH_TOTAL_PJT_DTL"
  },
  {
    "id": 40,
    "title": "지능형 해부학적 조직 인식 빅데이터를 활용한 줄기세포 분화용 3 차원 조직공학 지지체 제조",
    "organization": "한국연구재단",
    "program": "지역신산업선도인력양성사업",
    "period": "2016.06.01-2017.12.31",
    "budget": "160,057,060",
    "completed": true,
    "items": [
      "줄기세포의 형광 염색 이미지 처리 및 분석",
      "바이오 빅데이터 분석을 위한 플랫폼 구축 및 연구",
      "머신 러닝을 이용한 바이오 데이터 학습"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1711040697"
  },
  {
    "id": 41,
    "title": "맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP",
    "organization": "한국연구재단",
    "program": "지역대학우수과학자지원사업",
    "period": "2013.05.01-2014.04.30",
    "budget": "44,518,432",
    "completed": true,
    "items": [
      "모바일에서 사진 데이터의 효과적인 관리와 브라우징을 위한 데이터 큐브 저장 연구"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1345213337"
  },
  {
    "id": 42,
    "title": "대용량 센서 스트림 데이터를 실시간으로 처리하는 개방형 센서 DBMS 개발",
    "organization": "미래창조과학부",
    "program": "SW컴퓨팅산업원천기술개발",
    "period": "2013.01.01-2013.11.30",
    "budget": "1,000,000,000",
    "completed": true,
    "items": [
      "실시간 스트림 데이터 처리를 위한 CEP 엔진 연구",
      "센서 데이터의 효율적인 처리 및 저장을 위한 센서 DBMS 개발"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1711007575&pageCode=TH_PJT_PJT_DTL"
  },
  {
    "id": 43,
    "title": "맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP",
    "organization": "한국연구재단",
    "program": "지역대학우수과학자지원사업",
    "period": "2012.05.01-2012.10.18",
    "budget": "44,980,619",
    "completed": true,
    "items": [
      "맵리듀스를 이용한 대규모 데이터 큐브의 분산 병렬 계산 연구"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1345166629"
  },
  {
    "id": 44,
    "title": "맵리듀스 분산 컴퓨팅 프레임워크를 이용한 다차원 텍스트 데이터 베이스 및 모바일 OLAP",
    "organization": "한국연구재단",
    "program": "지역대학우수과학자지원사업",
    "period": "2011.05.01-2012.04.30",
    "budget": "44,450,000",
    "completed": true,
    "items": [
      "모바일에서 사진 데이터의 효과적인 관리와 브라우징을 위한 데이터 큐브 저장 연구"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1345148057"
  },
  {
    "id": 45,
    "title": "분산 컴퓨팅 환경에서 프라이버시 보호 시계열 데이터 마이닝",
    "organization": "한국연구재단",
    "program": "중견연구자지원사업",
    "period": "2011.05.01-2012.04.30",
    "budget": "100,000,000",
    "completed": true,
    "items": [
      "분산 컴퓨팅 환경에서 데이터 마이닝 알고리즘 연구"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1345146450"
  },
  {
    "id": 46,
    "title": "프로비넌스에 기반한 위치 및 스트림 데이터의 신뢰도 평가",
    "organization": "한국연구재단",
    "program": "신진연구지원사업",
    "period": "2011.05.01-2012.04.30",
    "budget": "50.000.000",
    "completed": true,
    "items": [
      "스트림 데이터의 다차원 분석 연구"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1345151450"
  },
  {
    "id": 47,
    "title": "앱(App)창작터 사업",
    "organization": "중소기업청",
    "program": "앱(App)창작터 지정운영사업",
    "period": "2010.07.01-2010.12.31",
    "budget": "118,000,000",
    "completed": true,
    "items": [
      "앱 개발 교육 지원",
      "안드로이드 개발 및 실습 지원"
    ]
  },
  {
    "id": 48,
    "title": "유비쿼터스 센서 네트워크에서 스트림 데이터에 대한 효율적인 다 차원 분석 기술",
    "organization": "강원대학교",
    "program": "캠퍼스간 공동연구비",
    "period": "2007.12.20-2008.12.19",
    "budget": "16,250,000",
    "completed": true,
    "items": [
      "스트림 데이터의 효과적인 처리 및 저장 기법 연구",
      "데이터 스트림 관리 시스템 (DSMS) 엔진 관련 연구"
    ]
  },
  {
    "id": 49,
    "title": "정보 및 지식관리 연구",
    "organization": "한국과학기술원(KAIST)",
    "program": "우수연구센터지원사업",
    "period": "2006.03.01-2007.02.28",
    "budget": "50,032,480",
    "completed": true,
    "items": [
      "스프레드시트를 이용한 OLAP 도구 엔진 설계 및 구현",
      "OLAP 엔진의 Pivot 알고리즘 개발"
    ],
    "url": "https://www.ntis.go.kr/project/pjtInfo.do?pjtId=1350021874"
  }
];

export const getActiveProjects = (): Project[] => {
  return projects.filter((p) => !p.completed);
};

export const getCompletedProjects = (): Project[] => {
  return projects.filter((p) => p.completed);
};
