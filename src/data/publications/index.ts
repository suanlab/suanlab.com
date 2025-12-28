export type PublicationType = 'journal' | 'conference' | 'djournal' | 'dconference' | 'book' | 'patent' | 'report' | 'column';

export interface Publication {
  id: number;
  type: PublicationType;
  title: string;
  authors: string;
  venue: string;
  date: string;
  abstract?: string;
  keywords?: string;
  url?: string;
  badge?: string;
  impact?: string;
}

export const publicationTypes: { key: PublicationType | 'all'; label: string; count: number }[] = [
  { key: 'all', label: 'All', count: 271 },
  { key: 'journal', label: 'International Journal', count: 31 },
  { key: 'conference', label: 'International Conference', count: 40 },
  { key: 'djournal', label: 'Domestic Journal', count: 28 },
  { key: 'dconference', label: 'Domestic Conference', count: 145 },
  { key: 'book', label: 'Book', count: 2 },
  { key: 'patent', label: 'Patent', count: 11 },
  { key: 'report', label: 'Report', count: 3 },
  { key: 'column', label: 'Column', count: 9 },
];

export const publications: Publication[] = [
  // ============ 2025 International Journals ============
  {
    "id": 200,
    "type": "journal",
    "title": "Internet fraud transaction detection based on temporal-aware heterogeneous graph oversampling and attention fusion network",
    "authors": "Sizheng Wei and Suan Lee",
    "venue": "PLoS One",
    "date": "2025",
    "badge": "[SCIE]",
    "impact": "(IF: 2.9)",
    "keywords": "fraud detection; heterogeneous graph; attention network; deep learning"
  },
  {
    "id": 201,
    "type": "journal",
    "title": "Rapid recognition and localization of virtual assembly components in bridge 3D point clouds based on supervoxel clustering and transformer",
    "authors": "Chenglong Huang, Chi-Ho Lin and Suan Lee",
    "venue": "Int. J. Simul. Multidisci. Des. Optim.",
    "date": "2025",
    "keywords": "3D point clouds; supervoxel clustering; transformer; bridge assembly"
  },
  {
    "id": 202,
    "type": "journal",
    "title": "Deep learning pathways for automatic sign language processing",
    "authors": "Mukhiddin Toshpulatov, Wookey Lee, Jaesung Jun, and Suan Lee",
    "venue": "Pattern Recognition",
    "date": "2025",
    "badge": "[SCIE]",
    "impact": "(IF: 7.5, JCR: Q1)",
    "keywords": "sign language processing; deep learning; gesture recognition"
  },
  {
    "id": 203,
    "type": "journal",
    "title": "Multi-Patch Time Series Transformer for Robust Bearing Fault Detection with Varying Noise",
    "authors": "Sangkeun Ko and Suan Lee",
    "venue": "Applied Sciences",
    "date": "2025",
    "badge": "[SCIE]",
    "impact": "(IF: 2.5)",
    "keywords": "bearing fault detection; time series transformer; noise robustness; deep learning"
  },
  // ============ 2024 International Journals ============
  {
    "id": 204,
    "type": "journal",
    "title": "Leveraging Feature Extraction and Risk-Based Clustering for Advanced Fault Diagnosis in Equipment",
    "authors": "Hyeonbin Ji, Ingeun Hwang, Junghwon Kim, Suan Lee, and Wookey Lee",
    "venue": "PLoS ONE",
    "date": "December 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 2.9)",
    "keywords": "fault diagnosis; feature extraction; clustering; equipment monitoring"
  },
  {
    "id": 205,
    "type": "journal",
    "title": "Explainable Neural Tensor Factorization for Commercial Alley Revenues Prediction",
    "authors": "Minkyu Kim, Suan Lee, and Jinho Kim",
    "venue": "Electronics",
    "date": "August 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 2.6)",
    "keywords": "tensor factorization; explainable AI; revenue prediction; commercial district"
  },
  {
    "id": 206,
    "type": "journal",
    "title": "Melanoma classification using generative adversarial network and proximal policy optimization",
    "authors": "Xiangui Ju, Chi-Ho Lin, Suan Lee, and Sizheng Wei",
    "venue": "Photochemistry and Photobiology",
    "date": "2024",
    "badge": "[SCIE]",
    "impact": "(IF: 2.6)",
    "keywords": "melanoma classification; GAN; reinforcement learning; medical imaging"
  },
  {
    "id": 207,
    "type": "journal",
    "title": "DDC3N: Doppler-Driven Convolutional 3D Network for Human Action Recognition",
    "authors": "Mukhiddin Toshpulatov, Wookey Lee, Suan Lee, et al.",
    "venue": "IEEE Access",
    "date": "2024",
    "badge": "[SCIE]",
    "impact": "(IF: 3.4)",
    "keywords": "human action recognition; Doppler; 3D CNN; deep learning"
  },
  {
    "id": 208,
    "type": "journal",
    "title": "Advancing the Robotic Vision Revolution: Development and Evaluation of a Bionic Binocular System for Enhanced Robotic Vision",
    "authors": "Hongxin Zhang and Suan Lee",
    "venue": "Biomimetics",
    "date": "June 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 3.4)",
    "keywords": "robotic vision; bionic eye; binocular system; computer vision"
  },
  {
    "id": 209,
    "type": "journal",
    "title": "Analyzing social media reactions to the FTX crisis: Unraveling the spillover effect on crypto markets",
    "authors": "Chunsik Lee, Suan Lee, et al.",
    "venue": "Journal of Contingencies and Crisis Management",
    "date": "June 2024",
    "badge": "[SSCI]",
    "impact": "(IF: 3.42)",
    "keywords": "social media analysis; cryptocurrency; FTX crisis; sentiment analysis"
  },
  {
    "id": 210,
    "type": "journal",
    "title": "A Multi-Input Convolutional Neural Network Model for Electric Motor Mechanical Fault Classification Using Multiple Image Transformation and Merging Methods",
    "authors": "Insu Bae and Suan Lee",
    "venue": "Machines",
    "date": "February 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 2.6)",
    "keywords": "fault classification; CNN; electric motor; image transformation"
  },
  {
    "id": 211,
    "type": "journal",
    "title": "Financial Anti-Fraud Based on Dual-Channel Graph Attention Network",
    "authors": "Sizheng Wei and Suan Lee",
    "venue": "Journal of Theoretical and Applied Electronic Commerce Research",
    "date": "March 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 5.6, JCR: Q1)",
    "keywords": "financial fraud detection; graph attention network; dual-channel; deep learning"
  },
  {
    "id": 212,
    "type": "journal",
    "title": "A deep learning model for predicting the number of stores and average sales in commercial district",
    "authors": "Suan Lee, Sangkeun Ko, Arousha Haghighian Roudsari, and Wookey Lee",
    "venue": "Data & Knowledge Engineering",
    "date": "March 2024",
    "badge": "[SCIE]",
    "impact": "(IF: 2.5)",
    "keywords": "deep learning; commercial district; sales prediction; store prediction"
  },
  // ============ 2023 International Journals (New) ============
  {
    "id": 213,
    "type": "journal",
    "title": "Machine Learning Model for Leak Detection Using Water Pipeline Vibration Sensor",
    "authors": "Suan Lee and Byeonghak Kim",
    "venue": "Sensors",
    "date": "October 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 3.9)",
    "keywords": "leak detection; machine learning; vibration sensor; water pipeline"
  },
  // ============ 2025 Domestic Journals (KCI) ============
  {
    "id": 214,
    "type": "djournal",
    "title": "LLM as a Search: 사용자 의도 분석을 통한 지능형 법률 문서 검색",
    "authors": "김재성, 고수윤, 정진근, 이수안",
    "venue": "정보과학회 컴퓨팅의 실제 논문지, 한국정보과학회",
    "date": "2025",
    "badge": "[KCI]",
    "keywords": "LLM; 법률 문서 검색; 사용자 의도 분석; 지능형 검색"
  },
  {
    "id": 215,
    "type": "djournal",
    "title": "강화학습을 활용한 소규모 언어 모델 기반 Text-to-SQL 성능 향상",
    "authors": "김동후, 이수안",
    "venue": "데이터베이스연구, 한국정보과학회",
    "date": "2025",
    "badge": "[KCI]",
    "keywords": "강화학습; Text-to-SQL; 소규모 언어 모델; sLLM"
  },
  {
    "id": 216,
    "type": "djournal",
    "title": "후반 레이어 Pruning과 다단계 Fine-tuning을 통한 한국어 언어 모델의 효율적 경량화",
    "authors": "김재성, 이수안",
    "venue": "정보과학회논문지, 한국정보과학회",
    "date": "2025",
    "badge": "[KCI]",
    "keywords": "모델 경량화; Pruning; Fine-tuning; 한국어 언어 모델"
  },
  // ============ 2024 Domestic Journals (KCI) ============
  {
    "id": 217,
    "type": "djournal",
    "title": "Step-by-Step과 Self-Debug 방식을 통한 Text-to-SQL 모델의 성능 향상",
    "authors": "김재성, 이수안",
    "venue": "데이터베이스연구, 한국정보과학회",
    "date": "2024",
    "badge": "[KCI]",
    "keywords": "Text-to-SQL; Self-Debug; 단계별 추론; 언어 모델"
  },
  {
    "id": 218,
    "type": "djournal",
    "title": "산업용 설비 소리의 고장 분류를 위한 이미지 변환 및 데이터 증강 모델",
    "authors": "고상근, 이수안",
    "venue": "정보과학회 컴퓨팅의 실제, 한국정보과학회",
    "date": "2024",
    "badge": "[KCI]",
    "keywords": "고장 분류; 이미지 변환; 데이터 증강; 산업용 설비"
  },
  // ============ 2023 Domestic Journals (KCI) ============
  {
    "id": 219,
    "type": "djournal",
    "title": "통계, 머신러닝, 딥러닝 기반 시계열 모델을 이용한 원자재 가격 예측",
    "authors": "서경식, 고상근, 이수안",
    "venue": "데이터베이스연구, 한국정보과학회",
    "date": "2023",
    "badge": "[KCI]",
    "keywords": "시계열 예측; 원자재 가격; 머신러닝; 딥러닝"
  },
  {
    "id": 220,
    "type": "djournal",
    "title": "타순별 통계와 기계학습의 회귀 모델을 활용한 한국 프로야구에서 팀 득점력 개선",
    "authors": "신동윤, 이수안, 김진호",
    "venue": "한국융합과학회지, 한국융합과학회",
    "date": "November 2023",
    "badge": "[KCI]",
    "keywords": "프로야구; 타순; 기계학습; 회귀 모델; 득점력"
  },
  // ============ 2024 International Conferences ============
  {
    "id": 221,
    "type": "conference",
    "title": "Course Recommendation System for Company Job Placement Using Collaborative Filtering and Hybrid Model",
    "authors": "Jaeheon Park, Suan Lee, Woncheol Lee, and Jinho Kim",
    "venue": "13th International Conference on Data Science, Technology and Applications (DATA 2024), SciTePress",
    "date": "2024",
    "keywords": "course recommendation; collaborative filtering; hybrid model; job placement"
  },
  {
    "id": 222,
    "type": "conference",
    "title": "Deep Learning Model for Traffic Accident Prediction Using Multiple Feature Interactions",
    "authors": "Namhyeon Kim, Minkyu Kim, and Suan Lee",
    "venue": "2024 IEEE International Conference on Big Data and Smart Computing (BigComp)",
    "date": "February 2024",
    "keywords": "traffic accident prediction; deep learning; feature interactions"
  },
  {
    "id": 223,
    "type": "conference",
    "title": "Noise-Robust Sleep States Classification Model Using Sound Feature Extraction and Conversion",
    "authors": "Sangkeun Ko, Seongho Min, Ye Shin Choi, Woo-Je Kim, and Suan Lee",
    "venue": "2024 IEEE International Conference on Big Data and Smart Computing (BigComp)",
    "date": "February 2024",
    "keywords": "sleep classification; sound feature extraction; noise-robust; deep learning"
  },
  {
    "id": 224,
    "type": "conference",
    "title": "Human Action Recognition Utilizing Doppler-Enhanced Convolutional 3D Networks",
    "authors": "Mukhiddin Toshpulatov, Wookey Lee, Chingiz Tursunbaev, and Suan Lee",
    "venue": "2024 IEEE International Conference on Big Data and Smart Computing (BigComp)",
    "date": "February 2024",
    "keywords": "human action recognition; Doppler; 3D CNN; deep learning"
  },
  // ============ 2023 International Conferences ============
  {
    "id": 225,
    "type": "conference",
    "title": "Diffusion-C: Unveiling the Generative Challenges of Diffusion Models through Corrupted Data",
    "authors": "Keywoong Bae, Suan Lee, and Wookey Lee",
    "venue": "NeurIPS 2023 Workshop on Diffusion Models",
    "date": "December 2023",
    "keywords": "diffusion models; generative AI; corrupted data; image generation"
  },
  // ============ 2025 Domestic Conferences ============
  {
    "id": 226,
    "type": "dconference",
    "title": "이미지 인코딩 기반 특징 결합을 활용한 시계열 이상 탐지 모델",
    "authors": "최하정, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "시계열 이상 탐지; 이미지 인코딩; 특징 결합; 딥러닝"
  },
  {
    "id": 227,
    "type": "dconference",
    "title": "잔여 수명 예측을 위한 딥러닝 기반 모델 비교 분석",
    "authors": "고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "잔여 수명 예측; 딥러닝; 모델 비교; RUL"
  },
  {
    "id": 228,
    "type": "dconference",
    "title": "멀티뷰 Video-LLM 기반 기업 IR 발표 영상 자동 분석 및 평가 시스템",
    "authors": "김혜진, 임채환, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "Video-LLM; IR 발표; 영상 분석; 멀티뷰"
  },
  {
    "id": 229,
    "type": "dconference",
    "title": "물리 일관성과 불확실성을 통합한 RGB 영상 기반 피부 생체 지표 복원 모델",
    "authors": "배인수, 김병학, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "피부 생체 지표; RGB 영상; 물리 일관성; 불확실성"
  },
  {
    "id": 230,
    "type": "dconference",
    "title": "Wi-Fi CSI 기반 시간 및 주파수 도메인 특성을 이용하여 강건한 비접촉식 호흡률 예측 모델",
    "authors": "김남현, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "Wi-Fi CSI; 호흡률 예측; 비접촉식; 시간-주파수 도메인"
  },
  {
    "id": 231,
    "type": "dconference",
    "title": "시각적 인지 보상 기반 위성 영상 이미지에 대한 강화학습",
    "authors": "임채환, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "위성 영상; 강화학습; 시각적 인지 보상; 이미지 분석"
  },
  {
    "id": 232,
    "type": "dconference",
    "title": "3D 얼굴 골격 분석과 멀티모달 생성을 활용한 개인 맞춤형 화장법 추천 에이전트",
    "authors": "이겸수, 장준혁, 정예림, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월",
    "keywords": "3D 얼굴 분석; 멀티모달; 화장법 추천; AI 에이전트"
  },
  {
    "id": 233,
    "type": "dconference",
    "title": "한국어 완곡 코팅을 통한 LLM 안전성 우회 분석",
    "authors": "길상현, 김재성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2025년 12월",
    "keywords": "LLM 안전성; 완곡 코팅; 우회 분석; 한국어"
  },
  // ============ 2025년 11월 추계학술대회 ============
  {
    "id": 234,
    "type": "dconference",
    "title": "고빈도 구문의 의미적 특성에 대한 단일 토큰으로의 전이 탐구: 트랜스포머 아키텍처를 중심으로",
    "authors": "김수웅, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 235,
    "type": "dconference",
    "title": "생성형 대규모 언어 모델을 활용한 계약서 요약 및 독소 조항 탐지",
    "authors": "서명관, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 236,
    "type": "dconference",
    "title": "설명 가능한 AI 기반의 유해 문장 판별 모델 구현 및 평가",
    "authors": "이찬우, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 237,
    "type": "dconference",
    "title": "설명가능 AI 기반의 인과추론을 이용한 당뇨병 환자 재입원 방지를 위한 개인 맞춤형 약물 처방",
    "authors": "이가연, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 238,
    "type": "dconference",
    "title": "제로샷 시계열 기초 모델의 주가 예측 성능 비교",
    "authors": "김동민, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 239,
    "type": "dconference",
    "title": "한국 세법 질의응답을 위한 VectorDB-GraphDB 하이브리드 검색 기반 RAG 시스템",
    "authors": "최윤성, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 240,
    "type": "dconference",
    "title": "AI 기반 스마트 냉장고 시스템: 자동 식재료 관리 및 맞춤형 레시피 추천",
    "authors": "나영민, 김진만, 김현기, 윤임주, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  {
    "id": 241,
    "type": "dconference",
    "title": "하루버디: 글로벌 한국을 위한 디지털 스케폴딩",
    "authors": "Chukwuka Chinechebem Yvette, 이수안",
    "venue": "추계학술대회논문집, 대한전자공학회",
    "date": "2025년 11월"
  },
  // ============ 2025년 07월 한국컴퓨터종합학술대회 ============
  {
    "id": 242,
    "type": "dconference",
    "title": "시간 경과에 따른 정적 악성코드 탐지 모델의 성능 저하 분석: OPCode 시퀀스와 트랜스포머 모델 기반 연구",
    "authors": "길상현, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 243,
    "type": "dconference",
    "title": "절연체 종류별 부분방전 분류를 위한 UFH 및 HFCT 센서 기반 딥러닝 시계열 모델",
    "authors": "최하정, 배인수, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 244,
    "type": "dconference",
    "title": "전력 수요 예측을 위한 시계열 딥러닝 모델 간의 장단기 예측 및 설명가능성 비교",
    "authors": "박주성, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 245,
    "type": "dconference",
    "title": "AI Agent를 이용한 개인 맞춤형 취업 매칭 및 추천 시스템",
    "authors": "장영진, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 246,
    "type": "dconference",
    "title": "강화학습을 통한 소규모 언어 모델 기반의 효율적인 Text-to-SQL 모델",
    "authors": "김동후, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 247,
    "type": "dconference",
    "title": "대규모 언어 모델을 이용한 진로 성숙도 검사 서술형 응답 자동 분석 및 피드백 생성",
    "authors": "김병학, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월",
    "badge": "[우수상]"
  },
  {
    "id": 248,
    "type": "dconference",
    "title": "모달리티 특성을 고려한 데이터 증강 기반 다중 모달 딥러닝 흉부 질환 분류",
    "authors": "정예림, 고상근, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 249,
    "type": "dconference",
    "title": "MCP를 이용한 LLM 기반 적응형 크롤링과 문서 분석을 위한 자동화 워크플로우 아키텍처 설계",
    "authors": "길상현, 이동혁, 유재익, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 250,
    "type": "dconference",
    "title": "대학생 역량 강화를 위한 학습 보조용 AI 챗봇 시스템",
    "authors": "권용, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 251,
    "type": "dconference",
    "title": "Grad-CAM을 적용한 객체 탐지 및 분할 모델 기반의 설명가능한 와이어 로프 결함 탐지",
    "authors": "서지훈, 조홍석, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 252,
    "type": "dconference",
    "title": "K-GovExam: 한국 공무원 시험 기반 LLM 평가용 데이터셋 구축 및 추론 언어 모델 분석",
    "authors": "김재성, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월",
    "badge": "[최우수상]"
  },
  {
    "id": 253,
    "type": "dconference",
    "title": "KO-SmallThinker: Reasoning 기반 소형 언어 모델을 활용한 초거대 언어 모델의 성능 한계 극복",
    "authors": "김재성, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 254,
    "type": "dconference",
    "title": "Minimal Tuning, Maximum Gains: 선택적 레이어 파인튜닝 기반 효율적 학습 전략",
    "authors": "김재성, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 255,
    "type": "dconference",
    "title": "Text-to-Image 생성 모델의 스타일 전이 성능 비교 연구",
    "authors": "백상렬, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 256,
    "type": "dconference",
    "title": "FLUX 및 LoRA 기반 텍스트-이미지 생성 모델을 활용한 웹툰 제작 자동화 모델 설계 및 구현",
    "authors": "이수현, 백상렬, 김정원, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 257,
    "type": "dconference",
    "title": "LoRA와 ControlNet Line-Art를 활용한 화풍 보존형 애니메이션 프레임 보간",
    "authors": "이하은, 김주일, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 258,
    "type": "dconference",
    "title": "텍스트에서 생성된 단일 이미지의 Gaussian Splatting을 이용한 3차원 객체 생성",
    "authors": "우강석, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 259,
    "type": "dconference",
    "title": "Wi-Fi CSI로부터 웹캠 영상 생성을 위한 1D-CNN과 트랜스포머를 결합한 MoPoE-VAE 모델",
    "authors": "진동민, 김재한, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 260,
    "type": "dconference",
    "title": "Wi-Fi MIMO CSI의 진폭과 위상 정보를 활용한 스펙트로그램 기반 실내 인원 수 추정 딥러닝 모델",
    "authors": "김남현, 진동민, 김재한, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  {
    "id": 261,
    "type": "dconference",
    "title": "SHAP 기반 설명 가능성을 활용한 모터 전류 시계열의 고장 분류 딥러닝 모델",
    "authors": "고상근, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2025년 07월"
  },
  // ============ 2024년 12월 한국소프트웨어종합학술대회 ============
  {
    "id": 262,
    "type": "dconference",
    "title": "효과적인 소리 특징 추출을 이용한 낙상 탐지 모델",
    "authors": "민성호, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 263,
    "type": "dconference",
    "title": "Depth-Up Scaling을 활용한 언어 모델 증강과 소규모 데이터 학습 성능 비교",
    "authors": "김재성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 264,
    "type": "dconference",
    "title": "시공간 그래프 신경망을 이용한 이동수단의 수요 예측 모델",
    "authors": "김지나, 김병학, 정은지, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 265,
    "type": "dconference",
    "title": "입면 정사 영상 데이터를 이용한 세그멘테이션 기반 아스팔트 균열탐지 모델",
    "authors": "김병학, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 266,
    "type": "dconference",
    "title": "EmoNeXt를 이용한 한국인 표정인식 모델",
    "authors": "김병학, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 267,
    "type": "dconference",
    "title": "서울시 상권 매출액 예측 모델: 특징 중요도 및 예측 기여도 분석",
    "authors": "정채윤, 고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 268,
    "type": "dconference",
    "title": "YOLOv10과 YOLOv11를 이용한 인쇄 회로 기판(PCB) 결함 탐지 및 분류",
    "authors": "박소연, 조홍석, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 269,
    "type": "dconference",
    "title": "경쟁 게임의 밸런스를 위한 머신 러닝 기반 승패 예측 및 특징 상호작용 분석",
    "authors": "이동혁, 길상현, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 270,
    "type": "dconference",
    "title": "NeRF와 Oculus Quest를 활용한 몰입형 3D 경험 구현 및 분석",
    "authors": "김주일, 이하은, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 271,
    "type": "dconference",
    "title": "텍스트와 이미지 프롬프트를 활용한 안경 가상 피팅 모델",
    "authors": "김정원, 백상렬, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 272,
    "type": "dconference",
    "title": "텍스트와 인물 이미지 기반 다각도의 고해상도 헤어 스타일 생성 및 변환",
    "authors": "이수현, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  {
    "id": 273,
    "type": "dconference",
    "title": "Auto Masking 기법을 이용한 레이어링 VTON",
    "authors": "최하정, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2024년 12월"
  },
  // ============ 2024년 06월 한국컴퓨터종합학술대회 ============
  {
    "id": 274,
    "type": "dconference",
    "title": "딥러닝 기반 전력 설비 이상 분류 시계열 모델",
    "authors": "고상근, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 275,
    "type": "dconference",
    "title": "Image Captioning과 Visual Question Answering를 이용한 위성 이미지 기반 도시 변화 탐지 모델",
    "authors": "임채환, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 276,
    "type": "dconference",
    "title": "Jetson Orin Nano 기반의 스테레오 카메라를 이용한 깊이 추정 및 객체 탐지",
    "authors": "임채환, 진동민, 장홍흠, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 277,
    "type": "dconference",
    "title": "사전 학습된 대규모 언어 모델을 이용한 주식 예측",
    "authors": "서세일, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 278,
    "type": "dconference",
    "title": "Mamba, Transformer, LSTM을 이용한 주식 예측 모델 분석",
    "authors": "서세일, 백상렬, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 279,
    "type": "dconference",
    "title": "YOLOv9 기반의 인쇄 회로 기판(PCB) 결함 탐지 및 분류",
    "authors": "조홍석, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 280,
    "type": "dconference",
    "title": "NVIDIA Jetson AGX Orin 기반의 YOLOv8, YOLOv9 모델을 이용한 화재 탐지",
    "authors": "조홍석, 권희준, 정해영, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 281,
    "type": "dconference",
    "title": "WiFi CSI 기반의 실내 재실자 수 예측 모델",
    "authors": "배인수, 진동민, 김재한, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 282,
    "type": "dconference",
    "title": "구름 요소 베어링의 진동 및 모터 전류 데이터의 고장 분류 모델",
    "authors": "최하정, 배인수, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월",
    "badge": "[장려상]"
  },
  {
    "id": 283,
    "type": "dconference",
    "title": "진동 센서 기반 상수관 누수 감지 및 분류 모델",
    "authors": "이수현, 고상근, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 284,
    "type": "dconference",
    "title": "실내외 CCTV를 이용한 군중 인원 수 계수 모델 비교",
    "authors": "김지나, 정은지, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 285,
    "type": "dconference",
    "title": "현실 공간 소리 탐지 및 분류 모델",
    "authors": "민성호, 김정원, 김재한, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 286,
    "type": "dconference",
    "title": "파인튜닝, RAG, 프롬프트 기반 언러닝 방법론의 효율성 비교 분석",
    "authors": "김재성, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 287,
    "type": "dconference",
    "title": "법률 QA데이터셋을 이용한 거대 언어 모델 학습",
    "authors": "김재성, 김강준, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 288,
    "type": "dconference",
    "title": "생성형 AI 기반 사용자 맞춤형 동화책 생성 및 구연 서비스",
    "authors": "안현주, 배지호, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  {
    "id": 289,
    "type": "dconference",
    "title": "Stable Diffusion 기반의 3D 캐릭터 생성에 대한 활용 및 비교 분석",
    "authors": "이하은, 김주일, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2024년 06월"
  },
  // ============ 2023년 12월 한국소프트웨어종합학술대회 ============
  {
    "id": 290,
    "type": "dconference",
    "title": "산업용 설비 소리의 이미지 변환을 이용한 딥러닝 기반의 고장 진단 및 분류 모델",
    "authors": "고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월",
    "badge": "[우수발표논문상]"
  },
  {
    "id": 291,
    "type": "dconference",
    "title": "딥러닝 모델을 이용한 시각 장애인 생활 보조",
    "authors": "최동현, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 292,
    "type": "dconference",
    "title": "ChatGPT와 DALL-E 3 기반의 AI 생성 모델을 활용한 AR 공간 내의 NPC 생성 및 활용",
    "authors": "임채환, 이수형, 이동근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 293,
    "type": "dconference",
    "title": "YOLOv8과 YOLO-NAS를 이용한 인쇄 회로 기판(PCB) 결함 탐지 및 분류",
    "authors": "조홍석, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 294,
    "type": "dconference",
    "title": "시계열의 이미지 인코딩을 이용한 전동기 기계시설물 고장 분류 모델",
    "authors": "배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월",
    "badge": "[우수상]"
  },
  {
    "id": 295,
    "type": "dconference",
    "title": "지도 위치 및 AR 기반 산책 애플리케이션",
    "authors": "현지원, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월",
    "badge": "[장려상]"
  },
  {
    "id": 296,
    "type": "dconference",
    "title": "AutoML을 이용한 GPS 궤적에 따른 이동 수단 분류 모델",
    "authors": "정은지, 배인수, 민성호, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 297,
    "type": "dconference",
    "title": "재난 분류를 위한 트윗 데이터 분석 및 언어 모델",
    "authors": "황민정, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 298,
    "type": "dconference",
    "title": "서로 다른 언어 모델의 상징적 지식 증류를 이용한 경량화된 감정 분석 모델",
    "authors": "김재성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 299,
    "type": "dconference",
    "title": "한국어 혐오 발언 댓글 분류를 위한 언어 모델별 비교",
    "authors": "어식, 배인수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 302,
    "type": "dconference",
    "title": "머신러닝을 이용한 투수와 타자 간의 출루율 예측 및 분석",
    "authors": "이정원, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 303,
    "type": "dconference",
    "title": "수면 소리 특징과 컨볼루션 신경망을 이용한 상태 분류 모델",
    "authors": "민성호, 최예신, 김우제, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 304,
    "type": "dconference",
    "title": "통계적, 머신러닝, 딥러닝 모델을 이용한 주가 시계열 예측 및 비교",
    "authors": "김지나, 고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 305,
    "type": "dconference",
    "title": "Neural Collaborative Filtering 기반의 기업 취업을 위한 과목 추천 시스템",
    "authors": "박재헌, 이원철, 은동진, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  {
    "id": 306,
    "type": "dconference",
    "title": "Graph Convolutional Matrix Completion을 이용한 수강 강의 추천 시스템",
    "authors": "추화랑, 이원철, 은동진, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2023년 12월"
  },
  // ============ 2022년 국내학술대회 ============
  {
    "id": 307,
    "type": "dconference",
    "title": "한국어 논문 요약을 위한 KoBART와 KoBERT 모델 비교",
    "authors": "전제성, 이수안",
    "venue": "제34회 한글 및 한국어 정보처리 학술대회, 한국정보과학회",
    "date": "2022년 10월"
  },
  {
    "id": 308,
    "type": "dconference",
    "title": "CNN-LSTM 기반 시계열 데이터의 이상치 분류를 위한 이미지 인코딩 방법 비교 분석",
    "authors": "신재광, 안동주, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월"
  },
  {
    "id": 309,
    "type": "dconference",
    "title": "3D 가상 피팅 딥러닝 모델과 VR 연동을 위한 웹 서비스 개발",
    "authors": "전제성, 진동민, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월"
  },
  {
    "id": 310,
    "type": "dconference",
    "title": "엣지 TPU 기반의 임베디드 기기별 경량화된 이미지 분류 모델 비교",
    "authors": "최동현, 김남현, 고상근, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월"
  },
  {
    "id": 311,
    "type": "dconference",
    "title": "임베디드 장비 환경에서 경량화된 객체 탐지 딥러닝 모델 비교",
    "authors": "김지나, 정은지, 전제성, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월"
  },
  {
    "id": 312,
    "type": "dconference",
    "title": "딥러닝 모델 기반 가상 피팅 웹 서비스 개발",
    "authors": "고상근, 장현수, 민정호, 이다혁, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월",
    "badge": "[장려상]"
  },
  {
    "id": 313,
    "type": "dconference",
    "title": "동질적 특징을 이용한 견고한 멀티모달 분류 모델",
    "authors": "배기웅, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월",
    "badge": "[우수상]"
  },
  {
    "id": 314,
    "type": "dconference",
    "title": "AIoT 환경에서 마커와 사람 인식을 이용한 식별된 사람 출입관리 시스템 개발",
    "authors": "임채환, 이수형, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월"
  },
  // ============ 2021년 국내학술대회 ============
  {
    "id": 315,
    "type": "dconference",
    "title": "윤리적 소비를 위한 지속가능한 거래: 중고거래 금지 품목 탐지를 중심으로",
    "authors": "이충성, 전서영, 이우기, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월"
  },
  {
    "id": 316,
    "type": "dconference",
    "title": "택시 데이터 분석을 통한 수요 예측과 승차대 위치 개선",
    "authors": "김예원, 김유정, 김주아, 이유정, 이우기, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월",
    "badge": "[장려상]"
  },
  {
    "id": 317,
    "type": "dconference",
    "title": "NVIDIA Jetson AGX Xavier 환경에서 경량화된 실시간 안전모 탐지 모델 개발",
    "authors": "조홍석, 이수형, 장현수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월"
  },
  {
    "id": 318,
    "type": "dconference",
    "title": "장애인 안전 이동을 위한 이미지 캡션 기반 시각 읽기 딥러닝 모델",
    "authors": "이정엽, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2021년 07월"
  },
  {
    "id": 319,
    "type": "dconference",
    "title": "모바일 자율주행 로봇을 위한 저비용 차선 인식 및 제어 알고리즘",
    "authors": "장현수, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2021년 07월",
    "badge": "[장려상]"
  },
  // ============ 2020년 국내학술대회 ============
  {
    "id": 320,
    "type": "dconference",
    "title": "Face detection을 위한 RetinaFace 기반 대용량 학습 데이터 자동 레이블링 프레임워크",
    "authors": "이효준, 이수안, 이우기",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2020년 12월"
  },
  {
    "id": 321,
    "type": "dconference",
    "title": "Generative Adversarial Networks 기반 개인정보 보호를 위한 얼굴 비식별화",
    "authors": "조시헌, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월"
  },
  {
    "id": 322,
    "type": "dconference",
    "title": "벡터화 경로 데이터에 노이즈 추가를 통한 프라이버시 보호 방안: 코로나 바이러스를 중심으로",
    "authors": "박수경, 라시드, 이철기, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월"
  },
  {
    "id": 323,
    "type": "dconference",
    "title": "적대적 공격을 이용한 객체 탐지 모델의 취약성 검증",
    "authors": "이효준, 이철기, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월"
  },
  {
    "id": 324,
    "type": "dconference",
    "title": "선행기술조사를 위한 딥러닝 언어 모델 기반 특허 문서 분류",
    "authors": "강명철, 이철기, 이수안, 이우기",
    "venue": "2020 춘계학술대회, 한국지식재산교육연구학회",
    "date": "2020년 06월",
    "badge": "[우수논문]"
  },
  // ============ 2019년 이전 국내학술대회 ============
  {
    "id": 325,
    "type": "dconference",
    "title": "로봇 판단지능을 위해 인간 작업 영상을 활용한 빅데이터 시스템 설계 및 구축",
    "authors": "장종원, 전호빈, 이수안, 김진호, 박홍성, 김미숙, 유수정, 지상훈",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2019년 12월"
  },
  {
    "id": 326,
    "type": "dconference",
    "title": "효율적인 회귀분석을 위한 Shared Wide & Deep 모델",
    "authors": "김민규, 이수안, 김진호",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2019년 07월"
  },
  {
    "id": 327,
    "type": "dconference",
    "title": "빅데이터 기술을 이용한 자유학기제 운영 실태 보고서 문서 분석",
    "authors": "김민규, 이수안, 김진호, 신혜숙",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2018년 12월"
  },
  {
    "id": 328,
    "type": "dconference",
    "title": "빅데이터 기반의 통합적 트래픽 분석 플랫폼을 위한 저장구조 설계",
    "authors": "장종원, 김희상, 김민규, 이수안, 김진호",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2018년 12월"
  },
  {
    "id": 329,
    "type": "dconference",
    "title": "초·중등 과학 교과서 용어주석 말뭉치 구축 및 검색을 위한 시스템 개발",
    "authors": "유영석, 이수안, 김진호, 윤은정, 박윤배",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2017년 06월"
  },
  {
    "id": 330,
    "type": "dconference",
    "title": "얼굴 인식과 SNS 정보를 이용한 모바일 기기에서 사진 자동 분류 및 검색",
    "authors": "최재용, 이수안, 김진호",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2012년 06월"
  },
  {
    "id": 331,
    "type": "dconference",
    "title": "모바일 단말기에서 효과적인 사진 관리를 위한 계층적 사진 탐색기",
    "authors": "이수안, 원지섭, 최재용, 김진호",
    "venue": "가을 학술발표논문집, 한국정보과학회",
    "date": "2011년 11월"
  },
  {
    "id": 332,
    "type": "dconference",
    "title": "SNMP MIB 기반 네트워크 관리 시스템의 다차원 분석을 위한 데이터 웨어하우스 설계",
    "authors": "이수안, 최미정, 김진호",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2010년 06월"
  },
  {
    "id": 333,
    "type": "dconference",
    "title": "클라우드 컴퓨팅 환경에서 데이터 웨어하우스 연구",
    "authors": "이수안, 김진호, 문양세",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2010년 06월"
  },
  {
    "id": 334,
    "type": "dconference",
    "title": "맵리듀스를 이용한 빙산 큐브 병렬 계산",
    "authors": "이수안, 김진호, 문양세, 노웅기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2010년 06월"
  },
  {
    "id": 335,
    "type": "dconference",
    "title": "유비쿼터스 센서 네트워크를 이용한 건물 화재 모니터링 시스템의 다차원 데이터베이스 설계",
    "authors": "김진호, 이수안, 민두환, 김석훈, 남시병",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2009년 06월"
  },
  // ============ Existing Publications ============
  {
    "id": 1,
    "type": "journal",
    "title": "A Hybrid Image Segmentation Method for Accurate Measurement of Urban Environments",
    "authors": "Hyungjoon Kim, Jae Ho Lee, and Suan Lee",
    "venue": "Electronics",
    "date": "April 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 2.690, JCR: Q2)",
    "abstract": "In the field of urban environment analysis research, image segmentation technology that groups important objects in the urban landscape image in pixel units has been the subject of increased attention. However, since a dataset consisting of a huge amount of image and label pairs is required to utilize this technology, in most cases, a model trained with a dataset having similar characteristics is used for analysis, and as a result, the quality of segmentation is poor. To overcome this limitation, we propose a hybrid model to leverage the strengths of each model in predicting specific classes. In particular, we first introduce a pre-processing operation to reduce the differences between the collected urban dataset and public dataset. Subsequently, we train several segmentation models with a pre-processed dataset then, based on the weight rule, the segmentation results are fused to create one segmentation map. To evaluate our proposal, we collected Google Street View images that do not have any labels and trained a model using the cityscapes dataset which contains foregrounds similar to the collected images. We quantitatively assessed its performance using the cityscapes dataset with ground truths and qualitatively evaluated the results of GSV data segmentation through user studies. Our approach outperformed existing methods and demonstrated the potential for accurate and efficient urban environment analysis using computer vision technology.",
    "keywords": "urban environment analysis; streetscapes; image segmentation; hybrid model; deep learning",
    "url": "https://www.mdpi.com/2079-9292/12/8/1845"
  },
  {
    "id": 2,
    "type": "journal",
    "title": "Dynamic Characteristics Prediction Model for Diesel Engine Valve Train Design Parameters Based on Deep Learning.",
    "authors": "Wookey Lee, Tae-Yun Jung, and Suan Lee",
    "venue": "Electronics",
    "date": "April 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 2.690, JCR: Q2)",
    "abstract": "This paper presents a comprehensive study on the utilization of machine learning and deep learning techniques to predict the dynamic characteristics of design parameters, exemplified by a diesel engine valve train. The research aims to address the challenging and time-consuming analysis required to optimize the performance and durability of valve train components, which are influenced by numerous factors. To this end, dynamic analyses data have been collected for diesel engine specifications and used to construct a regression prediction model using a gradient boosting regressor tree (GBRT), a deep neural network (DNN), a one-dimensional convolution neural network (1D-CNN), and long short-term memory (LSTM). The prediction model was utilized to estimate the force and valve seating velocity values of the valve train system. The dynamic characteristics of the case were evaluated by comparing the actual and predicted values. The results showed that the GBRT model had an R2 value of 0.90 for the valve train force and 0.97 for the valve seating velocity, while the 1D-CNN model had an R2 value of 0.89 for the valve train force and 0.98 for the valve seating velocity. The results of this study have important implications for advancing the design and development of efficient and reliable diesel engines.",
    "keywords": "diesel engine; valve train dynamics; deep learning; GBRT; DNN; LSTM; 1D-CNN",
    "url": "https://www.mdpi.com/2079-9292/12/3/698"
  },
  {
    "id": 3,
    "type": "journal",
    "title": "Talking human face generation: A survey.",
    "authors": "Mukhiddin Toshpulatov, Wookey Lee, and Suan Lee",
    "venue": "Expert Systems with Applications",
    "date": "June 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 8.665, JCR: Q1)",
    "abstract": "Talking human face generation aims at synthesizing a natural human face that talks in correspondence to the given text or audio series. Implementing the recently developed Deep Learning (DL) methods such as Convolutional Neural Networks (CNN), Generative Adversarial Networks (GAN)s, Neural Rendering Fields (NeRF) for data generation, and talking human face generation has attracted significant research interest from academia and industry. They have been explored and exploited recently and have been used to address several problems in image processing and computer vision. Notwithstanding notable advancements, implementing them to real-world problems such as talking human face generation remains challenging. The generation of deepfakes created by the abovementioned methods would greatly promote many fascinating applications, including augmented reality, virtual reality, computer games, teleconferencing, virtual try-on, special movie effects, and avatars. This research reviews and discusses DL related methods, including CNN, GANs, NeRF, and their implementation in talking human face generation. We aim to analyze existing approaches regarding their implementation to talking face generation, investigate the related general problems, and highlight the open study issues. We also provide quantitative and qualitative evaluations of the existing research approaches in the related field.",
    "keywords": "Talking human face animation; 3D face generation; Deep generative model; Autoencoder; Neural radiance field; Datasets; Evaluation metrics; Neural networks; Unsupervised learning; Mel spectogram",
    "url": "https://www.sciencedirect.com/science/article/pii/S0957417423001793"
  },
  {
    "id": 4,
    "type": "journal",
    "title": "Robot Bionic Eye Motion Posture Control System.",
    "authors": "Hongxin Zhang, and Suan Lee",
    "venue": "Electronics",
    "date": "January 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 2.690, JCR: Q2)",
    "abstract": "This paper mainly studies the structure and system of robot bionic eye. Given that most robots usually run on battery power, the STM32L053C8T6 with high efficiency and super low power consumption was selected as the main control. By carrying IMU, the bionic eye attitude data can be acquired quickly and accurately, and the measurement data of accelerometer and gyroscope can be fused by the algorithm to obtain stable and accurate bionic eye attitude data. Thus, the precise control of the motor can be realized through the drive control system equipped with PCA9685, which can enhance the motion control precision of robot bionic eye. In the present study, three kinds of IMU sensors, MPU6050, MPU9250, and WT9011G4K, were selected to carry out experiments. Finally, MPU9250 with better power consumption and adaptability is selected. This is the attitude acquisition device of bionic eye. In addition, three different filters, CF, GD, and EKF, were used for data fusion and comparison. The experimental result showed that the dynamic mean errors of CF, GD, and EKF are 0.62°, 0.61°, and 0.43°, respectively, and the static mean errors are 0.1017°, 0.1001°, and 0.0462°, respectively. The result showed that, after the use of EKF, the robot bionic eye system designed in this paper can significantly reduce the attitude angle error and effectively improve the image quality. It ensures accuracy and reduces power consumption and cost, which has lower requirements on hardware and is easier to popularize.",
    "keywords": "bionic eye; motion attitude detection; attitude sensor; Kalman filter",
    "url": "https://www.mdpi.com/2079-9292/12/3/698"
  },
  {
    "id": 5,
    "type": "journal",
    "title": "Robust and Lightweight Deep Learning Model for Industrial Fault Diagnosis in Low-Quality and Noisy Data.",
    "authors": "Jaegwang Shin, and Suan Lee",
    "venue": "Electronics",
    "date": "January 2023",
    "badge": "[SCIE]",
    "impact": "(IF: 2.690, JCR: Q2)",
    "abstract": "Machines in factories are typically operated 24 h a day to support production, which may result in malfunctions. Such mechanical malfunctions may disrupt factory output, resulting in financial losses or human casualties. Therefore, we investigate a deep learning model that can detect abnormalities in machines based on the operating noise. Various data preprocessing methods, including the discrete wavelet transform, the Hilbert transform, and short-time Fourier transform, were applied to extract characteristics from machine-operating noises. To create a model that can be used in factories, the environment of real factories was simulated by introducing noise and quality degradation to the sound dataset for Malfunctioning Industrial Machine Investigation and Inspection (MIMII). Thus, we proposed a lightweight model that runs reliably even in noisy and low-quality sound data environments, such as a real factory. We propose a Convolutional Neural Network–Long Short-Term Memory (CNN–LSTM) model using Short-Time Fourier Transforms (STFTs), and the proposed model can be very effective in terms of application because it is a lightweight model that requires only about 6.6% of the number of parameters used in the underlying CNN, and has only a performance difference within 0.5%.",
    "keywords": "fault diagnosis; deep learning; CNN; image representation; feature extraction",
    "url": "https://www.mdpi.com/2079-9292/12/2/409"
  },
  {
    "id": 6,
    "type": "dconference",
    "title": "Noise와 Curruption을 이용한 디퓨전 생성모델의 성능분석",
    "authors": "배기웅, 이수안, 이우기",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "디퓨전 생성모델은 가우시안 노이즈를 더해주는 디퓨전 과정과 노이즈를 걷어내면서 학습하는 리버스 과정을 통해 이미지를 만들어내는 생성모델이다. 본 논문에서는 획득비용이 높거나 정제가 어려워 다량의 노이즈가 포함된 데이터셋을 디퓨전 생성모델이 학습하는 과정을 통해 디퓨전 생성모델이 노이즈에 견고하며 일반화에 우수함을 증명하였다. 실험 과정에서는 총 3개의 노이즈를 사용하였으며, MNIST 데이터셋을 사용하였다. 검증은 프레쳇 인셉션 거리(Fréchet Inception Distance, FID)를 사용해 평가하였다. 실험을 통해 디퓨전 생성모델의 견고함을 확인하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224347"
  },
  {
    "id": 7,
    "type": "dconference",
    "title": "아파트 매매가 예측 모델을 위한 불규칙 시계열 데이터 보정 기법",
    "authors": "이수형, 고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "사람들이 생각하는 주택보유에 대한 인식은 일반적으로 집 한 채는 소유해야 한다는 의견이 많다. 그리하여 연구자들은 아파트 매매가를 예측하고 주택소유를 통해 주거문제를 완화하기 위해 노력하고 있다. 본 논문에서는 최근 아파트 거래량 감소로 인한 불규칙적인 시계열 데이터에 대한 예측을 위해 딥러닝 모델과 머신러닝 모델에 전처리 방법을 적용한 성능 비교분석과 어떤 전처리 방법과 모델이 불규칙적인 시계열 데이터에 강점을 보이는지에 대한 연구를 진행하였으며, SMA(6) 기법을 사용한 NeuralProphet 모델이 가장 우수한 성능을 보여주었다. 본 연구 결과를 통해 향후 불규칙적인 시계열 데이터의 예측에 있어서 도움이 될 것이다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224414"
  },
  {
    "id": 8,
    "type": "dconference",
    "title": "니켈 원자재 가격 예측을 위한 딥러닝 기반 시계열 모델 비교",
    "authors": "서경식, 고상근, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "badge": "[장려상]",
    "abstract": "원자재에는 금, 은과 같은 금속 뿐만 아니라 곡물, 육류, 에너지. 농산물 등 여러 가지 종류가 있다. 이러한 원자재는 우리의 생활들에 없어선 안될 요소들이다. 따라서 원자재 가격을 예측하는 연구가 필요하다. 본 논문에서는 니켈 가격데이터를 기반으로 Seq2seq, Transformer, Informer 세 가지 모델을 사용하여 시계열 예측 실험을 진행하였다. 실험은 학습 데이터를 4달, 6달로 나누고 테스트 데이터를 1달로 설정하여 두 가지 기준으로 진행하였다. 기준 1에서는 Transformer 모델이 기준 2에서는 Informer 모델이 가장 좋은 성능을 보여 기간이 장기간으로 갈수록 Informer 모델이 좋은 성능을 보인 것을 알 수 있다. 이러한 연구결과를 통해 앞으로의 원자재 가격 예측에 기여를 할 것으로 기대한다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224419"
  },
  {
    "id": 9,
    "type": "dconference",
    "title": "소음에 견고한 코골이 소리 분류를 위한 특징 추출 기반의 CNN-LSTM 모델",
    "authors": "신재광, 김남현, 최예신, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "최근 불면증 및 수면장애 등으로 건강에 문제가 있는 현대인들이 많고, 자신의 수면 상태와 패턴을 분석하고자 하는 요구가 증가하고 있다. 본 논문에서는 코골이 소리 데이터를 통하여 수면 중에 발생하는 코골이를 탐지 및 분류하는 딥러닝 모델을 연구하였다. 코골이 소리의 주요 특징들을 추출하기 위해 다양한 특징 추출 기법을 이용해 모델에 학습시키는 방법을 사용하였다. 소리의 특징 추출 방법으로는 STFT, Hilbert Transform, DWT를 사용하였고, STFT를 사용하였을 때 98.12%의 정확도로 모델의 성능이 가장 우수한 결과를 보여주었다. 본 연구 결과를 통하여 수면 중에 발생하는 코골이를 매우 높은 정확도로 탐지 및 분류가 가능하며, 향후 다양한 수면 분석에 활용될 수 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224437"
  },
  {
    "id": 10,
    "type": "dconference",
    "title": "인쇄 회로 기판(PCB) 결함 탐지 및 분류 딥러닝 모델 비교",
    "authors": "최동현, 전제성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "최근 일상생활에서의 전자제품 대중화로 인하여 인쇄 회로 기판(Printed Circuit Board; PCB)의 수요와 공급도 늘어나고 있다. PCB의 생산량이 늘어남에 따라 전자 제품 생산 회사 및 공장에서는 결함을 가진 PCB를 사전에 탐지하기 위해 많은 인력과 설비를 사용하고 큰 비용을 지출한다. 본 논문에서는 객체 탐지 모델인 YOLOv5, EfficientDet 2개의 모델을 사용하여 PCB 결함 탐지 작업에 대해 성능을 측정한 후, 각 모델의 실험 결과를 토대로 비교 분석하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224437"
  },
  {
    "id": 11,
    "type": "dconference",
    "title": "상수관로 진동 센서 데이터를 이용한 누수 감지 머신러닝 모델",
    "authors": "김병학, 전제성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "badge": "[장려상]",
    "abstract": "자동차의 증가, 도로의 확장 등으로 인해 교통사고 발생률이 증가하고 있으며, 이로 인해 인명피해, 기물 파손 등 금전적인 피해도 증가하고 있다. 본 논문에서는 교통사고에 취약한 도로형태, 도로의 종류에 따라 사고 건수를 예측하는 딥러닝 모델을 제안하였다. 서울시 사고 데이터를 구별 도로형태, 인구수, 자동차 수, 사고 건수를 나누어 사용하였고, Linear, TF, FM, NTF 모델을 사용하여 사고 건수 예측 실험을 하였다. 실험 결과로는 NTF 모델에서 가장 우수한 성능을 보여주었다. 본 연구를 통하여 NTF 모델로 교통사고를 예측하는 결과를 볼 수 있었고, 향후 연구를 통하여 모델의 성능을 올리고 경량화하여 실생활에서도 사용할 수 있게 만들어 교통사고 분석 및 예측에 활용할 수 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224438"
  },
  {
    "id": 12,
    "type": "dconference",
    "title": "도로 형태와 지역 정보를 결합한 서울시 교통사고 건수 예측 모델",
    "authors": "김남현, 고상근, 김성재, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "자동차의 증가, 도로의 확장 등으로 인해 교통사고 발생률이 증가하고 있으며, 이로 인해 인명피해, 기물 파손 등 금전적인 피해도 증가하고 있다. 본 논문에서는 교통사고에 취약한 도로형태, 도로의 종류에 따라 사고 건수를 예측하는 딥러닝 모델을 제안하였다. 서울시 사고 데이터를 구별 도로형태, 인구수, 자동차 수, 사고 건수를 나누어 사용하였고, Linear, TF, FM, NTF 모델을 사용하여 사고 건수 예측 실험을 하였다. 실험 결과로는 NTF 모델에서 가장 우수한 성능을 보여주었다. 본 연구를 통하여 NTF 모델로 교통사고를 예측하는 결과를 볼 수 있었고, 향후 연구를 통하여 모델의 성능을 올리고 경량화하여 실생활에서도 사용할 수 있게 만들어 교통사고 분석 및 예측에 활용할 수 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224449"
  },
  {
    "id": 13,
    "type": "dconference",
    "title": "위성 영상 기반 시맨틱 세그멘테이션을 이용한 도시화 분석",
    "authors": "임채환, 전제성, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2022년 12월",
    "abstract": "최근 자영업자들이 코로나 19 팬데믹 현상의 여파로 인해 이전 수익에 비해 수익 창출에 많은 어려움을 겪고 있다. 유동인구에 따라 자영업자들의 수익이 결정되기 때문에 유동인구가 많고, 이에 따라 도시화가 잘된 지역일수록 수익 창출에 유리하다. 따라서 어려움을 겪고 있는 자영업자와 더불어 창업을 시작하려는 예정자들에게 어느 곳이 도시화가 잘 되었는지 주거지역 및 상권 발달 경향 파악이 필요하다. 그리하여 본 논문에서는 서울시 한정으로 주거지역 및 주변 상권 건물을 분석하여 인구밀도를 예측할 수 있게 딥러닝 모델을 연구하였다. Semantic Segmentation 모델 중 메모리를 타 모델에 비교해 적게 사용하고, 빠른 속도와 파라미터 사용량 대비 높은 효율을 보이는 DeepLab V3+ 모델을 사용했다. 사용한 결과 주거 건물과 상권 건물에 대한 Segmentation이 잘 적용된 것을 확인했다. 본 연구결과를 통해 어느 지역이 사람이 많이 거주하고 있고 상권이 잘 발달하였는지 자영업자 및 창업 예정자들에게 정보를 제공하여 경향 파악에 도움을 주는 데 의의가 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224449"
  },
  {
    "id": 14,
    "type": "dconference",
    "title": "딥러닝 기반 시계열 예측모델: 경기도 가평군 유동인구를 중심으로",
    "authors": "고상근, 민종우, 이수안",
    "venue": "KDBC 2022(Korean DataBase Conference 2022)",
    "date": "2022년 11월 04일",
    "abstract": "유동인구는 도시의 발전 수준과 상권 및 관광 패턴을 분석함에 있어서 매우 중요한 요소이다. 유동인구의 수에 따라 각 지역의 발달, 위기 등이 존폐 가능성을 확인하는 데 큰 영향을 끼치는 요소이기에 때문이다. 그러므로 유동인구의 변화를 예측하는 연구가 필요하다. 이에 본 연구에서는 가평군의 유동인구 데이터 기반으로 Seq2seq, Transformer, Informer 세 가지의 딥러닝 모델을 사용하여 시계열 예측 실험을 진행하였다. 실험 결과, 손실값과 MAPE 측면에서는 Transformer 모델이, MAE, RMSE 측면에서 Informer 모델이 좋은 성능을 나타내었다. 이러한 연구 결과는 가평군뿐만 아니라 여러 지역의 유동인구 예측에 있어서 유의미한 기여를 할 것으로 기대된다."
  },
  {
    "id": 15,
    "type": "dconference",
    "title": "한국어 논문 요약을 위한 KoBART와 KoBERT 모델 비교",
    "authors": "전제성, 이수안",
    "venue": "제34회 한글 및 한국어 정보처리 학술대회, 한국정보과학회",
    "date": "2022년 10월 19일",
    "abstract": "통신 기술의 발전으로 일반인들도 다양한 자료들을 인터넷에서 손쉽게 찾아볼 수 있는 시대가 도래하였다. 개인이 접근할 수 있는 정보량이 기하급수적으로 많아 짐에 따라, 이를 효율적으로 요약, 정리하여 보여주는 서비스들의 필요성이 높아지기 시작했다. 본 논문에서는, 자연어 처리 모델인 BART를 40GB 이상의 한국어 텍스트로 미리 학습된 한국어 언어 모델 KoBART를 사용한 한국어 논문 요약 모델을 제안하고, KoBART와 KoBERT 모델의 한국어 논문 요약 성능을 비교한다."
  },
  {
    "id": 16,
    "type": "journal",
    "title": "Robot Bionic Vision Technologies: A Review.",
    "authors": "Hongxin Zhang, and Suan Lee",
    "venue": "Applied Sciences",
    "date": "August 2022",
    "badge": "[SCIE]",
    "impact": "(IF: 2.838, JCR: Q2)",
    "abstract": "The visual organ is important for animals to obtain information and understand the outside world; however, robots cannot do so without a visual system. At present, the vision technology of artificial intelligence has achieved automation and relatively simple intelligence; however, bionic vision equipment is not as dexterous and intelligent as the human eye. At present, robots can function as smartly as human beings; however, existing reviews of robot bionic vision are still limited. Robot bionic vision has been explored in view of humans and animals’ visual principles and motion characteristics. In this study, the development history of robot bionic vision equipment and related technologies are discussed, the most representative binocular bionic and multi-eye compound eye bionic vision technologies are selected, and the existing technologies are reviewed; their prospects are discussed from the perspective of visual bionic control. This comprehensive study will serve as the most up-to-date source of information regarding developments in the field of robot bionic vision technology.",
    "keywords": "artificial intelligence; robot bionic vision; optical devices; bionic eye; intelligent camera",
    "url": "https://www.mdpi.com/2076-3417/12/16/7970"
  },
  {
    "id": 17,
    "type": "dconference",
    "title": "CNN-LSTM 기반 시계열 데이터의 이상치 분류를 위한 이미지 인코딩 방법 비교 분석",
    "authors": "신재광, 안동주, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월",
    "abstract": "In this paper, we study an outlier detection modelusing the MIMII dataset to prevent machine failuresand accidents caused by failures in factories. Themodel was combined with CNN and LSTM, and thepreprocessing of data was performed by imageencoding. We compare methods to increase theaccuracy of the model through image encoding suchas STFT, DWT, and Hilbert transform, and as aresult, STFT was the best, followed by Hilberttransform. Through the results of this study, wedetected and classified outliers through the sound ofmachine operation, showing that failure prediction ispossible in the actual factory.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11132869"
  },
  {
    "id": 18,
    "type": "dconference",
    "title": "3D 가상 피팅 딥러닝 모델과 VR 연동을 위한 웹 서비스 개발",
    "authors": "전제성, 진동민, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월",
    "abstract": "With the recent development of artificial inteligencetechnology, various services utilizing artificialinteligence are becoming practical in variousindustries. Among them, the fashion industry isintroducing virtual try-on services using artificialintelligence technology. The virtual try-on service isa technology that synthesizes a person wearingclothes using only pictures of people and pictures ofclothes when purchasing cothes at an online storewithout going directly to the store. Among the deeplearning models implemented virtual try-on, theM3D-VTON[1] model is one of the models thatshowed good performances in its field. In this paper,we propose a react-based web frontend server anda 3D virtual try-on backend server using theM3D-VTON model based Rest API architecture. In addition, WebXR technology was used to enableusers to check the results of virtual try-on with VRon the web page.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11132869"
  },
  {
    "id": 19,
    "type": "dconference",
    "title": "엣지 TPU 기반의 임베디드 기기별 경량화된 이미지 분류 모델 비교",
    "authors": "최동현, 김남현, 고상근, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월",
    "abstract": "AIoT is emerging as it moves from the IT era ofthe tertiary industry to the fourth industrialrevolution. AIoT equipment requires00 lightweightmodel of deep learning because of its limitedperformance. In this paper, a comparative study wasconducted on the time required for inference oflightweight deep learning model on Edge TPUenvironment equipment optimized for artificialintelligence, and a significant speed difference couldbe confirmed. Using the results of this paper, it willhelp build an AIoT environment using Edge TPU.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11132897"
  },
  {
    "id": 20,
    "type": "dconference",
    "title": "임베디드 장비 환경에서 경량화된 객체 탐지 딥러닝 모델 비교",
    "authors": "김지나, 정은지, 전제성, 이수안",
    "venue": "하계학술대회논문집, 대한전자공학회",
    "date": "2022년 06월",
    "abstract": "Object detection, one of the representative technologies of artificial intelligence, is used in various fields such as smart cities and autonomous driving and is becoming increasingly important. Recently, in order to use artificial intelligence technology in the Edge environment, many researchers are studying how to reduce the weight of the model. In this paper, we compare YOLOv5, EfficientDet, SSD MobileNetV1, and spaghettinet models in terms of inference time. For comparative experiments, embedded devices used Google Coral Dev Board and ASUS Tinker Edge T.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11132898"
  },
  {
    "id": 21,
    "type": "dconference",
    "title": "딥러닝 모델 기반 가상 피팅 웹 서비스 개발",
    "authors": "고상근, 장현수, 민정호, 이다혁, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월",
    "badge": "[장려상]",
    "abstract": "본 연구에서는 데이터 획득 비용이 높아 충분한 학습 데이터를 획득하지 못하거나, 학습에 사용되는 데이터에 노이즈가 많이 포함되어 학습이 제대로 진행되지 못하는 상황에서 동일한 데이터이지만 서로 다른 모달의 동질적인 특징을 이용한 멀티모달 학습 방법이 노이즈에 견고하고, 일반화에 우수함을 실험을 통해 증명하였다. 실험 과정에서는 이미지 MNIST 데이터셋과 시퀀스 MNIST 데이터셋에 노이즈를 추가하여 사용하였다. 실험 비교를 위해 단일 CNN 계열 모델 3개, 단일 RNN 계열 모델 3개, 그리고 멀티모달 계열 모델 3개를 사용하였으며, 단일모달만을 사용하였을 때보다 멀티모달 모델을 사용했을 때 전반적으로 성능이 개선되는 것을 확인하였다",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11121476"
  },
  {
    "id": 22,
    "type": "dconference",
    "title": "동질적 특징을 이용한 견고한 멀티모달 분류 모델",
    "authors": "배기웅, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월",
    "badge": "[우수상]",
    "abstract": "본 연구에서는 데이터 획득 비용이 높아 충분한 학습 데이터를 획득하지 못하거나, 학습에 사용되는 데이터에 노이즈가 많이 포함되어 학습이 제대로 진행되지 못하는 상황에서 동일한 데이터이지만 서로 다른 모달의 동질적인 특징을 이용한 멀티모달 학습 방법이 노이즈에 견고하고, 일반화에 우수함을 실험을 통해 증명하였다. 실험 과정에서는 이미지 MNIST 데이터셋과 시퀀스 MNIST 데이터셋에 노이즈를 추가하여 사용하였다. 실험 비교를 위해 단일 CNN 계열 모델 3개, 단일 RNN 계열 모델 3개, 그리고 멀티모달 계열 모델 3개를 사용하였으며, 단일모달만을 사용하였을 때보다 멀티모달 모델을 사용했을 때 전반적으로 성능이 개선되는 것을 확인하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11121476"
  },
  {
    "id": 23,
    "type": "dconference",
    "title": "AIoT 환경에서 마커와 사람 인식을 이용한 식별된 사람 출입관리 시스템 개발",
    "authors": "임채환, 이수형, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2022년 06월",
    "abstract": "출입관리가 필요한 여러 기관과 장소 등에서 일일이 신원 확인을 하기 어려운 상황들이 상당수 발생한다. 또한, 한 번에 한 명씩 출입관리 시스템에 접근해야 하는 제한적인 조건 때문에 사람들이 몰리면 대기 시간이 발생하는 불편함과 혼잡함이 발생한다. 이를 한 번에 해소하고자 본 논문에서는 임베디드 장비에서 동작 가능한 실시간 사람 인식 및 마커 탐지 모델을 연구하였다. 객체 탐지 모델 중에서 빠른 속도와 높은 성능을 보이고 있는 YOLOv5 모델을 사용했고, 임베디드 장비인 NVIDIA Jetson AGX Xavier에서 사람과 마커를 탐지할 수 있는 모델 개발을 진행하였다. 이러한 개발환경에서 사람 탐지와 사람 몸에 부착된 마커를 실시간으로 잘 탐지하는 것을 확인했다. 본 연구결과를 통해 넓은 공간에서 많은 사람의 출입 및 신원을 효과적으로 관리한다. 더불어 제한된 출입 시스템의 체증에 대한 불편함과 혼잡함을 해소하고 편의성을 증대시켜 효율적인 작업자들의 출입관리에 기여를 하는데 의의가 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113809"
  },
  {
    "id": 24,
    "type": "djournal",
    "title": "전동기 기계시설물 고장 분류를 위한 이미지 인코딩 기반 경량화된 딥러닝 모델",
    "authors": "안동주, 신재광, 이수안",
    "venue": "대한전자공학회논문지, 대한전자공학회",
    "date": "2022년 04월",
    "badge": "[KCI]",
    "abstract": "산업 현장에서 사용되는 전동기 기계 설비들의 고장은 베어링, 회전체, 벨트, 축이 상당 부분을 차지한다. 설비들이 기계적 또는 전기적 원인에 의해 고장이 발생하거나 성능이 저하되면 공통 적으로 진동이 발생하고 전류 등이 이상 움직임을 보인다. 이러한 상황에서 불특정하게 발생하는 고장을 쉽게 감지하고 예측하는 것은 필수적이다. 따라서 본 논문에서는 전동기 기계 설비에 부착된 센서에서 생성되는 시계열 데이터를 이미지로 인코딩하는 방법을 사용하여 경량화된 딥러닝 모델을 제안하였다. 이미지 인코딩에는 세 개의 방식을 사용하였고, 각각의 방식에 대한 CNN 기반 딥러닝 분류모델을 생성하였다. CNN 모델은 작은 파라미터를 가지면서도 제일 정확도가 높은 모델을 실험을 통해 만들었다. CNN 모델의 정확도를 분석하고 어떠한 인코딩 방식이 학습에 효율적이고 더 적합한지 실험해보았더니 세 개의 이미지 인코딩 방식 중에서도 GASF 방식이 대체로 정확도가 높게 나온 것을 확인하였다. 본 논문에서 제안한 이미지 인코딩 기반의 경량화된 딥러닝 모델을 이용해 산업에서 활용되는 여러 센서 데이터에 대해 다양한 응용에 활용할 수 있을 것이라 예상된다.<br /> The failure of mechanical facilities used in industrial sites accounts for a significant portion of bearings, rotators, belts, and axes. When facilities fail due to mechanical or electrical causes or performance degrades, vibration is commonly generated and current or the like shows abnormal movement. It is essential to easily detect and predict failures that occur unspecified in this situation. Therefore, in this paper, a lightweight deep learning model was proposed using a method of encoding time series data generated by a sensor attached to a mechanical facility into an image. Three methods were used for image encoding, and a CNN-based deep learning classification model was created for each method. The CNN model was created through experiments with the most accurate model with small parameters. When analyzing the accuracy of the CNN model and experimenting with which encoding methods are more efficient and suitable for learning, it was confirmed that the GASF method was generally more accurate among the three image encoding methods. It is expected that various sensor data used in the industry can be used for various applications using the lightweight deep learning model based on image encoding proposed in this paper.",
    "keywords": "Timeseries classification, Image encoding, Deep learning, Lightweight model",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002841355"
  },
  {
    "id": 25,
    "type": "djournal",
    "title": "기계장비 진동 데이터를 이용한 딥러닝 기반의 고장 분류 모델",
    "authors": "이수형, 고상근, 이수안",
    "venue": "한국차세대컴퓨팅학회 논문지, 한국차세대컴퓨팅학회",
    "date": "2022년 02월",
    "badge": "[KCI]",
    "abstract": "최근 4차 산업 혁명으로 인해 공장에서는 기계 시설물의 고장으로 인한 제조 및 생산 시간 증가와 수리비용 증가를 최신 기술을 통해 예방하고자 한다. 그리하여 연구자들은 안전사고 및 고장을 예지(豫知)하고, 제품 불량 등으로 인한 시민 불편, 사회적 혼란 등의 문제를 방지하기 위해 노력하고 있다. 이런 문제 해결을 위해 장비에 부착된 센서로부터 받은 데이터를 통해 기계 상황을 모니터링 및 예측 가능한 시스템이 요구된다. 본 논문에서는 기계설비에서 생성되는 시계열 데이터를 다양한 딥러닝 기반의 시계열 분류 모델에서 비교분석을 수행하였다. 총 13개의 모델을 사용하여 어떤 딥러닝 모델이 학습에 효율적이고 성능이 좋은지 실험을 수행하였고, 시계열의 복잡한 패턴과 시간 및 공간 패턴을 효과적으로 학습하는 CNN 계열 모델이 정확도, 정밀도, 재현율, F1 성능지표에서 100% 성능을 달성하여 우수한 것을 확인할 수 있었다. 기존까지 진동 시계열 데이터에 대해 다양한 딥러닝 모델의 비교분석이 연구는 없었기에 본 논문의 결과를 통해 다른 기계 시설물의 고장 분류에 있어서 도움이 될 것이라 예상된다.<br /> Due to the recent Fourth Industrial Revolution, factories want to prevent the increase in manufacturing and production time and repair costs due to the failure of mechanical facilities through the latest technology. Therefore, researchers are trying to predict safety accidents and failures, and to prevent problems such as civil inconvenience and social confusion caused by product defects. To solve this problem, a system that can monitor and predict the machine situation through data received from sensors attached to the equipment is required. In this paper, comparative analysis was performed on time series data generated in machine facilities in various deep learning-based time series classification models. A total of 13 models were used to experiment with which deep learning models were efficient and performing well, and the models that effectively learned time and space patterns of time series recorded 100% performance in accuracy, precision, reproducibility, and F1 performance indicators. Since there have been no studies on the comparative analysis of various deep learning models for vibration time series data until now, the results of this paper are expected to help in classifying failures in other mechanical facilities.",
    "keywords": "시계열 데이터, 딥러닝, 기계학습, 분류, 예지 보전<br /> Time Series, Deep Learning, Machine learning, Classification, Predictive Maintenance",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002837888"
  },
  {
    "id": 26,
    "type": "journal",
    "title": "Human pose, hand and mesh estimation using deep learning: a survey.",
    "authors": "Mukhiddin Toshpulatov, Wookey Lee, Suan Lee and Arousha Haghighian Roudsari",
    "venue": "The Journal of Supercomputing",
    "date": "January 2022",
    "badge": "[SCIE]",
    "impact": "(IF: 2.474, JCR: Q2)",
    "abstract": "Human pose estimation is one of the issues that have gained many benefits from using state-of-the-art deep learning-based models. Human pose, hand and mesh estimation is a significant problem that has attracted the attention of the computer vision community for the past few decades. A wide variety of solutions have been proposed to tackle the problem. Deep Learning-based approaches have been extensively studied in recent years and used to address several computer vision problems. However, it is sometimes hard to compare these methods due to their intrinsic difference. This paper extensively summarizes the current deep learning-based 2D and 3D human pose, hand and mesh estimation methods with a single or multi-person, single or double-stage methodology-based taxonomy. The authors aim to make every step in the deep learning-based human pose, hand and mesh estimation techniques interpretable by providing readers with a readily understandable explanation. The presented taxonomy has clearly illustrated current research on deep learning-based 2D and 3D human pose, hand and mesh estimation. Moreover, it also provided dataset and evaluation metrics for both 2D and 3D HPE approaches.",
    "keywords": "3D pose estimation, Generator, Discriminator, Loss function, Deep neural network, Deep learning, Mesh estimation, Evaluation metric, Dataset",
    "url": "https://link.springer.com/article/10.1007/s11227-021-04184-7"
  },
  {
    "id": 27,
    "type": "journal",
    "title": "PatentNet: multi-label classification of patent documents using deep learning based language understanding.",
    "authors": "Haghighian Roudsari, Arousha, Jafar Afshar, Wookey Lee, and Suan Lee",
    "venue": "Scientometrics",
    "date": "January 2022",
    "badge": "[SSCI]",
    "impact": "(IF: 3.238, JCR: Q1)",
    "abstract": "Patent classification is an expensive and time-consuming task that has conventionally been performed by domain experts. However, the increase in the number of filed patents and the complexity of the documents make the classification task challenging. The text used in patent documents is not always written in a way to efficiently convey knowledge. Moreover, patent classification is a multi-label classification task with a large number of labels, which makes the problem even more complicated. Hence, automating this expensive and laborious task is essential for assisting domain experts in managing patent documents, facilitating reliable search, retrieval, and further patent analysis tasks. Transfer learning and pre-trained language models have recently achieved state-of-the-art results in many Natural Language Processing tasks. In this work, we focus on investigating the effect of fine-tuning the pre-trained language models, namely, BERT, XLNet, RoBERTa, and ELECTRA, for the essential task of multi-label patent classification. We compare these models with the baseline deep-learning approaches used for patent classification. We use various word embeddings to enhance the performance of the baseline models. The publicly available USPTO-2M patent classification benchmark and M-patent datasets are used for conducting experiments. We conclude that fine-tuning the pre-trained language models on the patent text improves the multi-label patent classification performance. Our findings indicate that XLNet performs the best and achieves a new state-of-the-art classification performance with respect to precision, recall, F1 measure, as well as coverage error, and LRAP.",
    "keywords": "Patent classification, Multi-label text classification, Pre-trained language model",
    "url": "https://link.springer.com/article/10.1007/s11192-021-04179-4"
  },
  {
    "id": 28,
    "type": "conference",
    "title": "Privacy-Preserving of Human Identification in CCTV Data using a Novel Deep Learning-Based Method",
    "authors": "Toshpulatov Mukhiddin, Haghighian Roudsari Arousha, Asatullaev Ubaydullo, Lee Wookey, and Suan Lee",
    "venue": "2022 IEEE International Conference on Big Data and Smart Computing (BigComp). IEEE, 2022.",
    "date": "January 17-20, 2022",
    "abstract": "With the rapid development of Information Tech-nologies, the number of surveillance cameras has increased, leading to a decrease in the rate of violence and crimes. Despite the advantages, it has some drawbacks, such as the lack of privacy-preserving in human identification when video data is shared. This paper focuses on the privacy-preserving issue and proposes a novel deep learning-based solution. The proposed approach uses the recent state-of-the-art models such as RetinaFace and TinaFace for human face detection from the input videos and other Computer Vision tools such as OpenCV for framing the input video and connecting them again for restoring the initial form of the video in the output. Moreover, computer vision tools such as Blur and Gaussian Blurring are used for anonymizing faces. Our proposed approach allows the closed-circuit television (CCTV) data to be shared for public use, where human identification is perfectly preserved. Experimental results show the effectiveness of our proposed method by outperforming the state-of-the-art methods in constrained conditions. Furthermore, we have created a face dataset from the input CCTV videos, where the face detection tools have failed. The created dataset is annotated with the five face landmarks and can be used for the face detection task.",
    "keywords": "Privacy preserving; human identification;dataset; evaluation metric; face anonymizing; human face de-tection; deep neural network; deep learning",
    "url": "https://ieeexplore.ieee.org/document/9736487"
  },
  {
    "id": 29,
    "type": "conference",
    "title": "Tracking Untrained Objects Based On Optical Flow Approach",
    "authors": "George Jung Yup Rhee, Suan Lee, and Wookey Lee",
    "venue": "2022 IEEE International Conference on Big Data and Smart Computing (BigComp). IEEE, 2022.",
    "date": "January 17-20, 2022",
    "abstract": "As Computer Vision is being used in various fields these days, the fields that require object tracking increase, and the performance starts to improve accordingly. In object tracking, State of the Art models are methodologies that obtain high accuracy by learning the objects to be tracked, whereas they have low accuracy for unlearned objects. To compensate for this, some models provide an environment for transfer learning on the object to be tracked, but there is rarely a large amount of preprocessed data that can be trained. In such a problem situation, it is possible to track an object universally and accurately using Lukas Kanade optical flow and additionally derive center coordinates.",
    "keywords": "Object Tracking, Lukas-Kanade Optical Flow, Untrained Object",
    "url": "https://ieeexplore.ieee.org/document/9736494"
  },
  {
    "id": 30,
    "type": "conference",
    "title": "Transformer Networks for Trajectory Classification",
    "authors": "Keywoong Bae, Suan Lee, and Wookey Lee",
    "venue": "2022 IEEE International Conference on Big Data and Smart Computing (BigComp). IEEE, 2022.",
    "date": "January 17-20, 2022",
    "abstract": "Research related to Trajectory Classification is actively underway, and its application fields are also very diverse. Existing studies related to trajectory classification mainly used RNN-based models such as SimpleRNN, LSTM, GRU, etc. However, these Seq2Seq models cause a bottle neck problem that does not reflect all information when the length of the input sequence increases during the encoding process. Therefore, we propose a Transformer model for more accurate trajectory classification even in situations where the trajectory input sequence is long. As a dataset, we use MNIST stroke sequence dataset, which expresses the stroke of the numbers of the MNIST as a unit vector trajectory. As a result, Transformer achieved comparable performance to LSTM.",
    "keywords": "Trajectory classification, MNIST stroke sequence data, Transformer",
    "url": "https://ieeexplore.ieee.org/document/9736500"
  },
  {
    "id": 31,
    "type": "dconference",
    "title": "윤리적 소비를 위한 지속가능한 거래: 중고거래 금지 품목 탐지를 중심으로",
    "authors": "이충성, 전서영, 이우기, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월",
    "abstract": "최근 온라인 중고거래 시장은 빠르게 성장하고 거래되는 품목도 다양해지고 있다. 중고거래 시장이 커지면서 정상 범주를 벗어난 거래 또한 증가하였다. 온라인 중고 거래 금지 품목 중 건강식품과 의약품이 거래되는 경우 약물 부작용 등 심각한 문제를 초래할 수 있다. 이러한 문제를 방지하기 위해 온라인 중고거래 시장에서 실시간으로 거래금지 품목을 식별하는 것은 매우 중요하다. 본 연구에서는 크롤링한 이미지를 이용한 딥러닝의 EfficientnetB0, Resnet50, Denset121 모델을 만들어 거래금지 품목의 이미지 분류를 하고 각 모델의 성능을 비교한다",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11036045"
  },
  {
    "id": 32,
    "type": "dconference",
    "title": "택시 데이터 분석을 통한 수요 예측과 승차대 위치 개선",
    "authors": "김예원, 김유정, 김주아, 이유정, 이우기, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월",
    "badge": "(장려상)",
    "abstract": "본 논문에서는 택시 승차 데이터 분석을 통해 수요와 공급이 일치하지 않는 곳에 위치한 택시 승차대의 위치와 서비스를 개선하고 ‘에코 스마트 택시 승차대’를 제안한다. 서울시 도로 네트워크 데이터와 택시 운행 분석 데이터, 택시 승차대 데이터를 K-means 기법과 Elbow 기법을 이용해 서울시 권역의 택시 승차 수요량과 승차대 현황을 분석한 후 서울시 지도 위에 시각화하였다. 데이터 분석 결과 수요 대비 승차대 위치와 수의 불균형이 확인되었고, 개선이 필요한 승차대의 위치를 추천하였다. 본 연구를 통해 택시 운수종사자, 이용 승객, 도시 시설의 모든 방면에서의 개선을 기대할 수 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11036014"
  },
  {
    "id": 33,
    "type": "dconference",
    "title": "NVIDIA Jetson AGX Xavier 환경에서 경량화된 실시간 안전모 탐지 모델 개발",
    "authors": "조홍석, 이수형, 장현수, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2021년 12월",
    "abstract": "최근 공사현장 및 건설현장에서 빈번히 발생하는 안전사고를 예방하고자 개인안전장비(PPE, Person Protective Equipment)를 착용해야 하며, 전동 킥보드와 자전거와 같은 이동 수단에서도 안전모를 착용해야 하는 법적 의무가 있다. 본 논문에서는 임베디드 모듈에서 동작 가능한 경량화된 실시간 안전모 탐지모델 연구를 진행하였다. 최근 객체탐지 분야에서 빠른 속도와 높은 성능을 보이는 YOLOv5 모델을 이용하였고, 외부나 공사현장에서 안전모 착용 여부를 탐지할 수 있도록 임베디드 환경인 NVIDIA Jetson AGX Xavier에서 개발하였다. 실제 사람이 안전모를 착용하거나 미착용한 경우에 대해서 실시간으로 잘 탐지하는 것을 확인하였으며, 본 연구결과를 통해 공사현장이나 이동 수단 이용 시에 단순 안전모 미착용으로 발생하는 안전사고 예방에 도움이 되는 데 의의가 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11035992"
  },
  {
    "id": 34,
    "type": "conference",
    "title": "Trajectory Privacy Preservationby Using Deep Learning: Transformer-TrajGAN",
    "authors": "Ellen S. Park, Hazel H. Kim, Suan Lee and Wookey Lee",
    "venue": "The 9th International Conference on Big Data Applications and Services (BIGDAS), 2021.",
    "date": "November 25-27, 2021",
    "abstract": "With the rapid development of mobile information and communication technology, the demand and supply of location-based services are increasing in many industries. Various trajectory data combining location data and spatio\u0002temporal information are being used, leading to serious privacy violations for individuals and countries. To solve this problem, we propose a research methodology to generate synthetic trajectory data that protects personal privacy by transducing a deep learning model, transformer, with a generative adversarial network (GAN) approach. The proposed model preserves the spatial and temporal characteristics of trajectory data while protecting location data-based path privacy.",
    "keywords": "Deep learning, Generative Adversarial Network, Transformer, Trajectory, Privacy Protection",
    "url": "http://www.kbigdata.or.kr/bigdas2021/"
  },
  {
    "id": 35,
    "type": "djournal",
    "title": "기계학습 모델을 이용한 이상기상에 따른 사일리지용 옥수수 생산량 피해량",
    "authors": "조현욱, 김민규, 김지융, 조무환, 김문주, 이수안, 김경대, 김병완, 성경일",
    "venue": "한국초지조사료학회지, 한국초지조사료학회",
    "date": "2021년 12월",
    "badge": "[KCI]",
    "abstract": "본 연구는 기계학습을 통한 수량예측모델을 이용하여 이상기상에 따른 WCM의 DMY 피해량을 산출하기 위한 목적으로 수행하였다. 수량예측모델은 WCM 데이터 및 기상 데이터를 수집 후 가공하여 8가지 기계학습을 통해 제작하였으며 실험지역은 경기도로 선정하였다. 수량예측모델은 기계학습 기법 중 정확성이 가장 높은 DeepCrossing (R2=0.5442, RMSE=0.1769) 기법을 통해 제작하였다. 피해량은 정상기상 및 이상기상의 DMY 예측값 간 차이로 산출하였다. 정상기상에서 WCM의 DMY 예측값은 지역에 따라 차이가 있으나 15,003~17,517 kg/ha 범위로 나타났다. 이상기온, 이상강수량 및 이상풍속에서 WCM의 DMY 예측 값은 지역 및 각 이상기상 수준에 따라 차이가 있었으며 각각 14,947~17,571 kg/ha, 14,986~17,525 kg/ha 및 14,920~17,557 kg/ha 범위로 나타났다. 이상기온, 이상강수량 및 이상풍속에서 WCM의 피해량은 각각 –68~89 kg/ha, -17~17 kg/ha 및 – 112~121 kg/ha 범위로 피해로 판단할 수 없는 수준이었다. WCM의 정확한 피해량을 산출하기 위해서는 수량예측모델에 이용하는 이상기상 데이터 수의 증가가 필요하다.",
    "keywords": "Abnormal climate, Whole crop maize, Machine learning, Forage yield prediction model, Dry matter yield damage",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002470779"
  },
  {
    "id": 36,
    "type": "dconference",
    "title": "장애인 안전 이동을 위한 이미지 캡션 기반 시각 읽기 딥러닝 모델",
    "authors": "이정엽, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2021년 07월",
    "abstract": "인간의 편의성을 높이기 위해 다양한 방면에서 도움을 주는 인공지능 모델들이 생겨나고 있지만, 장애인을 위한 인공지능 모델은 부족하며, 불편함을 근본적으로 해소하는 모델이나 서비스는 없다. 이러한 이유로 장애인의 이동권은 제한되어 있으며, 주체적 이동이 불가능한 것이 현실이다. 본 논문에서는 이러한 장애인들의 이동권에 기여하고자, 안전한 보행에 위협이 되는 요소들을 객체 인식 모델로 식별하고 이미지 캡셔닝 모델로 주위 상황에 대한 설명을 음성으로 제공하는 안전 보행 모델을 제안한다. 그리고 추가적인 데이터 셋을 활용하여 본 논문에서 제안한 안전 보행 모델을 보완하고 강화할 수 있는 방법에 대해 설명한다. 이를 통해 장애인들의 주체적인 보행을 보장하여 궁극적으로 이동권을 향상시킬 수 있을 것으로 기대한다."
  },
  {
    "id": 37,
    "type": "dconference",
    "title": "모바일 자율주행 로봇을 위한 저비용 차선 인식 및 제어 알고리즘",
    "authors": "장현수, 이수안",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2021년 07월",
    "badge": "(장려상)",
    "abstract": "최근 실내에서 작동하는 모바일 자율주행 로봇의 중요성은 크게 증가하고 있다. 본 논문에서는 제한된 실내 환경 조건 아래에 저비용으로도 자율주행이 가능한 모바일 로봇을 위해 저비용 차선 인식 및 제어 알고리즘을 제안하도록 제시한다. 특히 관심 영역 자동 추출, 히스토그램을 이용한 RGB 임계값 자동 추출을 통해 영상 처리 과정에서의 정확도를 높이고 연산을 줄일 수 있다. 또한, 저비용의 차선 인식을 바탕으로 보조선을 이용한 조향각 제어를 통해 자율주행에 사용한다. 주행에 있어 차선이 하나만 인식되거나 영상 중앙에 차선이 인식될 때 가중치를 주어 안전하게 주행이 이루어지도록 하였다"
  },
  {
    "id": 38,
    "type": "column",
    "title": "[테크인사이드] 메타버스 시대가 오고 있다",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2021년 04월 14일",
    "abstract": "최근 엔비디아 최고경영자인 젠슨 황은 '메타버스 시대가 오고 있다'고 말했다. 메타버스(Metaverse)는 가상·초월을 뜻하는 메타(Meta)와 세계·우주를 뜻하는 유니버스(Universe)의 합성어로 물리적 공간을 초월하는 새로운 디지털 세상, 가상현실 세계를 의미한다. 메타버스에서는 사람들은 원하는 아바타를 만들어 경제, 사회, 문화 활동 등 뭐든지 할 수 있다. 스티븐 스필버그 감독이 디지털 가상현실을 배경으로 제작한 영화 '레디 플레이어 원'이 대표적인 메타버스의 예다.",
    "url": "http://www.kwnews.co.kr/nview.asp?aid=221041300001"
  },
  {
    "id": 39,
    "type": "journal",
    "title": "OLGAVis: On-Line Graph Analysis and Visualization for Bibliographic Information Network",
    "authors": "Sunhwa Jo, Beomjun Park, Suan Lee, and Jinho Kim. 2021.",
    "venue": "Applied Sciences",
    "date": "April 2021",
    "badge": "[SCIE]",
    "impact": "(IF: 2.474, JCR: Q1)",
    "abstract": "Real-world systems that are composed of various types of components, their interactions, and relationships, and numerous applications are often modeled as graphs or network structures to represent and analyze the presence of relationship, shape, and meaning of objects. Network-structured data are used for various exploration and in-depth analysis through visualization of information in various fields. In particular, online bibliographic databases are a service that is used for a myriad of purposes, such as simple search of research materials as well as understanding the history and flow of research, current status, and trends. A visualization tool that can intuitively perform exploration and analysis by modeling the data provided by the online bibliographic database in a network structure will be a very meaningful study for the exploration of various information using a large amount of complex bibliographic data. This study has modeled an online bibliographic database as an information network, and further developed a prototype of a visualization tool that provides an interactive interface for easily and efficiently performing visual exploration and multidimensional analysis. The visualization tool that was developed through this study will be used to conveniently perform various online analysis of the bibliographic data, and the information and knowledge acquired as a result of the analysis are expected to contribute to the research development of various researchers. Furthermore, this visualization tool can be applied to other types of data in the future, and it is expected to develop into a useful tool for various information network analysis by improving, supplementing, and expanding the functions and performance of the developed prototype.",
    "keywords": "bibliographic information network; Information Network OLAP; information network visualization",
    "url": "https://www.mdpi.com/2076-3417/11/9/3862/htm"
  },
  {
    "id": 40,
    "type": "journal",
    "title": "Generative adversarial networks and their application to 3D face generation: A survey.",
    "authors": "Toshpulatov, Mukhiddin, Wookey Lee, and Suan Lee.",
    "venue": "Image and Vision Computing",
    "date": "February 2021",
    "badge": "[SCIE]",
    "impact": "(IF: 3.103, JCR: Q1)",
    "abstract": "Generative adversarial networks (GANs) have been extensively studied in recent years and have been used to address several problems in the fields of image generation and computer vision. Despite significant advancements in computer vision, applying GANs to real-world problems such as 3D face generation remains a challenge. Owing to the proliferation of fake images generated by GANs, it is important to analyze and build a taxonomy for providing an overall view of GANs. This, in turn, would facilitate many interesting applications, including virtual reality, augmented reality, computer games, teleconferencing, virtual try-on, special effects in movies, and 3D avatars. This paper reviews and discusses GANs and their application to 3D face generation. We aim to compare existing GANs methods in terms of their application to 3D face generation, investigate the related theoretical issues, and highlight the open research problems. Authors provided both qualitative and quantitative evaluations of the proposed approach. They claimed their results show the higher quality of the synthesized data compared to state-of-the-art ones.",
    "keywords": "Generative adversarial networks; 3D face generation; Generator; Discriminator; Deep neural network; Deep learning",
    "url": "https://www.sciencedirect.com/science/article/abs/pii/S026288562100024X"
  },
  {
    "id": 41,
    "type": "journal",
    "title": "Biosignal Sensors and Deep Learning-Based Speech Recognition: A Review",
    "authors": "Wookey Lee, Jessica J. Seong, Busra Ozlu, Bong S. Shim, Azizbek Marakhimov, and Suan Lee.",
    "venue": "Sensors",
    "date": "February 2021",
    "badge": "[SCIE]",
    "impact": "(IF: 3.275, JCR: Q1)",
    "abstract": "Voice is one of the essential mechanisms for communicating and expressing one’s intentions as a human being. There are several causes of voice inability, including disease, accident, vocal abuse, medical surgery, ageing, and environmental pollution, and the risk of voice loss continues to increase. Novel approaches should have been developed for speech recognition and production because that would seriously undermine the quality of life and sometimes leads to isolation from society. In this review, we survey mouth interface technologies which are mouth-mounted devices for speech recognition, production, and volitional control, and the corresponding research to develop artificial mouth technologies based on various sensors, including electromyography (EMG), electroencephalography (EEG), electropalatography (EPG), electromagnetic articulography (EMA), permanent magnet articulography (PMA), gyros, images and 3-axial magnetic sensors, especially with deep learning techniques. We especially research various deep learning technologies related to voice recognition, including visual speech recognition, silent speech interface, and analyze its flow, and systematize them into a taxonomy. Finally, we discuss methods to solve the communication problems of people with disabilities in speaking and future research with respect to deep learning components.",
    "keywords": "mouth interface; voice production; artificial larynx; EMG; biosignal; deep learning",
    "url": "https://www.mdpi.com/1424-8220/21/4/1399/htm"
  },
  {
    "id": 42,
    "type": "column",
    "title": "[테크인사이드] 바이러스 연구부터 뷰티·배달 AI 결합한 비즈니스 모델 주목",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2021년 01월 20일",
    "abstract": "코로나19 사태로 지금의 일상은 이전과 크게 바뀌었다. 포스트 코로나 시대는 오프라인 사회를 온라인으로 전환했고 인공지능이 실생활에 적용되며 새로운 비즈니스 모델이 발굴되는 세상이 찾아오고 있다. 코로나19로 인해 가장 시급한 건 인공지능 기술 기반 단백질 구조 분석을 통한 바이러스의 기전과 치료법 연구다. 인공지능 기업 블루닷은 코로나19로 인한 팬데믹 현상을 예측했다. 블루닷은 병원 시설, 지역 이동, 가축 및 해충 현황, 국제 항공 이동 데이터, 실시간 기후 변화 등의 데이터를 수집·분석해 의학·역학적 분석을 수행하는 기업이다. 최근 구글에서는 알파폴드(Alphafold) 시스템을 이용해 인공지능으로 단백질 구조 분석이 가능하도록 했다. 인공지능은 감염 경로 예측 및 조기 경보로 감염 확산을 저지하고, 바이러스 치료법 개발을 돕고 있다.",
    "url": "http://www.kwnews.co.kr/nview.asp?aid=221011900000"
  },
  {
    "id": 43,
    "type": "conference",
    "title": "Predicting Revenues of Seoul Commercial Alley using Neural Tensor Factorization",
    "authors": "Minkyu Kim and Suan Lee",
    "venue": "Big Data and Smart Computing (BigComp), 2021 IEEE International Conference on. IEEE, 2021.",
    "date": "January 17-20, 2021",
    "abstract": "Those who want to start their own businesses must decide a location and service to start. In order to make the decision, they must know characteristics of the location and service, such as average revenues and floating population. However, it is usually very difficult to collect and analyze these characteristics. Therefore, we propose a novel deep learning model named Neural Tensor Factorization (NeuralTF) that automatically analyzes the characteristics for predicting revenues, and a method for recommending appropriate location or service to start their businesses based on the predicted revenues. NeuralTF is a combination of Tensor Factorization(TF) and Deep Neural Network(DNN). We compare NeuralTF with other machine learning models using Seoul Commercial Alley dataset. In addition, we compare performances of NeuralTF when TF and DNN components share the embedding space and when they do not.",
    "keywords": "Recommender System, Tensor Factorization, Neural Network, Deep Learning, Neural Tensor Factorization",
    "url": "https://ieeexplore.ieee.org/document/9373114"
  },
  {
    "id": 44,
    "type": "conference",
    "title": "Comparison and Analysis of Embedding Methods for Patent Documents",
    "authors": "Arousha Haghighian Roudsari, Jafar Afshar, Suan Lee, and Wookey Lee",
    "venue": "Big Data and Smart Computing (BigComp), 2021 IEEE International Conference on. IEEE, 2021.",
    "date": "January 17-20, 2021",
    "abstract": "Patent text mining is an important task that requires domain knowledge. The patent text is sometimes not clear and contains many ambiguous and technical words. Traditional text mining approaches are not satisfactory enough for patent text mining. In this paper, we consider various embedding techniques for patent documents and try to find how to represent the patent text for other downstream tasks such as patent classification, patent recommendation, finding similar patents, knowledge mining, etc. We compare several embedding approaches with the patent classification task. The experimental results demonstrate that using contextual word embeddings can perform better than the conventional static word embedding approaches.",
    "url": "https://ieeexplore.ieee.org/document/9373099"
  },
  {
    "id": 600,
    "type": "conference",
    "title": "CHNE: Context-aware Heterogeneous Network Embedding",
    "authors": "Jihyeong Park, Suan Lee, and Jinho Kim",
    "venue": "2021 IEEE International Conference on Big Data and Smart Computing (BigComp)",
    "date": "January 17-20, 2021"
  },
  {
    "id": 601,
    "type": "conference",
    "title": "Multilingual Speech Synthesis for Voice Cloning",
    "authors": "Jiwon Seong, WooKey Lee, and Suan Lee",
    "venue": "2021 IEEE International Conference on Big Data and Smart Computing (BigComp)",
    "date": "January 17-20, 2021"
  },
  {
    "id": 602,
    "type": "conference",
    "title": "Combining Multiple Implicit-Explicit Interactions for Regression Analysis",
    "authors": "Minkyu Kim, Suan Lee, and Jinho Kim",
    "venue": "2020 IEEE International Conference on Big Data (IEEE BigData 2020)",
    "date": "December 2020",
    "badge": "[우수국제학술대회]"
  },
  {
    "id": 603,
    "type": "conference",
    "title": "ALTIBASE CEP: Real-Time Event Processing Engine",
    "authors": "Suan Lee, Jaenam Choi, Won Seo, Younghun Kim, Joonho Park, Kwangik Seo, and Nacwoo Kim",
    "venue": "Proceedings of the Fifth International Conference on Emerging Databases: Technologies, Applications, and Theory. ACM",
    "date": "2013"
  },
  {
    "id": 45,
    "type": "journal",
    "title": "Distributed graph cube generation using Spark framework",
    "authors": "Seok Kang, Suan Lee, and Jinho Kim",
    "venue": "Cluster Computing",
    "date": "October 2020",
    "badge": "[SCIE]",
    "impact": "(IF: 2.469, JCR: Q2)",
    "abstract": "Graph OLAP is a technology that generates aggregates or summaries of a large-scale graph based on the properties (or dimensions) associated with its nodes and edges, and in turn enables interactive analyses of the statistical information contained in the graph. To efficiently support these OLAP functions, a graph cube is widely used, which maintains aggregate graphs for all dimensions of the source graph. However, computing the graph cube for a large graph requires an enormous amount of time. While previous approaches have used the MapReduce framework to cut down on this computation time, the recently developed Spark environment offers superior computational performance. To leverage the advantages of Spark, we propose the GraphNaïve and GraphTDC algorithms. GraphNaïve sequentially computes graph cuboids for all dimensions in a graph, while GraphTDC computes them after first creating an execution plan. We also propose the Generate Multi-Dimension Table method to efficiently create a multidimensional graph table to express the graph. Evaluation experiments demonstrated that the GraphTDC algorithm significantly outperformed Spark SQL’s built-in library DataFrame, as the size of graphs increased.",
    "keywords": "Distributed parallel processing, Spark framework, Resilient distributed dataset, Graph cube, Data cube, Online analytical processing",
    "url": "https://link.springer.com/article/10.1007%2Fs11227-019-02746-4"
  },
  {
    "id": 46,
    "type": "journal",
    "title": "MRTensorCube: tensor factorization with data reduction for context-aware recommendations",
    "authors": "Svetlana Kim, Suan Lee, Jinho Kim, and Yong-Ik Yoon",
    "venue": "The Journal of Supercomputing",
    "date": "October 2020",
    "badge": "[SCIE]",
    "impact": "(IF: 2.469, JCR: Q2)",
    "abstract": "Context information can be an important factor of user behavior modeling and various context recognition recommendations. However, state-of-the-art context modeling methods cannot deal with contexts of other dimensions such as those of users and items and cannot extract special semantics. On the other hand, some tasks for predicting multidimensional relationships can be used to recommend context recognition, but there is a problem with the generation recommendations based on a variety of context information. In this paper, we propose MRTensorCube, which is a large-scale data cube calculation based on distributed parallel computing using MapReduce computation framework and supports efficient context recognition. The basic idea of MRTensorCube is the reduction of continuous data combined partial filter and slice when calculating using a four-way algorithm. From the experimental results, it is clear that MRTensor is superior to all other algorithms.",
    "keywords": "Context awareness, Tensor data cube, MapReduce framework",
    "url": "https://link.springer.com/article/10.1007/s11227-017-2002-1"
  },
  {
    "id": 47,
    "type": "column",
    "title": "[테크인사이드]이력서 작성·레시피 제공 다양하게 활용되는 GPT3",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2020년 10월 7일",
    "abstract": "최근 인공지능 분야에서 언어 생성 모델인 GPT3(Generative Pretrained Transformer 3)가 등장해 화제가 되고 있다. GPT 모델은 테슬라와 스페이스 X CEO인 일론 머스크(Elon Reeve Musk)가 창립한 비영리 연구단체 Open AI에서 만들어졌다. GPT3는 GPT2에 비해 무려 100배 정도 많은 1,750억개의 파라미터를 가진 모델이고, 인터넷에서 얻은 3,000억개 이상의 대량 데이터를 사전 학습해 만들어졌다. GPT3가 공개된 이후 기발하고 재미있게 활용한 사례들이 공개돼 많은 사람의 주목을 받고 있다.<br /> GPT3는 언어 생성 모델이라서 기본적으로 자연어를 다루는 일에 잘 활용될 수 있다. 간단하게는 문법을 체크하는 것, 필요한 단어들을 문장 앞뒤 맥락을 보고 추천하거나 다른 단어를 사용해 재구성해 주는 것이 있다. 그리고 다른 언어로 번역을 하거나 동시통역을 하는 등의 활용이 가능하다. 게다가 필요한 주요 내용만 적어주면 인사말과 격식을 갖춘 이메일 내용을 작성해주고, 인간이 작성한 건지 AI가 작성한 건지 구분할 수 없을 정도의 뉴스 기사, 소설과 시를 작성해준다.<br /> 버클리대 학생이 GPT3로 작성한 기사들을 올린 블로그를 2주간 운영했던 사례가 있다. 해당 기사들은 인간이 작성한 글과 구분하기 어려웠을 뿐만 아니라 IT 뉴스 서비스인 해커뉴스에서 1위를 차지할 정도로 인기를 얻었다. GPT는 대량의 인터넷 문서를 기반으로 질의응답을 하거나 내용을 보고 퀴즈를 만들어 주고, 이력서를 작성해주거나 해석이 어려운 법률용어 분석을 도와주기까지 한다. 게다가 음식 재료를 입력하면 요리 레시피를 알려주는 재미있는 활용도 가능하다.<br /> GPT3는 자연어 처리뿐만 아니라 다양한 분야에서의 활용도 가능하다. 먼저 수행해야 할 방법을 설명해주면 실행 가능한 코드를 작성해주거나 실제 작동하는 앱을 만들어준다. 만들고 싶은 웹페이지에 관해 설명하면 적합한 레이아웃을 구성하고 디자인이 된 웹사이트를 생성해준다.<br /> GPT3에는 많은 장점이 있지만, 단점이 없는 것은 아니다. 분별하기 어려운 가짜뉴스를 생성해 낼 수 있고, 인터넷에 존재하는 데이터가 편향돼 있어서 GPT3도 편견이 반영된 결과를 낼 수 있다. GPT3가 대량의 데이터를 학습했지만, 무엇을 하고 있는지 이해했다고 보기는 어렵다. 아직은 인공지능이 인간의 지능에 도달하기에는 더 많은 연구와 기술적 발전이 필요하다.",
    "url": "http://www.kwnews.co.kr/nview.asp?s=401&aid=220100600056"
  },
  {
    "id": 48,
    "type": "dconference",
    "title": "Face detection을 위한 RetinaFace 기반 대용량 학습 데이터 자동 레이블링 프레임워크",
    "authors": "이효준, 이수안, 이우기",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2020년 09월",
    "abstract": "컴퓨팅 파워의 발전과 새로운 알고리즘들의 연구 결과들로 큰 규모의 딥러닝 모델들이 많이 등장했다. 위 모델들을 학습하기 위해서는 많은 양의 데이터는 필수적이다. 특히 컴퓨터 비전 (Computer vision) 분야의 경우 좋은 성능은 결국 학습 데이터의 수에 직결되는 추세를 보인다. 학습 데이터를 구축하기 위해서는 해당 데이터에 대한 레이블링이 필수적이나, 사람이 직접 레이블링을 일일이 하는 것은 많은 비용을 필요로 하기에 딥러닝 모델의 성능을 올리는 데 있어서 상당 부분 장애물이 되었다. 본 논문에서는 이러한 문제점을 해결하기 위해 기존의 InsightFace의 RetinaFace를 활용하여 레이블링이 되지 않은 대용량 데이터를 받아 영역 검출 후 자동 레이블링 방식을 제안한다.",
    "url": "https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10529697"
  },
  {
    "id": 49,
    "type": "djournal",
    "title": "그래프 구조를 갖는 서지 빅데이터의 효율적인 온라인 탐색 및 분석을 지원하는 그래픽 인터페이스 개발",
    "authors": "유영석, 박범준, 조선화, 이수안, 김진호",
    "venue": "한국빅데이터학회지, 한국빅데이터학회",
    "date": "2020년 08월",
    "badge": "[KCI]",
    "abstract": "최근 다양한 실세계의 복잡한 관계를 그래프의 형태로 구성하고 분석하는 다양한 연구들이 진행되고 있다. 특히 DBLP와 같은 컴퓨터 분야 문헌 데이터 시스템은 논문의 저자, 그리고 논문과 논문들이 서로 인용 관계로 표현되는 대표적인 그래프 데이터이다. 그래프 데이터는 저장 구조 및 표현이 매우 복잡하므로, 문헌 빅데이터의 검색과 분석, 그리고 시각화는 매우 어려운 작업이다. 본 논문에서는 문헌 빅데이터를 그래프의 형태로 시각화한 그래픽 사용자 인터페이스 도구, 즉 EEUM을 개발하였다. EEUM은 그래프 데이터를 시각적으로 표시하여 연결된 그래프 구조에 따라 문헌 데이터를 브라우징 하는 기능을 제공하며, 문헌 빅데이터에 대한 검색 및 관리, 분석이 가능하도록 구현하였다. 또한 EEUM을 DBLP가 제공하는 문헌 그래프 빅데이터에 적용하여 편리하게 검색, 탐색 및 분석하는 할 수 있음을 시연한다. EEUM을 이용하여 모든 연구 분야에서 영향력 있는 저자나 논문을 쉽게 찾을 수 있으며, 여러 저자와 논문 사이의 모든 관계를한 눈에 볼 수 있는 등 복잡한 문헌 그래프 빅데이터의 검색 및 분석 도구로 편리하게 사용할 수 있다. Recently, many researches habe been done to organize and analyze various complex relationships in real world, represented in the form of graphs. In particular, the computer field literature data system, such as DBLP, is a representative graph data in which can be composed of papers, their authors, and citation among papers. Becasue graph data is very complex in storage structure and expression, it is very difficult task to search, analysis, and visualize a large size of bibliographic big data. In this paper, we develop a graphic user interface tool, called EEUM, which visualizes bibliographic big data in the form of graphs. EEUM provides the features to browse bibliographic big data according to the connected graph structure by visually displaying graph data, and implements search, management and analysis of the bibliographc big data. It also shows that EEUM can be conveniently used to search, explore, and analyze by applying EEUM to the bibliographic graph big data provided by DBLP. Through EEUM, you can easily find influential authors or papers in every research fields, and conveniently use it as a search and analysis tool for complex bibliographc big data, such as giving you a glimpse of all the relationships between several authors and papers.",
    "keywords": "그래프 데이터, 문헌 빅데이터, 그래픽 인터페이스, 시각화 도구 Graph Data, Bibliographic Big Data, Graphic Interface, Visualization Tool",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002623960"
  },
  {
    "id": 50,
    "type": "conference",
    "title": "Patent Prior Art Search using Deep Learning Language Model",
    "authors": "Dylan Myungchul Kang, Charles Cheolgi Lee, Suan Lee, and Wookey Lee",
    "venue": "24th International Database Engineering & Applications Symposium (IDEAS 2020)",
    "date": "August 2020",
    "abstract": "A patent is one of the essential indicators of new technologies and business processes, which becomes the main driving force of the companies and even the national competitiveness as well, that has recently been submitted and exploited in a large scale of quantities of information sources. Since the number of patent processing personnel, however, can hardly keep up with the increasing number of patents, and thus may have been worried about from deteriorating the quality of examinations. In this regard, the advancement of deep learning for the language processing capabilities has been developed significantly so that the prior art search by the deep learning models also can be accomplished for the labor-intensive and expensive patent document search tasks. The prior art search requires differentiation tasks, usually with the sheer volume of relevant documents; thus, the recall is much more important than the precision, which is the primary difference from the conventional search engines. This paper addressed a method to effectively handle the patent documents using BERT, one of the major deep learning-based language models. We proved through experiments that our model had outperformed the conventional approaches and the combinations of the key components with the recall value of up to '94.29%' from the real patent dataset.",
    "keywords": "Prior Art Search, Patent Document Classification, Language Model",
    "url": "https://dl.acm.org/doi/abs/10.1145/3410566.3410597"
  },
  {
    "id": 51,
    "type": "dconference",
    "title": "Generative Adversarial Networks 기반 개인정보 보호를 위한 얼굴 비식별화",
    "authors": "조시헌, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월",
    "abstract": "스마트시티가 활성화되고 도시의 치안 문제와 개인시설의 보안 문제 등으로 방범용 CCTV, IP 카메라 등이 지속적으로 증가하고있다. 그로 인해 영상 데이터들이 계속 많아지고, 이와함께 개인정보 유출에 대한 위험성도 커지고 있다. 개인정보는 주민등록번호, 이름, 주소 등 문서나 의료 데이터에서만 유출되는 것이 아닌 영상, 이미지에서도 유출된다. 본 논문에서는 영상에서 검출되는 얼굴을 비식별화하기 위해서 얼굴을 GAN(Generative Adversarial Network)으로 생성한 얼굴과 바꾸는 모델을 제안한다. 이미지에서 개인정보로 쉽게 알아볼 수 있는 얼굴을 탐지 후 GAN을 통하여 생성한 임의의 얼굴을 해당 이미지 내의 얼굴로 대체한다. 비식별화된 이미지 데이터들은 다른 영상관련 딥러닝 훈련에 개인정보 유출 없이 사용 될 수 있다. 본 논문에서 제시한 비식별화 기술을 통하여 영상이나 이미지 내에 개인 정보의 유출을 막을 수 있다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09874643"
  },
  {
    "id": 52,
    "type": "dconference",
    "title": "벡터화 경로 데이터에 노이즈 추가를 통한 프라이버시 보호 방안: 코로나 바이러스를 중심으로",
    "authors": "박수경, 라시드, 이철기, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월",
    "abstract": "최근 코로나 바이러스가 전세계적으로 확산됨에 따라 이를 차단하기 위한 많은 노력이 계속되고 있다. 국내에서는 COVID-19 확진 환자의 이동경로를 감염관리에 활용하고 있으며, 이 때 개인 정보 보호 차원에서 거주지 주소와 같은 세부 정보는 공개하지 않는다. 이와 같은 제한된 정보 공개는 국민들의 자가 감염 예방 활동이 제약되는 난점뿐만 아니라 여전한 개인 사생활 침해 가능성이란 문제점을 가진다. 본 연구에서는 이를 해결하기 위해 개인을 식별할 수 있는 위치 정보를 삭제하지 않고도 해당 데이터에 노이즈를 추가하는 익명화 방식을 통해 결과적으로 프라이버시를 보호할 수 있는 모델을 제안한다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09874634"
  },
  {
    "id": 53,
    "type": "dconference",
    "title": "적대적 공격을 이용한 객체 탐지 모델의 취약성 검증",
    "authors": "이효준, 이철기, 이수안, 이우기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2020년 07월",
    "abstract": "딥러닝 모델들은 높은 정확도로 인해 영상, 음성 및 자연어 처리 등 다양한 분야에 적용되며 산업에 활용 되고 있다. 이에 따라 딥러닝 모델에 대한 악의적인 공격에 관한 연구도 활발히 이루어지고 있다 .특히, 컴퓨터 비전 분야에서 영상 또는 이미지의 특성상 작은 노이즈에도 민감하게 반응할 수 있어 적대적 공격(Adversarial Attack)에 취약 하다. 만약 주민등록증이나 신용카드와 같은 개인정보를 포함하는 자료에 해당 공격이 가해진다면 다양한 인적, 물적 피해를 유발할 수 있기에 관련된 연구가 필요하다. 본 논문에서는 개인정보 자료들을 항한 적대적 공격의 유효성을 검증하기 위하여 개인 민감정보로서 쉽게 사용되어질 수 있는 차량 번호판에 적용하였다. 다양한 적대적 공격 방식을 대상으로 적대적 예제(Adversarial Example)들을 생성하였으며, 이를 통한 객체 탐지 모델의 오작동이 생길 수 있음을 검증하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09874643"
  },
  {
    "id": 54,
    "type": "dconference",
    "title": "선행기술조사를 위한 딥러닝 언어 모델 기반 특허 문서 분류",
    "authors": "강명철, 이철기, 이수안, 이우기",
    "venue": "2020 춘계학술대회, 한국지식재산교육연구학회",
    "date": "2020년 06월",
    "abstract": "전 세계적으로 최근 신기술에 대한 특허가 다량으로 출원되는 상황이지만, 특허 심사를 처리하는 인력은 증가하는 특허 수를 따라가지 못하고 있다. 이는 특허 심사 품질을 감소시키며 그와 동시에 개인 및 기업, 더 나아가 국가의 신뢰성 저하 및 경제적 손실을 야기한다. 특히, 선행기술조사는 제한된 시간 안에서 노동집약적 고비용 작업이기 때문에 최근 급속도로 발전하고 있는 딥러닝을 적용하기에 매우 적합한 대상이 된다. 본 논문을 통해 딥러닝 기반 언어 모델 중 하나인 매우 우수한 성능을 보이는 BERT에 기반하여 선행기술조사 때에 대용량 특허 문서에서 유효 특허를 효과적으로 찾는 방법을 제안하였으며, 실험을 통해 그 성능의 우월성을 입증하였다.",
    "keywords": "딥러닝, 인공지능, 문서 분류, 선행기술조사"
  },
  {
    "id": 55,
    "type": "column",
    "title": "[테크인사이드]인공지능의 보안 위협",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2020년 5월 20일",
    "abstract": "인공지능이 산업 전반에 영향을 미치며 기존 기술을 대체하거나 지능화되고 있다. 이러한 변화는 인공지능을 이용한 악의적인 공격과 개인정보 침해, 범죄의 지능화에도 영향을 주고 있다. 대표적으로 인공지능 발전의 핵심인 딥러닝 기술을 이용해 가짜를 만들어 내는 딥페이크(Deepfake) 기술이 있다.<br />\\딥페이크 기술을 활용한 예로는 성인비디오 출연 배우 얼굴을 연예인 등 유명인의 얼굴로 교체하거나 가짜 기사를 만들어 사회적인 혼란 조성, 특정 기업의 대표와 유사한 합성 음성을 만들어 공급사에 돈을 보내게 한 사례 등이 있다. 최근에는 구글 Vision AI가 체온계를 들고 있는 사람의 피부가 밝을 때는 체온계로 판단한 반면, 피부가 어두울 때는 총으로 인식해 논란이 된 적이 있다. 데이터를 통해 학습된 딥러닝 모델은 인간보다 뛰어난 결과들을 보여주고 있지만, 데이터 속 편견 또한 같이 학습해 편향된 결과를 내놓기도 한다. 마이크로소프트사에서 2016년 사람과 대화가 가능한 인공지능 챗봇 테이(Tay)를 개발해 공개했지만, 일부 사용자가 테이가 악의적 발언을 하도록 훈련시켰고 그 영향으로 테이가 욕설, 인종차별, 성차별 등을 하게 됨으로써 16시간 만에 운영이 중단된 바도 있다. 이렇게 의도적으로 악의적인 데이터를 학습시켜 영향을 주는 기법을 `중독 공격(Poisoning attack)'이라고 한다.<br /> 한편, 인공지능의 보안 위험이 발생하도록 모델에 내재된 취약점을 공격하는 `적대적 공격(Adversarial attack)'이라는 개념이 있다. 예를 들면 판다 이미지에 사람의 눈으로는 식별되지 않는 노이즈(Noise)를 추가해 긴팔원숭이로 인식하게 만드는 것이다. 적대적 공격 중 하나인 적대적 예제(Adversarial example)는 노이즈 데이터를 생성하고 학습시켜 모델이 잘못 인식하게 만든다. 적대적 스티커(Adversarial patch)는 바나나 옆에 특정 스티커를 붙여 토스터로 인식하게 만들거나, 도로 교통표지판에 스티커를 붙여 `정지' 표시를 `속도제한' 표시로 인식하게 만드는 것이다.<br /> 인공지능 보안 위협에는 모델이 잘못 인식하거나 오작동하게 만드는 공격뿐만 아니라, 수많은 질의를 통해 나온 모델의 결과를 분석해 모델 학습에 사용된 데이터를 추출하는 `전도 공격(Inversion attack)'이 있고, 질의 결과 분석을 통해 기존 모델과 유사한 모델을 만들도록 하는 `모델 추출 공격(Model extraction attack)'도 있다. 이러한 공격으로 학습 데이터에 기밀정보나 개인정보, 민감정보 등이 포함된 경우 데이터 유출 위험이 발생하고, 기업이 생성한 모델 자체가 추출된 경우 기업 자산 노출로 피해를 입게 된다. 산업 전반에 인공지능 기술이 적용되고 있는 만큼 인공지능이 가지는 취약점과 보안 위협을 염두에 둬야 한다. 인공지능 기술을 발전시키고 활용하는 것과 더불어 학습 데이터가 오염되지 않도록 관리하고, 취약점이 보완된 견고한 모델을 만들도록 해야만 인간의 삶을 이롭고 윤택하게 해줄 안전한 인공지능 세상이 가능할 것이다.",
    "url": "http://www.kwnews.co.kr/nview.asp?s=401&aid=220051900034"
  },
  {
    "id": 56,
    "type": "djournal",
    "title": "데이터이동권 도입을 위한 비교법적 연구",
    "authors": "고수윤, 이수안, 손경호",
    "venue": "과학기술법연구, 한남대학교 과학기술법연구원",
    "date": "2020.05.01",
    "badge": "[KCI]",
    "abstract": "우리나라는 해외 주요국과 같이 개인정보에 대한 정보주체의 권리를 보장하고자 마이데이터 정책을 점차 확장하고 있다. 마이데이터는 정보주체 중심의 데이터 생태계를 형성하는 것을 목적으로 하고 있으며 데이터 이동에 대한 통제권을 정보주체에게 보장함으로써 이를 달성하고자 하였다. 데이터이동권을 처음으로 규정한 EU GDPR은 공정한 시장경쟁을 위한 것을 목적으로 하고 있으며 특히 데이터이동권은 정보주체가 특정 서비스에 락인(lock-in)되는 현상을 방지하여 다른 서비스로 이전할 수 있도록 보장하고자 한 권리이다. 실제 거대 IT기업들이 정보유통시장을 과점하는 것을 방지하려는 목적이 있었다. 동일한 이유로, 단일화되고 있는 디지털 시장에서 우리나라 기업의 유효경쟁을 보장하고 정보주체의 서비스 선택에 자율성을 보장하기 위해 데이터이동권을 보장하는 것이 필요하다. 우리나라에서 데이터이동권은 데이터 이동에 대한 정보주체의 통제권을 의미한다는 점에서 데이터를 받을 수 있는 권리, 동의를 결정할 권리, 데이터 제공을 요청할 권리라는 모든 경우에 해당될 수 있다. 하지만 데이터이동권을 처음으로 명시한 EU GDPR의 규정에 따르면 데이터이동권은 정보주체가 개인정보를 기계판독이 가능한 포맷으로 제공받을 권리이다. 기계판독이 가능한 포맷 또는 상호운용성 있는 포맷의 보장은 해당 데이터가 이전받은 컨트롤러에 의한 이용가능성을 보장하고자 하는 의도가 있다. 우리나라 데이터이동권의 권리범위도 데이터 활용을 보장하며 정보주체가 데이터의 제공을 적극적으로 요청할 수 있는 권리로 보는 것이 타당할 것이다. 개인정보 일반에 대하여는 해당 규정이 존재하지 않으므로 이를 보장하기 위해서는 규정의 신설이 필요하다. 구체적으로 데이터이동권 규정을 도입하기 위하여 권리제한사유, 데이터 제공자, 제공자의 의무, 대상 데이터의 범위, 정보제공방법, 데이터 이동 요청에 따른 처리기간, 처리비용에 대하여 해외 주요국의 규정과 비교검토해 본다. Korea is gradually expanding its MyData policy to ensure the right of the data subject to personal data like major foreign countries. My Data aims to form a data ecosystem centered on the data subject, and aims to achieve this by ensuring control over data portability by the data subject. The EU GDPR, which first stipulated the right to data portability, aims to promote fair market competition. In particular, the right to data portability is intended to prevent data subjects from being locked in to a specific service and to be able to transfer to another service. Indeed, it generated to prevent IT giants from dominating the information distribution market. For the same reason, it is necessary to guarantee the right to data portability in order to ensure the effective competition of Korean companies in a single digital market and to ensure the autonomy in the choice of services by data subjects. In Korea, the right to data portability can refer to all cases of the right to receive data, the right to decide consent, and the right to request data in that it means the right to control the data subject. However, according to the EU GDPR, the right to data portability is the right of the data subject to receive personal data in a machine-readable format. The guarantee of a machine-readable format or an interoperable format is intended to guarantee the availability of the transferred data by a controller. It would be appropriate to view the right scope of data portability in Korea as the right to guarantee the utilization of data and to actively request the controller to provide the personal data. In Korea, there is no provision for the right to data portability, it is necessary to establish a new provision to ensure this. Specifically, this paper compares the reasons for restriction of the right, the data provider, the obligations of the providers, the scope of the target data, the method of providing data, and the processing period and the processing cost according to the request for data porting with the regulations of foreign countries to introduce the provision for the right to data portability.",
    "keywords": "데이터이동권, GDPR, 디지털공화국법, 기업 및 규제개혁법, 마이데이터 Data Portability, GDPR, Digital Republic Act, Enterprise and Regulatory Reform Act, MyData",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002590817"
  },
  {
    "id": 57,
    "type": "column",
    "title": "[테크 인사이드]데이터 경제 시대",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2020년 3월 4일",
    "abstract": "데이터3법(개인정보보호법, 신용정보법, 정보통신망법 개정안)이 국회 본회의를 통과했다. 이제 기업들은 개인정보가 포함된 빅데이터를 비식별화한 가명정보를 이용해 개인의 동의 없이도 통계·연구·기록 보존 등의 목적으로 활용이 가능해졌다. 다양한 분야에서 가명정보 활용을 통해 데이터 이동·융합 등이 활성화되면 데이터 경제 시대를 이뤄 나갈 수 있다.<br> 금융업계에서는 데이터 3법을 통해 다양한 서비스 발굴과 확장이 가능해졌다. 기존 자산관리 앱들은 공인인증서나 로그인 정보를 이용, 개별 금융사에 존재하는 입출금 내역, 소비정보, 자산·부채 등의 정보를 스크래핑해 긁어모았다. 그러나 이제는개인이 사용하는 금융 정보를 쉽게 이동하고 융합하며 자신의 신용정보를 통합적으로 관리하고 세분화할 수 있다. 통합된 금융 데이터로 좀 더 세밀한 맞춤형 투자나 대출, 금융상품 추천 등에 활용이 가능하다. 오랫동안 빅데이터를 축적해 온 금융업계에서는 다른 외부 데이터와의 결합과 융합을 통해 새로운 신산업 육성이 가능할 수 있다.<br> 의료 분야에서는 지금까지 활용하기 어려웠던 환자 데이터를 연구 목적이라면 동의 없이 활용이 가능해졌다. 의료·헬스케어 분야에서는 환자 인적사항, 발병, 병원 진료, 검진 결과, 처방 등의 데이터를 통합 분석해 새로운 의료 서비스가 탄생할 수 있다. 각종 의료 데이터를 상호 공유하고, 건강 정보를 통해 정밀의료, AI 진단과 같은 혁신적인 서비스가 가능하며, 국내 의료·바이오·헬스케어 분야의 산업 경쟁력 향상이 기대된다. 개인들은 고도화되고 체계적인 의료 서비스와 맞춤형 헬스케어 서비스, 건강보험 상품의 추천 등을 받을 수 있다.<br> 통신 분야에서는 이미 통신사업자들이 가입자 정보, 음성, 데이터 사용량, 위치, 네트워크 등의 다양한 정보를 축적하고 있다. 스마트폰에서 발생하는 데이터는 다양한 외부 데이터와 융합이 가능하며, 그로 인해 발생할 가치와 이익은 매우 크다. 그리고 통신 이용자의 사용 패턴과 성향을 분석해 맞춤형 서비스가 가능하다.<br> 데이터3법의 통과로 데이터 경제 시대는 한 발자국 더 다가왔다. 그러나 아직 해결해야 할 많은 문제가 남아 있다. 먼저 개인정보가 판매나 공유가 가능하면 개인정보 보호 문제는 더욱 커진다. 그리고 개인정보를 외부 데이터와 결합하며 개인이 재식별되는 문제 등도 있다. 이미 유출된 개인정보와 결합될 경우 범죄나 보이스 피싱, 스팸과 같은 악의적인 용도로 사용돼 피해가 커질 수 있다. 데이터 경제 시대가 자기 데이터의 주권과 결정권을 갖고 살아가는 데이터 민주주의 속에서 개인정보 보호와 안전한 활용으로 발전되길 기대해본다.",
    "url": "http://www.kwnews.co.kr/nview.asp?s=401&aid=220030300001"
  },
  {
    "id": 58,
    "type": "conference",
    "title": "A Wide & Deep Learning Sharing Input Data for Regression Analysis",
    "authors": "Minkyu Kim, Suan Lee, and Jinho Kim",
    "venue": "Big Data and Smart Computing (BigComp)",
    "date": "February 19-22, 2020",
    "abstract": "Wide & Deep model is deep learning model that jointed between of wide component and deep component for regression, recommendation, and classification. However, there is no study of regression analysis using Wide & Deep model. Also, Wide component of Wide & Deep model only deal with categorical variables and that need hand-crafted variables for efficient training. Therefore, this paper propose Lattice Wide & Deep Architecture which improve Wide & Deep model. Furthermore we show that the Lattice Wide & Deep model has better performance than Wide & Deep model in regression analysis.",
    "keywords": "lattice wide & deep, wide & deep learning, regression analysis",
    "url": "http://bigcomputing.org/conf2020/"
  },
  {
    "id": 59,
    "type": "conference",
    "title": "Estimating Revenues of Seoul Commercial Alley Services using Tensor Decomposition & Generating Recommendation System",
    "authors": "SungJin Park, Suan Lee, and Jinho Kim",
    "venue": "Big Data and Smart Computing (BigComp)",
    "date": "February 19-22, 2020",
    "abstract": "As for people who wish to start their own businesses, their concerns are whether they could survive during their operations because most stores or services in Seoul are not able to survive for more than 5 year possibly due to the poor decision of location/service to start. In order to solve this problem, using big data could be helpful to increase the survival rate. Singular Value Decomposition (SVD) has been widely used in finding the similarity between all pairs of alleys and obtaining predictions from unknown relevance scores. Since tensor decomposition is the extension of SVD for multi-dimensional data, using this method to find the similarity between all pair of alleys could be the solution of increasing survival rate. This paper aims to generate good prediction tensor, TENSORCABS, that is able to recommend users appropriate alley location to start their businesses or the appropriate services to start at the user’s desired location. Both CP and Tucker decompositions are used and compared to evaluate which method has better performance. Also, r-square for regression problem and precision & recall for top-k recommendation performance are used to evaluate the TENSORCABS. As results, actual and predicted values are good-fitted, and prediction tensor performs well on the top-k recommendation. In addition, Tucker outperforms CP for this situation. Therefore, the proposed method has advantages that can handle high-dimensional data and can use decomposition for recommendations of various perspectives. With this method, users are able to obtain recommendations of the appropriate alleys with predicted revenues for opening any business service or the appropriate services with predicted revenues on a user’s desired alley location.",
    "keywords": "Recommendation System, Tucker Decomposition, Canonical Polyadic, Collaborative Filtering, Top-k Recommendation, Tensor, Commercial Alley",
    "url": "http://bigcomputing.org/conf2020/"
  },
  {
    "id": 60,
    "type": "conference",
    "title": "Research Issues on Generative Adversarial Networks and Applications",
    "authors": "Toshpulatov Mukhiddin, WooKey Lee, Suan Lee, Tojiboev Rashid",
    "venue": "The Second International Workshop on Big Data, Cloud, and IoT Technologies for Smart Cities (IWBigDataCity2020)",
    "date": "February 19, 2020",
    "url": "http://iwbigdatacity.org/"
  },
  {
    "id": 61,
    "type": "conference",
    "title": "Prior Art Search Using Multi-Modal Embedding of Patent Documents",
    "authors": "Myungchul Kang, Suan Lee, and Wookey Lee",
    "venue": "The 3rd International Workshop on Dialog Systems (IWDS2020) 2020.",
    "date": "February 19, 2020",
    "url": "https://sigai.or.kr/workshop/bigcomp/2020/iwds/"
  },
  {
    "id": 62,
    "type": "conference",
    "title": "De-identification and Privacy Issues on Bigdata Transformation",
    "authors": "Allen Hyojun Lee, Steve Siheon Cho, Jessica Jiwon Seong, Suan Lee, Wookey Lee",
    "venue": "Industrial Security - eGovernance Vision and Strategy (ISComp2020)",
    "date": "February 19, 2020",
    "url": "https://sites.google.com/view/iscomp2020/"
  },
  {
    "id": 63,
    "type": "conference",
    "title": "A Study on the Voice Security System Using Sensor Technology",
    "authors": "Jessica Jiwon Seong, Steve Siheon Cho, Allen Hyojun Lee, Suan Lee, Wookey Lee",
    "venue": "Industrial Security - eGovernance Vision and Strategy (ISComp2020)",
    "date": "February 19, 2020",
    "url": "https://sites.google.com/view/iscomp2020/"
  },
  // ============ Books (저서) ============
  {
    "id": 300,
    "type": "book",
    "title": "파이썬으로 만드는 나만의 게임",
    "authors": "이수안",
    "venue": "비제이퍼블릭",
    "date": "2022년 01월",
    "url": "https://www.yes24.com/Product/Goods/106326014"
  },
  {
    "id": 301,
    "type": "book",
    "title": "파이썬으로 텍스트 분석하기: 전략커뮤니케이션을 위한 텍스트 마이닝",
    "authors": "윤태일, 이수안",
    "venue": "늘봄",
    "date": "2019년 08월",
    "url": "https://www.yes24.com/Product/Goods/78919398"
  },
  // ============ Reports (보고서) ============
  {
    "id": 64,
    "type": "report",
    "title": "데이터 이동권 도입 방안 연구",
    "authors": "손경호, 이수안, 고수윤",
    "venue": "한국데이터산업진흥원",
    "date": "2019년 12월",
    "url": "https://www.kdata.or.kr/info/info_01_download.html?dbnum=431"
  },
  {
    "id": 604,
    "type": "report",
    "title": "빅데이터 시대에 대응한 교육정보·통계 정책과제 개발 연구",
    "authors": "김진호, 손대형, 이기준, 신혜숙, 이수안",
    "venue": "교육부 교육안전정보국 교육통계과",
    "date": "2017년 09월"
  },
  {
    "id": 605,
    "type": "report",
    "title": "클라우드 컴퓨팅을 활용한 비즈니스 인텔리전스",
    "authors": "이수안, 문양세, 김진호",
    "venue": "정보통신산업진흥원 주간기술동향 1445호",
    "date": "2010년"
  },
  // ============ Patents (특허) - 등록 7건 ============
  {
    "id": 400,
    "type": "patent",
    "title": "위치 이동 인식 기반 객체 정보 인식 방법",
    "authors": "오영, 이수안, 김재형, 권봉기",
    "venue": "대한민국 특허청 (등록번호: 1028754430000)",
    "date": "2025년 10월",
    "url": "https://doi.org/10.8080/1020190085640"
  },
  {
    "id": 401,
    "type": "patent",
    "title": "회귀분석용 와이드 앤 딥 모델 기반 학습 방법",
    "authors": "김민규, 이수안, 김진호",
    "venue": "대한민국 특허청 (등록번호: 1028748890000)",
    "date": "2025년 10월",
    "url": "https://doi.org/10.8080/1020190174909"
  },
  {
    "id": 402,
    "type": "patent",
    "title": "동작 인식과 마커 기반 객체 정보 인식 방법",
    "authors": "오영, 이수안, 김재형, 권봉기",
    "venue": "대한민국 특허청 (등록번호: 1028521550000)",
    "date": "2025년 08월",
    "url": "https://doi.org/10.8080/1020190085636"
  },
  {
    "id": 403,
    "type": "patent",
    "title": "WiFi CSI 기반의 행동 예측 모델 시스템 및 이의 실행 방법",
    "authors": "김재한, 진동민, 이수안",
    "venue": "대한민국 특허청 (등록번호: 1028520250000)",
    "date": "2025년 08월",
    "url": "https://doi.org/10.8080/1020240094911"
  },
  {
    "id": 404,
    "type": "patent",
    "title": "구조 변형 인식 기반 객체 정보 인식 방법",
    "authors": "오영, 이수안, 김재형, 권봉기",
    "venue": "대한민국 특허청 (등록번호: 1028325900000)",
    "date": "2025년 07월",
    "url": "https://doi.org/10.8080/1020190085633"
  },
  {
    "id": 405,
    "type": "patent",
    "title": "인공신경망을 이용한 이미지 변환 기반의 코골이 분류 방법",
    "authors": "최예신, 이수안",
    "venue": "대한민국 특허청 (등록번호: 1025483790000)",
    "date": "2023년 06월",
    "url": "https://doi.org/10.8080/1020220186688"
  },
  {
    "id": 406,
    "type": "patent",
    "title": "사용자를 위한 수면 큐레이션 서비스 시스템의 동작 방법",
    "authors": "최예신, 김인헌, 이경수, 이수안",
    "venue": "대한민국 특허청 (등록번호: 1025219810000)",
    "date": "2023년 04월",
    "url": "https://doi.org/10.8080/1020230007147"
  },
  // ============ Patents (특허) - 출원 4건 ============
  {
    "id": 407,
    "type": "patent",
    "title": "WiFi CSI 기반의 자세 분류 및 이상 상태 판단 시스템 및 그 방법",
    "authors": "김재한, 진동민, 이수안",
    "venue": "대한민국 특허청 (출원번호: 1020250118370)",
    "date": "2025년 08월",
    "badge": "[출원]",
    "url": "https://doi.org/10.8080/1020250118370"
  },
  {
    "id": 408,
    "type": "patent",
    "title": "WiFi CSI 적응형 학습 시스템 및 이의 실행 방법",
    "authors": "김재한, 진동민, 이수안",
    "venue": "대한민국 특허청 (출원번호: 1020250118369)",
    "date": "2025년 08월",
    "badge": "[출원]",
    "url": "https://doi.org/10.8080/1020250118369"
  },
  {
    "id": 409,
    "type": "patent",
    "title": "WiFi CSI 기반의 객체 위치 추정 시스템 및 이의 실행 방법",
    "authors": "김재한, 진동민, 이수안",
    "venue": "대한민국 특허청 (출원번호: 1020250118368)",
    "date": "2025년 08월",
    "badge": "[출원]",
    "url": "https://doi.org/10.8080/1020250118368"
  },
  {
    "id": 410,
    "type": "patent",
    "title": "열화상 인식 및 마커 기반 객체 정보 인식 방법",
    "authors": "오영, 이수안, 김재형, 권봉기",
    "venue": "대한민국 특허청 (출원번호: 1020190085643)",
    "date": "2019년 07월",
    "badge": "[출원]",
    "url": "https://doi.org/10.8080/1020190085643"
  },
  // ============ Columns (칼럼) ============
  {
    "id": 65,
    "type": "column",
    "title": "[테크 인사이드]마이데이터 시대의 도래 데이터 주권과 새로운 가치",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2019년 12월 25일",
    "abstract": "국민홈피로 불린 싸이월드가 접속이 불가능한 상태가 됐다가 되살아났다. 이 사태로 많은 사람이 혼란스러움을 표현했다. 보통 서비스가 종료되기 전에 개인 데이터들을 백업해 사용자가 다운로드할 수 있도록 하는 경우가 많지만, 결국 업체가 문을 닫게 되면 데이터는 모두 삭제된다.<br > 이전까지 정보는 독재되거나 통제되는 구조에서 기업 내 공유되거나 기업 간 공유되는 형태로 변화됐고, 최근에는 데이터 생태계 속에서 정보가 개방되는 구조로 발전돼 데이터 민주주의 시대를 이끌어 가고 있다. 세계 각국에서는 개인들이 자기 데이터의 주권과 결정권을 갖고자 하는 움직임이 있다. 이미 구글, 페이스북, 인스타그램 등 주요 소셜 플랫폼은 개인 데이터의 다운로드 및 이동이 가능하도록 제공하고 있다. 유럽연합(EU)에서는 GDPR(General Data Protection Regulation·일반 개인정보보호법)을 발표해 개인 데이터에 대한 권리를 부여하고 있다.<br > 데이터 주권 확보의 결과로 개인이 자신의 데이터에 대한 통제와 권리를 부여받아 자유롭게 활용할 수 있도록 한 개념으로 마이데이터(MyData)가 등장했다. 마이데이터는 기존의 데이터에 새로운 가치를 더하고, 다양한 서비스로 활용이 가능해진다. 금융 분야에서는 개인 거래 내역을 통합적으로 조회 및 지출 관리 그리고 투자 서비스까지 제공하는 것이 가능하다. 대표적인 국외 서비스로 민트(mint)가 있고, 국내에는 뱅크샐러드가 있다. 헬스케어 분야에서는 자신의 건강검진 기록과 병원의 검사 및 영상 기록 등을 통해 개인화된 정밀의료 서비스를 받을 수 있다. 그리고 개인의 주요 질환 발생 예측과 맞춤형 약 제조와 건강식품 추천 등을 지원받을 수 있게 된다.<br > 교육 분야에서는 마이데이터를 기반으로 개인이 제공하는 학력, 경력, 자격 정보 등을 통해 맞춤형 교육과 취업, 진학, 진로 추천 등에 대해서 서비스할 수 있다. 그리고 기업은 맞춤형 인재 양성이 가능해진다.<br > 마이데이터는 흩어져 있는 나의 데이터를 모아 통합 관리하고, 승인 및 제어에 대한 주권을 확보하는 것이며, 더 나아가 개인 데이터를 활용한 통합 맞춤형 서비스를 누릴 수 있다. 그리고 마이데이터는 각종 공공 데이터나 오픈 데이터와 연계 결합돼 빅데이터 분석에 활용될 수 있고, 머신러닝·딥러닝 모델을 통한 맞춤형 개인화 추천, 예측, 의사결정 지원이 가능해진다. 앞으로 마이데이터는 데이터 민주주의의 개인 주권 확보를 넘어 새로운 가치를 만들어 낼 것이다.",
    "url": "http://www.kwnews.co.kr/nview.asp?s=1201&aid=219122400022"
  },
  {
    "id": 66,
    "type": "dconference",
    "title": "딥러닝 기반 개인정보 보호 영상 세그멘테이션 및 복원 가능한 영상 왜곡 모델",
    "authors": "박지형, 김진호, 이수안",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2019년 12월",
    "badge": "(장려상)",
    "abstract": "최근 기술의 발전으로 지능형 CCTV같은 영상 수집 장치의 유용성이 대두되고 있다. 하지만 이러한 유용성에도 불구하고 개인 정보 침해 문제 때문에 영상 수집 장치의 설치나 보급에 어려움을 겪고 있다. 이러한 문제를 해결하기 위해 모자이크 등의 여러 기술이 존재하나 이러한 기술들은 필요시 복원이 불가능하거나 여러 문제점을 가지고 있다. 따라서 본 논문은 이러한 문제를 해결하기 위한 복원 가능한 영상 왜곡 모델을 제안하고, 이 구조의 특징인 왜곡 후 복윈 시 원본과 차이가 없다는 것과 시드를 통해 의도하지 않은 조작과 왜곡을 방지할 수 있는 것을 실험으로 증명하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09301969"
  },
  {
    "id": 67,
    "type": "dconference",
    "title": "로봇 판단지능을 위해 인간 작업 영상을 활용한 빅데이터 시스템 설계 및 구축",
    "authors": "장종원, 전호빈, 이수안, 김진호, 박홍성, 김미숙, 유수정, 지상훈",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2019년 12월",
    "abstract": "최근 빅데이터와 AI를 이용한 다양한 활용성이 주요한 이슈로 떠오르고 있다. 그중에는 수많은 영상을 데이터로써 입력하여 로봇이 영상 속 인간 작업자의 동작을 따라 하게 하는 연구 또한 존재한다. 본 논문에서는 인간 작업자의 작업과정에 대한 영상 데이터를 저장, 분류하고, 각 영상에 annotation을 추가하여 빅데이터 시스템에 저장한다. 빅데이터 시스템을 이용해 필요한 데이터를 질의하고 처리 분석하고, 판단지능을 위한 학습 데이터를 가져올 수 있다. 또한, 지속적인 판단지능 향상을 위해 학습된 모델의 결과도 빅데이터 시스템에 저장한다. 본 논문에서는 빅데이터 시스템의 설계와 저장구조 모델을 설명하며 활용성을 보인다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09301504"
  },
  {
    "id": 68,
    "type": "column",
    "title": "[테크인사이드]유튜브 탄생과 크리에이터 시대",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2019년 9월 25일",
    "abstract": "누구나 쉽게 동영상을 공유할 수 있는 서비스 플랫폼인 유튜브는 전 세계를 장악했다. 전 세계 인터넷 사용자가 약 30억명인데 그중 유튜브 사용자는 10억명이 넘으며 매월 로그인 횟수는 19억 회가 넘는다. 약 91개 국가에서 현지화 버전의 유튜브를 사용하고 있으며 80개가 넘는 언어를 통해 사용자가 원하는 언어로 영상을 편하게 시청할 수 있다.<br /> 유튜브는 2005년 2월에 스티브 첸(Steve Chen)이 채리 헐리(Cha Meredith Hurley), 조드 카림(Jawed Karim)과 함께 창업해 탄생했다. 다양하고 수많은 영상이 넘쳐나는 지금과 달리 당시 창업 후 2개월 뒤 유튜브에 조드 카림이 동물원에서 코끼리 코를 보며 감탄하는 `Me at the zoo'라는 제목의 19초짜리 동영상 하나가 처음 업로드됐다. 하지만 유튜브는 설립된 지 1년 만에 월간 2,000만명의 방문, 일간 1억 조회 수를 달성하며 빠르게 성장했다. 이에 구글은 2006년 10월 유튜브를 16억5,000만 달러에 인수 합병하면서 전 세계로 확산됐다.",
    "url": "http://www.kwnews.co.kr/nview.asp?aid=219092400039"
  },
  {
    "id": 69,
    "type": "djournal",
    "title": "Visual Cell: 바이오세포 이미지 빅데이터를 위한 이미지 분석 및 시각적 검색 시스템",
    "authors": "박범준, 조선화, 이수안, 신지운, 유혁상, 김진호",
    "venue": "한국빅데이터학회지, 한국빅데이터학회",
    "date": "2019년 9월",
    "badge": "[KCI]",
    "abstract": "주변 세포의 구조적, 생화학적 지지체를 제공하는 세포 외 기질은 세포의 분열과 분화 등을 좌우하는 세포 생리 조절인자이다. 바이오 분야에서는 3차원 조직공학 지지체인 스캐폴드를 제작하고, 제작한 스캐폴드에 줄기세포를 배양해 동물에 이식해 조직 재생력을 평가한다. 이는 조직 내 콜라겐과 같은 구성성분에 좌우된다. 따라서 조직 내 구성성분의 포함율 및 분포를 파악하는 것이 매우 중요한데, 이에 관한 데이터를 염색된 조직 이미지의 색상을 분석함으로써 얻어낸다. 이때 이미지 수집부터 분석까지의 과정이 적지 않은 비용이 소모되고 있고, 수집되고 분석된 데이터를 연구 기관마다 상이한 포맷으로 관리하고 있다. 따라서 데이터 통합관리 및 분석결과 검색 등이 이루어지지 않고 있다. 본 논문에서는 관련 빅데이터를 통합적으로 관리할 수 있는 데이터베이스를 구축하고, 이 연구 분야에서 중요한 분석 척도인 색상을 기준으로 검색할 수 있는 바이오 이미지 통합 관리 및 검색 시스템을 제안한다. The extracellular matrix, which provides the structural and biochemical support of surrounding cells, is a cell physiological modulator that controls cell division and differentiation. In the bio sector, the company produces Scapold, a three-dimensional support for tissue engineering, and cultivates stem cells in the produced Scapold to be transplanted into animals to assess tissue regeneration. This depends on components such as collagen in the tissue. Therefore, it is very important to identify the inclusion rate and distribution of components in the tissue, and the data are obtained by analyzing the color of the dyed tissue image. The process from image collection to analysis is costly, and the data collected and analyzed are managed in different formats by different research institutions. Therefore, data integration management and analysis results search are not being performed. In this paper, we establish a database that can manage relevant bigdata in an integrated manner, and propose a bio-image integra ed management and retrieval system that can be searched based on color, an important analytical measure in this field of study.",
    "keywords": "바이오 세포 이미지, 세포외 기질, 이미지 분석, 시각적 검색, Bio Cell Image, extracellular matrix, image analysis, visual retrieval",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002501655"
  },
  {
    "id": 70,
    "type": "column",
    "title": "[테크인사이드]농업으로 들어간 인공지능",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2019년 9월 4일",
    "abstract": "Worldometers(실시간 세계 통계)에 따르면 전 세계 인구는 77억명(2019년 8월 기준)을 넘었으며 2055년에는 100억명을 넘어설 것으로 예측된다. 유한한 땅과 한정된 자원 속에서 인구 증가는 식량 부족을 초래할 수도 있다. 한국농촌경제연구원이 산출한 우리나라의 곡물지급률은 지난 3년간(2015~2017년) 평균 23%에 불과해 세계 꼴찌 수준이다. 폭발적인 인구 증가로 인한 식량 부족, 낮은 곡물지급률을 극복할 수 있는 방안은 농업을 더 발전시키는 것이다.<br /> 어그리테크는 농업(Agriculture)과 기술(Technology)의 합성어인 신조어다. 정보통신기술(ICT), 사물인터넷(IoT), 빅데이터, 인공지능(AI), 로봇 등의 첨단 기술 등이 도입돼 이전에는 찾아볼 수 없었던 농업의 혁신이 일어나고 있다. 미국과 유럽 등 선진국에서 어그리테크에 활발한 투자를 하고 있고, 어그리테크 스타트업이 이스라엘에는 500여 개, 브라질에는 200여 개가 있을 정도다. 뿐만 아니라 최근에는 인공지능과 빅데이터의 융합으로 농업의 새로운 흐름을 만들어 가고 있다. 국내에서도 KIST에서 미래농업 시스템으로 스마트 팜(Smart Farm)에 대한 연구개발을 진행하며 한국에 적합한 팜 테크(Farm Tech) 산업을 육성해 나가고 있다.",
    "url": "http://www.kwnews.co.kr/nview.asp?aid=219090300125"
  },
  {
    "id": 71,
    "type": "column",
    "title": "[테크 인사이드]AI시대 지배할 것인가 지배당하며 살 것인가",
    "authors": "이수안",
    "venue": "강원일보",
    "date": "2019년 8월 7일",
    "abstract": "손정의 소프트뱅크 그룹 회장은 최근 문재인 대통령과의 만남에서 한국이 4차 산업혁명을 선도하기 위해 집중해야 할 것은 첫째도, 둘째도, 셋째도 인공지능(AI)이라고 조언했다. 이미 현대 산업은 인공지능이 많은 부분을 차지하고 있으며 혁신을 위해 다양한 연구를 시도하고 있다.<br /> 인공지능은 글자를 인식하는 것을 넘어 100개가 넘는 언어를 다루고 번역을 하는 등 다양한 혁신을 이뤄 냈다. 서로 다른 언어로 대화하더라도 같은 언어로 대화하는 것처럼 느끼도록 실시간 번역을 도와주며 카메라로 찍은 문자를 인식해 번역하기도 한다. 또 인공지능은 챗봇 서비스를 지능형으로 발전시키고 있다. 사용자와의 대화를 통해 요구사항을 분석하고 사용자를 실질적으로 돕는 서비스로 발전하고 있다. 그 밖에 스포츠나 주식 등의 기사를 자동으로 생성하는 서비스와 문서 요약 등 연구도 진행되고 있다.",
    "url": "http://www.kwnews.co.kr/nview.asp?aid=219080600011"
  },
  {
    "id": 72,
    "type": "dconference",
    "title": "효율적인 회귀분석을 위한 Shared Wide & Deep 모델",
    "authors": "김민규, 이수안, 김진호",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2019년 7월",
    "abstract": "회귀분석은 시간에 따라 변화하는 데이터의 예측에 많이 이용되는 방법 중에 하나이다. 기존의 Wide & Deep 모델은 추천시스템에서 뛰어난 성능을 내는 방법이지만 회귀분석에서는 좋은 성능을 내지 못한다. 따라서 본 논문에서는 기존의 Wide & Deep 모델을 개선하여 회귀분석에서 잘 동작하는 Shared Wide & Deep 모델을 제안하였다. 제안한 모델을 검증하기 위해 여러 데이터들을 기존의 Wide & Deep 모델 그리고 다양한 회귀 분석 모델들과 비교하였다. 제안한 모델이 다른 모델들보다 높은 R^2 값을 가지고, 효율적으로 회귀 분석이 수행됨을 실험을 통해 확인하였다.",
    "url": "https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE08763671"
  },
  {
    "id": 73,
    "type": "djournal",
    "title": "자유학기제 운영계획서에 대한 텍스트 빅데이터 분석 및 요약",
    "authors": "이수안, 박범준, 김민규, 신혜숙, 김진호",
    "venue": "컴퓨터교육학회논문지, 한국컴퓨터교육학회",
    "date": "2017년 5월",
    "badge": "[KCI]",
    "abstract": "사회 각 분야에서 관련 주제에 대한 보다 직접적인 정보를 수집하고 분석하기 위하여 빅데이터 분석이 활발하게 활용되고 있다. 우리나라에서 사회적 관심과 파급 효과가 큰 교육 분야에서도 빅데이터 분석 기술을 활용하여 교육이나 정책의 효과를 파악하고 정책 수립에 활용하는 것에 관심이 높아지고 있다. 본 논문에서는 교육 분야에서 빅데이터 분석 기술을 활용하는 방안을 소개하고자 한다. 현재 핵심 교육정책 중의 하나인 자유학기제에 초점을 두고, 각 학교가 작성한 운영계획서에 대해 텍스트 분석과 시각화를 통하여 주요 관심 사항과 차이점에 대해 살펴보았다. 특히 서울과 강원도 지역의 중학교 자유학기제 운영계획서를 대상으로 지역적으로 주요 특성과 관심 사항이 서로 다르다는 것을 비교하였다. 본 연구는 빅데이터 분석 기술을 교육 분야의 필요와 요구에 따라 적용하고 활용하였다는 것에 큰 의의가 있다. Big data analysis is actively used for collecting and analyzing direct information on related topics in each field of society. Applying big data analysis technology in education field is increasingly interested in Korea, because applying this technology helps to identify the effectiveness of education methods and policies and applying them for policy formulation. In this paper, we propose our approach of utilizing big data analysis technology in education field. We focus on free semester program, one of the current core education policies, and we analyze the main points of interests and differences in the free semester through analysis and visualization of texts that are written on the operation reports prepared by each school. We compare regional differences in key characteristics and interests based on the free semester operation reports from middle schools particularly at Seoul and Gangwon-do regions. In conclusion, applying and utilizing big data analysis technology according to the needs and requirements of education field is a great significance.",
    "keywords": "빅데이터, 텍스트 분석, 교육정책, 자유학기제, 시각화, Big Data, Text Analysis, Education Policies, Free Semester, Visualization",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002470779"
  },
  {
    "id": 74,
    "type": "conference",
    "title": "Detecting Spammers on Social Networks using Strongly Connected Components in the Distributed Environment",
    "authors": "Heesang Kim, Suan Lee, SungJin Park, and Jinho Kim",
    "venue": "The 2nd International Workshop on Big Data Analysis for Smart Energy",
    "date": "14 Oct. 2017",
    "abstract": "Recently, several studies attempt to process big data using Apache Spark. In addition, spammers are increasing lately, and they are exposing indiscriminate information on social networks (SNS) that users do not want. Previous relationshipbased spammer detections are not suitable for big data graph, because the detecting spammers in big data graphs requires a long computation time. Therefore, we propose an efficient spammer detection scheme based on social relations using Strongly Connected Components (SCC), which quickly finds cyclical relationships on Spark GraphX. We test our proposed spammer detection method through experiments, and it is able to find spammers in big data graph quickly.",
    "keywords": "Spammer, Spark, Strongly Connected Components, Social Networks",
    "url": "http://sigai.or.kr/workshop/bigcomp/2019/big-data-for-smart-energy/"
  },
  {
    "id": 75,
    "type": "dconference",
    "title": "빅데이터 기반의 통합적 트래픽 분석 플랫폼을 위한 저장구조 설계",
    "authors": "장종원, 김희상, 김민규, 이수안, 김진호",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2018년 12월",
    "abstract": "오늘날 보안이라는 이슈가 점점 더 중요하게 여겨지고 있다. 다양한 보안 장비들로 구성된 보안관리 시스템에서는 장비마다 다른 보안 로그를 생성해내며, 그 로그들을 한데 모아 저장 및 분석할 필요가 있다. 많은 보안 장비에서 생성되는 로그의 양은 빅데이터라 할 수 있고, 기존 RDBMS를 이용하여 저장, 분석하기에는 힘든 상황이다. 따라서 본 논문에서는 대용량 로그를 저장할 수 있는 Hadoop 기반의 분산저장 데이터베이스 시스템인 HBase를 사용한 보안로그 분석시스템을 제안한다. 제안한 시스템과 기존 RDBMS와의 수행시간을 비교하였으며, 데이터가 많아질수록 HBase가 압도적으로 우수한 수행시간을 보였다.",
    "keywords": "##",
    "url": "#"
  },
  {
    "id": 76,
    "type": "dconference",
    "title": "빅데이터 기술을 이용한 자유학기제 운영 실태 보고서 문서 분석",
    "authors": "김민규, 이수안, 김진호, 신혜숙",
    "venue": "한국소프트웨어종합학술대회, 한국정보과학회",
    "date": "2018년 12월",
    "abstract": "빅데이터 기술은 많은 분야에서 다각적인 자료 분석이 가능하도록 활용되고 있다. 본 논문에서는 교육 분야에서 새로운 정책에 대한 효과를 평가하기 위해 빅데이터 기술을 활용하였다. 전국 중학교에서 운영되고 있는 자유학기제의 운영 실태 분석을 위하여 비정형 데이터 분석과 시각화를 통하여 주요 관심사항과 차이점에 대해서 살펴보았다. 특히 지역적으로 중요 특성과 관심 사항이 다르다는 것을 부산과 전남을 중심으로 살펴보았다. 본 연구는 교육 전문가와 함께 필요와 요구에 따라 빅데이터 분석 기술을 적용하였다는 것에 의미가 있다.",
    "keywords": "##",
    "url": "#"
  },
  {
    "id": 77,
    "type": "journal",
    "title": "Scalable distributed data cube computation for large-scale multidimensional data analysis on a Spark cluster",
    "authors": "Suan Lee, Seok Kang, Jinho Kim, and Eun Jung Yu",
    "venue": "Cluster Computing",
    "date": "01 Feb. 2018",
    "badge": "[SCI]",
    "impact": "(IF: 2.040)",
    "abstract": "A data cube is a powerful analytical tool that stores all aggregate values over a set of dimensions. It provides users with a simple and efficient means of performing complex data analysis while assisting in decision making. Since the computation time for building a data cube is very large, however, efficient methods for reducing the data cube computation time are needed. Previous works have developed various algorithms for efficiently generating data cubes using MapReduce, which is a large-scale distributed parallel processing framework. However, MapReduce incurs the overhead of disk I/Os and network traffic. To overcome these MapReduce limitations, Spark was recently proposed as a memory-based parallel/distributed processing framework. It has attracted considerable research attention owing to its high performance. In this paper, we propose two algorithms for efficiently building data cubes. The algorithms fully leverage Spark’s mechanisms and properties: Resilient Distributed Top-Down Computation (RDTDC) and Resilient Distributed Bottom-Up Computation (RDBUC). The former is an algorithm for computing the components (i.e., cuboids) of a data cube in a top-down approach; the latter is a bottom-up approach. The RDTDC algorithm has three key functions. (1) It approximates the size of the cuboid using the cardinality without additional Spark action computation to determine the size of each cuboid during top-down computation. Thus, one cuboid can be computed from the upper cuboid of a smaller size. (2) It creates an execution plan that is optimized to input the smaller sized cuboid. (3) Lastly, it uses a method of reusing the result of the already computed cuboid by top-down computation and simultaneously computes the cuboid of several dimensions. In addition, we propose the RDBUC bottom-up algorithm in Spark, which is widely used in computing Iceberg cubes to maintain only cells satisfying a certain condition of minimum support. This algorithm incorporates two primary strategies: (1) reducing the input size to compute aggregate values for a dimension combination (e.g., A, B, and C) by removing the input, which does not satisfy the Iceberg cube condition at its lower dimension combination (e.g., A and B) computed earlier. (2) We use a lazy materialization strategy that computes every combination of dimensions using only transformation operations without any action operation. It then stores them in a single action operation. To prove the efficiency of the proposed algorithms using a lazy materialization strategy by employing only one action operation, we conducted extensive experiments. We compared them to the cube() function, a built-in cube computation library of Spark SQL. The results showed that the proposed RDTDC and RDBUC algorithms outperformed Spark SQL cube().",
    "keywords": "Distributed processing, Spark framework, Resilient distributed dataset, Data warehousing, On-line analytical processing, Multidimensional data cube, Iceberg cube",
    "url": "https://link.springer.com/article/10.1007/s10586-018-1811-1"
  },
  {
    "id": 78,
    "type": "djournal",
    "title": "세포외 기질의 구성 및 구조의 모방을 위한 바이오 빅데이터 시스템 설계",
    "authors": "이수안, 이솔, 유혁상, 김진호",
    "venue": "데이터베이스연구, 한국정보과학회 데이터베이스 소사이어티",
    "date": "2017년 12월 29일",
    "badge": "[KCI]",
    "abstract": "세포외 기질은 세포가 성장하고 분화하는데 필요한 생화학적 인자들과 세포를 위한 환경을 제공한다. 생체재료 분야에서는 세포외 기질의 구성 및 구조를 모방한 생체재료를 만들고, 세포를 배양하여 조직을 만들어 내는 실험이 지속되고 있다. 그러나 세포가 원하는 조직으로 잘 분화되는 3차원 지지체를 만들기 위해서는 세포외 기질에 대한 구성과 구조, 특성을 분석하고, 물리적 환경에 따른 조절 요인 등에 대한 연구가 필요하다. 본 논문에서는 형광 염색된 세포외 기질 이미지로부터 추출/가공을 통해 다양한 데이터를 저장하고 분석하는 바이오 빅데이터 시스템을 제안한다. The extracellular matrix provides the environment for cells and biochemical factors necessary for cell growth and differentiation. In the field of biomaterials, experiments have been continuing to make biomaterials that mimic the structure and structure of extracellular matrix, and use them to culture cells to produce tissues. However, it is necessary to analyze the constitution, structure, and characteristics of the extracellular matrix and to study the factors regulating the physical environment in order to produce a three-dimensional scaffold that differentiates cells into desired tissues, In this paper, we propose a bio-data system for storing and analyzing various data through extraction/processing from fluorescence-stained extracellular matrix images.",
    "keywords": "Context awareness, Tensor data cube, MapReduce framework",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002302451"
  },
  {
    "id": 79,
    "type": "conference",
    "title": "EEUM: Explorable and Expandable User-interactive Model for Browsing Bibliographic Information Networks",
    "authors": "Suan Lee, Young-Seok You, Sungjin Park, and Jinho Kim",
    "venue": "Proceedings of the Seventh International Conference on Emerging Databases: Technologies, Applications, and Theory",
    "date": "14 Oct. 2017",
    "abstract": "A suitable user interactive model is required to navigate efficiently in information network for users. In this paper, we have developed EEUM (Explorable and Expandable User-interactive Model) that can be used conveniently and efficiently for users in bibliographic information networks. The system shows the demonstration of efficient search, exploration, and analysis of information network using EEUM. EEUM allows users to find influential authors or papers in any research field. Also, users can see all relationships between several authors and papers at a glance. Users are able to analyze after searching and exploring (or navigating) bibliographic information networks efficiently by using EEUM.",
    "keywords": "Information networks, Graph database, Data visualization, User-interactive model",
    "url": "https://link.springer.com/chapter/10.1007/978-981-10-6520-0_33"
  },
  {
    "id": 80,
    "type": "dconference",
    "title": "초·중등 과학 교과서 용어주석 말뭉치 구축 및 검색을 위한 시스템 개발",
    "authors": "유영석, 이수안, 김진호, 윤은정, 박윤배",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2017년 06월",
    "abstract": "초중등 학교의 과학 교수학습 상황에서 과학용어는 교사와 학생 모두에게 많은 어려움을 야기하는 주요한 요인 가운데 하나이다. 학생의 수준에 맞게 과학 용어를 사용하고 그 과학 용어의 의미를 정확하게 설명하는 것이 중요하다. 이를 위해 초∙중등 교육과정에서 사용하는 과학 교과서에 있는 과학 용어들을 추출하고, 교과서 내에서 그 과학 용어의 용례와 정의를 파악하는 것이 매우 효과적인 방법이다. 이 논문에서는 6차, 7차, 및 2009 개정 교육 과정의 모든 과학 교과서에서 과학용어 말뭉치를 추출하고, 이들용어들의 출현빈도와 출현한 문장들과 교과서 정보들을 데이터베이스에 저장한다. 과학용어 말뭉치는 표준국어대사전과 교과서의 품사 매칭을 이용해 과학용서를 식별한다. 이렇게 구축된 과학용어 데이터베이스를 이용하여 과학용어 키워드 질의에 대해 교과서 내의 예문과 과목이나 교육과정별 정보를 시각회된 차트로 제공하는 검색 시스템을 개발한다. 이 시스템을 활용하여, 원하는 과학용어에 대해 학년별/과목별 교과서에 대한 예문이나 교육과정에 대해 파악할 수 있으며, 학생 수준에 맞춰 학습 보조자료로 활용하거나 연구 보조자료 등으로 편리하게 활용할 수 있을 것이다.",
    "keywords": "유비쿼터스 센서 네트워크, 스트림 데이터, 저장 관리자, 경동 시간 구조, 데이터 축소",
    "url": "https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07207213"
  },
  {
    "id": 81,
    "type": "djournal",
    "title": "대학생 진로설계 및 취업 지원을 위한 미래진로 빅데이터 정보 시스템 설계",
    "authors": "이기준, 이수안, 구경아, 김진호",
    "venue": "정보화 연구, 한국엔터프라이즈아키텍처학회",
    "date": "2017년 06월",
    "badge": "[KCI]",
    "abstract": "전 세계적인 경기침체로 인해 우리나라에서도 많은 문제를 겪고 있는데 이 중 청년층의 취업률 및 실업률 문제가 큰 문제로 부각되고 있다. 이를 위해 정부에서는 많은 정책을 수립하여 시행하고 있는데 이 과정에서 사용되는 기초데이터로 많이 활용되는 데이터는 고등교육기관 졸업자 취업통계 정보이다. 그러나 현재 고등교육기관 졸업자 취업통계정보는 정부차원에서의 정책 수립 및 평가에 주로 활용되고 그 내용이 취업자 수에 의한 양적 정보에 한정됨에 따라 실제 청년층의 취업률 제고에는 효과적으로 활용되고 있지 못하다는 지적이 있으며 일선 대학에서는 진로 및 취업 상세정보의 부재로 학생들의 진로 및 취업상담에 애로를 겪고 있다. 따라서 대학생의 진로 설계와 취업 지원을 위한 미래진로 빅데이터가 필요한 시점이며 이를 활용하여 청년층 실업문제를 과학적으로 접근하는 시도가 필요하다. 본 논문에서는 기존 고등교육기관 졸업자 취업통계조사를 개선하여 대학생 진로설계와 취업지원이 가능하도록 데이터를 수집하는 데이터베이스를 구축하고 정보의 수요자들이 손쉽게 본인의 진로를 결정할 수 있는 정보를 전달하는 정보시스템을 제안한다. Due to the global economic downturn, many problems have been experienced in Korea. Among them, the employment rate and the unemployment rate of the young people are becoming big problems. To this end, the government establishes and implements a number of policies. Data that are often used as basic data in this process are statistical information on employment of grad-uates of higher education institutions. However, it is pointed out that employment statistics infor-mation of graduates of higher education institutions is mainly used for policy formulation and evaluation at the government level and its content is limited to quantitative information by the number of employed persons. Therefore, there is an indication that it is not being effectively used to raise the employment rate of young people. At University, students are struggling with career and career counseling due to lack of career and employment details. It is necessary to try to approach the youth unemployment problem scientifically by utilizing the future career big data for career planning and career support of university students. In this paper, we have developed an information system that improves the employment survey of graduates of existing higher educa-tion institutions and builds a database that collects data to enable college career design and employment support, and allows information consumers to easily determine their own career path.",
    "keywords": "빅데이터, 정보 시스템, 진로 및 취업, Big Data, Information System, Career Path, Supporting Employment",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002243258"
  },
  {
    "id": 82,
    "type": "djournal",
    "title": "경영 시뮬레이션 게임을 위한 빅 데이터 분석 플랫폼 설계 및 개발",
    "authors": "강석, 이수안, 김진호, 이강수",
    "venue": "정보화 연구, 한국엔터프라이즈아키텍처학회",
    "date": "2017년 03월",
    "badge": "[KCI]",
    "abstract": "스마트폰의 출현으로 SNS, 위치정보, 각종 로그들을 포함에 방대한 양의 데이터가 쌓이고 있다. 이런 데이터들을 활용하여 가치있는 정보로 활용하기 위해 빅 데이터 기술에 대한 관심을 가지고 있다. 사람들이 빅 데이터 처리를 통해 나온 결과를 쉽게 알아볼 수 있는 시각화 기술에 대해서도 관심이 뜨겁다. 본 논문에서는 경영 시뮬레이션 게임에서 의사 결정을 위해서 빅 데이터 분석을 수행하였고 가공처리되서 나온 데이터들을 이용해 시각화 도구를 개발하였다. 시각화 도구에는 워드 클라우드, 단어 빈도 분석, 네트워크 그래프 분석, 군집 분석 등 다양한 시각화 도구가 있다. 이런 시각화 도구를 통해 사용자는 앞서 말한 가치 있는 정보를 확인 할 수 있다. 또한 사용자는 데이터를 필터링하거나 단어의 빈도수 또는 단어의 개수를 조절하거나 군집의 개수를 설정하여 사용자가 좀 더 쉽게 이해 할 수 있도록 시각화를 다시 할 수 있다. 이를 통해 효과적이고 탄력적인 경영과 의사 결정에 도움을 줄 수 있다. SNS with the advent of smart phones, position information, has accumulated a huge amount of data to include a variety of log. By utilizing such data, in order to take advantage of the information that is of value, we have an interest in big data technology. People are hot interest in big data processing through the you can find the results that have been briefly out of visualization technology. In this paper, we run the big data analysis for decision-making in the management simulation game, we were using the data that came out be processed to develop a visualization tool. The visualization tool, the analysis of the word cloud, word of the frequency of use, network graph analysis, cluster analysis, etc., there are a variety of visualization tools. Using such visualization tools, the user can confirm the valuable information previously described. Also, users can filter the data, or to adjust the number of words in the frequency and the word, by setting the number of congestion, the user can repeat the visualization may be more easily understood. Thus, it is possible to support effective and flexible management and decision making.",
    "keywords": "빅 데이터, 워드 클라우드, 소셜 네트워크 분석, 군집 분석, 시뮬레이션 게임, Big Data, Word Cloud, Social Network Analysis, Clutering Analysis, Simulation Game",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002213667"
  },
  {
    "id": 83,
    "type": "conference",
    "title": "Time-Sensitive Multi-Dimensional Recommender in database system",
    "authors": "Sungjin Park, Suan Lee, and Jinho Kim",
    "venue": "Big Data and Smart Computing (BigComp), 2017 IEEE International Conference on. IEEE",
    "date": "13-16 Feb. 2017",
    "abstract": "A suitable user interactive model is required to navigate efficiently in information network for users. In this paper, we have developed EEUM (Explorable and Expandable User-interactive Model) that can be used conveniently and efficiently for users in bibliographic information networks. The system shows the demonstration of efficient search, exploration, and analysis of information network using EEUM. EEUM allows users to find influential authors or papers in any research field. Also, users can see all relationships between several authors and papers at a glance. Users are able to analyze after searching and exploring (or navigating) bibliographic information networks efficiently by using EEUM.",
    "keywords": "Information networks, Graph database, Data visualization, User-interactive model",
    "url": "https://ieeexplore.ieee.org/document/7881679/"
  },
  {
    "id": 84,
    "type": "conference",
    "title": "Design and Development of Visualization Tool for Movie Review and Sentiment Analysis",
    "authors": "Young Seok You, Suan Lee, and Jinho Kim",
    "venue": "Proceedings of the Sixth International Conference on Emerging Databases: Technologies, Applications, and Theory. ACM",
    "date": "17-19 Oct. 2016",
    "abstract": "Typically, application or website shows the comments of people in a list format. This list means in seeing chronologically or log of recommends. However, it is difficult to grasp because of reading and knowing all countless comments of the topic at a glance. Therefore, it requires a lot of ability to grasp information at a glance via picking only the important information. In this paper, we design and develop a visualization tool that can identify a number of reviews containing comments on the movie at a glance. Review assumed to be extracted from the Amazon and IMDb that are both subjective information. The tool that we develop visualizes sentimental analysis of the review on pre-made Sentiment Dictionary with objective information of a movie. Our proposed system can search and display one or more movies. Users can determine the relationship between movies by clustering sentiment of positive/negative reviews and movie's factors. In the future, based on all the reviews on Amazon and grasp the reviews on a variety of movies and products, as well, it will be used as tools to help users of a rational choice.",
    "keywords": "Review DAta, Sentiment Analysis, Visualization",
    "url": "https://dl.acm.org/citation.cfm?id=3007841"
  },
  {
    "id": 85,
    "type": "djournal",
    "title": "물류 산업에서 빅 데이터 분석을 위한 텍스트 시각화 도구 설계 및 개발",
    "authors": "이강수, 이수안, 강석, 박찬민, 김진호",
    "venue": "정보화연구, 한국엔터프라이즈아키텍처학회",
    "date": "2016년 06월 30일",
    "badge": "[KCI]",
    "abstract": "스마트폰의 출현으로 SNS, 위치정보, 각종 로그들을 포함에 방대한 양의 데이터가 쌓이고 있다. 이런 데이터들을 활용하여 가치있는 정보로 활용하기 위해 빅 데이터 기술에 대한 관심을 가지고 있다. 사람들이 빅 데이터 처리를 통해 나온 결과를 쉽게 알아볼 수 있는 시각화 기술에 대해서도 관심이 뜨겁다. 본 논문에서는 경영 시뮬레이션 게임에서 의사 결정을 위해서 빅 데이터 분석을 수행하였고 가공처리되서 나온 데이터들을 이용해 시각화 도구를 개발하였다. 시각화 도구에는 워드 클라우드, 단어 빈도 분석, 네트워크 그래프 분석, 군집 분석 등 다양한 시각화 도구가 있다. 이런 시각화 도구를 통해 사용자는 앞서 말한 가치 있는 정보를 확인 할 수 있다. 또한 사용자는 데이터를 필터링하거나 단어의 빈도수 또는 단어의 개수를 조절하거나 군집의 개수를 설정하여 사용자가 좀 더 쉽게 이해 할 수 있도록 시각화를 다시 할 수 있다. 이를 통해 효과적이고 탄력적인 경영과 의사 결정에 도움을 줄 수 있다. SNS with the advent of smart phones, position information, has accumulated a huge amount of data to include a variety of log. By utilizing such data, in order to take advantage of the information that is of value, we have an interest in big data technology. People are hot interest in big data processing through the you can find the results that have been briefly out of visualization technology. In this paper, we run the big data analysis for decision-making in the management simulation game, we were using the data that came out be processed to develop a visualization tool. The visualization tool, the analysis of the word cloud, word of the frequency of use, network graph analysis, cluster analysis, etc., there are a variety of visualization tools. Using such visualization tools, the user can confirm the valuable information previously described. Also, users can filter the data, or to adjust the number of words in the frequency and the word, by setting the number of congestion, the user can repeat the visualization may be more easily understood. Thus, it is possible to support effective and flexible management and decision making.",
    "keywords": "물류 데이터, 데이터 크롤러, 빅 데이터 분석, 텍스트 시각화, Logistics Data, Data Crawler, Big Data Analysis, Text Visualization",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002127079"
  },
  {
    "id": 86,
    "type": "djournal",
    "title": "클라우드 환경에서 전자문서 관리 및 가상 스토리지 시스템 기술",
    "authors": "이수안, 최재용, 강상원, 이기준, 한명훈, 김진호",
    "venue": "정보화연구, 한국엔터프라이즈아키텍처학회",
    "date": "2016년 03월 30일",
    "badge": "[KCI]",
    "abstract": "많은 공공기관과 기업들이 전자문서를 이용하고 있으며 자체적으로 전자문서관리시스템을 운영하고 있다. 그러나 최근 전자문서의 범위는 지식, 정보, 콘텐츠를 포함하는 것으로 확대되어가고 있다. 또한 사용자들은 전자문서의 공유 및 협업에 대한 요구가 증가되는 한편, 모바일과 사물인터넷 등으로 데이터의 양은 급증하고 있다. 이러한 대규모의 데이터를 처리하며 IT 자원을 효율적으로 관리해주는 클라우드 컴퓨팅이 활성화되고 있다. 그리하여 본 논문에서는 클라우드 환경에서 자동화 기술을 통해 IT 자원을 상황에 맞게 조절하고, 시스템의 확장 및 축소가 가능한 전자문서관리에 대해서 연구하였다. 또한 클라우드 환경에서 스토리지 관리 기법과 전자문서의 체계적인 복제본 관리기술 등에 대해서 제안하였다. Many public institutions and companies have been utilizing a lot of electronic documents and have been operating electronic document management systems on their own. However, the scope of the recent electronic documents is becoming enlarged to include knowledge, information, contents, etc. In addition, users have an increasing demand for sharing and collaborating the electronic documents. With the recent advance of mobile and Internet of Things(IoT) technologies, furthermore, the size of data are increasing very rapidly and tremendously. In order to manage a huge amount of data, cloud computing technology has been raving up more and more, which can handle IT resources efficiently. Thus, we have studied the electronic document management system which is possible to expand and to collapse to suit users' needs and IT resources automatically in cloud environments. In addition, we propose an efficient storage management system and a systematic replica management technology of electronic documents in cloud environments.",
    "keywords": "전자문서, 전사콘텐츠관리(ECM), 전자문서관리시스템(EDMS), 클라우드 컴퓨팅, 클라우드 스토리지 서비스, Electronic Document, Enterprise Contents Management, Electronic Document Management System, Cloud Computing, Cloud Storage Service",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002099017"
  },
  {
    "id": 87,
    "type": "conference",
    "title": "Performance evaluation of MRDataCube for data cube computation algorithm using MapReduce.",
    "authors": "Suan Lee, and Jinho Kim.",
    "venue": "Big Data and Smart Computing (BigComp), 2016 International Conference on. IEEE",
    "date": "18-20 Jan. 2016",
    "abstract": "This paper presents the performance evaluation of MRDataCube which we have previously proposed as an efficient algorithm for data cube computation with data reduction using MapReduce framework. We performed a large number of analyses and experiments to evaluate the MRDataCube algorithm in the MapReduce framework. In this paper, we compared it to simple MR-based data cube computation algorithms, e.g., MRNaive, MR2D as well as algorithms converted into MR paradigms from conventional ROLAP (relational OLAP) data cube algorithms, e.g., MRGBLP and MRPipeSort. From the experimental results, we observe that the MRDataCube algorithm outperforms the other algorithms in comparison tests by increasing the number of tuples and/or dimensions.",
    "keywords": "Data Warehouse, Data Cube, OLAP, MapReduce, Hadoop, Multidimensional Analysis, Distributed Parallel Processing",
    "url": "https://ieeexplore.ieee.org/document/7425939/"
  },
  {
    "id": 88,
    "type": "journal",
    "title": "Efficient Level-Based Top-Down Data Cube Computation Using MapReduce.",
    "authors": "Suan Lee, Jinho Kim, Yang-Sae Moon, and Wookey Lee",
    "venue": "Transactions on Large-Scale Data-and Knowledge-Centered Systems XXI",
    "date": "17 July 2015",
    "abstract": "Data cube is an essential part of OLAP(On-Line Analytical Processing) to support efficiently multidimensional analysis for a large size of data. The computation of data cube takes much time, because a data cube with d dimensions consists of 2 d (i.e., exponential order of d) cuboids. To build ROLAP (Relational OLAP) data cubes efficiently, many algorithms (e.g., GBLP, PipeSort, PipeHash, BUC, etc.) have been developed, which share sort cost and input data scan and/or reduce data computation time. Several parallel processing algorithms have been also proposed. On the other hand, MapReduce is recently emerging for the framework processing huge volume of data like web-scale data in a distributed/parallel manner by using a large number of computers (e.g., several hundred or thousands). In the MapReduce framework, the degree of parallel processing is more important to reduce total execution time than elaborate strategies like short-share and computation-reduction which existing ROLAP algorithms use. In this paper, we propose two distributed parallel processing algorithms. The first algorithm called MRLevel, which takes advantages of the MapReduce framework. The second algorithm called MRPipeLevel, which is based on the existing PipeSort algorithm which is one of the most efficient ones for top-down cube computation. (Top-down approach is more effective to handle big data, compared to others such as bottom-up and special data structures which are dependent on main-memory size.) The proposed MRLevel algorithm tries to parallelize cube computation and to reduce the number of data scan by level at the same time. The MRPipeLevel algorithm is based on the advantages of the MRLevel and to reduce the number of data scan by pipelining at the same time. We implemented and evaluated the performance of this algorithm under the MapReduce framework. Through the experiments, we also identify the factors for performance enhancement in MapReduce to process very huge data.",
    "keywords": "Data cube, ROLAP, MapReduce, Hadoop, Distributed parallel computing",
    "url": "https://link.springer.com/chapter/10.1007/978-3-662-47804-2_1"
  },
  {
    "id": 89,
    "type": "journal",
    "title": "Multidimensional Hierarchical Browser, Keyword Search, and Automatic Management of Photos within Smartphones.",
    "authors": "Suan Lee, Sunhwa Jo, Ji-Seop Won, Jinho Kim, and Yang-Sae Moon.",
    "venue": "Applied Mathematics & Information Sciences",
    "date": "1 May 2015",
    "badge": "[SCOPUS]",
    "abstract": "Recently new mobile devices such as cellular phones, smartphones, and digital cameras are popularly used to take photos. By virtue of these convenient instruments, we can take many photos easily, but we suffer from the difficulty of managing and searching photos due to their large volume. This paper develops a mobile application software, called Photo Cube, which automatically extracts various metadata for photos (e.g., date/time, address, place name, weather, personal event, etc.) by taking advantage of sensors and programming functions embedded in mobile smartphones like Android phones or iPhones. To avoid heavy network traffic and high processing overhead, it clusters photos into a set of clusters hierarchically by GPSs and it extracts the metadata for each centroid photo of clusters automatically. Then it constructs and stores the hierarchies of clusters based on the date/time, and address within the extracted metadata as well as the other metadata into photo database tables in the flash memory of smartphones. Furthermore, the system builds a multidimensional cube view for the photo database, which is popularly used in OLAP(On-Line Analytical Processing) applications and it facilitates the top-down browsing of photos over several dimensions such as date/time, address, etc. In addition to the hierarchical browsing, it provides users with keyword search function in order to find photos over every metadata of the photo database in a user-friendly manner. With these convenient features of the Photo Cube, therefore, users will be able to manage and search a large number of photos easily, without inputting any additional information but with clicking simply the shutter in a camera.",
    "keywords": "photo metadata, photo annotation, clustered databases, multidimensional data cube, OLAP, hierarchical clustering, keyword search, multidimensional hierarchical browsing, mobile application, smartphones",
    "url": "http://www.naturalspublishing.com/Article.asp?ArtcID=8783"
  },
  {
    "id": 90,
    "type": "conference",
    "title": "A graphical administration tool for managing cloud storage system.",
    "authors": "Jaeyong Choi, Suan Lee, Sangwon Kang, and Jinho Kim.",
    "venue": "Big Data and Smart Computing (BigComp), 2015 International Conference on. IEEE",
    "date": "9-11 Feb. 2015",
    "abstract": "In recent years, the amount of data produced by mobile devices and the Internet has increased rapidly. To facilitate the storage of such a large amount of data, open-source-based cloud storage services have also increased. However, most of administration tools which are specialized in managing open-source-based storage services have shortcomings such as lack of sufficient features and difficulty in operation. In this paper, we designed and implemented a GUI(Graphic User Interface) tool for managing OpenStack Swift, an open-source cloud storage services, to resolve these shortcomings. In addition, this tool incorporates a feature for the power management of computers/nodes within a cloud storage cluster to improve energy efficiency.",
    "keywords": "graphic monitoring tool, cloud service, cloud storage system, data center, OpenStack, Swift",
    "url": "https://ieeexplore.ieee.org/document/7072854/"
  },
  {
    "id": 91,
    "type": "conference",
    "title": "MRDataCube: Data cube computation using MapReduce.",
    "authors": "Suan Lee, Sunhwa Jo, and Jinho Kim.",
    "venue": "Big Data and Smart Computing (BigComp), 2015 International Conference on. IEEE",
    "date": "9-11 Feb. 2015",
    "abstract": "Data cube is used as an OLAP (On-Line Analytical Processing) model to implement multidimensional analyses in many fields of application. Computing a data cube requires a long sequence of basic operations and storage costs. Exponentially accumulating amounts of data have reached a magnitude that overwhelms the processing capacities of single computers. In this paper, we implement a large-scale data cube computation based on distributed parallel computing using the MapReduce (MR) computational framework. For this purpose, we developed a new algorithm, MRDataCube, which incorporates the MR mechanism into data cube computations such that effective data cube computations are enabled even when using the same computing resources. The proposed MRDataCube consists of two-level MR phases, namely, MRSpread and MRAssemble. The main feature of this algorithm is a continuous data reduction through the combination of partial cuboids and partial cells that are emitted when the computation undergoes these two phases. From the experimental results we revealed that MRDataCube outperforms all other algorithms.",
    "keywords": "distributed parallel algorithm, cube, OLAP, multi-dimensional analysis, data cube computation, MapReduce, Hadoop",
    "url": "https://ieeexplore.ieee.org/document/7072817/"
  },
  {
    "id": 92,
    "type": "conference",
    "title": "A Multi-Dimensional Analysis and Data Cube for Unstructured Text and Social Media.",
    "authors": "Suan Lee, Namsoo Kim, and Jinho Kim.",
    "venue": "Big Data and Cloud Computing (BdCloud), 2014 IEEE Fourth International Conference on. IEEE",
    "date": "3-5 Dec. 2014",
    "abstract": "Recently, unstructured data like texts, documents, or SNS messages has been increasingly being used in many applications, rather than structured data consisting of simple numbers or characters. Thus it becomes more important to analysis unstructured text data to extract valuable information for usres decision making. Like OLAP (On-Line Analytical Processing) analysis over structured data, Multi-dimensional analysis for these unstructured data is popularly being required. To facilitate these analysis requirements on the unstructured data, a text cube model on multi-dimensional text database has been proposed. In this paper, we extended the existing text cube model to incorporate TF-IDF (Term Frequency Inverse Document Frequrency) and LM (Language Model) as measurements. Because the proposed text cube model utilizes new measurements which are more popular in information retrieval systems, it is more efficient and effective to analysis text databases. Through experiments, we revealed that the performance and the effectiveness of the proposed text cube outperform the existing one.",
    "keywords": "language model, OLAP, Multi-dimensional analysis, text cube, data cube, text databases, information retrieval, TF-IDF",
    "url": "https://ieeexplore.ieee.org/document/7034871/"
  },
  {
    "id": 93,
    "type": "conference",
    "title": "Customized Information Interface with Web Applications.",
    "authors": "Wookey Lee, Suan Lee, and Jinho Kim.",
    "venue": "International Conference on Database Systems for Advanced Applications. Springer",
    "date": "11 July 2014",
    "abstract": "When information is searched via internet, a browser indicates information about web pages on a single window, but the existing browser shows only fragments of page information to web surfing users who visit several sites at once and in turn causes insufficiency and inconvenience to the users. Rich Internet Application techniques, which are web application techniques for the simple and easy operation and diverse and dynamic screen composition, have received a lot of attention as a next-generation UI technique emphasizing on users’ convenience. In this dissertation, a two-dimensional and sequential advanced search is realized with the use of dynamic UI so users can save and employ the customized search information for further web search. Also, the search structure has been designed with the use of user-oriented keyword preference to have more customizes search results than the existing web search. Furthermore, this paper has proven a decrease in the number of searched pages by employing the customized search administrator using RIA techniques. Thus, it could be concluded that the customized search administrator supports users of the more efficient and flexible customize web search.",
    "keywords": "Web browser, UI technique, Web search, Rich internet application technique, Customized search",
    "url": "https://link.springer.com/chapter/10.1007/978-3-662-43984-5_7"
  },
  {
    "id": 94,
    "type": "djournal",
    "title": "다차원 텍스트 큐브를 이용한 호텔 리뷰 데이터의 다차원 키워드 검색 및 분석",
    "authors": "김남수, 이수안, 조선화, 김진호",
    "venue": "정보화연구, 한국엔터프라이즈아키텍처학회",
    "date": "2014년 03월 30일",
    "badge": "[KCI]",
    "abstract": "웹의 발달로 텍스트 등으로 이루어진 비정형 데이터의 활용에 대한 관심이 높아지고 있다. 웹 상에서 사용자들이 작성한 대부분의 비정형 데이터는 사용자의 주관이 담겨져있어 이를 적절히 분석 할 경우 사용자의 취향이나 주관적인 관점 등의 아주 유용한 정보를 얻을 수 있다. 이 논문에서는 이러한 비정형 텍스트 문서를 다양한 차원으로 분석하기 하는데 OLAP(온라인 분석 처리)의 다차원 데 이터 큐브 기술을 활용한다. 다차원 데이터 큐브는 간단한 문자나 숫자 형태의 정형적인 데이터에 대 해 다차원 분석하는데 널리 사용되었지만, 텍스트 문장으로 이루어진 비정형 데이터에 대해서는 활용 되지 않았다. 이러한 텍스트 데이터베이스에 포함된 정보를 다차원으로 분석하기 위한 방법으로 텍스 트큐브 모델이 최근에 제안되었는데, 이 텍스트 큐브는 정보 검색에서 널리 사용하는 용어 빈도수 (Term Frequency)와 역 인덱스(Inverted Index)를 측정값으로 이용하여 텍스트 데이터베이스에 대 한 다차원 분석을 지원한다. 이 논문에서는 이러한 다차원 텍스트 큐브를 활용하여 실제 서비스되고 있는 호텔 정보 공유 사이트의 리뷰 데이터 분석에 활용하였다. 이를 위해 호텔 리뷰 데이터에 대한 다차원 텍스트 큐브를 생성하였으며, 이를 이용하여 다차원 키워드 검색 기능을 제공하여 사용자 중 심의 의미있는 정보 검색이 가능한 시스템을 설계 및 구현하였다. 또한, 본 논문에서 제안하는 시스템 에 대해 다양한 실험을 수행하였으며 이를 통해 제안된 시스템의 실효성을 검증하였다. As the advance of WWW, unstructured data including texts are taking users' interests more and more. These unstructured data created by WWW users represent users' subjective opinions thus we can get very useful information such as users' personal tastes or perspectives from them if we analyze appropriately. In this paper, we provide various analysis efficiently for unstructured text documents by taking advantage of OLAP (On-Line Analytical Processing) multidimen-sional cube technology. OLAP cubes have been widely used for the multidimensional analysis for structured data such as simple alphabetic and numberic data but they didn't have used for unstructured data consisting of long texts. In order to provide multidimensional analysis for unstructured text data, however, Text Cube model has been proposed precently. It incorporates term frequency and inverted index as measurements to search and analyze text databases which play key roles in information retrieval. The primary goal of this paper is to apply this text cube model to a real data set from in an Internet site sharing hotel information and to provide multidimensional analysis for users' reviews on hotels written in texts. To achieve this goal, we first build text cubes for the hotel review data. By using the text cubes, we design and implement the system which provides multidimensional keyword search features to search and to analyze review texts on various dimensions. This system will be able to help users to get valuable guest-subjective summary information easily. Furthermore, this paper evaluats the proposed systems through various experiments and it reveals the effectiveness of the system.",
    "keywords": "다차원 텍스트 데이터베이스, 텍스트 큐브, 온라인 다차원 분석, 사용자 리뷰 분석, 키워드 검색 Multi-dimensional Text Databases, Text Cubes, On-Line Analytical Processing (OLAP), Usres' review analysis, keyword search",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001866784"
  },
  {
    "id": 95,
    "type": "conference",
    "title": "An Efficient Keyword Search over Photo Images Within Mobile Smartphones.",
    "authors": "Suan Lee, Jinho Kim, Jiseop Won, Namsoo Kim, Johyeon Kang, and Sunhwa Jo.",
    "venue": "Cloud and Green Computing (CGC), 2013 Third International Conference on. IEEE",
    "date": "30 Sept.-2 Oct. 2013",
    "abstract": "Nowadays, it is popular for users to take photos through mobile devices like smartphones. In order to help users to search lots of photos within their smartphones easily, this paper develops a mobile application software system supporting a keyword search feature over photos just like searching web pages in the Internet. When a user takes a photo, the system extracts its meta-data of date/time and GPS as well as its various annotations automatically (e.g., mailing address, place names, event names, weather, etc.). Based on the annotations, we implemented a keyword search function over photos in smartphones. With this system, users can easily search photos with keyword conditions, even though they don't give any additional information.",
    "keywords": "Keyword Search Over Databases, Mobile Application, Smartphone, Image Retrieval, Image Annotation, Keyword Search",
    "url": "https://ieeexplore.ieee.org/document/6686086/"
  },
  {
    "id": 96,
    "type": "djournal",
    "title": "맵리듀스를 이용한 정렬 기반의 데이터 큐브 분산 병렬 계산 알고리즘",
    "authors": "이수안, 김진호",
    "venue": "전자공학회논문지, 대한전자공학회",
    "date": "2012년 09월 30일",
    "badge": "[KCI]",
    "abstract": "최근 많은 응용 분야에서 대규모 데이터에 대해 온라인 다차원 분석(OLAP)을 사용하고 있다. 다차원 데이터 큐브는 OLAP 분석에서 핵심 도구로 여긴다. 본 논문에서는 맵리듀스 분산 병렬 처리를 이용하여 효율적으로 데이터 큐브를 계산하는 방법을 연구하고자 한다. 이를 위해, 맵리듀스 프레임워크에서 데이터 큐브 계산 방법으로 잘 알려진 PipeSort 알고리즘을 구현하는 효율적인 방법에 대해서 살펴본다. PipeSort는 데이터 큐브의 한 큐보이드에서 동일한 정렬 순서를 갖는 여러 큐보이드를 한 파이프라인으로 한꺼번에 계산하는 효율적인 방식이다. 이 논문에서는 맵리듀스 프레임워크에서 PipeSort의 파이프라인을 구현한 네 가지 방법을 20대의 서버에서 수행하였다. 실험 결과를 보면, 고차원 데이터에 대해서는 PipeMap-NoReduce 알고리즘이 우수한 성능을 보였으며, 저차원 데이터에 대해서는 Post-Pipe 알고리즘이 더 우수함을 보였다. Recently, many applications perform OLAP(On-Line Analytical Processing) over a very large volume of data. Multidimensional data cube is regarded as a core tool in OLAP analysis. This paper focuses on the method how to efficiently compute data cubes in parallel by using a popular parallel processing tool, MapReduce. We investigate efficient ways to implement PipeSort algorithm, a well-known data cube computation method, on the MapReduce framework. The PipeSort executes several (descendant) cuboids at the same time as a pipeline by scanning one (ancestor) cuboid once, which have the same sorting order. This paper proposed four ways implementing the pipeline of the PipeSort on the MapReduce framework which runs across 20 servers. Our experiments show that PipeMap-NoReduce algorithm outperforms the rest algorithms for high-dimensional data. On the contrary, Post-Pipe stands out above the others for low-dimensional data.",
    "keywords": "multidimensional Data Cube, MapReduce, Distributed Parallel Computing, PipeSort",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001696347"
  },
  {
    "id": 97,
    "type": "djournal",
    "title": "스마트폰에서 계층 모델 기반의 사진 자동 분류 및 사진 탐색기 개발",
    "authors": "최재용, 원지섭, 이수안, 김진호",
    "venue": "정보과학회논문지, 한국정보과학회",
    "date": "2012년 09월 30일",
    "badge": "[KCI]",
    "abstract": "최근 스마트폰의 보급이 확산되고, 대중화됨에 따라 기존의 많은 모바일 기기들을 대체하고 있다. 많은 사용자들은 스마트폰을 이용한 사진 촬영을 취미뿐만 아니라 일상생활의 한 부분으로 많이 이용하고 있다. 하지만, PC에 비해 제한된 처리 능력과 디스플레이 크기를 가진 스마트폰에서 사진의 양이 증가함에 따라 단일 분류 기준으로는 관리 및 탐색에 어려움이 있다. 본 논문에서는 스마트폰에서 날짜/시간, GPS 정보를 추출한 뒤, 계층 모델을 생성하고, 계층 모델에 따라 사진 분류를 통해 효과적인 계층적 사진 탐색을 제공한다. 본 논문에서 제안한 시스템은 (1) 사용자 지정 가상 계층에 따른 사진 탐색, (2) 계층 트리 노드 병합을 이용한 사진 탐색, 그리고 (3) 균형 계층 트리를 이용한 사진 탐색 기법을 이용하여 사진 탐색의 편리함과 효율성을 극대화 하였으며, 구글 안드로이드 기반의 스마트폰에서 계층적 사진 탐색기를 설계 및 개발하였다. Recently smart phones are replacing a number of existing mobile devices while gaining wide popularity. Taking pictures with smart phones became a big part of our daily lives as well as hobbies. However, smart phones have limited processing capabilities and display size compared to a PC. Therefore, it is hard to manage and explore photos in a single category basis when the number of photos in a phone increase. This paper provides an effective hierarchical photo exploring system. As generating a hierarchical model by extracting date/time and GPS data from smartphones, this system offers us with an efficient way to explore photos. This photo exploring system features (1) using user customizable virtual hierarchy (2) using hierarchical tree nodes merge (3) maximizing efficiency and convenience by using balanced hierarchy tree. It was designed and developed using a Google Android smart phone.",
    "keywords": "사진 메타데이터, 사진 관리, 사진 검색, 계층 모델, 스마트폰 Photo Metadata, Photo Management, Photo Search, Hierarchy Model, Smartphone",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001695091"
  },
  {
    "id": 98,
    "type": "conference",
    "title": "Efficient Distributed Parallel Top-Down Computation of ROLAP Data Cube Using MapReduce.",
    "authors": "Suan Lee, Jinho Kim, Yang-Sae Moon, and Wookey Lee.",
    "venue": "Data Warehousing and Knowledge Discovery: 14th International Conference, DaWaK 2012",
    "date": "3-6 Sept. 2012",
    "abstract": "The computation of multidimensional OLAP(On-Line Analytical Processing) data cube takes much time, because a data cube with D dimensions consists of 2 D cuboids. To build ROLAP(Relational OLAP) data cubes efficiently, existing algorithms (e.g., GBLP, PipeSort, PipeHash, BUC, etc) use several strategies sharing sort cost and input data scan, reducing data computation, and utilizing parallel processing techniques. On the other hand, MapReduce is recently emerging for the framework processing a huge volume of data like web-scale data in a distributed/parallel manner by using a large number of computers (e.g., several hundred or thousands). In the MapReduce framework, the degree of parallel processing is more important to reduce total execution time than elaborate strategies. In this paper, we propose a distributed parallel processing algorithm, called MRPipeLevel, which takes advantage of the MapReduce framework. It is based on the existing PipeSort algorithm which is one of the most efficient ones for top-down cube computation. The proposed MRPipeLevel algorithm parallelizes cube computation and reduces the number of data scan by pipelining at the same time. We implemented and evaluated the proposed algorithm under the MapReduce framework. Through the experiments, we also identify factors for performance enhancement in MapReduce to process very huge data.",
    "keywords": "Data Cube, ROLAP, MapReduce, Hadoop, Distributed Parallel Computing",
    "url": "https://link.springer.com/chapter/10.1007/978-3-642-32584-7_14"
  },
  {
    "id": 99,
    "type": "dconference",
    "title": "얼굴 인식과 SNS 정보를 이용한 모바일 기기에서 사진 자동 분류 및 검색",
    "authors": "최재용, 이수안, 김진호",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2012년 06월",
    "abstract": "본 논문에서는 얼굴 인식 기술과 SNS 정보를 이용하여 사람의 얼굴을 기준으로 사진들을 효과적으로 분류하고 검색할 수 있는 시스템을 개발하였다. 얼굴 인식 기술을 이용하여 촬영된 사진의 분석을 통해 얼굴로부터 나이, 성별, 안경 착용 유무, 웃는 얼굴 판단 등의 의미적인 정보를 추출한다. 또한, 얼굴 인식을 통해 얻은 SNS 정보에서는 이름, 생일, 학력, 직업, 고향, 관심 분야, 종교 등의 개인적인 정보를 추출할 수 있다. 추출한 정보를 이용한 효과적인 사진 분류 및 검색을 통해 사용자의 편의를 극대화하였다. 본 논문에서는 구글 안드로이드 기반의 스마트폰에서 제안한 사진 자동 분류 및 검색 시스템을 구현하였다.",
    "url": "https://www.dbpia.co.kr/Journal/ArticleDetail/NODE01907205"
  },
  {
    "id": 100,
    "type": "djournal",
    "title": "맵리듀스를 이용한 데이터 큐브의 상향식 계산을 위한 반복적 알고리즘",
    "authors": "이수안, 김진호",
    "venue": "정보화연구, 한국엔터프라이즈아키텍처학회",
    "date": "2012년 03월 30일",
    "badge": "[KCI]",
    "abstract": "최근 데이터의 폭발적인 증가로 인해 대규모 데이터의 분석에 대한 요구를 충족할 수 있는 방법들이 계속 연구되고 있다. 본 논문에서는 맵리듀스를 이용한 분산 병렬 처리를 통해 대규모 데이터큐브의 효율적인 계산이 가능한 MRIterativeBUC 알고리즘을 제안하였다. MRIterativeBUC 알고리즘은 기존의 BUC 알고리즘을 맵리듀스의 반복적 단계에 따른 효율적인 동작이 가능하도록 개발되었고, 기존의 대규모 데이터 큐브 계산에 따른 문제인 데이터 크기와 저장 및 처리 능력의 한계를 해결하였다. 또한, 분석자의 관심 부분에 대해서만 계산하는 빙산 큐브 개념의 도입과 파티셔닝, 정렬과 같은 큐브 계산을 분산 병렬 처리하는 방법 등의 장점들을 통해 데이터 방출량을 줄여서 네트워크 부하를 줄이고, 각 노드의 처리량을 줄이며, 궁극적으로 전체 큐브 계산 비용을 줄일 수 있다. 본 연구결과는 맵리듀스를 이용한 데이터 큐브 계산에 대해서 상향식 처리와 반복적 알고리즘을 통해 다양한 확장이 가능하며, 여러 응용 분야에서 활용이 가능할 것으로 예상된다. Due to the recent data explosion, methods which can meet the requirement of large data analysis has been studying. This paper proposes MRIterativeBUC algorithm which enables efficient computation of large data cube by distributed parallel processing with MapReduce framework. MRIterativeBUC algorithm is developed for efficient iterative operation of the BUC method with MapReduce, and overcomes the limitations about the storage size and processing ability caused by large data cube computation. It employs the idea from the iceberg cube which computes only the interesting aspect of analysts and the distributed parallel process of cube computation by partitioning and sorting. Thus, it reduces data emission so that it can reduce network overload, processing amount on each node, and eventually the cube computation cost. The bottom-up cube computation and iterative algorithm using MapReduce, proposed in this paper, can be expanded in various way, and will make full use of many applications.",
    "keywords": "데이터 큐브, BUC 알고리즘, 맵리듀스, 분산 병렬 컴퓨팅 Data Cube, BUC Algorithm, MapReduce, Distributed Parallel Computing",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001735949"
  },
  {
    "id": 500,
    "type": "djournal",
    "title": "스마트폰에서 센서 데이터의 효과적인 저장 및 관리 방법",
    "authors": "이수안, 원지섭, 최미정",
    "venue": "Telecommunications Review, SK텔레콤",
    "date": "2012년",
    "badge": "[KCI]"
  },
  {
    "id": 101,
    "type": "conference",
    "title": "Photo cube: An automatic management and search for photos using mobile smartphones.",
    "authors": "Jinho Kim, Suan Lee, Ji-Seop Won, and Yang-Sae Moon.",
    "venue": "Dependable, Autonomic and Secure Computing (DASC), 2011 IEEE Ninth International Conference on. IEEE",
    "date": "12-14 Dec. 2011",
    "abstract": "Recently new mobile devices such as cellular phones, smart phones, and digital cameras are popularly used to take photos. By the virtue of these convenient instruments, we can take many photos easily, but we suffer from the difficulty of managing and searching photos due to their large volume. This paper develops a mobile application software, called Photo Cube, which automatically extracts various metadata for photos (e.g., date/time, place/address, weather, personal event, etc.) by taking advantage of sensors and networking functions embedded in mobile smart phones like Android phones or iPhones. The metadata can be used to manage and to search photos. Using this Photo Cube, users will be able to classify, store, manage, and search a large number of photos easily, without specifying any information but just clicking the shutter in a camera. The Photo Cube system was implemented on smart phones using Google's Android.",
    "keywords": "multidimensional search, photo metadata, photo annotation, image databases, mobile application, smartphones, text search",
    "url": "https://ieeexplore.ieee.org/document/6118849/"
  },
  {
    "id": 102,
    "type": "dconference",
    "title": "모바일 단말기에서 효과적인 사진 관리를 위한 계층적 사진 탐색기",
    "authors": "이수안, 원지섭, 최재용, 김진호",
    "venue": "가을 학술발표논문집, 한국정보과학회",
    "date": "2011년 11월",
    "abstract": "본 논문에서는 모바일에서 수천, 수만장의 사진을 효과적으로 관리하기 위한 사진 탐색기에서 대해서 제안하였다. 대부분의 사람들은 개인적인 분류 기준에 따라서 사진을 분류 및 관리한다. 하지만 사진의 양이 많아짐에 따라 기존의 단일 분류 기준으로는 관리 및 탐색의 어려움이 있다. 본 논문에서는 촬영된 사진에서 날짜/시간, 주소를 자동으로 추출하고, 추출된 정보의 계층 구조에 따라 사진을 효과적으로 자동 분류한다. 또한, 계층 모델의 구조에 따라 사용자가 언제든지 변경가능한 가상 계층 디렉토리 기능을 제공하고, 동적 계층 탐색 기능을 제공하여 사용자가 쉽고 빠르게 사진을 탐색할 수 있다. 사용자는 사진만 촬영하면, 편리하게 사진을 계층 분류 및 관리할 수 있는 기능을 사용할 수 있다. 본 논문에서는 구글 안드로이드 기반의 스마트 폰에서 제안한 계층적 사진 탐색기를 구현하였다.",
    "url": "https://www.dbpia.co.kr/Journal/ArticleDetail/NODE01746099"
  },
  {
    "id": 103,
    "type": "dconference",
    "title": "M2M 장비의 모바일 센서 데이터 관리를 위한 내장형 에이전트 설계",
    "authors": "최미정, 함효식, 이수안",
    "venue": "한국통신학회 종합 학술 발표회 논문집 (추계)",
    "date": "2011년 11월",
    "abstract": "최근 기술의 발달로 기계간의 통신을 지칭하는 M2M 개념이 등장하였고, 모바일 기기의 보급 확산으로 편의성을 더하고, 스마트한 세상을 열어가고 있다. 현재 스마트폰, 태블릿 PC 등 많은 모바일 장비들이 자체 내에 센서를 내장하고 있으며 다양한 센싱 정보를 제공하고 있다.본 논문에서는 M2M 모바일 장비에 탑재된 센서의 데이터를 효율적으로 관리하는 에이전트를 설계한다. 모바일 센서 관리 에이전트는 센서 데이터의 효율적인 저장 및 검색을 위해 정제 및 축소 기법을 사용하고, 센서 데이터의 통신 오버헤드를 줄이기 위한 통지 기법을 제안한다.",
    "url": "https://www.dbpia.co.kr/Journal/ArticleDetail/NODE02186330"
  },
  {
    "id": 104,
    "type": "conference",
    "title": "Distributed Parallel Top-Down Computation of Data Cube using MapReduce",
    "authors": "Suan Lee, Yang-Sae Moon, and Jinho Kim",
    "venue": "Proceedings of 3rd International Conference on Emerging Databases",
    "date": "25-27 Aug. 2011",
    "abstract": "Data cube has been studied for efficient analysis of large scale multidimensional data, and it has been used for multidimensional analysis and decision-making in various applications. Recently, MapReduce framework has been developed and utilized for distributed parallel processing of large scale data efficiently. This paper proposes the MRTDC algorithm to compute large scale multidimensional data cubes in top-down fashion by using the MapReduce framework. We reveal through experimental results that the MRTDC algorithm is quite efficiently operated within a little processing time as it reuses resulting data and reduces file I/O.",
    "keywords": "multidimensional database, data warehouse, data cube, mapreduce"
  },
  {
    "id": 105,
    "type": "dconference",
    "title": "클라우드 컴퓨팅 서비스의 효율성 사례에 관한 연구 (대회정보시스템 분석 사례를 중심으로)",
    "authors": "강상원, 최재용, 이수안, 김진호",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2011년 6월",
    "abstract": "클라우드 컴퓨팅은 차세대 인터넷 컴퓨팅 패러다임으로 등장하여 뛰어난 접근성, 확장성, 비용절감 등의 효과로 인해 인터넷 기업들을 중심으로 빠르게 도입되고 있다. 또한 스마트폰, 넷북 등 각종 모바일 기기의 진화와 맞물려 클라우드 컴퓨팅 서비스는 IT 트랜드의 중심으로 자리잡고 있다. 이에 본 논문에서는 클라우드 컴퓨팅에 대한 개념 및 동향과 함께 2014인천아시아경기 대회정보시스템의 분석 사례를 들어 클라우드 컴퓨팅의 효율성과 비용절감 효과에 대해서 살펴본다."
  },
  {
    "id": 106,
    "type": "conference",
    "title": "MapReduce-based Distributed and Parallel Computation of Iceberg Cubes",
    "authors": "Suan Lee, Yang-Sae Moon, and Jinho Kim",
    "venue": "Proceedings of 2nd International Conference on Emerging Databases",
    "date": "30-31 Aug. 2010",
    "abstract": "Data cubes enable us to efficiently analyze a large volume of data, but the computation of data cubes causes the severe processing time and space overhead. Iceberg cubes alleviate the overhead of data cube computation by performing the focused analysis on a small part of data cubes. However, iceberg cubes still require a lot of CPU and memory resources. To solve this problem, we adopt the MapReduce framework in computing iceberg cubes. We propose two MapReduce-based algorithms, MR-Naïve and MR-BUC, which efficiently compute iceberg cubes in a fully distributed and parallel manner. Experimental results show that, compared with the traditional algorithm, our MapReduce-based algorithms improve the computation performance by an order of magnitude.",
    "keywords": "OLAP, data cubes, iceberg cubes, MapReduce, cloud computing"
  },
  {
    "id": 107,
    "type": "dconference",
    "title": "SNMP MIB 기반 네트워크 관리 시스템의 다차원 분석을 위한 데이터 웨어하우스 설계",
    "authors": "이수안, 최미정, 김진호",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2010년 6월",
    "abstract": "최근 거대하고 다양한 네트워크의 관리를 위해 네트워크 관리 시스템을 많이 사용하고 있다. 하지만 단순히 네트워크에 대한 관리 정보 뿐만 아니라 프로토콜 별 트래픽 분석이나 QoS 등에 필요한 다차원 정보가 필요하다. 본 논문에서는 MIB 기반의 네트워크 관리 시스템의 효율적인 다차원 분석을 위해 데이터 웨어하우스를 설계하였다. 이러한 설계를 기반으로 기존 정보로는 분석하기 어려운 많은 요구 사항에 대해서 질의할 수 있으며, 네트워크 설계나 트래픽, QoS 등에 이용이 가능하다."
  },
  {
    "id": 108,
    "type": "dconference",
    "title": "클라우드 컴퓨팅 환경에서 데이터 웨어하우스 연구",
    "authors": "이수안, 김진호, 문양세",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2010년 6월",
    "abstract": "클라우드 컴퓨팅은 새로운 IT 기술의 중요한 패러다임으로 인터넷 기업들을 중심으로 적용되고 있으며, 관련된 연구 및 기술 개발 등이 활발히 이루어지고 있다. 지속적인 데이터 증가로 대규모 데이터를 체계화된 정보로 가공 및 저장 관리에 대한 요구가 확산되고 있다. 본 논문에서는 클라우드 컴퓨팅에 대한 기술과 데이터 웨어하우스를 위한 기술 동향, 그리고 클라우드 컴퓨팅 환경에서 데이터 웨어하우스 연구와 서비스 등을 통해 차세대 데이터 웨어하우스의 전망에 대해서 살펴본다."
  },
  {
    "id": 109,
    "type": "dconference",
    "title": "맵리듀스를 이용한 빙산 큐브 병렬 계산",
    "authors": "이수안, 김진호, 문양세, 노웅기",
    "venue": "한국컴퓨터종합학술대회, 한국정보과학회",
    "date": "2010년 6월",
    "abstract": "대용량 데이터의 효율적 분석을 위해 데이터 뷰브가 연구되었으며, 데이터 큐브 계산의 고비용 문제점을 해결하기 위하여 큐브의 일부 영역만을 계산하는 빙산 큐브가 등장하였다. 빙산 큐브는 저장 공간의 감소, 집중적인 분석 등의 장점이 있으나, 여전히 많은 계산과 저장 공간을 필요로 하는 단점이 있다. 본 논문에서는 이러한 문제점을 해결하는 실용적인 방법으로 대용량 문제를 분산하여 처리하는 분산 병렬 컴퓨팅 기술인 맵리듀스(MapReduce) 프레임워크를 사용하여 분산 병렬 빙산 큐브인 MR-Naive와 MR-BUC 알고리즘을 제안한다. 실험을 통해 맵리듀스 프레임워크를 통한 빙사 큐브 계산이 효율적으로 분산 병렬 처리 됨을 확인하였다.",
    "url": "http://www.dbpia.co.kr/Journal/ArticleDetail/NODE01471710"
  },
  {
    "id": 110,
    "type": "dconference",
    "title": "유비쿼터스 센서 네트워크를 이용한 건물 화재 모니터링 시스템의 다차원 데이터베이스 설계",
    "authors": "김진호, 이수안, 민두환, 김석훈, 남시병",
    "venue": "학술 심포지움 논문집, 한국정보과학회",
    "date": "2009년 6월",
    "abstract": "유비쿼터스 센서 네트워크를 이용하여 건물 화재 모니터링 시스템을 개발할 때, 대형 건물에 대한 방재 관련 정보와 센싱된 데이터를 통한 감시, 화재 위험요소에 대한 정보를 모니터해야 한다. 이 연구에서는 센서 네트워크에서 스트림 형태로 들어오는 화재 모니터링 데이터를 다양하게 분석하고 감시하는데 사용하기 위한 데이터베이스를 설계하였다. 다양한 관점에서 데이터를 모니터링하고 분석할 수 있도록 여러가지 차원을 기준으로 스타 스키마 형태의 다차원 구조로 설계하였다."
  },
  {
    "id": 111,
    "type": "journal",
    "title": "SAMSTARplus: An Automatic Tool for Generating Multi-Dimensional Schemas from an Entity-Relationship Diagram.",
    "authors": "Jinho Kim, Donghoo Kim, Suan Lee, Yang-Sae Moon, Il-Yeol Song, Ritu Khare, and Yuan An",
    "venue": "Revista de Informática Teórica e Aplicada 16.2",
    "date": "2009",
    "badge": "[SCOPUS]",
    "abstract": "This paper presents a tool that automatically generates multidimensional schemas for data warehouses from OLTP entity-relationship diagrams (ERDs). Based on user’s input parameters, it generates star schemas, snowflake schemas, or a fact constellation schema by taking advantage of only structural information of input ERDs. Hence, SAMSTARplus can help users reduce efforts for designing data warehouses and aids decision making.",
    "url": "http://www.seer.ufrgs.br/rita/article/view/rita_v16_n2_p79"
  },
  {
    "id": 112,
    "type": "conference",
    "title": "SAMSTARplus: An Automatic Tool for Generating Multi-Dimensional Schemas from an Entity-Relationship Diagram",
    "authors": "Jinho Kim, Donghoo Kim, Suan Lee, Yang-Sae Moon, Il-Yeol Song, Ritu Khare, and Yuan An",
    "venue": "International Conference on Conceptual Modeling",
    "date": "2009",
    "abstract": "This paper presents a tool that automatically generates multidimensional schemas for data warehouses from OLTP entity-relationship diagrams (ERDs). Based on user’s input parameters, it generates star schemas, snowflake schemas, or a fact constellation schema by taking advantage of only structural information of input ERDs. Hence, SAMSTARplus can help users reduce efforts for designing data warehouses and aids decision making.",
    "url": "http://citeseerx.ist.psu.edu/viewdoc/summary;jsessionid=55D08BE4216286C4B026CD075C595C0B?doi=10.1.1.918.6675"
  },
  {
    "id": 113,
    "type": "conference",
    "title": "SAMSTAR: An Automatic Tool for Generating Star Schema from Entity-Relationship Diagram",
    "authors": "Il-Yeol Song, Ritu Khare, Yuan An, Suan Lee, Sang-Pil Kim, Jinho Kim, and Yang-Sae Moon",
    "venue": "International Conference on Conceptual Modeling. Springer Berlin Heidelberg",
    "date": "2008",
    "abstract": "While online transaction processing (OLTP) databases are modeled with Entity-Relationship Diagrams (ERDs), data warehouses constructed from these OLTP DBs are usually represented as star schema. Designing data warehouse schemas, however, is very time consuming. We present a prototype system, SAMSTAR, which automatically generates star schemas from an ERD. The system takes an ERD drawn by ERwin Data Modeler as an input and generates star schemas. SAMSTAR uses the Connection Topology Value [1] which is the syntactic structural information embedded in an ERD. SAMSTAR displays the resulting star schemas on a computer screen graphically. With this automatic generation of star schema, this system helps designers reduce their efforts and time in building data warehouse schemas.",
    "keywords": "Prototype System, Automatic Generation, Automatic Tool, Connection Topology, Document Object Model",
    "url": "http://citeseerx.ist.psu.edu/viewdoc/summary;jsessionid=55D08BE4216286C4B026CD075C595C0B?doi=10.1.1.918.6675"
  },
  {
    "id": 114,
    "type": "djournal",
    "title": "유비쿼터스 센서 네트워크에서 스트림 데이터를 효율적으로 관리하는 저장 관리자 구현",
    "authors": "이수안, 김진호, 신성현, 남시병",
    "venue": "전자공학회논문지, 대한전자공학회",
    "date": "2009년 05월",
    "badge": "[KCI]",
    "abstract": "유비쿼터스 센서 네트워크를 통해 수집되는 데이터는 끊임없이 변화하는 스트림 데이터이다. 이 스트림 데이터는 기존의 데이터베이스와는 매우 다른 특성을 가지고 있어서, 이를 저장하고 분석 및 질의 처리하는 방법에 대한 새로운 기법이 필요하며, 이에 대한 연구가 최근에 많은 관심을 끌고 있다. 본 연구에서는 센서 네트워크로부터 끊임없이 들어오는 스트림 데이터를 수집하고 이를 효율적으로 데이터베이스에 저장하는 저장 관리자를 구현하였다. 이 저장 관리자는 무선 센서 환경에서 발생하는 오류에 대한 정제, 반복적으로 센싱되는 동일한 데이터에 대한 축소 기능, 장기간의 스트림 데이터를 경동 시간 구조로 유지하는 기능 등을 제공한다. 또 이 연구에서는, 구현된 저장 관리자를 건물의 온도, 습도, 조도 등을 수집하는 건물 화재 감시 센서 네트워크에 적용하여 그 성능을 측정하였다. 실험 결과, 이 저장 관리자는 스트림 데이터의 저장 공간을 현저히 줄이며, 건물 화재 감시를 위한 장기간의 스트림 데이터를 저장하는데 효과적임을 보였다. Stream data, gathered from ubiquitous sensor networks, change continuously over time. Because they have quite different characteristics from traditional databases, we need new techniques for storing and querying/analyzing these stream data, which are research issues recently emerging . In this research, we implemented a storage manager gathering stream data and storing them into databases, which are sampled continuously from sensor networks. The storage manager cleans faulty data occurred in mobile sensors and it also reduces the size of stream data by merging repeatedly-sampled values into one and by employing the tilted time frame which stores stream data with several different sampling rates. In this research, furthermore, we measured the performance of the storage manager in the context of a sensor network monitoring fires of a building. The experimental results reveal that the storage manager reduces significantly the size of storage spaces and it is effective to manage the data stream for real applications monitoring buildings and their fires.",
    "keywords": "유비쿼터스 센서 네트워크, 스트림 데이터, 저장 관리자, 경동 시간 구조, 데이터 축소",
    "url": "https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001342984"
  },
  {
    "id": 115,
    "type": "djournal",
    "title": "u강원 : 강원도의 u-City 구축 전략 및 사례",
    "authors": "김진호, 최황규, 김화종, 이수안, 김석훈",
    "venue": "정보과학회지, 한국정보과학회",
    "date": "2008년 08월",
    "badge": "[KCI]",
    "url": "https://www.dbpia.co.kr/Journal/ArticleDetail/NODE01030765?TotalCount=0&Seq=7&isIdentifyAuthor=1&Collection=0&isFullText=0&specificParam=0&SearchMethod=0&Page=1&PageSize=20"
  },
  {
    "id": 606,
    "type": "djournal",
    "title": "Word2Vec으로 생성한 유의어 사전을 이용한 그래프 기반 문서 요약",
    "authors": "박찬민, 이수안, 김진호",
    "venue": "정보통신논문지",
    "date": "2017년"
  },
  {
    "id": 607,
    "type": "djournal",
    "title": "스프레드시트 방식의 OLAP 인터페이스를 위한 시트 분할 기법",
    "authors": "신성현, 이수안, 최훈영, 김진호, 문양세",
    "venue": "정보통신논문지",
    "date": "2007년"
  },
  {
    "id": 116,
    "type": "conference",
    "title": "Query Optimization Techniques for Horizontal View Tables used for Multidimensional Analysis",
    "authors": "Sung-Hyun Shin, Yang-Sae Moon, Jinho Kim, Soo-Ahn Lee, and Sang-Wook Kim",
    "venue": "International Conference on Information and Knowledge Engineering (IKE'08)",
    "date": "2008",
    "abstract": "To support effective analyses in various business applications, On-Line Analytical Processing (OLAP) systems often represent multidimensional data as the horizontal format of tables whose columns are corresponding to values of dimension attributes. (Cross tabulation for statistical data is an example of horizontal tabular form.) These horizontal tables can have a lot of columns. Because conventional DBMSs have the limitation on the maximum number of attributes which tables can have (MS SQLServer and Oracle permit each table to have up to 1,024 columns), horizontal tables cannot be often stored directly into relational database systems. In that case, horizontal tables can be represented by equivalent vertical tables with the form of <attribute name, value> pairs and every queries for horizontal tables should be transformed into the ones for vertical tables. In this paper, we propose various optimization strategies in transforming horizontal table queries to equivalent vertical table ones. To achieve this goal, we first store a horizontal table by using an equivalent vertical table, then we develop various query transformation rules for horizontal table queries. We proposed various alternative query transformation rules for basic relational operators such as selection, projection, and join. (The PIVOT operator which the recent version of MS SQLServer newly provides is used in these transformation rules.) Here, we note that horizontal queries can be transformed/executed in several ways, and their execution times differ from each other. Thus, we propose various optimization strategies that transform horizontal queries to equivalent vertical queries. Finally, we evaluate these methods through experiments and identify optimal transformation strategies.",
    "keywords": "Multidimensional Data, Data Warehouse, PIVOT operation, Query Optimization"
  },
  {
    "id": 117,
    "type": "conference",
    "title": "Relationship Analysis on Academic Achievement and Learning Methods during Vacation: A Data Mining Approach",
    "authors": "Hea-Suk Kim, Yang-Sae Moon, Jinho Kim, Suan Lee, and Woong-Kee Loh",
    "venue": "International Conference. on e-Learning, e-Business, Enterprise Information Systems, and e-Government (EEE'08)",
    "date": "2008",
    "abstract": "Students are educated in various ways such as private tutoring, academic institute lessons, educational broadcasting, and Internet learning sites as well as regular school lessons to promote their academic achievement. These learning methods can affect differently academic achievement over student groups. In this paper, we analyze the effect of learning methods and living style of students during vacation on academic achievement using data mining techniques. To achieve this goal, we first identify various items of learning methods and living style which can affect academic achievement. Students are surveyed over these items through an Internet online site, and the data collected from students are stored into databases. We then present data filtering methods of these collected data to adopt data mining techniques. We also propose the methods of generating decision trees and association rules from the collected student data. Finally, we apply the proposed methods to middle school students in a city of Korea, and we analyze the effect of learning methods during vacation on their academic achievement. We believe that the analysis results presented in this paper would be helpful in establishing the guideline of living style and the studying plans for students during vacation.",
    "keywords": "data mining, academic achievement, learning method, decision tree, association rules"
  }
];

export const getPublicationsByType = (type: PublicationType | 'all'): Publication[] => {
  if (type === 'all') return publications;
  return publications.filter((p) => p.type === type);
};
