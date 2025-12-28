export interface ResearchTopic {
  title: string;
  description: string;
}

export interface ResearchArea {
  slug: string;
  titleKo: string;
  titleEn: string;
  image: string;
  description: string;
  icon: string;
  // Enhanced content
  overview: string;
  keyTechnologies: string[];
  researchTopics: ResearchTopic[];
  achievements: string[];
  keywords: string[];
  relatedCourses?: string[];
  representativeProjects?: string[];
}

export const researchAreas: ResearchArea[] = [
  {
    slug: 'ds',
    titleKo: '데이터과학 & 빅데이터',
    titleEn: 'Data Science & Big Data',
    image: '/assets/images/research/ds.jpg',
    description: 'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.',
    icon: 'et-gears',
    overview: `데이터 과학은 대규모 데이터에서 의미 있는 통찰을 추출하는 융합 학문입니다.
    본 연구실에서는 빅데이터 플랫폼 설계 및 구축, 분산 처리 시스템, 데이터 웨어하우스,
    OLAP(Online Analytical Processing), 데이터 마이닝 등 데이터 과학의 핵심 분야를 연구합니다.

    특히 Hadoop, Spark 기반의 대규모 분산 처리 시스템과 실시간 스트림 데이터 처리 엔진 개발에
    17년 이상의 연구 경험을 보유하고 있으며, 인메모리 데이터베이스 엔진 개발 경력도 갖추고 있습니다.`,
    keyTechnologies: [
      'Apache Hadoop (HDFS, MapReduce, YARN)',
      'Apache Spark (Spark SQL, MLlib, Streaming)',
      'Apache Kafka & Stream Processing',
      'HBase, Hive, ZooKeeper',
      'Data Warehouse & ETL Pipeline',
      'OLAP & Data Cube',
      'PostgreSQL, MySQL, MariaDB',
      'In-Memory Database (Altibase)',
      'Python (Pandas, NumPy, Dask)',
      'R & RStudio',
    ],
    researchTopics: [
      {
        title: '대규모 분산 데이터 처리',
        description: 'MapReduce 및 Spark 기반의 페타바이트급 데이터 분산 병렬 처리 기술 연구',
      },
      {
        title: '실시간 스트림 데이터 처리',
        description: 'CEP(Complex Event Processing) 엔진 및 실시간 데이터 분석 파이프라인 구축',
      },
      {
        title: '데이터 웨어하우스 & OLAP',
        description: '다차원 데이터 모델링, 데이터 큐브 연산, 인터랙티브 분석 시스템 개발',
      },
      {
        title: '센서 데이터베이스 시스템',
        description: 'IoT 센서 데이터의 효율적인 저장, 질의, 분석을 위한 DSMS(Data Stream Management System) 연구',
      },
      {
        title: '데이터 품질 관리',
        description: '데이터 정제, 결측치 처리, 이상치 탐지 및 데이터 품질 평가 기법 연구',
      },
    ],
    achievements: [
      '빅데이터 플랫폼 구축 관련 정부 R&D 과제 다수 수행',
      '인메모리 데이터베이스 엔진 개발 (Altibase, 3년)',
      '실시간 센서 DBMS 엔진 개발 참여 (10억원 규모)',
      '데이터 큐브 분산 병렬 계산 알고리즘 개발',
      '상권 데이터 분석 및 매출 추정 알고리즘 개발',
    ],
    keywords: ['Big Data', 'Data Science', 'Hadoop', 'Spark', 'Data Warehouse', 'OLAP', 'ETL', 'Data Mining', 'Stream Processing', 'Distributed Systems'],
    relatedCourses: ['빅데이터처리', '데이터베이스', '데이터마이닝', '데이터분석을 위한 SQL'],
    representativeProjects: [
      '대용량 센서 스트림 데이터를 실시간으로 처리하는 개방형 센서 DBMS 개발',
      '상권정보 기반 점포수 추정 알고리즘 개발 및 데이터 구축',
      '데이터산업 창업 활성화 및 인재양성 프로그램',
    ],
  },
  {
    slug: 'dl',
    titleKo: '딥러닝 & 머신러닝',
    titleEn: 'Deep Learning & Machine Learning',
    image: '/assets/images/research/dl.jpg',
    description: 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.',
    icon: 'et-layers',
    overview: `딥러닝과 머신러닝은 인공지능의 핵심 기술로, 데이터로부터 패턴을 학습하여
    예측, 분류, 생성 등의 작업을 수행합니다. 본 연구실에서는 다양한 신경망 아키텍처와
    학습 알고리즘을 연구하며, 실제 산업 문제에 적용하는 응용 연구를 수행합니다.

    특히 시계열 데이터 분석, 이상 탐지, 잔여수명(RUL) 예측, 메타러닝 등의 분야에서
    깊이 있는 연구를 진행하고 있으며, TensorFlow, PyTorch, Keras 등 다양한
    딥러닝 프레임워크를 활용한 모델 개발 경험을 보유하고 있습니다.`,
    keyTechnologies: [
      'TensorFlow / Keras',
      'PyTorch / PyTorch Lightning',
      'Transformers (Hugging Face)',
      'CNN, RNN, LSTM, GRU',
      'Transformer, Attention Mechanism',
      'GAN (Generative Adversarial Networks)',
      'AutoEncoder, VAE',
      'Meta-Learning (MAML, Prototypical Networks)',
      'Transfer Learning & Fine-tuning',
      'MLOps (MLflow, Weights & Biases)',
    ],
    researchTopics: [
      {
        title: '시계열 딥러닝',
        description: 'LSTM, Transformer 기반 시계열 예측, 분류, 이상 탐지 모델 개발',
      },
      {
        title: '이상 탐지 (Anomaly Detection)',
        description: 'AutoEncoder, One-Class SVM, Isolation Forest 등을 활용한 비지도 이상 탐지 연구',
      },
      {
        title: '잔여수명 예측 (RUL Prediction)',
        description: '설비 건전성 지표(Health Index) 기반 잔여수명 예측 AI 모델 개발',
      },
      {
        title: '메타러닝 & Few-shot Learning',
        description: '적은 데이터로 빠르게 학습하는 메타러닝 알고리즘 연구',
      },
      {
        title: '멀티모달 딥러닝',
        description: '텍스트, 이미지, 음성 등 다양한 모달리티를 통합하는 멀티모달 학습 연구',
      },
    ],
    achievements: [
      '멀티모달 음성 메타학습 (XVoice) 과학기술정보통신부 R&D (47.5억원)',
      'NeurIPS 2023, 2024 논문 발표',
      '수면 사운드 분석 AI 모델 개발',
      '자율주행 모형차 AI 모델 개발 (메가캡스톤)',
      '제조 설비 이상 탐지 및 예측 모델 다수 개발',
    ],
    keywords: ['Deep Learning', 'Machine Learning', 'Neural Networks', 'CNN', 'RNN', 'Transformer', 'Anomaly Detection', 'Time Series', 'Meta-Learning', 'MLOps'],
    relatedCourses: ['인공지능', '기계학습', '딥러닝', '데이터마이닝'],
    representativeProjects: [
      'XVoice: 멀티모달 음성 메타학습',
      '전류 기반 이상 탐지 및 잔여수명 예측 AI 모델 개발',
      '수면사운드 분석을 위한 인공지능 모델 구축',
    ],
  },
  {
    slug: 'nlp',
    titleKo: '자연어처리',
    titleEn: 'Natural Language Processing',
    image: '/assets/images/research/nlp.jpg',
    description: 'Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.',
    icon: 'et-document',
    overview: `자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 생성할 수 있게 하는
    인공지능의 핵심 분야입니다. 본 연구실에서는 대규모 언어 모델(LLM),
    Text-to-SQL, 질의응답 시스템, 문서 요약, 감성 분석 등 다양한 NLP 태스크를 연구합니다.

    특히 GPT, BERT, LLaMA 등 최신 언어 모델을 활용한 실용적인 응용 시스템 개발에
    주력하고 있으며, 한국어 자연어처리에 특화된 연구도 진행하고 있습니다.`,
    keyTechnologies: [
      'Large Language Models (GPT-4, Claude, LLaMA)',
      'BERT, RoBERTa, ELECTRA',
      'Hugging Face Transformers',
      'LangChain & LlamaIndex',
      'RAG (Retrieval-Augmented Generation)',
      'Text-to-SQL & Semantic Parsing',
      'Named Entity Recognition (NER)',
      'Sentiment Analysis',
      'Text Summarization',
      'Korean NLP (KoBERT, KoGPT)',
    ],
    researchTopics: [
      {
        title: 'Text-to-SQL',
        description: '자연어 질의를 SQL 쿼리로 자동 변환하는 시맨틱 파싱 연구',
      },
      {
        title: 'RAG 시스템',
        description: '검색 증강 생성(RAG) 기반 지식 기반 질의응답 시스템 개발',
      },
      {
        title: 'LLM 파인튜닝 & 프롬프트 엔지니어링',
        description: '도메인 특화 언어 모델 학습 및 효과적인 프롬프트 설계 연구',
      },
      {
        title: '문서 이해 & 정보 추출',
        description: 'NER, 관계 추출, 이벤트 추출 등 문서 분석 기술 연구',
      },
      {
        title: '한국어 자연어처리',
        description: '한국어 특성을 고려한 형태소 분석, 개체명 인식, 감성 분석 연구',
      },
    ],
    achievements: [
      '금융 특화 Text-to-SQL 솔루션 개발',
      'AI 기반 공무원 학습지원 플랫폼 개발',
      '언어모델 기반 SQL 생성 기술 자문',
      'AI 기반 추천 시스템 자문 다수',
    ],
    keywords: ['NLP', 'LLM', 'GPT', 'BERT', 'Text-to-SQL', 'RAG', 'Sentiment Analysis', 'NER', 'Question Answering', 'Korean NLP'],
    relatedCourses: ['자연어처리', '인공지능', '텍스트마이닝'],
    representativeProjects: [
      '금융분야 환경에 대응 가능한 생성형 언어모델 기반 Text-to-SQL 솔루션 서비스',
      '대규모 언어 모델 기반 공무원 수험생의 AI 학습지원 플랫폼',
      '언어모델 기반 SQL 생성 기술 자문',
    ],
  },
  {
    slug: 'cv',
    titleKo: '컴퓨터 비전',
    titleEn: 'Computer Vision',
    image: '/assets/images/research/cv.jpg',
    description: 'Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.',
    icon: 'et-pictures',
    overview: `컴퓨터 비전은 디지털 이미지와 비디오에서 고수준의 이해를 도출하는
    인공지능 분야입니다. 본 연구실에서는 객체 탐지, 영상 분류, 행동 인식,
    비식별화, 영상 생성 등 다양한 컴퓨터 비전 태스크를 연구합니다.

    특히 스마트시티 환경에서의 CCTV 영상 분석, 안전 모니터링, 개인정보 보호를 위한
    비식별화 기술 등 실제 산업 현장에 적용 가능한 연구를 수행하고 있습니다.
    또한 멀티모달 AI 기반 피칭 평가 시스템 개발 등 혁신적인 프로젝트를 진행 중입니다.`,
    keyTechnologies: [
      'CNN (ResNet, EfficientNet, ConvNeXt)',
      'Vision Transformer (ViT, Swin)',
      'Object Detection (YOLO, Faster R-CNN)',
      'Semantic/Instance Segmentation',
      'Video Understanding & Action Recognition',
      'Face Detection & Recognition',
      'Image Generation (Diffusion, GAN)',
      'OpenCV, Albumentations',
      'MMDetection, Detectron2',
      'MediaPipe, OpenPose',
    ],
    researchTopics: [
      {
        title: '객체 탐지 & 추적',
        description: 'YOLO, Transformer 기반 실시간 객체 탐지 및 다중 객체 추적 연구',
      },
      {
        title: '행동 인식 (Action Recognition)',
        description: '비디오 기반 인간 행동 분류 및 이상 행동 탐지 연구',
      },
      {
        title: '영상 비식별화',
        description: '개인정보 보호를 위한 얼굴, 번호판 등 비식별화 딥러닝 모델 개발',
      },
      {
        title: '안전 모니터링',
        description: 'CCTV 기반 안전모 착용 감지, 위험 지역 침입 탐지 등 산업 안전 AI',
      },
      {
        title: '멀티모달 영상 분석',
        description: '표정, 행동, 시선 등 복합 정보를 활용한 발표자 분석 시스템',
      },
    ],
    achievements: [
      '스마트시티 AI 융합 기술 개발 R&D (46억원 규모)',
      '영상 내 개인정보 비식별화 딥러닝 모델 개발',
      'CCTV 기반 건설현장 안전 AI 서비스 개발',
      '멀티모달 AI 기반 IR 피칭 평가 시스템 개발 (2억원)',
      '인종별 피부타입 분석 알고리즘 개발',
    ],
    keywords: ['Computer Vision', 'Object Detection', 'YOLO', 'Image Classification', 'Action Recognition', 'Video Analysis', 'De-identification', 'Multimodal', 'Safety AI'],
    relatedCourses: ['컴퓨터비전', '영상처리', '딥러닝'],
    representativeProjects: [
      '스마트시티 산업 생산성 혁신을 위한 AI융합 기술 개발',
      'CCTV기반 건설산업현장 출입관리 인공지능 서비스 개발',
      '멀티모달 AI 기반 IR 피칭 평가 및 피드백 시스템',
    ],
  },
  {
    slug: 'graphs',
    titleKo: '그래프 & 텐서',
    titleEn: 'Graphs and Tensors',
    image: '/assets/images/research/graphs.jpg',
    description: 'A graph is an abstract data type that is meant to implement the undirected graph and directed graph concepts from the field of graph theory within mathematics.',
    icon: 'et-linegraph',
    overview: `그래프와 텐서 연구는 복잡한 관계 데이터를 효과적으로 모델링하고 분석하는
    분야입니다. 본 연구실에서는 지식 그래프, 그래프 신경망(GNN), 텐서 분해 및
    다차원 데이터 분석 등을 연구합니다.

    특히 지식 그래프 기반 추론, 그래프 임베딩, 링크 예측 등의 연구를 수행하며,
    이를 자연어처리, 추천 시스템, 바이오인포매틱스 등 다양한 응용 분야에 적용하고 있습니다.
    또한 다차원 텐서 데이터의 효율적인 분석을 위한 알고리즘 연구도 진행하고 있습니다.`,
    keyTechnologies: [
      'Graph Neural Networks (GCN, GAT, GraphSAGE)',
      'Knowledge Graph (Neo4j, RDF)',
      'Graph Embedding (Node2Vec, TransE)',
      'PyTorch Geometric',
      'DGL (Deep Graph Library)',
      'NetworkX, igraph',
      'Tensor Decomposition (CP, Tucker)',
      'TensorLy',
      'SPARQL & Graph Query',
      'Ontology & Reasoning',
    ],
    researchTopics: [
      {
        title: '지식 그래프 구축 및 추론',
        description: '도메인 지식을 그래프로 모델링하고 추론 엔진을 통해 새로운 지식 도출',
      },
      {
        title: '그래프 신경망 (GNN)',
        description: 'GCN, GAT 등 그래프 구조 데이터에 대한 딥러닝 모델 연구',
      },
      {
        title: '링크 예측 & 관계 추론',
        description: '그래프 내 누락된 링크 예측 및 엔티티 간 관계 추론 연구',
      },
      {
        title: '텐서 분해 및 분석',
        description: '다차원 데이터의 차원 축소, 패턴 발견을 위한 텐서 분해 기법 연구',
      },
      {
        title: '온톨로지 기반 시스템',
        description: '의미적 상호운용성을 위한 온톨로지 설계 및 시맨틱 웹 기술 연구',
      },
    ],
    achievements: [
      '그래프 데이터베이스 기반 Text-to-SQL 지식베이스 개발',
      '온톨로지 및 에이전트 개발 프로젝트 수행',
      '다차원 텐서 기반 특징 선택 기법 개발 (XVoice 프로젝트)',
      'IEEE TKDE, TPAMI 등 저널 리뷰어',
      '지식 그래프 기반 추천 시스템 연구',
    ],
    keywords: ['Graph Neural Networks', 'Knowledge Graph', 'GNN', 'Tensor Decomposition', 'Link Prediction', 'Ontology', 'Neo4j', 'Graph Embedding', 'Semantic Web'],
    relatedCourses: ['데이터구조', '알고리즘', '데이터베이스', '인공지능'],
    representativeProjects: [
      '지휘결심 지원 및 가상 시뮬레이션을 위한 온톨로지 및 에이전트 개발',
      '금융분야 Text-to-SQL을 위한 그래프 데이터베이스 기반 지식베이스 개발',
    ],
  },
  {
    slug: 'st',
    titleKo: '시공간 데이터',
    titleEn: 'Spatio-Temporal',
    image: '/assets/images/research/st.jpg',
    description: 'Spatiotemporal analysis deals with data that has both spatial and temporal components, enabling understanding of phenomena across space and time.',
    icon: 'et-map',
    overview: `시공간 데이터 분석은 공간과 시간 정보를 동시에 포함하는 데이터를 다루는 분야입니다.
    본 연구실에서는 위치 기반 서비스, 궤적 분석, 실내 측위, 군중 동선 예측 등
    다양한 시공간 데이터 분석 연구를 수행합니다.

    특히 WiFi CSI(Channel State Information)를 활용한 실내 활동 감지,
    스마트시티 환경에서의 인파 관리, 상권 분석 등 실용적인 연구를 진행하고 있습니다.
    또한 시공간 딥러닝 모델을 활용한 예측 및 이상 탐지 연구도 수행하고 있습니다.`,
    keyTechnologies: [
      'Spatio-Temporal Neural Networks (ST-GNN)',
      'LSTM for Sequence Modeling',
      'Trajectory Analysis',
      'Indoor Positioning (WiFi, BLE, UWB)',
      'WiFi CSI (Channel State Information)',
      'GIS (Geographic Information System)',
      'PostGIS, GeoPandas',
      'Folium, Kepler.gl',
      'Time Series Forecasting',
      'Crowd Flow Prediction',
    ],
    researchTopics: [
      {
        title: 'WiFi CSI 기반 활동 감지',
        description: 'WiFi 신호 분석을 통한 비접촉 인간 활동 인식 및 이상 행동 탐지',
      },
      {
        title: '실내 측위 시스템',
        description: 'WiFi, BLE 신호 기반 고정밀 실내 위치 추정 기술 연구',
      },
      {
        title: '군중 동선 분석 및 예측',
        description: '인파 밀집 예측, 최적 경로 도출, 군중 흐름 시각화 연구',
      },
      {
        title: '상권 분석',
        description: '위치 데이터 기반 상권 특성 분석, 매출 추정, 입지 추천 연구',
      },
      {
        title: '시공간 이상 탐지',
        description: '시공간 패턴에서의 이상치 탐지 및 낙상 사고 실시간 감지',
      },
    ],
    achievements: [
      'WiFi CSI 기반 인간 활동 감지 연구 (Triangle Research Program)',
      '인파 관리를 위한 AI 모델 개발',
      '상권 데이터 매출 추정 Imputation 모델 개발 (서울신용보증재단)',
      '스마트 관광도시 비즈니스 모델 구축 (가평군)',
      '생활폐기물 효율적 수거를 위한 최적 경로 도출 (지역리빙랩)',
    ],
    keywords: ['Spatio-Temporal', 'WiFi CSI', 'Indoor Positioning', 'Trajectory Analysis', 'Crowd Flow', 'GIS', 'Location-Based Services', 'Time Series', 'Activity Recognition'],
    relatedCourses: ['데이터베이스', '빅데이터처리', '인공지능'],
    representativeProjects: [
      'WiFi CSI를 이용한 실내외 인간 활동 감지 및 분석',
      '사용자 위치 기반 행동 패턴 분석 AI 모델',
      '상권분석서비스 매출추정 방법론 개선 및 적용 연구',
    ],
  },
  {
    slug: 'asp',
    titleKo: '오디오 음성 처리',
    titleEn: 'Audio & Speech Processing',
    image: '/assets/images/research/asp.jpg',
    description: 'Audio and speech processing involves the analysis, synthesis, and recognition of audio signals and human speech using deep learning and signal processing techniques.',
    icon: 'et-music',
    overview: `오디오 및 음성 처리는 음성 인식, 음성 합성, 화자 인식, 음향 이벤트 감지 등
    소리 신호를 분석하고 생성하는 인공지능 분야입니다. 본 연구실에서는 딥러닝 기반의
    음성 처리 기술과 멀티모달 생체신호 분석 연구를 수행합니다.

    특히 멀티모달 생체신호로부터 음성을 합성하는 기술, 수면 사운드 분석,
    음향 이벤트 감지 등의 연구를 진행하고 있으며, 과학기술정보통신부의
    사람중심인공지능핵심원천기술개발 사업(XVoice)에 참여하고 있습니다.`,
    keyTechnologies: [
      'Speech Recognition (ASR)',
      'Text-to-Speech (TTS)',
      'Speaker Verification/Identification',
      'Audio Event Detection',
      'Librosa, torchaudio',
      'Wav2Vec 2.0, HuBERT',
      'Whisper (OpenAI)',
      'Mel-Spectrogram Analysis',
      'MFCC Feature Extraction',
      'Sound Classification (ESC, UrbanSound)',
    ],
    researchTopics: [
      {
        title: '멀티모달 음성 합성',
        description: 'EEG, EMG 등 생체신호로부터 자연스러운 음성을 합성하는 기술 연구',
      },
      {
        title: '수면 사운드 분석',
        description: '코골이, 수면무호흡 등 수면 중 소리 분석을 통한 수면 상태 분류',
      },
      {
        title: '음향 이벤트 감지',
        description: '환경음 분류, 비명/위험 소리 탐지, 실내 환경 모니터링',
      },
      {
        title: '화자 인식',
        description: '음성을 통한 화자 식별 및 검증 기술 연구',
      },
      {
        title: '음성 품질 향상',
        description: '잡음 제거, 음성 향상, 대역폭 확장 등 음질 개선 연구',
      },
    ],
    achievements: [
      'XVoice: 멀티모달 음성 메타학습 (과기정통부, 47.5억원)',
      '수면 사운드 분석 AI 모델 구축 (베러마인드)',
      'VOICE AI Workshop 2021, 2022 Chair',
      '멀티모달 생체신호 기반 음성 합성 기술 연구',
      '다차원 텐서 기반 특징 선택 기법 개발',
    ],
    keywords: ['Speech Processing', 'Audio Analysis', 'Speech Recognition', 'TTS', 'Sound Classification', 'Multimodal', 'Sleep Sound', 'Speaker Recognition', 'Wav2Vec'],
    relatedCourses: ['인공지능', '신호처리', '딥러닝'],
    representativeProjects: [
      'XVoice: 멀티모달 음성 메타학습',
      '수면사운드 분석을 위한 인공지능 모델 구축',
    ],
  },
];

export const getResearchBySlug = (slug: string): ResearchArea | undefined => {
  return researchAreas.find((r) => r.slug === slug);
};

// 연구 분야별 논문 검색을 위한 키워드 매핑
export const researchKeywordMap: Record<string, string[]> = {
  ds: [
    'big data', 'data science', 'hadoop', 'spark', 'data warehouse', 'olap', 'data cube',
    'etl', 'data mining', 'stream', 'database', 'sql', 'mapreduce', 'distributed',
    '빅데이터', '데이터 마이닝', '데이터베이스', '분산', '스트림'
  ],
  dl: [
    'deep learning', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'attention',
    'autoencoder', 'gan', 'generative', 'meta-learning', 'transfer learning', 'mlops',
    'classification', 'prediction', 'anomaly detection', 'fault', 'time series', 'timeseries',
    '딥러닝', '신경망', '이상 탐지', '시계열', '분류', '예측', '고장', '잔여 수명'
  ],
  nlp: [
    'nlp', 'natural language', 'language model', 'llm', 'bert', 'gpt', 'text-to-sql',
    'sentiment analysis', 'named entity', 'question answering', 'summarization',
    'machine translation', 'word embedding', 'seq2seq', 'dialogue', 'chatbot',
    'text classification', 'text mining', 'document', 'corpus', 'tokeniz',
    '자연어', '언어 모델', '감성 분석', '개체명', '질의응답', '문서', 'sllm',
    '법률 문서', '사용자 의도'
  ],
  cv: [
    'computer vision', 'image', 'video', 'object detection', 'segmentation', 'yolo',
    'face', 'action recognition', 'pose estimation', 'visual', 'cnn', 'vit',
    'de-identification', 'anonymiz', 'camera', 'bionic eye', 'melanoma', 'medical imaging',
    '영상', '이미지', '객체', '얼굴', '행동 인식', '비식별화', '컴퓨터 비전'
  ],
  graphs: [
    'graph', 'knowledge graph', 'gnn', 'tensor', 'network', 'node', 'link prediction',
    'ontology', 'neo4j', 'rdf', 'embedding', 'heterogeneous', 'attention network',
    '그래프', '텐서', '지식', '네트워크', '온톨로지'
  ],
  st: [
    'spatio', 'temporal', 'spatial', 'location', 'gis', 'trajectory', 'gps',
    'indoor', 'positioning', 'wifi', 'csi', 'crowd', 'traffic', 'urban',
    'commercial', 'store', 'revenue', 'sales', 'district',
    '시공간', '위치', '궤적', '실내', '상권', '매출', '교통'
  ],
  asp: [
    'audio', 'speech', 'voice', 'sound', 'acoustic', 'speaker', 'asr', 'tts',
    'wav2vec', 'mel', 'spectrogram', 'music', 'sleep', 'snoring',
    '음성', '음향', '오디오', '화자', '수면', '사운드'
  ],
};
