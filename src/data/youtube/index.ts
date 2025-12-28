export interface YouTubeVideo {
  id: string;
  titleKo: string;
  titleEn: string;
  youtubeId: string;
  description?: string;
  colabUrl?: string;
  pdfUrl?: string;
}

export interface YouTubePlaylist {
  slug: string;
  titleKo: string;
  titleEn: string;
  icon: string;
  videoCount: number;
  playlistId?: string;
  videos: YouTubeVideo[];
}

export const playlists: YouTubePlaylist[] = [
  {
    "slug": "pp",
    "titleKo": "파이썬 프로그래밍",
    "titleEn": "Python Programming",
    "icon": "fa fa-code",
    "videoCount": 9,
    "playlistId": "PL7ZVZgsnLwEEdhCYInwxRpj1Rc4EGmCUc",
    "videos": [
      {
        "id": "01",
        "titleKo": "01 파이썬 프로그래밍 언어",
        "titleEn": "",
        "youtubeId": "mWLoxTvDoDQ",
        "colabUrl": "https://colab.research.google.com/drive/1XmLpgK5z0CvCFW8IivLC64H4FTMy8v3O?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP01.pdf"
      },
      {
        "id": "02",
        "titleKo": "02 변수, 자료형, 연산자",
        "titleEn": "",
        "youtubeId": "a0O-TDHVPo0",
        "colabUrl": "https://colab.research.google.com/drive/17QdUyjooYwmFsq4rBFx6hRtumqudBz2-?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP02.pdf"
      },
      {
        "id": "03",
        "titleKo": "03 문자열",
        "titleEn": "",
        "youtubeId": "98jmfUeAje4",
        "colabUrl": "https://colab.research.google.com/drive/1Hlk7Xt8Nykk0qZP8ZyCWSav982C7s8Ix?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP03.pdf"
      },
      {
        "id": "04",
        "titleKo": "04 리스트, 튜플, 세트, 딕셔너리",
        "titleEn": "",
        "youtubeId": "GTx9gxzUzAg",
        "colabUrl": "https://colab.research.google.com/drive/1ZN5TkfGaKppvieM0mJS1e1NcWefwaV-J?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP04.pdf"
      },
      {
        "id": "05",
        "titleKo": "05 제어문(조건문, 반복문), 에러와 예외처리",
        "titleEn": "",
        "youtubeId": "Dvd_R4DjCho",
        "colabUrl": "https://colab.research.google.com/drive/1p76W2o6vLprAfBsnykuzzsMlBZSYVGUG?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP05.pdf"
      },
      {
        "id": "06",
        "titleKo": "06 입력과 출력",
        "titleEn": "",
        "youtubeId": "_ByfSqzatp4",
        "colabUrl": "https://colab.research.google.com/drive/1CZcYzJSSDfOulHBVLnYcAXy_EQ9vvaw4?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP06.pdf"
      },
      {
        "id": "07",
        "titleKo": "07 함수",
        "titleEn": "",
        "youtubeId": "HlUWhxhC16w",
        "colabUrl": "https://colab.research.google.com/drive/1vXZeQdic41BDKTLg3jfiVoBLAp1AtNoT?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP07.pdf"
      },
      {
        "id": "08",
        "titleKo": "08 객체와 클래스",
        "titleEn": "",
        "youtubeId": "h-OwxPqjMpc",
        "colabUrl": "https://colab.research.google.com/drive/1gdGqlDkoPcBtgffhzlz4-6H4jxVM3_fM?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP08.pdf"
      },
      {
        "id": "09",
        "titleKo": "09 모듈과 패키지",
        "titleEn": "",
        "youtubeId": "F1jKFsbjns0",
        "colabUrl": "https://colab.research.google.com/drive/1wDXsOesu4vWKMVnu-qP1G2aTCwhYBP7Q?usp=sharing",
        "pdfUrl": "/assets/youtubes/pp/PP09.pdf"
      }
    ]
  },
  {
    "slug": "pg",
    "titleKo": "파이썬 게임",
    "titleEn": "Python Game",
    "icon": "fa fa-tasks",
    "videoCount": 5,
    "playlistId": "PL7ZVZgsnLwEH_cLdFK67ygJWPpL2rR8XH",
    "videos": [
      {
        "id": "01",
        "titleKo": "Python Shooting Game ver.2",
        "titleEn": "",
        "youtubeId": "W92RjjptAsM"
      },
      {
        "id": "02",
        "titleKo": "Creating a Spaceship Game with Python",
        "titleEn": "",
        "youtubeId": "TQKxx5WwIe8"
      },
      {
        "id": "03",
        "titleKo": "Creating a Python Racing Car Game with pygame",
        "titleEn": "",
        "youtubeId": "37a7cBmCvB8"
      },
      {
        "id": "04",
        "titleKo": "Creating a Python Game with pygame",
        "titleEn": "",
        "youtubeId": "rtwtOcfYKqc"
      },
      {
        "id": "05",
        "titleKo": "Creating a Shooting Game with Python",
        "titleEn": "",
        "youtubeId": "-e_5sOsKqrU"
      }
    ]
  },
  {
    "slug": "web",
    "titleKo": "웹 프로그래밍",
    "titleEn": "Web Programming",
    "icon": "fa fa-html5",
    "videoCount": 3,
    "playlistId": "PL7ZVZgsnLwEHAehyS4ult4cgRpa7Ll_eU",
    "videos": [
      {
        "id": "01",
        "titleKo": "Web and Internet Concepts",
        "titleEn": "",
        "youtubeId": "Pt5xkFPOPGs"
      },
      {
        "id": "02",
        "titleKo": "HTML Full Tutorial",
        "titleEn": "",
        "youtubeId": "VozMYcCYvtg"
      },
      {
        "id": "03",
        "titleKo": "CSS Full Tutorial",
        "titleEn": "",
        "youtubeId": "J3ef9c-sZ14"
      }
    ]
  },
  {
    "slug": "ds",
    "titleKo": "데이터 과학",
    "titleEn": "Data Science",
    "icon": "fa fa-flask",
    "videoCount": 6,
    "playlistId": "PL7ZVZgsnLwEGnhhjdZ2LH9LyBbDQoxaPk",
    "videos": [
      {
        "id": "01",
        "titleKo": "History of Data",
        "titleEn": "",
        "youtubeId": "CBmGTfeNksw"
      },
      {
        "id": "02",
        "titleKo": "Concept of Data",
        "titleEn": "",
        "youtubeId": "s4OCLDdC5so"
      },
      {
        "id": "03",
        "titleKo": "Open Data",
        "titleEn": "",
        "youtubeId": "bIXyIv-gIbo"
      },
      {
        "id": "04",
        "titleKo": "Data Era",
        "titleEn": "",
        "youtubeId": "98CQHA6ODjM"
      },
      {
        "id": "05",
        "titleKo": "Data Science",
        "titleEn": "",
        "youtubeId": "lcV4WJsshVQ"
      },
      {
        "id": "06",
        "titleKo": "Steps of Data Science",
        "titleEn": "",
        "youtubeId": "tyxVZBbR7oA"
      }
    ]
  },
  {
    "slug": "dc",
    "titleKo": "데이터 수집",
    "titleEn": "Data Collection",
    "icon": "fa fa-gears",
    "videoCount": 23,
    "playlistId": "PL7ZVZgsnLwEFbtQ9LkKkzTBRDkEz3YHsQ",
    "videos": [
      {
        "id": "01",
        "titleKo": "Overview of Data Collection",
        "titleEn": "",
        "youtubeId": "0OY6-04fT9Y"
      },
      {
        "id": "02",
        "titleKo": "Classification of Data Collection",
        "titleEn": "",
        "youtubeId": "xsV2j6SIZ0I"
      },
      {
        "id": "03",
        "titleKo": "Data Collection Procedure",
        "titleEn": "",
        "youtubeId": "8GKhFa4XCgE"
      },
      {
        "id": "04",
        "titleKo": "Methods and Techniques for Data Collection",
        "titleEn": "",
        "youtubeId": "GQPhbXfehLw"
      },
      {
        "id": "05",
        "titleKo": "네이버 웹문서, 지식iN, 뉴스, 블로그, 책, 영화, 쇼핑 데이터 다 가져오기",
        "titleEn": "",
        "youtubeId": "C8-SII6S4Bc"
      },
      {
        "id": "06",
        "titleKo": "대기 오염 데이터 수집",
        "titleEn": "",
        "youtubeId": "Hi9C2Xl_ES8"
      },
      {
        "id": "07",
        "titleKo": "상가 상권 데이터 수집",
        "titleEn": "",
        "youtubeId": "7PzAIjHy-gI"
      },
      {
        "id": "08",
        "titleKo": "강력한 웹 데이터 스크래핑",
        "titleEn": "",
        "youtubeId": "TbPD9Ndnt04"
      },
      {
        "id": "09",
        "titleKo": "전 세계 웹사이트 순위 스크래핑",
        "titleEn": "",
        "youtubeId": "2iGIlEQFsuM"
      },
      {
        "id": "10",
        "titleKo": "네이버 뮤직 TOP 100 음악 순위 스크래핑",
        "titleEn": "",
        "youtubeId": "uuJMeBojbBk"
      },
      {
        "id": "11",
        "titleKo": "네이버 영화 리뷰 스크래핑",
        "titleEn": "",
        "youtubeId": "s-BRTDfyp5E"
      },
      {
        "id": "12",
        "titleKo": "네이버 뉴스 기사 스크래핑",
        "titleEn": "",
        "youtubeId": "In5KCBqzViU"
      },
      {
        "id": "13",
        "titleKo": "네이버 블로그 스크래핑",
        "titleEn": "",
        "youtubeId": "gy38Kh2arHY"
      },
      {
        "id": "14",
        "titleKo": "셀레니움과 웹 드라이버 설치부터 웹사이트 스크래핑 따라하기",
        "titleEn": "",
        "youtubeId": "dDEESB4Iw8g"
      },
      {
        "id": "15",
        "titleKo": "네이버 웹툰 스크래핑",
        "titleEn": "",
        "youtubeId": "mpEsdbT52Pg"
      },
      {
        "id": "16",
        "titleKo": "CGV 영화 리뷰 스크래핑",
        "titleEn": "",
        "youtubeId": "qil0BtP8kLY"
      },
      {
        "id": "17",
        "titleKo": "구글 이미지 스크래핑",
        "titleEn": "",
        "youtubeId": "pQ7dOg9c4NI"
      },
      {
        "id": "18",
        "titleKo": "네이버 뉴스 댓글 스크래핑",
        "titleEn": "",
        "youtubeId": "IFyhTEBSto0"
      },
      {
        "id": "19",
        "titleKo": "국회의원 정보, 관련 뉴스, 회의록 스크래핑",
        "titleEn": "",
        "youtubeId": "LXZdQCVTUlc"
      },
      {
        "id": "20",
        "titleKo": "네이버 급상승 검색어 스크래핑",
        "titleEn": "",
        "youtubeId": "b5SynNjL03c"
      },
      {
        "id": "21",
        "titleKo": "Get Email Contents into Excel using VBA",
        "titleEn": "",
        "youtubeId": "UoZr2xziDsc"
      },
      {
        "id": "22",
        "titleKo": "Scraping Naver with Excel",
        "titleEn": "",
        "youtubeId": "RIQjLpsNqbg"
      },
      {
        "id": "23",
        "titleKo": "Naver Blog Web Scraping with Python",
        "titleEn": "",
        "youtubeId": "cB8bRCTgqJ8"
      }
    ]
  },
  {
    "slug": "da",
    "titleKo": "데이터 분석",
    "titleEn": "Data Analysis",
    "icon": "fa fa-wrench",
    "videoCount": 11,
    "playlistId": "PL7ZVZgsnLwEEZcVusN-fV_sJhQHq833OS",
    "videos": [
      {
        "id": "01",
        "titleKo": "Exploratory Data Analysis",
        "titleEn": "",
        "youtubeId": "0eCbAYX-_UQ"
      },
      {
        "id": "02",
        "titleKo": "Data Preprocessing, Quality, Techniques",
        "titleEn": "",
        "youtubeId": "qGg3DOKPUTM"
      },
      {
        "id": "03",
        "titleKo": "Data Cleaning",
        "titleEn": "",
        "youtubeId": "tMkYyMUYldQ"
      },
      {
        "id": "04",
        "titleKo": "Data Integration",
        "titleEn": "",
        "youtubeId": "eBuRFT6jb1U"
      },
      {
        "id": "05",
        "titleKo": "Data Reduction",
        "titleEn": "",
        "youtubeId": "24xeF0P9ePM"
      },
      {
        "id": "06",
        "titleKo": "Data Transformation",
        "titleEn": "",
        "youtubeId": "tnAMYJu2vWQ"
      },
      {
        "id": "07",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "9eP4bMqJYX4"
      },
      {
        "id": "08",
        "titleKo": "데이터 탐색, 정제, 변환을 도와주는 막강한 전처리 도구",
        "titleEn": "",
        "youtubeId": "oRH-1RG8oQY"
      },
      {
        "id": "09",
        "titleKo": "데이터 과학 핵심 도구, 고차원 배열 생성, 처리, 연산 집계",
        "titleEn": "",
        "youtubeId": "mirZPrWwvao",
        "colabUrl": "https://colab.research.google.com/drive/1qEBbLwNJ0FZA6h1BWHm5wu4mrJhbg3ty?usp=sharing",
        "pdfUrl": "/assets/youtubes/da/NumPy.pdf"
      },
      {
        "id": "10",
        "titleKo": "데이터 과학의 핵심 패키지, 데이터 처리, 연산, 집계",
        "titleEn": "",
        "youtubeId": "lG8pEwvYwCw",
        "colabUrl": "https://colab.research.google.com/drive/1nt8EA_2tC3DjAoUcqm89xy0RyF5qBL_y?usp=sharing",
        "pdfUrl": "/assets/youtubes/da/Pandas.pdf"
      },
      {
        "id": "11",
        "titleKo": "(using Pandas & Seaborn)",
        "titleEn": "",
        "youtubeId": "wBQnyesc9bU"
      }
    ]
  },
  {
    "slug": "dv",
    "titleKo": "데이터 시각화",
    "titleEn": "Data Visualization",
    "icon": "fa fa-bar-chart",
    "videoCount": 10,
    "playlistId": "PL7ZVZgsnLwEGR11m7oLOSa0pBWCc3TMaL",
    "videos": [
      {
        "id": "01",
        "titleKo": "Concepts and History of Data Visualization",
        "titleEn": "",
        "youtubeId": "LjWmZkW1fy0"
      },
      {
        "id": "02",
        "titleKo": "Data Visualization Technique",
        "titleEn": "",
        "youtubeId": "m-vYq5rZS3M"
      },
      {
        "id": "03",
        "titleKo": "Interactive Data Visualization",
        "titleEn": "",
        "youtubeId": "2xufjZJ4jJw"
      },
      {
        "id": "04",
        "titleKo": "Visualization Design Principles",
        "titleEn": "",
        "youtubeId": "CC5OmZ8gTtA"
      },
      {
        "id": "05",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "5DfACSYgP0U",
        "colabUrl": "https://colab.research.google.com/drive/1ZRBjyNnrQYOmcuT-PQOzrxJG0HhldaIH?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Matplotlib.pdf"
      },
      {
        "id": "06",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "gWhwAY5Q9Ec",
        "colabUrl": "https://colab.research.google.com/drive/1PFx8PEvm9FEyVph8QZmn4Zevi_51WDWq?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Seaborn.pdf"
      },
      {
        "id": "07",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "wBQnyesc9bU",
        "colabUrl": "https://colab.research.google.com/drive/1o7_8S1w0Mmvw6QLE5FNhvTrzJ6Oblhh6?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Titanic.pdf"
      },
      {
        "id": "08",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "Jt9nycjTGZs",
        "colabUrl": "https://colab.research.google.com/drive/1jQBf8NMVBmlg89hdN5KJwb8sMpkHJkPy?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Folium.pdf"
      },
      {
        "id": "09",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "qt6rtokj7rw",
        "colabUrl": "https://colab.research.google.com/drive/1sEnO-tXYBH2YpQJLM3DLZf-Da6Rvpvz7?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Bokeh.pdf"
      },
      {
        "id": "10",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "i-xbj0owgEE",
        "colabUrl": "https://colab.research.google.com/drive/1o7_8S1w0Mmvw6QLE5FNhvTrzJ6Oblhh6?usp=sharing",
        "pdfUrl": "/assets/youtubes/dv/Plotly.pdf"
      }
    ]
  },
  {
    "slug": "db",
    "titleKo": "데이터베이스",
    "titleEn": "Database",
    "icon": "fa fa-database",
    "videoCount": 6,
    "playlistId": "PL7ZVZgsnLwEEMDG02R-ThBc1cDTdT97z6",
    "videos": [
      {
        "id": "01",
        "titleKo": "Database Overview",
        "titleEn": "",
        "youtubeId": "HmVAN1xq9KI"
      },
      {
        "id": "02",
        "titleKo": "Data Modeling",
        "titleEn": "",
        "youtubeId": "7GCz1HnRZsM"
      },
      {
        "id": "03",
        "titleKo": "Relational Data Operators",
        "titleEn": "",
        "youtubeId": "Zv3qsPxY_qk"
      },
      {
        "id": "04",
        "titleKo": "Structured Query Language",
        "titleEn": "",
        "youtubeId": "gNBDl-apgoQ"
      },
      {
        "id": "05",
        "titleKo": "Database Development and Types",
        "titleEn": "",
        "youtubeId": "7FOFueWyUS8"
      },
      {
        "id": "06",
        "titleKo": "SQL Full Tutorial Course using MySQL Database",
        "titleEn": "",
        "youtubeId": "vgIc4ctNFbc"
      }
    ]
  },
  {
    "slug": "bd",
    "titleKo": "빅데이터",
    "titleEn": "Big Data",
    "icon": "fa fa-server",
    "videoCount": 7,
    "playlistId": "PL7ZVZgsnLwEF4wvGL2OHZbodXCLWM2Uhn",
    "videos": [
      {
        "id": "01",
        "titleKo": "Big Data Concept",
        "titleEn": "",
        "youtubeId": "Zhw-0Vw4jZc"
      },
      {
        "id": "02",
        "titleKo": "Planning and Leveraging of Big Data",
        "titleEn": "",
        "youtubeId": "vSJWS8GLOTs"
      },
      {
        "id": "03",
        "titleKo": "Big Data Collection Technology",
        "titleEn": "",
        "youtubeId": "YR_0fl2_HEY"
      },
      {
        "id": "04",
        "titleKo": "Big Data Storage Technology",
        "titleEn": "",
        "youtubeId": "N2-_el-rH1k"
      },
      {
        "id": "05",
        "titleKo": "Big Data Processing Technology",
        "titleEn": "",
        "youtubeId": "HmYiUXxqVk0"
      },
      {
        "id": "06",
        "titleKo": "Big Data Analysis Technology",
        "titleEn": "",
        "youtubeId": "km_-Vhj3PrM"
      },
      {
        "id": "07",
        "titleKo": "Big Data Representation Technology",
        "titleEn": "",
        "youtubeId": "jmcr0Oz49lI"
      }
    ]
  },
  {
    "slug": "ml",
    "titleKo": "머신러닝",
    "titleEn": "Machine Learning",
    "icon": "fa fa-gears",
    "videoCount": 14,
    "playlistId": "PL7ZVZgsnLwEEd3-h-jySLz4wT154r7VVr",
    "videos": [
      {
        "id": "01",
        "titleKo": "Machine Learning",
        "titleEn": "",
        "youtubeId": "rWz582-yKuQ",
        "colabUrl": "https://colab.research.google.com/drive/1_Ri4Bv_PWSgy-O8h_pj3e-SBk1AXi_Mn",
        "pdfUrl": "/assets/youtubes/ml/ML01.pdf"
      },
      {
        "id": "02",
        "titleKo": "scikit-learn",
        "titleEn": "",
        "youtubeId": "eVxGhCRN-xA",
        "colabUrl": "https://colab.research.google.com/drive/10yp9-8ZtDjfVuh0H1lU5N9nA1y60eLzj",
        "pdfUrl": "/assets/youtubes/ml/ML02.pdf"
      },
      {
        "id": "03",
        "titleKo": "Linear Models",
        "titleEn": "",
        "youtubeId": "KLgjSGrl_WI",
        "colabUrl": "https://colab.research.google.com/drive/1p6piFoPCAt2jgQ6FWCDearaB6bojNlJE",
        "pdfUrl": "/assets/youtubes/ml/ML03.pdf"
      },
      {
        "id": "04",
        "titleKo": "Logistic Regression",
        "titleEn": "",
        "youtubeId": "5HdxWe8T4sQ",
        "colabUrl": "https://colab.research.google.com/drive/19yh1SDvichat9wCC2_Dc1V4-fNpMeOJx",
        "pdfUrl": "/assets/youtubes/ml/ML04.pdf"
      },
      {
        "id": "05",
        "titleKo": "Support Vector Machine",
        "titleEn": "",
        "youtubeId": "dGjBhSHW9lg",
        "colabUrl": "https://colab.research.google.com/drive/1tTbmeS0Bp0_NkFDsT1DvwvnRjkZUY9Yz",
        "pdfUrl": "/assets/youtubes/ml/ML05.pdf"
      },
      {
        "id": "06",
        "titleKo": "k Nearest Neighbor",
        "titleEn": "",
        "youtubeId": "R9hx09guvzA",
        "colabUrl": "https://colab.research.google.com/drive/1PA3jt6Tn5srFDrBP5byoMcskyJ_ACopE",
        "pdfUrl": "/assets/youtubes/ml/ML06.pdf"
      },
      {
        "id": "07",
        "titleKo": "Naive Bayes",
        "titleEn": "",
        "youtubeId": "4q2TXGc5HcA",
        "colabUrl": "https://colab.research.google.com/drive/1HjsftqRzYSVIipR_OJmnzfioPUwd28Tw",
        "pdfUrl": "/assets/youtubes/ml/ML07.pdf"
      },
      {
        "id": "08",
        "titleKo": "Decision Tree",
        "titleEn": "",
        "youtubeId": "YEt0ViG_VXk",
        "colabUrl": "https://colab.research.google.com/drive/1nhGXYUY-1TK1NmvzSGwghnG3TjU41bEv",
        "pdfUrl": "/assets/youtubes/ml/ML08.pdf"
      },
      {
        "id": "09",
        "titleKo": "Ensemble",
        "titleEn": "",
        "youtubeId": "5PX1ivMiLMA",
        "colabUrl": "https://colab.research.google.com/drive/1Li8YxdE0rIZdtdo_EkyZ0YZp1ZI8WT0q",
        "pdfUrl": "/assets/youtubes/ml/ML09.pdf"
      },
      {
        "id": "10",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "4Jz4_IOgS4c",
        "colabUrl": "https://colab.research.google.com/drive/1cVC0gnrdN_U8zms1RdW_EKsqTDQU1SOd",
        "pdfUrl": "/assets/youtubes/ml/ML09.1.pdf"
      },
      {
        "id": "11",
        "titleKo": "Clustering",
        "titleEn": "",
        "youtubeId": "jn2HNDJmBZ8",
        "colabUrl": "https://colab.research.google.com/drive/1kLL1Dkpt7NPgOv9au_MOa4zwIdNWVuqJ",
        "pdfUrl": "/assets/youtubes/ml/ML10.pdf"
      },
      {
        "id": "12",
        "titleKo": "Manifold Learning",
        "titleEn": "",
        "youtubeId": "jBOVTimnlA8",
        "colabUrl": "https://colab.research.google.com/drive/1oAvgWY5JUBCyqkl3TIz2d6f_A9VHqSRW",
        "pdfUrl": "/assets/youtubes/ml/ML11.pdf"
      },
      {
        "id": "13",
        "titleKo": "Decomposition",
        "titleEn": "",
        "youtubeId": "tI0D15lH2iU",
        "colabUrl": "https://colab.research.google.com/drive/1uAMLUiujcKbtEPlr27RUzvQbiAoiZfHz",
        "pdfUrl": "/assets/youtubes/ml/ML12.pdf"
      },
      {
        "id": "14",
        "titleKo": "Recommender System",
        "titleEn": "",
        "youtubeId": "6TP51jvjLsE",
        "colabUrl": "https://colab.research.google.com/drive/1j39NJ-HSyAeRn-QekORfpkLcW3h5cg1W",
        "pdfUrl": "/assets/youtubes/ml/ML13.pdf"
      }
    ]
  },
  {
    "slug": "dl",
    "titleKo": "딥러닝",
    "titleEn": "Deep Learning",
    "icon": "fa fa-sliders",
    "videoCount": 12,
    "playlistId": "PL7ZVZgsnLwEHTS9YdnJw3fYWRqy46cOVB",
    "videos": [
      {
        "id": "01",
        "titleKo": "딥러닝 개요 Deep Learning Overview",
        "titleEn": "",
        "youtubeId": "0r_QueHF3Qg"
      },
      {
        "id": "02",
        "titleKo": "신경망 기초수학",
        "titleEn": "",
        "youtubeId": "RZB6fwX_ixY",
        "colabUrl": "https://colab.research.google.com/drive/13ZwqCpFknt022ONPP1MXwt2kkwuCB4jH?usp=sharing"
      },
      {
        "id": "03",
        "titleKo": "신경망 데이터 표현",
        "titleEn": "",
        "youtubeId": "-5XGjsjec3Y",
        "colabUrl": "https://colab.research.google.com/drive/1ZM-bVqUpOuauK6KBStMGaDv94iuqox45?usp=sharing"
      },
      {
        "id": "04",
        "titleKo": "신경망 구조",
        "titleEn": "",
        "youtubeId": "kHXrjyqyfE4",
        "colabUrl": "https://colab.research.google.com/drive/13mKh25DM0U-zzmMRz9i6Xg8XZIMvxJad?usp=sharing"
      },
      {
        "id": "05",
        "titleKo": "모델 학습과 손실 함수",
        "titleEn": "",
        "youtubeId": "9flte5nLbw8",
        "colabUrl": "https://colab.research.google.com/drive/12dWiy-3GRLTTdngonQEAKssRPUw4x8g4?usp=sharing"
      },
      {
        "id": "06",
        "titleKo": "경사하강법 Gradient Decent",
        "titleEn": "",
        "youtubeId": "GwR1ivsUiAI",
        "colabUrl": "https://colab.research.google.com/drive/1IAk_GCNjaOEWTsGrBLa-M_LrFSiMGYuV?usp=sharing"
      },
      {
        "id": "07",
        "titleKo": "신경망 학습",
        "titleEn": "",
        "youtubeId": "yocALgANi28",
        "colabUrl": "https://colab.research.google.com/drive/1hk4I8R9TidWzFQrKTJjuij54kN_QC3jY?usp=sharing"
      },
      {
        "id": "08",
        "titleKo": "오차역전파 Backpropagation",
        "titleEn": "",
        "youtubeId": "3OLh7bb_53k",
        "colabUrl": "https://colab.research.google.com/drive/1XTRcNbxFCqXbi5LJPLNfN6PEDPUvS8Qk?usp=sharing"
      },
      {
        "id": "09",
        "titleKo": "딥러닝 학습 기술",
        "titleEn": "",
        "youtubeId": "3WzkrGyZhFo",
        "colabUrl": "https://colab.research.google.com/drive/1zx9k16VII8MZ3_n6VcnsBeiOoQQxavBF?usp=sharing"
      },
      {
        "id": "10",
        "titleKo": "CNN 컨볼루션 신경망",
        "titleEn": "",
        "youtubeId": "OAksbx2bTVc",
        "colabUrl": "https://colab.research.google.com/drive/1xhL09gss_KwPwJkYf2iBxg4zbuJe_XIE?usp=sharing"
      },
      {
        "id": "11",
        "titleKo": "RNN 순환신경망",
        "titleEn": "",
        "youtubeId": "P9xMyjBYl_g"
      },
      {
        "id": "12",
        "titleKo": "RNN 순환신경망 (실습)",
        "titleEn": "",
        "youtubeId": "r92QktoXsas",
        "colabUrl": "https://colab.research.google.com/drive/1OEOdA2JEH2V9zrgqFYywvwxUxdcZGyg5?usp=sharing"
      }
    ]
  },
  {
    "slug": "dlf",
    "titleKo": "딥러닝 프레임워크",
    "titleEn": "Deep Learning Framework",
    "icon": "fa fa-magic",
    "videoCount": 20,
    "playlistId": "PL7ZVZgsnLwEHGS6EId3B_AnRYSCi_35rj",
    "videos": [
      {
        "id": "01",
        "titleKo": "TensorFlow",
        "titleEn": "",
        "youtubeId": "B961QM47g64",
        "colabUrl": "https://colab.research.google.com/drive/10Ro_I62m_crG1Xg4FO_kJ0GAUBT2HlPh?usp=sharing"
      },
      {
        "id": "02",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "ZmlqsOidnWw",
        "colabUrl": "https://colab.research.google.com/drive/17E9JLj0cDZF6j7qEfbFdrrjXVMi4pxlk?usp=sharing"
      },
      {
        "id": "03",
        "titleKo": "Keras",
        "titleEn": "",
        "youtubeId": "mzOpojTpliA",
        "colabUrl": "https://colab.research.google.com/drive/1TKOlZDI16mmcnPDY1ymbiznRo5gcI1ZT?usp=sharing"
      },
      {
        "id": "04",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "utP3gh9DZI8",
        "colabUrl": "https://colab.research.google.com/drive/1btPUxWBcHkmhbHz8VcWi-VvgIzxOq0Xy?usp=sharing"
      },
      {
        "id": "05",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "uc4Kh2MdKgI",
        "colabUrl": "https://colab.research.google.com/drive/1q3UiR96pryKIDJdf7pxHTYqKvSd-ftN8?usp=sharing"
      },
      {
        "id": "06",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "M_65kGUWB2s",
        "colabUrl": "https://colab.research.google.com/drive/1HRzZAW88LJuIoHYfGILciIRMRju5OfvS?usp=sharing"
      },
      {
        "id": "07",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "E3HhuCjyZoA",
        "colabUrl": "https://colab.research.google.com/drive/1kTkDk_5lAqVxaY6lVEGF4Z8w1rMb1aG4?usp=sharing"
      },
      {
        "id": "08",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "xrfQAMwHysc",
        "colabUrl": "https://colab.research.google.com/drive/1VKuhGaEgksuqoK-H_VOn8TmaOBjx2U1W?usp=sharing"
      },
      {
        "id": "09",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "WIdALlK3wXg",
        "colabUrl": "https://colab.research.google.com/drive/1yipTP0a8bdBch1ivCV9oQMrpIxHq8jZf?usp=sharing"
      },
      {
        "id": "10",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "sQVhs16SDAI",
        "colabUrl": "https://colab.research.google.com/drive/1_7GCqvwci4BWl35tbVr5NyNj9gODzwPd?usp=sharing"
      },
      {
        "id": "11",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "PGI84VwTT-4",
        "colabUrl": "https://colab.research.google.com/drive/10tSxnkKO6hf9wgTRJEq1DeS_YBo7lkYM?usp=sharing"
      },
      {
        "id": "12",
        "titleKo": "Transfer Learning",
        "titleEn": "",
        "youtubeId": "BJwcMFAKqXo",
        "colabUrl": "https://colab.research.google.com/drive/18Fo8SI66od6NVrlxz8B3uQ1ncDQsXcIp?usp=sharing"
      },
      {
        "id": "13",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "FBbnbAbpKX8",
        "colabUrl": "https://colab.research.google.com/drive/1FykGuiZ9Y0G3zJhVPHrMZCt_0kVsacKK?usp=sharing"
      },
      {
        "id": "14",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "13HOC0z6WLM",
        "colabUrl": "https://colab.research.google.com/drive/1qGmut4DLqj1hgiJur9wjxFWKtbvHSxqt?usp=sharing"
      },
      {
        "id": "15",
        "titleKo": "PyTorch",
        "titleEn": "",
        "youtubeId": "C1P7PaIeKvU",
        "colabUrl": "https://colab.research.google.com/drive/1yURN8yVBax7qoHttCZoDkD93TxWdaZK_?usp=sharing"
      },
      {
        "id": "16",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "9AMeeRL9Wmc",
        "colabUrl": "https://colab.research.google.com/drive/1Wqo--p1hkZheq8mPzP2zzCToaARiEBpx?usp=sharing"
      },
      {
        "id": "17",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "IwLOWwrz26w",
        "colabUrl": "https://colab.research.google.com/drive/1Xtqqd-SJ4ySNyNU_HJPbbo3XnA1puq0Q?usp=sharing"
      },
      {
        "id": "18",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "E-LodDU6pIA",
        "colabUrl": "https://colab.research.google.com/drive/1-dh19UM7nBZIP_essHwUAUTX7OY0W87R?usp=sharing"
      },
      {
        "id": "19",
        "titleKo": "Transfer Learning",
        "titleEn": "",
        "youtubeId": "szfjmRYPX-4",
        "colabUrl": "https://colab.research.google.com/drive/16Q0AGU6X0d1o9V9Gjw9QosGSE2Sj9eny?usp=sharing"
      },
      {
        "id": "20",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "izQZNWpOrGU",
        "colabUrl": "https://colab.research.google.com/drive/1-0cRgo7FWwnJsWdbijwY04hNewBQ5Z2-?usp=sharing"
      }
    ]
  },
  {
    "slug": "cv",
    "titleKo": "컴퓨터 비전",
    "titleEn": "Computer Vision",
    "icon": "fa fa-desktop",
    "videoCount": 19,
    "playlistId": "PL7ZVZgsnLwEElur9LpRW9w3NYGVgA3n6t",
    "videos": [
      {
        "id": "01",
        "titleKo": "이미지 입출력, 컬러 공간, 도형 그리기",
        "titleEn": "",
        "youtubeId": "-zSi7nbvejA",
        "colabUrl": "https://colab.research.google.com/drive/1iXOYMKO0-jHS9BtQx0IOE5zvQmPAkd-d?usp=sharing"
      },
      {
        "id": "02",
        "titleKo": "Image Processing",
        "titleEn": "",
        "youtubeId": "XLkMfXz3r4g",
        "colabUrl": "https://colab.research.google.com/drive/1eErrGFvEpYSzjGxnvKQozC5T0NI1td9f?usp=sharing"
      },
      {
        "id": "03",
        "titleKo": "Image Operation",
        "titleEn": "",
        "youtubeId": "FTSgFFQGkho",
        "colabUrl": "https://colab.research.google.com/drive/12DN_EFMhpltsERABoi9D174sWXKtT4LC?usp=sharing"
      },
      {
        "id": "04",
        "titleKo": "Image Thresholding",
        "titleEn": "",
        "youtubeId": "fNq-CluTJmU",
        "colabUrl": "https://colab.research.google.com/drive/1n1OsVCtw_IaGvU7iNd_Xkaf9fTC9hKX5?usp=sharing"
      },
      {
        "id": "05",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "YwDFKQF7R2I",
        "colabUrl": "https://colab.research.google.com/drive/1AcgwNZlMfvQk7ACEJmdA5W07DNC5nAko?usp=sharing"
      },
      {
        "id": "06",
        "titleKo": "Morphological Transformations)",
        "titleEn": "",
        "youtubeId": "gAB5-knzKNo",
        "colabUrl": "https://colab.research.google.com/drive/1MWyhk6tZUsjjJK_GJHtZetguHZspcKN9?usp=sharing"
      },
      {
        "id": "07",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "zBGne43CNBA",
        "colabUrl": "https://colab.research.google.com/drive/1uZcqz-E_pAJVpZsNKb3PtQJlsLIinDVo?usp=sharing"
      },
      {
        "id": "08",
        "titleKo": "Histogram",
        "titleEn": "",
        "youtubeId": "_VJleCR-1As",
        "colabUrl": "https://colab.research.google.com/drive/1Zs5weClEWAxBl-qMvvhKO8QuTNwnYkLy?usp=sharing"
      },
      {
        "id": "09",
        "titleKo": "CNN, VGGNet, GoogLeNet, ResNet",
        "titleEn": "",
        "youtubeId": "BEfsSIOL-8k",
        "colabUrl": "https://colab.research.google.com/drive/1AjUXL50x3JRhWUx4FpLPqKewIZOoSwBr?usp=sharing"
      },
      {
        "id": "10",
        "titleKo": "Object Detection - YOLOv3",
        "titleEn": "",
        "youtubeId": "5ev0MMBzY3E",
        "colabUrl": "https://colab.research.google.com/drive/1UCYSgGzDiV0C5UKOcMd9PSuqLiI6SyhK?usp=sharing"
      },
      {
        "id": "11",
        "titleKo": "Image Segmentation - Mask R-CNN",
        "titleEn": "",
        "youtubeId": "argrFpobxYE",
        "colabUrl": "https://colab.research.google.com/drive/15d6ND25Sldt2k22d99dM9pwNQ9zn8tlc?usp=sharing"
      },
      {
        "id": "12",
        "titleKo": "Autoencoder",
        "titleEn": "",
        "youtubeId": "0PqWUsKPZdQ",
        "colabUrl": "https://colab.research.google.com/drive/13W6jA7N61NS-YPRn_jADIqBrMmIYart-?usp=sharing"
      },
      {
        "id": "13",
        "titleKo": "Denoise",
        "titleEn": "",
        "youtubeId": "QUSJ5dTLd8E",
        "colabUrl": "https://colab.research.google.com/drive/1YIvHWN5YBbMmlaecLu8wzoBfpD-ur8Ga?usp=sharing"
      },
      {
        "id": "14",
        "titleKo": "Variational Autoencoder",
        "titleEn": "",
        "youtubeId": "1O80j29vwjY",
        "colabUrl": "https://colab.research.google.com/drive/1SsmYPVepmNpAEpG0I2OuHQx8C6nx2XYx?usp=sharing"
      },
      {
        "id": "15",
        "titleKo": "Generative Adversarial Networks",
        "titleEn": "",
        "youtubeId": "oK4SrEdimaU",
        "colabUrl": "https://colab.research.google.com/drive/19qjn40qltSBO5CKt4sXwS6m2xsnB3x8R?usp=sharing"
      },
      {
        "id": "16",
        "titleKo": "DeepDream",
        "titleEn": "",
        "youtubeId": "Lmp6M2x2lig",
        "colabUrl": "https://colab.research.google.com/drive/1yZ1LI1SOXuQUQ9KPONTz5gJmVb3N-ruB?usp=sharing"
      },
      {
        "id": "17",
        "titleKo": "Neural Style Transfer",
        "titleEn": "",
        "youtubeId": "l_jk13ChzP4",
        "colabUrl": "https://colab.research.google.com/drive/1j3MmG2rg6mRBLiDyUsl9SSB4IhAz4Gyk?usp=sharing"
      },
      {
        "id": "18",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "pYN1tCmn4V0",
        "colabUrl": "https://colab.research.google.com/drive/1aBqrroDvd5xS8htFo9IgJ4PFeV5zn1rG?usp=sharing"
      },
      {
        "id": "19",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "kZBousGA0xg",
        "colabUrl": "https://colab.research.google.com/drive/1xg12EtoBTKKcQ2Y-bBV2fCwWWmumbdYr?usp=sharing"
      }
    ]
  },
  {
    "slug": "nlp",
    "titleKo": "자연어 처리",
    "titleEn": "Natural Language Processing",
    "icon": "fa fa-search",
    "videoCount": 17,
    "playlistId": "PL7ZVZgsnLwEEoHQAElEPg7l7T6nt25I3N",
    "videos": [
      {
        "id": "01",
        "titleKo": "Natural Language Processing",
        "titleEn": "",
        "youtubeId": "2e9wnwuAVv0",
        "colabUrl": "https://colab.research.google.com/drive/1UJ36KTBTgw8fvBBvsdQjx-OUP0YrGNet?usp=sharing"
      },
      {
        "id": "02",
        "titleKo": "Keyword Analysis",
        "titleEn": "",
        "youtubeId": "5P6nG8xHKbU",
        "colabUrl": "https://colab.research.google.com/drive/1HdLLGVY-59yc8nMVdFdKRXrepxqcNcAD?usp=sharing"
      },
      {
        "id": "03",
        "titleKo": "Cluster Analysis",
        "titleEn": "",
        "youtubeId": "YJSHBQj8zbU",
        "colabUrl": "https://colab.research.google.com/drive/10YlHniw_tI3iYGD61ZPWAsaT-2Z6gN6f?usp=sharing"
      },
      {
        "id": "04",
        "titleKo": "Document Classification",
        "titleEn": "",
        "youtubeId": "xegxbgsnYko",
        "colabUrl": "https://colab.research.google.com/drive/1NlSZKwocO_9Z6Tbw7X4v9YsISHAZfJlK?usp=sharing"
      },
      {
        "id": "05",
        "titleKo": "Semantic Network Analysis",
        "titleEn": "",
        "youtubeId": "jnoKa44OZv8",
        "colabUrl": "https://colab.research.google.com/drive/18az5ur4JDVwxJz9nQfLMylVJM1fG1iND?usp=sharing"
      },
      {
        "id": "06",
        "titleKo": "Topic Modeling",
        "titleEn": "",
        "youtubeId": "Xt607xhpF6U",
        "colabUrl": "https://colab.research.google.com/drive/1mIm9o41JOa-oFodaj0j26SjLx6y1xNVD?usp=sharing"
      },
      {
        "id": "07",
        "titleKo": "Embedding",
        "titleEn": "",
        "youtubeId": "hR8Rvp-YNGg",
        "colabUrl": "https://colab.research.google.com/drive/1mBJgcpLrcyJeytlgarr2Gftff8JROPd8?usp=sharing"
      },
      {
        "id": "08",
        "titleKo": "Recurrent Neural Network",
        "titleEn": "",
        "youtubeId": "hGuUFVZ_tSs",
        "colabUrl": "https://colab.research.google.com/drive/1PELxaO_9Pzov6svb0LlVLtCkNlTb_dFi?usp=sharing"
      },
      {
        "id": "09",
        "titleKo": "Convolution Neural Network",
        "titleEn": "",
        "youtubeId": "2oFx3DPf_Uo",
        "colabUrl": "https://colab.research.google.com/drive/19Xq9UE_qi8jx8xLeHuW2GohPpdQmKvai?usp=sharing"
      },
      {
        "id": "10",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "L4p-ju44spQ",
        "colabUrl": "https://colab.research.google.com/drive/1Xkb-zjhofm4Rnv1ZsQoCi4w5bn7iY4f5?usp=sharing"
      },
      {
        "id": "11",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "QejZQ0Dh5x8",
        "colabUrl": "https://colab.research.google.com/drive/1bkzZh_JbjatK-47sF8LUaEhzWwLT-RSD?usp=sharing"
      },
      {
        "id": "12",
        "titleKo": "Sentiment Analysis",
        "titleEn": "",
        "youtubeId": "7GUoDHxN5NM",
        "colabUrl": "https://colab.research.google.com/drive/1CFBtnM5W7bGOp0SVhZeHymb78dMgeSBu?usp=sharing"
      },
      {
        "id": "13",
        "titleKo": "Named Entity Recognition",
        "titleEn": "",
        "youtubeId": "XETjf2CX4xU",
        "colabUrl": "https://colab.research.google.com/drive/1L3FXPOL6aBPyTOgHmJUUkYHX-ldm2E5l?usp=sharing"
      },
      {
        "id": "14",
        "titleKo": "",
        "titleEn": "",
        "youtubeId": "aUsGQaqYYBk",
        "colabUrl": "https://colab.research.google.com/drive/1MGAdSfhrtT15MhLWzly87fZ_k0q2g_Zd?usp=sharing"
      },
      {
        "id": "15",
        "titleKo": "Transformer",
        "titleEn": "",
        "youtubeId": "Izi9trF3nKY",
        "colabUrl": "https://colab.research.google.com/drive/1qBC_BPdmQgTWSd6n1B4d-pNAaATy_xVC?usp=sharing"
      },
      {
        "id": "16",
        "titleKo": "(Bidirectional Encoder Representations from Transformers)",
        "titleEn": "",
        "youtubeId": "LEtLfx1dS7Q",
        "colabUrl": "https://colab.research.google.com/drive/1-chrVi5GtTwxd6KNYthnX8hvDVFwHnhG?usp=sharing"
      },
      {
        "id": "17",
        "titleKo": "(Generative Pre-trained Transformer 2)",
        "titleEn": "",
        "youtubeId": "t43qcsVydnY",
        "colabUrl": "https://colab.research.google.com/drive/1cyRyTv3BZHahrjmbvnNW0tZkFDT0lM-G?usp=sharing"
      }
    ]
  },
  {
    "slug": "asp",
    "titleKo": "오디오 음성 처리",
    "titleEn": "Audio Speech Processing",
    "icon": "fa fa-tasks",
    "videoCount": 5,
    "playlistId": "PL7ZVZgsnLwEGskuPmm2-pYsNKY8Ihs5AP",
    "videos": [
      {
        "id": "01",
        "titleKo": "Audio Processing",
        "titleEn": "",
        "youtubeId": "oltGIc4uo5c",
        "colabUrl": "https://colab.research.google.com/drive/1N4TV3hCobwGLTMngp1vkOgKK3r3l0ZMJ?usp=sharing"
      },
      {
        "id": "02",
        "titleKo": "Audio Classification",
        "titleEn": "",
        "youtubeId": "Cf6QFjdU_KY",
        "colabUrl": "https://colab.research.google.com/drive/1roAAOvw-d_B4rn49JEoyMv2mPUNrEQCU"
      },
      {
        "id": "03",
        "titleKo": "Speech Recognition",
        "titleEn": "",
        "youtubeId": "WZt2_-S261g",
        "colabUrl": "https://colab.research.google.com/drive/19ralgDPXHHcJtoB-7Y7MPw4FcInLE2h0?usp=sharing"
      },
      {
        "id": "04",
        "titleKo": "Speaker Diarization",
        "titleEn": "",
        "youtubeId": "DLTij46bFsA",
        "colabUrl": "https://colab.research.google.com/drive/1XZVBq6GlheRooFo2jNzsIHY3WbFJVDzY?usp=sharing"
      },
      {
        "id": "05",
        "titleKo": "Speech Synthesis",
        "titleEn": "",
        "youtubeId": "3rpdqw_0dyU",
        "colabUrl": "https://colab.research.google.com/drive/1o0g9IsP6mmUVyTRRm60oMugg2JjJjE-a?usp=sharing"
      }
    ]
  }
];

export const getPlaylistBySlug = (slug: string): YouTubePlaylist | undefined => {
  return playlists.find((p) => p.slug === slug);
};
