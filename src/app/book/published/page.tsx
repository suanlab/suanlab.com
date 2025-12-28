import Image from 'next/image';
import { BookOpen, User, Building, Calendar, ShoppingCart, Download } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export const metadata = {
  title: 'Published Book | SuanLab',
  description: 'Published books by Professor Suan Lee',
};

interface BookChapter {
  title: string;
  items: string[];
  downloadUrl?: string;
}

interface Book {
  id: string;
  title: string;
  description: string;
  author: string;
  publisher: string;
  date: string;
  image: string;
  url: string;
  chapters: BookChapter[];
}

const books: Book[] = [
  {
    id: 'python_game',
    title: '파이썬으로 만드는 나만의 게임',
    description: 'Pygame을 통하여 기초 예제부터 심화 예제까지 실습이 아닌 놀이처럼 배우는 파이썬. 파이썬의 기초 지식과 응용 지식을 함께 함양하며, 기본 문법과 연산자, 제어문, 함수를 익히며 프로젝트를 실습합니다.',
    author: '이수안',
    publisher: '비제이퍼블릭(BJ퍼블릭)',
    date: '2022년 01월 28일',
    image: '/assets/images/book/python_game.jpg',
    url: 'http://www.yes24.com/Product/Goods/106349915',
    chapters: [
      {
        title: 'Chapter 1. 파이썬과 IDE 소개 및 설치',
        items: ['파이썬 소개 및 특징', '파이썬 설치 및 환경설정', 'IDLE', 'Visual Studio Code 설치', '파이참(PyCharm) 설치'],
      },
      {
        title: 'Chapter 2. 파이썬 배우기',
        items: ['변수와 자료형', '연산자(Operators)', '제어문', '함수(Function)', '객체(Object)와 클래스(Class)', '모듈(Module)'],
        downloadUrl: '/assets/books/python_game/2_Python_Basic.zip',
      },
      {
        title: 'Chapter 3. pygame 배우기',
        items: ['pygame 소개 및 특징', 'pygame 라이브러리 설치', 'pygame 모듈', 'pygame 기본'],
        downloadUrl: '/assets/books/python_game/3_Pygame.zip',
      },
      {
        title: 'Chapter 4. 스네이크 게임(Snake Game)',
        items: ['스네이크 게임 규칙', '스네이크 게임 만들기', '스네이크 게임 실행'],
        downloadUrl: '/assets/books/python_game/4_Snake_Game.zip',
      },
      {
        title: 'Chapter 5. 핑퐁 게임(Ping Pong Game)',
        items: ['핑퐁 게임 규칙', '핑퐁 게임 리소스', '핑퐁 게임 만들기'],
        downloadUrl: '/assets/books/python_game/5_Ping_Pong_Game.zip',
      },
      {
        title: 'Chapter 6. 물고기 게임(Fish Game)',
        items: ['물고기 게임 규칙', '물고기 게임 리소스', '물고기 게임 만들기', '물고기 게임 실행', '물고기 게임 실행 파일 만들기'],
        downloadUrl: '/assets/books/python_game/6_Fish_Game.zip',
      },
      {
        title: 'Chapter 7. 자동차 게임(Racing Car Game)',
        items: ['자동차 게임 규칙', '자동차 게임 리소스', '자동차 게임 만들기', '자동차 게임 실행', '자동차 게임 실행 파일 만들기'],
        downloadUrl: '/assets/books/python_game/7_Racing_Car_Game.zip',
      },
      {
        title: 'Chapter 8. 우주선 게임(Spaceship Game)',
        items: ['우주선 게임 규칙', '우주선 게임 리소스', '우주선 게임 만들기', '우주선 게임 실행', '우주선 게임 실행 파일 만들기'],
        downloadUrl: '/assets/books/python_game/8_Spaceship_Game.zip',
      },
      {
        title: 'Chapter 9. 슈팅 게임(Shooting Game)',
        items: ['슈팅 게임 규칙', '슈팅 게임 리소스', '슈팅 게임 만들기', '슈팅 게임 실행', '슈팅 게임 실행 파일 만들기'],
        downloadUrl: '/assets/books/python_game/9_Shooting_Game.zip',
      },
    ],
  },
  {
    id: 'python_text',
    title: '파이썬으로 텍스트 분석하기: 전략커뮤니케이션을 위한 텍스트 마이닝',
    description: '『파이썬으로 텍스트 분석하기』는 파이썬 텍스트 분석에 앞서 선행되어야 할 부분인 파이썬 설치 및 코딩하는 방법부터 시작해서, 본격적인 텍스트 분석에 앞서 꼭 알아두어야 할 통계분석을 파이썬으로 어떻게 코딩하는가를 설명했고, 이어서 방대한 빅데이터를 어디에서 어떻게 수집하는가 하는 웹 크롤링, 그리고 수집된 텍스트 데이터를 정제하는 자연어처리를 설명하였다. 또한, 이러한 내용을 숙지한 후 핵심어 분석, 의미 연결망분석, 군집분석, 토픽 모델링, 단어임베딩, 감정분석 등 많이 활용되는 텍스트 분석기법을 파이썬으로 어떻게 코딩하는가를 기본 개념과 예제 중심으로 설명하였다.',
    author: '윤태일, 이수안',
    publisher: '늘봄',
    date: '2019년 08월 20일',
    image: '/assets/images/book/python_text.jpg',
    url: 'http://www.yes24.com/Product/Goods/77708059',
    chapters: [
      {
        title: '01. 왜 파이썬을 활용한 텍스트 마이닝인가?',
        items: ['왜 텍스트 마이닝인가?', '왜 파이썬인가?', '파이썬으로 하는 텍스트 마이닝의 절차 및 이 책의 구성'],
      },
      {
        title: '02. 파이썬 설치하고 시작하기',
        items: ['파이썬 설치하기', '파이썬 실행 방법 및 핵심 라이브러리'],
      },
      {
        title: '03. 기초 파이썬 코딩',
        items: ['파이썬의 입력과 출력', '데이터의 유형과 처리', '제어문'],
      },
      {
        title: '04. 파이썬으로 하는 통계기초',
        items: ['이 책에서 사용하는 데이터에 대한 설명', '데이터 불러오기 및 데이터 프레임 변환', '데이터 탐사와 통계분석 방법의 결정', '파이썬으로 하는 통계분석'],
      },
      {
        title: '05. 텍스트 수집하기',
        items: ['엑셀을 활용한 데이터 수집', '공개 API 활용', '공공데이터 포털을 활용하여 데이터 수집하기', '한국언론진흥재단의 빅 카인즈로 언론기사 수집하기'],
      },
      {
        title: '06. 텍스트 정제하기',
        items: ['자연어 처리의 기본개념과 절차', '영어 텍스트의 자연어 처리', '한국어 텍스트의 자연어 처리'],
      },
      {
        title: '07. 핵심어 빈도분석',
        items: ['단순 빈도분석', '단어 구름으로 시각화하기', '어휘 빈도-문서 역빈도(TF-IDF) 분석'],
      },
      {
        title: '08. 의미 연결망분석',
        items: ['사회(의미) 연결망분석의 기본개념', '의미 연결망의 속성'],
      },
      {
        title: '09. 군집분석',
        items: ['군집분석의 기본개념', '비계층적 군집분석', '계층적 군집분석'],
      },
      {
        title: '10. 토픽 모델링과 단어임베딩',
        items: ['토픽 모델링과 LDA의 이해', '단어임베딩과 Word2Vec의 이해'],
      },
      {
        title: '11. 감정분석',
        items: ['감정분석의 기본개념', '감정어휘 사전을 이용한 문서 감정분석', '공개 API를 활용한 이미지 감정분석'],
      },
      {
        title: '12. 마무리',
        items: ['참고문헌'],
        downloadUrl: '/assets/books/python_text_mining.zip',
      },
    ],
  },
];

export default function PublishedBookPage() {
  return (
    <>
      <PageHeader
        title="Published Book"
        subtitle="Published books by Professor Suan Lee"
        breadcrumbs={[
          { label: 'Book', href: '/book' },
          { label: 'Published Book' },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
              Published <span className="text-primary">Books</span>
            </h2>
            <p className="mt-4 text-muted-foreground">
              Learn programming and data science through comprehensive books
            </p>
          </div>

          <div className="space-y-12">
            {books.map((book) => (
              <Card key={book.id} className="overflow-hidden" id={book.id}>
                <div className="flex flex-col lg:flex-row lg:items-start gap-8">
                  {/* Book Cover */}
                  <div className="flex-shrink-0 flex justify-center lg:justify-start p-6 lg:p-8 lg:pb-0">
                    <Image
                      src={book.image}
                      alt={book.title}
                      width={280}
                      height={400}
                      className="rounded-lg shadow-lg object-contain"
                    />
                  </div>

                  {/* Book Details */}
                  <CardContent className="flex-1 p-6 lg:p-8 lg:pl-0">
                    <div className="mb-4">
                      <Badge variant="secondary" className="mb-3">
                        <BookOpen className="mr-1 h-3 w-3" />
                        Book
                      </Badge>
                      <h3 className="text-2xl font-bold">{book.title}</h3>
                    </div>

                    <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-6">
                      <div className="flex items-center gap-1">
                        <User className="h-4 w-4" />
                        <span>{book.author}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Building className="h-4 w-4" />
                        <span>{book.publisher}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="h-4 w-4" />
                        <span>{book.date}</span>
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="font-semibold mb-2">About this book</h4>
                      <p className="text-muted-foreground">{book.description}</p>
                    </div>

                    {book.chapters.length > 0 && (
                      <div className="mb-6">
                        <h4 className="font-semibold mb-4">Table of Contents</h4>
                        <div className="space-y-4">
                          {book.chapters.map((chapter, idx) => (
                            <div key={idx} className="border rounded-lg p-4">
                              <div className="flex items-start justify-between">
                                <div>
                                  <p className="font-medium">{chapter.title}</p>
                                  <div className="mt-2 flex flex-wrap gap-2">
                                    {chapter.items.slice(0, 3).map((item, itemIdx) => (
                                      <Badge key={itemIdx} variant="outline" className="text-xs">
                                        {item}
                                      </Badge>
                                    ))}
                                    {chapter.items.length > 3 && (
                                      <Badge variant="outline" className="text-xs">
                                        +{chapter.items.length - 3} more
                                      </Badge>
                                    )}
                                  </div>
                                </div>
                                {chapter.downloadUrl && (
                                  <Button variant="outline" size="sm" asChild>
                                    <a href={chapter.downloadUrl} target="_blank" rel="noopener noreferrer">
                                      <Download className="h-4 w-4" />
                                    </a>
                                  </Button>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <Button size="lg" asChild>
                      <a href={book.url} target="_blank" rel="noopener noreferrer">
                        <ShoppingCart className="mr-2 h-4 w-4" />
                        Purchase Book
                      </a>
                    </Button>
                  </CardContent>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
