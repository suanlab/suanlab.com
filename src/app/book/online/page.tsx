import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { BookOpen } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { getAllBookPosts } from '@/lib/books';
import '@/styles/bookshelf.css';

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  title: { absolute: 'Online Book | SuanLab' },
  description: '이수안 교수의 온라인 도서 - 데이터 사이언스, 머신러닝, 딥러닝 학습 자료',
  openGraph: {
    title: 'Online Book | SuanLab',
    description: '이수안 교수의 온라인 도서 - 데이터 사이언스, 머신러닝, 딥러닝 학습 자료',
    url: `${BASE_URL}/book/online`,
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Online Book | SuanLab',
    description: '이수안 교수의 온라인 도서 - 데이터 사이언스, 머신러닝, 딥러닝 학습 자료',
  },
  alternates: {
    canonical: `${BASE_URL}/book/online/`,
  },
};

export default async function OnlineBookPage() {
  const books = await getAllBookPosts();

  return (
    <>
      <PageHeader
        title="Online Book"
        subtitle="책을 클릭하여 내용을 확인하세요"
        breadcrumbs={[
          { label: 'Book', href: '/book' },
          { label: 'Online Book' },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          {books.length > 0 ? (
            <div className="bookshelf">
              {/* Bookshelf with books */}
              <div className="bookshelf-row">
                {books.map((book, index) => (
                  <Link
                    key={book.slug}
                    href={`/book/online/${book.slug}`}
                    className="book-item"
                    style={{ '--book-index': index } as React.CSSProperties}
                  >
                    <div className="book-cover">
                      {book.image ? (
                        <Image
                          src={book.image}
                          alt={book.title}
                          fill
                          className="book-image"
                        />
                      ) : (
                        <div className="book-placeholder">
                          <BookOpen className="w-8 h-8" />
                        </div>
                      )}
                      <div className="book-spine"></div>
                      <div className="book-overlay">
                        <span className="book-read-text">읽기</span>
                      </div>
                    </div>
                    <div className="book-info">
                      <h3 className="book-title">{book.title}</h3>
                      <p className="book-author">{book.author}</p>
                    </div>
                  </Link>
                ))}
              </div>
              {/* Shelf */}
              <div className="shelf"></div>
              <div className="shelf-shadow"></div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                <BookOpen className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold">No online books available</h3>
              <p className="mt-2 text-muted-foreground">
                온라인 도서 콘텐츠가 준비 중입니다.
              </p>
            </div>
          )}
        </div>
      </section>
    </>
  );
}
