import { getQTByBibleBook, getQTStats, getAllQTEntries } from '@/lib/qt';
import QTClientPage from './QTClientPage';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Quiet Time | SuanLab',
  description: '성경과 함께하는 매일 묵상 일지 - 이수안의 QT 기록',
  openGraph: {
    title: 'Quiet Time | SuanLab',
    description: '성경과 함께하는 매일 묵상 일지',
    type: 'website',
  },
};

export default function QTPage() {
  const byBook = getQTByBibleBook();
  const stats = getQTStats();
  const allEntries = getAllQTEntries();
  const recentEntries = allEntries.slice(0, 30);

  // Serialize for client component
  const serializedByBook = byBook.map(b => ({
    book: b.book,
    bookInfo: b.bookInfo,
    entries: b.entries.map(e => ({
      slug: e.slug,
      date: e.date,
      title: e.title,
      bibleBook: e.bibleBook,
      chapter: e.chapter,
      bibleReference: e.bibleReference,
    })),
  }));

  const serializedRecent = recentEntries.map(e => ({
    slug: e.slug,
    date: e.date,
    title: e.title,
    bibleBook: e.bibleBook,
    chapter: e.chapter,
    bibleReference: e.bibleReference,
  }));

  return (
    <QTClientPage
      byBook={serializedByBook}
      stats={stats}
      recentEntries={serializedRecent}
    />
  );
}
