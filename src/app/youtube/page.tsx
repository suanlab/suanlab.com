import { Metadata } from 'next';
import Link from 'next/link';
import { Play, Youtube, ChevronRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { VideoCollectionJsonLd } from '@/components/seo/JsonLd';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { playlists } from '@/data/youtube';
import YouTubeClient from './YouTubeClient';

export const metadata: Metadata = {
  title: 'YouTube | SuanLab',
  description:
    '이수안 교수의 데이터 사이언스, 머신러닝, 딥러닝 유튜브 강의 - 파이썬, PyTorch, TensorFlow 튜토리얼',
  keywords: [
    'YouTube',
    '유튜브',
    '강의',
    '튜토리얼',
    '파이썬',
    'Python',
    'PyTorch',
    'TensorFlow',
    'Deep Learning',
    'Machine Learning',
    '이수안',
  ],
  openGraph: {
    title: 'YouTube | SuanLab',
    description: '이수안 교수의 데이터 사이언스, 머신러닝, 딥러닝 유튜브 강의',
    url: 'https://suanlab.com/youtube',
    siteName: 'SuanLab',
    type: 'website',
    locale: 'ko_KR',
    images: [
      {
        url: '/assets/images/slider/blue.jpg',
        width: 1200,
        height: 630,
        alt: 'SuanLab YouTube 강의',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'YouTube | SuanLab',
    description: '이수안 교수의 데이터 사이언스, 머신러닝, 딥러닝 유튜브 강의',
    images: ['/assets/images/slider/blue.jpg'],
  },
  alternates: {
    canonical: 'https://suanlab.com/youtube',
  },
};

const popularVideos = playlists.slice(0, 6).map(p => ({
  youtubeId: p.videos[0]?.youtubeId || '',
  title: p.videos[0]?.titleKo || p.titleKo,
  playlistTitle: p.titleKo,
})).filter(v => v.youtubeId);

const recentVideos = playlists.slice(0, 6).map(p => ({
  youtubeId: p.videos[p.videos.length - 1]?.youtubeId || '',
  title: p.videos[p.videos.length - 1]?.titleKo || p.titleKo,
  playlistTitle: p.titleKo,
})).filter(v => v.youtubeId);

export default function YouTubePage() {
  const allVideos = playlists.flatMap((p) =>
    p.videos.filter((v) => v.youtubeId && (v.titleKo || v.titleEn)).map((v) => ({
      title: v.titleKo || v.titleEn || p.titleKo,
      videoId: v.youtubeId,
    }))
  );

  return (
    <>
      <VideoCollectionJsonLd videos={allVideos} />
      <PageHeader
        title="YouTube"
        subtitle="영상으로 데이터 사이언스와 AI를 배워보세요"
        breadcrumbs={[{ label: 'YouTube' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            <aside className="lg:col-span-1">
              <div className="sticky top-24">
                <div className="mb-4 flex items-center gap-2">
                  <Youtube className="h-5 w-5 text-red-500" />
                  <h2 className="font-semibold">재생목록</h2>
                </div>
                <nav className="space-y-1">
                  {playlists.map((playlist) => (
                    <Link
                      key={playlist.slug}
                      href={`/youtube/${playlist.slug}`}
                      className="group flex items-center justify-between rounded-lg px-3 py-2 text-sm hover:bg-muted transition-colors"
                    >
                      <div>
                        <p className="font-medium group-hover:text-primary transition-colors">
                          {playlist.titleKo}
                        </p>
                        <p className="text-xs text-muted-foreground">{playlist.titleEn}</p>
                      </div>
                      <Badge variant="secondary" className="text-xs">
                        {playlist.videoCount}
                      </Badge>
                    </Link>
                  ))}
                </nav>
              </div>
            </aside>

            <div className="lg:col-span-3">
              <YouTubeClient popularVideos={popularVideos} recentVideos={recentVideos} />

              <div className="mt-12">
                <h3 className="text-lg font-semibold mb-6">전체 재생목록</h3>
                <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3">
                  {playlists.map((playlist) => (
                    <Link key={playlist.slug} href={`/youtube/${playlist.slug}`}>
                      <Card className="group h-full transition-all hover:shadow-md hover:border-primary/50">
                        <CardContent className="flex items-center justify-between p-4">
                          <div className="flex items-center gap-3">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-500/10">
                              <Play className="h-5 w-5 text-red-500" />
                            </div>
                            <div>
                              <p className="font-medium group-hover:text-primary transition-colors">
                                {playlist.titleKo}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {playlist.videoCount}개 영상
                              </p>
                            </div>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                        </CardContent>
                      </Card>
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
