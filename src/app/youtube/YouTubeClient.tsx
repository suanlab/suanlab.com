'use client';
import LocalizedContent, { OriginalLanguageNotice, useDisplayText } from '@/components/localized-content';
import SourceText from '@/components/source-text';
import { catalogEnglish } from '@/data/i18n/catalog';

import Image from 'next/image';
import { useId, useState } from 'react';
import { Play, Map, TrendingUp, Clock } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface VideoItem {
  youtubeId: string;
  title: string;
  playlistTitle: string;
}

interface YouTubeClientProps {
  popularVideos: VideoItem[];
  recentVideos: VideoItem[];
}

const tabs = [
  { id: 'roadmap', label: '로드맵', icon: Map },
  { id: 'popular', label: '인기 영상', icon: TrendingUp },
  { id: 'recent', label: '최근 영상', icon: Clock },
];

function LazyVideo({ video, badge }: { video: VideoItem; badge: React.ReactNode }) {
  const displayText = useDisplayText();
  const titleId = useId();
  const title = displayText(video.title);
  const [loaded, setLoaded] = useState(false);
  const thumbnailUrl = `https://img.youtube.com/vi/${video.youtubeId}/maxresdefault.jpg`;

  return <LocalizedContent>
    <Card className="overflow-hidden">
      <div className="relative aspect-video bg-black">
        {loaded ? (
          <iframe
            className="h-full w-full"
            src={`https://www.youtube.com/embed/${video.youtubeId}`}
            title={title}
            lang={/[가-힣]/.test(title) ? 'ko' : undefined}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        ) : (
          <button
            onClick={() => setLoaded(true)}
            className="relative h-full w-full cursor-pointer"
            aria-labelledby={`${titleId}-play ${titleId}`}
          >
            <span id={`${titleId}-play`} className="sr-only">영상 재생</span>
            <Image
              width={480}
              height={360}
              src={thumbnailUrl}
              alt=""
              className="h-full w-full object-cover"
              loading="lazy"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 transition-colors hover:bg-black/40">
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-red-600 shadow-lg transition-transform hover:scale-110">
                <Play className="h-6 w-6 text-white ml-1" fill="white" />
              </div>
            </div>
          </button>
        )}
      </div>
      <CardContent className="p-4">
        {badge}
        <p id={titleId} className="text-sm font-medium mt-1"><SourceText text={title} /></p>
        <p className="text-xs text-muted-foreground">{video.playlistTitle}</p>
      </CardContent>
    </Card>
  </LocalizedContent>;
}

export default function YouTubeClient({ popularVideos, recentVideos }: YouTubeClientProps) {
  const [activeTab, setActiveTab] = useState('roadmap');

  return <LocalizedContent translations={catalogEnglish}>
    <>
      <div className="mb-8 flex flex-wrap gap-2 border-b">
        {tabs.map((tab) => {
          const TabIcon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              <TabIcon className="h-4 w-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === 'roadmap' && (
        <div className="overflow-hidden rounded-xl border bg-white dark:bg-gray-900 shadow-lg">
          <Image
            src="/assets/youtubes/roadmap.png"
            alt="YouTube Roadmap"
            width={822}
            height={1100}
            className="w-full h-auto"
          />
        </div>
      )}

      {activeTab !== 'roadmap' && <OriginalLanguageNotice />}
      {activeTab === 'popular' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold">인기 영상</h3>
          <div className="grid gap-6 md:grid-cols-2">
            {popularVideos.map((video, index) => (
              <LazyVideo
                key={video.youtubeId}
                video={video}
                badge={
                  <Badge variant="outline" className="mb-2">
                    <TrendingUp className="mr-1 h-3 w-3" />
                    인기 #{index + 1}
                  </Badge>
                }
              />
            ))}
          </div>
        </div>
      )}

      {activeTab === 'recent' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold">최근 업로드</h3>
          <div className="grid gap-6 md:grid-cols-2">
            {recentVideos.map((video) => (
              <LazyVideo
                key={video.youtubeId}
                video={video}
                badge={
                  <Badge variant="outline" className="mb-2">
                    <Clock className="mr-1 h-3 w-3" />
                    최근
                  </Badge>
                }
              />
            ))}
          </div>
        </div>
      )}
    </>
  </LocalizedContent>;
}
