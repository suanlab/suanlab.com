'use client';

import { useState } from 'react';
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
  const [loaded, setLoaded] = useState(false);
  const thumbnailUrl = `https://img.youtube.com/vi/${video.youtubeId}/maxresdefault.jpg`;

  return (
    <Card className="overflow-hidden">
      <div className="relative aspect-video bg-black">
        {loaded ? (
          <iframe
            className="h-full w-full"
            src={`https://www.youtube.com/embed/${video.youtubeId}`}
            title={video.title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        ) : (
          <button
            onClick={() => setLoaded(true)}
            className="relative h-full w-full cursor-pointer"
            aria-label={`영상 재생: ${video.title}`}
          >
            <img
              src={thumbnailUrl}
              alt={video.title}
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
        <p className="text-sm font-medium mt-1">{video.title}</p>
        <p className="text-xs text-muted-foreground">{video.playlistTitle}</p>
      </CardContent>
    </Card>
  );
}

export default function YouTubeClient({ popularVideos, recentVideos }: YouTubeClientProps) {
  const [activeTab, setActiveTab] = useState('roadmap');

  return (
    <>
      <div className="mb-8 flex gap-2 border-b">
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
          <img
            src="/assets/youtubes/roadmap.png"
            alt="YouTube Roadmap"
            width={822}
            height={1100}
            className="w-full h-auto"
          />
        </div>
      )}

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
  );
}
