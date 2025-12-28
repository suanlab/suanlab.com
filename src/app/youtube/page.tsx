'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Play, Map, TrendingUp, Clock, Youtube, ChevronRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { playlists } from '@/data/youtube';

const tabs = [
  { id: 'roadmap', label: 'Roadmap', icon: Map },
  { id: 'popular', label: 'Popular', icon: TrendingUp },
  { id: 'recent', label: 'Recent', icon: Clock },
];

const popularVideos = [
  { id: 'vgIc4ctNFbc', title: 'Popular Video 1' },
  { id: 'rtwtOcfYKqc', title: 'Popular Video 2' },
  { id: '37a7cBmCvB8', title: 'Popular Video 3' },
];

const recentVideos = [
  { id: 'kZBousGA0xg', title: 'Recent Video 1' },
  { id: 'pYN1tCmn4V0', title: 'Recent Video 2' },
  { id: 'l_jk13ChzP4', title: 'Recent Video 3' },
];

export default function YouTubePage() {
  const [activeTab, setActiveTab] = useState('roadmap');

  return (
    <>
      <PageHeader
        title="YouTube"
        subtitle="Learn data science and AI through video tutorials"
        breadcrumbs={[{ label: 'YouTube' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            {/* Sidebar */}
            <aside className="lg:col-span-1">
              <div className="sticky top-24">
                <div className="mb-4 flex items-center gap-2">
                  <Youtube className="h-5 w-5 text-red-500" />
                  <h2 className="font-semibold">Playlists</h2>
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

            {/* Main Content */}
            <div className="lg:col-span-3">
              {/* Tabs */}
              <div className="mb-8 flex gap-2 border-b">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : 'border-transparent text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <tab.icon className="h-4 w-4" />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'roadmap' && (
                <div className="overflow-hidden rounded-xl border bg-white shadow-lg">
                  <Image
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
                  <h3 className="text-lg font-semibold">Most Popular Videos</h3>
                  <div className="grid gap-6 md:grid-cols-2">
                    {popularVideos.map((video, index) => (
                      <Card key={video.id} className="overflow-hidden">
                        <div className="aspect-video">
                          <iframe
                            className="h-full w-full"
                            src={`https://www.youtube.com/embed/${video.id}`}
                            title={video.title}
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                          />
                        </div>
                        <CardContent className="p-4">
                          <Badge variant="outline" className="mb-2">
                            <TrendingUp className="mr-1 h-3 w-3" />
                            #{index + 1} Popular
                          </Badge>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'recent' && (
                <div className="space-y-6">
                  <h3 className="text-lg font-semibold">Recently Uploaded</h3>
                  <div className="grid gap-6 md:grid-cols-2">
                    {recentVideos.map((video) => (
                      <Card key={video.id} className="overflow-hidden">
                        <div className="aspect-video">
                          <iframe
                            className="h-full w-full"
                            src={`https://www.youtube.com/embed/${video.id}`}
                            title={video.title}
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                          />
                        </div>
                        <CardContent className="p-4">
                          <Badge variant="outline" className="mb-2">
                            <Clock className="mr-1 h-3 w-3" />
                            Recent
                          </Badge>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {/* All Playlists Grid (for mobile and desktop) */}
              <div className="mt-12">
                <h3 className="text-lg font-semibold mb-6">All Playlists</h3>
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
                                {playlist.videoCount} videos
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
