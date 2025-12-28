import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Play, Youtube, FileText, ExternalLink, Video, ChevronRight } from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { playlists, getPlaylistBySlug } from '@/data/youtube';

export async function generateStaticParams() {
  return playlists.map((playlist) => ({
    playlist: playlist.slug,
  }));
}

export async function generateMetadata({ params }: { params: { playlist: string } }) {
  const playlist = getPlaylistBySlug(params.playlist);
  if (!playlist) return { title: 'Not Found' };

  return {
    title: `${playlist.titleEn} | YouTube | SuanLab`,
    description: `${playlist.titleKo} - ${playlist.titleEn} YouTube 강좌`,
  };
}

export default function YouTubePlaylistPage({ params }: { params: { playlist: string } }) {
  const currentPlaylist = getPlaylistBySlug(params.playlist);

  if (!currentPlaylist) {
    notFound();
  }

  return (
    <>
      <PageHeader
        title={currentPlaylist.titleEn}
        subtitle={currentPlaylist.titleKo}
        breadcrumbs={[
          { label: 'YouTube', href: '/youtube' },
          { label: currentPlaylist.titleEn },
        ]}
      />

      <section className="py-16 md:py-20">
        <div className="container">
          <div className="grid gap-8 lg:grid-cols-4">
            {/* Sidebar */}
            <aside className="lg:col-span-1 order-2 lg:order-1">
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
                      className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${
                        playlist.slug === params.playlist
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-muted'
                      }`}
                    >
                      <div>
                        <p className={`font-medium ${playlist.slug !== params.playlist ? 'group-hover:text-primary' : ''} transition-colors`}>
                          {playlist.titleKo}
                        </p>
                        <p className={`text-xs ${playlist.slug === params.playlist ? 'text-primary-foreground/80' : 'text-muted-foreground'}`}>
                          {playlist.titleEn}
                        </p>
                      </div>
                      <Badge
                        variant={playlist.slug === params.playlist ? 'secondary' : 'outline'}
                        className="text-xs"
                      >
                        {playlist.videoCount}
                      </Badge>
                    </Link>
                  ))}
                </nav>
              </div>
            </aside>

            {/* Main Content */}
            <div className="lg:col-span-3 order-1 lg:order-2">
              <div className="mb-8 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold">{currentPlaylist.titleKo}</h2>
                  <p className="text-muted-foreground">{currentPlaylist.titleEn}</p>
                </div>
                <Badge variant="secondary" className="text-sm">
                  <Video className="mr-1 h-4 w-4" />
                  {currentPlaylist.videoCount} videos
                </Badge>
              </div>

              {currentPlaylist.videos.length > 0 ? (
                <div className="space-y-8">
                  {currentPlaylist.videos.map((video) => (
                    <Card key={video.id} className="overflow-hidden">
                      <div className="grid gap-6 md:grid-cols-3">
                        <div className="md:col-span-2">
                          <div className="aspect-video">
                            <iframe
                              className="h-full w-full"
                              src={`https://www.youtube.com/embed/${video.youtubeId}${currentPlaylist.playlistId ? `?list=${currentPlaylist.playlistId}` : ''}`}
                              title={video.titleKo}
                              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                              allowFullScreen
                            />
                          </div>
                        </div>
                        <CardContent className="flex flex-col justify-center p-6">
                          <Badge variant="outline" className="w-fit mb-3">
                            Episode {video.id}
                          </Badge>
                          <h3 className="text-lg font-semibold">{video.titleKo}</h3>
                          <p className="mt-1 text-sm text-muted-foreground">{video.titleEn}</p>
                          {video.description && (
                            <p className="mt-3 text-sm text-muted-foreground">{video.description}</p>
                          )}
                          <div className="mt-4 flex flex-wrap gap-2">
                            {video.colabUrl && (
                              <Button variant="outline" size="sm" asChild>
                                <a href={video.colabUrl} target="_blank" rel="noopener noreferrer">
                                  <ExternalLink className="mr-1 h-3 w-3" />
                                  Colab
                                </a>
                              </Button>
                            )}
                            {video.pdfUrl && (
                              <Button variant="outline" size="sm" asChild>
                                <a href={video.pdfUrl} target="_blank" rel="noopener noreferrer">
                                  <FileText className="mr-1 h-3 w-3" />
                                  PDF
                                </a>
                              </Button>
                            )}
                          </div>
                        </CardContent>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center">
                  {currentPlaylist.playlistId ? (
                    <Card className="overflow-hidden">
                      <div className="aspect-video">
                        <iframe
                          className="h-full w-full"
                          src={`https://www.youtube.com/embed/videoseries?list=${currentPlaylist.playlistId}`}
                          title={currentPlaylist.titleEn}
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                          allowFullScreen
                        />
                      </div>
                    </Card>
                  ) : (
                    <Card className="p-12">
                      <div className="flex flex-col items-center text-center">
                        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                          <Play className="h-8 w-8 text-muted-foreground" />
                        </div>
                        <h3 className="text-lg font-semibold">
                          {currentPlaylist.videoCount}개의 영상이 있습니다
                        </h3>
                        <p className="mt-2 text-muted-foreground">
                          영상 데이터가 곧 추가될 예정입니다.
                        </p>
                      </div>
                    </Card>
                  )}
                </div>
              )}

              {/* Navigation */}
              <div className="mt-12 flex justify-between">
                <Button variant="outline" asChild>
                  <Link href="/youtube">
                    <ChevronRight className="mr-2 h-4 w-4 rotate-180" />
                    All Playlists
                  </Link>
                </Button>
                <Button asChild>
                  <a
                    href={`https://www.youtube.com/channel/UCFfALXX0DOx7zv6VeR5U_Bg`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Youtube className="mr-2 h-4 w-4" />
                    Visit Channel
                  </a>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
