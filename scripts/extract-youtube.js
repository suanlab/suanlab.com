const fs = require('fs');
const path = require('path');

const youtubeDir = path.join(__dirname, '../../WWW/youtube');
const files = fs.readdirSync(youtubeDir).filter(f => f.endsWith('.html') && f !== 'index.html');

const playlistInfo = {
  pp: { titleKo: '파이썬 프로그래밍', titleEn: 'Python Programming', icon: 'fa fa-code' },
  pg: { titleKo: '파이썬 게임', titleEn: 'Python Game', icon: 'fa fa-tasks' },
  web: { titleKo: '웹 프로그래밍', titleEn: 'Web Programming', icon: 'fa fa-html5' },
  ds: { titleKo: '데이터 과학', titleEn: 'Data Science', icon: 'fa fa-flask' },
  dc: { titleKo: '데이터 수집', titleEn: 'Data Collection', icon: 'fa fa-gears' },
  da: { titleKo: '데이터 분석', titleEn: 'Data Analysis', icon: 'fa fa-wrench' },
  dv: { titleKo: '데이터 시각화', titleEn: 'Data Visualization', icon: 'fa fa-bar-chart' },
  db: { titleKo: '데이터베이스', titleEn: 'Database', icon: 'fa fa-database' },
  bd: { titleKo: '빅데이터', titleEn: 'Big Data', icon: 'fa fa-server' },
  ml: { titleKo: '머신러닝', titleEn: 'Machine Learning', icon: 'fa fa-gears' },
  dl: { titleKo: '딥러닝', titleEn: 'Deep Learning', icon: 'fa fa-sliders' },
  dlf: { titleKo: '딥러닝 프레임워크', titleEn: 'Deep Learning Framework', icon: 'fa fa-magic' },
  cv: { titleKo: '컴퓨터 비전', titleEn: 'Computer Vision', icon: 'fa fa-desktop' },
  nlp: { titleKo: '자연어 처리', titleEn: 'Natural Language Processing', icon: 'fa fa-search' },
  asp: { titleKo: '오디오 음성 처리', titleEn: 'Audio Speech Processing', icon: 'fa fa-tasks' },
};

const playlists = [];

for (const file of files) {
  const slug = file.replace('.html', '');
  const html = fs.readFileSync(path.join(youtubeDir, file), 'utf-8');

  const info = playlistInfo[slug] || { titleKo: slug, titleEn: slug, icon: 'fa fa-play' };

  // Extract videos
  const videos = [];

  // Find all POST ITEM sections
  const postItemRegex = /<!-- POST ITEM -->([\s\S]*?)<!-- \/POST ITEM -->/g;
  let match;
  let videoId = 1;

  while ((match = postItemRegex.exec(html)) !== null) {
    const section = match[1];

    // Extract YouTube ID from iframe
    const ytMatch = section.match(/youtube\.com\/embed\/([a-zA-Z0-9_-]+)/);
    if (!ytMatch) continue;
    const youtubeId = ytMatch[1];

    // Extract title from h2
    const titleMatch = section.match(/<h2 class="size-20">([^<]*)<br ?\/?><span>([^<]*)<\/span>/);
    let titleKo = '';
    if (titleMatch) {
      titleKo = titleMatch[2].trim();
    }

    // Extract Colab URL
    const colabMatch = section.match(/<a href="(https:\/\/colab\.research\.google\.com[^"]+)"/);
    const colabUrl = colabMatch ? colabMatch[1] : undefined;

    // Extract PDF URL
    const pdfMatch = section.match(/<a href="(\.\.\/assets\/youtubes\/[^"]+\.pdf)"/);
    let pdfUrl = pdfMatch ? pdfMatch[1].replace('../', '/') : undefined;

    const video = {
      id: String(videoId).padStart(2, '0'),
      titleKo: titleKo,
      titleEn: '',
      youtubeId: youtubeId,
    };

    if (colabUrl) video.colabUrl = colabUrl;
    if (pdfUrl) video.pdfUrl = pdfUrl;

    videos.push(video);
    videoId++;
  }

  // Get playlist ID from first video iframe
  const playlistIdMatch = html.match(/list=([A-Za-z0-9_-]+)/);
  const playlistId = playlistIdMatch ? playlistIdMatch[1] : undefined;

  const playlist = {
    slug,
    titleKo: info.titleKo,
    titleEn: info.titleEn,
    icon: info.icon,
    videoCount: videos.length,
  };

  if (playlistId) playlist.playlistId = playlistId;
  playlist.videos = videos;

  playlists.push(playlist);
  console.log(slug + ': ' + videos.length + ' videos');
}

// Sort playlists by the order in playlistInfo
const order = Object.keys(playlistInfo);
playlists.sort((a, b) => order.indexOf(a.slug) - order.indexOf(b.slug));

// Generate TypeScript
const output = `export interface YouTubeVideo {
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

export const playlists: YouTubePlaylist[] = ${JSON.stringify(playlists, null, 2)};

export const getPlaylistBySlug = (slug: string): YouTubePlaylist | undefined => {
  return playlists.find((p) => p.slug === slug);
};
`;

fs.writeFileSync(path.join(__dirname, '../src/data/youtube/index.ts'), output);
console.log('Total playlists: ' + playlists.length);
console.log('Total videos: ' + playlists.reduce((sum, p) => sum + p.videoCount, 0));
