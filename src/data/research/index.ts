export interface ResearchArea {
  slug: string;
  titleKo: string;
  titleEn: string;
  image: string;
  description: string;
  icon: string;
}

export const researchAreas: ResearchArea[] = [
  {
    slug: 'ds',
    titleKo: '데이터과학 & 빅데이터',
    titleEn: 'Data Science & Big Data',
    image: '/assets/images/research/ds.jpg',
    description: 'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.',
    icon: 'et-gears',
  },
  {
    slug: 'dl',
    titleKo: '딥러닝 & 머신러닝',
    titleEn: 'Deep Learning & Machine Learning',
    image: '/assets/images/research/dl.jpg',
    description: 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.',
    icon: 'et-layers',
  },
  {
    slug: 'nlp',
    titleKo: '자연어처리',
    titleEn: 'Natural Language Processing',
    image: '/assets/images/research/nlp.jpg',
    description: 'Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.',
    icon: 'et-document',
  },
  {
    slug: 'cv',
    titleKo: '컴퓨터 비전',
    titleEn: 'Computer Vision',
    image: '/assets/images/research/cv.jpg',
    description: 'Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.',
    icon: 'et-pictures',
  },
  {
    slug: 'graphs',
    titleKo: '그래프 & 텐서',
    titleEn: 'Graphs and Tensors',
    image: '/assets/images/research/graphs.jpg',
    description: 'A graph is an abstract data type that is meant to implement the undirected graph and directed graph concepts from the field of graph theory within mathematics.',
    icon: 'et-linegraph',
  },
  {
    slug: 'st',
    titleKo: '시공간 데이터',
    titleEn: 'Spatio-Temporal',
    image: '/assets/images/research/st.jpg',
    description: 'Spatiotemporal analysis deals with data that has both spatial and temporal components, enabling understanding of phenomena across space and time.',
    icon: 'et-map',
  },
  {
    slug: 'asp',
    titleKo: '오디오 음성 처리',
    titleEn: 'Audio & Speech Processing',
    image: '/assets/images/research/asp.jpg',
    description: 'Audio and speech processing involves the analysis, synthesis, and recognition of audio signals and human speech using deep learning and signal processing techniques.',
    icon: 'et-music',
  },
];

export const getResearchBySlug = (slug: string): ResearchArea | undefined => {
  return researchAreas.find((r) => r.slug === slug);
};
