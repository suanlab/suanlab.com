'use client';

import { useLanguage } from '@/components/language-provider';

export default function AboutMe() {
  const { language } = useLanguage();

  if (language === 'en') {
    return (
      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">
          About <span className="text-primary">Me</span>
        </h2>
        <p className="text-muted-foreground leading-relaxed">
          Suan Lee received the B.S., M.S. and Ph.D. in Computer Science from Kangwon National University, Korea, in 2008, 2010
          and 2017. Since 2021, he has been an assistant professor with the School of Computer Science, Semyung University,
          Korea, where he has been serving as the Head of the School since 2024. He has developed an in-memory database
          engine and real-time stream data processing engine as a researcher at Altibase company for over three years since 2012.
          He was a senior researcher at Kangwon National University&apos;s Information and Communication Research Center in 2018
          and a Research Professor/Visiting Professor at the National Program of Excellence in Software, at Kangwon National
          University in 2019. He was a Principal Researcher at Inha University VOICE AI Institute in 2020. He has received over 20
          paper awards from various conferences and academic societies, including BigComp, BigData, the Korean Institute of
          Information Scientists and Engineers (KIISE), and the Institute of Electronics and Information Engineers (IEIE).
          Furthermore, he has served as a Chair on the Organizing Committees for several prominent academic conferences,
          including ACM KDD, IEEE BigComp, DASFAA, KDBC, and KJDB. His current research interests include: trustworthy and
          causal machine intelligence (Interpretability, Explainability, and Causal Reasoning); geometric and relational representation
          learning (tensors, hypergraphs, knowledge graphs, and networks); high-dimensional structured data analytics (tabular
          data, time series, and spatio-temporal modeling); multimodal foundation models; and autonomous agency and collective
          intelligence (AI agents, multi-agent systems, and meta-policy optimization).
        </p>
      </div>
    );
  }

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-bold mb-6">
        About <span className="text-primary">Me</span>
      </h2>
      <p className="text-lg text-muted-foreground leading-relaxed mb-4">
        &quot;데이터와 AI를 이용해 세상을 이롭게하자!&quot;라는 생각을 가진 데이터 과학자이자 AI 연구자입니다.
      </p>
      <p className="text-muted-foreground leading-relaxed">
        새로운 연구와 기술에 흥미를 가지며, 인공지능, 머신러닝, 딥러닝, 자연어처리, 컴퓨터비전,
        오디오음성처리, 빅데이터에 관심이 많습니다. 머신러닝, 딥러닝, 데이터마이닝, 데이터웨어하우스,
        데이터베이스 분야에서 19년간 연구하였고, 인메모리 데이터베이스와 실시간 스트림 데이터 처리 엔진,
        빅데이터 플랫폼과 관련해 3년 이상의 개발 경력을 쌓았습니다.
      </p>
    </div>
  );
}
