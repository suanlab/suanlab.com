import { Metadata } from 'next';
import Link from 'next/link';
import PageHeader from '@/components/layout/PageHeader';

export const metadata: Metadata = {
  title: '개인정보처리방침 | SuanLab',
  description: 'SuanLab 개인정보처리방침',
};

export default function PrivacyPage() {
  return (
    <>
      <PageHeader
        title="개인정보처리방침"
        subtitle="SuanLab의 개인정보 보호 정책"
        breadcrumbs={[{ label: '개인정보처리방침' }]}
      />

      <section className="py-16 md:py-20">
        <div className="container max-w-3xl">
          <article className="prose prose-invert max-w-none">
            <div className="space-y-8 text-foreground">
              {/* Last Updated */}
              <div className="text-sm text-muted-foreground border-l-4 border-primary pl-4">
                <p>최종 수정일: 2026년 3월 2일</p>
              </div>

              {/* Introduction */}
              <div>
                <p className="text-lg text-muted-foreground leading-relaxed">
                  이수안컴퓨터연구소(이하 &quot;SuanLab&quot;)는 개인정보보호법을 준수하며, 이용자의 개인정보를 보호하고 개인정보와 관련된 이용자의 고충을 신속하고 성실하게 처리하기 위하여 다음과 같이 개인정보처리방침을 수립·공개합니다.
                </p>
              </div>

              {/* Section 1 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">1. 개인정보의 수집 및 이용</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 다음과 같은 개인정보를 수집하고 있습니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>이메일 주소 (문의 및 연락용)</li>
                    <li>이름 (선택사항)</li>
                    <li>접속 로그, 쿠키, 접속 IP 주소</li>
                  </ul>
                  <p className="mt-4">
                    수집된 개인정보는 다음의 목적으로만 이용됩니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>이용자 문의에 대한 응답 및 처리</li>
                    <li>서비스 개선 및 통계 분석</li>
                    <li>법적 의무 이행</li>
                  </ul>
                </div>
              </div>

              {/* Section 2 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">2. 쿠키 및 추적 기술</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 Google Analytics를 통해 사용자의 방문 통계를 수집합니다. Google Analytics는 쿠키를 사용하여 사용자의 행동을 추적합니다.
                  </p>
                  <p>
                    <strong>Google Analytics 추적 ID:</strong> G-PYEC6PCW0P
                  </p>
                  <p>
                    사용자는 브라우저 설정을 통해 쿠키를 거부할 수 있으며, 이 경우 일부 서비스 이용에 제한이 있을 수 있습니다.
                  </p>
                </div>
              </div>

              {/* Section 3 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">3. 외부 서비스 연동</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 다음의 외부 서비스를 사용합니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li><strong>YouTube 임베드:</strong> 교육용 비디오 콘텐츠 제공. YouTube의 개인정보처리방침이 적용됩니다.</li>
                    <li><strong>Google Analytics:</strong> 웹사이트 사용 통계 분석</li>
                  </ul>
                  <p className="mt-4">
                    이러한 외부 서비스 제공자의 개인정보처리방침을 확인하시기 바랍니다.
                  </p>
                </div>
              </div>

              {/* Section 4 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">4. 개인정보의 보호</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 개인정보의 안전성 확보를 위해 다음과 같은 조치를 취하고 있습니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>개인정보에 대한 접근 제한</li>
                    <li>암호화를 통한 데이터 보호</li>
                    <li>정기적인 보안 점검</li>
                  </ul>
                </div>
              </div>

              {/* Section 5 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">5. 개인정보의 제3자 제공</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 이용자의 개인정보를 제3자에게 제공하지 않습니다. 단, 다음의 경우는 예외입니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>법령에 의한 요구가 있는 경우</li>
                    <li>이용자의 명시적 동의가 있는 경우</li>
                  </ul>
                </div>
              </div>

              {/* Section 6 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">6. 개인정보의 보유 및 이용 기간</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    개인정보는 수집 목적이 달성될 때까지 보유하며, 목적이 달성된 후에는 지체 없이 파기합니다. 단, 관련 법령에 의해 보존이 필요한 경우는 해당 기간 동안 보유합니다.
                  </p>
                </div>
              </div>

              {/* Section 7 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">7. 이용자의 권리</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    이용자는 다음의 권리를 가집니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>개인정보 열람 요청</li>
                    <li>개인정보 수정 요청</li>
                    <li>개인정보 삭제 요청</li>
                    <li>개인정보 처리 정지 요청</li>
                  </ul>
                  <p className="mt-4">
                    위의 권리 행사는 아래의 연락처로 요청할 수 있습니다.
                  </p>
                </div>
              </div>

              {/* Section 8 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">8. 개인정보 처리방침의 변경</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    이 개인정보처리방침은 법령의 변경이나 서비스 정책의 변경에 따라 수정될 수 있습니다. 변경 사항은 웹사이트에 공지됩니다.
                  </p>
                </div>
              </div>

              {/* Contact Section */}
              <div className="border-t border-muted pt-8">
                <h2 className="text-2xl font-bold mb-4">문의</h2>
                <div className="space-y-3 text-muted-foreground">
                  <p>
                    개인정보 처리와 관련하여 문의사항이 있으시면 아래로 연락주시기 바랍니다:
                  </p>
                  <div className="bg-muted/30 p-4 rounded-lg space-y-2">
                    <p><strong>기관명:</strong> 이수안컴퓨터연구소</p>
                    <p><strong>이메일:</strong> <a href="mailto:suanlab@gmail.com" className="text-primary hover:underline">suanlab@gmail.com</a></p>
                  </div>
                </div>
              </div>

              {/* Footer Links */}
              <div className="border-t border-muted pt-8 flex gap-4 text-sm">
                <Link href="/terms" className="text-primary hover:underline">
                  이용약관
                </Link>
                <span className="text-muted-foreground">|</span>
                <Link href="/" className="text-primary hover:underline">
                  홈으로
                </Link>
              </div>
            </div>
          </article>
        </div>
      </section>
    </>
  );
}
