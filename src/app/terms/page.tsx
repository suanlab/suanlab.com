import { Metadata } from 'next';
import Link from 'next/link';
import PageHeader from '@/components/layout/PageHeader';

export const metadata: Metadata = {
  title: '이용약관 | SuanLab',
  description: 'SuanLab 이용약관',
};

export default function TermsPage() {
  return (
    <>
      <PageHeader
        title="이용약관"
        subtitle="SuanLab 서비스 이용약관"
        breadcrumbs={[{ label: '이용약관' }]}
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
                  이수안컴퓨터연구소(이하 &quot;SuanLab&quot;)가 제공하는 웹사이트 및 서비스(이하 &quot;서비스&quot;)를 이용하실 기 전에 다음의 이용약관을 읽어주실 기 바랍니다. 본 약관에 동의함으로써 서비스 이용이 가능합니다.
                </p>
              </div>

              {/* Section 1 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">1. 서비스의 성격</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 데이터 과학, 인공지능, 머신러닝, 딥러닝 등의 연구 및 교육 콘텐츠를 제공하는 학술 연구 웹사이트입니다. 본 서비스는 교육 및 정보 제공을 목적으로 하며, 상업적 목적으로 운영되지 않습니다.
                  </p>
                </div>
              </div>

              {/* Section 2 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">2. 서비스 이용</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    이용자는 본 약관에 동의한 경우에만 서비스를 이용할 수 있습니다. 서비스 이용 시 다음을 준수해야 합니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>법령 및 본 약관을 준수할 것</li>
                    <li>타인의 권리를 침해하지 않을 것</li>
                    <li>서비스의 정상적인 운영을 방해하지 않을 것</li>
                    <li>불법적인 목적으로 서비스를 이용하지 않을 것</li>
                  </ul>
                </div>
              </div>

              {/* Section 3 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">3. 지적재산권</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab이 제공하는 모든 콘텐츠(텍스트, 이미지, 비디오, 코드 등)의 저작권은 이수안컴퓨터연구소 또는 해당 저작권자에게 있습니다.
                  </p>
                  <p>
                    이용자는 개인적인 학습 목적으로만 콘텐츠를 이용할 수 있으며, 저작권자의 명시적 동의 없이 다음을 금지합니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>콘텐츠의 복제, 배포, 전송</li>
                    <li>콘텐츠의 수정 또는 변경</li>
                    <li>콘텐츠의 상업적 이용</li>
                    <li>콘텐츠의 무단 전재</li>
                  </ul>
                  <p className="mt-4">
                    단, 교육 및 학술 목적의 인용은 출처를 명시하는 경우 허용됩니다.
                  </p>
                </div>
              </div>

              {/* Section 4 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">4. 면책조항</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 다음에 대해 책임을 지지 않습니다:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-2">
                    <li>서비스 이용으로 인한 직접적, 간접적 손해</li>
                    <li>콘텐츠의 정확성, 완전성, 유용성</li>
                    <li>외부 링크 또는 제3자 서비스의 내용</li>
                    <li>서비스의 중단 또는 오류</li>
                  </ul>
                  <p className="mt-4">
                    본 서비스는 &quot;있는 그대로&quot; 제공되며, 명시적 또는 묵시적 보증이 없습니다.
                  </p>
                </div>
              </div>

              {/* Section 5 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">5. 서비스 변경 및 중단</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 사전 공지 없이 서비스의 전부 또는 일부를 변경하거나 중단할 수 있습니다. 이로 인한 손해에 대해 책임을 지지 않습니다.
                  </p>
                </div>
              </div>

              {/* Section 6 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">6. 이용자 콘텐츠</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    이용자가 서비스에 제공하는 모든 콘텐츠(댓글, 문의 등)에 대해 이용자는 모든 책임을 집니다. SuanLab은 이용자 콘텐츠를 자유롭게 사용, 수정, 배포할 수 있습니다.
                  </p>
                </div>
              </div>

              {/* Section 7 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">7. 약관의 변경</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    SuanLab은 법령의 변경이나 서비스 정책의 변경에 따라 본 약관을 수정할 수 있습니다. 변경된 약관은 웹사이트에 공지되며, 공지 후 계속 서비스를 이용하는 경우 변경된 약관에 동의한 것으로 간주됩니다.
                  </p>
                </div>
              </div>

              {/* Section 8 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">8. 준거법 및 관할</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    본 약관은 대한민국 법령에 따라 해석되며, 본 약관과 관련된 분쟁은 대한민국 법원의 관할에 속합니다.
                  </p>
                </div>
              </div>

              {/* Section 9 */}
              <div>
                <h2 className="text-2xl font-bold mb-4">9. 기타</h2>
                <div className="space-y-4 text-muted-foreground">
                  <p>
                    본 약관의 일부가 무효하거나 집행 불가능한 경우, 나머지 약관은 계속 유효합니다. SuanLab이 본 약관의 어떤 조항을 행사하지 않는 것은 그 조항의 포기를 의미하지 않습니다.
                  </p>
                </div>
              </div>

              {/* Contact Section */}
              <div className="border-t border-muted pt-8">
                <h2 className="text-2xl font-bold mb-4">문의</h2>
                <div className="space-y-3 text-muted-foreground">
                  <p>
                    본 약관에 대한 문의사항이 있으시면 아래로 연락주시기 바랍니다:
                  </p>
                  <div className="bg-muted/30 p-4 rounded-lg space-y-2">
                    <p><strong>기관명:</strong> 이수안컴퓨터연구소</p>
                    <p><strong>이메일:</strong> <a href="mailto:suanlab@gmail.com" className="text-primary hover:underline">suanlab@gmail.com</a></p>
                  </div>
                </div>
              </div>

              {/* Footer Links */}
              <div className="border-t border-muted pt-8 flex gap-4 text-sm">
                <Link href="/privacy" className="text-primary hover:underline">
                  개인정보처리방침
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
