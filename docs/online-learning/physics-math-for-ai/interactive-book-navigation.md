# Interactive Book 메뉴와 진입 경로

2026-11-06. 진행 중인 인터랙티브 교재는 **도서 → Interactive Book**에서 열도록 분리했다.

- 도서 서브메뉴와 `/book/` 소개에 Interactive Book을 추가했다.
- `/book/interactive/`에서 현재 교재를 연다. Online Book 페이지에는 해당 교재의 진입 카드를 넣지 않는다.
- 교재 목차·읽기 화면의 breadcrumb와 데스크톱/모바일 메뉴 활성 표시는 Interactive Book을 가리킨다.
- 기존 교재 URL과 북마크·검색 링크는 유지한다. 기존 Online Book 도서 목록과 출판 도서 분류도 유지한다.
- 카드의 중첩 버튼을 비대화형 요소로 바꿔 전체 카드 링크 하나로 동작하도록 했고, 한글 제목의 단어 단위 줄바꿈을 적용했다.

한영·명암·모바일320px/데스크톱1600px의8개 조합에서 메뉴의 키보드/클릭 진입, 활성 분류, 소개→목차→읽기 흐름, Online Book의 교재 링크 부재와 기존 도서 링크를 검증했다. 소개·목록의 접근성·영어 표시·가로 넘침도 확인했다. 최종 정적 출력과 영어571경로·629상태 검사를 통과했다.

로컬 미리보기: <http://localhost:4175/book/interactive/>. 공개 배포는 이번 작업에 포함하지 않았다. 근거는 `.runtime/learning/continuum-stress-strain-integrated/navigation.json`과 최종 검증 기록이다.
