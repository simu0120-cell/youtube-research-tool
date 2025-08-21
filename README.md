# YouTube 탐색 & 대량 비교 (MVP)

한 번에 키워드 기반으로 유튜브 영상을 탐색하고, 조회수/길이/날짜 필터, 커스텀 지표(CII), 채널 기여도(%)를 표시합니다. 선택한 영상의 공개 자막을 TXT로 일괄 다운로드할 수 있습니다.

## 빠른 시작

1) 파이썬 3.10+ 준비 후 패키지 설치
```bash
pip install -r requirements.txt
```

2) YouTube Data API v3 키 준비 후 환경변수 설정
```bash
# macOS/Linux
export YT_API_KEY="YOUR_API_KEY"

# Windows PowerShell
setx YT_API_KEY "YOUR_API_KEY"
```
터미널을 새로 열어 환경변수 반영 후 실행하세요.

3) 실행
```bash
streamlit run app.py
```

## 기능
- 키워드 검색 + 업로드 시기(이번달/최근7일/제한없음)
- 영상 길이 필터(4분 미만/4~20분/20분 이상)
- 정렬(조회수/날짜/관련성/평점) + 커스텀 지표 CII
- 채널 기여도(%) 계산
- 공개 자막 일괄 수집 및 TXT 다운로드
- 결과 CSV 다운로드

## 주의
- 공개 자막만 수집됩니다.
- API 쿼터 초과/제한에 대비해 검색/조회는 50개 배치로 요청합니다.
- 상업적 운용 전, 개인정보·저작권·플랫폼 약관을 준수하세요.
