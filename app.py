# app.py — YouTube 탐색 & 대량 비교 (1단계 업그레이드 반영본)
# - Streamlit Cloud: Settings > Secrets 에 YT_API_KEY="..." 저장
# - 로컬 실행: 환경변수 YT_API_KEY 설정 후 `streamlit run app.py`

import os
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

KST = timezone(timedelta(hours=9))

# -----------------------------
# 기본 유틸
# -----------------------------
def get_api_key() -> str:
    # Streamlit Cloud 권장: st.secrets
    key = st.secrets.get("YT_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("YT_API_KEY", "").strip()
    return key

def build_youtube():
    key = get_api_key()
    if not key:
        st.error("API 키가 없습니다. Streamlit Secrets에 `YT_API_KEY=\"...\"` 를 저장하거나, 환경변수로 설정하세요.")
        st.stop()
    return build('youtube', 'v3', developerKey=key)

def iso8601_to_seconds(iso: str) -> int:
    # 'PT1H2M10S' → 초
    h=m=s=0
    if not iso:
        return 0
    t = iso.split('T')[-1]
    num = ''
    for ch in t:
        if ch.isdigit():
            num += ch
        else:
            if ch == 'H' and num: h = int(num)
            if ch == 'M' and num: m = int(num)
            if ch == 'S' and num: s = int(num)
            num = ''
    return h*3600 + m*60 + s

def seconds_to_mmss(s: int) -> str:
    return f"{s//60}:{s%60:02d}"

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def calc_cii(row: pd.Series, now: datetime) -> float:
    # CII = 참여도(좋아요+댓글×3 / 조회수) × 채널규모(log10(구독자+1)) × 신선도(exp(-일수/30)) × 100
    views = max(1, safe_int(row.get('view_count', 0)))
    likes = safe_int(row.get('like_count', 0))
    comments = safe_int(row.get('comment_count', 0))
    e = (likes + 3*comments) / views

    subs = safe_int(row.get('channel_subscribers', 0))
    b = math.log10(subs + 1) if subs >= 0 else 0.0

    published_at = row.get('published_at_dt', now)
    days = max(0.0, (now - published_at).total_seconds() / 86400.0)
    f = math.exp(-days / 30.0)

    return 100.0 * e * b * f

def pick_duration(label: str) -> str:
    return {"전체":"any","4분 미만":"short","4~20분":"medium","20분 이상":"long"}.get(label, "any")

def pick_order(label: str) -> str:
    # 내부 API용(검색 단계) — UI 정렬에는 CII 옵션이 따로 있음
    return {"조회수(내림차순)":"viewCount","최근 업로드":"date","관련성":"relevance","평점":"rating"}.get(label, "viewCount")

def start_of_this_month_kst() -> datetime:
    now = datetime.now(KST)
    return datetime(now.year, now.month, 1, tzinfo=KST)

def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')

# -----------------------------
# API 호출
# -----------------------------
def search_video_ids(client, q: str, max_results: int, order: str, published_after: str, video_duration: str) -> List[str]:
    ids = []
    page_token = None
    try:
        while len(ids) < max_results:
            req = client.search().list(
                q=q, part='id', type='video',
                maxResults=min(50, max_results - len(ids)),
                order=order,
                publishedAfter=published_after if published_after else None,
                videoDuration=video_duration if video_duration != 'any' else None,
                safeSearch='none'
            )
            if page_token:
                req.uri += f"&pageToken={page_token}"
            resp = req.execute()
            for item in resp.get('items', []):
                vid = item.get('id', {}).get('videoId')
                if vid:
                    ids.append(vid)
            page_token = resp.get('nextPageToken')
            if not page_token:
                break
    except HttpError as e:
        st.warning(f"Search API 오류: {e}")
    return ids

def fetch_videos_and_channels(client, video_ids: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    ch_ids = set()
    # videos
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = client.videos().list(
                part='snippet,contentDetails,statistics',
                id=",".join(batch)
            ).execute()
        except HttpError as e:
            st.warning(f"Videos API 오류: {e}")
            continue
        for item in resp.get('items', []):
            vid = item['id']
            sn = item.get('snippet', {})
            stc = item.get('statistics', {})
            cd = item.get('contentDetails', {})
            ch_id = sn.get('channelId', '')
            ch_ids.add(ch_id)
            rows.append({
                'video_id': vid,
                'channel_id': ch_id,
                'channel_title': sn.get('channelTitle',''),
                'title': sn.get('title',''),
                'published_at': sn.get('publishedAt',''),
                'thumbnail_url': sn.get('thumbnails',{}).get('default',{}).get('url',''),
                'duration_iso': cd.get('duration','PT0S'),
                'view_count': safe_int(stc.get('viewCount', 0)),
                'like_count': safe_int(stc.get('likeCount', 0)),
                'comment_count': safe_int(stc.get('commentCount', 0)),
            })
    df = pd.DataFrame(rows)

    # channels
    channels = {}
    if ch_ids:
        ids = list(ch_ids)
        for i in range(0, len(ids), 50):
            batch = ids[i:i+50]
            try:
                resp = client.channels().list(
                    part='statistics',
                    id=",".join(batch)
                ).execute()
            except HttpError as e:
                st.warning(f"Channels API 오류: {e}")
                continue
            for item in resp.get('items', []):
                cid = item['id']
                stc = item.get('statistics', {})
                channels[cid] = {'subscribers': safe_int(stc.get('subscriberCount', 0))}
    return df, channels

def enrich_dataframe(df: pd.DataFrame, channels: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now(KST)
    df = df.copy()
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['duration_sec'] = df['duration_iso'].apply(iso8601_to_seconds)
    df['duration_mmss'] = df['duration_sec'].apply(seconds_to_mmss)
    df['channel_subscribers'] = df['channel_id'].map(lambda x: channels.get(x, {}).get('subscribers', 0))
    df['CII'] = df.apply(lambda r: calc_cii(r, now), axis=1)

    # 채널 점유율(%) 계산
    counts = df['channel_title'].value_counts(dropna=False)
    share = {k: (counts[k] / len(df)) * 100.0 for k in counts.index}
    df['channel_share_pct'] = df['channel_title'].map(lambda x: share.get(x, 0.0))

    return df

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="YouTube 탐색 & 비교", layout="wide")
st.title("YouTube 탐색 & 대량 비교 (1단계 업그레이드)")
st.caption("키워드로 영상을 탐색하고, 한글 지표/정렬/다운로드/자막 일괄 수집까지 지원합니다.")

with st.sidebar:
    st.subheader("검색 조건")
    keyword = st.text_input("키워드", value="쿠팡꿀템")
    uploaded_when = st.selectbox("업로드 시기", ["이번달","최근 7일","제한 없음"], index=0)
    duration_label = st.selectbox("영상 길이", ["전체","4분 미만","4~20분","20분 이상"], index=1)
    # ⭐ 정렬 옵션에 CII(내림차순) 포함
    order = st.selectbox("정렬", ["조회수(내림차순)", "최근 업로드", "CII(내림차순)"], 0)
    max_results = st.slider("최대 결과 수", 10, 200, 50, 10)
    sample_take = st.number_input("한 번에 선택할 개수", 1, 50, 5, 1)
    run = st.button("검색 실행", type="primary")

client = build_youtube()

published_after = ""
if uploaded_when == "이번달":
    published_after = to_rfc3339(start_of_this_month_kst())
elif uploaded_when == "최근 7일":
    published_after = to_rfc3339(datetime.now(KST) - timedelta(days=7))

if run:
    with st.spinner("검색 중..."):
        ids = search_video_ids(
            client=client,
            q=keyword,
            max_results=max_results,
            order=pick_order(order),  # 검색 API용
            published_after=published_after,
            video_duration=pick_duration(duration_label)
        )
        if not ids:
            st.error("검색 결과가 없거나 API 오류가 발생했습니다. (API 키/권한/쿼터/필터 확인)")
            st.stop()

        df_raw, ch_map = fetch_videos_and_channels(client, ids)
        df = enrich_dataframe(df_raw, ch_map)

        # ====== 지표 컬럼 추가 (좋아요/조회수 %, 댓글/조회수 ‰, 채널 점유율 반올림) ======
        if not df.empty:
            safe_views = df["view_count"].replace(0, 1)  # 0 나눗셈 방지
            df["like_rate_pct"] = (df["like_count"] / safe_views * 100).round(2)          # 좋아요/조회수(%)
            df["comment_per_mille"] = (df["comment_count"] / safe_views * 1000).round(2)  # 댓글/조회수(‰)
            df["channel_share_pct"] = df["channel_share_pct"].round(2)                    # 보기 좋은 소수 2자리

        # ====== 정렬 적용 ======
        if not df.empty:
            if order == "조회수(내림차순)":
                df = df.sort_values("view_count", ascending=False)
            elif order == "최근 업로드":
                df = df.sort_values("published_at_dt", ascending=False)
            elif order == "CII(내림차순)":
                df = df.sort_values("CII", ascending=False)

        st.success(f"가져온 영상 {len(df)}개")

        # ====== 표시 & CSV 다운로드 ======
        if df.empty:
            st.warning("표시할 데이터가 없습니다.")
        else:
            show_cols = [
                "video_id","channel_title","title","published_at_dt",
                "channel_subscribers","view_count","like_count","comment_count",
                "like_rate_pct","comment_per_mille","duration_mmss",
                "CII","channel_share_pct","thumbnail_url"
            ]
            nice_df = df[show_cols].rename(columns={
                "video_id":"영상ID",
                "channel_title":"채널명",
                "title":"제목",
                "published_at_dt":"게시일",
                "channel_subscribers":"구독자수",
                "view_count":"조회수",
                "like_count":"좋아요수",
                "comment_count":"댓글수",
                "like_rate_pct":"좋아요/조회수(%)",
                "comment_per_mille":"댓글/조회수(‰)",
                "duration_mmss":"영상 길이",
                "CII":"CII",
                "channel_share_pct":"채널 점유율(%)",
                "thumbnail_url":"썸네일"
            })

            st.dataframe(nice_df, use_container_width=True, hide_index=True)

            # CSV 다운로드 (엑셀 호환: utf-8-sig)
            csv = nice_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("현재 결과 CSV 다운로드", data=csv, file_name="youtube_results.csv")

            # ====== 자막 일괄 수집 / TXT 다운로드 ======
            st.markdown("---")
            st.subheader("선택한 영상 자막 일괄 수집")
            options = df["video_id"].tolist()
            selected_ids = st.multiselect("자막을 수집할 영상 선택", options=options, default=options[:min(5, len(options))])
            if st.button("선택 영상 자막 수집 → TXT 다운로드", type="primary"):
                texts = []
                for vid in selected_ids:
                    try:
                        tr = YouTubeTranscriptApi.get_transcript(vid, languages=["ko","ko-KR","ko+en","en"])
                        text = "\n".join([x.get("text","") for x in tr if x.get("text")])
                    except (TranscriptsDisabled, NoTranscriptFound):
                        text = ""
                    except Exception:
                        text = ""
                    texts.append({"video_id": vid, "caption": text})
                txt = ""
                for item in texts:
                    txt += f"=== {item['video_id']} ===\n{item['caption']}\n\n"
                st.download_button("자막 TXT 다운로드", data=txt.encode("utf-8"), file_name="captions.txt")
