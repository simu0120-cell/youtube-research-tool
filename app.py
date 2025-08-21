# app.py
# YouTube 탐색 & 대량 비교(MVP) – Streamlit 버전
# 필요: 환경변수 YT_API_KEY 설정
# 실행: streamlit run app.py

import os
import math
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# -----------------------------
# 유틸
# -----------------------------

KST = timezone(timedelta(hours=9))

def get_api_key() -> str:
    key = os.environ.get("YT_API_KEY", "").strip()
    return key

def build_youtube():
    key = get_api_key()
    if not key:
        st.error("환경변수 YT_API_KEY가 비었습니다. 터미널에서 'export YT_API_KEY=your_api_key' 설정 후 다시 실행하세요.")
        st.stop()
    return build('youtube', 'v3', developerKey=key)

def iso8601_duration_to_seconds(duration: str) -> int:
    # e.g., 'PT1H2M10S', 'PT3M40S', 'PT45S'
    hours, minutes, seconds = 0, 0, 0
    if duration.startswith('P'):
        duration = duration[1:]
    if 'T' in duration:
        date_part, time_part = duration.split('T', 1)
    else:
        date_part, time_part = duration, ''
    num = ''
    for ch in time_part:
        if ch.isdigit():
            num += ch
        else:
            if ch == 'H' and num:
                hours = int(num)
            elif ch == 'M' and num:
                minutes = int(num)
            elif ch == 'S' and num:
                seconds = int(num)
            num = ''
    return hours*3600 + minutes*60 + seconds

def seconds_to_mmss(s: int) -> str:
    m = s // 60
    s = s % 60
    return f"{m}:{s:02d}"

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def calc_cii(row: pd.Series, now: datetime) -> float:
    # 참여도 e = (like + 3*comment) / views
    views = max(1, safe_int(row.get('view_count', 0)))
    likes = safe_int(row.get('like_count', 0))
    comments = safe_int(row.get('comment_count', 0))
    e = (likes + 3*comments) / views

    # 채널 크기 보정 b = log10(subscribers+1)
    subs = safe_int(row.get('channel_subscribers', 0))
    b = math.log10(subs + 1) if subs >= 0 else 0.0

    # 신선도 f = exp(- days / 30)
    published_at = row.get('published_at_dt', now)
    days = max(0.0, (now - published_at).total_seconds() / 86400.0)
    f = math.exp(-days / 30.0)

    return 100.0 * e * b * f

def channel_contribution(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    K = len(df)
    counts = df['channel_title'].value_counts(dropna=False)
    return {ch: (counts.get(ch, 0) / K) * 100.0 for ch in counts.index}

@st.cache_data(show_spinner=False)
def search_video_ids(client, q: str, max_results: int, order: str, published_after: str, video_duration: str) -> List[str]:
    ids = []
    page_token = None
    while len(ids) < max_results:
        try:
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
            break
    return ids

@st.cache_data(show_spinner=False)
def fetch_videos_and_channels(client, video_ids: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    ch_ids = set()
    # videos.list
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
            ch_id = sn.get('channelId')
            ch_ids.add(ch_id)
            rows.append({
                'video_id': vid,
                'channel_id': ch_id,
                'channel_title': sn.get('channelTitle', ''),
                'title': sn.get('title', ''),
                'published_at': sn.get('publishedAt', ''),
                'thumbnail_url': sn.get('thumbnails', {}).get('default', {}).get('url', ''),
                'duration_iso': cd.get('duration', 'PT0S'),
                'view_count': safe_int(stc.get('viewCount', 0)),
                'like_count': safe_int(stc.get('likeCount', 0)),
                'comment_count': safe_int(stc.get('commentCount', 0)),
            })
    df = pd.DataFrame(rows)

    # channels.list
    channels = {}
    for i in range(0, len(ch_ids), 50):
        batch = list(ch_ids)[i:i+50]
        try:
            resp = client.channels().list(
                part='statistics,snippet',
                id=",".join(batch)
            ).execute()
        except HttpError as e:
            st.warning(f"Channels API 오류: {e}")
            continue
        for item in resp.get('items', []):
            cid = item['id']
            stc = item.get('statistics', {})
            channels[cid] = {
                'subscribers': safe_int(stc.get('subscriberCount', 0)),
                'videoCount': safe_int(stc.get('videoCount', 0)),
            }

    return df, channels

def enrich_dataframe(df: pd.DataFrame, channels: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now(KST)
    df = df.copy()
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['duration_sec'] = df['duration_iso'].apply(iso8601_duration_to_seconds)
    df['duration_mmss'] = df['duration_sec'].apply(seconds_to_mmss)
    df['channel_subscribers'] = df['channel_id'].map(lambda x: channels.get(x, {}).get('subscribers', 0))
    df['CII'] = df.apply(lambda r: calc_cii(r, now), axis=1)
    # 채널 기여도(%) 계산
    share_map = channel_contribution(df)
    df['channel_share_pct'] = df['channel_title'].map(lambda x: share_map.get(x, 0.0))
    return df

def pick_video_duration_filter(choice: str) -> str:
    # Streamlit UI의 라벨 -> API 파라미터
    return {
        "전체": "any",
        "4분 미만": "short",     # < 4 minutes
        "4~20분": "medium",       # 4-20 minutes
        "20분 이상": "long"
    }.get(choice, "any")

def pick_order(choice: str) -> str:
    return {
        "조회수(내림차순)": "viewCount",
        "최근 업로드": "date",
        "관련성": "relevance",
        "평점": "rating"
    }.get(choice, "viewCount")

def default_start_of_this_month_kst() -> datetime:
    now = datetime.now(KST)
    return datetime(now.year, now.month, 1, tzinfo=KST)

def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

def fetch_captions_text(video_id: str) -> str:
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "ko-KR", "ko+en", "en"])
        return "\n".join([x.get("text","") for x in tr if x.get("text")])
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception:
        return ""

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="YouTube 탐색 & 비교(MVP)", layout="wide")
st.title("YouTube 탐색 & 대량 비교 (MVP)")
st.caption("키워드로 영상을 탐색하고, 조회수/길이/날짜 필터와 커스텀 지표(CII), 채널 기여도를 보며 비교할 수 있습니다. 자막도 일괄 수집 가능.")

with st.sidebar:
    st.subheader("검색 조건")
    keyword = st.text_input("키워드", value="쿠팡꿀템")
    uploaded_when = st.selectbox("업로드 시기", ["이번달", "최근 7일", "제한 없음"], index=0)
    duration_label = st.selectbox("영상 길이", ["전체", "4분 미만", "4~20분", "20분 이상"], index=1)
    order_label = st.selectbox("정렬", ["조회수(내림차순)", "최근 업로드", "관련성", "평점"], index=0)
    max_results = st.slider("최대 결과 수", min_value=10, max_value=200, value=50, step=10)
    sample_take = st.number_input("한 번에 선택할 개수", min_value=1, max_value=50, value=5, step=1)
    run = st.button("검색 실행", type="primary")

col_info, col_actions = st.columns([3,1])

with col_info:
    st.markdown("#### 결과")

client = build_youtube()

published_after = ""
if uploaded_when == "이번달":
    published_after = to_rfc3339(default_start_of_this_month_kst())
elif uploaded_when == "최근 7일":
    published_after = to_rfc3339(datetime.now(KST) - timedelta(days=7))

if run:
    with st.spinner("검색 중..."):
        ids = search_video_ids(
            client=client,
            q=keyword,
            max_results=max_results,
            order=pick_order(order_label),
            published_after=published_after,
            video_duration=pick_video_duration_filter(duration_label)
        )
        df_raw, ch_map = fetch_videos_and_channels(client, ids)
        df = enrich_dataframe(df_raw, ch_map)

        # 정렬 적용
        if order_label == "조회수(내림차순)":
            df = df.sort_values("view_count", ascending=False)
        elif order_label == "최근 업로드":
            df = df.sort_values("published_at_dt", ascending=False)
        else:
            df = df.sort_values("CII", ascending=False)

        st.success(f"가져온 영상 {len(df)}개")
        if not df.empty:
            show_cols = [
                "video_id","channel_title","title","published_at_dt","channel_subscribers",
                "view_count","duration_mmss","comment_count","CII","channel_share_pct","thumbnail_url"
            ]
            nice_df = df[show_cols].rename(columns={
                "video_id":"영상ID","channel_title":"채널명","title":"제목","published_at_dt":"게시일",
                "channel_subscribers":"구독자수","view_count":"조회수","duration_mmss":"영상 길이",
                "comment_count":"댓글수","CII":"CII","channel_share_pct":"채널 기여도(%)","thumbnail_url":"썸네일"
            })
            st.dataframe(nice_df, use_container_width=True, hide_index=True)

            # 선택 섹션
            st.markdown("---")
            st.subheader("선택 & 액션")
            options = df["video_id"].tolist()
            selected_ids = st.multiselect("자막 수집/다운로드할 영상 선택", options=options, default=options[:sample_take])
            do_captions = st.button("선택 영상 자막 수집 → TXT 다운로드")
            if do_captions and selected_ids:
                texts = []
                with st.spinner("자막 수집 중..."):
                    for vid in selected_ids:
                        t = fetch_captions_text(vid)
                        texts.append({"video_id": vid, "caption": t})
                # TXT로 합치기
                txt = ""
                for item in texts:
                    txt += f"=== {item['video_id']} ===\n{item['caption']}\n\n"
                st.download_button("자막 TXT 다운로드", data=txt.encode("utf-8"), file_name="captions.txt")

            # CSV 익스포트
            csv = nice_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("현재 결과 CSV 다운로드", data=csv, file_name="youtube_results.csv")

with col_actions:
    st.markdown("#### 도움말")
    st.markdown("- 키워드, 기간, 길이, 정렬을 정하고 '검색 실행'을 누르세요.")
    st.markdown("- CII는 참여도×채널규모×신선도를 반영한 지표입니다.")
    st.markdown("- '선택 영상 자막 수집'으로 공개 자막만 모읍니다.")
    st.markdown("- .env 대신 OS 환경변수 YT_API_KEY를 사용합니다.")
