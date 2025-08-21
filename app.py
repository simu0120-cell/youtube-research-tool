# app.py — 대표님 전용 유튜브 분석기 (최신 최종본, 요청 수정 반영)
# 기간(올해/이번달/이번주/최근7일/오늘/무제한), 확장검색, 한글지표,
# 썸네일 이미지 확대(원본비율), 영상 링크, 미리보기, CSV, 자막 일괄 수집
# 정렬: 조회수 / 최근 업로드 / 관련성  (CII 노출/정렬 제거)
# 표에서 "영상ID" 열 숨김 (내부 사용만)

import os
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# -----------------------------
# 공통 상수/유틸
# -----------------------------
KST = timezone(timedelta(hours=9))

def get_api_key() -> str:
    """Secrets/환경변수에서 YT_API_KEY 또는 YOUTUBE_API_KEY 자동 인식"""
    key = None
    try:
        key = st.secrets.get("YT_API_KEY") or st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("YT_API_KEY") or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        st.error(
            "API 키가 없습니다.\n"
            "Streamlit Secrets에 `YT_API_KEY=\"...\"` 또는 `YOUTUBE_API_KEY=\"...\"` 저장하거나\n"
            "환경변수로 설정해 주세요."
        )
        st.stop()
    return key

def build_youtube():
    return build("youtube", "v3", developerKey=get_api_key())

def iso8601_to_seconds(iso: str) -> int:
    # 'PT1H2M10S' → 초
    h = m = s = 0
    if not iso:
        return 0
    t = iso.split("T")[-1]
    num = ""
    for ch in t:
        if ch.isdigit():
            num += ch
        else:
            if ch == "H" and num:
                h = int(num)
            if ch == "M" and num:
                m = int(num)
            if ch == "S" and num:
                s = int(num)
            num = ""
    return h * 3600 + m * 60 + s

def seconds_to_mmss(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"

def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def calc_cii(row: pd.Series, now: datetime) -> float:
    """
    CII(가중 반응지표) 내부 계산(현재는 화면 미노출)
      참여도 e = (좋아요 + 3*댓글) / 조회수
      채널규모 b = log10(구독자+1)
      신선도 f = exp(-일수/30)
    """
    views = max(1, safe_int(row.get("view_count", 0)))
    likes = safe_int(row.get("like_count", 0))
    comments = safe_int(row.get("comment_count", 0))
    e = (likes + 3 * comments) / views
    subs = safe_int(row.get("channel_subscribers", 0))
    b = math.log10(subs + 1) if subs >= 0 else 0.0
    published_at = row.get("published_at_dt", now)
    days = max(0.0, (now - published_at).total_seconds() / 86400.0)
    f = math.exp(-days / 30.0)
    return 100.0 * e * b * f

def pick_duration(label: str) -> str:
    return {"전체": "any", "4분 미만": "short", "4~20분": "medium", "20분 이상": "long"}.get(label, "any")

def pick_order(label: str) -> str:
    # 검색 API용 정렬 매핑(화면 정렬은 아래에서 별도 처리)
    return {
        "조회수(내림차순)": "viewCount",
        "최근 업로드": "date",
        "관련성": "relevance",
    }.get(label, "viewCount")

def start_of_this_month_kst() -> datetime:
    now = datetime.now(KST)
    return datetime(now.year, now.month, 1, 0, 0, 0, tzinfo=KST)

def start_of_this_year_kst() -> datetime:
    now = datetime.now(KST)
    return datetime(now.year, 1, 1, 0, 0, 0, tzinfo=KST)

def start_of_today_kst() -> datetime:
    now = datetime.now(KST)
    return datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=KST)

def start_of_this_week_kst() -> datetime:
    # 한국 기준 월요일 00:00
    now = datetime.now(KST)
    monday = now - timedelta(days=now.weekday())
    return datetime(monday.year, monday.month, monday.day, 0, 0, 0, tzinfo=KST)

def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def now_utc_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# 확장 검색(다변형)
def expand_queries(q: str) -> list[str]:
    q = q.strip()
    variants = [
        q,
        f"\"{q}\"",
        q.replace(" ", ""),
    ]
    tails = ["추천", "꿀템", "리뷰", "후기", "TOP", "Best", "모음", "가성비", "쇼츠"]
    for t in tails:
        variants.append(f"{q} {t}")
    seen, uniq = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

# -----------------------------
# API 호출
# -----------------------------
def search_video_ids(
    client, q: str, max_results: int, order: str,
    published_after: str, video_duration: str,
    published_before: Optional[str] = None
) -> List[str]:
    ids: List[str] = []
    page_token = None
    try:
        while len(ids) < max_results:
            req = client.search().list(
                q=q, part="id", type="video",
                maxResults=min(50, max_results - len(ids)),
                order=order,
                publishedAfter=published_after if published_after else None,
                publishedBefore=published_before if published_before else None,
                videoDuration=video_duration if video_duration != "any" else None,
                safeSearch="none",
            )
            if page_token:
                req.uri += f"&pageToken={page_token}"
            resp = req.execute()
            for item in resp.get("items", []):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    ids.append(vid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        st.warning(f"Search API 오류: {e}")
    return ids

def _best_thumb(snippet_thumbs: dict) -> str:
    # 잘림 없이 크게 보기: high → medium → default 우선 사용
    return (
        snippet_thumbs.get("high", {}).get("url")
        or snippet_thumbs.get("medium", {}).get("url")
        or snippet_thumbs.get("default", {}).get("url", "")
    )

def fetch_videos_and_channels(client, video_ids: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    ch_ids = set()

    # videos
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = client.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch)
            ).execute()
        except HttpError as e:
            st.warning(f"Videos API 오류: {e}")
            continue
        for item in resp.get("items", []):
            vid = item["id"]
            sn = item.get("snippet", {})
            stc = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            ch_id = sn.get("channelId", "")
            ch_ids.add(ch_id)
            rows.append({
                "video_id": vid,
                "channel_id": ch_id,
                "channel_title": sn.get("channelTitle", ""),
                "title": sn.get("title", ""),
                "published_at": sn.get("publishedAt", ""),
                "thumbnail_url": _best_thumb(sn.get("thumbnails", {})),
                "duration_iso": cd.get("duration", "PT0S"),
                "view_count": safe_int(stc.get("viewCount", 0)),
                "like_count": safe_int(stc.get("likeCount", 0)),
                "comment_count": safe_int(stc.get("commentCount", 0)),
            })
    df = pd.DataFrame(rows)

    # channels (구독자)
    channels: Dict[str, Dict[str, Any]] = {}
    if ch_ids:
        ids = list(ch_ids)
        for i in range(0, len(ids), 50):
            batch = ids[i:i+50]
            try:
                resp = client.channels().list(
                    part="statistics",
                    id=",".join(batch)
                ).execute()
            except HttpError as e:
                st.warning(f"Channels API 오류: {e}")
                continue
            for item in resp.get("items", []):
                cid = item["id"]
                stc = item.get("statistics", {})
                channels[cid] = {"subscribers": safe_int(stc.get("subscriberCount", 0))}
    return df, channels

def enrich_dataframe(df: pd.DataFrame, channels: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now(KST)
    df = df.copy()
    df["published_at_dt"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["duration_sec"] = df["duration_iso"].apply(iso8601_to_seconds)
    df["duration_mmss"] = df["duration_sec"].apply(seconds_to_mmss)
    df["channel_subscribers"] = df["channel_id"].map(lambda x: channels.get(x, {}).get("subscribers", 0))
    # CII는 내부만 (노출 X)
    df["CII"] = df.apply(lambda r: calc_cii(r, now), axis=1)

    # 채널 점유율(%)
    counts = df["channel_title"].value_counts(dropna=False)
    share = {k: (counts[k] / len(df)) * 100.0 for k in counts.index}
    df["channel_share_pct"] = df["channel_title"].map(lambda x: share.get(x, 0.0))
    return df

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="대표님 전용 유튜브 분석기", layout="wide")
st.title("YouTube 탐색 & 대량 비교 (최종본)")
st.caption("기간 확장 + 확장검색 + 한글지표 + 썸네일 확대 + 링크 + 미리보기 + CSV + 자막 일괄 수집")

with st.sidebar:
    st.subheader("검색 조건")
    keyword = st.text_input("키워드", value="쿠팡 꿀템")

    uploaded_when = st.selectbox(
        "업로드 시기",
        ["올해", "이번달", "이번주", "최근 7일", "오늘", "제한 없음"],
        index=1  # 기본: 이번달
    )

    duration_label = st.selectbox("영상 길이", ["전체", "4분 미만", "4~20분", "20분 이상"], index=1)
    # 🔽 정렬에서 'CII' 제거, '관련성' 추가
    order = st.selectbox("정렬", ["조회수(내림차순)", "최근 업로드", "관련성"], 0)

    # 확장 검색(다변형)
    expand_mode = st.checkbox("확장 검색(다변형)", value=True, help="따옴표/공백제거/연관단어 조합으로 더 많은 소스 수집")

    # 결과 수 상향
    max_results = st.slider("최대 결과 수", 10, 300, 100, 10)

    sample_take = st.number_input("한 번에 선택할 개수", 1, 50, 5, 1)
    run = st.button("검색 실행", type="primary")

client = build_youtube()

# === 날짜 범위 계산 ===
published_after = ""
published_before = now_utc_rfc3339()  # 상한은 현재 시각

if uploaded_when == "올해":
    start_dt = start_of_this_year_kst()
    published_after = to_rfc3339(start_dt)
elif uploaded_when == "이번달":
    start_dt = start_of_this_month_kst()
    published_after = to_rfc3339(start_dt)
elif uploaded_when == "이번주":
    start_dt = start_of_this_week_kst()
    published_after = to_rfc3339(start_dt)
elif uploaded_when == "최근 7일":
    start_dt = datetime.now(KST) - timedelta(days=7)
    published_after = to_rfc3339(start_dt)
elif uploaded_when == "오늘":
    start_dt = start_of_today_kst()
    published_after = to_rfc3339(start_dt)
else:  # 제한 없음
    published_after = ""
    published_before = None  # 상한 미적용

# -----------------------------
# 실행
# -----------------------------
if run:
    with st.spinner("검색 중..."):
        # 확장 검색
        if expand_mode:
            queries = expand_queries(keyword)
            want = max_results
            ids_set: set[str] = set()
            per = max(10, want // max(1, len(queries)))  # 쿼리당 최소 10개
            for qv in queries:
                if len(ids_set) >= want:
                    break
                got = search_video_ids(
                    client=client,
                    q=qv,
                    max_results=min(per, want - len(ids_set)),
                    order=pick_order(order),
                    published_after=published_after,
                    video_duration=pick_duration(duration_label),
                    published_before=published_before,
                )
                ids_set.update(got)
            ids = list(ids_set)[:want]
        else:
            ids = search_video_ids(
                client=client,
                q=keyword,
                max_results=max_results,
                order=pick_order(order),
                published_after=published_after,
                video_duration=pick_duration(duration_label),
                published_before=published_before,
            )

        if not ids:
            st.error("검색 결과가 없거나 API 오류가 발생했습니다. (API 키/권한/쿼터/필터 확인)")
            st.stop()

        df_raw, ch_map = fetch_videos_and_channels(client, ids)
        df = enrich_dataframe(df_raw, ch_map)

        # ===== 지표 컬럼 추가 (CII는 내부만) =====
        if not df.empty:
            safe_views = df["view_count"].replace(0, 1)
            df["like_rate_pct"] = (df["like_count"] / safe_views * 100).round(2)          # 좋아요/조회수(%)
            df["comment_per_mille"] = (df["comment_count"] / safe_views * 1000).round(2)  # 댓글/조회수(‰)
            df["channel_share_pct"] = df["channel_share_pct"].round(2)

        # ===== 화면 정렬 =====
        if not df.empty:
            if order == "조회수(내림차순)":
                df = df.sort_values("view_count", ascending=False)
            elif order == "최근 업로드":
                df = df.sort_values("published_at_dt", ascending=False)
            # '관련성'은 검색 API가 반환한 순서를 유지 (추가 소트 없음)

        st.success(f"가져온 영상 {len(df)}개")

        if df.empty:
            st.warning("표시할 데이터가 없습니다.")
            st.stop()

        # ===== 표시용 DF 구성 =====
        show_cols = [
            "video_id","channel_title","title","published_at_dt",
            "channel_subscribers","view_count","like_count","comment_count",
            "like_rate_pct","comment_per_mille","duration_mmss",
            "channel_share_pct","thumbnail_url"
        ]
        nice_df = df[show_cols].copy()
        nice_df["영상 링크"] = "https://www.youtube.com/watch?v=" + nice_df["video_id"]
        nice_df["썸네일"] = nice_df["thumbnail_url"]

        # 한글 머리말로 변경 (영상ID는 내부만 사용)
        nice_df = nice_df.rename(columns={
            # "video_id":"영상ID",  # ❌ 테이블 노출 제거
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
            "channel_share_pct":"채널 점유율(%)",
        })

        # 노출 컬럼(영상ID, CII 미포함)
        display_cols = [
            "썸네일","영상 링크",
            "채널명","제목","게시일",
            "구독자수","조회수","좋아요수","댓글수",
            "좋아요/조회수(%)","댓글/조회수(‰)","영상 길이","채널 점유율(%)"
        ]

        st.dataframe(
            nice_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                # 썸네일 크게(원본 비율 그대로 렌더)
                "썸네일": st.column_config.ImageColumn("썸네일", width=180),
                "영상 링크": st.column_config.LinkColumn("영상 보기", display_text="열기"),
            },
        )

        # CSV 다운로드 (엑셀 호환: utf-8-sig)
        csv = nice_df[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("현재 결과 CSV 다운로드", data=csv, file_name="youtube_results.csv")

        # ===== 미리보기 =====
        st.markdown("---")
        st.subheader("영상 미리보기")
        # 영상ID는 표에 보이지 않지만 선택 문자열엔 포함해도 무방(확인 용)
        sel_options = (nice_df["제목"] + " | " + nice_df["채널명"] + " | " + df["video_id"]).tolist()
        sel = st.selectbox("미리볼 영상 선택", options=["선택 안 함"] + sel_options, index=0)
        if sel != "선택 안 함":
            vid = sel.split("|")[-1].strip()
            st.video(f"https://www.youtube.com/watch?v={vid}")

        # ===== 자막 일괄 수집 / TXT 다운로드 =====
        st.markdown("---")
        st.subheader("선택한 영상 자막 일괄 수집")
        options = df["video_id"].tolist()
        default_pick = options[:min(5, len(options))]
        selected_ids = st.multiselect("자막을 수집할 영상 선택", options=options, default=default_pick)
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
