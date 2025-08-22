# app.py — 최소 비용/최대 효과 통합본
# 포함: 저쿼터 모드(기본 ON), 키 로테이션, 트렌딩 대체 모드, 캐시, 최근 성공본 백업
# UI: 큰 캔버스 표(썸네일, 영상링크, 한글 지표), CSV 다운로드
# 정렬: 조회수/최근 업로드/관련성
# 기간: 올해/이번달/이번주/최근7일/오늘/제한 없음
# 길이: 전체/4분 미만/4~20분/20분 이상

import os
from math import ceil
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -------------------
# 공통/유틸
# -------------------
KST = timezone(timedelta(hours=9))

def iso8601_to_seconds(iso: str) -> int:
    if not iso:
        return 0
    h = m = s = 0
    t = iso.split("T")[-1]
    num = ""
    for ch in t:
        if ch.isdigit():
            num += ch
        else:
            if ch == "H" and num: h = int(num)
            if ch == "M" and num: m = int(num)
            if ch == "S" and num: s = int(num)
            num = ""
    return h*3600 + m*60 + s

def seconds_to_mmss(sec: int) -> str:
    return f"{sec//60}:{sec%60:02d}"

def safe_int(x) -> int:
    try: return int(x)
    except: return 0

def pick_duration(label: str) -> str:
    return {"전체": "any", "4분 미만": "short", "4~20분": "medium", "20분 이상": "long"}.get(label, "any")

def pick_order(label: str) -> str:
    return {"조회수(내림차순)": "viewCount", "최근 업로드": "date", "관련성": "relevance"}.get(label, "viewCount")

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
    now = datetime.now(KST)
    monday = now - timedelta(days=now.weekday())
    return datetime(monday.year, monday.month, monday.day, 0, 0, 0, tzinfo=KST)

def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def now_utc_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def expand_queries(q: str) -> list[str]:
    q = q.strip()
    variants = [q, f"\"{q}\"", q.replace(" ", "")]
    tails = ["추천", "꿀템", "리뷰", "후기", "TOP", "Best", "모음", "가성비", "쇼츠"]
    for t in tails:
        variants.append(f"{q} {t}")
    seen, uniq = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq

# -------------------
# API 키 로딩 / 로테이션
# -------------------
def get_api_keys() -> list[str]:
    keys = []
    try:
        v = st.secrets.get("YT_API_KEYS")
        if isinstance(v, list):
            keys.extend(v)
    except Exception:
        pass
    # 단일 키도 보조로
    single = None
    try:
        single = st.secrets.get("YT_API_KEY") or st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        single = None
    if not single:
        single = os.environ.get("YT_API_KEY") or os.environ.get("YOUTUBE_API_KEY")
    if single:
        keys.append(single)

    keys = [k for k in keys if k]
    if not keys:
        st.error("API 키가 없습니다. YT_API_KEYS(배열) 또는 YT_API_KEY / YOUTUBE_API_KEY를 설정해 주세요.")
        st.stop()
    return keys

def build_youtube_with_key(k: str):
    return build("youtube", "v3", developerKey=k)

def with_key_rotation(callable_func, *args, **kwargs):
    last_err = None
    for k in get_api_keys():
        try:
            client = build_youtube_with_key(k)
            return callable_func(client, *args, **kwargs)
        except HttpError as e:
            last_err = e
            if "quotaExceeded" in str(e):
                continue
            else:
                raise
    raise last_err if last_err else RuntimeError("API 호출 실패")

# -------------------
# API 호출 함수
# -------------------
def search_video_ids(client, q: str, max_results: int, order: str,
                     published_after: str, video_duration: str,
                     published_before: Optional[str] = None) -> List[str]:
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
                if vid: ids.append(vid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        st.warning(f"Search API 오류: {e}")
    return ids

def _best_thumb(snippet_thumbs: dict) -> str:
    return (
        snippet_thumbs.get("high", {}).get("url")
        or snippet_thumbs.get("medium", {}).get("url")
        or snippet_thumbs.get("default", {}).get("url", "")
    )

def fetch_videos_and_channels(client, video_ids: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    ch_ids = set()

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
    if df.empty: return df
    df = df.copy()
    df["published_at_dt"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["duration_sec"] = df["duration_iso"].apply(iso8601_to_seconds)
    df["duration_mmss"] = df["duration_sec"].apply(seconds_to_mmss)
    df["channel_subscribers"] = df["channel_id"].map(lambda x: channels.get(x, {}).get("subscribers", 0))
    counts = df["channel_title"].value_counts(dropna=False)
    share = {k: (counts[k] / len(df)) * 100.0 for k in counts.index}
    df["channel_share_pct"] = df["channel_title"].map(lambda x: share.get(x, 0.0))
    return df

# -------------------
# 캐시(1시간)
# -------------------
@st.cache_data(ttl=3600)
def cached_search_ids(q, max_results, order, published_after, video_duration, published_before):
    # 캐시 적중 시 API 0
    keys = get_api_keys()  # 없으면 여기서 stop
    client = build("youtube", "v3", developerKey=keys[0])
    return search_video_ids(client, q, max_results, order, published_after, video_duration, published_before)

@st.cache_data(ttl=3600)
def cached_fetch_videos_and_channels(video_ids):
    keys = get_api_keys()
    client = build("youtube", "v3", developerKey=keys[0])
    return fetch_videos_and_channels(client, video_ids)

def trending_fallback_filter(client, keyword: str, region_code: str = "KR", max_results: int = 50) -> list[str]:
    """검색 막힐 때: 트렌딩에서 제목 포함만 필터"""
    try:
        resp = client.videos().list(
            part="id,snippet",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=min(50, max_results),
        ).execute()
        vids = []
        kw = keyword.lower().strip()
        for it in resp.get("items", []):
            vid = it.get("id")
            title = it.get("snippet", {}).get("title", "")
            if vid and (kw in title.lower()):
                vids.append(vid)
        return vids
    except HttpError:
        return []

def estimate_quota_cost(num_queries:int, want:int, ch_count:int) -> int:
    search_pages = num_queries * ceil(max(1, want)/50)
    video_calls  = ceil(max(1, want)/50)
    ch_calls     = ceil(max(1, ch_count)/50)
    return 100*search_pages + video_calls + ch_calls

# -------------------
# UI
# -------------------
st.set_page_config(page_title="대표님 전용 유튜브 분석기", layout="wide")
st.title("YouTube 탐색 & 대량 비교 (최저 쿼터)")
st.caption("키 로테이션 + 대체모드 + 캐시 | 큰 캔버스 표 · 썸네일 · 링크 · CSV")

with st.sidebar:
    st.subheader("검색 조건")
    keyword = st.text_input("키워드", value="쿠팡 꿀템")

    uploaded_when = st.selectbox("업로드 시기",
                                 ["올해", "이번달", "이번주", "최근 7일", "오늘", "제한 없음"], index=1)
    duration_label = st.selectbox("영상 길이", ["전체", "4분 미만", "4~20분", "20분 이상"], index=1)
    order = st.selectbox("정렬", ["조회수(내림차순)", "최근 업로드", "관련성"], 0)

    low_quota_mode = st.toggle("저쿼터 모드(권장)", value=True,
                               help="확장 검색 OFF + 결과 자동 제한(≤50) + 호출 최소화")
    expand_mode = st.checkbox("확장 검색(다변형)", value=False if low_quota_mode else True)

    max_results = st.slider("최대 결과 수", 10, 300, 100, 10)

    run = st.button("검색 실행", type="primary")

# 기간 변환
published_after = ""
published_before = now_utc_rfc3339()
if uploaded_when == "올해":
    published_after = to_rfc3339(start_of_this_year_kst())
elif uploaded_when == "이번달":
    published_after = to_rfc3339(start_of_this_month_kst())
elif uploaded_when == "이번주":
    published_after = to_rfc3339(start_of_this_week_kst())
elif uploaded_when == "최근 7일":
    published_after = to_rfc3339(datetime.now(KST) - timedelta(days=7))
elif uploaded_when == "오늘":
    published_after = to_rfc3339(start_of_today_kst())
else:
    published_after = ""
    published_before = None

# -------------------
# RUN
# -------------------
if run:
    with st.spinner("검색 중..."):
        want = max_results
        use_expand = expand_mode
        if low_quota_mode:
            use_expand = False
            want = min(want, 50)
            st.info("저쿼터 모드: 확장 검색 OFF, 결과 수 50으로 제한합니다.")

        # 1) 캐시 우선 → 2) 키 로테이션 → 3) 대체 모드
        ids = []
        num_queries = 1
        try:
            if use_expand:
                queries = expand_queries(keyword)
                ids_set = set()
                per = max(10, want // max(1, len(queries)))
                for qv in queries:
                    if len(ids_set) >= want:
                        break
                    got = cached_search_ids(qv, min(per, want - len(ids_set)),
                                            pick_order(order),
                                            published_after,
                                            pick_duration(duration_label),
                                            published_before)
                    ids_set.update(got)
                ids = list(ids_set)[:want]
                num_queries = len(queries)
            else:
                ids = cached_search_ids(keyword, want, pick_order(order),
                                        published_after, pick_duration(duration_label),
                                        published_before)
                num_queries = 1

        except HttpError as e:
            if "quotaExceeded" in str(e):
                st.warning("기본 키 쿼터 소진 → 다른 키로 재시도합니다.")
                def _search_call(client, q, want, order, pa, vd, pb):
                    return search_video_ids(client, q, want, order, pa, vd, pb)
                if use_expand:
                    ids_set = set()
                    queries = expand_queries(keyword)
                    per = max(10, want // max(1, len(queries)))
                    for qv in queries:
                        if len(ids_set) >= want: break
                        got = with_key_rotation(_search_call, qv, min(per, want - len(ids_set)),
                                                pick_order(order), published_after,
                                                pick_duration(duration_label), published_before)
                        ids_set.update(got)
                    ids = list(ids_set)[:want]
                    num_queries = len(queries)
                else:
                    ids = with_key_rotation(_search_call, keyword, want, pick_order(order),
                                            published_after, pick_duration(duration_label),
                                            published_before)
                    num_queries = 1
            else:
                st.error(f"Search API 오류: {e}")
                st.stop()

        if not ids:
            st.info("검색 API가 막혀 대체 모드로 전환합니다. (트렌딩 + 제목 필터)")
            def _trend_call(client, kw, region, mx):
                return trending_fallback_filter(client, kw, region_code=region, max_results=mx)
            try:
                ids = with_key_rotation(_trend_call, keyword, "KR", want)
            except Exception as e:
                st.error(f"대체 모드 실패: {e}")
                st.stop()
            if not ids:
                st.error("대체 모드도 결과가 없습니다. 키워드를 더 구체적으로/기간을 좁혀보세요.")
                # 최근 성공본 있으면 보여주기
                if "latest_df" in st.session_state:
                    st.info("최근 성공한 결과를 대신 표시합니다.")
                    st.dataframe(st.session_state["latest_df"], use_container_width=True, hide_index=True)
                st.stop()

        st.caption(f"예상 쿼터 사용(대략): 약 {estimate_quota_cost(num_queries, len(ids), 50)} 유닛")

        # 상세 정보 (캐시 → 로테이션 백업)
        try:
            df_raw, ch_map = cached_fetch_videos_and_channels(ids)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                st.warning("상세 가져오기도 다른 키로 재시도합니다.")
                def _fetch_call(client, vids):
                    return fetch_videos_and_channels(client, vids)
                df_raw, ch_map = with_key_rotation(_fetch_call, ids)
            else:
                st.error(f"Videos/Channels API 오류: {e}")
                st.stop()

        df = enrich_dataframe(df_raw, ch_map)
        if df.empty:
            st.warning("표시할 데이터가 없습니다.")
            # 최근 성공본 표시
            if "latest_df" in st.session_state:
                st.info("최근 성공한 결과를 대신 표시합니다.")
                st.dataframe(st.session_state["latest_df"], use_container_width=True, hide_index=True)
            st.stop()

        # 한글 지표
        safe_views = df["view_count"].replace(0, 1)
        df["like_rate_pct"] = (df["like_count"] / safe_views * 100).round(2)
        df["comment_per_mille"] = (df["comment_count"] / safe_views * 1000).round(2)
        df["channel_share_pct"] = df["channel_share_pct"].round(2)

        # 화면 정렬
        if order == "조회수(내림차순)":
            df = df.sort_values("view_count", ascending=False)
        elif order == "최근 업로드":
            df = df.sort_values("published_at_dt", ascending=False)

        st.success(f"가져온 영상 {len(df)}개")

        show_cols = [
            "video_id","channel_title","title","published_at_dt",
            "channel_subscribers","view_count","like_count","comment_count",
            "like_rate_pct","comment_per_mille","duration_mmss",
            "channel_share_pct","thumbnail_url"
        ]
        nice_df = df[show_cols].copy()
        nice_df["영상 링크"] = "https://www.youtube.com/watch?v=" + nice_df["video_id"]
        nice_df["썸네일"] = nice_df["thumbnail_url"]
        nice_df = nice_df.rename(columns={
            "channel_title":"채널명","title":"제목","published_at_dt":"게시일",
            "channel_subscribers":"구독자수","view_count":"조회수","like_count":"좋아요수",
            "comment_count":"댓글수","like_rate_pct":"좋아요/조회수(%)",
            "comment_per_mille":"댓글/조회수(‰)","duration_mmss":"영상 길이",
            "channel_share_pct":"채널 점유율(%)",
        })

        display_cols = [
            "썸네일","영상 링크","채널명","제목","게시일",
            "구독자수","조회수","좋아요수","댓글수",
            "좋아요/조회수(%)","댓글/조회수(‰)","영상 길이","채널 점유율(%)"
        ]

        height = int(min(1800, 46 * (len(nice_df) + 2)))
        if height < 600: height = 600

        st.dataframe(
            nice_df[display_cols],
            use_container_width=True,
            hide_index=True,
            height=height,
            column_config={
                "썸네일": st.column_config.ImageColumn("썸네일", width=180),
                "영상 링크": st.column_config.LinkColumn("영상 보기", display_text="열기"),
            },
        )

        # 최근 성공본으로 백업
        st.session_state["latest_df"] = nice_df[display_cols].copy()

        # CSV 다운로드
        csv = nice_df[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("현재 결과 CSV 다운로드", data=csv, file_name="youtube_results.csv")
