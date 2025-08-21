import streamlit as st
import pandas as pd
import datetime
from googleapiclient.discovery import build

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="대표님 전용 유튜브 분석기", layout="wide")

API_KEY = st.secrets["YOUTUBE_API_KEY"]  # 🔑 streamlit secrets에 저장 필요
client = build("youtube", "v3", developerKey=API_KEY)

# =========================
# 유틸 함수들
# =========================
def pick_order(order_label: str) -> str:
    if "조회수" in order_label:
        return "viewCount"
    elif "최근" in order_label:
        return "date"
    else:
        return "relevance"

def pick_duration(label: str) -> str:
    if label == "4분 미만":
        return "short"
    elif label == "4~20분":
        return "medium"
    elif label == "20분 이상":
        return "long"
    return "any"

def expand_queries(q: str) -> list[str]:
    """간단한 다변형 쿼리 생성: 따옴표, 공백제거, 연관 단어 조합."""
    q = q.strip()
    variants = [
        q,
        f"\"{q}\"",
        q.replace(" ", ""),
    ]
    tails = ["추천","꿀템","리뷰","후기","TOP","Best","모음","가성비","쇼츠"]
    for t in tails:
        variants.append(f"{q} {t}")
    seen, uniq = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq

def search_video_ids(client, q, max_results, order, published_after=None, video_duration="any", published_before=None):
    ids = []
    req = client.search().list(
        q=q,
        part="id",
        type="video",
        order=order,
        maxResults=min(50, max_results),
        publishedAfter=published_after,
        publishedBefore=published_before,
        videoDuration=video_duration
    )
    res = req.execute()
    for item in res.get("items", []):
        ids.append(item["id"]["videoId"])
    return ids

def fetch_videos_and_channels(client, ids):
    # 비디오 메타 수집
    df_list = []
    ch_map = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i+50]
        res = client.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        ).execute()
        for item in res.get("items", []):
            vid = item["id"]
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            df_list.append({
                "video_id": vid,
                "channel_id": snippet["channelId"],
                "channel_title": snippet["channelTitle"],
                "title": snippet["title"],
                "published_at": snippet["publishedAt"],
                "thumbnail_url": snippet["thumbnails"]["default"]["url"],
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": item["contentDetails"]["duration"],
            })
    df = pd.DataFrame(df_list)
    return df, ch_map

def enrich_dataframe(df, ch_map):
    if df.empty:
        return df
    df["published_at_dt"] = pd.to_datetime(df["published_at"])
    df["duration_mmss"] = df["duration"].astype(str)
    df["like_rate_pct"] = df["like_count"] / df["view_count"].replace(0,1) * 100
    df["comment_per_mille"] = df["comment_count"] / df["view_count"].replace(0,1) * 1000
    df["CII"] = (
        df["like_rate_pct"].fillna(0) * 0.7 +
        df["comment_per_mille"].fillna(0) * 0.3
    )
    df["channel_share_pct"] = 100 / df.groupby("channel_title")["channel_title"].transform("count")
    return df

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.subheader("검색 조건")
    keyword = st.text_input("키워드", value="쿠팡 꿀템")

    uploaded_when = st.selectbox(
        "업로드 시기",
        ["올해", "이번달", "이번주", "최근 7일", "오늘", "제한 없음"],
        index=1
    )

    duration_label = st.selectbox("영상 길이", ["전체","4분 미만","4~20분","20분 이상"], index=1)
    order = st.selectbox("정렬", ["조회수(내림차순)", "최근 업로드", "CII(내림차순)"], 0)

    expand_mode = st.checkbox("확장 검색(다변형)", value=True, help="따옴표/공백제거/연관단어 조합으로 더 많은 소스 수집")

    max_results = st.slider("최대 결과 수", 10, 300, 80, 10)

    sample_take = st.number_input("한 번에 선택할 개수", 1, 50, 5, 1)
    run = st.button("검색 실행", type="primary")

# =========================
# 업로드 시기 필터 계산
# =========================
published_after, published_before = None, None
today = datetime.datetime.utcnow()

if uploaded_when != "제한 없음":
    if uploaded_when == "오늘":
        published_after = (today - datetime.timedelta(days=1)).isoformat("T")+"Z"
    elif uploaded_when == "최근 7일":
        published_after = (today - datetime.timedelta(days=7)).isoformat("T")+"Z"
    elif uploaded_when == "이번주":
        published_after = (today - datetime.timedelta(days=7)).isoformat("T")+"Z"
    elif uploaded_when == "이번달":
        published_after = (today - datetime.timedelta(days=30)).isoformat("T")+"Z"
    elif uploaded_when == "올해":
        published_after = (today - datetime.timedelta(days=365)).isoformat("T")+"Z"

# =========================
# 메인 실행
# =========================
if run:
    with st.spinner("검색 중..."):
        if expand_mode:
            queries = expand_queries(keyword)
            want = max_results
            ids_set: set[str] = set()
            per = max(10, want // max(1, len(queries)))
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
                    published_before=published_before
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
                published_before=published_before
            )

        if not ids:
            st.error("검색 결과가 없습니다. API 키/쿼터/필터 확인 필요")
            st.stop()

        df_raw, ch_map = fetch_videos_and_channels(client, ids)
        df = enrich_dataframe(df_raw, ch_map)

        # 표 준비
        show_cols = [
            "video_id","channel_title","title","published_at_dt",
            "view_count","like_count","comment_count",
            "like_rate_pct","comment_per_mille","duration_mmss",
            "CII","channel_share_pct","thumbnail_url"
        ]
        nice_df = df[show_cols].copy()
        nice_df["영상 링크"] = "https://www.youtube.com/watch?v=" + nice_df["video_id"]
        nice_df["썸네일"] = nice_df["thumbnail_url"]

        nice_df = nice_df.rename(columns={
            "video_id":"영상ID",
            "channel_title":"채널명",
            "title":"제목",
            "published_at_dt":"게시일",
            "view_count":"조회수",
            "like_count":"좋아요수",
            "comment_count":"댓글수",
            "like_rate_pct":"좋아요/조회수(%)",
            "comment_per_mille":"댓글/조회수(‰)",
            "duration_mmss":"영상 길이",
            "CII":"CII",
            "channel_share_pct":"채널 점유율(%)",
        })

        display_cols = [
            "썸네일", "영상 링크",
            "영상ID","채널명","제목","게시일",
            "조회수","좋아요수","댓글수",
            "좋아요/조회수(%)","댓글/조회수(‰)","영상 길이","CII","채널 점유율(%)"
        ]

        st.dataframe(
            nice_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "썸네일": st.column_config.ImageColumn("썸네일", width="small"),
                "영상 링크": st.column_config.LinkColumn("영상 보기", display_text="열기"),
            },
        )

        csv = nice_df[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("현재 결과 CSV 다운로드", data=csv, file_name="youtube_results.csv")

        st.markdown("---")
        st.subheader("영상 미리보기")
        sel_options = (nice_df["제목"] + " | " + nice_df["채널명"] + " | " + nice_df["영상ID"]).tolist()
        sel = st.selectbox("미리볼 영상 선택", options=["선택 안 함"] + sel_options, index=0)
        if sel != "선택 안 함":
            vid = sel.split("|")[-1].strip()
            st.video(f"https://www.youtube.com/watch?v={vid}")
