# app.py
import os, math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

KST = timezone(timedelta(hours=9))

def get_api_key() -> str:
    key = st.secrets.get("YT_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("YT_API_KEY", "").strip()
    return key

def build_youtube():
    key = get_api_key()
    if not key:
        st.error("API 키가 없습니다. Secrets에 YT_API_KEY를 저장하세요.")
        st.stop()
    return build('youtube','v3',developerKey=key)

def iso8601_to_seconds(iso:str)->int:
    h=m=s=0
    if not iso: return 0
    t = iso.split('T')[-1]
    num=''
    for ch in t:
        if ch.isdigit(): num+=ch
        else:
            if ch=='H' and num: h=int(num)
            if ch=='M' and num: m=int(num)
            if ch=='S' and num: s=int(num)
            num=''
    return h*3600+m*60+s

def seconds_to_mmss(s:int)->str:
    return f"{s//60}:{s%60:02d}"

def safe_int(x)->int:
    try: return int(x)
    except: return 0

def calc_cii(row:pd.Series, now:datetime)->float:
    views=max(1,safe_int(row.get('view_count',0)))
    likes=safe_int(row.get('like_count',0))
    comments=safe_int(row.get('comment_count',0))
    e=(likes+3*comments)/views
    subs=safe_int(row.get('channel_subscribers',0))
    b=math.log10(subs+1) if subs>=0 else 0.0
    days=(now-row.get('published_at_dt',now)).total_seconds()/86400.0
    f=math.exp(-days/30.0)
    return 100.0*e*b*f

def pick_duration(l): return {"전체":"any","4분 미만":"short","4~20분":"medium","20분 이상":"long"}.get(l,"any")
def pick_order(l): return {"조회수(내림차순)":"viewCount","최근 업로드":"date","관련성":"relevance","평점":"rating"}.get(l,"viewCount")
def start_of_this_month_kst(): 
    now=datetime.now(KST); return datetime(now.year,now.month,1,tzinfo=KST)
def to_rfc3339(dt:datetime)->str: return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')

def search_video_ids(client,q,max_results,order,published_after,video_duration)->List[str]:
    ids=[]; page=None
    try:
        while len(ids)<max_results:
            req=client.search().list(q=q,part='id',type='video',
                maxResults=min(50,max_results-len(ids)),
                order=order,
                publishedAfter=published_after or None,
                videoDuration=None if video_duration=='any' else video_duration,
                safeSearch='none')
            if page: req.uri+=f"&pageToken={page}"
            resp=req.execute()
            for it in resp.get('items',[]): ids.append(it['id']['videoId'])
            page=resp.get('nextPageToken')
            if not page: break
    except HttpError as e: st.warning(f"Search API 오류: {e}")
    return ids

def fetch_videos_and_channels(client,video_ids:List[str]):
    rows=[]; ch_ids=set()
    for i in range(0,len(video_ids),50):
        batch=video_ids[i:i+50]
        try: resp=client.videos().list(part='snippet,contentDetails,statistics',id=",".join(batch)).execute()
        except HttpError as e: st.warning(f"Videos API 오류:{e}"); continue
        for it in resp.get('items',[]):
            vid=it['id']; sn=it.get('snippet',{}); stc=it.get('statistics',{}); cd=it.get('contentDetails',{})
            ch=sn.get('channelId',''); ch_ids.add(ch)
            rows.append({'video_id':vid,'channel_id':ch,'channel_title':sn.get('channelTitle',''),
                'title':sn.get('title',''),'published_at':sn.get('publishedAt',''),
                'thumbnail_url':sn.get('thumbnails',{}).get('default',{}).get('url',''),
                'duration_iso':cd.get('duration','PT0S'),
                'view_count':safe_int(stc.get('viewCount',0)),
                'like_count':safe_int(stc.get('likeCount',0)),
                'comment_count':safe_int(stc.get('commentCount',0))})
    df=pd.DataFrame(rows)
    channels={}
    if ch_ids:
        for i in range(0,len(ch_ids),50):
            batch=list(ch_ids)[i:i+50]
            try: resp=client.channels().list(part='statistics',id=",".join(batch)).execute()
            except HttpError as e: st.warning(f"Channels API 오류:{e}"); continue
            for it in resp.get('items',[]):
                channels[it['id']]={'subscribers':safe_int(it.get('statistics',{}).get('subscriberCount',0))}
    return df,channels

def enrich_dataframe(df,channels):
    if df.empty: return df
    now=datetime.now(KST); df=df.copy()
    df['published_at_dt']=pd.to_datetime(df['published_at'],errors='coerce')
    df['duration_sec']=df['duration_iso'].apply(iso8601_to_seconds)
    df['duration_mmss']=df['duration_sec'].apply(seconds_to_mmss)
    df['channel_subscribers']=df['channel_id'].map(lambda x: channels.get(x,{}).get('subscribers',0))
    df['CII']=df.apply(lambda r: calc_cii(r,now),axis=1)
    counts=df['channel_title'].value_counts()
    share={k:(counts[k]/len(df))*100 for k in counts.index}
    df['channel_share_pct']=df['channel_title'].map(lambda x:share.get(x,0.0))
    return df

# UI
st.set_page_config(page_title="YouTube 탐색 & 비교",layout="wide")
st.title("YouTube 탐색 & 대량 비교 (클린)")

with st.sidebar:
    keyword=st.text_input("키워드","쿠팡꿀템")
    uploaded=st.selectbox("업로드 시기",["이번달","최근 7일","제한 없음"],0)
    duration=st.selectbox("영상 길이",["전체","4분 미만","4~20분","20분 이상"],1)
    order=st.selectbox("정렬",["조회수(내림차순)","최근 업로드","관련성","평점"],0)
    max_results=st.slider("최대 결과 수",10,200,50,10)
    sample_take=st.number_input("한 번에 선택할 개수",1,50,5,1)
    run=st.button("검색 실행",type="primary")

client=build_youtube()
published_after=""
if uploaded=="이번달": published_after=to_rfc3339(start_of_this_month_kst())
elif uploaded=="최근 7일": published_after=to_rfc3339(datetime.now(KST)-timedelta(days=7))

if run:
    with st.spinner("검색 중..."):
        ids=search_video_ids(client,keyword,max_results,pick_order(order),published_after,pick_duration(duration))
        if not ids: st.error("결과 없음 또는 API 오류"); st.stop()
        df_raw,ch_map=fetch_videos_and_channels(client,ids)
        df=enrich_dataframe(df_raw,ch_map)
        if not df.empty:
            if order=="조회수(내림차순)": df=df.sort_values("view_count",ascending=False)
            elif order=="최근 업로드": df=df.sort_values("published_at_dt",ascending=False)
            else: df=df.sort_values("CII",ascending=False)
        st.success(f"가져온 영상 {len(df)}개")
        if df.empty: st.warning("표시할 데이터 없음")
        else:
            show=["video_id","channel_title","title","published_at_dt","channel_subscribers","view_count","duration_mmss","comment_count","CII","channel_share_pct"]
            st.dataframe(df[show],use_container_width=True,hide_index=True)
