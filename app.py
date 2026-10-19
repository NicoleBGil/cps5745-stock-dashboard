import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Comparison | CPS 5745", page_icon="📈", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.6rem}.stApp{background:#f8fafc}h1,h2,h3{color:#142b49}</style>""", unsafe_allow_html=True)
st.title("📈 Interactive Stock Comparison Dashboard")
st.caption("Nicole Bencosme-Gil  |  Developed for CPS*5745")
st.write("Compare publicly traded companies using historical market data. All six charts respond to the shared filters below.")

COMPANIES = {"Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "NVIDIA (NVDA)": "NVDA", "Alphabet (GOOGL)": "GOOGL", "Amazon (AMZN)": "AMZN", "Meta (META)": "META"}
today = dt.date.today()
with st.container(border=True):
    a,b,c = st.columns([2,1,1])
    with a:
        chosen = st.multiselect("Companies (select at least three)", list(COMPANIES), default=list(COMPANIES)[:3])
    with b:
        start = st.date_input("Start date", value=today-dt.timedelta(days=365), max_value=today)
    with c:
        end = st.date_input("End date", value=today, max_value=today)
if len(chosen)<3:
    st.warning("Select at least three companies to compare.")
    st.stop()
if start>=end:
    st.error("The start date must be earlier than the end date.")
    st.stop()

symbols=[COMPANIES[x] for x in chosen]
@st.cache_data(ttl=3600, show_spinner=False)
def load(tickers, start_date, end_date):
    raw=yf.download(list(tickers), start=start_date.isoformat(), end=(end_date+dt.timedelta(days=1)).isoformat(), auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame(),pd.DataFrame()
    if isinstance(raw.columns,pd.MultiIndex):
        close=raw['Close'].copy()
        volume=raw['Volume'].copy()
    else:
        close=raw[['Close']].rename(columns={'Close':tickers[0]})
        volume=raw[['Volume']].rename(columns={'Volume':tickers[0]})
    return close.reindex(columns=list(tickers)).sort_index(),volume.reindex(columns=list(tickers)).sort_index()

with st.spinner("Loading latest available historical stock data..."):
    try:
        prices,volume=load(tuple(symbols),start,end)
    except Exception as e:
        st.error(f"Could not retrieve market data: {e}")
        st.stop()
if prices.empty or prices.dropna(how='all').empty:
    st.error("No market data was returned. Try a wider date range or refresh later.")
    st.stop()
prices=prices.ffill().dropna(how='all')
valid=[s for s in symbols if prices[s].notna().sum()>1]
if len(valid)<3:
    st.error("Fewer than three companies have enough data for the selected dates. Change the date range.")
    st.stop()
prices=prices[valid]
volume=volume[valid]
common=prices.dropna(how='any')
if len(common)<2:
    st.error("Not enough overlapping trading dates to compare these companies.")
    st.stop()
returns=common.pct_change(fill_method=None).dropna(how='all')*100
normalized=common.div(common.iloc[0]).mul(100)
volatility=returns.rolling(20,min_periods=20).std()*np.sqrt(252)
labels={s: next(k for k,v in COMPANIES.items() if v==s) for s in valid}

def line_chart(frame,title,yaxis):
    long=frame.rename(columns=labels).reset_index().melt(id_vars=frame.index.name or 'Date',var_name='Company',value_name='Value')
    fig=px.line(long,x=long.columns[0],y='Value',color='Company',title=title,labels={'Value':yaxis,long.columns[0]:'Trading date'})
    fig.update_layout(hovermode='x unified',legend_title_text='Company',margin=dict(t=55,b=35,l=30,r=15))
    fig.update_yaxes(title_text=yaxis)
    return fig

st.caption(f"Source: Yahoo Finance via yfinance • Latest available trading date: {prices.index.max():%B %d, %Y} • Prices are split/dividend adjusted (auto-adjusted close) • USD • Market data may be delayed")
left,right=st.columns(2)
with left:
    st.plotly_chart(line_chart(common,"1. Adjusted Closing Price by Company","Adjusted price (USD/share)"),use_container_width=True)
with right:
    st.plotly_chart(line_chart(normalized,"2. Normalized Price Performance","Index (first common day = 100)"),use_container_width=True)
with left:
    v=volume.loc[common.index].rename(columns=labels).reset_index().melt(id_vars=volume.index.name or 'Date',var_name='Company',value_name='Shares')
    fig=px.bar(v,x=v.columns[0],y='Shares',color='Company',barmode='group',title='3. Daily Trading Volume',labels={v.columns[0]:'Trading date','Shares':'Shares traded'})
    fig.update_layout(margin=dict(t=55,b=35,l=30,r=15))
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.plotly_chart(line_chart(returns,"4. Daily Returns","Daily return (%)"),use_container_width=True)
with left:
    st.plotly_chart(line_chart(volatility,"5. Rolling 20-Day Annualized Volatility","Annualized volatility (%)"),use_container_width=True)
with right:
    corr=returns.corr().rename(index=labels,columns=labels)
    fig=px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='6. Daily Return Correlation',aspect='auto',labels={'color':'Correlation'})
    fig.update_layout(margin=dict(t=55,b=35,l=30,r=15))
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Key differences revealed by the visualizations")
performance=(normalized.iloc[-1]-100).sort_values(ascending=False)
most=performance.index[0]
least=performance.index[-1]
vol=volatility.iloc[-1].dropna()
text=(f"Over the selected period, {labels[most]} had the highest normalized price change ({performance[most]:+.1f}%), "
      f"while {labels[least]} had the lowest ({performance[least]:+.1f}%). "
      "The adjusted-price chart shows share prices in USD, while the normalized chart provides a fairer comparison of relative performance. ")
if not vol.empty:
    text+=f"{labels[vol.idxmax()]} has the highest latest 20-day annualized volatility ({vol.max():.1f}%), indicating greater recent return variability. "
text+="The volume chart compares trading activity, and the correlation heatmap shows how similarly daily returns moved. These are historical observations, not predictions."
st.write(text)
st.caption("Methodology: Adjusted closing prices account for corporate actions; normalized values use the first common trading day = 100. Daily return = percent change in adjusted close. Volatility = 20-day standard deviation of daily percentage returns × √252. Correlations use overlapping daily returns. Trading volume is raw shares and is not adjusted for company size.")
st.markdown("---")
st.markdown("**Nicole Bencosme-Gil** · **Developed for CPS*5745**")
