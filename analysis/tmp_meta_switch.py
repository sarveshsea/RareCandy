import datetime as dt
import numpy as np
import pandas as pd
import ccxt
import sys
sys.path.append('/app/analysis')
import backtest_pine_batch as base
import backtest_pine_walkforward as wf
import adaptive_online_ensemble as aoe

START=dt.datetime(2025,5,1,tzinfo=dt.timezone.utc)
END=dt.datetime(2026,2,3,tzinfo=dt.timezone.utc)
start_ms=int(START.timestamp()*1000)
end_ms=int(END.timestamp()*1000)

symbols=['BTC/USD','ETH/USD']
bos_symbols=['BTC/USD','ETH/USD','SOL/USD','LTC/USD','XRP/USD','ADA/USD','DOGE/USD','BCH/USD']

ex=ccxt.coinbase({'enableRateLimit':True})
all_syms=sorted(set(symbols+bos_symbols))
data={s: base.clean_df(base.fetch_ohlcv(ex,s,'15m',start_ms,end_ms)) for s in all_syms}
bos_data={s:data[s] for s in bos_symbols}

experts=[(name,var) for name in wf.STRATEGIES for var in ['fast','base','slow']]

def run(sym,use_bos=True,lookback=672,topk=1,min_score=0.0):
    df=data[sym]
    idx=df.index
    # build positions
    cols=[]
    for name,var in experts:
        sig=wf.STRATEGIES[name](df,var)
        pos=aoe.build_position_state(df,sig)
        cols.append(pos.rename(f'{name}:{var}'))
    ep=pd.concat(cols,axis=1).fillna(0.0)
    M=ep.to_numpy(float) # n x m
    n,m=M.shape
    close=df['close'].to_numpy(float)
    r=np.zeros(n)
    r[1:]=close[1:]/close[:-1]-1
    # expert instantaneous reward approx
    dpos=np.abs(np.diff(M,axis=0,prepend=np.zeros((1,m))))
    rew=M*np.expand_dims(r,1)-dpos*0.0006
    csum=np.cumsum(rew,axis=0)

    if use_bos:
        long_ok,short_ok,_=wf.bos_filter(idx,bos_data)
    else:
        long_ok=pd.Series(True,index=idx)
        short_ok=pd.Series(True,index=idx)

    eq=1.0; pos=0.0; peak=1.0
    curve=np.ones(n)
    trades=[]; entry=np.nan
    for i in range(1,n):
        eq*=1+pos*r[i]
        if i>lookback:
            win=csum[i]-csum[i-lookback]
        else:
            win=csum[i]
        rank=np.argsort(win)[::-1]
        chosen=rank[:topk]
        scores=win[chosen]
        scores=np.where(scores>min_score,scores,0)
        sig=np.mean(M[i,chosen]) if scores.sum()==0 else np.dot(scores, M[i,chosen])/scores.sum()
        desired=1 if sig>0.15 else (-1 if sig<-0.15 else 0)
        if desired>0 and not bool(long_ok.iloc[i]): desired=0
        if desired<0 and not bool(short_ok.iloc[i]): desired=0
        if abs(desired-pos)>1e-9:
            eq*=1-0.0006*abs(desired-pos)
            if abs(pos)<1e-9 and abs(desired)>1e-9:
                entry=eq
            elif abs(pos)>1e-9 and abs(desired)<1e-9 and not np.isnan(entry):
                trades.append(eq/entry-1); entry=np.nan
            elif pos*desired<0 and not np.isnan(entry):
                trades.append(eq/entry-1); entry=eq
        pos=float(desired)
        curve[i]=eq
        peak=max(peak,eq)
    if abs(pos)>0 and not np.isnan(entry):
        trades.append(eq/entry-1)
    curve=pd.Series(curve,index=idx)
    running=curve.cummax(); dd=((curve/running)-1).min()
    ret=(curve.iloc[-1]-1)*100
    win_rate=(sum(1 for t in trades if t>0)/len(trades)*100) if trades else 0
    pf=(sum(t for t in trades if t>0)/abs(sum(t for t in trades if t<0))) if any(t<0 for t in trades) else 99
    std=curve.pct_change().fillna(0).std(); sharpe=(curve.pct_change().fillna(0).mean()/std*(365*24*4)**0.5) if std>1e-12 else 0
    return {'ret':ret,'dd':abs(dd*100),'trades':len(trades),'win':win_rate,'pf':pf,'sh':sharpe}

for use_bos in [False,True]:
    print('\nuse_bos',use_bos)
    best=None
    for look in [288,576,864,1152]:
      for topk in [1,2,3,4,6]:
        for ms in [0.0,0.01,0.03]:
          mets=[]
          for sym in symbols:
            m=run(sym,use_bos=use_bos,lookback=look,topk=topk,min_score=ms)
            mets.append(m)
          avg_ret=np.mean([x['ret'] for x in mets]); avg_dd=np.mean([x['dd'] for x in mets]); avg_sh=np.mean([x['sh'] for x in mets])
          score=avg_ret+avg_sh*8-avg_dd*0.5
          row=(score,avg_ret,avg_dd,avg_sh,look,topk,ms,mets)
          if best is None or score>best[0]: best=row
    print('best',best[:7])
    print('symbol details',best[7])
