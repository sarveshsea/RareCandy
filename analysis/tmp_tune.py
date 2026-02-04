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

print('loading data...')
ex=ccxt.coinbase({'enableRateLimit':True})
all_syms=sorted(set(symbols+bos_symbols))
data={s: base.clean_df(base.fetch_ohlcv(ex,s,'15m',start_ms,end_ms)) for s in all_syms}
bos_data={s:data[s] for s in bos_symbols}
print('loaded', {k:len(v) for k,v in data.items() if k in symbols})

expert_specs=[
    ('Jurik_Breakouts','fast'),
    ('BackQuant_VolSkew','base'),
    ('DAFE_Bands_SmartComposite','base'),
    ('GK_Trend_Ribbon','base'),
]

cache={}
for sym in symbols:
    df=data[sym]
    cols=[]
    for name,var in expert_specs:
        sig=wf.STRATEGIES[name](df,var)
        pos=aoe.build_position_state(df,sig)
        cols.append(pos.rename(f'{name}:{var}'))
    expos=pd.concat(cols,axis=1).fillna(0.0)
    long_ok,short_ok,_=wf.bos_filter(df.index,bos_data)
    cache[sym]=(df,expos,long_ok,short_ok)
print('experts cached')

params=[]
for reward_scale in [80,120,160,220]:
    for decay in [0.97,0.98,0.99]:
        for min_conf in [0.12,0.18,0.25]:
            for dd_kill in [0.12,0.16,0.2]:
                params.append((reward_scale,decay,min_conf,dd_kill))

rows=[]
for reward_scale,decay,min_conf,dd_kill in params:
    rets=[]; dds=[]; sh=[]
    for sym in symbols:
        df,ep,long_ok,short_ok=cache[sym]
        m,_,_=aoe.run_adaptive_ensemble(
            df,ep,long_ok,short_ok,
            reward_scale=reward_scale,
            score_decay=decay,
            top_k=3,
            min_confidence=min_conf,
            dd_kill_threshold=dd_kill,
            target_bar_vol=0.003,
            vol_window=96,
            dd_cooldown_bars=96,
            max_leverage=1.0,
        )
        rets.append(m.total_return_pct)
        dds.append(m.max_drawdown_pct)
        sh.append(m.sharpe)
    avg_ret=float(np.mean(rets))
    avg_dd=float(np.mean(dds))
    avg_sh=float(np.mean(sh))
    score=avg_ret + avg_sh*8 - avg_dd*0.5
    rows.append((score,avg_ret,avg_dd,avg_sh,reward_scale,decay,min_conf,dd_kill))

rows=sorted(rows,reverse=True)
print('top20:')
for r in rows[:20]:
    print(r)
