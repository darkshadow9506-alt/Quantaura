from quantaura import manage
from quantaura.models import Side

CFG = {"breakeven_R": 1.0, "trail_atr_mult": 3.0, "near_target_atr": 0.5,
       "danger_on_ma_break": True, "danger_on_macd_flip": True}


def test_long_in_profit_moves_to_breakeven():
    # long entry 100, stop 90 (risk 10); price at 112 -> +1.2R
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=120, current=112,
                      atr=2.0, ma_trend=95, macd_hist=0.5, hi_since=113, lo_since=99, cfg=CFG)
    assert r.R_now == 1.2
    assert r.recommended_sl >= 100              # at least breakeven
    assert r.at_breakeven or r.trailed
    assert r.actionable
    assert not r.danger


def test_long_trails_above_breakeven():
    # big run: hi_since 130, atr 2 -> trail = 130-6 = 124 > entry -> trailed
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=140, current=128,
                      atr=2.0, ma_trend=110, macd_hist=1.0, hi_since=130, lo_since=99, cfg=CFG)
    assert r.trailed
    assert abs(r.recommended_sl - 124.0) < 1e-9


def test_long_danger_on_ma_break():
    # price back below the trend MA -> danger
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=120, current=96,
                      atr=2.0, ma_trend=99, macd_hist=-0.2, hi_since=108, lo_since=95, cfg=CFG)
    assert r.danger
    assert r.actionable


def test_holding_not_actionable():
    # only +0.3R, above MA, macd positive, far from target -> just hold
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=140, current=103,
                      atr=2.0, ma_trend=98, macd_hist=0.4, hi_since=104, lo_since=99, cfg=CFG)
    assert r.R_now == 0.3
    assert not r.danger and not r.at_breakeven and not r.trailed
    assert not r.actionable


def test_short_mirror():
    # short entry 100, stop 110 (risk 10); price 88 -> +1.2R
    r = manage.review(side=Side.SHORT, entry=100, stop=110, target=80, current=88,
                      atr=2.0, ma_trend=105, macd_hist=-0.5, hi_since=101, lo_since=87, cfg=CFG)
    assert r.R_now == 1.2
    assert r.recommended_sl <= 100              # breakeven or trailed down
    # trail = lo_since + 3*atr = 87+6 = 93 < entry -> trailed to 93
    assert abs(r.recommended_sl - 93.0) < 1e-9
    assert not r.danger


def test_short_danger_on_macd_flip():
    r = manage.review(side=Side.SHORT, entry=100, stop=110, target=80, current=98,
                      atr=2.0, ma_trend=102, macd_hist=0.3, hi_since=101, lo_since=97, cfg=CFG)
    assert r.danger        # macd flipped up against the short


def test_near_target_flag():
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=120, current=119.5,
                      atr=2.0, ma_trend=110, macd_hist=1.0, hi_since=120, lo_since=99, cfg=CFG)
    assert r.near_target


def test_live_tp_banks_before_level_ahead():
    # long in profit; a resistance at 118 sits before the 130 target ->
    # recommend taking profit just before it
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=130, current=112,
                      atr=2.0, ma_trend=105, macd_hist=0.5, hi_since=113, lo_since=99,
                      cfg=CFG, next_level=118.0)
    assert abs(r.recommended_tp - (118.0 - 0.25 * 2.0)) < 1e-9   # 117.5
    assert "resistance" in r.tp_reason


def test_live_tp_rides_to_target_when_clear():
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=130, current=112,
                      atr=2.0, ma_trend=105, macd_hist=0.5, hi_since=113, lo_since=99,
                      cfg=CFG, next_level=None)
    assert r.recommended_tp == 130.0


def test_live_tp_take_now_on_danger():
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=130, current=104,
                      atr=2.0, ma_trend=106, macd_hist=-0.3, hi_since=108, lo_since=99,
                      cfg=CFG, next_level=120.0)
    assert r.danger and r.recommended_tp == 104   # take profit / close at current


def test_level_beyond_target_ignored():
    # resistance at 140 is beyond the 130 target -> ride to target
    r = manage.review(side=Side.LONG, entry=100, stop=90, target=130, current=112,
                      atr=2.0, ma_trend=105, macd_hist=0.5, hi_since=113, lo_since=99,
                      cfg=CFG, next_level=140.0)
    assert r.recommended_tp == 130.0


def test_past_target_take_profit_now():
    # SHORT already trading below its target -> recommend banking now, not the
    # (now worse) target price
    r = manage.review(side=Side.SHORT, entry=220, stop=231, target=198, current=168,
                      atr=8, ma_trend=210, macd_hist=-1, hi_since=221, lo_since=167, cfg=CFG)
    assert r.recommended_tp == 168 and r.near_target
    assert "reached" in r.tp_reason
    # LONG already above target
    r2 = manage.review(side=Side.LONG, entry=100, stop=90, target=120, current=125,
                       atr=2, ma_trend=110, macd_hist=1, hi_since=126, lo_since=99, cfg=CFG)
    assert r2.recommended_tp == 125


def test_zero_risk_safe():
    r = manage.review(side=Side.LONG, entry=100, stop=100, target=120, current=110,
                      atr=2.0, ma_trend=99, macd_hist=1.0, hi_since=111, lo_since=99, cfg=CFG)
    assert r.recommended_sl == 100      # degenerate risk -> no crash
