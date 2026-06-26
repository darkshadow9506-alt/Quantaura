from quantaura import journal
from quantaura.models import AssetClass, Side, Signal
from quantaura.storage import Store


def _sig(symbol="AAPL", side=Side.LONG, strategy="trend_breakout"):
    return Signal(symbol=symbol, asset_class=AssetClass.STOCK, strategy=strategy,
                  side=side, entry=100, stop=98, target=104, risk_per_unit=2,
                  reward_per_unit=4, rr_ratio=2.0, confidence=0.6)


# ---------------- storage ----------------
def test_record_and_dedup(tmp_path):
    st = Store(tmp_path / "s.db")
    assert st.record_signal(_sig(), cooldown_days=3) is True
    # same symbol+strategy+side while open & within cooldown -> deduped
    assert st.record_signal(_sig(), cooldown_days=3) is False
    # different side is a different signal
    assert st.record_signal(_sig(side=Side.SHORT), cooldown_days=3) is True
    assert len(st.open_signals()) == 2
    st.close()


def test_cooldown_zero_allows_repeat(tmp_path):
    st = Store(tmp_path / "s.db")
    assert st.record_signal(_sig(), cooldown_days=0) is True
    # cooldown 0 -> no longer "recent", repeat allowed
    assert st.record_signal(_sig(), cooldown_days=0) is True
    st.close()


def test_close_and_performance(tmp_path):
    st = Store(tmp_path / "s.db")
    st.record_signal(_sig("AAA"), 3)
    st.record_signal(_sig("BBB"), 3)
    st.record_signal(_sig("CCC"), 3)
    rows = st.open_signals()
    st.close_signal(rows[0]["id"], "tp", 2.0)
    st.close_signal(rows[1]["id"], "sl", -1.0)
    perf = st.performance()
    assert perf["total"] == 3 and perf["open"] == 1 and perf["closed"] == 2
    assert perf["tp"] == 1 and perf["sl"] == 1
    assert perf["win_rate"] == 0.5
    assert abs(perf["total_R"] - 1.0) < 1e-9
    st.close()


def test_subscribers(tmp_path):
    st = Store(tmp_path / "s.db")
    assert st.add_subscriber(111) is True
    st.add_subscriber(111)            # idempotent
    st.add_subscriber(222)
    assert st.subscribers() == [111, 222]
    assert st.remove_subscriber(111) is True
    assert st.remove_subscriber(999) is False
    assert st.subscribers() == [222]
    st.close()


# ---------------- journal resolvers ----------------
def test_resolve_bar_long():
    assert journal.resolve_bar("LONG", 100, 98, 104, high=105, low=99) == ("tp", 2.0)
    assert journal.resolve_bar("LONG", 100, 98, 104, high=101, low=97) == ("sl", -1.0)
    # tie -> stop wins (pessimistic)
    assert journal.resolve_bar("LONG", 100, 98, 104, high=105, low=97) == ("sl", -1.0)
    assert journal.resolve_bar("LONG", 100, 98, 104, high=101, low=99) == (None, None)


def test_resolve_bar_short():
    assert journal.resolve_bar("SHORT", 100, 102, 96, high=101, low=95) == ("tp", 2.0)
    assert journal.resolve_bar("SHORT", 100, 102, 96, high=103, low=99) == ("sl", -1.0)


def test_resolve_path_first_event_wins():
    bars = [(101, 99), (101, 99), (105, 100)]   # tp on 3rd bar
    assert journal.resolve_path("LONG", 100, 98, 104, bars) == ("tp", 2.0)
    bars2 = [(101, 99), (101, 97)]              # sl on 2nd bar
    assert journal.resolve_path("LONG", 100, 98, 104, bars2) == ("sl", -1.0)
    assert journal.resolve_path("LONG", 100, 98, 104, [(101, 99)]) == (None, None)


def test_mark_to_market():
    assert journal._mark_to_market_R("LONG", 100, 98, 102) == 1.0   # +2 / risk 2
    assert journal._mark_to_market_R("SHORT", 100, 102, 98) == 1.0


def test_format_performance_empty_and_full():
    assert "No signals recorded" in journal.format_performance(
        {"total": 0, "open": 0, "closed": 0, "tp": 0, "sl": 0,
         "win_rate": 0, "avg_R": 0, "total_R": 0})
    txt = journal.format_performance(
        {"total": 10, "open": 3, "closed": 7, "tp": 4, "sl": 3,
         "win_rate": 0.57, "avg_R": 0.3, "total_R": 2.1})
    assert "Win rate" in txt and "57%" in txt
