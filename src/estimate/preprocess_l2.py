import pandas as pd
from datetime import datetime
import numpy as np


import pandas_market_calendars as mcal

def load_raw_feed(path: str) -> pd.DataFrame:
    """
    - Read a Databento `.csv` 
    - Keep only these columns: `ts_recv` (int64 nanoseconds), `action` (ADD/CANCEL/TRADE), `side` (BID/ASK), `price` (int64 in price ticks), `size` (int64), and `bid_sz_00..03` / `ask_sz_00..03` (top-4 queue snapshots).
    - Discard the first and last 30 minutes of each trading day to exclude open/close artefacts (keep 10:00–15:30 ET).
    - Return a dataframe sorted ascending by `ts_recv`.
    """
    df = pd.read_csv(path)
    df = filter_trading_hours(df, "ts_recv")
    
def orderbook_preprocess(df:pd.DataFrame) -> pd.DataFrame:
    """
    - Group the event_log by 'ts_recv'
    - Calculate the best ask/bid
    - find the cell
    - aggregate trades, creates
    """
    # Pre-group once — O(n) instead of O(n²)
    grouped = df.groupby("ts_recv", sort=False)

    book_states = {}
    # keys: bid_00, ask_00, bid_01, ask_01, ... bid_03, ask_03
    # data: [price, size]
    delta = []
    pre_ts = None
                               
    for ts, temp_df in grouped:
        last = temp_df.iloc[-1]
        # recording book states
        if (last["flags"] in [128,130]) and (last["action"]!="T"):
            book_states[ts] = {
                "bid_03": [last["bid_px_03"], last["bid_sz_03"]],
                "bid_02": [last["bid_px_02"], last["bid_sz_02"]],
                "bid_01": [last["bid_px_01"], last["bid_sz_01"]],
                "bid_00": [last["bid_px_00"], last["bid_sz_00"]],
                "ask_00": [last["ask_px_00"], last["ask_sz_00"]],
                "ask_01": [last["ask_px_01"], last["ask_sz_01"]],
                "ask_02": [last["ask_px_02"], last["ask_sz_02"]],
                "ask_03": [last["ask_px_03"], last["ask_sz_03"]]
            }
            
        else:
            continue
        
        # decideing delta
        if "N" in temp_df["side"].values:
            row = temp_df.iloc[-1]       
            if row.flags == 0:
                continue
            else:
                # The profile is right, compare the previous ts data to decide the side, action
                if row["action"] != "F":
                    if pre_ts is None or pre_ts not in book_states:
                        continue
                    prev_state = book_states[pre_ts]
                    prices = []
                    sizes  = []
                    current_prices = []
                    current_sizes = []
                    for value in prev_state.values():
                        prices.append(value[0])
                        sizes.append(value[1])

                    for value in book_states[ts].values():
                        current_prices.append(value[0])
                        current_sizes.append(value[1])

                    
                    # queue level index mapping:
                    # i=0 → q-4,  i=1 → q-3,  i=2 → q-2,  i=3 → q-1 (best bid)
                    # i=4 → q+1 (best ask),  i=5 → q+2,  i=6 → q+3,  i=7 → q+4

                    if prices == current_prices:
                        # No price level appeared or disappeared — only sizes changed
                        for i in range(len(current_prices)):
                            change = current_sizes[i] - sizes[i]
                            if change != 0:
                                delta.append([ts, queue_level(i), change, "C" if change < 0 else "A"])

                    else:
                        # At least one price level changed — determine what happened
                        # Split into bid and ask sides
                        # Bid: indices 0-3, ordered worst to best (index 3 = best bid)
                        # Ask: indices 4-7, ordered best to worst (index 4 = best ask)
                        # Scan bid from best inward (index 3 → 0)
                        # Scan ask from best inward (index 4 → 7)

                        prev_ask    = prices[4:]         # [best_ask, 2nd, 3rd, 4th]
                        current_ask = current_prices[4:]
                        prev_ask_sz    = sizes[4:]
                        current_ask_sz = current_sizes[4:]

                        prev_bid    = prices[:4][::-1]         # reverse to [best_bid, 2nd, 3rd, 4th]
                        current_bid = current_prices[:4][::-1]
                        prev_bid_sz    = sizes[:4][::-1]
                        current_bid_sz = current_sizes[:4][::-1]

                        

                        delta += scan_side(ts, prev_ask, current_ask, prev_ask_sz, current_ask_sz, "ask")
                        delta += scan_side(ts, prev_bid, current_bid, prev_bid_sz, current_bid_sz, "bid")

                                    
                        
        else:
            # for normal trade there is an extra cancel event
            is_normal_trade = temp_df[(temp_df["flags"] == 130) | (temp_df["flags"] == 128)].shape[0] > 1 and temp_df.iloc[-2]["action"] == "T" and temp_df.iloc[-1]["action"] == "C"
            if is_normal_trade:
                temp_df = temp_df.iloc[:-1] 
            for row in temp_df.itertuples(index=False):
                delta.append([ts, row.depth+1 if row.side=="A" else -(row.depth+1), row.size, row.action])
        pre_ts = ts
        
    return book_states, pd.DataFrame(delta, columns=["ts", "level", "size", "action"])

def queue_level(i):
        return i - 4 if i < 4 else i - 3   # -4,-3,-2,-1,+1,+2,+3,+4

def scan_side(ts, prev_px, curr_px, prev_sz, curr_sz, side):
    """
    Scan from best price outward.
    First difference found determines the event type for that level and beyond.
    Returns list of [queue_level, size_change, action]
    """
    result = []
    n = len(prev_px)

    # Find the first position where price differs
    first_diff = None
    for i in range(n):
        if prev_px[i] != curr_px[i]:
            first_diff = i
            break

    # No price change on this side — only size changes
    if first_diff is None:
        for i in range(n):
            change = curr_sz[i] - prev_sz[i]
            if change != 0:
                level = (i + 1) if side == "ask" else -(i + 1)
                result.append([ts, level, change, "C" if change < 0 else "A"])
        return result

    # --- Determine what happened at first_diff ---
    prev_price    = prev_px[first_diff]
    current_price = curr_px[first_diff]

    if side == "ask":
        price_improved = current_price < prev_price
    else:
        price_improved = current_price > prev_price

    if price_improved:
        # New best price appeared inside the spread — CREATE event
        # The new price is not in the previous book at all
        if current_price not in set(prev_px):
            level = (first_diff + 1) if side == "ask" else -(first_diff + 1)
            result.append([ts, level, curr_sz[first_diff], "CREATE_ASK" if side == "ask" else "CREATE_BID"])

            # All levels from first_diff+1 onward shifted one step deeper
            # previous level i is now at position i+1
            for i in range(first_diff + 1, n):
                prev_idx = i - 1   # this level's previous position
                if prev_idx < n:
                    change = curr_sz[i] - prev_sz[prev_idx]
                    if change != 0:
                        level = (i + 1) if side == "ask" else -(i + 1)
                        result.append([ts, level, change, "C" if change < 0 else "A"])
                else:
                    # Pushed beyond tracking depth — record as new addition
                    level = (i + 1) if side == "ask" else -(i + 1)
                    result.append([ts, level, curr_sz[i], "A"])

        else:
            # Price improved but exists in prev book — level shift (price move)
            # Best level was consumed, everything shifted inward
            level = (first_diff + 1) if side == "ask" else -(first_diff + 1)
            result.append([ts, level, -prev_sz[first_diff], "C"])   # old best depleted

            # Remaining levels shifted one step toward best
            for i in range(first_diff + 1, n):
                prev_idx = i + 1   # this level was previously one step deeper
                if prev_idx < n:
                    change = curr_sz[i] - prev_sz[prev_idx]
                    if change != 0:
                        level = (i + 1) if side == "ask" else -(i + 1)
                        result.append([ts, level, change, "C" if change < 0 else "A"])
                else:
                    # Newly revealed deepest level
                    level = (i + 1) if side == "ask" else -(i + 1)
                    result.append([ts, level, curr_sz[i], "A"])

    else:
        # Price worsened at first_diff — a gap level appeared between existing levels
        # New price that did not exist before inserted between levels
        if current_price not in set(prev_px):
            level = (first_diff + 1) if side == "ask" else -(first_diff + 1)
            result.append([ts, level, curr_sz[first_diff], "A"])

            # Levels deeper than insertion point shifted one step outward
            for i in range(first_diff + 1, n):
                prev_idx = i - 1
                if prev_idx < n and prev_px[prev_idx] == curr_px[i]:
                    # This level just shifted deeper — record size change
                    change = curr_sz[i] - prev_sz[prev_idx]
                    if change != 0:
                        level = (i + 1) if side == "ask" else -(i + 1)
                        result.append([ts, level, change, "C" if change < 0 else "A"])
        else:
            # Price worsened and exists in prev — old level was fully removed
            # Record the removal and check remaining levels
            level = (first_diff + 1) if side == "ask" else -(first_diff + 1)
            result.append([ts, level, -prev_sz[first_diff], "C"])

            # Continue scanning from first_diff+1 for further changes
            for i in range(first_diff + 1, n):
                if prev_px[i] != curr_px[i]:
                    change = curr_sz[i] - prev_sz[i]
                    if change != 0:
                        level = (i + 1) if side == "ask" else -(i + 1)
                        result.append([ts, level, change, "C" if change < 0 else "A"])
                else:
                    change = curr_sz[i] - prev_sz[i]
                    if change != 0:
                        level = (i + 1) if side == "ask" else -(i + 1)
                        result.append([ts, level, change, "C" if change < 0 else "A"])

    return result

def convert_book_states(book_states: dict, tick_size: float = 1e9) -> pd.DataFrame:
    """
    Convert book_states dict into a structured DataFrame.
    
    book_states: {ts -> {"bid_00": [price, size], ..., "ask_00": [price, size], ...}}
    tick_size:   one tick in Databento fixed-point units (1e9 = 1 cent)
    
    Output columns:
        ts, spread, imbalance, q-4, q-3, q-2, q-1, q+1, q+2, q+3, q+4
    """
    rows = []

    # --- Step 1: Compute MES per queue slot for normalisation ---
    # Collect all sizes per slot across all timestamps
    slot_sizes = {f"bid_0{i}": [] for i in range(4)}
    slot_sizes.update({f"ask_0{i}": [] for i in range(4)})

    for state in book_states.values():
        for key, (price, size) in state.items():
            if size > 0:
                slot_sizes[key].append(size)

    # MES = median size per slot, symmetrised between bid and ask at same depth
    mes = {}
    for i in range(4):
        bid_sizes = slot_sizes[f"bid_0{i}"]
        ask_sizes = slot_sizes[f"ask_0{i}"]
        combined  = bid_sizes + ask_sizes
        mes[i] = float(np.median(combined)) if combined else 1.0  # fallback to 1

    # --- Step 2: Process each timestamp ---
    for ts, state in book_states.items():

        # Unpack prices and sizes
        bid_px = [state[f"bid_0{i}"][0] for i in range(4)]   # [best, 2nd, 3rd, 4th]
        bid_sz = [state[f"bid_0{i}"][1] for i in range(4)]
        ask_px = [state[f"ask_0{i}"][0] for i in range(4)]   # [best, 2nd, 3rd, 4th]
        ask_sz = [state[f"ask_0{i}"][1] for i in range(4)]

        best_bid = bid_px[0]
        best_ask = ask_px[0]

        # --- Spread ---
        # In Databento fixed-point units; divide by tick_size to get ticks
        if best_bid is None or best_ask is None or pd.isna(best_bid) or pd.isna(best_ask):
            continue   # skip incomplete snapshots
        spread_ticks = int(round((best_ask - best_bid) / tick_size))

        # --- Imbalance at best level ---
        q_bid1 = bid_sz[0]
        q_ask1 = ask_sz[0]
        total  = q_bid1 + q_ask1
        imbalance = (q_bid1 - q_ask1) / total if total > 0 else 0.0

        # --- Step 3: Redefine queues by tick distance ---
        # For bid side: q-k sits (k-1) ticks BELOW best bid
        #   q-1 = best_bid price
        #   q-2 = best_bid - 1 tick
        #   q-3 = best_bid - 2 ticks
        #   q-4 = best_bid - 3 ticks
        # For ask side: q+k sits (k-1) ticks ABOVE best ask
        #   q+1 = best_ask price
        #   q+2 = best_ask + 1 tick
        #   q+3 = best_ask + 2 ticks
        #   q+4 = best_ask + 3 ticks

        # Build price -> size lookup from the snapshot
        bid_lookup = {px: sz for px, sz in zip(bid_px, bid_sz) if px is not None and not pd.isna(px)}
        ask_lookup = {px: sz for px, sz in zip(ask_px, ask_sz) if px is not None and not pd.isna(px)}

        # Assign sizes by tick distance — zero if no resting queue at that tick level
        def bid_queue(k):
            """k=1 → best bid, k=2 → one tick behind, etc."""
            target_price = best_bid - (k - 1) * tick_size
            raw_size = bid_lookup.get(target_price, 0)
            return raw_size / mes[k - 1] if mes[k - 1] > 0 else 0.0

        def ask_queue(k):
            """k=1 → best ask, k=2 → one tick behind, etc."""
            target_price = best_ask + (k - 1) * tick_size
            raw_size = ask_lookup.get(target_price, 0)
            return raw_size / mes[k - 1] if mes[k - 1] > 0 else 0.0

        # --- Step 4 & 5: Build row with normalised volumes ---
        rows.append({
            "ts":        ts,
            "spread":    spread_ticks,
            "imbalance": round(imbalance, 4),
            "best_size": (round(bid_queue(1), 4)+round(ask_queue(1), 4))*1e-9,
            "q-4":       round(bid_queue(4), 4),
            "q-3":       round(bid_queue(3), 4),
            "q-2":       round(bid_queue(2), 4),
            "q-1":       round(bid_queue(1), 4),
            "q+1":       round(ask_queue(1), 4),
            "q+2":       round(ask_queue(2), 4),
            "q+3":       round(ask_queue(3), 4),
            "q+4":       round(ask_queue(4), 4),
        })

    result = pd.DataFrame(rows, columns=[
        "ts", "spread", "imbalance",
        "q-4", "q-3", "q-2", "q-1",
        "q+1", "q+2", "q+3", "q+4"
    ])

    return result

    
def aggregate_trades(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Group consecutive TRADE rows that share the same `ts_recv` value into one row with summed `size`.
    - Rationale: a single aggressive order hitting multiple resting orders generates one message per fill; these must be treated as one event for volume estimation.
    - Implementation: use `groupby` on a "burst ID" that increments whenever `ts_recv` changes or `action != TRADE`. Take the first row's metadata and sum `size`.
    """
    trades = delta_df[delta_df["action"] == "T"].copy()
    
    # Burst ID: increments when ts changes between consecutive trade rows
    trades["burst_id"] = (trades["ts"] != trades["ts"].shift()).cumsum()
    
    aggregated = trades.groupby("burst_id", sort=False).agg(
        ts     = ("ts",     "first"),
        level  = ("level",  "first"),
        size   = ("size",   "sum"),     # key aggregation
        action = ("action", "first"),
    ).reset_index(drop=True)
    
    non_trades = delta_df[delta_df["action"] != "T"]
    return pd.concat([aggregated, non_trades]).sort_values("ts").reset_index(drop=True)
    


def aggregate_creates(delta_df: list) -> list:
    """
    Aggregate CREATE events with any Add events at the same ts and level.
    When a new price level is created, other participants immediately join it —
    those Add events at the same ts and level should be folded into the Create.
    
    Also aggregates consecutive Add events at the same ts and level that
    follow a Create, even if they appear non-consecutively among other events,
    as long as they share the same ts.
    """
    creates = delta_df[delta_df["action"].isin(["CREATE_ASK", "CREATE_BID"])].copy()
    others  = delta_df[~delta_df["action"].isin(["CREATE_ASK", "CREATE_BID"])].copy()

    # Burst ID: increments when ts, level, or action changes
    creates["burst_id"] = (
        (creates["ts"]     != creates["ts"].shift())     |
        (creates["level"]  != creates["level"].shift())  |
        (creates["action"] != creates["action"].shift())
    ).cumsum()

    aggregated = creates.groupby("burst_id", sort=False).agg(
        ts     = ("ts",     "first"),
        level  = ("level",  "first"),
        size   = ("size",   "sum"),     # accumulate all participants joining new level
        action = ("action", "first"),
    ).reset_index(drop=True)

    return pd.concat([aggregated, others]).sort_values("ts").reset_index(drop=True)

def save_processed(df: pd.DataFrame, ticker: str):
    """
    - Write the cleaned, annotated dataframe to `data/processed/{ticker}_events.parquet`.
    - Schema: `[ts_recv, action, side, level, price, size, vol_norm, dt_ns, log10_dt, imb_bin, spread_bin, is_fast]`.
    """
    


def filter_trading_hours(df: pd.DataFrame, ts_col: str = "ts_recv") -> pd.DataFrame:
    """
    filter the first and last 30 min of each trading day
    remaining: 10:00~15:00 each trading day
    """
    # Convert nanosecond UTC to ET
    ts_et = pd.to_datetime(df[ts_col], unit="ns", utc=True).dt.tz_convert("America/New_York")
    
    # Get the date range covered by the data
    start_date = ts_et.dt.date.min()
    end_date   = ts_et.dt.date.max()
    
    # Get the NYSE trading calendar for that range
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # schedule gives you market_open and market_close in UTC for each trading day
    # but we will use fixed 9:30-15:30 ET since regular session times don't change
    trading_dates = set(schedule.index.date)
    
    # Build mask: must be a trading day AND within session hours
    date_only = ts_et.dt.date
    time_only = ts_et.dt.time
    
    market_open  = pd.Timestamp("10:00:00").time()
    market_close = pd.Timestamp("15:30:00").time()
    
    is_trading_day   = date_only.apply(lambda d: d in trading_dates)
    is_trading_hours = (time_only >= market_open) & (time_only <= market_close)
    
    mask = is_trading_day & is_trading_hours
    
    return df[mask].reset_index(drop=True)