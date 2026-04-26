import pandas as pd
from datetime import datetime

def load_raw_feed(path: str) -> pd.DataFrame:
    """
    - Read a Databento `.csv` 
    - Keep only these columns: `ts_recv` (int64 nanoseconds), `action` (ADD/CANCEL/TRADE), `side` (BID/ASK), `price` (int64 in price ticks), `size` (int64), and `bid_sz_00..03` / `ask_sz_00..03` (top-4 queue snapshots).
    - Discard the first and last 30 minutes of each trading day to exclude open/close artefacts (keep 10:00–15:30 ET).
    - Return a dataframe sorted ascending by `ts_recv`.
    """
    df = pd.read_csv(path)
    df["price"] *= 1e-9
    event_log = []
    """
    event_log columns
        ts: time
        side: bid(B)/ask(A)
        price
        size
        event_type
    """
    last_seen = {}
    for temp_order in df.itertuples(index=False):
        # drop out the first and last 10 minute data
        ### The time drop will be processed later since the open time is unknown
        
        
        # Trade, Fill, or None: no change
        if temp_order.action in ("T", "F", "N"):
            continue

        # Clear book: remove all resting orders
        if temp_order.action == "R":
            current_event = [temp_order.ts_recv,
                            temp_order.side,
                            temp_order.price,
                            temp_order.size,
                            "Clear"
                            ]

        # Add: insert a new order
        elif temp_order.action == "A":
            # For top-of-book publishers, remove previous order associated with this side
            # Modification of best ask/bid
            if temp_order.flags==64:
                current_event = [temp_order.ts_recv,
                                temp_order.side,
                                temp_order.price,
                                temp_order.size,
                                "Modify best"
                            ]
                # UNDEF_PRICE indicates the price level was removed. There's no new level to add
                if pd.isna(temp_order["price"]):
                    continue
            else:
                current_event = [temp_order.ts_recv,
                            temp_order.side,
                            temp_order.price,
                            temp_order.size,
                            "Add"
                            ]

        # Cancel: partially or fully cancel some size from a resting order
        elif temp_order.action == "C":
            if temp_order.flags == 128:
                sequence = temp_order.sequence
                
                sequence_to_action = df.set_index("sequence").action.to_dict()
                if sequence_to_action.get(sequence) in {"T", "F"}:
                    event_type = "Trade"
                else:
                    event_type = "Cancel"
                    
            current_event = [temp_order.ts_recv,
                            temp_order.side,
                            temp_order.price,
                            temp_order.size,
                            event_type
                            ]

        # Modify: change the price and/or size of a resting order
        elif temp_order.action == "M":
            
            previous = last_seen.get(temp_order.order_id)
            if previous is None:
                # Never seen this order before — can't compute delta
                # This can happen if the Add arrived before your data window
                continue
            else:
                delta_price = previous.price - temp_order.price
                delta_size  = previous.size  - temp_order.size
            current_event = [temp_order.ts_recv,
                            temp_order.side,
                            delta_price,
                            delta_size,
                            "Modify"
                            ]
        

        event_log.append(current_event)

    return pd.DataFrame(event_log, columns=["ts", "side", "price", "size", "event_type"])



    
def orderbook_converter(df: pd.DataFrame):
    """
    - Group the event_log by 'ts_recv'
    - Calculate the best ask/bid
    - aggregate creates
    """
    
    best_ask_series = {}
    best_bid_series = {}

    current_ask_prices = {}   # price -> size
    current_bid_prices = {}   # price -> size
    ask_book_series = {}   # ts -> full dict snapshot
    bid_book_series = {}   # ts -> full dict snapshot
    
    create_ask = []
    create_bid = []

    # Create aggregation state
    pending_create = None   # {"side": ..., "price": ..., "size": ..., "ts": ...}
    prev_ts = None   # track previous timestamp

    def _flush_create():
        """Apply the accumulated create event to the book."""
        if pending_create is None:
            return
        if pending_create["side"] == "A":
            current_ask_prices[pending_create["price"]] = pending_create["size"]
            create_ask.append([ts, pending_create["price"], pending_create["size"]])
        else:
            current_bid_prices[pending_create["price"]] = pending_create["size"]
            create_bid.append([ts, pending_create["price"], pending_create["size"]])
            
    def _snapshot(ts):
        """Record full book state and best prices at this timestamp."""
        best_ask_series[ts] = min(current_ask_prices) if current_ask_prices else None
        best_bid_series[ts] = max(current_bid_prices) if current_bid_prices else None
        # Store a copy — not a reference — so future updates don't mutate it
        ask_book_series[ts] = dict(current_ask_prices)
        bid_book_series[ts] = dict(current_bid_prices)


    for row in df.itertuples(index=False):
        ts = row.ts

        # When timestamp changes, snapshot the completed previous timestamp
        # But only if no Create burst is pending (book not fully updated yet)
        if prev_ts is not None and ts != prev_ts and pending_create is None:
            _snapshot(prev_ts)
            
        
        # Check if this row continues an ongoing Create burst
        # A burst continues only if same side, same price, and still an Add
        is_create_continuation = (
            pending_create is not None
            and row.event_type == "Add"
            and row.side == pending_create["side"]
            and row.price == pending_create["price"]
        )

        if is_create_continuation:
            # Accumulate into the pending create
            pending_create["size"] += row.size
            pending_create["ts"] = ts
            # Do not apply to book yet — keep accumulating

        else:
            # This row breaks the burst — flush whatever was pending
            _flush_create()
            pending_create = None

            # Now handle the current row normally
            if row.event_type == "Modify best":
                # F_TOB update — replace the entire side's best level
                if row.side == "A":
                    # Remove old best ask entry and write new one
                    if current_ask_prices:
                        old_best = min(current_ask_prices)
                        current_ask_prices.pop(old_best, None)
                    if row.price is not None:
                        current_ask_prices[row.price] = row.size
                else:
                    if current_bid_prices:
                        old_best = max(current_bid_prices)
                        current_bid_prices.pop(old_best, None)
                    if row.price is not None:
                        current_bid_prices[row.price] = row.size

            elif row.event_type == "Add":
                # Check if this is the start of a new Create burst
                # A Create is an Add that places a new best level
                # i.e. ask price below current best ask, or bid price above current best bid
                is_new_best_ask = (
                    row.side == "A"
                    and current_ask_prices
                    and row.price < min(current_ask_prices)
                )
                is_new_best_bid = (
                    row.side == "B"
                    and current_bid_prices
                    and row.price > max(current_bid_prices)
                )

                if is_new_best_ask or is_new_best_bid:
                    # Start accumulating a new Create burst
                    pending_create = {
                        "side":  row.side,
                        "price": row.price,
                        "size":  row.size,
                        "ts":    ts,
                    }
                    # Do not apply to book yet
                else:
                    # Regular add at an existing or non-best level
                    if row.side == "A":
                        current_ask_prices[row.price] = current_ask_prices.get(row.price, 0) + row.size
                    else:
                        current_bid_prices[row.price] = current_bid_prices.get(row.price, 0) + row.size

            elif (row.event_type == "Cancel") or (row.event_type == "Trade"):
                if row.side == "A":
                    current_ask_prices[row.price] -= row.size
                    # Only remove the level entirely if no shares remains
                    if current_ask_prices[row.price] <= 0:
                        current_ask_prices.pop(row.price)
                else:
                    current_bid_prices[row.price] -= row.size
                    if current_bid_prices[row.price] <= 0:
                        current_bid_prices.pop(row.price)

            elif row.event_type == "Clear":
                current_ask_prices.clear()
                current_bid_prices.clear()
            
        prev_ts = ts

        # Snapshot current best after every row
        # Skip snapshot during an ongoing Create burst — book not updated yet
        if pending_create is None:
            best_ask_series[ts] = min(current_ask_prices) if current_ask_prices else None
            best_bid_series[ts] = max(current_bid_prices) if current_bid_prices else None

    # Flush any pending create at end of data
    _flush_create()
    # Snapshot the final timestamp
    if prev_ts is not None:
        _snapshot(prev_ts)

    best_ask = pd.Series(best_ask_series)
    best_bid = pd.Series(best_bid_series)
    
    create_ask_df = pd.DataFrame(create_ask, columns=["ts", "price", "size"])
    create_bid_df = pd.DataFrame(create_bid, columns=["ts", "price", "size"])

    return best_ask, best_bid, ask_book_series, bid_book_series, create_ask_df, create_bid_df
    
def _aggregate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Group consecutive TRADE rows that share the same `ts_recv` value into one row with summed `size`.
    - Rationale: a single aggressive order hitting multiple resting orders generates one message per fill; these must be treated as one event for volume estimation.
    - Implementation: use `groupby` on a "burst ID" that increments whenever `ts_recv` changes or `action != TRADE`. Take the first row's metadata and sum `size`.
    """
    df[df["event_type"]=="Trade"].groupby("ts")

def _compute_spread_and_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    - After each event, recompute `spread_n` as `(best_ask_price - best_bid_price)` in ticks, clamped to `[1, 5]` (treat 5+ as one bin).
    - Compute `imbalance_raw = (bid_sz_00 - ask_sz_00) / (bid_sz_00 + ask_sz_00)`, range `[-1, 1]`.
    - Add columns `imb_bin` (int, 0–20 per the 21-bin scheme) and `spread_bin` (int, 0–4).
    - Binning rule: 21 bins of width 0.1; bins are right-closed for positive imbalance, left-closed for negative; the exact-zero bin is a point mass. See Section 2.3 of the paper.
    """
def compute_inter_event_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Add column `dt_ns` = `ts_recv.diff()` in nanoseconds (first row gets NaN, drop it).
    - Add `log10_dt` = `log10(dt_ns)` for GMM fitting downstream.
    - Flag rows where `dt_ns < delta_rtl` as `is_fast = True` (used for race model validation).
    """
def normalise_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Compute the Median Event Size (MES) separately for each queue level (q1, q2, q3, q4) and each event type (ADD, CANCEL, TRADE), symmetrised between BID and ASK.
    - Add column `vol_norm = size / MES[level, event_type]`, rounded to nearest integer, minimum 1.
    - Cap at 50 MES (set `vol_norm = min(vol_norm, 50)`).
    - Save MES table to `data/params/{ticker}_mes.npz` for use during simulation (needed when queues shift levels after a price move).
    """
def save_processed(df: pd.DataFrame, ticker: str):
    """
    - Write the cleaned, annotated dataframe to `data/processed/{ticker}_events.parquet`.
    - Schema: `[ts_recv, action, side, level, price, size, vol_norm, dt_ns, log10_dt, imb_bin, spread_bin, is_fast]`.
    """