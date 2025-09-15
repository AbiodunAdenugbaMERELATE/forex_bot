import pandas as pd
import math
from datetime import datetime

JOURNAL_PATH = 'trade_log.csv'

def load_journal(path=JOURNAL_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Journal file {path} not found.")
        return None
    if df.empty:
        print("Journal empty.")
        return None
    return df

def summarize_trades(df):
    # We consider phase=='filled' entries as trade entries; need to infer exits from 'trailing_update' or future 'filled' of opposite? For now rely on journal for entries only.
    filled = df[df['phase']=='filled'].copy()
    if filled.empty:
        print("No filled trades.")
        return
    # Compute approximate R if sl & price present
    def calc_r(row):
        try:
            price = float(row.get('price'))
            sl = float(row.get('sl')) if not pd.isna(row.get('sl')) else None
            tp = float(row.get('tp')) if not pd.isna(row.get('tp')) else None
            if sl is None or price is None:
                return None
            risk_per_unit = abs(price - sl)
            # If pl is known in journal, approximate realized R else None
            pl = row.get('pl')
            if pd.isna(pl) or pl is None:
                return None
            # Without units, approximate R per unit using pl / risk_per_unit when risk_per_unit>0
            if risk_per_unit > 0:
                try:
                    pnl_val = float(pl)
                    r = pnl_val / risk_per_unit if risk_per_unit else None
                    return r
                except Exception:
                    return None
            return None
        except Exception:
            return None
    filled['approx_r'] = filled.apply(calc_r, axis=1)
    avg_r = filled['approx_r'].dropna().mean() if 'approx_r' in filled else 0
    win_mask = filled['pl'].astype(float) > 0
    wins = filled[win_mask]
    losses = filled[~win_mask]
    win_rate = (len(wins)/len(filled))*100 if len(filled) else 0
    gross_win = wins['pl'].astype(float).sum() if not wins.empty else 0
    gross_loss = abs(losses['pl'].astype(float).sum()) if not losses.empty else 0
    profit_factor = (gross_win / gross_loss) if gross_loss else math.inf
    expectancy = avg_r  # Approximated since we lack full exit R normalization
    # Regime stats
    regime_counts = filled['regime'].value_counts().to_dict() if 'regime' in filled else {}

    print("=== Journal Summary ===")
    print(f"Total Trades (filled): {len(filled)} | Wins: {len(wins)} | Losses: {len(losses)} | WinRate: {win_rate:.2f}%")
    print(f"Gross Win: {gross_win:.2f} | Gross Loss: -{gross_loss:.2f} | Profit Factor: {profit_factor:.2f}")
    print(f"Average Approx R: {expectancy:.3f}")
    if regime_counts:
        print("Regime distribution:")
        for k,v in regime_counts.items():
            print(f"  {k}: {v}")

if __name__ == '__main__':
    journal_df = load_journal()
    if journal_df is not None:
        summarize_trades(journal_df)
