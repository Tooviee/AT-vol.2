"""
Diagnostic script to analyze why signals aren't being generated.
Shows current indicator values and how close each symbol is to triggering.
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding for emoji support
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from modules.config_loader import load_config
from modules.kis_api_manager import KISAPIManager
from modules.strategy import USAStrategy
import pandas as pd

def diagnose():
    print("=" * 70)
    print("SIGNAL DIAGNOSIS - Why aren't trades executing?")
    print("=" * 70)
    
    # Load config
    config = load_config()
    
    # Initialize components
    kis_api = KISAPIManager(config.model_dump())
    strategy = USAStrategy(config.strategy.model_dump())
    
    for symbol in config.symbols:
        print(f"\n{'='*70}")
        print(f"📊 {symbol}")
        print("=" * 70)
        
        # Get data
        hist_data = kis_api.get_historical_data(symbol, period='1y', interval='1d')
        
        if hist_data is None or hist_data.empty:
            print(f"   ❌ No data available")
            continue
        
        # Calculate indicators
        df = strategy.calculate_indicators(hist_data)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        
        print(f"\n📈 INDICATOR VALUES (latest bar):")
        print(f"   Close Price:    ${latest['Close']:.2f}")
        print(f"   SMA Short ({strategy.sma_short}): ${latest['SMA_short']:.2f}")
        print(f"   SMA Long ({strategy.sma_long}):  ${latest['SMA_long']:.2f}")
        print(f"   MACD:           {latest['MACD']:.4f}")
        print(f"   MACD Signal:    {latest['MACD_signal']:.4f}")
        print(f"   MACD Histogram: {latest['MACD_hist']:.4f}")
        print(f"   RSI:            {latest['RSI']:.1f}")
        print(f"   ATR:            ${latest['ATR']:.2f}")
        print(f"   Volatility:     {latest['Volatility']*100:.2f}%")
        
        # Check BUY conditions
        print(f"\n🟢 BUY CONDITIONS (need 2+ for signal):")
        buy_count = 0
        
        # Condition 1: SMA golden cross
        sma_cross_change = latest['SMA_cross_change']
        if sma_cross_change == 2:
            print(f"   ✅ SMA Golden Cross (just happened!)")
            buy_count += 1
        else:
            sma_status = "above" if latest['SMA_short'] > latest['SMA_long'] else "below"
            print(f"   ❌ SMA Golden Cross (short {sma_status} long, no crossover today)")
        
        # Condition 2: MACD bullish crossover
        macd_cross_change = latest['MACD_cross_change']
        if macd_cross_change == 2:
            print(f"   ✅ MACD Bullish Crossover (just happened!)")
            buy_count += 1
        else:
            macd_status = "above" if latest['MACD'] > latest['MACD_signal'] else "below"
            print(f"   ❌ MACD Bullish Crossover (MACD {macd_status} signal, no crossover today)")
        
        # Condition 3: RSI oversold recovery
        if prev['RSI'] < strategy.rsi_oversold and latest['RSI'] >= strategy.rsi_oversold:
            print(f"   ✅ RSI Oversold Recovery (RSI: {prev['RSI']:.1f} → {latest['RSI']:.1f})")
            buy_count += 1
        else:
            if latest['RSI'] < strategy.rsi_oversold:
                print(f"   ⏳ RSI Oversold Recovery (RSI={latest['RSI']:.1f}, still oversold)")
            else:
                print(f"   ❌ RSI Oversold Recovery (RSI={latest['RSI']:.1f}, not oversold)")
        
        # Condition 4: Trend aligned bullish
        trend_bullish = (latest['Close'] > latest['SMA_short'] > latest['SMA_long'] and 
                        latest['MACD'] > 0 and latest['RSI'] < 65)
        if trend_bullish:
            print(f"   ✅ Trend Aligned Bullish (Price > SMA_short > SMA_long, MACD+, RSI ok)")
            buy_count += 1
        else:
            reasons = []
            if not latest['Close'] > latest['SMA_short']:
                reasons.append(f"Price ${latest['Close']:.2f} < SMA_short ${latest['SMA_short']:.2f}")
            if not latest['SMA_short'] > latest['SMA_long']:
                reasons.append(f"SMA_short < SMA_long")
            if not latest['MACD'] > 0:
                reasons.append(f"MACD={latest['MACD']:.4f} < 0")
            if not latest['RSI'] < 65:
                reasons.append(f"RSI={latest['RSI']:.1f} > 65")
            print(f"   ❌ Trend Aligned Bullish: {', '.join(reasons)}")
        
        print(f"\n   📊 BUY CONDITIONS MET: {buy_count}/4 (need 2+)")
        
        # Check SELL conditions
        print(f"\n🔴 SELL CONDITIONS (need 2+ for signal):")
        sell_count = 0
        
        # Condition 1: SMA death cross
        if sma_cross_change == -2:
            print(f"   ✅ SMA Death Cross (just happened!)")
            sell_count += 1
        else:
            sma_status = "below" if latest['SMA_short'] < latest['SMA_long'] else "above"
            print(f"   ❌ SMA Death Cross (short {sma_status} long, no crossover today)")
        
        # Condition 2: MACD bearish crossover
        if macd_cross_change == -2:
            print(f"   ✅ MACD Bearish Crossover (just happened!)")
            sell_count += 1
        else:
            macd_status = "below" if latest['MACD'] < latest['MACD_signal'] else "above"
            print(f"   ❌ MACD Bearish Crossover (MACD {macd_status} signal, no crossover today)")
        
        # Condition 3: RSI overbought reversal
        if prev['RSI'] > strategy.rsi_overbought and latest['RSI'] <= strategy.rsi_overbought:
            print(f"   ✅ RSI Overbought Reversal (RSI: {prev['RSI']:.1f} → {latest['RSI']:.1f})")
            sell_count += 1
        else:
            if latest['RSI'] > strategy.rsi_overbought:
                print(f"   ⏳ RSI Overbought Reversal (RSI={latest['RSI']:.1f}, still overbought)")
            else:
                print(f"   ❌ RSI Overbought Reversal (RSI={latest['RSI']:.1f}, not overbought)")
        
        # Condition 4: Trend aligned bearish
        trend_bearish = (latest['Close'] < latest['SMA_short'] < latest['SMA_long'] and 
                        latest['MACD'] < 0 and latest['RSI'] > 35)
        if trend_bearish:
            print(f"   ✅ Trend Aligned Bearish (Price < SMA_short < SMA_long, MACD-, RSI ok)")
            sell_count += 1
        else:
            reasons = []
            if not latest['Close'] < latest['SMA_short']:
                reasons.append(f"Price > SMA_short")
            if not latest['SMA_short'] < latest['SMA_long']:
                reasons.append(f"SMA_short > SMA_long")
            if not latest['MACD'] < 0:
                reasons.append(f"MACD > 0")
            if not latest['RSI'] > 35:
                reasons.append(f"RSI < 35")
            print(f"   ❌ Trend Aligned Bearish: {', '.join(reasons)}")
        
        print(f"\n   📊 SELL CONDITIONS MET: {sell_count}/4 (need 2+)")
        
        # Generate actual signal
        signal = strategy.generate_signal(hist_data, symbol, None)
        
        # Calculate confidence
        confidence = strategy._calculate_confidence(df)
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"   Signal: {signal.signal.value.upper()}")
        print(f"   Reason: {signal.reason}")
        print(f"   Confidence: {confidence:.2f} (threshold: 0.40 for TA-only)")
        
        if buy_count >= 2:
            if confidence >= 0.4:
                print(f"   → ✅ WOULD EXECUTE BUY")
            else:
                print(f"   → ❌ Signal blocked by confidence threshold")
        elif buy_count == 1:
            print(f"   → ⏳ 1 more BUY condition needed")
        else:
            print(f"   → ⏳ Waiting for trend conditions to align")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print("""
The strategy requires CROSSOVER EVENTS (the exact moment of crossing).
With daily data, these are rare - you might wait days/weeks for signals.

Options to get more trades:
1. Use shorter SMA periods (e.g., 5/15 instead of 10/30)
2. Lower required conditions from 2 to 1 (riskier)
3. Add more condition types (e.g., volume breakout)
4. Use intraday data for faster signals (but more noise)
""")

if __name__ == "__main__":
    diagnose()
