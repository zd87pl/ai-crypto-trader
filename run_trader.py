import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from auto_trader import AutoTrader
from services.strategy_selection_service import StrategySelectionService
from services.market_regime_service import MarketRegimeService
from services.social_strategy_integrator import SocialStrategyIntegrator
from services.feature_importance_analyzer import FeatureImportanceAnalyzer
from services.social_risk_adjuster import SocialRiskAdjuster
from services.monte_carlo_service import MonteCarloService
from services.neural_network_service import NeuralNetworkService
from services.pattern_recognition_service import PatternRecognitionService
from services.news_analysis_service import NewsAnalysisService
from services.enhanced_social_monitor_service import EnhancedSocialMonitorService
from services.order_book_analysis_service import OrderBookAnalysisService

def setup_logging():
    """Setup logging configuration"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_status(trader, strategy_selector=None):
    """Print current trading status"""
    print("\n=== Trading Status ===")
    print(f"Active Positions: {len(trader.trade_executor.active_trades)}/{trader.config['trading_params']['max_positions']}")
    
    # Print active strategy information if available
    if strategy_selector and hasattr(strategy_selector, 'active_strategy_id'):
        print(f"Active Strategy: {strategy_selector.active_strategy_id}")
        print(f"Current Risk Profile: {strategy_selector.current_risk_profile}")
    
    # Print current market regime if available
    try:
        market_regime_data = trader.redis.get('market_regime_history')
        if market_regime_data:
            regime_history = json.loads(market_regime_data)
            if regime_history and len(regime_history) > 0:
                # Get the most recent regime
                current_regime = regime_history[-1]
                regime_name = current_regime.get('regime', 'unknown')
                confidence = current_regime.get('confidence', 0)
                timestamp = current_regime.get('timestamp', '')
                
                # Format the timestamp
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                # Show the regime with confidence
                print(f"Market Regime: {regime_name.upper()} (Confidence: {confidence:.2f})")
                print(f"Regime Last Updated: {timestamp}")
                
                # Show regime probabilities if available
                if 'probs' in current_regime:
                    print("\nRegime Probabilities:")
                    probs = current_regime['probs']
                    for regime, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        print(f"- {regime.upper()}: {prob:.2f}")
    except Exception as e:
        pass
    
    # Print active trades
    if trader.trade_executor.active_trades:
        print("\nCurrent Positions:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Entry Price':<12} {'Current Price':<12} {'PnL %':<8} {'Duration':<15}")
        print("-" * 80)
        
        total_pnl = 0
        for symbol, trade in trader.trade_executor.active_trades.items():
            try:
                current_price = float(trader.client.get_symbol_ticker(symbol=symbol)['price'])
                entry_price = trade['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                entry_time = datetime.fromisoformat(trade['entry_time'])
                duration = datetime.now() - entry_time
                hours = duration.total_seconds() / 3600
                
                print(f"{symbol:<10} "
                      f"${entry_price:<11.4f} "
                      f"${current_price:<11.4f} "
                      f"{pnl_pct:>7.2f}% "
                      f"{f'{hours:.1f}h':<15}")
                
                total_pnl += (current_price - entry_price) * trade['quantity']
            except Exception as e:
                print(f"Error getting data for {symbol}: {str(e)}")
        
        print("-" * 80)
        print(f"Total PnL: ${total_pnl:.2f}")
    
    # Print current opportunities
    if not trader.opportunity_queue.empty():
        print("\nCurrent Opportunities:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Price':<12} {'Volume':<15} {'Change %':<8}")
        print("-" * 80)
        
        # Get opportunities without removing them from queue
        opportunities = []
        while not trader.opportunity_queue.empty():
            opp = trader.opportunity_queue.get()
            opportunities.append(opp)
            print(f"{opp['symbol']:<10} "
                  f"${opp['price']:<11.4f} "
                  f"${opp['volume']:>14,.0f} "
                  f"{opp['price_change']:>7.2f}%")
        
        # Put opportunities back in queue
        for opp in opportunities:
            trader.opportunity_queue.put(opp)

    # Print strategy selection metrics if available
    try:
        strategy_metrics = trader.redis.get('strategy_selection_metrics') 
        if strategy_metrics:
            metrics = json.loads(strategy_metrics)
            
            # Only print if the metrics are recent (less than 10 minutes old)
            metrics_time = datetime.fromisoformat(metrics['timestamp'])
            if datetime.now() - metrics_time < timedelta(minutes=10):
                print("\nStrategy Selection Metrics:")
                print("-" * 80)
                print(f"Optimal Strategy: {metrics['optimal_strategy_id']} (Score: {metrics['optimal_score']:.4f})")
                print(f"Current Strategy: {metrics['active_strategy_id']}")
                print(f"Market Regime: {metrics['market_regime']}")
                
                # Print factor scores if available
                if 'factor_scores' in metrics:
                    print("\nFactor Scores:")
                    for factor, score in metrics['factor_scores'].items():
                        print(f"- {factor}: {score:.4f}")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print feature importance information if available
    try:
        feature_importance = trader.redis.get('feature_importance')
        if feature_importance:
            importance_data = json.loads(feature_importance)
            
            # Only print if the data is recent (less than 1 hour old)
            importance_time = datetime.fromisoformat(importance_data['timestamp'])
            if datetime.now() - importance_time < timedelta(hours=1):
                print("\nFeature Importance Analysis:")
                print("-" * 80)
                
                # Print top 5 most important features for trading success
                if 'classification' in importance_data:
                    # Sort features by importance
                    features = [(k, v) for k, v in importance_data['classification'].items()]
                    sorted_features = sorted(features, key=lambda x: x[1], reverse=True)[:5]
                    
                    print("Top 5 features for trade success prediction:")
                    for feature, importance in sorted_features:
                        print(f"- {feature}: {importance:.4f}")
                
                # Print feature group importance if available
                if 'feature_groups' in importance_data:
                    print("\nFeature Group Importance:")
                    groups = importance_data['feature_groups']
                    for group, data in sorted(groups.items(), 
                                              key=lambda x: x[1]['classification'], 
                                              reverse=True):
                        print(f"- {group}: {data['classification']:.4f}")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print social risk adjustment information if available
    try:
        social_risk_report = trader.redis.get('social_risk_report')
        if social_risk_report:
            risk_data = json.loads(social_risk_report)
            
            # Only print if the data is recent (less than 10 minutes old)
            report_time = datetime.fromisoformat(risk_data['timestamp'])
            if datetime.now() - report_time < timedelta(minutes=10):
                print("\nSocial Risk Adjustments:")
                print("-" * 80)
                print(f"Active Adjustments: {risk_data.get('active_adjustments', 0)}")
                
                # Print a summary of the active adjustments
                if 'adjustments' in risk_data and risk_data['adjustments']:
                    print("\nActive Risk Adjustments by Symbol:")
                    for symbol, adj in risk_data['adjustments'].items():
                        sentiment_type = adj.get('sentiment_type', 'NEUTRAL')
                        sentiment_score = adj.get('sentiment_score', 0.5)
                        pos_adj = adj.get('position_size_adj', 0) * 100
                        sl_adj = adj.get('stop_loss_adj', 0) * 100
                        tp_adj = adj.get('take_profit_adj', 0) * 100
                        
                        print(f"- {symbol}: {sentiment_type} (Score: {sentiment_score:.2f}) | " +
                              f"Pos: {pos_adj:+.1f}%, SL: {sl_adj:+.1f}%, TP: {tp_adj:+.1f}%")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print Monte Carlo simulation results if available
    try:
        monte_carlo_report = trader.redis.get('monte_carlo_latest_report')
        if monte_carlo_report:
            report_data = json.loads(monte_carlo_report)
            
            # Only print if the data is recent (less than 1 day old)
            if 'timestamp' in report_data:
                report_time = datetime.fromisoformat(report_data['timestamp'])
                if datetime.now() - report_time < timedelta(days=1):
                    print("\nMonte Carlo Risk Projection:")
                    print("-" * 80)
                    
                    # Portfolio value and projections
                    print(f"Current Portfolio Value: ${report_data.get('portfolio_value', 0):.2f}")
                    print(f"Expected Value (30 days): {report_data.get('expected_change', '0.00%')}")
                    
                    # Value at Risk
                    var_section = report_data.get('value_at_risk', {})
                    print(f"Value at Risk (95%): {var_section.get('var_percent', '0.00%')} (${var_section.get('var_amount', '$0.00')})")
                    
                    # Scenario analysis 
                    if 'scenario_analysis' in report_data and report_data['scenario_analysis']:
                        print("\nScenario Analysis:")
                        scenarios = report_data['scenario_analysis']
                        for scenario, data in scenarios.items():
                            expected_return = data.get('expected_return', '0.00%')
                            var = data.get('var', '0.00%')
                            print(f"- {scenario.capitalize()}: Return {expected_return}, VaR {var}")
                    
                    # Individual asset analysis
                    if 'asset_analysis' in report_data and report_data['asset_analysis']:
                        print("\nAsset Risk Analysis:")
                        asset_data = report_data['asset_analysis']
                        for symbol, data in asset_data.items():
                            expected_return = data.get('expected_return', '0.00%')
                            var = data.get('var', '0.00%')
                            prob_profit = data.get('prob_profit', '0.0%')
                            print(f"- {symbol}: Return {expected_return}, VaR {var}, Prob. Profit {prob_profit}")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print neural network price predictions
    try:
        symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]  # Default symbols to check
        intervals = ["1h", "4h", "24h"]  # Default prediction intervals
        
        print("\nNeural Network Price Predictions:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Interval':<8} {'Current':<10} {'Predicted':<10} {'Change':<10} {'Confidence':<10}")
        print("-" * 80)
        
        predictions_found = False
        
        for symbol in symbols:
            for interval in intervals:
                prediction_key = f'nn_prediction_{symbol}_{interval}'
                prediction_json = trader.redis.get(prediction_key)
                
                if not prediction_json:
                    continue
                
                prediction = json.loads(prediction_json)
                
                if prediction.get('status') != 'success':
                    continue
                    
                predictions_found = True
                current_price = prediction.get('current_price', 0)
                predicted_price = prediction.get('predicted_price', 0)
                change_pct = prediction.get('change_pct', 0)
                confidence = prediction.get('confidence', 0)
                
                # Format the prediction time
                prediction_time = prediction.get('prediction_time', '')
                if prediction_time:
                    try:
                        dt = datetime.fromisoformat(prediction_time)
                        time_str = dt.strftime("%H:%M %m/%d")
                    except:
                        time_str = prediction_time
                else:
                    time_str = 'N/A'
                
                # Add emoji based on direction
                direction_emoji = "🟢" if change_pct > 0 else "🔴"
                
                print(f"{symbol:<10} "
                     f"{interval:<8} "
                     f"${current_price:<9.4f} "
                     f"${predicted_price:<9.4f} "
                     f"{direction_emoji} {abs(change_pct):<7.2f}% "
                     f"{confidence*100:<9.1f}%")
        
        if not predictions_found:
            print("No predictions available yet")
    
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print chart pattern recognition data
    try:
        print("\nChart Pattern Recognition:")
        print("-" * 80)
        
        # Get combined pattern analysis report
        pattern_report = trader.redis.get('pattern_analysis_report')
        
        if pattern_report:
            report_data = json.loads(pattern_report)
            
            # Only print if the data is recent (less than 30 minutes old)
            if 'timestamp' in report_data:
                report_time = datetime.fromisoformat(report_data['timestamp'])
                if datetime.now() - report_time < timedelta(minutes=30):
                    
                    # Print summary statistics
                    summary = report_data.get('summary', {})
                    bullish = summary.get('bullish_patterns', 0)
                    bearish = summary.get('bearish_patterns', 0)
                    neutral = summary.get('neutral_patterns', 0)
                    
                    print(f"Patterns found: {bullish + bearish + neutral} (Bullish: {bullish}, Bearish: {bearish}, Neutral: {neutral})")
                    
                    # Print strongest signal if available
                    strongest = summary.get('strongest_signal', (None, None))
                    if strongest[0]:
                        symbol, signal = strongest
                        print(f"Strongest pattern: {symbol} - {signal['pattern'].replace('_', ' ').title()} (Strength: {signal['strength']:.2f})")
                    
                    # Print top pattern signals
                    if 'signals' in report_data and report_data['signals']:
                        signals = report_data['signals']
                        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['strength'], reverse=True)[:3]
                        
                        print("\nTop Pattern Signals:")
                        print(f"{'Symbol':<10} {'Pattern':<25} {'Signal':<8} {'Strength':<10} {'Completion':<10}")
                        print("-" * 80)
                        
                        for symbol, data in sorted_signals:
                            pattern = data.get('pattern', 'unknown').replace('_', ' ').title()
                            signal = data.get('signal', 'neutral').upper()
                            strength = data.get('strength', 0)
                            completion = data.get('completion', 0)
                            
                            # Add color/emoji based on signal
                            signal_indicator = ""
                            if signal == "BUY":
                                signal_indicator = "🟢 BUY"
                            elif signal == "SELL":
                                signal_indicator = "🔴 SELL"
                            else:
                                signal_indicator = "⚪ HOLD"
                            
                            print(f"{symbol:<10} {pattern:<25} {signal_indicator:<8} {strength:<10.2f} {completion:<10}%")
                else:
                    print("No recent pattern data available")
            else:
                print("No pattern data available")
        else:
            # Check for individual pattern signals - at most 3 individual examples
            displayed = 0
            signals_shown = False
            symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]  # Default symbols to check
            
            for symbol in symbols:
                pattern_key = f"pattern:{symbol}"
                pattern_json = trader.redis.get(pattern_key)
                
                if not pattern_json:
                    continue
                    
                if not signals_shown:
                    print("Recent Detected Patterns:")
                    print(f"{'Symbol':<10} {'Pattern':<25} {'Confidence':<10} {'Completion':<10}")
                    print("-" * 80)
                    signals_shown = True
                
                pattern_data = json.loads(pattern_json)
                
                # Only show if data is recent (less than 30 minutes old)
                pattern_time = datetime.fromisoformat(pattern_data.get('timestamp', '2000-01-01T00:00:00'))
                if datetime.now() - pattern_time > timedelta(minutes=30):
                    continue
                    
                pattern = pattern_data.get('primary_pattern', 'unknown').replace('_', ' ').title()
                confidence = pattern_data.get('confidence', 0)
                completion = pattern_data.get('completion_percentage', 0)
                
                print(f"{symbol:<10} {pattern:<25} {confidence:<10.2f} {completion:<10}%")
                
                displayed += 1
                if displayed >= 3:
                    break
                    
            if not signals_shown:
                print("No pattern data available yet")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print news analysis data
    try:
        print("\nNews Analysis:")
        print("-" * 80)
        
        # Get news summary report
        news_report = trader.redis.get('news_summary_report')
        
        if news_report:
            report_data = json.loads(news_report)
            
            # Only print if the data is recent (less than 30 minutes old)
            if 'timestamp' in report_data:
                report_time = datetime.fromisoformat(report_data['timestamp'])
                if datetime.now() - report_time < timedelta(minutes=30):
                    
                    # Print overall market sentiment
                    if 'market_sentiment' in report_data:
                        sentiment = report_data['market_sentiment']
                        sentiment_score = report_data.get('sentiment_score', 0)
                        
                        # Add emoji based on sentiment
                        sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "⚪"
                        print(f"Market News Sentiment: {sentiment_emoji} {sentiment.title()} (Score: {sentiment_score:.2f})")
                    
                    # Print top news topics
                    if 'top_topics' in report_data and report_data['top_topics']:
                        topics = report_data['top_topics'][:3]  # Show top 3 topics
                        print("Top News Topics:")
                        for topic in topics:
                            topic_name = topic.get('topic', '')
                            topic_count = topic.get('count', 0)
                            topic_sentiment = topic.get('sentiment', 0)
                            
                            # Add emoji based on sentiment
                            topic_emoji = "🟢" if topic_sentiment > 0.6 else "🔴" if topic_sentiment < 0.4 else "⚪"
                            print(f"- {topic_name} ({topic_count} articles): {topic_emoji} {topic_sentiment:.2f}")
                    
                    # Print asset-specific news
                    if 'asset_news' in report_data and report_data['asset_news']:
                        news_items = report_data['asset_news']
                        print("\nAsset-Specific News Sentiment:")
                        print(f"{'Symbol':<10} {'Sentiment':<15} {'Articles':<10} {'Top Entity':<20}")
                        print("-" * 80)
                        
                        for symbol, data in sorted(news_items.items(), 
                                                  key=lambda x: abs(x[1].get('sentiment_score', 0) - 0.5), 
                                                  reverse=True)[:5]:  # Sort by most extreme sentiment and show top 5
                            sentiment = data.get('sentiment', 'neutral').upper()
                            sentiment_score = data.get('sentiment_score', 0.5)
                            article_count = data.get('article_count', 0)
                            
                            # Get top entity if available
                            top_entity = "N/A"
                            if 'entities' in data and data['entities']:
                                top_entity = data['entities'][0].get('entity', 'N/A')
                            
                            # Add emoji based on sentiment
                            sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "⚪"
                            
                            print(f"{symbol:<10} {sentiment_emoji} {sentiment:<12} {article_count:<10} {top_entity:<20}")
                    
                    # Print hot news summary if available
                    if 'hot_news' in report_data and report_data['hot_news']:
                        print("\nKey News Items:")
                        for i, news in enumerate(report_data['hot_news'][:3]):  # Show top 3 news items
                            title = news.get('title', 'N/A')
                            source = news.get('source', 'N/A')
                            sentiment = news.get('sentiment_score', 0.5)
                            
                            # Add emoji based on sentiment
                            sentiment_emoji = "🟢" if sentiment > 0.6 else "🔴" if sentiment < 0.4 else "⚪"
                            
                            print(f"{i+1}. {title} ({source}) {sentiment_emoji}")
                else:
                    print("No recent news data available")
            else:
                print("No news data available")
        else:
            # Check for individual symbol news
            displayed = 0
            news_shown = False
            symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]  # Default symbols to check
            
            for symbol in symbols:
                # Convert symbol to base asset for news lookup (e.g., BTCUSDC -> BTC)
                base_asset = symbol.replace("USDC", "").replace("USDT", "")
                news_key = f"news:{base_asset}"
                news_json = trader.redis.get(news_key)
                
                if not news_json:
                    continue
                    
                if not news_shown:
                    print("Recent News Analysis:")
                    print(f"{'Asset':<10} {'Sentiment':<15} {'Articles':<10} {'Hot Topic':<20}")
                    print("-" * 80)
                    news_shown = True
                
                news_data = json.loads(news_json)
                
                # Only show if data is recent (less than 30 minutes old)
                news_time = datetime.fromisoformat(news_data.get('timestamp', '2000-01-01T00:00:00'))
                if datetime.now() - news_time > timedelta(minutes=30):
                    continue
                    
                sentiment = news_data.get('sentiment', 'neutral').upper()
                sentiment_score = news_data.get('sentiment_score', 0.5)
                article_count = news_data.get('article_count', 0)
                
                # Get hot topic if available
                hot_topic = "N/A"
                if 'topics' in news_data and news_data['topics']:
                    hot_topic = news_data['topics'][0].get('topic', 'N/A')
                
                # Add emoji based on sentiment
                sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "⚪"
                
                print(f"{base_asset:<10} {sentiment_emoji} {sentiment:<12} {article_count:<10} {hot_topic:<20}")
                
                displayed += 1
                if displayed >= 3:
                    break
                    
            if not news_shown:
                print("No news analysis data available yet")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print enhanced social metrics data
    try:
        print("\nEnhanced Social Metrics:")
        print("-" * 80)
        
        # Get social accuracy report
        accuracy_report = trader.redis.get('social_accuracy_report')
        
        if accuracy_report:
            report_data = json.loads(accuracy_report)
            
            # Only print if the data is recent (less than 12 hours old)
            if 'timestamp' in report_data:
                report_time = datetime.fromisoformat(report_data['timestamp'])
                if datetime.now() - report_time < timedelta(hours=12):
                    # Print overall accuracy metrics
                    avg_accuracy = report_data.get('average_direction_accuracy', 0.0)
                    total_symbols = report_data.get('total_symbols', 0)
                    
                    # Add emoji based on accuracy
                    accuracy_emoji = "🟢" if avg_accuracy > 0.6 else "🔴" if avg_accuracy < 0.45 else "⚪"
                    print(f"Social Metrics Prediction Accuracy: {accuracy_emoji} {avg_accuracy:.2f} (across {total_symbols} symbols)")
                    
                    # Print symbols with highest accuracy
                    if 'symbols' in report_data and report_data['symbols']:
                        print("\nTop Social Metrics Accuracy by Symbol:")
                        print(f"{'Symbol':<10} {'Direction Acc.':<15} {'Correlation':<15} {'Optimal Lag':<15}")
                        print("-" * 80)
                        
                        # Sort symbols by accuracy
                        sorted_symbols = sorted(
                            [(symbol, data) for symbol, data in report_data['symbols'].items()],
                            key=lambda x: x[1].get('direction_accuracy', 0.0),
                            reverse=True
                        )[:5]  # Show top 5
                        
                        for symbol, data in sorted_symbols:
                            direction_acc = data.get('direction_accuracy', 0.0)
                            correlation = data.get('correlation', 0.0)
                            optimal_lag = data.get('optimal_lag', 0)
                            
                            # Add emoji based on accuracy
                            acc_emoji = "🟢" if direction_acc > 0.6 else "🔴" if direction_acc < 0.45 else "⚪"
                            
                            # Format optimal lag with direction
                            if optimal_lag > 0:
                                lag_str = f"+{optimal_lag}h (leads)"
                            elif optimal_lag < 0:
                                lag_str = f"{optimal_lag}h (lags)"
                            else:
                                lag_str = "0h (concurrent)"
                            
                            print(f"{symbol:<10} {acc_emoji} {direction_acc:.2f} {correlation:>14.2f} {lag_str:<15}")
        
        # Get performance report
        performance_report = trader.redis.get('enhanced_social_monitor_performance')
        
        if performance_report:
            perf_data = json.loads(performance_report)
            
            # Only print if the data is recent (less than 10 minutes old)
            if 'timestamp' in perf_data:
                report_time = datetime.fromisoformat(perf_data['timestamp'])
                if datetime.now() - report_time < timedelta(minutes=10):
                    print("\nSocial Monitor Performance:")
                    print(f"Monitored symbols: {perf_data.get('monitored_symbols', 0)}")
                    print(f"Anomalies detected: {perf_data.get('anomalies_detected', 0)}")
                    print(f"Processing time: {perf_data.get('avg_processing_time', 0.0):.3f}s")
                    
        # Get enhanced social metrics for a few key symbols
        symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]  # Default symbols to check
        displayed = 0
        metrics_shown = False
        
        for symbol in symbols:
            metrics_key = f"enhanced_social_metrics:{symbol}"
            metrics_json = trader.redis.get(metrics_key)
            
            if not metrics_json:
                # Try the regular key for backward compatibility
                metrics_json = trader.redis.hget('enhanced_social_metrics', symbol)
                
            if not metrics_json:
                continue
                
            if not metrics_shown:
                print("\nEnhanced Social Sentiment:")
                print(f"{'Symbol':<10} {'Raw':<10} {'Enhanced':<10} {'Lead/Lag':<15} {'Anomaly':<10}")
                print("-" * 80)
                metrics_shown = True
            
            metrics_data = json.loads(metrics_json)
            
            # Only show if data is recent (less than 15 minutes old)
            metrics_time = datetime.fromisoformat(metrics_data.get('timestamp', '2000-01-01T00:00:00'))
            if datetime.now() - metrics_time > timedelta(minutes=15):
                continue
                
            # Get sentiment data
            enhanced_sentiment = metrics_data.get('enhanced_sentiment', {})
            raw_sentiment = enhanced_sentiment.get('raw_sentiment', 0.5)
            enhanced_sentiment_value = enhanced_sentiment.get('enhanced_sentiment', 0.5)
            optimal_lag = enhanced_sentiment.get('optimal_lag', 0)
            is_anomaly = enhanced_sentiment.get('is_anomaly', False)
            
            # Format optimal lag with direction
            if optimal_lag > 0:
                lag_str = f"+{optimal_lag}h (leads)"
            elif optimal_lag < 0:
                lag_str = f"{optimal_lag}h (lags)"
            else:
                lag_str = "0h"
            
            # Add emoji based on sentiment
            raw_emoji = "🟢" if raw_sentiment > 0.6 else "🔴" if raw_sentiment < 0.4 else "⚪"
            enhanced_emoji = "🟢" if enhanced_sentiment_value > 0.6 else "🔴" if enhanced_sentiment_value < 0.4 else "⚪"
            
            # Format symbol (strip USDC/USDT)
            display_symbol = symbol.replace("USDC", "").replace("USDT", "")
            
            print(f"{display_symbol:<10} {raw_emoji} {raw_sentiment:.2f} {enhanced_emoji} {enhanced_sentiment_value:.2f} {lag_str:<15} {'⚠️ Yes' if is_anomaly else 'No':<10}")
            
            displayed += 1
            if displayed >= 5:  # Limit to 5 symbols
                break
                
        if not metrics_shown:
            print("No enhanced social metrics data available yet")
    except Exception as e:
        pass  # Silently ignore any errors here
        
    # Print order book analysis data
    try:
        print("\nOrder Book Analysis:")
        print("-" * 80)
        
        # Get order book analysis summary
        summary_json = trader.redis.get('order_book_analysis_summary')
        
        if summary_json:
            summary = json.loads(summary_json)
            
            # Only print if the data is recent (less than 5 minutes old)
            if 'timestamp' in summary:
                summary_time = datetime.fromisoformat(summary['timestamp'])
                if datetime.now() - summary_time < timedelta(minutes=5):
                    # Print summary statistics
                    monitored = summary.get('monitored_symbols', 0)
                    buy_signals = summary.get('buy_signals', 0)
                    sell_signals = summary.get('sell_signals', 0)
                    neutral_signals = summary.get('neutral_signals', 0)
                    
                    # Display summary
                    print(f"Monitored symbols: {monitored} | Signals: {buy_signals} Buy, {sell_signals} Sell, {neutral_signals} Neutral")
                    
                    # Print strongest buy/sell signals
                    strongest_buy = summary.get('strongest_buy')
                    strongest_buy_conf = summary.get('strongest_buy_confidence', 0)
                    strongest_sell = summary.get('strongest_sell')
                    strongest_sell_conf = summary.get('strongest_sell_confidence', 0)
                    
                    if strongest_buy:
                        print(f"Strongest buy signal: {strongest_buy} (Confidence: {strongest_buy_conf:.2f})")
                    if strongest_sell:
                        print(f"Strongest sell signal: {strongest_sell} (Confidence: {strongest_sell_conf:.2f})")
                    
                    # Print signals by symbol
                    print("\nOrder Book Trading Signals:")
                    print(f"{'Symbol':<10} {'Signal':<10} {'Confidence':<12} {'Buy Signals':<12} {'Sell Signals':<12}")
                    print("-" * 80)
                    
                    signals_by_symbol = summary.get('signals_by_symbol', {})
                    displayed = 0
                    
                    # Show signals sorted by confidence
                    sorted_symbols = []
                    for symbol, data in signals_by_symbol.items():
                        # Skip neutral signals with low confidence for cleaner display
                        if data.get('signal') == 'neutral' and data.get('confidence', 0) < 0.7:
                            continue
                        
                        # Calculate a sort value that prioritizes non-neutral signals and high confidence
                        sort_value = 0
                        if data.get('signal') != 'neutral':
                            sort_value = data.get('confidence', 0)
                        else:
                            sort_value = data.get('confidence', 0) * 0.5
                            
                        sorted_symbols.append((symbol, data, sort_value))
                    
                    # Sort and display top signals
                    for symbol, data, _ in sorted(sorted_symbols, key=lambda x: x[2], reverse=True)[:5]:
                        signal = data.get('signal', 'neutral').upper()
                        confidence = data.get('confidence', 0)
                        buy_count = data.get('buy_signals', 0)
                        sell_count = data.get('sell_signals', 0)
                        
                        # Add signal emoji
                        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                        
                        # Format symbol for display
                        display_symbol = symbol.replace("USDC", "").replace("USDT", "")
                        
                        print(f"{display_symbol:<10} {signal_emoji} {signal:<8} {confidence:<12.2f} {buy_count:<12} {sell_count:<12}")
                        
                        displayed += 1
                        if displayed >= 5:  # Limit to 5 symbols
                            break
        
        # Get specific order book metrics for a few key symbols
        symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC"]  # Default symbols to check
        displayed = 0
        metrics_shown = False
        
        for symbol in symbols:
            # Get aggregated metrics for cleaner display
            metrics_key = f"order_book_agg:{symbol}"
            metrics_json = trader.redis.get(metrics_key)
            
            if not metrics_json:
                continue
                
            if not metrics_shown:
                print("\nOrder Book Liquidity Metrics:")
                print(f"{'Symbol':<10} {'Imbalance':<15} {'Bid/Ask Ratio':<15} {'Near Pressure':<15} {'Spread %':<10}")
                print("-" * 80)
                metrics_shown = True
            
            metrics = json.loads(metrics_json)
            
            # Only show if data is recent (less than 5 minutes old)
            metrics_time = datetime.fromisoformat(metrics.get('timestamp', '2000-01-01T00:00:00'))
            if datetime.now() - metrics_time > timedelta(minutes=5):
                continue
            
            # Get current metrics
            current = metrics.get('current_metrics', {})
            imbalance = current.get('liquidity_imbalance', 0)
            bid_ask_ratio = current.get('bid_ask_ratio', 1.0)
            near_pressure = current.get('near_pressure', 0)
            spread = current.get('spread_percentage', 0)
            
            # Add emojis based on metrics
            imbalance_emoji = "🟢" if imbalance > 0.1 else "🔴" if imbalance < -0.1 else "⚪"
            pressure_emoji = "🟢" if near_pressure > 0.1 else "🔴" if near_pressure < -0.1 else "⚪"
            
            # Format symbol for display
            display_symbol = symbol.replace("USDC", "").replace("USDT", "")
            
            print(f"{display_symbol:<10} {imbalance_emoji} {imbalance:>8.2f} {bid_ask_ratio:>14.2f} {pressure_emoji} {near_pressure:>8.2f} {spread:>9.2f}%")
            
            displayed += 1
            if displayed >= 5:  # Limit to 5 symbols
                break
        
        # Check for support/resistance levels for key symbols
        support_resist_shown = False
        displayed = 0
        
        for symbol in symbols:
            # Get full order book analysis for support/resistance
            order_book_key = f"order_book:{symbol}"
            order_book_json = trader.redis.get(order_book_key)
            
            if not order_book_json:
                continue
            
            order_book_data = json.loads(order_book_json)
            
            # Only show if data is recent (less than 5 minutes old)
            data_time = datetime.fromisoformat(order_book_data.get('timestamp', '2000-01-01T00:00:00'))
            if datetime.now() - data_time > timedelta(minutes=5):
                continue
            
            # Get support/resistance levels
            support_levels = order_book_data.get('support_levels', [])
            resistance_levels = order_book_data.get('resistance_levels', [])
            
            if not support_levels and not resistance_levels:
                continue
                
            if not support_resist_shown:
                print("\nKey Support & Resistance Levels:")
                print(f"{'Symbol':<10} {'Type':<12} {'Price':<12} {'Strength':<10} {'Distance':<10}")
                print("-" * 80)
                support_resist_shown = True
            
            # Format symbol for display
            display_symbol = symbol.replace("USDC", "").replace("USDT", "")
            current_price = order_book_data.get('current_price', 0)
            
            # Show top support level
            if support_levels:
                top_support = support_levels[0]
                support_price = top_support.get('price', 0)
                support_strength = top_support.get('strength', 0)
                support_distance = (current_price - support_price) / current_price * 100
                
                print(f"{display_symbol:<10} {'Support':<12} ${support_price:<11.4f} {support_strength:<10.2f} {support_distance:>9.2f}%")
            
            # Show top resistance level
            if resistance_levels:
                top_resistance = resistance_levels[0]
                resistance_price = top_resistance.get('price', 0)
                resistance_strength = top_resistance.get('strength', 0)
                resistance_distance = (resistance_price - current_price) / current_price * 100
                
                print(f"{display_symbol:<10} {'Resistance':<12} ${resistance_price:<11.4f} {resistance_strength:<10.2f} {resistance_distance:>9.2f}%")
            
            displayed += 1
            if displayed >= 5:  # Limit to 5 symbols
                break
                
        if not metrics_shown and not support_resist_shown:
            print("No order book analysis data available yet")
    except Exception as e:
        pass  # Silently ignore any errors here

async def run_strategy_selection_service():
    """Run the strategy selection service"""
    service = StrategySelectionService()
    await service.run()

async def run_market_regime_service():
    """Run the market regime service"""
    service = MarketRegimeService()
    await service.run()

async def run_social_strategy_service():
    """Run the social strategy integrator service"""
    service = SocialStrategyIntegrator()
    await service.run()

async def run_feature_importance_service():
    """Run the feature importance analysis service"""
    service = FeatureImportanceAnalyzer()
    await service.start()
    
async def run_social_risk_adjuster_service():
    """Run the social risk adjuster service"""
    service = SocialRiskAdjuster()
    await service.run()
    
async def run_monte_carlo_service():
    """Run the Monte Carlo simulation service"""
    service = MonteCarloService()
    await service.run()
    
async def run_neural_network_service():
    """Run the neural network price prediction service"""
    service = NeuralNetworkService()
    await service.run()
    
async def run_pattern_recognition_service():
    """Run the pattern recognition service"""
    service = PatternRecognitionService()
    await service.run()
    
async def run_news_analysis_service():
    """Run the news analysis service"""
    service = NewsAnalysisService()
    await service.run()
    
async def run_enhanced_social_monitor_service():
    """Run the enhanced social monitor service"""
    service = EnhancedSocialMonitorService()
    await service.run()

async def run_order_book_analysis_service():
    """Run the order book analysis service"""
    service = OrderBookAnalysisService()
    await service.run()

def run_async_service(async_func):
    """Run an async service in the current thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func())
    loop.close()

def main():
    """Main function to run the trader with status monitoring"""
    setup_logging()
    logging.info("Starting trading bot with automated strategy selection...")
    
    # Flag to control service threads
    running = True
    
    try:
        # Start market regime service in a separate thread
        regime_thread = threading.Thread(
            target=run_async_service,
            args=(run_market_regime_service,),
            daemon=True
        )
        regime_thread.start()
        logging.info("Market regime service started")
        
        # Start strategy selection service in a separate thread
        strategy_thread = threading.Thread(
            target=run_async_service,
            args=(run_strategy_selection_service,),
            daemon=True
        )
        strategy_thread.start()
        logging.info("Strategy selection service started")
        
        # Start social strategy integrator in a separate thread
        social_thread = threading.Thread(
            target=run_async_service,
            args=(run_social_strategy_service,),
            daemon=True
        )
        social_thread.start()
        logging.info("Social strategy integrator service started")
        
        # Start feature importance analysis service in a separate thread
        feature_importance_thread = threading.Thread(
            target=run_async_service,
            args=(run_feature_importance_service,),
            daemon=True
        )
        feature_importance_thread.start()
        logging.info("Feature importance analysis service started")
        
        # Start social risk adjuster service in a separate thread
        social_risk_thread = threading.Thread(
            target=run_async_service,
            args=(run_social_risk_adjuster_service,),
            daemon=True
        )
        social_risk_thread.start()
        logging.info("Social risk adjuster service started")
        
        # Start Monte Carlo simulation service in a separate thread
        monte_carlo_thread = threading.Thread(
            target=run_async_service,
            args=(run_monte_carlo_service,),
            daemon=True
        )
        monte_carlo_thread.start()
        logging.info("Monte Carlo simulation service started")
        
        # Start Neural Network price prediction service in a separate thread
        neural_network_thread = threading.Thread(
            target=run_async_service,
            args=(run_neural_network_service,),
            daemon=True
        )
        neural_network_thread.start()
        logging.info("Neural Network price prediction service started")
        
        # Start Pattern Recognition service in a separate thread
        pattern_recognition_thread = threading.Thread(
            target=run_async_service,
            args=(run_pattern_recognition_service,),
            daemon=True
        )
        pattern_recognition_thread.start()
        logging.info("Pattern Recognition service started")
        
        # Start News Analysis service in a separate thread
        news_analysis_thread = threading.Thread(
            target=run_async_service,
            args=(run_news_analysis_service,),
            daemon=True
        )
        news_analysis_thread.start()
        logging.info("News Analysis service started")
        
        # Start Enhanced Social Monitor service in a separate thread
        enhanced_social_thread = threading.Thread(
            target=run_async_service,
            args=(run_enhanced_social_monitor_service,),
            daemon=True
        )
        enhanced_social_thread.start()
        logging.info("Enhanced Social Monitor service started")
        
        # Start Order Book Analysis service in a separate thread
        order_book_thread = threading.Thread(
            target=run_async_service,
            args=(run_order_book_analysis_service,),
            daemon=True
        )
        order_book_thread.start()
        logging.info("Order Book Analysis service started")
        
        # Small delay to allow services to initialize
        time.sleep(3)
        
        # Initialize trader
        trader = AutoTrader()
        
        # Get strategy selection service instance for status display
        # Note: We only use this for display purposes, as the actual service runs in its own thread
        strategy_selector = StrategySelectionService()
        
        # Get initial balance
        account = trader.client.get_account()
        initial_balance = next(
            (float(asset['free']) for asset in account['balances'] 
             if asset['asset'] == 'USDC'),
            0.0
        )
        
        # Print initial configuration
        print("\n=== Trading Bot Configuration ===")
        print(f"Initial USDC Balance: ${initial_balance:.2f}")
        print(f"Reserve Ratio: {trader.config['trading_params'].get('reserve_ratio', 0.1) * 100}%")
        print(f"Max Positions: {trader.config['trading_params']['max_positions']}")
        print(f"Min Volume: ${trader.config['trading_params']['min_volume_usdc']:,}")
        print(f"Position Size: {trader.config['trading_params'].get('position_size', 0.1) * 100}% of available capital")
        print(f"Strategy Selection: Automated (Market Regime: {strategy_selector.selection_weights['market_regime']:.2f}, Performance: {strategy_selector.selection_weights['historical_performance']:.2f})")
        
        # Start the trader
        trader.start()
        
        # Monitor and print status
        while running:
            print_status(trader, strategy_selector)
            time.sleep(5)  # Update status every 5 seconds
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading bot...")
        print("\nShutting down gracefully...")
        
        # Set flag to stop all services
        running = False
        
        # Stop the trader
        trader.stop()
        
        # Close all positions if needed
        if trader.trade_executor.active_trades:
            print("\nClosing all active positions...")
            for symbol, trade in trader.trade_executor.active_trades.items():
                try:
                    # Cancel all open orders
                    trader.client.cancel_open_orders(symbol=symbol)
                    # Place market sell order
                    trader.client.create_order(
                        symbol=symbol,
                        side='SELL',
                        type='MARKET',
                        quantity=trade['quantity']
                    )
                    print(f"Closed position for {symbol}")
                except Exception as e:
                    print(f"Error closing position for {symbol}: {str(e)}")
        
        print("\nTrading bot stopped.")
        
        # Give threads a moment to clean up
        time.sleep(2)
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        running = False
        raise

if __name__ == "__main__":
    main()
