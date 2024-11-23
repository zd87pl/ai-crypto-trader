# Crypto Trading Bot - Microservices Architecture

A sophisticated cryptocurrency trading bot built with a microservices architecture, leveraging AI for market analysis and automated trading decisions.

## Architecture Overview

The system consists of three main microservices:

1. **Market Monitor Service**
   - Streams real-time market data from Binance
   - Processes and filters trading opportunities
   - Publishes market updates to Redis

2. **AI Analyzer Service**
   - Analyzes market data using OpenAI
   - Generates trading signals
   - Performs risk analysis

3. **Trade Executor Service**
   - Executes trades based on AI signals
   - Manages positions and risk
   - Handles order lifecycle

## Prerequisites

- Docker and Docker Compose
- Binance API credentials
- OpenAI API key
- Redis (automatically set up via Docker Compose)

## Configuration

1. Create and configure the `.env` file:
```bash
cp .env.example .env
```

Edit with your credentials:
```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
OPENAI_API_KEY=your_openai_api_key
```

2. Configure trading parameters in `config.json`:
```json
{
  "trading_params": {
    "min_volume_usdc": 100000,
    "position_size": 0.15,
    "max_positions": 5,
    "stop_loss_pct": 2,
    "take_profit_pct": 4
  }
}
```

## Installation & Deployment

1. Build the Docker images:
```bash
docker-compose build
```

2. Start the services:
```bash
docker-compose up -d
```

3. Monitor the logs:
```bash
docker-compose logs -f
```

4. Stop the services:
```bash
docker-compose down
```

## Service Ports

- Market Monitor: 8001
- Trade Executor: 8002
- AI Analyzer: 8003
- Redis: 6379

## Monitoring

Monitor service health:
```bash
docker-compose ps
```

View service logs:
```bash
docker-compose logs -f market-monitor
docker-compose logs -f trade-executor
docker-compose logs -f ai-analyzer
```

## Directory Structure

```
├── services/
│   ├── market_monitor_service.py
│   ├── trade_executor_service.py
│   └── ai_analyzer_service.py
├── config.json
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env
```

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run services individually:
```bash
python3 services/market_monitor_service.py
python3 services/trade_executor_service.py
python3 services/ai_analyzer_service.py
```

## Testing

Run tests:
```bash
python3 -m pytest tests/
```

## Contributing

1. Check the BACKLOG.md for planned improvements
2. Create a feature branch
3. Submit a pull request

## Security

- API keys are stored in environment variables
- Inter-service communication is done through Redis
- Services run in isolated containers
- Regular security updates via Docker

## Backup & Recovery

1. Redis data is persisted to a Docker volume
2. Logs are stored in the `logs` directory
3. Trading data is stored in the `data` directory

## Troubleshooting

Common issues:

1. Connection errors:
   - Check Binance API credentials
   - Verify Redis connection
   - Check network connectivity

2. Trading errors:
   - Verify sufficient balance
   - Check trading rules compliance
   - Review log files

## License

MIT License - see LICENSE file

## Support

For issues and feature requests:
1. Check existing issues
2. Review BACKLOG.md
3. Submit detailed bug reports

## Future Improvements

See BACKLOG.md for planned enhancements and features.
