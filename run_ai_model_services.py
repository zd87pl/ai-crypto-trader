#!/usr/bin/env python3
import os
import sys
import asyncio
import argparse
import logging as logger
from typing import Dict, List
from datetime import datetime

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logger.FileHandler(f'logs/ai_model_services_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logger.StreamHandler(sys.stdout)
    ]
)

# Import services
try:
    from services.model_registry_service import ModelRegistryService
    from services.ai_explainability_service import AIExplainabilityService
except ImportError as e:
    logger.error(f"Error importing services: {str(e)}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

async def run_services(args):
    """Run the AI model services"""
    services = []
    tasks = []

    try:
        # Start model registry if requested
        if args.model_registry:
            logger.info("Starting Model Registry Service...")
            model_registry = ModelRegistryService()
            services.append(model_registry)
            tasks.append(asyncio.create_task(model_registry.run()))
            
        # Start AI explainability service if requested
        if args.explainability:
            logger.info("Starting AI Explainability Service...")
            ai_explainability = AIExplainabilityService()
            services.append(ai_explainability)
            tasks.append(asyncio.create_task(ai_explainability.run()))
            
        if not tasks:
            logger.error("No services specified to run. Use --model-registry or --explainability")
            return
            
        # Run all services
        logger.info(f"Running {len(tasks)} services...")
        await asyncio.gather(*tasks)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down services...")
        
    except Exception as e:
        logger.error(f"Error running services: {str(e)}")
        
    finally:
        # Stop all services
        for service in services:
            try:
                await service.stop()
            except Exception as e:
                logger.error(f"Error stopping service: {str(e)}")
                
        logger.info("All services stopped")

def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='AI Model Services CLI')
    
    parser.add_argument('--model-registry', action='store_true', help='Run the Model Registry Service')
    parser.add_argument('--explainability', action='store_true', help='Run the AI Explainability Service')
    parser.add_argument('--all', action='store_true', help='Run all services')
    
    return parser

def main():
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # If --all is specified, enable all services
    if args.all:
        args.model_registry = True
        args.explainability = True
    
    # Run the services
    try:
        asyncio.run(run_services(args))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())