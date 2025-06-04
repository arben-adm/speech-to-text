"""
Example script demonstrating how to use the Agent class with MCP servers
"""
import asyncio
import json
from typing import Dict, List, Any

from agents.agent import Agent
from agents.tools.think import ThinkTool
from utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    """
    Example of using the Agent class with MCP servers
    """
    # Create an agent with both local tools and MCP server tools
    agent = Agent(
        name="SpeechToTextAssistant",
        system="You are a helpful assistant that can analyze audio transcriptions and provide insights.",
        tools=[ThinkTool()],  # Local tools
        mcp_servers=[
            {
                "type": "stdio",
                "name": "sequential-thinking",  # Name for this server
                "command": "uv",
                "args": [
                    "--directory",
                    "C:\\Users\\arben\\Desktop\\Github\\mcp-python\\mcp-sequential-thinking\\mcp_sequential_thinking",
                    "run",
                    "server.py"
                ]
            },
            {
                "type": "stdio",
                "name": "brave-search",  # Name for this server
                "command": "uv",
                "args": [
                    "--directory",
                    "C:\\Users\\arben\\Desktop\\Github\\mcp-python\\brave-mcp-search\\src",
                    "run",
                    "server.py"
                ],
                "env": {
                    "BRAVE_API_KEY": "YOUR_BRAVE_API_KEY_HERE"
                }
            }
        ]
    )
    
    # Connect to all MCP servers
    agent.connect()
    
    try:
        # List all available tools
        tools = agent.get_available_tools()
        logger.info(f"Available tools: {json.dumps(tools, indent=2)}")
        
        # Execute a local tool
        result = await agent.execute_tool("think", {"thought": "Let's analyze this audio transcription"})
        logger.info(f"Local tool result: {result}")
        
        # Execute an MCP tool (if available)
        if "sequential-thinking:think" in tools:
            result = await agent.execute_tool("sequential-thinking:think", {
                "thought": "How can I analyze this audio transcript step by step?"
            })
            logger.info(f"MCP tool result: {result}")
            
        # Process a user query
        response = await agent.process("Can you analyze this transcript for key topics?")
        logger.info(f"Agent response: {response}")
        
    finally:
        # Disconnect from MCP servers
        agent.disconnect()

# Run the example asynchronously
if __name__ == "__main__":
    asyncio.run(main())