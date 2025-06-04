# MCP Server Integration for Speech-to-Text

This module provides integration between the speech-to-text functionality and MCP (Model Context Protocol) servers.

## Overview

The MCP (Model Context Protocol) is an open standard that enables AI assistants to connect with data systems, including content repositories, business tools, and development environments. This implementation allows you to easily integrate any MCP server with the speech-to-text application.

## Features

- Create AI agents that combine speech-to-text functionality with tools from MCP servers
- Easy configuration of MCP servers via JSON
- Loosely coupled architecture that allows adding tools without modifying existing code
- Local tools for transcription and text processing

## Usage

### Basic Agent Usage

```python
from src.agents.agent import Agent
from src.agents.tools.think import ThinkTool

# Create an agent with both local tools and MCP server tools
agent = Agent(
    name="MyAgent",
    system="You are a helpful assistant.",
    tools=[ThinkTool()],  # Local tools
    mcp_servers=[
        {
            "type": "stdio",
            "name": "sequential-thinking",
            "command": "python",
            "args": ["-m", "mcp_server"],
        },
    ]
)

# Connect to all MCP servers
agent.connect()

# List all available tools (both local and MCP)
tools = agent.get_available_tools()

# Execute a local tool
result = await agent.execute_tool("think", {"thought": "Analysis process"})

# Execute an MCP tool
result = await agent.execute_tool("sequential-thinking:think", {"thought": "Step-by-step analysis"})

# Process a user query
response = await agent.process("Analyze this data")

# Disconnect when done
agent.disconnect()
```

### Speech Agent Usage

The `SpeechAgent` class combines the speech-to-text functionality with MCP server tools:

```python
from src.agents.speech_agent import SpeechAgent

# Create a speech agent with MCP server integration
speech_agent = SpeechAgent(
    name="SpeechAssistant",
    system="You are a helpful assistant that can process and analyze audio transcriptions.",
    provider="openai",
    api_key="your-api-key",
    transcription_model="whisper-1",
    chat_model="gpt-4",
    mcp_servers=[
        {
            "type": "stdio",
            "name": "sequential-thinking",
            "command": "python",
            "args": ["-m", "mcp_sequential_thinking.server"],
        }
    ]
)

# Connect to all MCP servers
speech_agent.connect()

# Transcribe and process audio
result = await speech_agent.transcribe_and_process(
    audio_bytes=audio_data,
    transcription_model="whisper-1",
    chat_model="gpt-4",
    system_prompt="Analyze this transcript and highlight key points."
)

# Access results
original_text = result["original_text"]
processed_text = result["processed_text"]

# Disconnect when done
speech_agent.disconnect()
```

## MCP Server Configuration

MCP servers are configured in the `mcp_config.json` file. You can add, remove, or modify servers through the API or by editing this file directly.

Example configuration:

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\mcp-sequential-thinking\\",
        "run",
        "server.py"
      ]
    },
    "brave-search": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\brave-mcp-search\\src",
        "run",
        "server.py"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR_BRAVE_API_KEY_HERE"
      }
    }
  }
}
```

### Cross-Platform Path Handling

The MCP client now includes automatic path conversion to handle different platforms:

- Windows paths (`C:\path\to\file`) are automatically converted to WSL paths (`/mnt/c/path/to/file`) when running on WSL
- WSL paths (`/mnt/c/path`) are automatically converted to Windows paths (`C:\path`) when running on Windows

This ensures that your MCP server configurations work seamlessly across different platforms.
```

## Available Tools

### Local Tools

- `transcribe`: Transcribe audio files to text
- `process_text`: Process text using AI models
- `think`: Think through a problem step by step

### MCP Server Tools

Each MCP server can provide its own set of tools, which will be automatically discovered when connecting to the server.

## API Reference

### Agent Class

The base class for AI agents that can use both local tools and MCP server tools.

```python
Agent(
    name: str,
    system: str,
    tools: List[Tool] = None,
    mcp_servers: List[Dict[str, Any]] = None,
    mcp_config_path: str = "mcp_config.json"
)
```

#### Key Methods

- `connect()`: Connect to all configured MCP servers
- `disconnect()`: Disconnect from all MCP servers
- `get_available_tools()`: Get all available tools (both local and MCP)
- `execute_tool(tool_id, arguments)`: Execute a specific tool
- `process(input_text, callback=None)`: Process input text using AI model and available tools

### SpeechAgent Class

Extension of the Agent class that provides integration with speech-to-text functionality.

```python
SpeechAgent(
    name: str,
    system: str,
    provider: str,
    api_key: str,
    transcription_model: str,
    chat_model: str,
    mcp_servers: List[Dict[str, Any]] = None,
    mcp_config_path: str = "mcp_config.json"
)
```

#### Key Methods

In addition to all methods from the Agent class, SpeechAgent provides:

- `transcribe_and_process(audio_bytes, transcription_model, chat_model, system_prompt, callback=None)`: Transcribe audio and process the resulting text

### MCPServerIntegration Class

Handles the integration with MCP servers.

```python
MCPServerIntegration(config_path: str = "mcp_config.json")
```

#### Key Methods

- `add_server(server_name, server_config)`: Add a new MCP server configuration
- `remove_server(server_name)`: Remove an MCP server configuration
- `connect()`: Connect to all configured MCP servers
- `disconnect()`: Disconnect from all MCP servers
- `get_available_servers()`: Get a list of all connected servers
- `get_available_tools(server_name)`: Get available tools from a specific server
- `call_tool(server_name, tool_name, args)`: Call a specific tool on a server

### MCPClient Class

Low-level client for interacting with MCP servers.

```python
# Get the singleton instance
from mcp_client import get_mcp_client
client = get_mcp_client(config_path="mcp_config.json")
```

#### Key Methods

- `load_config()`: Load MCP server configuration from the JSON file
- `connect_to_servers()`: Connect to all configured MCP servers
- `connect_to_server(server_name, server_config)`: Connect to a specific MCP server
- `list_servers()`: List all connected servers
- `list_tools(server_name)`: List all available tools on a server
- `call_tool(server_name, tool_name, arguments)`: Call a tool on a server
- `close()`: Close all connections and free resources

## Examples

See the `src/agents/example.py` file for a complete example of using the Agent class with MCP servers.

## Troubleshooting

### MCP Server Connection Issues

If you encounter problems connecting to MCP servers:

1. **Check Server Status**: Ensure the MCP server is running and accessible
2. **Check Configuration**: Verify the server command and arguments in `mcp_config.json`
3. **Path Issues**: Ensure paths in your configuration are valid for your platform
4. **Environment Variables**: Make sure required environment variables are set correctly
5. **Permissions**: Check that the MCP server has the necessary permissions to execute

### Common Error Messages

- `Error connecting to server <name>`: The server couldn't be started or connected to
- `Server <name> not connected`: Trying to use a server that isn't connected
- `Tool not found: <tool_name>`: The specified tool doesn't exist on the server
- `Error executing tool: <error>`: An error occurred while executing the tool

### Debugging Tips

1. Use `print(f"Server {server_name} connected with tools: {tools}")` to see available tools
2. Check the server's output for error messages
3. Use the Streamlit UI's MCP Tools tab to test server connections and tools manually
4. If running on WSL or across platforms, check the path conversion is working correctly