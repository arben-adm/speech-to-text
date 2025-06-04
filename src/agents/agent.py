import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable

from agents.mcp_integration import get_mcp_integration
from utils.logger import get_logger

logger = get_logger(__name__)

class Tool:
    """Base class for all local tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments"""
        raise NotImplementedError("Subclasses must implement execute method")


class Agent:
    """
    AI Agent that can use both local tools and MCP server tools.
    Maintains compatibility with speech-to-text functionality.
    """
    
    def __init__(
        self,
        name: str,
        system: str,
        tools: List[Tool] = None,
        mcp_servers: List[Dict[str, Any]] = None,
        mcp_config_path: str = "mcp_config.json"
    ):
        """
        Initialize AI Agent
        
        Args:
            name: Agent name
            system: System prompt for the agent
            tools: List of local tools
            mcp_servers: List of MCP server configurations to add
            mcp_config_path: Path to MCP configuration file
        """
        self.name = name
        self.system = system
        self.local_tools = tools or []
        self.mcp_integration = get_mcp_integration(mcp_config_path)
        
        # Add MCP servers if provided
        if mcp_servers:
            for server_config in mcp_servers:
                server_type = server_config.get("type", "")
                if server_type == "stdio":
                    self._add_stdio_mcp_server(server_config)
        
        # Dictionary to track available MCP tools
        self.mcp_tools = {}
        
    def _add_stdio_mcp_server(self, server_config: Dict[str, Any]) -> None:
        """
        Add a stdio-based MCP server
        
        Args:
            server_config: Server configuration dictionary
        """
        server_name = server_config.get("name", f"mcp-server-{len(self.mcp_integration.get_available_servers()) + 1}")
        
        # Create MCP server configuration
        mcp_config = {
            "command": server_config.get("command", ""),
            "args": server_config.get("args", []),
            "env": server_config.get("env", {})
        }
        
        # Add server to MCP integration
        self.mcp_integration.add_server(server_name, mcp_config)
        
    def connect(self) -> None:
        """
        Connect to all configured MCP servers and discover available tools
        """
        # Connect to MCP servers
        self.mcp_integration.connect()
        
        # Get available servers
        servers = self.mcp_integration.get_available_servers()
        
        # Discover tools from each server
        for server in servers:
            tools = self.mcp_integration.get_available_tools(server)
            for tool in tools:
                tool_id = f"{server}:{tool['name']}"
                self.mcp_tools[tool_id] = {
                    "server": server,
                    "name": tool["name"],
                    "description": tool.get("description", "")
                }
                
        logger.info(f"Connected to {len(servers)} MCP servers with {len(self.mcp_tools)} available tools")
        
    def disconnect(self) -> None:
        """
        Disconnect from all MCP servers
        """
        self.mcp_integration.disconnect()
        self.mcp_tools = {}
        
    def get_available_tools(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available tools (both local and MCP)
        
        Returns:
            Dictionary of tool information
        """
        tools = {}
        
        # Add local tools
        for tool in self.local_tools:
            tools[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "type": "local"
            }
            
        # Add MCP tools
        for tool_id, tool_info in self.mcp_tools.items():
            tools[tool_id] = {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "type": "mcp",
                "server": tool_info["server"]
            }
            
        return tools
        
    async def execute_tool(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool (local or MCP) with given arguments
        
        Args:
            tool_id: Tool identifier (tool name for local tools, server:tool for MCP tools)
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        # Check if it's a local tool
        for tool in self.local_tools:
            if tool.name == tool_id:
                try:
                    return await tool.execute(arguments)
                except Exception as e:
                    logger.error(f"Error executing local tool {tool_id}: {str(e)}")
                    return {
                        "error": f"Error executing tool: {str(e)}",
                        "isError": True
                    }
        
        # Check if it's an MCP tool
        if ":" in tool_id:
            server_name, tool_name = tool_id.split(":", 1)
            if server_name in self.mcp_integration.get_available_servers():
                try:
                    return self.mcp_integration.call_tool(server_name, tool_name, arguments)
                except Exception as e:
                    logger.error(f"Error executing MCP tool {tool_id}: {str(e)}")
                    return {
                        "error": f"Error executing MCP tool: {str(e)}",
                        "isError": True
                    }
        
        return {
            "error": f"Tool not found: {tool_id}",
            "isError": True
        }
        
    async def process(self, input_text: str, callback: Optional[Callable[[str], Awaitable[None]]] = None) -> str:
        """
        Process input text using AI model and available tools
        
        Args:
            input_text: User input text
            callback: Optional callback function for streaming responses
            
        Returns:
            Agent response
        """
        from api_providers.provider_factory import ProviderFactory
        from prompts import PromptTemplate
        import os
        import json
        import time
        
        try:
            # Get provider information from the connected text processor
            # This assumes SpeechAgent has already set up a text_processor attribute
            if not hasattr(self, 'text_processor'):
                error_msg = "Agent not properly initialized with a text processor"
                logger.error(error_msg)
                return error_msg
            
            provider = self.text_processor.provider
            
            # Get available tools for the agent
            tools = self.get_available_tools()
            
            # Format tools for inclusion in the prompt
            tools_description = ""
            for tool_id, tool_info in tools.items():
                tool_type = tool_info.get("type", "unknown")
                server = tool_info.get("server", "local") if tool_type == "mcp" else "local"
                tools_description += f"- {tool_info['name']} ({server}): {tool_info['description']}\n"
            
            # Create a special prompt for the agent
            agent_prompt = PromptTemplate(
                name="agent_prompt",
                description="AI Agent with tool access",
                system_prompt=f"""You are {self.name}, an AI assistant with access to various tools.

SYSTEM INSTRUCTIONS:
{self.system}

AVAILABLE TOOLS:
{tools_description}

To use a tool, respond with JSON in the following format:
```json
{{
  "tool": "tool_name",
  "args": {{
    "arg1": "value1",
    "arg2": "value2"
  }}
}}
```

If you don't need to use a tool, simply respond with normal text.
"""
            )
            
            # Process the input using the text processor
            current_response = ""
            message_history = [
                {"role": "system", "content": agent_prompt.system_prompt},
                {"role": "user", "content": input_text}
            ]
            
            # Send to AI provider
            model = None  # Use default model
            if hasattr(self, 'chat_model'):
                model = self.chat_model
            
            # Start with a simple response
            raw_response = self.text_processor.process_text(
                text=input_text,
                prompt_template=agent_prompt,
                model=model
            )
            
            if not raw_response:
                error_msg = "Failed to get a response from the AI provider"
                logger.error(error_msg)
                if callback:
                    await callback(error_msg)
                return error_msg
            
            # Check if the response contains a tool call (JSON format)
            tool_call = None
            try:
                # Look for JSON block in the response
                if "```json" in raw_response:
                    json_part = raw_response.split("```json")[1].split("```")[0].strip()
                    tool_call = json.loads(json_part)
                elif raw_response.strip().startswith("{") and raw_response.strip().endswith("}"):
                    tool_call = json.loads(raw_response.strip())
            except (json.JSONDecodeError, IndexError):
                # Not a tool call, just a regular response
                tool_call = None
            
            # If it's a tool call, execute it and send result back to AI
            if tool_call and "tool" in tool_call and "args" in tool_call:
                tool_id = tool_call["tool"]
                tool_args = tool_call["args"]
                
                # If it's a streaming callback, update the user
                if callback:
                    await callback(f"Executing tool: {tool_id}...")
                
                # Execute the tool
                tool_result = await self.execute_tool(tool_id, tool_args)
                
                # Append tool call and result to message history
                message_history.append({
                    "role": "assistant", 
                    "content": raw_response
                })
                
                result_content = "Tool execution result:\n"
                if tool_result.get("isError", False):
                    result_content += f"ERROR: {tool_result.get('error', 'Unknown error')}"
                else:
                    result_content += str(tool_result.get("result", tool_result))
                
                message_history.append({
                    "role": "user", 
                    "content": result_content
                })
                
                # Get AI response to the tool result
                final_response = self.text_processor.process_text(
                    text=result_content,
                    prompt_template=agent_prompt,
                    model=model
                )
                
                # Stream the final response if callback provided
                if callback:
                    await callback(final_response)
                
                return final_response
            else:
                # No tool call, just return the raw response
                if callback:
                    await callback(raw_response)
                
                return raw_response
                
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            logger.error(error_msg)
            if callback:
                await callback(error_msg)
            return error_msg