import asyncio
import json
import os
from typing import Dict, List, Optional, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, config_path: str = "mcp_config.json"):
        """
        Initializes the MCP client with configuration from a JSON file
        
        Args:
            config_path: Path to the MCP configuration file
        """
        self.config_path = config_path
        self.servers = {}
        self.sessions = {}
        self.exit_stack = None
        self.load_config()
        
    def load_config(self) -> None:
        """Loads the MCP server configuration from the JSON file"""
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump({"mcpServers": {}}, f, indent=2)
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.servers = config.get("mcpServers", {})
    
    async def connect_to_servers(self) -> None:
        """Connects to all configured MCP servers"""
        from contextlib import AsyncExitStack
        
        self.exit_stack = AsyncExitStack()
        for server_name, server_config in self.servers.items():
            try:
                await self.connect_to_server(server_name, server_config)
            except Exception as e:
                print(f"Error connecting to server {server_name}: {str(e)}")
    
    async def connect_to_server(self, server_name: str, server_config: Dict) -> None:
        """
        Connects to a single MCP server
        
        Args:
            server_name: Name of the server
            server_config: Configuration of the server
        """
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        # Handle paths across different platforms
        import platform
        import re
        import os
        
        modified_args = []
        system = platform.system().lower()
        
        for arg in args:
            if not isinstance(arg, str):
                modified_args.append(arg)
                continue
                
            # Handle Windows paths on WSL
            if system == 'linux' and 'microsoft' in platform.release().lower():
                # Convert Windows paths (C:\path\to\file) to WSL paths (/mnt/c/path/to/file)
                if re.match(r'^[A-Za-z]:\\', arg):
                    drive_letter = arg[0].lower()
                    wsl_path = f"/mnt/{drive_letter}/{arg[3:].replace('\\', '/')}"
                    modified_args.append(wsl_path)
                else:
                    modified_args.append(arg)
                    
            # Handle WSL paths on Windows
            elif system == 'windows':
                # Convert WSL paths (/mnt/c/path) to Windows paths (C:\path)
                wsl_path_match = re.match(r'^/mnt/([a-z])/(.*)', arg)
                if wsl_path_match:
                    drive_letter = wsl_path_match.group(1).upper()
                    path_part = wsl_path_match.group(2).replace('/', '\\')
                    win_path = f"{drive_letter}:\\{path_part}"
                    modified_args.append(win_path)
                else:
                    modified_args.append(arg)
            
            # No conversion needed for other cases
            else:
                modified_args.append(arg)
        
        server_params = StdioServerParameters(
            command=command,
            args=modified_args,
            env=env or None
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdin, stdout = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdin, stdout))
            
            # Server initialisieren
            await session.initialize()
            
            # Session speichern
            self.sessions[server_name] = session
            
            # List available tools and resources
            tools_response = await session.list_tools()
            print(f"Server {server_name} connected with tools: {[tool.name for tool in tools_response.tools]}")
            
        except Exception as e:
            print(f"Error connecting to {server_name}: {str(e)}")
            # Show full error for debugging
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            raise
    
    async def list_servers(self) -> List[str]:
        """List of all connected servers"""
        return list(self.sessions.keys())
    
    async def list_tools(self, server_name: str) -> List[Dict]:
        """
        Lists all available tools from a server
        
        Args:
            server_name: Name of the server
        
        Returns:
            List of tool information
        """
        if server_name not in self.sessions:
            return []
            
        session = self.sessions[server_name]
        response = await session.list_tools()
        
        return [{
            "name": tool.name,
            "description": tool.description,
            "server": server_name
        } for tool in response.tools]
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict:
        """
        Calls a tool on a server
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Arguments for the tool
        
        Returns:
            Result of the tool call
        """
        if server_name not in self.sessions:
            return {"error": f"Server {server_name} not connected"}
        
        session = self.sessions[server_name]
        try:
            result = await session.call_tool(tool_name, arguments)
            return {
                "content": result.content,
                "isError": result.isError
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self) -> None:
        """Closes all connections and frees resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.sessions = {}

# Singleton-Instanz
_mcp_client_instance = None

def get_mcp_client(config_path: str = "mcp_config.json") -> MCPClient:
    """
    Returns a singleton instance of the MCP client
    
    Args:
        config_path: Path to the MCP configuration file
    
    Returns:
        MCPClient instance
    """
    global _mcp_client_instance
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPClient(config_path)
    return _mcp_client_instance

# Asynchronous helper functions for Streamlit
def run_async(coroutine):
    """Executes a coroutine in an asyncio event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()