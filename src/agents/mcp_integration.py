import json
import os
from typing import Dict, List, Optional, Any, Union

from mcp_client import get_mcp_client, run_async

class MCPServerIntegration:
    """
    Integration class for MCP servers.
    Provides an easy way to add and use MCP servers in the project.
    """
    def __init__(self, config_path: str = "mcp_config.json"):
        """
        Initialize MCP server integration
        
        Args:
            config_path: Path to the MCP configuration file
        """
        self.config_path = config_path
        self.mcp_client = get_mcp_client(config_path)
        self.connected = False
        
    def add_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        """
        Add a new MCP server to the configuration
        
        Args:
            server_name: Name of the server
            server_config: Server configuration dict with keys:
                - command: Command to start the server
                - args: List of arguments
                - env: Optional environment variables
        """
        # Load current configuration
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}
            
        # Add or update server configuration
        config["mcpServers"][server_name] = server_config
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Reload the configuration in the client
        self.mcp_client.load_config()
    
    def remove_server(self, server_name: str) -> bool:
        """
        Remove an MCP server from the configuration
        
        Args:
            server_name: Name of the server to remove
            
        Returns:
            True if server was removed, False if not found
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            if server_name in config.get("mcpServers", {}):
                del config["mcpServers"][server_name]
                
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                # Reload the configuration in the client
                self.mcp_client.load_config()
                return True
                
        return False
    
    def connect(self) -> None:
        """Connect to all configured MCP servers"""
        run_async(self.mcp_client.connect_to_servers())
        self.connected = True
    
    def disconnect(self) -> None:
        """Disconnect from all MCP servers"""
        if self.connected:
            run_async(self.mcp_client.close())
            self.connected = False
    
    def get_available_servers(self) -> List[str]:
        """
        Get a list of available MCP servers
        
        Returns:
            List of server names
        """
        if not self.connected:
            return []
            
        return run_async(self.mcp_client.list_servers())
    
    def get_available_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Get available tools from a specific server
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of tool information dictionaries
        """
        if not self.connected:
            return []
            
        return run_async(self.mcp_client.list_tools(server_name))
    
    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool call
        """
        if not self.connected:
            return {"error": "Not connected to MCP servers", "isError": True}
            
        return run_async(self.mcp_client.call_tool(server_name, tool_name, arguments))


# Singleton instance
_mcp_integration_instance = None

def get_mcp_integration(config_path: str = "mcp_config.json") -> MCPServerIntegration:
    """
    Get a singleton instance of the MCP server integration
    
    Args:
        config_path: Path to the MCP configuration file
        
    Returns:
        MCPServerIntegration instance
    """
    global _mcp_integration_instance
    if _mcp_integration_instance is None:
        _mcp_integration_instance = MCPServerIntegration(config_path)
    return _mcp_integration_instance