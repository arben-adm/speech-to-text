"""
Unit tests for the MCP client
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
import tempfile
from src.mcp_client import MCPClient, get_mcp_client, run_async

class TestMCPClient:
    """Test cases for the MCPClient class"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing"""
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "python",
                    "args": ["-m", "test_server"],
                    "env": {"TEST_KEY": "test-value"}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
            
        yield temp_file_path
        
        # Clean up the file after the test
        os.unlink(temp_file_path)
    
    def test_initialization_and_load_config(self, temp_config_file):
        """Test that the client initializes and loads config correctly"""
        # Arrange & Act
        client = MCPClient(config_path=temp_config_file)
        
        # Assert
        assert "test-server" in client.servers
        assert client.servers["test-server"]["command"] == "python"
        assert client.servers["test-server"]["args"] == ["-m", "test_server"]
        assert client.servers["test-server"]["env"] == {"TEST_KEY": "test-value"}
    
    def test_singleton_pattern(self, temp_config_file):
        """Test that get_mcp_client returns a singleton instance"""
        # Arrange & Act
        client1 = get_mcp_client(config_path=temp_config_file)
        client2 = get_mcp_client(config_path=temp_config_file)
        
        # Assert
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_connect_to_server(self, temp_config_file):
        """Test connecting to a server"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        
        # Mock the exit stack and stdio client
        with patch('src.mcp_client.AsyncExitStack') as MockExitStack, \
             patch('src.mcp_client.stdio_client') as mock_stdio_client, \
             patch('src.mcp_client.ClientSession') as MockClientSession:
            
            # Setup mocks
            mock_exit_stack = MagicMock()
            MockExitStack.return_value = mock_exit_stack
            
            mock_transport = (MagicMock(), MagicMock())  # stdin, stdout
            mock_exit_stack.enter_async_context.side_effect = [
                mock_transport,
                MagicMock()  # session
            ]
            
            mock_session = MagicMock()
            mock_session.initialize = AsyncMock()
            mock_session.list_tools = AsyncMock()
            mock_session.list_tools.return_value.tools = [
                MagicMock(name="tool1"),
                MagicMock(name="tool2")
            ]
            
            MockClientSession.return_value = mock_session
            
            # Act
            client.exit_stack = mock_exit_stack
            await client.connect_to_server("test-server", client.servers["test-server"])
            
            # Assert
            assert "test-server" in client.sessions
            assert client.sessions["test-server"] == mock_session
            mock_session.initialize.assert_called_once()
            mock_session.list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_servers(self, temp_config_file):
        """Test listing servers"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        client.sessions = {
            "server1": MagicMock(),
            "server2": MagicMock()
        }
        
        # Act
        servers = await client.list_servers()
        
        # Assert
        assert set(servers) == {"server1", "server2"}
    
    @pytest.mark.asyncio
    async def test_list_tools(self, temp_config_file):
        """Test listing tools for a server"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        mock_session = MagicMock()
        tool1 = MagicMock(name="tool1", description="Tool 1 description")
        tool2 = MagicMock(name="tool2", description="Tool 2 description")
        mock_session.list_tools = AsyncMock()
        mock_session.list_tools.return_value.tools = [tool1, tool2]
        
        client.sessions = {"test-server": mock_session}
        
        # Act
        tools = await client.list_tools("test-server")
        
        # Assert
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"
        assert tools[0]["description"] == "Tool 1 description"
        assert tools[0]["server"] == "test-server"
        assert tools[1]["name"] == "tool2"
        assert tools[1]["description"] == "Tool 2 description"
        assert tools[1]["server"] == "test-server"
    
    @pytest.mark.asyncio
    async def test_call_tool(self, temp_config_file):
        """Test calling a tool on a server"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock()
        mock_session.call_tool.return_value.content = [{"type": "text", "text": "Tool result"}]
        mock_session.call_tool.return_value.isError = False
        
        client.sessions = {"test-server": mock_session}
        
        # Act
        result = await client.call_tool("test-server", "test-tool", {"arg": "value"})
        
        # Assert
        assert result["content"] == [{"type": "text", "text": "Tool result"}]
        assert result["isError"] is False
        mock_session.call_tool.assert_called_once_with("test-tool", {"arg": "value"})
    
    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self, temp_config_file):
        """Test calling a tool on a server that's not connected"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        client.sessions = {}
        
        # Act
        result = await client.call_tool("nonexistent-server", "test-tool", {"arg": "value"})
        
        # Assert
        assert "error" in result
        assert "not connected" in result["error"]
    
    @pytest.mark.asyncio
    async def test_close(self, temp_config_file):
        """Test closing all connections"""
        # Arrange
        client = MCPClient(config_path=temp_config_file)
        mock_exit_stack = MagicMock()
        mock_exit_stack.aclose = AsyncMock()
        client.exit_stack = mock_exit_stack
        client.sessions = {"server1": MagicMock()}
        
        # Act
        await client.close()
        
        # Assert
        mock_exit_stack.aclose.assert_called_once()
        assert client.sessions == {}