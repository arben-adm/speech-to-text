"""
Unit tests for the Agent class
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from src.agents.agent import Agent, Tool

class TestAgent:
    """Test cases for the Agent class"""
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing"""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "mock_tool"
        mock_tool.description = "A mock tool for testing"
        mock_tool.execute = AsyncMock(return_value={"result": "Tool executed", "isError": False})
        return mock_tool
    
    @pytest.fixture
    def mock_mcp_integration(self):
        """Create a mock MCP integration for testing"""
        with patch('src.agents.agent.get_mcp_integration') as mock_get_integration:
            mock_integration = MagicMock()
            mock_integration.get_available_servers.return_value = ["test_server"]
            mock_integration.get_available_tools.return_value = [
                {"name": "test_tool", "description": "A test MCP tool"}
            ]
            mock_integration.call_tool.return_value = {"result": "MCP tool executed", "isError": False}
            
            mock_get_integration.return_value = mock_integration
            yield mock_integration
    
    def test_initialization(self, mock_tool, mock_mcp_integration):
        """Test that the agent initializes correctly"""
        # Arrange & Act
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool],
            mcp_servers=[]
        )
        
        # Assert
        assert agent.name == "TestAgent"
        assert agent.system == "Test system prompt"
        assert len(agent.local_tools) == 1
        assert agent.local_tools[0] == mock_tool
        assert agent.mcp_integration is not None
    
    def test_get_available_tools(self, mock_tool, mock_mcp_integration):
        """Test getting available tools"""
        # Arrange
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool]
        )
        
        # Populate the mcp_tools dictionary
        agent.mcp_tools = {
            "server:tool": {
                "server": "server",
                "name": "tool",
                "description": "An MCP tool"
            }
        }
        
        # Act
        tools = agent.get_available_tools()
        
        # Assert
        assert len(tools) == 2
        assert "mock_tool" in tools
        assert "server:tool" in tools
        assert tools["mock_tool"]["type"] == "local"
        assert tools["server:tool"]["type"] == "mcp"
    
    @pytest.mark.asyncio
    async def test_execute_local_tool(self, mock_tool, mock_mcp_integration):
        """Test executing a local tool"""
        # Arrange
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool]
        )
        
        # Act
        result = await agent.execute_tool("mock_tool", {"arg": "value"})
        
        # Assert
        assert result == {"result": "Tool executed", "isError": False}
        mock_tool.execute.assert_called_once_with({"arg": "value"})
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, mock_tool, mock_mcp_integration):
        """Test executing an MCP tool"""
        # Arrange
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool]
        )
        
        # Setup MCP tool
        agent.mcp_tools = {
            "test_server:test_tool": {
                "server": "test_server",
                "name": "test_tool",
                "description": "A test MCP tool"
            }
        }
        
        # Act
        result = await agent.execute_tool("test_server:test_tool", {"arg": "value"})
        
        # Assert
        assert result == {"result": "MCP tool executed", "isError": False}
        mock_mcp_integration.call_tool.assert_called_once_with("test_server", "test_tool", {"arg": "value"})
    
    @pytest.mark.asyncio
    async def test_process_method(self, mock_tool, mock_mcp_integration):
        """Test the process method with the new implementation"""
        # Arrange
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool]
        )
        
        # Mock the text processor
        agent.text_processor = MagicMock()
        agent.text_processor.provider = "groq"
        agent.text_processor.process_text.return_value = "Response from the AI model"
        
        # Act
        response = await agent.process("Hello, agent!")
        
        # Assert
        assert response == "Response from the AI model"
        agent.text_processor.process_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_with_tool_call(self, mock_tool, mock_mcp_integration):
        """Test the process method with a tool call"""
        # Arrange
        agent = Agent(
            name="TestAgent",
            system="Test system prompt",
            tools=[mock_tool]
        )
        
        # Mock the text processor
        agent.text_processor = MagicMock()
        agent.text_processor.provider = "groq"
        # First response is a tool call, second is the final response
        agent.text_processor.process_text.side_effect = [
            '```json\n{"tool": "mock_tool", "args": {"arg": "value"}}\n```',
            "Final response after tool execution"
        ]
        
        # Act
        response = await agent.process("Hello, use a tool!")
        
        # Assert
        assert response == "Final response after tool execution"
        assert agent.text_processor.process_text.call_count == 2
        mock_tool.execute.assert_called_once_with({"arg": "value"})