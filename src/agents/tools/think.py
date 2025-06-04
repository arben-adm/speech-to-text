from typing import Dict, Any
from agents.agent import Tool

class ThinkTool(Tool):
    """
    Tool that allows the agent to think through a problem step by step.
    This is a local tool that doesn't require external API calls.
    """
    
    def __init__(self):
        super().__init__(
            name="think",
            description="Think through a problem step by step"
        )
        
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the thinking process.
        
        Args:
            args: Dictionary containing the 'thought' key
            
        Returns:
            Dictionary with the result of thinking
        """
        thought = args.get("thought", "")
        
        # The thought is simply returned as the result
        # This tool is mostly for the agent to articulate its reasoning
        return {
            "result": f"I thought about: {thought}",
            "isError": False
        }