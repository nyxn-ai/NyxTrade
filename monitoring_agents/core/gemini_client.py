"""
Gemini AI client for intelligent analysis and insights
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .config_manager import ConfigManager


class GeminiClient:
    """
    Client for interacting with Google's Gemini AI
    Provides intelligent analysis and insights for monitoring agents
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger("gemini_client")
        
        # Load configuration
        self.config = self.config_manager.get_gemini_config()
        
        # Initialize Gemini
        self._initialize_gemini()
        
        # Response cache for optimization
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        self.logger.info("Initialized Gemini client")
    
    def _initialize_gemini(self):
        """Initialize Gemini AI with configuration"""
        try:
            # Configure API key
            genai.configure(api_key=self.config.get("api_key"))
            
            # Initialize model
            model_name = self.config.get("model", "gemini-pro")
            
            generation_config = {
                "temperature": self.config.get("temperature", 0.7),
                "top_p": self.config.get("top_p", 0.8),
                "top_k": self.config.get("top_k", 40),
                "max_output_tokens": self.config.get("max_tokens", 2048),
            }
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info(f"Initialized Gemini model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze data using Gemini AI
        
        Args:
            prompt: Analysis prompt
            context: Additional context information
            
        Returns:
            Analysis results as dictionary
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(prompt, context)
            cached_result = self._get_cached_response(cache_key)
            if cached_result:
                self.logger.debug("Returning cached Gemini response")
                return cached_result
            
            # Prepare full prompt with context
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Generate response
            self.logger.debug("Sending request to Gemini")
            response = await self._generate_response(full_prompt)
            
            # Parse response
            parsed_response = self._parse_response(response)
            
            # Cache response
            self._cache_response(cache_key, parsed_response)
            
            self.logger.info("Gemini analysis completed successfully")
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            return {
                "error": str(e),
                "analysis": "Analysis failed due to AI service error",
                "confidence": 0.0,
                "recommendations": ["Unable to provide recommendations due to analysis failure"]
            }
    
    def _prepare_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare the full prompt with context and formatting instructions"""
        
        system_prompt = """
You are an expert cryptocurrency market analyst with deep knowledge of:
- Technical analysis and market indicators
- Blockchain and on-chain analytics
- Market psychology and sentiment analysis
- Risk management and trading strategies
- Macroeconomic factors affecting crypto markets

Please provide analysis in the following JSON format:
{
    "analysis": "Detailed analysis of the provided data",
    "key_insights": ["List of key insights"],
    "market_sentiment": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "recommendations": ["List of actionable recommendations"],
    "timeframe": "short/medium/long term outlook",
    "supporting_factors": ["Factors supporting the analysis"],
    "risk_factors": ["Potential risks and concerns"]
}

Current timestamp: {timestamp}
""".format(timestamp=datetime.now().isoformat())
        
        context_str = ""
        if context:
            context_str = f"\nContext Information:\n{json.dumps(context, indent=2)}\n"
        
        full_prompt = f"{system_prompt}{context_str}\nAnalysis Request:\n{prompt}"
        
        return full_prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response from Gemini model"""
        try:
            # Use asyncio to run the synchronous generate_content method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.model.generate_content, 
                prompt
            )
            
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            else:
                raise Exception("No valid response generated")
                
        except Exception as e:
            self.logger.error(f"Failed to generate Gemini response: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif response.startswith("{") and response.endswith("}"):
                json_str = response
            else:
                # If no JSON found, create structured response
                return {
                    "analysis": response,
                    "key_insights": [],
                    "market_sentiment": "neutral",
                    "confidence": 0.5,
                    "risk_level": "medium",
                    "recommendations": ["Review the analysis for actionable insights"],
                    "timeframe": "medium",
                    "supporting_factors": [],
                    "risk_factors": []
                }
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["analysis", "confidence", "recommendations"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            return parsed
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return {
                "analysis": response,
                "error": "Failed to parse structured response",
                "confidence": 0.3,
                "recommendations": ["Manual review required due to parsing error"]
            }
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            "analysis": "Analysis not available",
            "key_insights": [],
            "market_sentiment": "neutral",
            "confidence": 0.5,
            "risk_level": "medium",
            "recommendations": [],
            "timeframe": "medium",
            "supporting_factors": [],
            "risk_factors": []
        }
        return defaults.get(field, "")
    
    def _get_cache_key(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for request"""
        import hashlib
        content = prompt + str(context or {})
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if datetime.now().timestamp() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future use"""
        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now().timestamp()
        }
        
        # Clean old cache entries
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now().timestamp()
        expired_keys = [
            key for key, data in self.response_cache.items()
            if current_time - data["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "cache_size": len(self.response_cache),
            "model_name": self.config.get("model", "gemini-pro"),
            "cache_ttl": self.cache_ttl
        }
