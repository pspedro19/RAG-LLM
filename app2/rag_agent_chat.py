#!/usr/bin/env python3
"""
Reflective RAG Agent Chat - Advanced conversational interface with RAG, LLMs, and structured reflection

This script enhances the existing RAG chat system with improved conversation handling and
specialized historical query processing for Curaçao tourism information.

Usage:
    python rag_agent_chat.py [--model openai|claude] [--top-k 5] [--temperature 0.2]
"""

import asyncio
import argparse
import logging
import os
import sys
import textwrap
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Literal, Annotated, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import re

# Load environment variables
load_dotenv()

# Add parent directory to path for importing modules
sys.path.insert(0, os.path.abspath('..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_agent_chat.log"),
    ]
)

logger = logging.getLogger("rag-agent-chat")

# Suppress low-priority logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("app2").setLevel(logging.WARNING)

# Import from existing RAG system
from app2.core.config.config import Config
from app2.core.pipelines.search_pipeline import SearchPipeline

# Import LLM APIs if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool, BaseTool
    
    # LangGraph imports
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode
    
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangChain/LangGraph not installed. Agent capabilities will be limited.")

# ANSI Colors for terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Check if terminal supports colors
def supports_color():
    """Verify if the terminal supports ANSI colors."""
    if os.name == 'nt':  # Windows
        return False
    
    if 'NO_COLOR' in os.environ:
        return False
    
    if not sys.stdout.isatty():
        return False
    
    return True

# Disable colors if no support
if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, "")

# ===== REFLECTION STRUCTURES =====
class RetrievalReflection(BaseModel):
    """Structured reflection on retrieved information."""
    relevance: str = Field(description="Assessment of how relevant the retrieved information is to the query")
    completeness: str = Field(description="Analysis of whether the retrieved information fully answers the query")
    reliability: str = Field(description="Evaluation of the reliability and accuracy of the retrieved information")
    gaps: str = Field(description="Identification of any missing information that would improve the answer")
    model_used: str = Field(description="Model used for retrieval and analysis")
    needs_reasoning: bool = Field(description="Whether additional reasoning is needed beyond the retrieved information")

class ReasoningReflection(BaseModel):
    """Structured reflection on the reasoning process."""
    assumptions: str = Field(description="Key assumptions made during reasoning")
    logic: str = Field(description="Assessment of the logical coherence of the reasoning")
    alternatives: str = Field(description="Alternative perspectives or approaches not considered")
    confidence: str = Field(description="Overall confidence in the reasoning and conclusions")

# ===== STATE MANAGEMENT =====
if LANGGRAPH_AVAILABLE:
    class EnhancedState(TypedDict):
        """State management for the agent workflow with support for conversation history and context tracking."""
        messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
        context: Dict[str, Any]  # Stores retrieved information, reflection notes, etc.
        rag_session: Optional[Any]  # Will store the RAG chat instance

# ===== CORE RAG CHAT CLASS =====
class ReflectiveRAGChat:
    """Enhanced conversational interface that integrates RAG with LLMs and structured reflection."""
    
    def __init__(
        self, 
        model: str = 'openai',
        top_k: int = 5,
        temperature: float = 0.2,
        use_localhost: bool = True,
        max_conversation_history: int = 10,
        use_agent: bool = True  # Whether to use LangGraph agent architecture
    ):
        """
        Initialize the reflective RAG chat.
        
        Args:
            model: Model to use ('openai' or 'claude')
            top_k: Number of results to return from RAG
            temperature: Controls randomness in LLM responses
            use_localhost: If True, use localhost instead of postgres_pgvector
            max_conversation_history: Maximum number of messages to retain in history
            use_agent: Whether to use LangGraph agent architecture
        """
        self.config = Config()
        
        # Use localhost for database if specified
        if use_localhost:
            logger.info("Using localhost for PostgreSQL connection")
            self.config.DB_HOST = "localhost"
        
        # Search configuration
        self.top_k = top_k
        self.model_name = model
        self.temperature = temperature
        self.max_conversation_history = max_conversation_history
        self.use_agent = use_agent and LANGGRAPH_AVAILABLE
        
        # Initialize components
        self.search_pipeline = None
        
        # Conversation history
        self.conversation = []
        self.last_query_topic = ""
        
        # Get terminal size
        self.term_width, self.term_height = shutil.get_terminal_size()
        
        # Validate API availability
        self.openai_available = self._check_openai_availability()
        self.claude_available = self._check_claude_availability()
        
        # Determine model to use based on availability
        self.active_model = self._determine_active_model()
        
        # Initialize LangGraph agent if enabled
        self.agent = None
        if self.use_agent:
            self._initialize_agent()
        
        logger.info(f"Reflective RAG Chat initialized (model: {self.active_model}, top-k: {top_k}, agent: {self.use_agent})")
    
    def _check_openai_availability(self) -> bool:
        """Check if OpenAI is available and configured."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not installed")
            return False
        
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return False
        
        # Initialize client
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return False
    
    def _check_claude_availability(self) -> bool:
        """Check if Claude is available and configured."""
        if not CLAUDE_AVAILABLE:
            logger.warning("Anthropic library not installed")
            return False
        
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            return False
        
        # Initialize client
        try:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Claude client: {e}")
            return False
    
    def _determine_active_model(self) -> str:
        """Determine which model to use based on availability."""
        # Check first preference
        if self.model_name == 'openai' and self.openai_available:
            return 'openai'
        elif self.model_name == 'claude' and self.claude_available:
            return 'claude'
        
        # If first preference is not available, try alternative
        if self.openai_available:
            return 'openai'
        elif self.claude_available:
            return 'claude'
        
        # If none are available, use RAG only
        logger.warning("No external LLM available. Using RAG-only mode.")
        return 'rag-only'
    
    def _initialize_agent(self):
        """Initialize the LangGraph agent if LangGraph is available."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, skipping agent initialization")
            return
        
        try:
            self.agent = ReflectiveAgent(rag_chat=self)
            logger.info("LangGraph agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LangGraph agent: {e}")
            self.use_agent = False
    
    async def initialize(self) -> bool:
        """Initialize the system components."""
        logger.info("Initializing components...")
        
        try:
            # Create search pipeline
            self.search_pipeline = SearchPipeline(config=self.config)
            logger.info("Components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    async def close(self):
        """Close connections and clean up resources."""
        if self.search_pipeline:
            await self.search_pipeline.close()
        logger.info("Reflective RAG Chat resources closed")
    
    def _is_history_query(self, query: str) -> bool:
        """Determine if a query is asking about history or historical information."""
        history_keywords = [
            "history", "historical", "heritage", "origin", "past", "colonization", 
            "timeline", "era", "background", "ancient", "story", "stories",
            "historia", "histórico", "patrimonio", "origen", "pasado", "colonización"
        ]
        
        query_lower = query.lower()
        
        # Check for direct history keywords
        if any(kw in query_lower for kw in history_keywords):
            return True
            
        # Check for implied historical questions
        if "when" in query_lower and any(w in query_lower for w in ["founded", "established", "discovered", "settled"]):
            return True
            
        # Check if it's a follow-up about history
        if ("tell me about" in query_lower or "what about" in query_lower) and self.last_query_topic == "history":
            return True
        
        return False
    
    async def _query_needs_rag(self, query: str) -> bool:
        """
        Determine if a query needs to use RAG or can be answered directly.
        
        Args:
            query: User query
            
        Returns:
            True if the query needs RAG, False if not
        """
        # Update the last query topic for context tracking
        if self._is_history_query(query):
            self.last_query_topic = "history"
            logger.info(f"Historical query detected: '{query}'")
            return True
        
        # Generic queries that don't need RAG
        generic_greetings = [
            "hola", "hello", "hi", "buenos días", "buenas tardes",
            "cómo estás", "how are you", "saludos", "qué tal",
            "adiós", "bye", "hasta luego", "gracias", "thank you",
        ]
        
        # Check if it's a simple greeting
        if query.lower().strip().strip("?!.,") in generic_greetings:
            logger.info(f"Generic query detected, won't use RAG: '{query}'")
            self.last_query_topic = "greeting"
            return False
        
        # Keywords indicating RAG is likely needed
        rag_keywords = [
            "curaçao", "curacao", "playas", "beaches", "atracciones", "attractions",
            "hotel", "restaurante", "restaurant", "actividades", "activities",
            "tourism", "turismo", "isla", "island", "caribe", "caribbean",
            "willemstad", "playa", "beach", "museo", "museum", "snorkel",
            "buceo", "diving", "historia", "history", "precio", "price"
        ]
        
        # Check if it contains keywords related to Curaçao
        for keyword in rag_keywords:
            if keyword.lower() in query.lower():
                logger.info(f"Domain-related query, will use RAG: '{query}'")
                # Extract main topic from query
                topic_words = [w for w in query.lower().split() if w in rag_keywords]
                self.last_query_topic = topic_words[0] if topic_words else "tourism"
                return True
        
        # For other messages, use model to decide, but simple for now
        if len(query.split()) > 5:  # If more than 5 words, could be a complex question
            return True
        
        # By default, don't use RAG
        logger.info(f"Uncategorized query, won't use RAG: '{query}'")
        self.last_query_topic = "general"
        return False
    
    async def _perform_rag_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a search in the RAG system with enhanced historical query handling.
        
        Args:
            query: Query to perform
            
        Returns:
            Search results or empty dictionary if it fails
        """
        if not self.search_pipeline:
            logger.error("Search pipeline not initialized")
            return {}
        
        # Check if this is a historical query
        is_historical = self._is_history_query(query)
        if is_historical:
            logger.info(f"Performing historical search for: '{query}'")
            # Add historical context to the query for better retrieval
            enhanced_query = f"history of Curaçao {query}"
        else:
            enhanced_query = query
        
        logger.info(f"Searching in RAG: '{enhanced_query}'")
        
        try:
            # Perform search
            results = await self.search_pipeline.search(
                query_text=enhanced_query,
                mode='hybrid',  # Combine semantic and keyword search
                top_k=self.top_k,
                strategy='relevance'
            )
            
            # Log search details for debugging
            chunks_found = len(results.get('chunks', []))
            logger.info(f"Search completed: {chunks_found} chunks found")
            
            # If historical query but no good results, try a more generic history search
            if is_historical and chunks_found < 2:
                logger.info("Few historical results found, trying broader historical search")
                backup_results = await self.search_pipeline.search(
                    query_text="history of Curaçao origins colonization heritage",
                    mode='hybrid', 
                    top_k=self.top_k,
                    strategy='relevance'
                )
                
                # If backup search gave more results, use it
                if len(backup_results.get('chunks', [])) > chunks_found:
                    logger.info(f"Using broader historical search results instead ({len(backup_results.get('chunks', []))} chunks)")
                    results = backup_results
            
            # Check if we got meaningful results for historical queries
            if is_historical and self._contains_historical_content(results):
                logger.info("Historical content confirmed in search results")
            elif is_historical:
                logger.warning("Query was about history but results may not contain historical content")
            
            return results
        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return {}
    
    def _contains_historical_content(self, results: Dict) -> bool:
        """
        Check if the search results contain historical content.
        
        Args:
            results: Search results from RAG
            
        Returns:
            True if historical content is present, False otherwise
        """
        if not results or not results.get('context'):
            return False
            
        context = results.get('context', '').lower()
        
        # Historical indicators in content
        historical_indicators = [
            'year', 'century', 'colonization', 'settlement', 'timeline', 
            'history', 'historical', 'founded', 'established', 'colonial',
            'dutch', 'spanish', 'heritage', 'unesco', 'slave', 'trade'
        ]
        
        # Check for date patterns (years)
        contains_dates = bool(re.search(r'\b1[5-9]\d\d\b|\b20\d\d\b', context))
        
        # Check for historical terms
        contains_historical_terms = any(indicator in context for indicator in historical_indicators)
        
        return contains_dates or contains_historical_terms
    
    def _create_prompt_with_rag_context(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a prompt for the LLM including RAG context with enhanced historical handling.
        
        Args:
            query: User query
            rag_results: RAG results
            
        Returns:
            Formatted prompt for the LLM
        """
        # Get context
        context = rag_results.get('context', "")
        
        # Check if this is a historical query
        is_historical = self._is_history_query(query)
        
        # Create custom instructions for historical queries
        history_instructions = ""
        if is_historical:
            history_instructions = """
            This query is about the history of Curaçao. When answering:
            1. Organize historical information chronologically when possible
            2. Highlight key events, periods, and historical figures
            3. Include dates for important historical events
            4. Explain historical context and significance
            5. If the information is incomplete, acknowledge limitations
            
            Present historical information in a structured, educational format.
            """
        
        # Create message system for history
        if self.active_model == 'openai':
            # Format for OpenAI
            messages = []
            
            # System message
            system_msg = f"""
            You are a reflective tourism assistant specializing in Curaçao. Use the provided information
            to answer user questions. If the information is not in the context,
            indicate that you don't have sufficient information. Respond naturally and conversationally,
            without mentioning technical terms like "chunks", "context", "vectors", etc.
            
            For each response:
            1. First assess the relevance and completeness of the information you have
            2. Consider whether reasoning is needed beyond the provided facts
            3. Provide a concise, well-organized answer that directly addresses the query
            4. When appropriate, mention limitations in your knowledge or suggest alternatives
            
            {history_instructions}
            
            IMPORTANT: Maintain conversation continuity. Use the conversation history to provide
            context-aware responses. Don't repeat information unless specifically requested.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Add conversation history
            for msg in self.conversation:
                messages.append(msg)
            
            # Add RAG context
            if context:
                context_msg = f"""
                Here is relevant information about the query:
                
                {context}
                
                Use this information to answer the current query if relevant. Consider:
                - How relevant is this information to the specific question?
                - Does it fully answer the question or are there gaps?
                - Is any of this information potentially outdated or uncertain?
                - What additional information might improve your answer?
                """
                messages.append({"role": "system", "content": context_msg.strip()})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            return messages
        
        elif self.active_model == 'claude':
            # Format for Claude (single message with all context)
            system_prompt = f"""
            \n\nHuman: You are a reflective tourism assistant specializing in Curaçao. Use the provided information
            to answer user questions. If the information is not in the context,
            indicate that you don't have sufficient information. Respond naturally and conversationally,
            without mentioning technical terms like "chunks", "context", "vectors", etc.
            
            For each response:
            1. First assess the relevance and completeness of the information you have
            2. Consider whether reasoning is needed beyond the provided facts
            3. Provide a concise, well-organized answer that directly addresses the query
            4. When appropriate, mention limitations in your knowledge or suggest alternatives
            
            {history_instructions}
            
            IMPORTANT: Maintain conversation continuity. Use the conversation history to provide
            context-aware responses. Don't repeat information unless specifically requested.
            """
            
            # Add conversation history
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Add RAG context
            context_text = ""
            if context:
                context_text = f"""
                \n\nHuman: Here is relevant information about the query:
                
                {context}
                
                Use this information to answer the current query if relevant. Consider:
                - How relevant is this information to the specific question?
                - Does it fully answer the question or are there gaps?
                - Is any of this information potentially outdated or uncertain?
                - What additional information might improve your answer?
                """
            
            # Add current query
            query_text = f"\n\nHuman: {query}"
            
            # Combine everything
            prompt = system_prompt + conversation_text + context_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Simple format for fallback
            return {"query": query, "context": context, "is_historical": is_historical}
    
    def _create_simple_prompt(self, query: str) -> Dict[str, Any]:
        """
        Create a prompt for the LLM without RAG context.
        
        Args:
            query: User query
            
        Returns:
            Formatted prompt for the LLM
        """
        # Check if this is a historical query
        is_historical = self._is_history_query(query)
        
        # Create custom instructions for historical queries
        history_instructions = ""
        if is_historical:
            history_instructions = """
            This query is about the history of Curaçao. If you have general knowledge about this topic:
            1. Organize historical information chronologically when possible
            2. Mention key historical events and periods
            3. Include dates when you're confident about them
            4. Be transparent about the limitations of your knowledge
            
            If you lack specific historical details, acknowledge this and offer to provide general
            information about Curaçao's colonial history, cultural heritage, or related topics.
            """
        
        if self.active_model == 'openai':
            # Format for OpenAI
            messages = []
            
            # System message
            system_msg = f"""
            You are a reflective tourism assistant specializing in Curaçao. Respond concisely and naturally.
            If you don't have sufficient information to answer, clearly indicate this and suggest what
            information would be needed for a more complete answer.
            
            Always consider the logical implications of your responses and be transparent about 
            any assumptions you make. When uncertain, acknowledge limitations in your knowledge.
            
            {history_instructions}
            
            IMPORTANT: Maintain conversation continuity. Remember previous interactions and build upon them.
            Don't repeat information unless specifically requested.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Add conversation history
            for msg in self.conversation:
                messages.append(msg)
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            return messages
        
        elif self.active_model == 'claude':
            # Format for Claude
            system_prompt = f"""
            \n\nHuman: You are a reflective tourism assistant specializing in Curaçao. Respond concisely and naturally.
            If you don't have sufficient information to answer, clearly indicate this and suggest what
            information would be needed for a more complete answer.
            
            Always consider the logical implications of your responses and be transparent about 
            any assumptions you make. When uncertain, acknowledge limitations in your knowledge.
            
            {history_instructions}
            
            IMPORTANT: Maintain conversation continuity. Remember previous interactions and build upon them.
            Don't repeat information unless specifically requested.
            """
            
            # Add conversation history
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Add current query
            query_text = f"\n\nHuman: {query}"
            
            # Combine everything
            prompt = system_prompt + conversation_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Simple format for fallback
            return {"query": query, "is_historical": is_historical}
    
    async def _generate_reflection(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured reflection on the retrieved information.
        
        Args:
            query: User query
            rag_results: Results from RAG search
            
        Returns:
            Structured reflection
        """
        context = rag_results.get('context', "")
        chunks = rag_results.get('chunks', [])
        
        # Special handling for historical queries
        is_historical = self._is_history_query(query)
        
        # If no results, reflection is simple
        if not context or not chunks:
            return {
                "relevance": "No relevant information was found in the knowledge base.",
                "completeness": "The query cannot be answered with the available information.",
                "reliability": "No information to assess for reliability.",
                "gaps": "Complete information would be needed to answer this query.",
                "model_used": self.active_model,
                "needs_reasoning": True
            }
        
        # For historical queries, check if we have historical content
        if is_historical and not self._contains_historical_content(rag_results):
            return {
                "relevance": "Limited historical information found in the knowledge base.",
                "completeness": "Historical details are sparse or missing.",
                "reliability": "Available information may not provide accurate historical context.",
                "gaps": "Detailed historical timeline and key events are missing.",
                "model_used": self.active_model,
                "needs_reasoning": True
            }
        
        # For actual results, check if we have LLM available for detailed reflection
        if self.active_model in ['openai', 'claude']:
            # Use available LLM to generate reflection
            reflection_prompt = f"""
            Analyze the following information retrieved for the query: "{query}"
            
            RETRIEVED INFORMATION:
            {context}
            
            Provide a structured analysis with these components:
            1. RELEVANCE: How directly does this information address the query?
            2. COMPLETENESS: Does this information fully answer the query or are there gaps?
            3. RELIABILITY: How reliable and accurate is this information likely to be?
            4. GAPS: What additional information would improve the answer?
            5. REASONING NEEDED: Does answering this query require reasoning beyond the facts?
            
            Format your response as a JSON object with the keys: relevance, completeness, reliability, gaps, and needs_reasoning (boolean).
            """
            
            try:
                if self.active_model == 'openai':
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": reflection_prompt}],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    reflection_text = response.choices[0].message.content
                    reflection = json.loads(reflection_text)
                    reflection["model_used"] = self.active_model
                    return reflection
                
                elif self.active_model == 'claude':
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=500,
                        temperature=0.2,
                        messages=[
                            {"role": "user", "content": reflection_prompt}
                        ]
                    )
                    reflection_text = response.content[0].text
                    # Extract JSON from Claude's response
                    try:
                        import re
                        json_match = re.search(r'({.*})', reflection_text.replace('\n', ''), re.DOTALL)
                        if json_match:
                            reflection = json.loads(json_match.group(1))
                        else:
                            # Fallback to simple reflection
                            reflection = {
                                "relevance": "Information appears relevant but detailed analysis failed.",
                                "completeness": "Partial information available.",
                                "reliability": "Unknown reliability.",
                                "gaps": "Could not analyze gaps properly.",
                                "needs_reasoning": True
                            }
                    except Exception as e:
                        logger.error(f"Error parsing reflection JSON: {e}")
                        reflection = {
                            "relevance": "Information retrieved but analysis failed.",
                            "completeness": "Unknown completeness.",
                            "reliability": "Unknown reliability.",
                            "gaps": "Analysis error occurred.",
                            "needs_reasoning": True
                        }
                    
                    reflection["model_used"] = self.active_model
                    return reflection
            
            except Exception as e:
                logger.error(f"Error generating reflection: {e}")
        
        # Fallback to simple heuristic reflection
        chunk_count = len(chunks)
        
        try:
            # Calculate average score - protect against type errors
            if chunk_count > 0:
                scores = []
                for chunk in chunks:
                    score = chunk.get("score", 0)
                    if isinstance(score, (int, float)):
                        scores.append(score)
                
                avg_score = sum(scores) / len(scores) if scores else 0
            else:
                avg_score = 0
            
            relevance_msg = f"Retrieved {chunk_count} potentially relevant chunks with average score {avg_score:.2f}."
        except Exception as e:
            logger.error(f"Error calculating reflection metrics: {e}")
            relevance_msg = f"Retrieved {chunk_count} potentially relevant chunks."
        
        return {
            "relevance": relevance_msg,
            "completeness": "Information availability unknown without semantic analysis.",
            "reliability": "Information comes from the knowledge base with unknown reliability.",
            "gaps": "Detailed gap analysis not available in fallback mode.",
            "model_used": "fallback",
            "needs_reasoning": True
        }
    
    async def _generate_response_with_openai(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            messages: List of messages in OpenAI format
            
        Returns:
            Generated response
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI: {e}")
            return f"I'm sorry, an error occurred while processing your query. Details: {str(e)}"
    
    async def _generate_response_with_claude(self, prompt: str) -> str:
        """
        Generate a response using the Claude API.
        
        Args:
            prompt: Prompt for Claude
            
        Returns:
            Generated response
        """
        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude: {e}")
            return f"I'm sorry, an error occurred while processing your query. Details: {str(e)}"
    
    async def _generate_response_with_rag_only(self, query_data: Dict[str, Any]) -> str:
        """
        Generate a simple response using only the RAG results.
        
        Args:
            query_data: Query data
            
        Returns:
            Generated response
        """
        context = query_data.get("context", "")
        is_historical = query_data.get("is_historical", False)
        
        if not context:
            # Special handling for historical queries with no results
            if is_historical:
                return """
I don't have detailed historical information about Curaçao in my knowledge base at the moment. 

Curaçao has a rich history that includes indigenous Arawak settlements, Spanish discovery in the late 15th century, and Dutch colonization in the 17th century. The island became an important center for the Dutch West India Company and has a complex colonial past that includes involvement in the slave trade.

For more detailed historical information, I'd recommend asking about specific periods or aspects of Curaçao's history, such as:
1. The colonial period and Dutch influence
2. The architectural heritage of Willemstad (a UNESCO World Heritage site)
3. The cultural history and development of Papiamentu
4. Modern history and the relationship with the Netherlands

Would you like to know more about any of these specific aspects of Curaçao's history?
"""
            
            # Generic no-results response
            return "I'm sorry, I couldn't find relevant information for your query in my knowledge base."
        
        # Format historical information in a more structured way
        if is_historical:
            # Extract potential dates from the context
            date_pattern = r'\b(1[4-9]\d\d|20\d\d)\b'
            dates = re.findall(date_pattern, context)
            
            if dates:
                # Try to organize by dates if available
                response = "Here's some historical information about Curaçao:\n\n"
                
                # Simple extraction of sentences with dates
                sentences = re.split(r'(?<=[.!?])\s+', context)
                dated_info = []
                
                for sentence in sentences:
                    if any(date in sentence for date in dates):
                        dated_info.append(sentence)
                
                if dated_info:
                    response += "\n".join(dated_info)
                    response += "\n\nThis historical information comes from our knowledge base about Curaçao."
                    return response
            
            # If we couldn't organize by dates, return formatted context
            return "Historical information about Curaçao:\n\n" + context
        
        # Regular response for non-historical queries
        return (
            "Here's the information I found:\n\n" +
            context +
            "\n\nThis information comes directly from our knowledge base about Curaçao."
        )
    
    async def _process_query_standard(self, query: str) -> str:
        """
        Process a user query and generate a response without using the agent.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        # Determine if the query needs RAG
        needs_rag = await self._query_needs_rag(query)
        
        # If needs RAG, perform search
        rag_results = {}
        reflection = {}
        if needs_rag:
            print(f"{Colors.DIM}Searching for relevant information...{Colors.RESET}", flush=True)
            rag_results = await self._perform_rag_search(query)
            
            # Generate reflection on retrieved information
            if rag_results:
                reflection = await self._generate_reflection(query, rag_results)
                logger.info(f"Reflection generated: {reflection}")
                # Decide whether to use RAG based on reflection
                relevance = reflection.get("relevance", "")
                if isinstance(relevance, str) and relevance.startswith("No relevant"):
                    needs_rag = False
        
        # Generate prompt based on active model
        if needs_rag and rag_results:
            # Prompt with RAG context
            prompt = self._create_prompt_with_rag_context(query, rag_results)
        else:
            # Prompt without RAG context
            prompt = self._create_simple_prompt(query)
        
        # Generate response based on active model
        if self.active_model == 'openai':
            print(f"{Colors.DIM}Generating response with OpenAI...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_openai(prompt)
        elif self.active_model == 'claude':
            print(f"{Colors.DIM}Generating response with Claude...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_claude(prompt)
        else:
            # Fallback to simple RAG
            response = await self._generate_response_with_rag_only(prompt)
        
        # Update conversation history
        self.conversation.append({"role": "user", "content": query})
        self.conversation.append({"role": "assistant", "content": response})
        
        # Truncate history if exceeds maximum
        if len(self.conversation) > self.max_conversation_history * 2:
            self.conversation = self.conversation[-self.max_conversation_history*2:]
        
        return response
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        try:
            # Use agent if available and enabled
            if self.use_agent and self.agent:
                logger.info(f"Processing query with agent: {query[:50]}...")
                try:
                    # Try processing with agent first
                    response = self.agent.process_query(query)
                    
                    # Check if we got a meaningful response
                    if not response or response.strip() == "" or "I apologize" in response and len(response) < 100:
                        logger.warning("Agent produced potentially problematic response, falling back to standard processing")
                        response = await self._process_query_standard(query)
                    
                    # Update this instance's conversation history too (for redundancy)
                    self.conversation.append({"role": "user", "content": query})
                    self.conversation.append({"role": "assistant", "content": response})
                    
                    # Truncate history if exceeds maximum
                    if len(self.conversation) > self.max_conversation_history * 2:
                        self.conversation = self.conversation[-self.max_conversation_history*2:]
                        
                    return response
                except Exception as e:
                    logger.error(f"Error in agent processing, falling back to standard: {e}")
                    return await self._process_query_standard(query)
            else:
                logger.info(f"Processing query with standard pipeline: {query[:50]}...")
                return await self._process_query_standard(query)
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return f"I'm sorry, I encountered an unexpected error. Please try again or rephrase your question."
    
    def print_welcome(self):
        """Print welcome message."""
        title = "Reflective RAG Agent Chat - Curaçao Tourism Assistant"
        subtitle = f"Active model: {self.active_model.upper()} | Agent mode: {'Enabled' if self.use_agent else 'Disabled'}"
        
        print("\n" + "=" * self.term_width)
        print(f"{Colors.BOLD}{Colors.GREEN}{title.center(self.term_width)}{Colors.RESET}")
        print(f"{Colors.DIM}{subtitle.center(self.term_width)}{Colors.RESET}")
        print("=" * self.term_width)
        
        print(f"\n{Colors.CYAN}Welcome to the Curaçao tourism assistant!{Colors.RESET}")
        print("I can help you with information about the island, attractions, beaches, restaurants, and more.")
        print("I'll search for relevant information and reflect on my knowledge to provide the best answers.")
        
        if self.active_model == 'rag-only':
            print(f"\n{Colors.YELLOW}Notice: No valid LLM API key detected.{Colors.RESET}")
            print(f"{Colors.YELLOW}Using RAG-only mode for responses.{Colors.RESET}")
            print(f"{Colors.YELLOW}For a complete experience, configure OPENAI_API_KEY or ANTHROPIC_API_KEY.{Colors.RESET}")
        
        print("\nType your query and press Enter to begin.")
        print("Type '/exit' to end the conversation.")
        print("=" * self.term_width + "\n")
    
    def format_response(self, response: str) -> str:
        """
        Format a response for display in the terminal.
        
        Args:
            response: Response to format
            
        Returns:
            Formatted response
        """
        # Apply wrap to lines
        lines = []
        for paragraph in response.split('\n'):
            if not paragraph.strip():
                lines.append("")
                continue
            
            # Wrap each paragraph
            wrapped = textwrap.wrap(paragraph, width=self.term_width-4)
            lines.extend(wrapped)
        
        # Join with indentation
        formatted = "\n".join(["    " + line for line in lines])
        
        return formatted
    
    async def run(self):
        """Run the interactive chat."""
        # Initialize components
        if not await self.initialize():
            print(f"\n{Colors.RED}Error initializing the system. Check logs for details.{Colors.RESET}\n")
            return
        
        # Show welcome
        self.print_welcome()
        
        # Main loop
        try:
            while True:
                # Update terminal size
                self.term_width, self.term_height = shutil.get_terminal_size()
                
                # Request input
                query = input(f"\n{Colors.BOLD}{Colors.GREEN}You:{Colors.RESET} ")
                
                # Check if empty
                if not query.strip():
                    continue
                
                # Check if exit command
                if query.lower() in ['/exit', '/quit', '/salir']:
                    print(f"\n{Colors.GREEN}Thank you for using the Curaçao tourism assistant! Goodbye!{Colors.RESET}\n")
                    break
                
                # Process query
                response = await self.process_query(query)
                
                # Show response
                print(f"\n{Colors.BOLD}{Colors.BLUE}Assistant:{Colors.RESET}")
                formatted_response = self.format_response(response)
                print(formatted_response)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye!{Colors.RESET}\n")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}\n")
        finally:
            # Close resources
            await self.close()

# ===== LANGGRAPH AGENT IMPLEMENTATION =====
if LANGGRAPH_AVAILABLE:
    # System prompt for agent behavior
    AGENT_SYSTEM_PROMPT = """You are a Reflective Tourism Assistant that combines knowledge retrieval with critical thinking,
    specializing in providing information about Curaçao.

    IMPORTANT: Always maintain conversation continuity. Remember previous exchanges and don't repeat information
    unless explicitly asked. If the user mentions something vague like "history" or "beaches", understand this
    in the context of Curaçao tourism.

    Your approach to answering questions follows this methodology:

    1. ASSESS: Determine if the question requires factual knowledge about Curaçao tourism, reasoning, or both
       - For factual questions (e.g., "What beaches are in Curaçao?"), use the reflective_retrieval tool
       - For analytical questions (e.g., "What's the best time to visit?"), use the structured_reasoning tool
       - For complex questions, you may use both tools in sequence

    2. REFLECT: After using any tool, carefully analyze the results
       - Consider the relevance, completeness, and reliability of information
       - Identify any gaps, assumptions, or limitations in your understanding
       - Determine if additional tool use would improve your answer

    3. RESPOND: Provide a clear, accurate answer that:
       - Directly addresses the user's question about Curaçao tourism
       - Incorporates insights from your tools and reflection
       - Acknowledges any limitations or uncertainties
       - Uses natural, conversational language

    CONVERSATION CONSISTENCY:
    - Track what the user has already asked and what you've already shared
    - Avoid repeating the same information unless requested
    - If the user asks a vague follow-up (like "tell me more" or "what about history?"), 
      interpret it in context of the conversation about Curaçao

    For questions about Curaçao's history:
    - Organize information chronologically when possible
    - Include relevant dates and time periods
    - Mention key historical events and their significance
    - Explain cultural and historical context

    Remember: Your goal is to provide not just accurate information about Curaçao, but also to 
    maintain a natural, coherent conversation that builds on previous exchanges.
    """

    # LangGraph agent tools
    @tool
    def reflective_retrieval(query: str, context: Optional[Dict] = None) -> str:
        """
        Retrieve information from your knowledge base and reflect on its quality and relevance.
        
        Args:
            query: The user's question or information need
            context: Optional context from previous interactions
            
        Returns:
            Retrieved information with structured reflection
        """
        try:
            # Extract RAG session from context
            rag_session = context.get("rag_session") if context else None
            if not rag_session:
                return "Error: No RAG session available for retrieval."
            
            # Log what we're searching for
            logger.info(f"Performing reflective retrieval for: {query}")
            
            # Check if this is a historical query
            is_historical = rag_session._is_history_query(query)
            if is_historical:
                # Add historical context to the query
                enhanced_query = f"history of Curaçao {query}"
                logger.info(f"Enhanced historical query: {enhanced_query}")
            else:
                enhanced_query = query
            
            # Perform RAG search asynchronously
            rag_results = asyncio.run(rag_session._perform_rag_search(enhanced_query))
            
            # Check if we got results
            if not rag_results or not rag_results.get('chunks'):
                logger.warning(f"No results found for query: {query}")
                
                if is_historical:
                    return """
    RETRIEVED INFORMATION:
    No specific historical information about Curaçao was found for this query.

    REFLECTION ON RETRIEVED INFORMATION:
    - Relevance: The knowledge base doesn't contain directly relevant historical information.
    - Completeness: Historical information is incomplete in the knowledge base.
    - Reliability: Cannot assess reliability of non-existent information.
    - Knowledge Gaps: Detailed historical timeline, colonial history, and key historical events are missing.
    - Reasoning Required: Yes, will need to rely on general knowledge about Curaçao's history.
    """
                else:
                    return """
    RETRIEVED INFORMATION:
    No specific information about this topic was found in the knowledge base.

    REFLECTION ON RETRIEVED INFORMATION:
    - Relevance: The knowledge base doesn't contain directly relevant information for this query.
    - Completeness: The information is incomplete as no specific data was found.
    - Reliability: Cannot assess reliability of non-existent information.
    - Knowledge Gaps: Complete information about this topic is missing from the knowledge base.
    - Reasoning Required: Yes, will need to rely on general knowledge.
    """
            
            # Generate reflection
            reflection = asyncio.run(rag_session._generate_reflection(query, rag_results))
            
            # Format response with results and reflection
            retrieved_context = rag_results.get('context', "No relevant information found.")
            
            # Create a more detailed, structured response
            response = f"""
    RETRIEVED INFORMATION:
    {retrieved_context}

    REFLECTION ON RETRIEVED INFORMATION:
    - Relevance: {reflection.get('relevance', 'Unknown')}
    - Completeness: {reflection.get('completeness', 'Unknown')}
    - Reliability: {reflection.get('reliability', 'Unknown')}
    - Knowledge Gaps: {reflection.get('gaps', 'Unknown')}
    - Reasoning Required: {"Yes" if reflection.get('needs_reasoning', True) else "No"}
    """
            
            logger.info(f"Reflective retrieval completed successfully for: {query}")
            return response
        except Exception as e:
            # Graceful error handling
            error_msg = f"Error during retrieval: {str(e)}"
            logger.error(error_msg)
            return f"""
    RETRIEVED INFORMATION:
    I encountered a problem with the retrieval system.

    REFLECTION ON RETRIEVED INFORMATION:
    - Relevance: Unable to assess due to retrieval error.
    - Completeness: Unable to assess due to retrieval error.
    - Reliability: Unable to assess due to retrieval error.
    - Knowledge Gaps: Cannot identify specific gaps due to system error.
    - Reasoning Required: Yes, will need to rely on general knowledge instead of retrieval.

    Technical details: {str(e)}
    """

    @tool
    def structured_reasoning(query: str, context: Optional[Dict] = None) -> str:
        """
        Perform step-by-step reasoning with structured reflection on the thinking process.
        
        Args:
            query: The question or problem requiring reasoning
            context: Optional context from previous steps or retrieval
            
        Returns:
            Step-by-step reasoning with structured reflection
        """
        try:
            # Extract RAG session from context
            rag_session = context.get("rag_session") if context else None
            if not rag_session:
                return "Error: No LLM session available for reasoning."
            
            # Check if this is a historical query
            is_historical = rag_session._is_history_query(query)
            
            # Create reasoning prompt
            if rag_session.active_model == 'openai':
                system_content = """
                Perform step-by-step reasoning to analyze the given query.
                Break down the problem, analyze key components, and reach a logical conclusion.
                Structure your analysis with clear steps and reflect on your reasoning process.
                """
                
                if is_historical:
                    system_content += """
                    Since this query is about Curaçao's history:
                    1. Consider the chronological development of events
                    2. Think about colonial influences (Spanish, Dutch)
                    3. Consider cultural and economic historical factors
                    4. Reflect on how historical events shaped modern Curaçao
                    """
                
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"""
                    Please perform structured reasoning on this query: "{query}"
                    
                    If there is any relevant context from previous retrieval, consider it in your analysis.
                    
                    Format your response as:
                    1. REASONING STEPS: A numbered list of your step-by-step reasoning
                    2. REFLECTION: An analysis of your assumptions, logical structure, alternative approaches, and confidence
                    """}
                ]
                
                response = asyncio.run(rag_session._generate_response_with_openai(messages))
            
            elif rag_session.active_model == 'claude':
                system_content = """
                Perform step-by-step reasoning to analyze the given query.
                Break down the problem, analyze key components, and reach a logical conclusion.
                Structure your analysis with clear steps and reflect on your reasoning process.
                """
                
                if is_historical:
                    system_content += """
                    Since this query is about Curaçao's history:
                    1. Consider the chronological development of events
                    2. Think about colonial influences (Spanish, Dutch)
                    3. Consider cultural and economic historical factors
                    4. Reflect on how historical events shaped modern Curaçao
                    """
                
                prompt = f"""
                \n\nHuman: {system_content}
                
                Please perform structured reasoning on this query: "{query}"
                
                If there is any relevant context from previous retrieval, consider it in your analysis.
                
                Format your response as:
                1. REASONING STEPS: A numbered list of your step-by-step reasoning
                2. REFLECTION: An analysis of your assumptions, logical structure, alternative approaches, and confidence
                
                \n\nAssistant:
                """
                
                response = asyncio.run(rag_session._generate_response_with_claude(prompt))
            
            else:
                # Simple fallback for RAG-only mode
                if is_historical:
                    response = f"""
                    REASONING PROCESS:
                    1. Breaking down historical query about Curaçao: "{query}"
                    2. Considering key historical periods: pre-colonial, Spanish discovery, Dutch colonization, modern era
                    3. Analyzing available historical information in the knowledge base
                    4. Drawing preliminary conclusions based on limited historical data
                    
                    REFLECTION ON REASONING:
                    - Key Assumptions: Limited to basic understanding of Caribbean colonial history
                    - Logical Structure: Chronological analysis of historical developments
                    - Alternative Approaches: Would benefit from specialized historical sources
                    - Confidence Assessment: Low due to limited historical information in knowledge base
                    """
                else:
                    response = f"""
                    REASONING PROCESS:
                    1. Breaking down query: "{query}"
                    2. Identifying key components of the question
                    3. Applying logical analysis
                    4. Drawing preliminary conclusions
                    
                    REFLECTION ON REASONING:
                    - Key Assumptions: Limited to basic understanding of Curaçao tourism
                    - Logical Structure: Sequential analysis of query components
                    - Alternative Approaches: Would benefit from external knowledge
                    - Confidence Assessment: Low due to limited information
                    """
            
            return response
        except Exception as e:
            # Graceful error handling
            error_msg = f"Error during reasoning: {str(e)}"
            logger.error(error_msg)
            return f"I encountered a problem during the reasoning process: {str(e)}. Let me simplify and try again."

    # List of tools available to the agent
    agent_tools = [reflective_retrieval, structured_reasoning]

    class ReflectiveAgent:
        """Implementation of a reflective agent using LangGraph."""
        
        def __init__(self, rag_chat: 'ReflectiveRAGChat'):
            """
            Initialize the reflective agent.
            
            Args:
                rag_chat: The RAG chat instance to use for processing
            """
            self.rag_chat = rag_chat
            self.app = self._create_agent_workflow()
            self.conversation_history = []
            self.state = None  # Maintain persistent state between conversations
        
        def _create_agent_workflow(self):
            """
            Create and configure the agent workflow graph.
            
            Returns:
                Compiled workflow
            """
            # Initialize LLM based on rag_chat's active model
            if self.rag_chat.active_model == 'openai':
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo", 
                    temperature=self.rag_chat.temperature
                )
            elif self.rag_chat.active_model == 'claude':
                llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=self.rag_chat.temperature
                )
            else:
                # Fallback to OpenAI if available
                if OPENAI_AVAILABLE:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
                else:
                    raise ValueError("No suitable LLM available for the agent")
            
            # Bind tools to the LLM
            agent_llm = llm.bind_tools(agent_tools)
            
            # Create the workflow graph
            workflow = StateGraph(EnhancedState)
            
            # Define the agent node function
            def agent_node(state: EnhancedState):
                """Process messages and decide whether to use tools."""
                try:
                    messages = state["messages"]
                    
                    # Add system message
                    system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)
                    full_messages = [system_message] + messages
                    
                    # Ensure rag_session is available
                    if "rag_session" not in state["context"]:
                        state["context"]["rag_session"] = self.rag_chat
                    
                    # Log for debugging
                    logger.debug(f"Agent processing with {len(full_messages)} messages")
                    
                    # Get response from LLM
                    response = agent_llm.invoke(full_messages)
                    logger.debug(f"Agent LLM response received")
                    
                    # Update state
                    return {"messages": [response]}
                except Exception as e:
                    error_msg = f"Agent node error: {str(e)}"
                    logger.error(error_msg)
                    # Return graceful error message
                    error_response = AIMessage(content=f"I encountered an issue processing your request: {str(e)}. Let me try a different approach.")
                    return {"messages": [error_response]}
            
            # Create tool execution node
            tool_node = ToolNode(agent_tools)
            
            # Decision function for next step - IMPROVED VERSION
            def route_next(state: EnhancedState) -> Literal["agent_node", "tool_node", END]:
                """Determine whether to use tools, continue processing, or end."""
                messages = state["messages"]
                last_message = messages[-1]
                
                # Log the decision point
                logger.debug(f"Routing decision for message type: {type(last_message)}")
                
                # If agent requested a tool, route to tool node
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    logger.debug(f"Tool call detected: {last_message.tool_calls}")
                    return "tool_node"
                
                # Otherwise end this processing cycle
                logger.debug("No tool calls, ending processing cycle")
                return END
            
            # Add nodes to graph
            workflow.add_node("agent_node", agent_node)
            workflow.add_node("tool_node", tool_node)
            
            # Define workflow edges
            workflow.add_edge(START, "agent_node")
            workflow.add_conditional_edges("agent_node", route_next)
            workflow.add_edge("tool_node", "agent_node")
            
            # Compile graph
            return workflow.compile()
        
        def process_query(self, query: str) -> str:
            """
            Process a user query and return the assistant's response.
            
            Args:
                query: User query
                
            Returns:
                Assistant's response
            """
            try:
                # Create initial state or use existing state
                if self.state is None:
                    logger.info("Creating new agent state")
                    self.state = {
                        "messages": [],
                        "context": {"rag_session": self.rag_chat, "conversation_id": id(self)}
                    }
                
                # Add the new user query to messages
                human_message = HumanMessage(content=query)
                self.state["messages"].append(human_message)
                
                # Run workflow with current state
                logger.info(f"Running agent workflow for query: {query[:50]}...")
                updated_state = self.app.invoke(self.state)
                
                # Update our persistent state with the new state
                self.state = updated_state
                
                # Get the latest AI message
                if not self.state["messages"] or not any(isinstance(m, AIMessage) for m in self.state["messages"]):
                    logger.warning("No AI response found in updated state")
                    return "I apologize, but I couldn't generate a proper response. Please try asking again."
                
                # Find the latest AI message
                ai_messages = [m for m in self.state["messages"] if isinstance(m, AIMessage)]
                if not ai_messages:
                    return "I apologize, but I couldn't generate a proper response. Please try asking again."
                
                latest_ai_message = ai_messages[-1]
                
                # Update conversation history for future reference
                self.conversation_history = self.state["messages"][-10:]  # Keep last 10 messages
                
                # Return final response
                return latest_ai_message.content
                
            except Exception as e:
                error_msg = f"Error processing query with agent: {str(e)}"
                logger.error(error_msg)
                # Add error message to conversation history so we maintain flow
                error_response = AIMessage(content=f"I experienced a technical issue processing your request. Could you please rephrase your question?")
                if self.state and "messages" in self.state:
                    self.state["messages"].append(error_response)
                return error_response.content
        
        def reset_conversation(self):
            """Reset the conversation history and state."""
            self.conversation_history = []
            self.state = None
            return "Conversation history has been reset."

# ===== MAIN FUNCTION =====
async def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Reflective RAG Agent Chat - Curaçao Tourism Assistant')
    
    parser.add_argument('--model', choices=['openai', 'claude'],
                       default='openai', help='Model to use')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return from RAG')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for LLM responses (0.0 to 1.0)')
    parser.add_argument('--use-container-name', action='store_true',
                       help='Use postgres_pgvector instead of localhost')
    parser.add_argument('--no-agent', action='store_true',
                       help='Disable LangGraph agent architecture')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app2").setLevel(logging.DEBUG)
    
    # Initialize chat
    chat = ReflectiveRAGChat(
        model=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
        use_localhost=not args.use_container_name,
        use_agent=not args.no_agent
    )
    
    # Run chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)