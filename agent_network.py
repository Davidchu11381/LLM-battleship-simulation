"""
Clean LLM Agent Network Framework

Simple framework that creates a network of LLM agents from a configuration file.
Supports Ollama, OpenAI, Claude (Anthropic), and Gemini using AG2 (AutoGen 2).
Agents communicate based on network topology.

Usage:
    from agent_network import AgentNetwork
    
    network = AgentNetwork.from_config("network_config.json")
    result = network.send_message("agent1", "agent2", "Hello!")
"""

import json
import asyncio
import networkx as nx
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import uuid
from datetime import datetime
import os

from autogen import ConversableAgent, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationTracker:
    """Custom conversation tracking system independent of AutoGen"""
    
    def __init__(self):
        self.conversations = []
        self.agent_conversations = {}  # agent_id -> list of conversation_ids
        self.current_conversation_id = 0
    
    def start_conversation(self, sender_id: str, receiver_id: str, initial_message: str) -> str:
        """Start tracking a new conversation"""
        conversation_id = f"conv_{self.current_conversation_id}"
        self.current_conversation_id += 1
        
        conversation = {
            'id': conversation_id,
            'sender': sender_id,
            'receiver': receiver_id,
            'initial_message': initial_message,
            'messages': [],
            'timestamp': asyncio.get_event_loop().time(),
            'status': 'active'
        }
        
        self.conversations.append(conversation)
        
        # Track by agent
        if sender_id not in self.agent_conversations:
            self.agent_conversations[sender_id] = []
        if receiver_id not in self.agent_conversations:
            self.agent_conversations[receiver_id] = []
            
        self.agent_conversations[sender_id].append(conversation_id)
        self.agent_conversations[receiver_id].append(conversation_id)
        
        # Add initial message
        self.add_message(conversation_id, sender_id, initial_message, 'request')
        
        return conversation_id
    
    def add_message(self, conversation_id: str, speaker_id: str, content: str, message_type: str = 'response'):
        """Add a message to an ongoing conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return
        
        message = {
            'speaker': speaker_id,
            'content': content,
            'type': message_type,  # 'request', 'response', 'system'
            'timestamp': asyncio.get_event_loop().time()
        }
        
        conversation['messages'].append(message)
    
    def complete_conversation(self, conversation_id: str):
        """Mark conversation as complete"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation['status'] = 'complete'
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID"""
        for conv in self.conversations:
            if conv['id'] == conversation_id:
                return conv
        return None
    
    def get_agent_conversations(self, agent_id: str, recent_only: bool = True) -> List[Dict]:
        """Get all conversations involving an agent"""
        if agent_id not in self.agent_conversations:
            return []
        
        conversations = []
        for conv_id in self.agent_conversations[agent_id]:
            conv = self.get_conversation(conv_id)
            if conv:
                if recent_only:
                    # Only conversations from last 5 minutes
                    current_time = asyncio.get_event_loop().time()
                    if current_time - conv['timestamp'] < 300:
                        conversations.append(conv)
                else:
                    conversations.append(conv)
        
        # Sort by timestamp (most recent first)
        conversations.sort(key=lambda x: x['timestamp'], reverse=True)
        return conversations
    
    def extract_advice_for_agent(self, agent_id: str) -> List[str]:
        """Extract all advice given TO a specific agent, from both assistants and teammates."""
        # Get all recent convs involving this agent
        conversations = self.get_agent_conversations(agent_id)
        advice_items = []
        seen = set()

        for conv in conversations:
            # We no longer skip based on conv['receiver']
            for msg in conv['messages']:
                # only look at messages *to* our agent
                if msg['speaker'] == agent_id:
                    continue

                content = msg['content']
                giver = msg['speaker']

                # Coordinate-based advice
                coord = self._parse_coordinate_from_text(content)
                if coord:
                    key = (giver, 'coord', coord)
                    if key not in seen:
                        prefix = '🤖' if 'assistant' in giver else '👥'
                        advice_items.append(f"{prefix} {giver}: Suggests {coord}")
                        seen.add(key)
                    continue

                # Strategic advice based on keywords
                strategic_keywords = {'target','focus','avoid','center','edge','corner','pattern','cluster','switch','continue'}
                if any(k in content.lower() for k in strategic_keywords) and len(content) > 20:
                    key = (giver, 'strat', content[:30])
                    if key not in seen:
                        summary = content[:60].rstrip() + ('…' if len(content) > 60 else '')
                        prefix = '🤖' if 'assistant' in giver else '👥'
                        advice_items.append(f"{prefix} {giver}: {summary}")
                        seen.add(key)

        return advice_items

    
    def _parse_coordinate_from_text(self, text: str) -> Optional[str]:
        """Parse coordinate from text using regex"""
        import re
        
        # Look for COORDINATE: [A1] format first
        coordinate_pattern = r'COORDINATE:\s*\[?([A-J][1-9]|[A-J]10)\]?'
        match = re.search(coordinate_pattern, text)
        if match:
            return match.group(1)
        
        # Look for standalone coordinates like "E5", "D4", etc.
        standalone_pattern = r'\b([A-J][1-9]|[A-J]10)\b'
        matches = re.findall(standalone_pattern, text)
        if matches:
            return matches[0]
        
        return None

@dataclass
class NetworkConfig:
    """Simple network configuration"""
    nodes: Dict[str, Dict[str, Any]]  # node_id -> {name, model, provider, system_message, ...}
    edges: List[List[str]]  # [[node1, node2], ...] - who can talk to whom
    global_config: Dict[str, Any] = None  # Global settings
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path], edges_file: Union[str, Path] = None) -> 'NetworkConfig':
        """Load configuration from JSON file and optional edges file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Load edges from separate file if provided
        if edges_file and Path(edges_file).exists():
            edges = cls._load_edges_from_file(edges_file)
            logger.info(f"📊 Loaded {len(edges)} edges from {edges_file}")
        else:
            # Fallback to edges in JSON file
            edges = data.get('edges', [])
            if edges:
                logger.info(f"📊 Using {len(edges)} edges from JSON config")
        
        return cls(
            nodes=data.get('nodes', {}),
            edges=edges,
            global_config=data.get('global_config', {})
        )
    
    @staticmethod
    def _load_edges_from_file(edges_file: Union[str, Path]) -> List[List[str]]:
        """Load edges from a text file
        
        Supported formats:
        - Simple: agent1 -> agent2
        - JSON: [["agent1", "agent2"], ["agent2", "agent3"]]
        - CSV: agent1,agent2
        """
        edges = []
        
        with open(edges_file, 'r') as f:
            content = f.read().strip()
        
        # Try to parse as JSON first
        if content.startswith('['):
            try:
                edges = json.loads(content)
                return edges
            except json.JSONDecodeError:
                pass
        
        # Parse line by line
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            
            try:
                # Format: agent1 -> agent2
                if '->' in line:
                    sender, receiver = line.split('->', 1)
                    edges.append([sender.strip(), receiver.strip()])
                
                # Format: agent1,agent2
                elif ',' in line:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) == 2:
                        edges.append(parts)
                    else:
                        logger.warning(f"Invalid edge format on line {line_num}: {line}")
                
                # Format: agent1 agent2 (space separated)
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        edges.append(parts)
                    else:
                        logger.warning(f"Invalid edge format on line {line_num}: {line}")
            
            except Exception as e:
                logger.error(f"Error parsing line {line_num}: {line} - {e}")
        
        return edges
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for easy network operations"""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, attributes in self.nodes.items():
            G.add_node(node_id, **attributes)
        
        # Add edges
        for edge in self.edges:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
        
        return G


class LLMProvider:
    """Factory for creating LLM configurations"""
    
    @staticmethod
    def create_config(provider: str, model: str, **kwargs) -> LLMConfig:
        """Create LLM config for different providers"""
        provider = provider.lower()
        
        if provider == "ollama":
            return LLMConfig(
                model=model,
                api_type="ollama",
                client_host=kwargs.get("client_host", "http://localhost:11434"),
                temperature=kwargs.get("temperature", 0.7),
                stream=False,
                **{k: v for k, v in kwargs.items() if k not in ["client_host", "temperature"]}
            )
        
        elif provider == "openai":
            return LLMConfig(
                model=model,
                api_type="openai",
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "temperature", "max_tokens"]}
            )
        
        elif provider == "anthropic" or provider == "claude":
            return LLMConfig(
                model=model,
                api_type="anthropic",
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "temperature", "max_tokens"]}
            )
        
        elif provider == "gemini" or provider == "google":
            return LLMConfig(
                model=model,
                api_type="google",
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.7),
                **{k: v for k, v in kwargs.items() if k not in ["api_key", "temperature"]}
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class AgentNetwork:
    """Main agent network class"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.graph = config.to_networkx()
        self.agents: Dict[str, ConversableAgent] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.simulation_rounds: List[Dict[str, Any]] = []
        self.current_round = 0
        
        # Logging configuration
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.current_log_file = None
        self.verbose_logs = []
        
        self._initialize_agents()
        logger.info(f"Initialized network with {len(self.agents)} agents")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], edges_file: Union[str, Path] = None) -> 'AgentNetwork':
        """Create network from configuration file and optional edges file"""
        config = NetworkConfig.from_file(config_path, edges_file)
        network = cls(config)
        
        # Store file paths for logging
        network.config_file_path = config_path
        network.edges_file_path = edges_file
        
        return network
    
    def _initialize_agents(self):
        """Initialize all agents from configuration with role-based system messages"""
        global_config = self.config.global_config or {}
        role_definitions = global_config.get('role_definitions', {})
        provider_defaults = global_config.get('provider_defaults', {})
        api_keys = global_config.get('api_keys', {})
        
        for node_id, node_data in self.config.nodes.items():
            try:
                # Extract basic configuration
                name = node_data.get('name', node_id)
                model = node_data['model']
                provider = node_data['provider']
                role = node_data.get('role', 'assistant')  # Default to assistant if no role specified
                
                # Get role definition
                role_def = role_definitions.get(role, {})
                if not role_def:
                    logger.warning(f"Role '{role}' not found in role_definitions, using basic config")
                    system_message = node_data.get('system_message', f"You are {name}, an AI assistant.")
                else:
                    # Build system message from template
                    system_message = self._build_system_message(
                        role_def, node_data, name, role
                    )
                
                # Merge provider defaults with node-specific config
                provider_config = provider_defaults.get(provider, {}).copy()
                
                # Add API key from global keys if not specified in node
                if 'api_key' not in node_data and provider in api_keys:
                    provider_config['api_key'] = api_keys[provider]
                
                # Override with node-specific settings
                for key, value in node_data.items():
                    if key not in ['name', 'model', 'provider', 'role', 'specialization']:
                        provider_config[key] = value
                
                # Apply specialization overrides, but filter out template-only params
                specialization = node_data.get('specialization', {})
                
                # Template-only parameters that shouldn't go to LLM config
                template_only_params = {'word_limit', 'additional_instructions', 'team_name', 
                                    'leadership_style', 'risk_tolerance', 'assistant_reliance',
                                    'team_input_consideration', 'strategy_focus', 'communication_style',
                                    'strategy_type', 'suggestion_style', 'coordinate_preference',
                                    'player_name', 'description', 'decision_speed'}
                
                # Only add specialization params that are valid for LLM config
                for key, value in specialization.items():
                    if key not in template_only_params:
                        provider_config[key] = value
                
                # Create LLM config (without template-only parameters)
                llm_config = LLMProvider.create_config(
                    provider=provider,
                    model=model,
                    **provider_config
                )
                
                # Create agent
                with llm_config:
                    agent = ConversableAgent(
                        name=node_id,
                        system_message=system_message,
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=node_data.get('max_consecutive_auto_reply', 5),
                        description=node_data.get('description', f"{name} - {role_def.get('description', 'AI agent')}")
                    )
                
                self.agents[node_id] = agent
                logger.info(f"✓ Created {role} agent '{node_id}' ({provider}/{model})")
                
            except Exception as e:
                logger.error(f"✗ Failed to create agent '{node_id}': {e}")
                raise

    # ADD these new helper methods to the AgentNetwork class:

    def _build_system_message(self, role_def: Dict[str, Any], node_data: Dict[str, Any], 
                            name: str, role: str) -> str:
        """Build system message from role template and node specialization"""
        
        # Get template and default params
        template = role_def.get('system_message_template', 
                            "You are a {role} agent named {name}.")
        default_params = role_def.get('default_params', {})
        
        # Merge default params with specialization
        specialization = node_data.get('specialization', {})
        params = {**default_params, **specialization}
        
        # Add standard template variables
        params.update({
            'role': role,
            'name': name,
            'additional_instructions': params.get('additional_instructions', '')
        })
        
        try:
            return template.format(**params)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} for agent {name}, using basic template")
            return f"You are a {role} agent named {name}."

    def get_agent_role(self, agent_id: str) -> str:
        """Get the role of a specific agent"""
        if agent_id in self.config.nodes:
            return self.config.nodes[agent_id].get('role', 'unknown')
        return 'unknown'

    def get_agents_by_role(self, role: str) -> List[str]:
        """Get all agent IDs that have a specific role"""
        return [
            agent_id for agent_id, config in self.config.nodes.items()
            if config.get('role') == role and agent_id in self.agents
        ]

    def get_role_description(self, role: str) -> str:
        """Get the description of a role"""
        global_config = self.config.global_config or {}
        role_definitions = global_config.get('role_definitions', {})
        role_def = role_definitions.get(role, {})
        return role_def.get('description', f'{role} agent')
    
    def _initialize_logging(self, network_file: str = None, prompts_file: str = None, 
                          rounds: int = None, max_turns: int = None, delay: float = None):
        """Initialize logging for the simulation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename components
        filename_parts = []
        
        # Network topology part
        if network_file:
            network_name = Path(network_file).stem
            filename_parts.append(f"net_{network_name}")
        else:
            filename_parts.append("net_default")
        
        # Prompts part
        if prompts_file:
            prompts_name = Path(prompts_file).stem
            filename_parts.append(f"prompts_{prompts_name}")
        else:
            filename_parts.append("prompts_default")
        
        # Parameters part
        params = []
        if rounds is not None:
            params.append(f"r{rounds}")
        if max_turns is not None:
            params.append(f"t{max_turns}")
        if delay is not None:
            params.append(f"d{delay}")
        
        if params:
            filename_parts.append("_".join(params))
        
        # Add timestamp
        filename_parts.append(timestamp)
        
        # Create final filename
        log_filename = "_".join(filename_parts) + ".txt"
        self.current_log_file = self.output_dir / log_filename
        
        # Initialize log with header
        self._write_log_header(network_file, prompts_file, rounds, max_turns, delay)
        
        logger.info(f"📝 Logging to: {self.current_log_file}")
        
        return self.current_log_file
    
    def _write_log_header(self, network_file: str = None, prompts_file: str = None,
                         rounds: int = None, max_turns: int = None, delay: float = None):
        """Write detailed header to log file"""
        header_lines = [
            "=" * 80,
            "AGENT NETWORK SIMULATION LOG",
            "=" * 80,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Config File: {getattr(self, 'config_file_path', 'Unknown')}",
            f"Network Topology: {network_file or 'Default/Embedded'}",
            f"Pilot Prompts: {prompts_file or 'Default/Generated'}",
            "",
            "SIMULATION PARAMETERS:",
            f"  Rounds: {rounds or 'N/A'}",
            f"  Max Turns per Conversation: {max_turns or 'N/A'}",
            f"  Delay Between Rounds: {delay or 'N/A'}s",
            "",
            "NETWORK CONFIGURATION:",
            f"  Total Agents: {len(self.agents)}",
            f"  Total Connections: {self.graph.number_of_edges()}",
            "",
            "AGENTS:"
        ]
        
        # Add agent details - only show connected agents
        connected_agents = set()
        for agent_id in self.agents:
            neighbors_out = self.get_neighbors(agent_id)
            neighbors_in = self.get_incoming_neighbors(agent_id)
            if neighbors_out or neighbors_in:
                connected_agents.add(agent_id)
                connected_agents.update(neighbors_out)
                connected_agents.update(neighbors_in)
        
        for agent_id in connected_agents:
            if agent_id in self.config.nodes:
                config = self.config.nodes[agent_id]
                header_lines.extend([
                    f"  {agent_id}:",
                    f"    Name: {config.get('name', agent_id)}",
                    f"    Model: {config.get('provider', 'unknown')}/{config.get('model', 'unknown')}",
                    f"    Role: {'Pilot' if 'pilot' in agent_id else 'Assistant'}",
                    f"    Temperature: {config.get('temperature', 'N/A')}",
                    f"    Can send to: {self.get_neighbors(agent_id)}",
                    ""
                ])
        
        if len(connected_agents) < len(self.agents):
            disconnected = len(self.agents) - len(connected_agents)
            header_lines.extend([
                f"  ({disconnected} agents have no connections and won't participate)",
                ""
            ])
        
        header_lines.extend([
            "=" * 80,
            "SIMULATION START",
            "=" * 80,
            ""
        ])
        
        # Write to file
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header_lines))
    
    def _log_verbose(self, message: str, level: str = "INFO"):
        """Simplified verbose logging - only log important events"""
        # Only log important battleship events, skip conversation details
        important_keywords = ["ATTACK", "HIT", "MISS", "SUNK", "GAME_OVER", "ROUND", "PHASE"]
        
        if any(keyword in message.upper() for keyword in important_keywords):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {level}: {message}"
            
            # Write to file if available
            if self.current_log_file:
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            
            # Also log to console for important events only
            if level == "ERROR":
                logger.error(message)
            elif "ATTACK" in message or "SUNK" in message or "GAME_OVER" in message:
                logger.info(message)

    
    def can_communicate(self, sender: str, receiver: str) -> bool:
        """Check if two agents can communicate based on network topology"""
        return self.graph.has_edge(sender, receiver)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all agents that this agent can send messages to"""
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id))
    
    def get_incoming_neighbors(self, node_id: str) -> List[str]:
        """Get all agents that can send messages to this agent"""
        if node_id not in self.graph:
            return []
        return list(self.graph.predecessors(node_id))
    
    async def send_message(self, sender_id: str, receiver_id: str, message: str, max_turns: int = 2) -> Dict[str, Any]:
        """Send message between two agents with custom conversation tracking"""
        # Validate agents exist
        if sender_id not in self.agents:
            raise ValueError(f"Sender agent '{sender_id}' not found")
        if receiver_id not in self.agents:
            raise ValueError(f"Receiver agent '{receiver_id}' not found")
        
        # Check if communication is allowed
        if not self.can_communicate(sender_id, receiver_id):
            raise ValueError(f"Communication not allowed: {sender_id} -> {receiver_id}")
        
        # Initialize conversation tracker if not exists
        if not hasattr(self, 'conversation_tracker'):
            self.conversation_tracker = ConversationTracker()
        
        sender_agent = self.agents[sender_id]
        receiver_agent = self.agents[receiver_id]
        
        try:
            conversation_start = datetime.now()
            
            # Start tracking conversation
            conv_id = self.conversation_tracker.start_conversation(sender_id, receiver_id, message)
            
            # Start AutoGen conversation
            chat_result = sender_agent.initiate_chat(
                recipient=receiver_agent,
                message=message,
                max_turns=max_turns,
                verbose=False 
            )
            
            # Extract and track all messages from chat_result
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in chat_result.chat_history:
                    speaker = msg.get('name', 'unknown')
                    content = msg.get('content', '')
                    
                    if speaker == sender_id:
                        self.conversation_tracker.add_message(conv_id, speaker, content, 'request')
                    elif speaker == receiver_id:
                        self.conversation_tracker.add_message(conv_id, speaker, content, 'response')
            
            # Complete conversation tracking
            self.conversation_tracker.complete_conversation(conv_id)
            
            conversation_end = datetime.now()
            duration = (conversation_end - conversation_start).total_seconds()
            
            # Store in history (keep existing format for compatibility)
            conversation_record = {
                'sender': sender_id,
                'receiver': receiver_id,
                'initial_message': message,
                'chat_result': chat_result,
                'duration_seconds': duration,
                'timestamp': asyncio.get_event_loop().time(),
                'conversation_start': conversation_start.isoformat(),
                'conversation_end': conversation_end.isoformat(),
                'custom_tracking_id': conv_id  # Reference to our tracking
            }
            
            self.conversation_history.append(conversation_record)
            
            return conversation_record
            
        except Exception as e:
            self._log_verbose(f"❌ CONVERSATION FAILED: {sender_id} -> {receiver_id}", "ERROR")
            raise
    
    def _log_detailed_conversation(self, sender_id: str, receiver_id: str, chat_result, duration: float):
        """Log detailed conversation with all messages - FILE ONLY, not console"""
        # Only log to file, not console
        if not self.current_log_file:
            return
            
        log_lines = [
            f"📋 DETAILED CONVERSATION LOG",
            f"🔄 Participants: {sender_id} <-> {receiver_id}",
            f"⏱️  Duration: {duration:.2f} seconds",
            ""
        ]
        
        if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
            log_lines.append("💬 CHAT HISTORY:")
            for i, msg in enumerate(chat_result.chat_history, 1):
                speaker = msg.get('name', msg.get('role', 'Unknown'))
                content = msg.get('content', '')
                
                log_lines.append(f"  Message {i} - {speaker}:")
                # Split long messages into multiple lines for readability
                if len(content) > 100:
                    lines = [content[i:i+100] for i in range(0, len(content), 100)]
                    for line in lines:
                        log_lines.append(f"    {line}")
                else:
                    log_lines.append(f"    {content}")
                log_lines.append("")
        
        # Log summary if available
        if hasattr(chat_result, 'summary') and chat_result.summary:
            log_lines.extend([
                "📄 CONVERSATION SUMMARY:",
                f"  {chat_result.summary}",
                ""
            ])
        
        # Log cost info if available
        if hasattr(chat_result, 'cost') and chat_result.cost:
            log_lines.append("💰 COST INFORMATION:")
            for key, value in chat_result.cost.items():
                log_lines.append(f"  {key}: {value}")
            log_lines.append("")
        
        # Write to file only
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(log_lines) + '\n')    
    
    def broadcast_message(self, sender_id: str, message: str, max_turns: int = 2) -> List[Dict[str, Any]]:
        """Send message to all connected agents"""
        if sender_id not in self.agents:
            raise ValueError(f"Sender agent '{sender_id}' not found")
        
        neighbors = self.get_neighbors(sender_id)
        if not neighbors:
            logger.warning(f"Agent '{sender_id}' has no outgoing connections")
            return []
        
        results = []
        for receiver_id in neighbors:
            try:
                result = asyncio.create_task(
                    self.send_message(sender_id, receiver_id, message, max_turns)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Broadcast failed to {receiver_id}: {e}")
        
        # Wait for all to complete
        completed_results = []
        for task in results:
            try:
                completed_results.append(asyncio.get_event_loop().run_until_complete(task))
            except Exception as e:
                logger.error(f"Broadcast task failed: {e}")
        
        return completed_results
    
    async def run_simulation_round(self, 
                                  initial_messages: List[Dict[str, str]] = None,
                                  pilot_prompts: Dict[str, str] = None,
                                  max_turns: int = 2) -> Dict[str, Any]:
        """
        Run a single simulation round with initial messages or pilot prompts
        
        Args:
            initial_messages: List of {"sender": "agent_id", "receiver": "agent_id", "message": "text"}
            pilot_prompts: Dict of {"pilot_agent_id": "task_prompt"} - will auto-assign to assistants
            max_turns: Maximum turns per conversation
            
        Returns:
            Round results with all conversations and metadata
        """
        self.current_round += 1
        round_start_time = asyncio.get_event_loop().time()
        
        self._log_verbose(f"🎯 Starting simulation round {self.current_round}")
        
        # Generate messages from pilot prompts if provided
        if pilot_prompts:
            initial_messages = self._generate_messages_from_pilot_prompts(pilot_prompts)
            self._log_verbose(f"   Generated {len(initial_messages)} messages from pilot prompts")
        elif initial_messages:
            self._log_verbose(f"   Using {len(initial_messages)} provided messages")
        else:
            raise ValueError("Must provide either initial_messages or pilot_prompts")
        
        round_conversations = []
        
        for msg_config in initial_messages:
            try:
                sender = msg_config["sender"]
                receiver = msg_config["receiver"]
                message = msg_config["message"]
                
                self._log_verbose(f"   📤 Initiating: {sender} -> {receiver}")
                
                conversation = await self.send_message(sender, receiver, message, max_turns)
                round_conversations.append(conversation)
                
            except Exception as e:
                self._log_verbose(f"   ❌ Failed message {sender} -> {receiver}: {e}", "ERROR")
                round_conversations.append({
                    'sender': msg_config.get("sender", "unknown"),
                    'receiver': msg_config.get("receiver", "unknown"),
                    'initial_message': msg_config.get("message", ""),
                    'error': str(e),
                    'timestamp': asyncio.get_event_loop().time()
                })
        
        round_end_time = asyncio.get_event_loop().time()
        round_duration = round_end_time - round_start_time
        
        round_result = {
            'round_number': self.current_round,
            'start_time': round_start_time,
            'end_time': round_end_time,
            'duration_seconds': round_duration,
            'conversations': round_conversations,
            'success_count': len([c for c in round_conversations if 'error' not in c]),
            'error_count': len([c for c in round_conversations if 'error' in c]),
            'total_messages': len(initial_messages),
            'pilot_prompts': pilot_prompts if pilot_prompts else None
        }
        
        self.simulation_rounds.append(round_result)
        
        self._log_verbose(f"✅ Round {self.current_round} completed in {round_duration:.2f}s")
        self._log_verbose(f"   Success: {round_result['success_count']}/{round_result['total_messages']}")
        
        return round_result
    
    def _generate_messages_from_pilot_prompts(self, pilot_prompts: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate messages by assigning pilot prompts to their connected assistants"""
        messages = []
        
        for pilot_id, task_prompt in pilot_prompts.items():
            if pilot_id not in self.agents:
                continue  # Skip silently
            
            # Find assistants this pilot can communicate with
            assistants = self.get_neighbors(pilot_id)
            
            if not assistants:
                continue  # Skip silently - don't warn about disconnected pilots
            
            # Send the task prompt to each connected assistant
            for assistant_id in assistants:
                messages.append({
                    "sender": pilot_id,
                    "receiver": assistant_id,
                    "message": task_prompt
                })
                self._log_verbose(f"   📋 Task assigned: {pilot_id} -> {assistant_id}")
        
        return messages
    
    async def run_simulation(self, 
                           rounds: int,
                           messages_per_round: List[Dict[str, str]] = None,
                           pilot_prompts_per_round: Dict[str, str] = None,
                           pilot_prompts_file: str = None,
                           message_generator: callable = None,
                           max_turns: int = 2,
                           delay_between_rounds: float = 0.0,
                           network_file: str = None) -> Dict[str, Any]:
        """
        Run multiple simulation rounds with detailed logging
        
        Args:
            rounds: Number of rounds to run
            messages_per_round: Static messages to send each round, or None
            pilot_prompts_per_round: Static pilot prompts to use each round, or None
            pilot_prompts_file: File containing pilot prompts (txt/json), or None
            message_generator: Function that generates messages for each round: f(round_num, network) -> List[Dict]
            max_turns: Maximum turns per conversation
            delay_between_rounds: Seconds to wait between rounds
            network_file: Network topology file name (for logging)
            
        Returns:
            Complete simulation results
        """
        # Initialize logging
        log_file = self._initialize_logging(
            network_file=network_file,
            prompts_file=pilot_prompts_file,
            rounds=rounds,
            max_turns=max_turns,
            delay=delay_between_rounds
        )
        
        # Load pilot prompts from file if provided
        if pilot_prompts_file:
            pilot_prompts_per_round = self._load_pilot_prompts_from_file(pilot_prompts_file)
            self._log_verbose(f"📋 Loaded pilot prompts from {pilot_prompts_file}")
            
            # Log only prompts for active pilots
            active_pilots = [p for p in pilot_prompts_per_round.keys() if self.get_neighbors(p)]
            if active_pilots:
                self._log_verbose("🎯 ACTIVE PILOT PROMPTS:")
                for pilot_id in active_pilots:
                    prompt = pilot_prompts_per_round[pilot_id]
                    self._log_verbose(f"  {pilot_id}:")
                    prompt_lines = prompt.split('. ')
                    for line in prompt_lines:
                        if line.strip():
                            self._log_verbose(f"    {line.strip()}.")
                    self._log_verbose("")
            else:
                self._log_verbose("⚠️  No active pilots found with connections")
        
        # Validate inputs
        input_count = sum([
            bool(messages_per_round),
            bool(pilot_prompts_per_round), 
            bool(message_generator)
        ])
        
        if input_count != 1:
            raise ValueError("Must provide exactly one of: messages_per_round, pilot_prompts_per_round, or message_generator")
        
        simulation_start_time = asyncio.get_event_loop().time()
        
        self._log_verbose(f"🚀 SIMULATION START")
        self._log_verbose(f"📊 Parameters: {rounds} rounds, {max_turns} max turns, {delay_between_rounds}s delay")
        self._log_verbose("")
        
        all_rounds = []
        
        for round_num in range(1, rounds + 1):
            self._log_verbose(f"🎯 ===== ROUND {round_num}/{rounds} =====")
            
            # Determine what to use for this round
            if message_generator:
                round_messages = message_generator(round_num, self)
                round_result = await self.run_simulation_round(
                    initial_messages=round_messages, 
                    max_turns=max_turns
                )
            elif pilot_prompts_per_round:
                round_result = await self.run_simulation_round(
                    pilot_prompts=pilot_prompts_per_round,
                    max_turns=max_turns
                )
            else:  # messages_per_round
                round_result = await self.run_simulation_round(
                    initial_messages=messages_per_round,
                    max_turns=max_turns
                )
            
            all_rounds.append(round_result)
            
            # Log round summary
            self._log_verbose(f"📈 ROUND {round_num} SUMMARY:")
            self._log_verbose(f"  ✅ Successful conversations: {round_result['success_count']}")
            self._log_verbose(f"  ❌ Failed conversations: {round_result['error_count']}")
            self._log_verbose(f"  ⏱️  Round duration: {round_result['duration_seconds']:.2f}s")
            self._log_verbose("")
            
            # Delay before next round (except for last round)
            if delay_between_rounds > 0 and round_num < rounds:
                self._log_verbose(f"⏳ Waiting {delay_between_rounds}s before next round...")
                await asyncio.sleep(delay_between_rounds)
        
        simulation_end_time = asyncio.get_event_loop().time()
        total_duration = simulation_end_time - simulation_start_time
        
        # Calculate statistics
        total_conversations = sum(r['total_messages'] for r in all_rounds)
        total_successes = sum(r['success_count'] for r in all_rounds)
        total_errors = sum(r['error_count'] for r in all_rounds)
        
        simulation_result = {
            'total_rounds': rounds,
            'total_duration_seconds': total_duration,
            'total_conversations': total_conversations,
            'total_successes': total_successes,
            'total_errors': total_errors,
            'success_rate': total_successes / total_conversations if total_conversations > 0 else 0,
            'avg_round_duration': total_duration / rounds,
            'network_info': self.get_network_info(),
            'log_file': str(log_file)
        }
        
        # Write final summary to log
        self._write_final_summary(simulation_result)
        
        self._log_verbose("🏁 SIMULATION COMPLETED!")
        self._log_verbose(f"📝 Complete log saved to: {log_file}")
        
        return simulation_result
    
    def _write_final_summary(self, simulation_result: Dict[str, Any]):
        """Write final summary to log file"""
        try:
            summary_lines = [
                "",
                "=" * 80,
                "SIMULATION SUMMARY", 
                "=" * 80,
                f"Total Rounds: {simulation_result.get('total_rounds', 'N/A')}",
                f"Total Duration: {simulation_result.get('total_duration_seconds', 0):.2f} seconds",
                f"Total Conversations: {simulation_result.get('total_conversations', 0)}",
                f"Successful Conversations: {simulation_result.get('total_successes', 0)}",
                f"Failed Conversations: {simulation_result.get('total_errors', 0)}",
                f"Success Rate: {simulation_result.get('success_rate', 0):.1%}",
                f"Average Round Duration: {simulation_result.get('avg_round_duration', 0):.2f} seconds",
                "",
                "ROUND-BY-ROUND BREAKDOWN:",
            ]
            
            rounds_data = simulation_result.get('rounds', [])
            for i, round_data in enumerate(rounds_data, 1):
                summary_lines.extend([
                    f"  Round {i}:",
                    f"    Duration: {round_data.get('duration_seconds', 0):.2f}s",
                    f"    Success: {round_data.get('success_count', 0)}/{round_data.get('total_messages', 0)}",
                    f"    Success Rate: {round_data.get('success_count', 0)/max(round_data.get('total_messages', 1), 1):.1%}",
                    ""
                ])
            
            summary_lines.extend([
                "=" * 80,
                "END OF SIMULATION LOG",
                "=" * 80
            ])
            
            # Write to file
            if self.current_log_file:
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(summary_lines))
                    
        except Exception as e:
            # Fallback summary if there's an error
            fallback_lines = [
                "",
                "=" * 80,
                "SIMULATION SUMMARY (FALLBACK)",
                "=" * 80,
                "Simulation completed but summary generation failed.",
                f"Error: {str(e)}",
                "Check the detailed logs above for conversation results.",
                "=" * 80,
                "END OF SIMULATION LOG", 
                "=" * 80
            ]
            
            if self.current_log_file:
                with open(self.current_log_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(fallback_lines))
    
    def _load_pilot_prompts_from_file(self, prompts_file: str) -> Dict[str, str]:
        """Load pilot prompts from file
        
        Supported formats:
        - JSON: {"pilot_id": "task prompt", ...}
        - Text: pilot_id: task prompt (one per line)
        """
        prompts = {}
        
        with open(prompts_file, 'r') as f:
            content = f.read().strip()
        
        # Try JSON first
        if content.startswith('{'):
            try:
                prompts = json.loads(content)
                return prompts
            except json.JSONDecodeError:
                pass
        
        # Parse as text format
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                pilot_id, task_prompt = line.split(':', 1)
                prompts[pilot_id.strip()] = task_prompt.strip()
            else:
                logger.warning(f"Invalid prompt format on line {line_num}: {line}")
        
        return prompts
    
    def reset_simulation(self):
        """Reset simulation state for a fresh start"""
        self.simulation_rounds = []
        self.conversation_history = []
        self.current_round = 0
        self.verbose_logs = []
        logger.info("🔄 Simulation state reset")
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary statistics from all simulation rounds"""
        if not self.simulation_rounds:
            return {"message": "No simulation rounds completed yet"}
        
        total_rounds = len(self.simulation_rounds)
        total_conversations = sum(r['total_messages'] for r in self.simulation_rounds)
        total_successes = sum(r['success_count'] for r in self.simulation_rounds)
        total_errors = sum(r['error_count'] for r in self.simulation_rounds)
        total_duration = sum(r['duration_seconds'] for r in self.simulation_rounds)
        
        return {
            'total_rounds_completed': total_rounds,
            'total_conversations': total_conversations,
            'total_successes': total_successes,
            'total_errors': total_errors,
            'overall_success_rate': total_successes / total_conversations if total_conversations > 0 else 0,
            'total_simulation_time': total_duration,
            'avg_round_duration': total_duration / total_rounds if total_rounds > 0 else 0,
            'avg_conversations_per_round': total_conversations / total_rounds if total_rounds > 0 else 0
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network structure"""
        return {
            'num_agents': len(self.agents),
            'num_connections': self.graph.number_of_edges(),
            'agents': {
                agent_id: {
                    'name': self.config.nodes[agent_id].get('name', agent_id),
                    'model': self.config.nodes[agent_id]['model'],
                    'provider': self.config.nodes[agent_id]['provider'],
                    'can_send_to': self.get_neighbors(agent_id),
                    'can_receive_from': self.get_incoming_neighbors(agent_id)
                }
                for agent_id in self.agents.keys()
            },
            'conversation_count': len(self.conversation_history),
            'simulation_rounds_completed': len(self.simulation_rounds)
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def visualize_network(self):
        """Print a simple visualization of ONLY the agents involved in simulation"""
        print("\n🕸️  Active Network Topology:")
        print("=" * 50)
        
        # Get only agents that have actual connections in this simulation
        active_agents = set()
        
        # Find agents that can send messages
        for agent_id in self.agents:
            neighbors_out = self.get_neighbors(agent_id)
            if neighbors_out:
                active_agents.add(agent_id)
                active_agents.update(neighbors_out)
        
        # Also check for agents that can receive messages
        for agent_id in self.agents:
            neighbors_in = self.get_incoming_neighbors(agent_id)
            if neighbors_in:
                active_agents.add(agent_id)
                active_agents.update(neighbors_in)
        
        if not active_agents:
            print("⚠️  No active connections found!")
            return
        
        # Show only active agents with their roles
        pilots_shown = []
        assistants_shown = []
        
        for agent_id in sorted(active_agents):
            if agent_id in self.agents:
                agent_info = self.config.nodes[agent_id]
                name = agent_info.get('name', agent_id)
                model = agent_info['model']
                provider = agent_info['provider']
                
                # Determine if pilot or assistant
                role = "🎯 PILOT" if 'pilot' in agent_id else "🤖 ASSISTANT"
                
                neighbors_out = self.get_neighbors(agent_id)
                neighbors_in = self.get_incoming_neighbors(agent_id)
                
                print(f"{role}: {name}")
                print(f"   Model: {provider}/{model}")
                
                if neighbors_out:
                    print(f"   → Sends to: {neighbors_out}")
                if neighbors_in:
                    print(f"   ← Receives from: {neighbors_in}")
                print()
                
                # Track for summary
                if 'pilot' in agent_id:
                    pilots_shown.append(agent_id)
                else:
                    assistants_shown.append(agent_id)
        
        # Simplified summary
        print(f"📊 Active: {len(pilots_shown)} pilots, {len(assistants_shown)} assistants")
        print(f"🔗 Total connections: {self.graph.number_of_edges()}")
        
        # Hide inactive agents warning unless there are many
        inactive_count = len(self.agents) - len(active_agents)
        if inactive_count > 0:
            print(f"💤 ({inactive_count} inactive agents hidden)")