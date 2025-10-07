"""
agent_network.py — Clean Simultaneous Battleship Agent Network

Essential features for 3-phase Alpha coordination vs Beta individual decisions:
1. Team Strategy Proposals (Parallel Broadcast)
2. Democratic Voting (Sequential)
3. Plan Selection (Majority rule with tie-breaking)

Core functionality only - no bloat.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
from autogen import ConversableAgent, LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

@contextmanager
def _suppress_autogen_spam():
    """Suppress AutoGen noise"""
    class _Filter(io.StringIO):
        def write(self, s):
            return 0 if "TERMINATING" in s else super().write(s)
    
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = _Filter()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _normalize_ids(text: str, candidates: List[str]) -> str:
    """Map aliases like 'Alpha-2' to 'player_a2'"""
    if not text:
        return text
    
    # Build alias mapping
    alias_map = {}
    for cid in candidates:
        m = re.search(r'player_(a|b)(\d+)', cid, re.I)
        if m:
            team, num = m.group(1).lower(), m.group(2)
            aliases = [f"{team}{num}", f"{team}-{num}", f"alpha{num}" if team == "a" else f"beta{num}"]
            for alias in aliases:
                alias_map[alias.lower()] = cid
    
    # Replace aliases
    def replace(match):
        return alias_map.get(match.group(0).lower(), match.group(0))
    
    return re.sub(r'\b(?:alpha|beta|a|b)[\s_-]*\d+\b', replace, text, flags=re.I)


# ---------------------------------------------------------------------
# Minimal Conversation Tracking
# ---------------------------------------------------------------------

class ConversationTracker:
    """Minimal tracking for advice extraction"""
    
    def __init__(self):
        self.conversations = []
    
    def add_conversation(self, sender: str, receiver: str, content: str):
        """Add conversation for advice tracking"""
        self.conversations.append({
            "sender": sender, "receiver": receiver, "content": content,
            "timestamp": asyncio.get_event_loop().time()
        })
        # Keep only recent conversations
        if len(self.conversations) > 50:
            self.conversations = self.conversations[-25:]
    
    def get_recent_advice_for_player(self, player_id: str) -> List[str]:
        """Get recent action suggestions for player"""
        advice = []
        for conv in self.conversations[-20:]:
            if player_id in (conv["sender"], conv["receiver"]):
                content = conv["content"]
                # Extract BOMB/MOVE suggestions
                bomb_match = re.search(r'\bBOMB\s+([A-J](?:10|[1-9]))\b', content, re.I)
                if bomb_match:
                    advice.append(f"BOMB {bomb_match.group(1)}")
                move_match = re.search(r'\bMOVE\s+(UP|DOWN|LEFT|RIGHT|ROTATE)\b', content, re.I)
                if move_match:
                    advice.append(f"MOVE {move_match.group(1)}")
        return advice[-3:]  # Last 3 pieces of advice


# ---------------------------------------------------------------------
# Network Configuration
# ---------------------------------------------------------------------

@dataclass
class NetworkConfig:
    nodes: Dict[str, Dict[str, Any]]
    edges: List[List[str]]
    global_config: Dict[str, Any] = None
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path], edges_file: Optional[Union[str, Path]] = None):
        data = json.loads(Path(config_path).read_text())
        
        if edges_file and Path(edges_file).exists():
            edges_text = Path(edges_file).read_text().strip()
            if edges_text.startswith("["):
                edges = json.loads(edges_text)
            else:
                edges = []
                for line in edges_text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "->" in line:
                            a, b = line.split("->", 1)
                            edges.append([a.strip(), b.strip()])
                        elif "," in line:
                            parts = [p.strip() for p in line.split(",", 1)]
                            if len(parts) == 2:
                                edges.append(parts)
        else:
            edges = data.get("edges", [])
        
        return cls(data.get("nodes", {}), edges, data.get("global_config", {}))
    
    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node_id, attrs in self.nodes.items():
            graph.add_node(node_id, **attrs)
        for a, b in self.edges:
            graph.add_edge(a, b)
        return graph


# ---------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------

class LLMProvider:
    @staticmethod
    def create_config(provider: str, model: str, **kwargs) -> LLMConfig:
        provider = provider.lower()
        
        if provider == "ollama":
            return LLMConfig(
                model=model, api_type="ollama",
                client_host=kwargs.get("client_host", "http://localhost:11434"),
                temperature=kwargs.get("temperature", 0.1), stream=False
            )
        elif provider == "openai":
            return LLMConfig(
                model=model, api_type="openai", api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.1), max_tokens=kwargs.get("max_tokens", 1000)
            )
        elif provider in ("anthropic", "claude"):
            return LLMConfig(
                model=model, api_type="anthropic", api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.1), max_tokens=kwargs.get("max_tokens", 1000)
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------
# Main Agent Network
# ---------------------------------------------------------------------

class AgentNetwork:
    """Clean agent network for Simultaneous Battleship"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.graph = config.to_networkx()
        self.agents: Dict[str, ConversableAgent] = {}
        self.tracker = ConversationTracker()
        self._init_agents()
        logger.info("🎮 Agent Network initialized")
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], edges_file: Optional[Union[str, Path]] = None):
        return cls(NetworkConfig.from_file(config_path, edges_file))
    
    def _init_agents(self):
        """Initialize all agents"""
        global_config = self.config.global_config or {}
        provider_defaults = global_config.get("provider_defaults", {})
        api_keys = global_config.get("api_keys", {})
        role_definitions = global_config.get("role_definitions", {})

        for node_id, node_data in self.config.nodes.items():
            name = node_data.get("name", node_id)
            model = node_data["model"]
            provider = node_data["provider"]
            role = node_data.get("role", "player")
            
            # Determine team
            team = self._infer_team(node_id, node_data)
            team_name = "Alpha Squadron" if team == "a" else "Beta Fleet" if team == "b" else "Unknown Team"
            
            # Get team behavior instructions from node config
            team_behavior = node_data.get("team_behavior_instructions", "Coordinate effectively to win.")
            
            # Create system message using the template from your config
            if role == "player":
                tmpl = role_definitions.get("player", {}).get("system_message_template", 
                    "You are {name} on {team_name}. {team_behavior_instructions}")
                try:
                    system_message = tmpl.format(
                        name=name, 
                        team_name=team_name,
                        team_behavior_instructions=team_behavior,
                        additional_instructions="Be strategic and decisive."
                    )
                except Exception as e:
                    logger.warning(f"Template formatting failed for {node_id}: {e}")
                    system_message = f"You are {name} on {team_name}. {team_behavior}"
            else:
                system_message = f"You are {name}, a {role}."
            
            # Build LLM config
            provider_config = dict(provider_defaults.get(provider, {}))
            if "api_key" not in node_data and provider in api_keys:
                provider_config["api_key"] = api_keys[provider]
            
            # Node-level overrides
            for key, value in node_data.items():
                if key not in {"name", "model", "provider", "role", "team", "team_behavior_instructions"}:
                    provider_config[key] = value
            
            llm_config = LLMProvider.create_config(provider, model, **provider_config)
            
            agent = ConversableAgent(
                name=node_id, system_message=system_message, llm_config=llm_config,
                human_input_mode="NEVER", max_consecutive_auto_reply=1,
                description=f"{name} ({provider}/{model})"
            )
            
            self.agents[node_id] = agent
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get structured network information"""
        return {
            "total_agents": len(self.agents),
            "total_connections": self.graph.number_of_edges(),
            "alpha_squadron": self.get_team_members("a"),
            "beta_fleet": self.get_team_members("b"),
        }
    
    def _infer_team(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Infer team from node ID or data"""
        explicit = (node_data.get("team") or "").lower()
        if explicit in ("a", "b", "alpha", "beta"):
            return explicit[0]
        
        nid = node_id.lower()
        if any(p in nid for p in ["a_", "player_a", "alpha"]):
            return "a"
        if any(p in nid for p in ["b_", "player_b", "beta"]):
            return "b"
        return ""
    
    def can_communicate(self, sender: str, receiver: str) -> bool:
        """Check if direct communication is allowed"""
        return sender == "game_master" or self.graph.has_edge(sender, receiver)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get communication neighbors"""
        if node_id == "game_master":
            return list(self.agents.keys())
        return list(self.graph.successors(node_id)) if node_id in self.graph else []
    
    def get_team_members(self, team_prefix: str) -> List[str]:
        """Get team members (a/alpha or b/beta)"""
        target_team = team_prefix.lower()[0]
        members = []
        for node_id, node_data in self.config.nodes.items():
            if self._infer_team(node_id, node_data) == target_team and "player" in node_id.lower():
                members.append(node_id)
        return members
    
    def visualize_network(self):
        """Print network topology"""
        alpha = self.get_team_members("a")
        beta = self.get_team_members("b")
        
        logger.info("🎮 Simultaneous Battleship Network")
        logger.info(f"🔵 Alpha Squadron (Coordinated): {alpha}")
        logger.info(f"🔴 Beta Fleet (Individual): {beta}")
    
    async def send_message(self, sender_id: str, receiver_id: str, message: str, *,
                          max_turns: int = 1, conversation_type: str = "general") -> Dict[str, Any]:
        """Send message between agents or from GM"""
        message = str(message or "").strip()
        
        # GM → Agent (silent)
        if sender_id == "game_master":
            return await self._send_gm_message(receiver_id, message, conversation_type, max_turns)
        
        # Agent → Agent (logged)
        if sender_id not in self.agents or receiver_id not in self.agents:
            raise ValueError(f"Unknown agent: {sender_id} or {receiver_id}")
        if not self.can_communicate(sender_id, receiver_id):
            raise ValueError(f"No communication edge: {sender_id} → {receiver_id}")
        
        # Log the outgoing message (FULL MESSAGE - NO TRUNCATION)
        logger.info(f"🗣️ {sender_id} → {receiver_id}: {message}")
        
        with _suppress_autogen_spam():
            chat_result = await asyncio.to_thread(
                self.agents[sender_id].initiate_chat,
                recipient=self.agents[receiver_id], message=message, max_turns=max_turns, verbose=False
            )
        
        # Log the response (FULL RESPONSE - NO TRUNCATION)
        if hasattr(chat_result, "chat_history"):
            for msg in chat_result.chat_history:
                if msg.get("name") == receiver_id:
                    response_content = msg.get("content", "")
                    logger.info(f"💬 {receiver_id} replied: {response_content}")
                    break
        
        # Track conversation for advice
        self.tracker.add_conversation(sender_id, receiver_id, message)
        
        return {"sender": sender_id, "receiver": receiver_id, "chat_result": chat_result}
    
    # agent_network.py
    async def _send_gm_message(self, receiver_id: str, message: str, conv_type: str, max_turns: int):
        if receiver_id not in self.agents:
            raise ValueError(f"Unknown agent: {receiver_id}")

        with _suppress_autogen_spam():
            response = await asyncio.to_thread(
                self.agents[receiver_id].generate_reply,
                messages=[{"role": "user", "content": message}],
                sender=None
            )

        # Fallback if the provider returns None (observed in Beta Round 2)
        if response is None:
            # Create a tiny, ephemeral GM agent and use initiate_chat (reliable)
            gm = ConversableAgent(
                name="game_master_ephemeral",
                system_message="You are the game master issuing a single instruction.",
                llm_config=self.agents[receiver_id].llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
            )
            with _suppress_autogen_spam():
                chat_result = await asyncio.to_thread(
                    gm.initiate_chat,
                    recipient=self.agents[receiver_id],
                    message=message,
                    max_turns=1,
                    verbose=False
                )
            # Extract agent reply from chat_result for logging + downstream parsing
            if hasattr(chat_result, "chat_history"):
                for msg in chat_result.chat_history:
                    if msg.get("name") == receiver_id:
                        content = msg.get("content", "")
                        logger.info(f"💬 {receiver_id}: {content}")
                        # Track GM conversation with the actual content
                        self.tracker.add_conversation("game_master", receiver_id, content)
                        return {"sender": "game_master", "receiver": receiver_id, "chat_result": chat_result}

            # If somehow still nothing, fall back to a sentinel
            content = "No response generated"

        else:
            # Original path: extract content from 'response'
            if isinstance(response, dict):
                content = response.get("content", str(response))
            elif isinstance(response, str):
                content = response
            elif hasattr(response, "content"):
                content = str(response.content)
            else:
                content = str(response)

            logger.info(f"💬 {receiver_id}: {content}")
            class MockChat:
                chat_history = [
                    {"name": "game_master", "content": message},
                    {"name": receiver_id, "content": content}
                ]

            self.tracker.add_conversation("game_master", receiver_id, content)
            return {"sender": "game_master", "receiver": receiver_id, "chat_result": MockChat()}

        # Default final fallback
        logger.info(f"💬 {receiver_id}: {content}")
        class MockChat2:
            chat_history = [
                {"name": "game_master", "content": message},
                {"name": receiver_id, "content": content}
            ]
        self.tracker.add_conversation("game_master", receiver_id, content)
        return {"sender": "game_master", "receiver": receiver_id, "chat_result": MockChat2()}

    
    async def send_team_message(self, sender_id: str, recipients: List[str], message: str, *,
                               conversation_type: str = "team_broadcast", max_turns: int = 1) -> List[Dict[str, Any]]:
        """Send message to multiple teammates"""
        message = _normalize_ids(str(message), recipients)
        results = []
        
        # Log team broadcast (FULL MESSAGE - NO TRUNCATION)
        logger.info(f"📢 {sender_id} → TEAM [{', '.join(recipients)}]: {message}")
        
        for recipient in recipients:
            if recipient == sender_id:
                continue
            
            try:
                if self.can_communicate(sender_id, recipient):
                    result = await self.send_message(sender_id, recipient, message, 
                                                   max_turns=max_turns, conversation_type=conversation_type)
                else:
                    # GM relay for agents without direct connection
                    relay_msg = f"[RELAY FROM {sender_id}] {message}"
                    result = await self._send_gm_message(recipient, relay_msg, conversation_type, 1)
                    logger.info(f"🔄 {sender_id} → {recipient} (via GM relay)")
                
                results.append(result)
            except Exception as e:
                logger.error(f"Team message failed {sender_id} → {recipient}: {e}")
                results.append({"error": str(e), "sender": sender_id, "receiver": recipient})
        
        return results
    
    def get_recent_advice_for_player(self, player_id: str) -> List[str]:
        """Get recent advice for player from conversations"""
        return self.tracker.get_recent_advice_for_player(player_id)


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def create_battleship_network(llm_config_path: str, edges_file: Optional[str] = None) -> AgentNetwork:
    """Create battleship network from config files"""
    network = AgentNetwork.from_config(llm_config_path, edges_file)
    network.visualize_network()
    return network


if __name__ == "__main__":
    print("🎮 Simultaneous Battleship Agent Network")
    print("Use via battleship_runner.py for game execution.")