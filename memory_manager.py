"""
Comprehensive Memory Management System for Battleship AI Agents

This system provides persistent, searchable memory for agents including:
- Battlefield coordinate tracking (hits/misses/sunk ships)
- Player behavior patterns and profiles
- Conversation history and advice received
- Game state and strategic insights

Designed to be easily extensible for future memory types.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories that can be stored"""
    BATTLEFIELD = "battlefield"          # Coordinate attacks and results
    PLAYER_PROFILE = "player_profile"    # Observed player behaviors
    CONVERSATION = "conversation"        # Important conversations and advice
    STRATEGIC = "strategic"             # Strategic insights and patterns
    GAME_STATE = "game_state"          # Current game state information


@dataclass
class CoordinateMemory:
    """Memory of a specific coordinate attack"""
    coordinate: str                      # e.g., "A1"
    attacker: str                       # Who attacked
    target_team: str                    # Which team was targeted
    result: str                         # "HIT", "MISS", "SUNK"
    sunk_ship: Optional[str] = None     # Ship name if sunk
    round_number: int = 0               # When it happened
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_natural_language(self) -> str:
        """Convert to human-readable format"""
        base = f"{self.attacker} attacked {self.coordinate} → {self.result}"
        if self.sunk_ship:
            base += f" (sunk {self.sunk_ship})"
        return base


@dataclass
class PlayerBehaviorMemory:
    """Memory of observed player behavior patterns"""
    player_id: str
    behavior_type: str                  # "coordinate_preference", "risk_tolerance", etc.
    observations: List[str] = field(default_factory=list)
    confidence: float = 0.5             # How confident we are in this pattern
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_observation(self, observation: str):
        """Add a new behavioral observation"""
        self.observations.append(observation)
        self.last_updated = datetime.now().isoformat()
        # Increase confidence with more observations (max 1.0)
        self.confidence = min(1.0, self.confidence + 0.1)


@dataclass
class ConversationMemory:
    """Memory of important conversations and advice"""
    conversation_id: str
    participants: List[str]
    key_points: List[str] = field(default_factory=list)
    coordinates_mentioned: List[str] = field(default_factory=list)
    advice_given: Optional[str] = None
    advice_quality: str = "unknown"     # "good", "bad", "neutral"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StrategicMemory:
    """Memory of strategic insights and patterns"""
    insight_type: str                   # "enemy_pattern", "team_strategy", etc.
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    reliability: float = 0.5           # How reliable this insight is
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentMemoryManager:
    """Comprehensive memory management for a single agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memories: Dict[MemoryType, List[Any]] = {
            MemoryType.BATTLEFIELD: [],
            MemoryType.PLAYER_PROFILE: [],
            MemoryType.CONVERSATION: [],
            MemoryType.STRATEGIC: [],
            MemoryType.GAME_STATE: []
        }
        self.memory_index: Dict[str, List[Tuple[MemoryType, int]]] = {}  # For fast lookup
        
    def add_coordinate_memory(self, coordinate: str, attacker: str, target_team: str, 
                            result: str, sunk_ship: Optional[str] = None, round_number: int = 0):
        """Add memory of a coordinate attack"""
        memory = CoordinateMemory(
            coordinate=coordinate,
            attacker=attacker,
            target_team=target_team,
            result=result,
            sunk_ship=sunk_ship,
            round_number=round_number
        )
        
        self.memories[MemoryType.BATTLEFIELD].append(memory)
        self._index_memory("coordinate", coordinate, MemoryType.BATTLEFIELD, len(self.memories[MemoryType.BATTLEFIELD]) - 1)
        self._index_memory("attacker", attacker, MemoryType.BATTLEFIELD, len(self.memories[MemoryType.BATTLEFIELD]) - 1)
        
        logger.debug(f"[MEMORY] {self.agent_id} recorded: {memory.to_natural_language()}")
    
    def add_player_behavior(self, player_id: str, behavior_type: str, observation: str):
        """Add or update player behavior observation"""
        # Look for existing behavior memory
        for memory in self.memories[MemoryType.PLAYER_PROFILE]:
            if memory.player_id == player_id and memory.behavior_type == behavior_type:
                memory.add_observation(observation)
                logger.debug(f"[MEMORY] {self.agent_id} updated {player_id} behavior: {observation}")
                return
        
        # Create new behavior memory
        memory = PlayerBehaviorMemory(
            player_id=player_id,
            behavior_type=behavior_type,
            observations=[observation]
        )
        self.memories[MemoryType.PLAYER_PROFILE].append(memory)
        self._index_memory("player", player_id, MemoryType.PLAYER_PROFILE, len(self.memories[MemoryType.PLAYER_PROFILE]) - 1)
        
        logger.debug(f"[MEMORY] {self.agent_id} recorded new {player_id} behavior: {observation}")
    
    def add_conversation_memory(self, conversation_id: str, participants: List[str], 
                              key_points: List[str], coordinates_mentioned: List[str] = None,
                              advice_given: str = None):
        """Add memory of an important conversation"""
        memory = ConversationMemory(
            conversation_id=conversation_id,
            participants=participants,
            key_points=key_points,
            coordinates_mentioned=coordinates_mentioned or [],
            advice_given=advice_given
        )
        
        self.memories[MemoryType.CONVERSATION].append(memory)
        for participant in participants:
            self._index_memory("participant", participant, MemoryType.CONVERSATION, len(self.memories[MemoryType.CONVERSATION]) - 1)
        
        logger.debug(f"[MEMORY] {self.agent_id} recorded conversation with {participants}")
    
    def add_strategic_memory(self, insight_type: str, description: str, evidence: List[str] = None):
        """Add strategic insight or pattern"""
        memory = StrategicMemory(
            insight_type=insight_type,
            description=description,
            supporting_evidence=evidence or []
        )
        
        self.memories[MemoryType.STRATEGIC].append(memory)
        self._index_memory("insight", insight_type, MemoryType.STRATEGIC, len(self.memories[MemoryType.STRATEGIC]) - 1)
        
        logger.debug(f"[MEMORY] {self.agent_id} recorded strategic insight: {description}")
    
    def _index_memory(self, key_type: str, key_value: str, memory_type: MemoryType, index: int):
        """Index memory for fast retrieval"""
        index_key = f"{key_type}:{key_value}"
        if index_key not in self.memory_index:
            self.memory_index[index_key] = []
        self.memory_index[index_key].append((memory_type, index))
    
    def get_coordinate_history(self, include_all_teams: bool = True) -> List[CoordinateMemory]:
        """Get all coordinate attack memories"""
        return [mem for mem in self.memories[MemoryType.BATTLEFIELD] if isinstance(mem, CoordinateMemory)]
    
    def get_coordinate_result(self, coordinate: str) -> Optional[CoordinateMemory]:
        """Get the result for a specific coordinate"""
        for memory in self.memories[MemoryType.BATTLEFIELD]:
            if isinstance(memory, CoordinateMemory) and memory.coordinate == coordinate:
                return memory
        return None
    
    def get_player_patterns(self, player_id: str) -> List[PlayerBehaviorMemory]:
        """Get observed behavior patterns for a specific player"""
        return [mem for mem in self.memories[MemoryType.PLAYER_PROFILE] 
                if isinstance(mem, PlayerBehaviorMemory) and mem.player_id == player_id]
    
    def get_successful_coordinates(self) -> List[str]:
        """Get coordinates that resulted in hits or sunk ships"""
        successful = []
        for memory in self.memories[MemoryType.BATTLEFIELD]:
            if isinstance(memory, CoordinateMemory) and memory.result in ["HIT", "SUNK"]:
                successful.append(memory.coordinate)
        return successful
    
    def get_failed_coordinates(self) -> List[str]:
        """Get coordinates that resulted in misses"""
        failed = []
        for memory in self.memories[MemoryType.BATTLEFIELD]:
            if isinstance(memory, CoordinateMemory) and memory.result == "MISS":
                failed.append(memory.coordinate)
        return failed
    
    def get_all_attempted_coordinates(self) -> Set[str]:
        """Get all coordinates that have been attempted"""
        attempted = set()
        for memory in self.memories[MemoryType.BATTLEFIELD]:
            if isinstance(memory, CoordinateMemory):
                attempted.add(memory.coordinate)
        return attempted
    
    def get_strategic_insights(self, insight_type: str = None) -> List[StrategicMemory]:
        """Get strategic insights, optionally filtered by type"""
        insights = [mem for mem in self.memories[MemoryType.STRATEGIC] if isinstance(mem, StrategicMemory)]
        if insight_type:
            insights = [mem for mem in insights if mem.insight_type == insight_type]
        return insights
    
    def generate_battlefield_summary(self) -> str:
        """Generate a natural language summary of battlefield knowledge"""
        coord_memories = self.get_coordinate_history()
        if not coord_memories:
            return "No battlefield activity recorded yet."
        
        total_attacks = len(coord_memories)
        hits = len([m for m in coord_memories if m.result in ["HIT", "SUNK"]])
        misses = len([m for m in coord_memories if m.result == "MISS"])
        sunk_ships = [m.sunk_ship for m in coord_memories if m.sunk_ship]
        
        summary = f"Battlefield Knowledge ({total_attacks} total attacks):\n"
        summary += f"- Successful attacks: {hits} hits, {len(sunk_ships)} ships sunk\n"
        summary += f"- Failed attacks: {misses} misses\n"
        
        if sunk_ships:
            summary += f"- Ships destroyed: {', '.join(sunk_ships)}\n"
        
        # Recent activity
        recent_attacks = sorted(coord_memories, key=lambda x: x.timestamp)[-5:]
        if recent_attacks:
            summary += f"- Recent activity: {', '.join([m.to_natural_language() for m in recent_attacks])}"
        
        return summary
    
    def generate_player_profile_summary(self, player_id: str) -> str:
        """Generate summary of what we know about a specific player"""
        patterns = self.get_player_patterns(player_id)
        if not patterns:
            return f"No behavioral patterns recorded for {player_id}."
        
        summary = f"Player Profile - {player_id}:\n"
        for pattern in patterns:
            latest_obs = pattern.observations[-1] if pattern.observations else "No observations"
            summary += f"- {pattern.behavior_type}: {latest_obs} (confidence: {pattern.confidence:.1f})\n"
        
        return summary.strip()
    
    def search_memories(self, query: str, memory_types: List[MemoryType] = None) -> List[Any]:
        """Search memories by text content"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        results = []
        query_lower = query.lower()
        
        for memory_type in memory_types:
            for memory in self.memories[memory_type]:
                # Convert memory to string and search
                memory_str = json.dumps(asdict(memory) if hasattr(memory, '__dict__') else str(memory)).lower()
                if query_lower in memory_str:
                    results.append(memory)
        
        return results
    
    def export_memories(self, memory_types: List[MemoryType] = None) -> Dict[str, Any]:
        """Export memories to a serializable format"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        export_data = {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.now().isoformat(),
            "memories": {}
        }
        
        for memory_type in memory_types:
            memories = []
            for memory in self.memories[memory_type]:
                if hasattr(memory, '__dict__'):
                    memories.append(asdict(memory))
                else:
                    memories.append(str(memory))
            export_data["memories"][memory_type.value] = memories
        
        return export_data
    
    def clear_memories(self, memory_types: List[MemoryType] = None, older_than_hours: int = None):
        """Clear memories, optionally filtered by type and age"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        cutoff_time = None
        if older_than_hours:
            from datetime import timedelta
            cutoff_time = (datetime.now() - timedelta(hours=older_than_hours)).isoformat()
        
        for memory_type in memory_types:
            if cutoff_time:
                # Keep only recent memories
                self.memories[memory_type] = [
                    mem for mem in self.memories[memory_type]
                    if hasattr(mem, 'timestamp') and mem.timestamp >= cutoff_time
                ]
            else:
                # Clear all memories of this type
                self.memories[memory_type] = []
        
        # Rebuild index
        self.memory_index = {}
        for memory_type in MemoryType:
            for i, memory in enumerate(self.memories[memory_type]):
                # Re-index based on memory type
                if isinstance(memory, CoordinateMemory):
                    self._index_memory("coordinate", memory.coordinate, memory_type, i)
                    self._index_memory("attacker", memory.attacker, memory_type, i)
                elif isinstance(memory, PlayerBehaviorMemory):
                    self._index_memory("player", memory.player_id, memory_type, i)
                # Add more indexing as needed
        
        logger.info(f"[MEMORY] {self.agent_id} cleared memories: {[mt.value for mt in memory_types]}")


class GlobalMemoryManager:
    """Manages memory for all agents in the game"""
    
    def __init__(self):
        self.agent_memories: Dict[str, AgentMemoryManager] = {}
        self.global_events: List[Dict[str, Any]] = []
        
    def get_agent_memory(self, agent_id: str) -> AgentMemoryManager:
        """Get or create memory manager for an agent"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemoryManager(agent_id)
        return self.agent_memories[agent_id]
    
    def record_coordinate_attack(self, coordinate: str, attacker: str, target_team: str, 
                               result: str, sunk_ship: Optional[str] = None, round_number: int = 0,
                               inform_agents: List[str] = None):
        """Record a coordinate attack for multiple agents"""
        if inform_agents is None:
            inform_agents = list(self.agent_memories.keys())
        
        for agent_id in inform_agents:
            agent_memory = self.get_agent_memory(agent_id)
            agent_memory.add_coordinate_memory(coordinate, attacker, target_team, result, sunk_ship, round_number)
        
        # Record as global event
        self.global_events.append({
            "type": "coordinate_attack",
            "coordinate": coordinate,
            "attacker": attacker,
            "target_team": target_team,
            "result": result,
            "sunk_ship": sunk_ship,
            "round": round_number,
            "timestamp": datetime.now().isoformat()
        })
    
    def analyze_player_behavior(self, player_id: str, recent_coordinates: List[str]) -> Dict[str, str]:
        """Analyze player behavior and update memories"""
        analysis = {}
        
        if not recent_coordinates:
            return analysis
        
        # Analyze coordinate preferences
        edge_coords = sum(1 for coord in recent_coordinates if coord[0] in 'AJ' or coord[1:] in ['1', '10'])
        center_coords = sum(1 for coord in recent_coordinates if coord[0] in 'DEF' and coord[1:] in ['4', '5', '6'])
        
        if edge_coords > center_coords:
            analysis["coordinate_preference"] = "Prefers edge/corner attacks"
        elif center_coords > edge_coords:
            analysis["coordinate_preference"] = "Prefers center area attacks"
        else:
            analysis["coordinate_preference"] = "Mixed attack pattern"
        
        # Analyze systematic vs random approach
        coord_numbers = []
        for coord in recent_coordinates:
            try:
                row = ord(coord[0]) - ord('A')
                col = int(coord[1:]) - 1
                coord_numbers.append((row, col))
            except (ValueError, IndexError):
                continue
        
        if len(coord_numbers) >= 3:
            # Check for sequential patterns
            sequential = 0
            for i in range(1, len(coord_numbers)):
                prev_row, prev_col = coord_numbers[i-1]
                curr_row, curr_col = coord_numbers[i]
                if abs(prev_row - curr_row) <= 1 and abs(prev_col - curr_col) <= 1:
                    sequential += 1
            
            if sequential >= len(coord_numbers) * 0.6:
                analysis["attack_strategy"] = "Systematic/methodical approach"
            else:
                analysis["attack_strategy"] = "Random/scattered approach"
        
        # Update all agent memories with this analysis
        for agent_id in self.agent_memories:
            agent_memory = self.agent_memories[agent_id]
            for behavior_type, observation in analysis.items():
                agent_memory.add_player_behavior(player_id, behavior_type, observation)
        
        return analysis
    
    def generate_global_battlefield_state(self) -> str:
        """Generate a comprehensive battlefield state summary"""
        if not self.global_events:
            return "No battlefield activity recorded."
        
        attacks_by_result = {}
        attacks_by_round = {}
        sunk_ships = []
        
        for event in self.global_events:
            if event["type"] == "coordinate_attack":
                result = event["result"]
                round_num = event["round"]
                
                attacks_by_result[result] = attacks_by_result.get(result, 0) + 1
                attacks_by_round[round_num] = attacks_by_round.get(round_num, 0) + 1
                
                if event["sunk_ship"]:
                    sunk_ships.append(event["sunk_ship"])
        
        total_attacks = len([e for e in self.global_events if e["type"] == "coordinate_attack"])
        summary = f"Global Battlefield State:\n"
        summary += f"- Total attacks: {total_attacks}\n"
        summary += f"- Results: {dict(attacks_by_result)}\n"
        summary += f"- Ships sunk: {len(sunk_ships)} ({', '.join(sunk_ships) if sunk_ships else 'none'})\n"
        summary += f"- Current round: {max(attacks_by_round.keys()) if attacks_by_round else 0}\n"
        
        return summary
    
    def export_all_memories(self) -> Dict[str, Any]:
        """Export all agent memories and global state"""
        export_data = {
            "global_events": self.global_events,
            "agent_memories": {},
            "export_timestamp": datetime.now().isoformat()
        }
        
        for agent_id, agent_memory in self.agent_memories.items():
            export_data["agent_memories"][agent_id] = agent_memory.export_memories()
        
        return export_data