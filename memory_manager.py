"""
Enhanced Memory Management System for Simultaneous Battleship AI Agents

Extended to support:
- Individual ship ownership and elimination tracking
- Dual action space (BOMB vs MOVE actions)
- Team deliberation session memories (3-phase Alpha protocol)
- Ship movement and position tracking
- Simultaneous execution coordination patterns
- Alpha vs Beta decision pattern analysis
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
    MOVEMENT = "movement"                # Ship movements and positions
    DELIBERATION = "deliberation"        # Team deliberation sessions (3-phase Alpha)
    PLAYER_PROFILE = "player_profile"    # Observed player behaviors
    SHIP_OWNERSHIP = "ship_ownership"    # Ship ownership and elimination tracking
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
    ship_owner: Optional[str] = None    # Player who owned the sunk ship
    player_eliminated: bool = False     # Was the ship owner eliminated
    round_number: int = 0               # When it happened
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_natural_language(self) -> str:
        """Convert to human-readable format"""
        base = f"{self.attacker} attacked {self.coordinate} → {self.result}"
        if self.sunk_ship:
            base += f" (sunk {self.ship_owner}'s {self.sunk_ship})"
            if self.player_eliminated:
                base += f" - {self.ship_owner} ELIMINATED"
        return base


@dataclass
class MovementMemory:
    """Memory of ship movements and repositioning"""
    player_id: str                      # Who moved their ship
    ship_name: str                      # Which ship was moved
    movement_type: str                  # "UP", "DOWN", "LEFT", "RIGHT", "ROTATE"
    from_position: Optional[str] = None # Starting coordinate (if trackable)
    to_position: Optional[str] = None   # Ending coordinate (if trackable)
    reason: Optional[str] = None        # Strategic reason if known
    round_number: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_natural_language(self) -> str:
        """Convert to human-readable format"""
        base = f"{self.player_id} moved {self.ship_name} {self.movement_type}"
        if self.from_position and self.to_position:
            base += f" from {self.from_position} to {self.to_position}"
        if self.reason:
            base += f" ({self.reason})"
        return base


@dataclass
class DeliberationMemory:
    """Memory of team deliberation sessions (3-phase Alpha protocol)"""
    session_id: str                     # Unique session identifier
    team_id: str                        # Which team deliberated
    round_number: int                   # Game round
    phase: str                          # "proposals", "voting", "selection", "complete"
    participants: List[str] = field(default_factory=list)
    proposals: Dict[str, str] = field(default_factory=dict)      # player_id -> proposal
    votes: Dict[str, str] = field(default_factory=dict)         # voter_id -> voted_for
    final_actions: Dict[str, str] = field(default_factory=dict) # player_id -> final action
    winning_proposer: Optional[str] = None                      # Who's plan was selected
    consensus_reached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_natural_language(self) -> str:
        """Convert to human-readable format"""
        base = f"{self.team_id} deliberation R{self.round_number}: {len(self.participants)} participants"
        if self.winning_proposer:
            base += f", {self.winning_proposer}'s plan selected"
        return base


@dataclass
class ShipOwnershipMemory:
    """Memory of ship ownership and player elimination"""
    player_id: str                      # Player who owns/owned the ship
    ship_name: str                      # Name of the ship
    ship_size: int                      # Size of the ship
    status: str                         # "ACTIVE", "DAMAGED", "SUNK"
    elimination_round: Optional[int] = None  # Round when player was eliminated
    eliminated_by: Optional[str] = None     # Who eliminated this player
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_natural_language(self) -> str:
        """Convert to human-readable format"""
        if self.status == "SUNK":
            return f"{self.player_id}'s {self.ship_name} SUNK by {self.eliminated_by} in round {self.elimination_round}"
        return f"{self.player_id} owns {self.ship_name} ({self.status})"


@dataclass
class PlayerBehaviorMemory:
    """Memory of observed player behavior patterns"""
    player_id: str
    behavior_type: str                  # "coordinate_preference", "action_preference", etc.
    observations: List[str] = field(default_factory=list)
    confidence: float = 0.5             # How confident we are in this pattern
    team_coordination: Optional[bool] = None  # Does this player coordinate with team
    decision_style: Optional[str] = None      # "deliberative", "individual", "mixed"
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
    conversation_type: str = "general"  # "proposal", "voting", "advice", "general"
    key_points: List[str] = field(default_factory=list)
    coordinates_mentioned: List[str] = field(default_factory=list)
    actions_mentioned: List[str] = field(default_factory=list)  # BOMB/MOVE actions
    advice_given: Optional[str] = None
    advice_quality: str = "unknown"     # "good", "bad", "neutral"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StrategicMemory:
    """Memory of strategic insights and patterns"""
    insight_type: str                   # "enemy_pattern", "team_strategy", "movement_pattern", etc.
    description: str
    supporting_evidence: List[str] = field(default_factory=list)
    reliability: float = 0.5           # How reliable this insight is
    team_context: Optional[str] = None  # "alpha", "beta", "both"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentMemoryManager:
    """Enhanced memory management for simultaneous battleship agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memories: Dict[MemoryType, List[Any]] = {
            MemoryType.BATTLEFIELD: [],
            MemoryType.MOVEMENT: [],
            MemoryType.DELIBERATION: [],
            MemoryType.PLAYER_PROFILE: [],
            MemoryType.SHIP_OWNERSHIP: [],
            MemoryType.CONVERSATION: [],
            MemoryType.STRATEGIC: [],
            MemoryType.GAME_STATE: []
        }
        self.memory_index: Dict[str, List[Tuple[MemoryType, int]]] = {}  # For fast lookup
        
    def add_coordinate_memory(self, coordinate: str, attacker: str, target_team: str, 
                            result: str, sunk_ship: Optional[str] = None, 
                            ship_owner: Optional[str] = None, player_eliminated: bool = False,
                            round_number: int = 0):
        """Add memory of a coordinate attack with ownership info"""
        memory = CoordinateMemory(
            coordinate=coordinate,
            attacker=attacker,
            target_team=target_team,
            result=result,
            sunk_ship=sunk_ship,
            ship_owner=ship_owner,
            player_eliminated=player_eliminated,
            round_number=round_number
        )
        
        self.memories[MemoryType.BATTLEFIELD].append(memory)
        self._index_memory("coordinate", coordinate, MemoryType.BATTLEFIELD, len(self.memories[MemoryType.BATTLEFIELD]) - 1)
        self._index_memory("attacker", attacker, MemoryType.BATTLEFIELD, len(self.memories[MemoryType.BATTLEFIELD]) - 1)
    
    def add_movement_memory(self, player_id: str, ship_name: str, movement_type: str,
                          from_position: str = None, to_position: str = None,
                          reason: str = None, round_number: int = 0):
        """Add memory of ship movement"""
        memory = MovementMemory(
            player_id=player_id,
            ship_name=ship_name,
            movement_type=movement_type,
            from_position=from_position,
            to_position=to_position,
            reason=reason,
            round_number=round_number
        )
        
        self.memories[MemoryType.MOVEMENT].append(memory)
        self._index_memory("mover", player_id, MemoryType.MOVEMENT, len(self.memories[MemoryType.MOVEMENT]) - 1)
    
    def add_deliberation_memory(self, session_id: str, team_id: str, round_number: int,
                              phase: str, participants: List[str] = None,
                              proposals: Dict[str, str] = None, votes: Dict[str, str] = None,
                              final_actions: Dict[str, str] = None, winning_proposer: str = None,
                              consensus_reached: bool = False):
        """Add memory of team deliberation session (3-phase Alpha protocol)"""
        memory = DeliberationMemory(
            session_id=session_id,
            team_id=team_id,
            round_number=round_number,
            phase=phase,
            participants=participants or [],
            proposals=proposals or {},
            votes=votes or {},
            final_actions=final_actions or {},
            winning_proposer=winning_proposer,
            consensus_reached=consensus_reached
        )
        
        self.memories[MemoryType.DELIBERATION].append(memory)
        self._index_memory("deliberation", team_id, MemoryType.DELIBERATION, len(self.memories[MemoryType.DELIBERATION]) - 1)
    
    def add_ship_ownership_memory(self, player_id: str, ship_name: str, ship_size: int,
                                status: str = "ACTIVE", eliminated_by: str = None, 
                                elimination_round: int = None):
        """Add or update ship ownership memory"""
        # Check if we already have ownership memory for this ship
        for i, memory in enumerate(self.memories[MemoryType.SHIP_OWNERSHIP]):
            if memory.player_id == player_id and memory.ship_name == ship_name:
                if ship_size and getattr(memory, "ship_size", 0) == 0:
                    memory.ship_size = ship_size
                # Update existing memory
                memory.status = status
                memory.eliminated_by = eliminated_by
                memory.elimination_round = elimination_round
                memory.timestamp = datetime.now().isoformat()
                return
        
        # Create new ownership memory
        memory = ShipOwnershipMemory(
            player_id=player_id,
            ship_name=ship_name,
            ship_size=ship_size,
            status=status,
            eliminated_by=eliminated_by,
            elimination_round=elimination_round
        )
        
        self.memories[MemoryType.SHIP_OWNERSHIP].append(memory)
        self._index_memory("ship_owner", player_id, MemoryType.SHIP_OWNERSHIP, len(self.memories[MemoryType.SHIP_OWNERSHIP]) - 1)
    
    def add_player_behavior(self, player_id: str, behavior_type: str, observation: str,
                          team_coordination: bool = None, decision_style: str = None):
        """Add or update player behavior observation with team context"""
        # Look for existing behavior memory
        for memory in self.memories[MemoryType.PLAYER_PROFILE]:
            if memory.player_id == player_id and memory.behavior_type == behavior_type:
                memory.add_observation(observation)
                if team_coordination is not None:
                    memory.team_coordination = team_coordination
                if decision_style is not None:
                    memory.decision_style = decision_style
                return
        
        # Create new behavior memory
        memory = PlayerBehaviorMemory(
            player_id=player_id,
            behavior_type=behavior_type,
            observations=[observation],
            team_coordination=team_coordination,
            decision_style=decision_style
        )
        self.memories[MemoryType.PLAYER_PROFILE].append(memory)
        self._index_memory("player", player_id, MemoryType.PLAYER_PROFILE, len(self.memories[MemoryType.PLAYER_PROFILE]) - 1)
    
    def add_conversation_memory(self, conversation_id: str, participants: List[str], 
                              conversation_type: str = "general", key_points: List[str] = None,
                              coordinates_mentioned: List[str] = None, actions_mentioned: List[str] = None,
                              advice_given: str = None):
        """Add memory of an important conversation with action tracking"""
        memory = ConversationMemory(
            conversation_id=conversation_id,
            participants=participants,
            conversation_type=conversation_type,
            key_points=key_points or [],
            coordinates_mentioned=coordinates_mentioned or [],
            actions_mentioned=actions_mentioned or [],
            advice_given=advice_given
        )
        
        self.memories[MemoryType.CONVERSATION].append(memory)
        for participant in participants:
            self._index_memory("participant", participant, MemoryType.CONVERSATION, len(self.memories[MemoryType.CONVERSATION]) - 1)
    
    def add_strategic_memory(self, insight_type: str, description: str, evidence: List[str] = None,
                           team_context: str = None):
        """Add strategic insight with team context"""
        memory = StrategicMemory(
            insight_type=insight_type,
            description=description,
            supporting_evidence=evidence or [],
            team_context=team_context
        )
        
        self.memories[MemoryType.STRATEGIC].append(memory)
        self._index_memory("insight", insight_type, MemoryType.STRATEGIC, len(self.memories[MemoryType.STRATEGIC]) - 1)
    
    def _index_memory(self, key_type: str, key_value: str, memory_type: MemoryType, index: int):
        """Index memory for fast retrieval"""
        index_key = f"{key_type}:{key_value}"
        if index_key not in self.memory_index:
            self.memory_index[index_key] = []
        self.memory_index[index_key].append((memory_type, index))
    
    # Enhanced query methods
    
    def get_movement_history(self, player_id: str = None) -> List[MovementMemory]:
        """Get movement history, optionally filtered by player"""
        movements = [mem for mem in self.memories[MemoryType.MOVEMENT] if isinstance(mem, MovementMemory)]
        if player_id:
            movements = [mem for mem in movements if mem.player_id == player_id]
        return movements
    
    def get_deliberation_history(self, team_id: str = None) -> List[DeliberationMemory]:
        """Get deliberation history, optionally filtered by team"""
        deliberations = [mem for mem in self.memories[MemoryType.DELIBERATION] if isinstance(mem, DeliberationMemory)]
        if team_id:
            deliberations = [mem for mem in deliberations if mem.team_id == team_id]
        return deliberations
    
    def get_ship_ownership_status(self, player_id: str = None) -> List[ShipOwnershipMemory]:
        """Get ship ownership status, optionally filtered by player"""
        ownerships = [mem for mem in self.memories[MemoryType.SHIP_OWNERSHIP] if isinstance(mem, ShipOwnershipMemory)]
        if player_id:
            ownerships = [mem for mem in ownerships if mem.player_id == player_id]
        return ownerships
    
    def get_eliminated_players(self) -> List[str]:
        """Get list of eliminated players"""
        eliminated = {m.player_id for m in self.memories[MemoryType.SHIP_OWNERSHIP]
                    if isinstance(m, ShipOwnershipMemory) and m.status == "SUNK"}
        return sorted(eliminated)
    
    def get_player_action_preferences(self, player_id: str) -> Dict[str, int]:
        """Analyze player's action preferences (BOMB vs MOVE)"""
        bomb_count = 0
        move_count = 0
        
        # Count from battlefield memories
        for memory in self.memories[MemoryType.BATTLEFIELD]:
            if isinstance(memory, CoordinateMemory) and memory.attacker == player_id:
                bomb_count += 1
        
        # Count from movement memories
        for memory in self.memories[MemoryType.MOVEMENT]:
            if isinstance(memory, MovementMemory) and memory.player_id == player_id:
                move_count += 1
        
        return {"BOMB": bomb_count, "MOVE": move_count}
    
    def analyze_team_coordination_patterns(self, team_id: str) -> Dict[str, Any]:
        """Analyze how well a team coordinates (3-phase Alpha protocol)"""
        deliberations = self.get_deliberation_history(team_id)
        
        if not deliberations:
            return {"coordination_level": "none", "analysis": "No deliberation sessions recorded"}
        
        total_sessions = len(deliberations)
        consensus_sessions = len([d for d in deliberations if d.consensus_reached])
        
        # Analyze proposal diversity and voting patterns
        proposal_diversity = 0
        voting_consensus = 0
        
        for delib in deliberations:
            if len(delib.proposals) > 1:
                proposal_diversity += 1
            if len(set(delib.votes.values())) == 1:  # All votes for same proposer
                voting_consensus += 1
        
        coordination_score = (consensus_sessions + voting_consensus) / (total_sessions * 2) if total_sessions > 0 else 0
        
        analysis = {
            "coordination_level": "high" if coordination_score > 0.7 else "medium" if coordination_score > 0.3 else "low",
            "total_deliberations": total_sessions,
            "consensus_rate": consensus_sessions / total_sessions if total_sessions > 0 else 0,
            "proposal_diversity_rate": proposal_diversity / total_sessions if total_sessions > 0 else 0,
            "voting_consensus_rate": voting_consensus / total_sessions if total_sessions > 0 else 0,
            "coordination_score": coordination_score,
            "analysis": f"Team shows {'strong' if coordination_score > 0.7 else 'moderate' if coordination_score > 0.3 else 'weak'} coordination via 3-phase protocol"
        }
        
        return analysis
    
    def generate_dynamic_battlefield_summary(self) -> str:
        """Generate comprehensive battlefield summary including movements"""
        coord_memories = [mem for mem in self.memories[MemoryType.BATTLEFIELD] if isinstance(mem, CoordinateMemory)]
        move_memories = [mem for mem in self.memories[MemoryType.MOVEMENT] if isinstance(mem, MovementMemory)]
        elimination_memories = [mem for mem in self.memories[MemoryType.SHIP_OWNERSHIP] 
                               if isinstance(mem, ShipOwnershipMemory) and mem.status == "SUNK"]
        
        if not coord_memories and not move_memories:
            return "No battlefield activity recorded yet."
        
        total_attacks = len(coord_memories)
        total_moves = len(move_memories)
        hits = len([m for m in coord_memories if m.result in ["HIT", "SUNK"]])
        misses = len([m for m in coord_memories if m.result == "MISS"])
        eliminations = len(elimination_memories)
        
        summary = f"Simultaneous Battlefield Knowledge:\n"
        summary += f"- Actions: {total_attacks} attacks, {total_moves} movements\n"
        summary += f"- Attack results: {hits} hits, {misses} misses\n"
        summary += f"- Player eliminations: {eliminations}\n"
        
        if elimination_memories:
            eliminated_players = [mem.player_id for mem in elimination_memories]
            summary += f"- Eliminated players: {', '.join(eliminated_players)}\n"
        
        # Recent activity (sorted by timestamp)
        recent_actions = sorted(coord_memories + move_memories, key=lambda x: x.timestamp)[-5:]
        if recent_actions:
            summary += f"- Recent activity:\n"
            for action in recent_actions:
                summary += f"  • {action.to_natural_language()}\n"
        
        return summary.strip()
    
    def export_memories(self, memory_types: List[MemoryType] = None) -> Dict[str, Any]:
        """Export memories with enhanced simultaneous battleship data"""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        export_data = {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.now().isoformat(),
            "memories": {},
            "analytics": {
                "total_eliminations": len(self.get_eliminated_players()),
                "action_preferences": {},
                "coordination_analysis": {}
            }
        }
        
        # Export standard memories
        for memory_type in memory_types:
            memories = []
            for memory in self.memories[memory_type]:
                if hasattr(memory, '__dict__'):
                    memories.append(asdict(memory))
                else:
                    memories.append(str(memory))
            export_data["memories"][memory_type.value] = memories
        
        # Add analytics
        all_players = set()
        for memory in self.memories[MemoryType.BATTLEFIELD] + self.memories[MemoryType.MOVEMENT]:
            if hasattr(memory, 'attacker'):
                all_players.add(memory.attacker)
            if hasattr(memory, 'player_id'):
                all_players.add(memory.player_id)
        
        for player in all_players:
            export_data["analytics"]["action_preferences"][player] = self.get_player_action_preferences(player)
        
        return export_data


class GlobalMemoryManager:
    """Enhanced global memory manager for simultaneous battleship"""
    
    def __init__(self):
        self.agent_memories: Dict[str, AgentMemoryManager] = {}
        self.global_events: List[Dict[str, Any]] = []
        
    def get_agent_memory(self, agent_id: str) -> AgentMemoryManager:
        """Get or create memory manager for an agent"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemoryManager(agent_id)
        return self.agent_memories[agent_id]
    
    def record_coordinate_attack(self, coordinate: str, attacker: str, target_team: str,
                               result: str, sunk_ship: Optional[str] = None,
                               ship_owner: Optional[str] = None, player_eliminated: bool = False,
                               round_number: int = 0, inform_agents: List[str] = None) -> None:
        """Record a coordinate attack with ownership/elimination info and emit a global event"""
        # Determine who should receive this memory
        if inform_agents is None:
            inform_agents = list(self.agent_memories.keys())
        if not inform_agents:
            # Fallback so the action is never dropped
            inform_agents = [attacker]
        
        elim_flag = bool(player_eliminated) or (str(result).upper() == "SUNK" and ship_owner is not None)
        
        # Store the attack memory for all informed agents
        for agent_id in inform_agents:
            agent_memory = self.get_agent_memory(agent_id)
            agent_memory.add_coordinate_memory(
                coordinate=coordinate,
                attacker=attacker,
                target_team=target_team,
                result=result,
                sunk_ship=sunk_ship,
                ship_owner=ship_owner,
                player_eliminated=elim_flag,
                round_number=round_number,
            )

        # Update ownership on elimination
        if elim_flag and ship_owner:
            recovered_size = 0
            if sunk_ship:
                for aid, am in self.agent_memories.items():
                    for mem in am.get_ship_ownership_status(player_id=ship_owner):
                        if mem.ship_name == sunk_ship and mem.ship_size:
                            recovered_size = mem.ship_size
                            break
                    if recovered_size:
                        break
            for agent_id in inform_agents:
                self.get_agent_memory(agent_id).add_ship_ownership_memory(
                    player_id=ship_owner,
                    ship_name=sunk_ship or "Unknown Ship",
                    ship_size=recovered_size,
                    status="SUNK",
                    eliminated_by=attacker,
                    elimination_round=round_number,
                )

        # Record global event
        self.global_events.append({
            "type": "coordinate_attack",
            "coordinate": coordinate,
            "attacker": attacker,
            "target_team": target_team,
            "result": result,
            "sunk_ship": sunk_ship,
            "ship_owner": ship_owner,
            "player_eliminated": elim_flag,
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
        })
    
    def record_ship_movement(self, player_id: str, ship_name: str, movement_type: str,
                           from_position: str = None, to_position: str = None,
                           reason: str = None, round_number: int = 0,
                           inform_agents: List[str] = None):
        """Record ship movement for all relevant agents"""
        if inform_agents is None:
            inform_agents = list(self.agent_memories.keys())
        
        if not inform_agents:
            inform_agents = [player_id]
        
        for agent_id in inform_agents:
            agent_memory = self.get_agent_memory(agent_id)
            agent_memory.add_movement_memory(player_id, ship_name, movement_type,
                                           from_position, to_position, reason, round_number)
        
        # Record as global event
        self.global_events.append({
            "type": "ship_movement",
            "player_id": player_id,
            "ship_name": ship_name,
            "movement_type": movement_type,
            "from_position": from_position,
            "to_position": to_position,
            "reason": reason,
            "round": round_number,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_team_deliberation(self, session_id: str, team_id: str, round_number: int,
                               participants: List[str], proposals: Dict[str, str] = None,
                               votes: Dict[str, str] = None, final_actions: Dict[str, str] = None,
                               winning_proposer: str = None, consensus_reached: bool = False,
                               inform_agents: List[str] = None):
        """Record team deliberation session (3-phase Alpha protocol)"""
        if inform_agents is None:
            inform_agents = list(self.agent_memories.keys())
        
        if not inform_agents:
            inform_agents = participants or []
        
        for agent_id in inform_agents:
            agent_memory = self.get_agent_memory(agent_id)
            agent_memory.add_deliberation_memory(session_id, team_id, round_number, "complete",
                                                participants, proposals, votes, final_actions,
                                                winning_proposer, consensus_reached)
        
        # Record as global event
        self.global_events.append({
            "type": "team_deliberation",
            "session_id": session_id,
            "team_id": team_id,
            "round": round_number,
            "participants": participants,
            "winning_proposer": winning_proposer,
            "consensus_reached": consensus_reached,
            "timestamp": datetime.now().isoformat()
        })
    
    def initialize_ship_ownership(self, ship_assignments: Dict[str, Dict[str, Any]], inform_agents: List[str] = None):
        """Initialize ship ownership for all players"""
        if inform_agents is None:
            inform_agents = list(self.agent_memories.keys())
        if not inform_agents:
            # Ensure write-through even at start of game
            inform_agents = list(ship_assignments.keys())

        for player_id, ship_info in ship_assignments.items():
            for agent_id in inform_agents:
                agent_memory = self.get_agent_memory(agent_id)
                agent_memory.add_ship_ownership_memory(
                    player_id, ship_info['name'], ship_info['size'], "ACTIVE"
                )

    def get_player_context(self, player_id: str) -> str:
        """Get battlefield context for a specific player"""
        try:
            return self.generate_dynamic_battlefield_summary()
        except Exception:
            return "Battlefield memory unavailable"

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary for simultaneous battleship game"""
        analysis = self.analyze_experiment_results()
        return {
            "coordination_effectiveness": analysis.get("coordination_comparison", {}),
            "action_balance": analysis.get("action_patterns", {}),
            "elimination_summary": analysis.get("elimination_patterns", {}),
            "total_events": len(self.global_events)
        }
    
    def analyze_experiment_results(self) -> Dict[str, Any]:
        """Analyze the 3-phase deliberation vs individual decision experiment"""
        analysis = {
            "coordination_comparison": {},
            "action_patterns": {},
            "elimination_patterns": {},
            "decision_effectiveness": {}
        }
        
        # Analyze coordination by team (3-phase Alpha vs Individual Beta)
        teams = set()
        for event in self.global_events:
            if event["type"] == "team_deliberation":
                teams.add(event["team_id"])
        
        for team_id in teams:
            if self.agent_memories:
                sample_agent = next(iter(self.agent_memories.values()))
                team_analysis = sample_agent.analyze_team_coordination_patterns(team_id)
                analysis["coordination_comparison"][team_id] = team_analysis
        
        # Analyze action patterns (BOMB vs MOVE in simultaneous execution)
        bomb_actions = len([e for e in self.global_events if e["type"] == "coordinate_attack"])
        move_actions = len([e for e in self.global_events if e["type"] == "ship_movement"])
        
        analysis["action_patterns"] = {
            "total_bomb_actions": bomb_actions,
            "total_move_actions": move_actions,
            "mobility_ratio": move_actions / (bomb_actions + move_actions) if (bomb_actions + move_actions) > 0 else 0,
            "strategic_balance": "high_mobility" if move_actions > bomb_actions * 0.3 else "low_mobility"
        }
        
        # Analyze elimination patterns
        elimination_events = [e for e in self.global_events if e["type"] == "coordinate_attack" and e.get("player_eliminated")]
        analysis["elimination_patterns"] = {
            "total_eliminations": len(elimination_events),
            "elimination_timeline": [{"round": e["round"], "eliminated": e["ship_owner"], "by": e["attacker"]} for e in elimination_events],
            "average_elimination_round": sum(e["round"] for e in elimination_events) / len(elimination_events) if elimination_events else 0
        }
        
        return analysis
    
    def generate_dynamic_battlefield_summary(self) -> str:
        """Generate comprehensive battlefield summary"""
        if not self.agent_memories:
            return "No battlefield activity recorded yet."
        
        # Use first agent's memory as representative view
        sample_agent = next(iter(self.agent_memories.values()))
        return sample_agent.generate_dynamic_battlefield_summary()
    
    def generate_experiment_report(self) -> str:
        """Generate comprehensive experiment report for simultaneous battleship"""
        analysis = self.analyze_experiment_results()
        
        report = "🧪 SIMULTANEOUS BATTLESHIP EXPERIMENT ANALYSIS\n"
        report += "=" * 50 + "\n\n"
        
        # Coordination Analysis (3-phase Alpha vs Individual Beta)
        report += "🤝 TEAM COORDINATION ANALYSIS:\n"
        for team_id, coord_data in analysis["coordination_comparison"].items():
            report += f"  {team_id.upper()}:\n"
            report += f"    Coordination Level: {coord_data['coordination_level']}\n"
            report += f"    Consensus Rate: {coord_data['consensus_rate']:.1%}\n"
            if 'voting_consensus_rate' in coord_data:
                report += f"    Voting Consensus: {coord_data['voting_consensus_rate']:.1%}\n"
            report += f"    Analysis: {coord_data['analysis']}\n\n"
        
        # Action Patterns (Simultaneous Execution)
        action_data = analysis["action_patterns"]
        report += "⚔️ SIMULTANEOUS ACTION UTILIZATION:\n"
        report += f"  Bomb Actions: {action_data['total_bomb_actions']}\n"
        report += f"  Move Actions: {action_data['total_move_actions']}\n"
        report += f"  Mobility Ratio: {action_data['mobility_ratio']:.1%}\n"
        report += f"  Strategic Balance: {action_data['strategic_balance']}\n\n"
        
        # Elimination Analysis
        elim_data = analysis["elimination_patterns"]
        report += "💀 ELIMINATION ANALYSIS:\n"
        report += f"  Total Eliminations: {elim_data['total_eliminations']}\n"
        report += f"  Average Elimination Round: {elim_data['average_elimination_round']:.1f}\n"
        if elim_data["elimination_timeline"]:
            report += "  Elimination Timeline:\n"
            for elim in elim_data["elimination_timeline"]:
                report += f"    Round {elim['round']}: {elim['eliminated']} eliminated by {elim['by']}\n"
        
        report += "\n🎯 EXPERIMENTAL INSIGHTS:\n"
        report += "  • Simultaneous execution eliminates turn order bias\n"
        report += "  • 3-phase Alpha protocol enables structured coordination\n"
        report += "  • Individual Beta decisions provide coordination-free baseline\n"
        
        return report
    
    def export_all_memories(self) -> Dict[str, Any]:
        """Export all memories with enhanced simultaneous battleship analytics"""
        export_data = {
            "global_events": self.global_events,
            "agent_memories": {},
            "experiment_analysis": self.analyze_experiment_results(),
            "experiment_report": self.generate_experiment_report(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        for agent_id, agent_memory in self.agent_memories.items():
            export_data["agent_memories"][agent_id] = agent_memory.export_memories()
        
        return export_data


# =============================================================================
# Convenience Functions for Simultaneous Battleship Integration
# =============================================================================

def create_simultaneous_memory_manager() -> GlobalMemoryManager:
    """Create a memory manager optimized for simultaneous battleship"""
    return GlobalMemoryManager()


def record_ship_assignment(memory_manager: GlobalMemoryManager, 
                         player_assignments: Dict[str, Dict[str, Any]]):
    """Record initial ship assignments for all players"""
    memory_manager.initialize_ship_ownership(player_assignments)
    logger.info(f"[MEMORY] Initialized ship ownership for {len(player_assignments)} players")


def record_player_action(memory_manager: GlobalMemoryManager, player_id: str, 
                        action_type: str, target: str, result: str = None,
                        round_number: int = 0, **kwargs):
    """Record a player action (BOMB or MOVE) with appropriate memory type"""
    if action_type == "BOMB":
        memory_manager.record_coordinate_attack(
            coordinate=target,
            attacker=player_id,
            target_team=kwargs.get('target_team', 'enemy'),
            result=result or 'UNKNOWN',
            sunk_ship=kwargs.get('sunk_ship'),
            ship_owner=kwargs.get('ship_owner'),
            player_eliminated=kwargs.get('player_eliminated', False),
            round_number=round_number
        )
    elif action_type == "MOVE":
        memory_manager.record_ship_movement(
            player_id=player_id,
            ship_name=kwargs.get('ship_name', 'Unknown Ship'),
            movement_type=target,
            from_position=kwargs.get('from_position'),
            to_position=kwargs.get('to_position'),
            reason=kwargs.get('reason'),
            round_number=round_number
        )


def record_deliberation_session(memory_manager: GlobalMemoryManager, team_id: str,
                              round_number: int, participants: List[str],
                              proposals: Dict[str, str] = None, votes: Dict[str, str] = None,
                              final_actions: Dict[str, str] = None, winning_proposer: str = None,
                              consensus_reached: bool = False):
    """Record a team deliberation session (3-phase Alpha protocol)"""
    session_id = f"{team_id}_r{round_number}_{datetime.now().strftime('%H%M%S')}"
    memory_manager.record_team_deliberation(
        session_id=session_id,
        team_id=team_id,
        round_number=round_number,
        participants=participants,
        proposals=proposals,
        votes=votes,
        final_actions=final_actions,
        winning_proposer=winning_proposer,
        consensus_reached=consensus_reached
    )


if __name__ == "__main__":
    # Example usage and testing
    print("🧠 Enhanced Memory Manager for Simultaneous Battleship")
    print("Key Features:")
    print("✅ Individual ship ownership tracking")
    print("✅ Dual action space (BOMB/MOVE) memory")
    print("✅ 3-phase team deliberation recording")
    print("✅ Player elimination tracking")
    print("✅ Movement and positioning history")
    print("✅ Simultaneous execution coordination analysis")
    print("✅ Alpha vs Beta decision pattern comparison")
    print("✅ Clean integration with simultaneous battleship game")