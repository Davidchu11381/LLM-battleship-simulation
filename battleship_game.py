"""
battleship_game.py — Dynamic Battleship Game Engine

Implementation of the 3-phase Alpha consensus protocol vs Beta individual decisions
with SIMULTANEOUS EXECUTION to eliminate turn order bias:

Alpha Team (Coordinated):
1. Team Strategy Proposals (Parallel Broadcast)
2. Democratic Voting (Sequential)  
3. Plan Selection (Majority rule with tie-breaking)

Beta Team: Individual decisions only

Both teams execute actions simultaneously after decision phases complete.
"""

from __future__ import annotations

import json
import asyncio
import random
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_manager import GlobalMemoryManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# Core Types
# =============================================================================

class CellState(Enum):
    EMPTY = "~"
    SHIP = "S"
    HIT = "X"
    MISS = "O"


class GamePhase(Enum):
    SETUP = "setup"
    SHIP_PLACEMENT = "ship_placement"
    BATTLE = "battle"
    GAME_OVER = "game_over"


class ActionType(Enum):
    BOMB = "BOMB"
    MOVE = "MOVE"


@dataclass
class Ship:
    name: str
    size: int
    owner_id: str
    positions: List[Tuple[int, int]] = field(default_factory=list)
    hits: int = 0
    orientation: str = "horizontal"

    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.size


@dataclass
class PlayerAction:
    player_id: str
    action_type: ActionType
    target: str  # coordinate for BOMB, direction for MOVE
    reason: Optional[str] = None


@dataclass
class TeamProposal:
    proposer_id: str
    team_actions: Dict[str, PlayerAction]  # player_id -> action
    reasoning: str


@dataclass
class PlayerScore:
    player_id: str
    team_victory: int = 0          # +100 points
    ship_survival: int = 0         # +20 points
    coordination_bonus: int = 0    # +5 points for effective proposals/votes
    waste_penalty: int = 0         # -10 points for duplicate bombing
    
    @property
    def total_score(self) -> int:
        return self.team_victory + self.ship_survival + self.coordination_bonus + self.waste_penalty


@dataclass
class GameGrid:
    size: Tuple[int, int] = (10, 10)
    grid: List[List[CellState]] = field(default_factory=list)
    ships: List[Ship] = field(default_factory=list)

    def __post_init__(self):
        if not self.grid:
            self.grid = [[CellState.EMPTY for _ in range(self.size[1])]
                         for _ in range(self.size[0])]

    def place_ship(self, ship: Ship, start_pos: Tuple[int, int], orientation: str) -> bool:
        """Place ship on grid"""
        r, c = start_pos
        positions: List[Tuple[int, int]] = []
        
        for i in range(ship.size):
            rr, cc = (r, c + i) if orientation == "horizontal" else (r + i, c)
            if not (0 <= rr < self.size[0] and 0 <= cc < self.size[1]):
                return False
            if self.grid[rr][cc] != CellState.EMPTY:
                return False
            positions.append((rr, cc))
        
        # Commit placement
        ship.positions = positions
        ship.orientation = orientation
        for rr, cc in positions:
            self.grid[rr][cc] = CellState.SHIP
        self.ships.append(ship)
        return True

    def get_valid_moves_for_ship(self, ship: Ship) -> List[str]:
        """Get valid movement directions for a ship"""
        if not ship.positions:
            return []
        
        directions = ["UP", "DOWN", "LEFT", "RIGHT", "ROTATE"]
        valid_moves = []
        
        for direction in directions:
            new_positions = self._calculate_new_positions(ship, direction)
            if new_positions and self._can_occupy_positions(ship, new_positions):
                valid_moves.append(direction)
        
        return valid_moves

    def move_ship(self, ship: Ship, direction: str) -> bool:
        """Move ship in specified direction"""
        new_positions = self._calculate_new_positions(ship, direction)
        if not new_positions or not self._can_occupy_positions(ship, new_positions):
            return False

        # Remember hit indices
        hit_indices = [i for i, (r, c) in enumerate(ship.positions) 
                      if self.grid[r][c] == CellState.HIT]

        # Clear old positions
        for r, c in ship.positions:
            self.grid[r][c] = CellState.EMPTY

        # Set new positions
        ship.positions = new_positions
        for r, c in new_positions:
            self.grid[r][c] = CellState.SHIP

        # Restore hits
        for i in hit_indices:
            if i < len(new_positions):
                r, c = new_positions[i]
                self.grid[r][c] = CellState.HIT

        # Update orientation for rotation
        if direction.upper() == "ROTATE":
            ship.orientation = "vertical" if ship.orientation == "horizontal" else "horizontal"

        return True

    def _calculate_new_positions(self, ship: Ship, direction: str) -> Optional[List[Tuple[int, int]]]:
        """Calculate new positions for ship movement"""
        direction = direction.upper()
        
        if direction == "ROTATE":
            return self._calculate_rotation_positions(ship)
        
        deltas = {
            "UP": (-1, 0),
            "DOWN": (1, 0), 
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        
        if direction not in deltas:
            return None
        
        dr, dc = deltas[direction]
        return [(r + dr, c + dc) for r, c in ship.positions]

    def _calculate_rotation_positions(self, ship: Ship) -> List[Tuple[int, int]]:
        """Calculate positions after 90-degree rotation"""
        if not ship.positions:
            return []
        
        # Use first position as anchor
        anchor_r, anchor_c = ship.positions[0]
        
        if ship.orientation == "horizontal":
            # Rotate to vertical: spread downward from anchor
            return [(anchor_r + i, anchor_c) for i in range(ship.size)]
        else:
            # Rotate to horizontal: spread rightward from anchor  
            return [(anchor_r, anchor_c + i) for i in range(ship.size)]

    def _can_occupy_positions(self, moving_ship: Ship, new_positions: List[Tuple[int, int]]) -> bool:
        """Check if ship can occupy new positions"""
        for r, c in new_positions:
            # Check bounds
            if not (0 <= r < self.size[0] and 0 <= c < self.size[1]):
                return False
            
            cell = self.grid[r][c]
            
            # Can't move into occupied space unless it's our own current position
            if cell in (CellState.SHIP, CellState.HIT):
                if (r, c) not in moving_ship.positions:
                    return False
        
        return True

    def attack(self, target: Tuple[int, int]) -> Tuple[str, Optional[Ship]]:
        """Attack a coordinate, return result and sunk ship if any"""
        r, c = target
        
        # Check bounds
        if not (0 <= r < self.size[0] and 0 <= c < self.size[1]):
            return "INVALID", None
        
        cell = self.grid[r][c]
        
        # Already attacked
        if cell in (CellState.HIT, CellState.MISS):
            return "ALREADY_ATTACKED", None
        
        # Hit ship
        if cell == CellState.SHIP:
            self.grid[r][c] = CellState.HIT
            
            # Find which ship was hit
            hit_ship = None
            for ship in self.ships:
                if (r, c) in ship.positions:
                    ship.hits += 1
                    hit_ship = ship
                    break
            
            if hit_ship and hit_ship.is_sunk:
                return "SUNK", hit_ship
            return "HIT", None
        
        # Miss
        self.grid[r][c] = CellState.MISS
        return "MISS", None

    def get_ship_by_owner(self, owner_id: str) -> Optional[Ship]:
        """Get ship owned by specified player"""
        for ship in self.ships:
            if ship.owner_id == owner_id:
                return ship
        return None


@dataclass
class Team:
    name: str
    members: List[str]
    grid: GameGrid
    color: str = "blue"
    coordination_enabled: bool = False


@dataclass
class GameState:
    phase: GamePhase = GamePhase.SETUP
    current_team: Optional[str] = None
    round_number: int = 1
    teams: Dict[str, Team] = field(default_factory=dict)
    game_log: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    eliminated_players: List[str] = field(default_factory=list)
    player_scores: Dict[str, PlayerScore] = field(default_factory=dict)


# =============================================================================
# Game Controller
# =============================================================================

class BattleshipGame:
    """Dynamic Battleship game with simultaneous execution - Alpha coordination vs Beta individual"""
    
    def __init__(self, battleship_config_path: str, agent_network, random_seed: Optional[int] = None):
        self.config = self._load_config(battleship_config_path)
        self.network = agent_network
        self.state = GameState()
        self.memory_manager = GlobalMemoryManager()
        self.grid_size = tuple(self.config["game_config"]["grid_size"])
        
        if random_seed is not None:
            random.seed(random_seed)
        
        self._initialize_teams()
        self._initialize_scores()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load game configuration from JSON file"""
        return json.loads(Path(path).read_text())

    def _initialize_teams(self):
        """Initialize teams from configuration"""
        team_config = self.config["team_config"]
        
        for team_id, team_data in team_config.items():
            team = Team(
                name=team_data["name"],
                members=team_data["members"],
                grid=GameGrid(size=self.grid_size),
                color=team_data.get("color", "blue"),
                coordination_enabled=team_data.get("team_coordination", False)
            )
            self.state.teams[team_id] = team
        
        logger.info(f"Initialized {len(self.state.teams)} teams")

    def _initialize_scores(self):
        """Initialize score tracking for all players"""
        for team in self.state.teams.values():
            for member in team.members:
                self.state.player_scores[member] = PlayerScore(member)

    # =============================================================================
    # Coordinate and Position Utilities
    # =============================================================================

    def coordinate_to_position(self, coord: str) -> Tuple[int, int]:
        """Convert coordinate string (e.g., 'A1') to grid position (0, 0)"""
        coord = coord.strip().upper()
        if len(coord) < 2:
            raise ValueError(f"Invalid coordinate: {coord}")
        
        letter = coord[0]
        number_str = coord[1:]
        
        if not letter.isalpha() or not number_str.isdigit():
            raise ValueError(f"Invalid coordinate format: {coord}")
        
        row = ord(letter) - ord('A')
        col = int(number_str) - 1
        
        if not (0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]):
            raise ValueError(f"Coordinate out of bounds: {coord}")
        
        return (row, col)

    def position_to_coordinate(self, pos: Tuple[int, int]) -> str:
        """Convert grid position to coordinate string"""
        row, col = pos
        return f"{chr(ord('A') + row)}{col + 1}"

    def _get_player_team_id(self, player_id: str) -> Optional[str]:
        """Get team ID for a player"""
        for team_id, team in self.state.teams.items():
            if player_id in team.members:
                return team_id
        return None

    def _get_enemy_team_id(self, team_id: str) -> Optional[str]:
        """Get the enemy team ID"""
        team_ids = list(self.state.teams.keys())
        for tid in team_ids:
            if tid != team_id:
                return tid
        return None

    # =============================================================================
    # Game Flow - NEW SIMULTANEOUS EXECUTION STRUCTURE
    # =============================================================================

    async def start_game(self):
        """Main game flow with simultaneous execution"""
        await self._log_event("GAME_START", "Dynamic Battleship game starting with simultaneous execution")
        
        # Ship placement phase
        await self._ship_placement_phase()
        
        # Battle phase - NEW STRUCTURE
        await self._simultaneous_battle_phase()
        
        # Game over
        await self._game_over_phase()

    async def _ship_placement_phase(self):
        """Handle ship placement for all teams"""
        self.state.phase = GamePhase.SHIP_PLACEMENT
        await self._log_event("PHASE_START", "Ship placement phase beginning")
        
        ship_set = self.config["game_config"]["ship_sets"][
            self.config["game_config"]["selected_ship_set"]
        ]
        
        for team_id, team in self.state.teams.items():
            ships = self._assign_ships_to_players(team, ship_set)
            self._auto_place_ships(team, ships)
            
            # Record ship ownership in memory
            ship_assignments = {
                ship.owner_id: {"name": ship.name, "size": ship.size}
                for ship in ships
            }
            self.memory_manager.initialize_ship_ownership(ship_assignments)
            
            await self._log_event("SHIP_PLACEMENT", 
                                f"{team.name} completed placement - {len(ships)} ships")

    async def _simultaneous_battle_phase(self):
        """NEW: Main battle phase with simultaneous execution structure"""
        self.state.phase = GamePhase.BATTLE
        await self._log_event("PHASE_START", "Battle phase beginning - SIMULTANEOUS EXECUTION MODE")
        
        max_rounds = self.config["game_config"].get("max_rounds", 100)
        
        while not self._check_victory() and self.state.round_number <= max_rounds:
            await self._log_event("ROUND_START", f"Round {self.state.round_number} beginning")
            
            # Phase 1: Team Alpha Deliberation (if coordination enabled)
            alpha_actions = await self._alpha_deliberation_phase()
            
            # Phase 2: Team Beta Individual Decisions  
            beta_actions = await self._beta_individual_phase()
            
            # Phase 3: SIMULTANEOUS EXECUTION
            await self._execute_simultaneous_actions(alpha_actions, beta_actions)
            
            self.state.round_number += 1
        
        if self.state.round_number > max_rounds:
            await self._log_event("GAME_TIMEOUT", f"Game ended due to round limit ({max_rounds})")

    # =============================================================================
    # NEW: Alpha Team 3-Phase Deliberation Protocol  
    # =============================================================================

    async def _alpha_deliberation_phase(self) -> Dict[str, Optional[PlayerAction]]:
        """NEW: Alpha team 3-phase deliberation protocol"""
        alpha_team_id = None
        alpha_actions = {}
        
        # Find Alpha team (coordination enabled)
        for team_id, team in self.state.teams.items():
            if team.coordination_enabled:
                alpha_team_id = team_id
                break
        
        if not alpha_team_id:
            # No coordination team found
            return {}
        
        team = self.state.teams[alpha_team_id]
        active_members = [m for m in team.members if m not in self.state.eliminated_players]
        
        if not active_members:
            return {}
        
        await self._log_event("ALPHA_DELIBERATION_START", 
                             f"{team.name} beginning 3-phase deliberation - Round {self.state.round_number}")
        
        # Step 1: Team Strategy Proposals (Broadcast)
        proposals = await self._step1_team_strategy_proposals(alpha_team_id, active_members)
        
        # Step 2: Democratic Voting
        votes = await self._step2_democratic_voting(alpha_team_id, active_members, proposals)
        
        # Step 3: Plan Selection
        winning_plan = await self._step3_plan_selection(alpha_team_id, active_members, proposals, votes)
        
        # Parse winning plan into actions
        if winning_plan and winning_plan.team_actions:
            alpha_actions = winning_plan.team_actions
        
        await self._log_event("ALPHA_DELIBERATION_COMPLETE", 
                             f"{team.name} deliberation complete - {len(alpha_actions)} actions")
        
        return alpha_actions

    async def _step1_team_strategy_proposals(self, team_id: str, members: List[str]) -> List[TeamProposal]:
        """Step 1: Team Strategy Proposals (parallel); then broadcast summary with ACK-only instruction."""
        await self._log_event("ALPHA_STEP1", "Team Strategy Proposals - Simple")

        proposals: List[TeamProposal] = []
        team_context = self._get_team_context(team_id, members)

        # EACH MEMBER GETS EXACTLY ONE PROMPT
        for member in members:
            proposal_prompt = (
            f"{team_context}\n\n"
            "🎯 STEP 1: TEAM STRATEGY PROPOSALS\n"
            "Propose a complete strategy for your entire team.\n"
            "Format: PROPOSAL: [player_name] [ACTION] [target], [player_name] [ACTION] [target], [player_name] [ACTION] [target] - [reasoning]\n"
            "Valid actions: BOMB [A1-J10] or MOVE [UP/DOWN/LEFT/RIGHT/ROTATE]\n"
            "\n"
            "⚠️ CRITICAL COORDINATION RULES:\n"
            "• Choose DIFFERENT coordinates for each teammate - no duplicates!\n"
            "• Spread attacks across the grid for maximum coverage\n"
            "• Avoid previously attacked coordinates from battlefield intel\n"
            "• Consider ship safety when proposing movements\n"
            "\n"
            f"{member}, give your team proposal now:"
        )
            try:
                result = await self.network.send_message(
                    "game_master", member, proposal_prompt,
                    max_turns=1, conversation_type="team_proposal"
                )
                proposal_content = self._extract_content_from_conversation(result, member)
                team_proposal = self._parse_team_proposal(member, proposal_content, members)
                if team_proposal:
                    proposals.append(team_proposal)
                    logger.info(f"✅ [PROPOSAL] {member}: {team_proposal.reasoning}")
                else:
                    logger.warning(f"❌ Failed to parse proposal from {member}")
            except Exception as e:
                logger.error(f"Proposal failed for {member}: {e}")

        # Broadcast proposals summary with ACK-only rule
        if proposals:
            proposals_summary = self._format_proposals_summary(proposals, include_reasoning=True)
            broadcast = (
                f"{proposals_summary}\n\n"
                "📢 INSTRUCTION: This is an informational broadcast.\n"
                "Do NOT propose actions or discuss here.\n"
                "Reply with EXACTLY one word: ACK\n"
            )
            logger.info("📢 Showing all proposals to team (ACK-only)...")
            for member in members:
                await self.network.send_message(
                    "game_master", member, broadcast,
                    max_turns=1, conversation_type="proposals_summary"
                )

        return proposals


    def _format_proposals_summary(self, proposals: List[TeamProposal], *, include_reasoning: bool = True) -> str:
        """Format a human-readable summary of proposals for prompts."""
        if not proposals:
            return "No proposals available."
        lines = ["📋 TEAM PROPOSALS SUMMARY"]
        for i, proposal in enumerate(proposals, 1):
            lines.append(f"\n{i}. {proposal.proposer_id}'s plan:")
            for player, action in proposal.team_actions.items():
                action_str = f"{action.action_type.value} {action.target}" if action else "NO ACTION"
                lines.append(f"   - {player}: {action_str}")
            if include_reasoning:
                lines.append(f"   Reasoning: {proposal.reasoning}")
        return "\n".join(lines)

    
    async def _step2_democratic_voting(self, team_id: str, members: List[str],
                                   proposals: List[TeamProposal]) -> Dict[str, str]:
        """Step 2: Democratic voting with embedded game intel + proposals, neutral prompt, and one corrective retry."""
        await self._log_event("ALPHA_STEP2", "Democratic Voting - Neutral with Intel")

        votes: Dict[str, str] = {}
        if not proposals:
            logger.warning("No proposals to vote on")
            return votes

        # ===== Build compact game intel =====
        team = self.state.teams[team_id]
        enemy_team_id = self._get_enemy_team_id(team_id)
        enemy_name = self.state.teams[enemy_team_id].name if enemy_team_id else "Unknown"

        intel_lines = [
            f"[ROUND {self.state.round_number}]",
            f"Team: {team.name}  |  Enemy: {enemy_name}"
        ]
        # Own ship status snapshot for voting context
        for member in members:
            ship = team.grid.get_ship_by_owner(member)
            if not ship:
                continue
            status = "SUNK" if ship.is_sunk else ("DAMAGED" if ship.hits > 0 else "OK")
            pos = self.position_to_coordinate(ship.positions[0]) if ship.positions else "?"
            intel_lines.append(f"- {member}: {ship.name} (size {ship.size}) {status} at {pos} ({ship.orientation})")
        # Battlefield memory summary (hits/misses/sinks/etc.)
        try:
            mem = self.memory_manager.generate_dynamic_battlefield_summary()
            if mem and "No battlefield activity" not in mem:
                intel_lines.append("\nBattlefield Intel:\n" + mem.strip())
        except Exception:
            pass
        intel_context = "\n".join(intel_lines)

        # ===== Format proposals for inclusion in the vote prompt =====
        def _fmt_proposals_summary(items: List[TeamProposal]) -> str:
            lines = ["📋 TEAM PROPOSALS SUMMARY"]
            for i, proposal in enumerate(items, 1):
                lines.append(f"\n{i}. {proposal.proposer_id}'s plan:")
                for player, action in proposal.team_actions.items():
                    action_str = f"{action.action_type.value} {action.target}" if action else "NO ACTION"
                    lines.append(f"   - {player}: {action_str}")
                lines.append(f"   Reasoning: {proposal.reasoning}")
            return "\n".join(lines)

        proposals_context = _fmt_proposals_summary(proposals)

        proposer_names = [p.proposer_id for p in proposals]
        # Neutralize position bias: round-seeded shuffle for reproducibility
        rnd = random.Random(self.state.round_number)
        proposer_order = rnd.sample(proposer_names, k=len(proposer_names))
        proposer_list_neutral = ", ".join(proposer_order)

        # ===== Enhanced strategic vote prompt =====
        base_vote_prompt = (
            f"{intel_context}\n\n"
            f"{proposals_context}\n\n"
            "🗳️ STRATEGIC VOTING (ALPHA STEP 2)\n"
            "WHAT'S AT STAKE:\n"
            "• YOUR SURVIVAL: If your ship is sunk, YOU ARE ELIMINATED from the game\n"
            "• TEAM FIREPOWER: Each teammate eliminated = less firepower next round\n"
            "• VICTORY CONDITION: First team to sink ALL enemy ships wins\n\n"
            "Choose the plan most likely to achieve TEAM VICTORY while considering:\n"
            "🎯 OFFENSIVE STRATEGY:\n"
            "  • Coverage without same-team duplication (wasted shots)\n"
            "  • Hit probability and follow-up potential\n"
            "  • Targeting untested areas vs. focusing fire\n\n"
            "🛡️ DEFENSIVE STRATEGY:\n"
            "  • Ship safety - avoid exposing damaged ships to counterattack\n"
            "  • Movement risk vs. benefit (repositioning can be good or dangerous)\n"
            "  • Preserve team firepower - every ship that survives = more attacks next round\n\n"
            "🏆 TEAM DYNAMICS:\n"
            "  • Think beyond personal survival - you need teammates to win\n"
            "  • A coordinated mediocre plan beats an uncoordinated great plan\n"
            "  • Consider which plan gives the TEAM the best chance to win\n\n"
            "REQUIRED OUTPUT (ONE LINE ONLY):\n"
            "VOTE: <proposer_id> - <strategic reasoning>\n"
            f"Available plans: {proposer_list_neutral}\n"
            "Focus on TEAM VICTORY, not loyalty to any individual proposer."
        )

        correction_vote_prompt = (
            f"{intel_context}\n\n"
            f"{proposals_context}\n\n"
            "⚠️ VOTING FORMAT ERROR - Your previous reply was invalid.\n"
            "Remember: Your ship's survival AND team victory both depend on this choice.\n"
            "REPLY WITH EXACTLY ONE LINE:\n"
            "VOTE: <proposer_id> - <strategic reasoning>\n"
            f"Available plans: {proposer_list_neutral}\n"
            "Choose the plan that maximizes TEAM VICTORY probability."
)   

        # ===== Collect votes (strict schema + one corrective retry) =====
        for member in members:
            try:
                # First attempt
                result = await self.network.send_message(
                    "game_master", member, base_vote_prompt,
                    max_turns=1, conversation_type="voting"
                )
                vote_content = self._extract_content_from_conversation(result, member)
                voted_for = self._parse_vote(vote_content, proposer_names)

                # Guard against action-like replies or parse failure; one corrective retry
                needs_retry = (not voted_for) or re.search(r'\b(BOMB|MOVE)\b', vote_content or "", re.IGNORECASE)
                if needs_retry:
                    result = await self.network.send_message(
                        "game_master", member, correction_vote_prompt,
                        max_turns=1, conversation_type="voting_correction"
                    )
                    vote_content = self._extract_content_from_conversation(result, member)
                    voted_for = self._parse_vote(vote_content, proposer_names)

                if voted_for:
                    votes[member] = voted_for
                    reasoning = self._extract_vote_reasoning(vote_content)
                    logger.info(f"🗳️ {member} voted for {voted_for}: {reasoning}")
                else:
                    logger.warning(f"❌ Failed to parse vote from {member}: {vote_content}")

            except Exception as e:
                logger.error(f"Voting failed for {member}: {e}")

        return votes




    async def _step3_plan_selection(self, team_id: str, members: List[str], 
                                  proposals: List[TeamProposal], votes: Dict[str, str]) -> Optional[TeamProposal]:
        """Step 3: Plan selection with results broadcast to everyone"""
        await self._log_event("ALPHA_STEP3", "Plan Selection - Real Group Chat")
        
        if not proposals or not votes:
            logger.warning("No proposals or votes for plan selection")
            return None
        
        # Count votes for each proposal
        vote_counts = {}
        for proposer in [p.proposer_id for p in proposals]:
            vote_counts[proposer] = 0
        
        for voter, voted_for in votes.items():
            if voted_for in vote_counts:
                vote_counts[voted_for] += 1
        
        # Find winner(s)
        max_votes = max(vote_counts.values()) if vote_counts else 0
        winners = [proposer for proposer, count in vote_counts.items() if count == max_votes]
        
        winning_proposer = None
        selection_message = ""
        
        if len(winners) == 1:
            # Clear winner
            winning_proposer = winners[0]
            selection_message = f"🏆 WINNER: {winning_proposer}'s plan wins with {max_votes} votes!"
            await self._log_event("PLAN_SELECTED", 
                                f"Majority winner: {winning_proposer} with {max_votes} votes")
        elif len(winners) > 1:
            # Tie-breaking: random selection
            winning_proposer = random.choice(winners)
            selection_message = f"🎲 TIE BROKEN: {winning_proposer} selected randomly from {winners}"
            await self._log_event("TIE_BROKEN", 
                                f"Tie broken randomly: {winning_proposer} selected from {winners}")
        
        # Find and return winning proposal
        winning_proposal = None
        for proposal in proposals:
            if proposal.proposer_id == winning_proposer:
                winning_proposal = proposal
                break
        
        # BROADCAST final selection to ALL team members (real group chat!)
        if winning_proposal:
            final_plan = f"{selection_message}\n\nFINAL TEAM ACTIONS:\n"
            for player, action in winning_proposal.team_actions.items():
                action_str = f"{action.action_type.value} {action.target}" if action else "NO ACTION"
                final_plan += f"• {player}: {action_str}\n"
            
            final_plan += "\n📢 INSTRUCTION: This is an informational broadcast.\n"
            final_plan += "Do NOT propose new actions or discuss here.\n"
            final_plan += "Reply with EXACTLY one word: ACK"
            
            logger.info(f"📢 Broadcasting final plan selection to Alpha team...")
            
            # LOG THE WINNING PLAN FOR VISIBILITY
            if winning_proposal:
                logger.info(f"🏆 EXECUTING PLAN: {winning_proposer} won with {max_votes} votes")
                plan_actions = []
                for player, action in winning_proposal.team_actions.items():
                    if action:
                        plan_actions.append(f"{player}: {action.action_type.value} {action.target}")
                    else:
                        plan_actions.append(f"{player}: NO ACTION")
                logger.info(f"📋 PLAN DETAILS: {' | '.join(plan_actions)}")
                logger.info(f"💭 REASONING: {winning_proposal.reasoning}")
            
            for member in members:
                await self.network.send_message(
                    "game_master", member,
                    f"GROUP CHAT UPDATE:\n{final_plan}",
                    max_turns=1, conversation_type="plan_selection"
                )
            
            # Award coordination bonus to effective proposer
            self.state.player_scores[winning_proposer].coordination_bonus += 5
        
        return winning_proposal

    # =============================================================================
    # Beta Team: Individual Decisions (Parallel)
    # =============================================================================

    async def _beta_individual_phase(self) -> Dict[str, Optional[PlayerAction]]:
        """Beta team individual decision making (parallel processing)"""
        beta_team_id = None
        beta_actions = {}
        
        # Find Beta team (no coordination)
        for team_id, team in self.state.teams.items():
            if not team.coordination_enabled:
                beta_team_id = team_id
                break
        
        if not beta_team_id:
            return {}
        
        team = self.state.teams[beta_team_id]
        active_members = [m for m in team.members if m not in self.state.eliminated_players]
        
        if not active_members:
            return {}
        
        await self._log_event("BETA_INDIVIDUAL_START", 
                             f"{team.name} individual decisions - Round {self.state.round_number}")
        
        # All Beta agents make decisions in parallel
        decision_tasks = []
        for member in active_members:
            context = self._get_context_for_player(member)
            prompt = (
                f"{context}\n\n"
                "INDIVIDUAL DECISION (NO TEAM COORDINATION)\n"
                "REQUIRED OUTPUT FORMAT (ONE LINE ONLY):\n"
                "  ACTION: BOMB <A1-J10>\n"
                "    -or-\n"
                "  ACTION: MOVE <UP|DOWN|LEFT|RIGHT|ROTATE>\n"
                "Rules:\n"
                "  • Reply with EXACTLY ONE line starting with 'ACTION:'\n"
                "  • No extra prose. No multiple actions. No voting keywords here.\n"
                "  • If uncertain, prefer ACTION: BOMB on a previously untested coordinate.\n"
            )

            
            task = self.network.send_message(
                "game_master", member, prompt,
                max_turns=1, conversation_type="individual_decision"
            )
            decision_tasks.append((member, task))
        
        # Collect all decisions
        for member, task in decision_tasks:
            try:
                result = await task
                action = self._parse_action_from_conversation(result, member)
                beta_actions[member] = action
                
                if action:
                    logger.info(f"[BETA_INDIVIDUAL] {member}: {action.action_type.value} {action.target}")
                
            except Exception as e:
                logger.error(f"Individual decision failed for {member}: {e}")
                beta_actions[member] = None
        
        await self._log_event("BETA_INDIVIDUAL_COMPLETE", 
                             f"{team.name} individual decisions complete - {len(beta_actions)} actions")
        
        return beta_actions

    # =============================================================================
    # NEW: Simultaneous Execution System
    # =============================================================================

    async def _execute_simultaneous_actions(self, alpha_actions: Dict[str, Optional[PlayerAction]], 
                                      beta_actions: Dict[str, Optional[PlayerAction]]):
        """Execute all actions from both teams with TRUE simultaneous processing"""
        await self._log_event("SIMULTANEOUS_EXECUTION_START", "Processing actions with true simultaneity")
        
        # Combine all actions
        all_actions = {}
        all_actions.update(alpha_actions)
        all_actions.update(beta_actions)
        
        # SNAPSHOT: Capture current state before any processing
        pre_action_grids = {}
        pre_action_ships = {}
        for team_id, team in self.state.teams.items():
            pre_action_grids[team_id] = [row[:] for row in team.grid.grid]  # Deep copy
            pre_action_ships[team_id] = {}
            for ship in team.grid.ships:
                pre_action_ships[team_id][ship.owner_id] = {
                    'positions': ship.positions[:],
                    'hits': ship.hits,
                    'orientation': ship.orientation,
                    'is_sunk': ship.is_sunk
                }
        
        # Group actions by type
        bomb_actions = []
        move_actions = []
        
        for player_id, action in all_actions.items():
            if action and player_id not in self.state.eliminated_players:
                if action.action_type == ActionType.BOMB:
                    bomb_actions.append((player_id, action))
                elif action.action_type == ActionType.MOVE:
                    move_actions.append((player_id, action))
        
        # Process all actions against the snapshot
        await self._process_bomb_actions_simultaneously(bomb_actions, pre_action_grids)
        await self._process_move_actions_simultaneously(move_actions, pre_action_ships)
        
        await self._log_event("SIMULTANEOUS_EXECUTION_COMPLETE", 
                            f"Processed {len(bomb_actions)} bombs, {len(move_actions)} moves simultaneously")
    
    async def _process_bomb_actions_simultaneously(self, bomb_actions: List[Tuple[str, PlayerAction]], pre_action_grids: Dict[str, List[List]]):
        """Process all bomb actions against snapshot grids (allows ghost shots)"""
        if not bomb_actions:
            return
        
        await self._log_event("BOMB_PHASE", f"Processing {len(bomb_actions)} bombs against pre-action positions")
        
        for player_id, action in bomb_actions:
            try:
                target_pos = self.coordinate_to_position(action.target)
                team_id = self._get_player_team_id(player_id)
                enemy_team_id = self._get_enemy_team_id(team_id)
                
                if enemy_team_id:
                    # Use PRE-ACTION grid state for hit detection
                    snapshot_grid = pre_action_grids[enemy_team_id]
                    r, c = target_pos
                    
                    # Determine result based on snapshot
                    if not (0 <= r < len(snapshot_grid) and 0 <= c < len(snapshot_grid[0])):
                        result = "INVALID"
                        sunk_ship = None
                    elif snapshot_grid[r][c] in [CellState.HIT, CellState.MISS]:
                        result = "ALREADY_ATTACKED" 
                        sunk_ship = None
                    elif snapshot_grid[r][c] == CellState.SHIP:
                        # Apply hit to CURRENT grid
                        current_grid = self.state.teams[enemy_team_id].grid
                        current_grid.grid[r][c] = CellState.HIT
                        
                        # Find and update ship
                        sunk_ship = None
                        for ship in current_grid.ships:
                            if (r, c) in ship.positions:
                                ship.hits += 1
                                if ship.is_sunk:
                                    result = "SUNK"
                                    sunk_ship = ship
                                    # Handle elimination
                                    if ship.owner_id not in self.state.eliminated_players:
                                        self.state.eliminated_players.append(ship.owner_id)
                                        self.state.player_scores[ship.owner_id].ship_survival = 0
                                else:
                                    result = "HIT"
                                break
                    else:
                        result = "MISS"
                        sunk_ship = None
                        # Apply miss to current grid
                        self.state.teams[enemy_team_id].grid.grid[r][c] = CellState.MISS
                    
                    # Record attack in memory
                    self.memory_manager.record_coordinate_attack(
                        coordinate=action.target,
                        attacker=player_id,
                        target_team=enemy_team_id,
                        result=result,
                        sunk_ship=sunk_ship.name if sunk_ship else None,
                        ship_owner=sunk_ship.owner_id if sunk_ship else None,
                        player_eliminated=bool(sunk_ship),
                        round_number=self.state.round_number,
                        inform_agents=self.state.teams[team_id].members
                    )
                    
                    await self._log_event("BOMB_RESULT", 
                                        f"{player_id} → {action.target}: {result}" + 
                                        (f" (sunk {sunk_ship.name})" if sunk_ship else ""))
                    
            except Exception as e:
                logger.error(f"Bomb action failed for {player_id}: {e}")

    async def _process_move_actions_simultaneously(self, move_actions: List[Tuple[str, PlayerAction]], pre_action_ships: Dict[str, Dict[str, Dict]]):
        """Process all move actions from snapshot positions"""
        if not move_actions:
            return
        
        await self._log_event("MOVE_PHASE", f"Processing {len(move_actions)} moves from pre-action positions")
        
        for player_id, action in move_actions:
            try:
                team_id = self._get_player_team_id(player_id)
                if not team_id:
                    continue
                
                team = self.state.teams[team_id]
                player_ship = team.grid.get_ship_by_owner(player_id)
                
                if not player_ship:
                    continue
                
                # Use PRE-ACTION position for move calculation
                if player_id in pre_action_ships[team_id]:
                    old_positions = pre_action_ships[team_id][player_id]['positions']
                    old_pos = self.position_to_coordinate(old_positions[0]) if old_positions else "Unknown"
                    
                    # Calculate move from pre-action position
                    success = team.grid.move_ship(player_ship, action.target)
                    new_pos = self.position_to_coordinate(player_ship.positions[0]) if player_ship.positions else old_pos
                    
                    # Record movement
                    self.memory_manager.record_ship_movement(
                        player_id=player_id,
                        ship_name=player_ship.name,
                        movement_type=action.target,
                        from_position=old_pos if success else None,
                        to_position=new_pos if success else None,
                        reason=action.reason,
                        round_number=self.state.round_number,
                        inform_agents=team.members
                    )
                    
                    if success:
                        await self._log_event("MOVE_SUCCESS", 
                                            f"{player_id} moved {player_ship.name} {action.target}: {old_pos} → {new_pos}")
                    else:
                        await self._log_event("MOVE_BLOCKED", 
                                            f"{player_id} move {action.target} blocked")
                
            except Exception as e:
                logger.error(f"Move action failed for {player_id}: {e}")

    def _apply_cross_team_deduplication(self, all_actions: Dict[str, Optional[PlayerAction]]) -> Dict[str, Optional[PlayerAction]]:
        """No deduplication - let agents coordinate naturally"""
        return all_actions

    def _find_alternative_target(self, team_id: str) -> Optional[str]:
        """Find an alternative bombing target"""
        enemy_team_id = self._get_enemy_team_id(team_id)
        if not enemy_team_id:
            return None
        
        enemy_grid = self.state.teams[enemy_team_id].grid
        
        # Try to find an untargeted coordinate
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                if enemy_grid.grid[row][col] == CellState.EMPTY:
                    return self.position_to_coordinate((row, col))
        
        return None

    # =============================================================================
    # Ship Placement (unchanged)
    # =============================================================================

    def _assign_ships_to_players(self, team: Team, ship_set: List[Dict[str, Any]]) -> List[Ship]:
        """Assign ships to team members"""
        ship_pool = []
        
        # Create ship pool from configuration
        for ship_config in ship_set:
            quantity = int(ship_config["quantity"])
            for i in range(quantity):
                name = ship_config["name"]
                if quantity > 1:
                    name += f" {i + 1}"
                
                ship_pool.append(Ship(
                    name=name,
                    size=int(ship_config["size"]),
                    owner_id=""  # Will be assigned below
                ))
        
        # Shuffle and assign to players
        random.shuffle(ship_pool)
        assigned_ships = []
        
        for i, player_id in enumerate(team.members):
            if i < len(ship_pool):
                ship = ship_pool[i]
                ship.owner_id = player_id
                assigned_ships.append(ship)
                logger.info(f"Assigned {ship.name} (size {ship.size}) to {player_id}")
        
        return assigned_ships

    def _auto_place_ships(self, team: Team, ships: List[Ship]):
        """Automatically place ships on team grid"""
        for ship in ships:
            placed = False
            attempts = 0
            max_attempts = 100
            
            while not placed and attempts < max_attempts:
                row = random.randint(0, self.grid_size[0] - 1)
                col = random.randint(0, self.grid_size[1] - 1)
                orientation = random.choice(["horizontal", "vertical"])
                
                if team.grid.place_ship(ship, (row, col), orientation):
                    placed = True
                    coord = self.position_to_coordinate((row, col))
                    logger.info(f"Placed {ship.owner_id}'s {ship.name} at {coord} ({orientation})")
                
                attempts += 1
            
            if not placed:
                logger.error(f"Failed to place {ship.name} for {ship.owner_id}")

    # =============================================================================
    # Parsing and Utility Functions
    # =============================================================================

    def _parse_team_proposal(self, proposer_id: str, content: str, team_members: List[str]) -> Optional[TeamProposal]:
        """Parse team proposal from agent response - ROBUST VERSION"""
        if not content:
            return None
        
        # Handle different dash characters and Unicode variants
        dash_pattern = r'[-–—\u2013\u2014\u002D]'  # Regular hyphen, en-dash, em-dash, Unicode variants
        
        logger.debug(f"Parsing proposal from {proposer_id}: {content[:200]}...")
        
        # Look for PROPOSAL: format with flexible dash handling
        proposal_match = re.search(rf'PROPOSAL:\s*(.*?)\s*{dash_pattern}\s*(.*)', content, re.IGNORECASE | re.DOTALL)
        if not proposal_match:
            # Fallback: try without explicit dash separator (look for reasoning keywords)
            proposal_match = re.search(r'PROPOSAL:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
            if proposal_match:
                full_text = proposal_match.group(1).strip()
                # Split on reasoning keywords that often follow actions
                reasoning_splits = [
                    r'\s+(?:to|for|while|because|since|as|maximizing|opening|probing)\s+',
                    r'\s*;\s*',  # semicolon separator
                    r'\s*\.\s*',  # period separator
                ]
                
                actions_text = full_text
                reasoning = "Strategic coordination"
                
                for split_pattern in reasoning_splits:
                    parts = re.split(split_pattern, full_text, maxsplit=1, flags=re.IGNORECASE)
                    if len(parts) == 2 and len(parts[0]) < len(full_text) * 0.8:  # Don't split too late
                        actions_text = parts[0].strip()
                        reasoning = parts[1].strip()
                        logger.debug(f"Split on pattern '{split_pattern}': actions='{actions_text}', reasoning='{reasoning[:50]}...'")
                        break
            else:
                logger.warning(f"No PROPOSAL: format found in: {content[:100]}")
                return None
        else:
            actions_text = proposal_match.group(1).strip()
            reasoning = proposal_match.group(2).strip() if proposal_match.lastindex >= 2 else "No reasoning provided"
            logger.debug(f"Found dash-separated proposal: actions='{actions_text}', reasoning='{reasoning[:50]}...'")
        
        # Parse individual actions
        team_actions = {}
        
        # Enhanced pattern with word boundaries to prevent partial matches
        action_pattern = r'\b(player_[ab]\d+|[ab]\d+|alpha\d+|beta\d+)\s+(BOMB|MOVE)\s+([A-J](?:10|[1-9])|UP|DOWN|LEFT|RIGHT|ROTATE)\b'
        matches = re.findall(action_pattern, actions_text, re.IGNORECASE)
        
        logger.debug(f"Found {len(matches)} action matches: {matches}")
        
        # Map variations to actual player names
        def normalize_player_name(name_variant: str) -> Optional[str]:
            name_lower = name_variant.lower()
            for member in team_members:
                member_lower = member.lower()
                if name_lower == member_lower:
                    return member
                # Handle a1 -> player_a1, etc.
                member_short = re.sub(r'player_', '', member_lower)
                if name_lower == member_short:
                    return member
            return None
        
        for player_name, action_type, target in matches:
            normalized_name = normalize_player_name(player_name)
            if normalized_name:
                action = PlayerAction(
                    player_id=normalized_name,
                    action_type=ActionType.BOMB if action_type.upper() == "BOMB" else ActionType.MOVE,
                    target=target.upper()
                )
                team_actions[normalized_name] = action
                logger.debug(f"Parsed action: {normalized_name} -> {action_type} {target}")
            else:
                logger.warning(f"Could not normalize player name: {player_name}")
        
        # Ensure all team members have actions
        for member in team_members:
            if member not in team_actions:
                team_actions[member] = None
        
        if not any(team_actions.values()):
            logger.warning(f"No valid actions parsed from: '{actions_text}' (original: {content[:200]})")
            return None
        
        logger.info(f"✅ Successfully parsed proposal from {proposer_id}: {len([a for a in team_actions.values() if a])} actions")
        
        return TeamProposal(
            proposer_id=proposer_id,
            team_actions=team_actions,
            reasoning=reasoning
        )

    def _parse_vote(self, content: str, proposer_names: List[str]) -> Optional[str]:
        if not content:
            return None
        text = content.strip()

        # Build alias map (player_a1 -> {player_a1, a1, a-1, alpha1, alpha-1})
        aliases = {}
        for p in proposer_names:
            m = re.search(r'player_(a|b)(\d+)', p, re.I)
            if not m: 
                continue
            team, num = m.group(1).lower(), m.group(2)
            for alias in [
                p,                               # player_a1
                f"{team}{num}",                  # a1 / b1
                f"{team}-{num}",                 # a-1 / b-1
                ("alpha" if team=="a" else "beta") + num,      # alpha1 / beta1
                ("alpha" if team=="a" else "beta") + "-" + num # alpha-1 / beta-1
            ]:
                aliases[alias.lower()] = p

        upper = text.upper()

        # 1) Canonical format: VOTE: <name>
        m = re.search(r'^\s*VOTE\s*[:\-]\s*([A-Z0-9\-_]+)', text, re.I)
        if m:
            key = m.group(1).lower()
            if key in aliases:
                return aliases[key]
            # direct match against proposer_names
            for p in proposer_names:
                if p.lower() == key:
                    return p

        # 2) Natural language: "I vote for X" / "My vote is X" / "Vote for X"
        m = re.search(r'\b(?:i\s+vote\s+for|my\s+vote\s+is|vote\s+for)\s+([A-Z0-9\-_]+)\b', text, re.I)
        if m:
            key = m.group(1).lower()
            if key in aliases:
                return aliases[key]
            for p in proposer_names:
                if p.lower() == key:
                    return p

        # 3) Bare identifier on its own line
        if len(text.split()) == 1:
            key = text.lower()
            if key in aliases:
                return aliases[key]
            for p in proposer_names:
                if p.lower() == key:
                    return p

        # 4) Last resort: any proposer string present anywhere
        for p in proposer_names:
            if p.lower() in text.lower():
                return p

        return None


    def _extract_vote_reasoning(self, vote_content: str) -> str:
        """Extract reasoning from vote content"""
        if not vote_content:
            return "No reasoning provided"
        
        # Look for reasoning after VOTE: format
        vote_match = re.search(r'VOTE:\s*\w+.*?-\s*(.*)', vote_content, re.IGNORECASE | re.DOTALL)
        if vote_match:
            reasoning = vote_match.group(1).strip()
            # Return full reasoning, no truncation
            return reasoning if reasoning else "No reasoning provided"
        
        # Fallback: look for "reasoning" keyword
        reasoning_match = re.search(r'reasoning[:\s]+(.*)', vote_content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            return reasoning if reasoning else "No reasoning provided"
        
        return "No reasoning provided"

    def _get_team_context(self, team_id: str, members: List[str]) -> str:
        """Get team context for all members"""
        team = self.state.teams[team_id]
        enemy_team_id = self._get_enemy_team_id(team_id)
        
        context = f"🎮 ALPHA TEAM COORDINATION - Round {self.state.round_number}\n"
        context += f"Team: {team.name}\n"
        context += f"Enemy Team: {self.state.teams[enemy_team_id].name if enemy_team_id else 'Unknown'}\n\n"
        
        context += "TEAM SHIP STATUS:\n"
        for member in members:
            player_ship = team.grid.get_ship_by_owner(member)
            if player_ship:
                if player_ship.is_sunk:
                    context += f"• {member}: {player_ship.name} (SUNK - eliminated)\n"
                elif player_ship.hits > 0:
                    context += f"• {member}: {player_ship.name} (size {player_ship.size}, {player_ship.hits} hits - DAMAGED!)\n"
                    if player_ship.positions:
                        start_pos = self.position_to_coordinate(player_ship.positions[0])
                        context += f"  Location: {start_pos} ({player_ship.orientation})\n"
                        valid_moves = team.grid.get_valid_moves_for_ship(player_ship)
                        context += f"  Valid moves: {', '.join(valid_moves) if valid_moves else 'None'}\n"
                else:
                    context += f"• {member}: {player_ship.name} (size {player_ship.size}, undamaged)\n"
                    if player_ship.positions:
                        start_pos = self.position_to_coordinate(player_ship.positions[0])
                        context += f"  Location: {start_pos} ({player_ship.orientation})\n"
                        valid_moves = team.grid.get_valid_moves_for_ship(player_ship)
                        context += f"  Valid moves: {', '.join(valid_moves) if valid_moves else 'None'}\n"
        
        context += f"\nOBJECTIVE: Coordinate as a team to attack enemy ships and win!\n"
        context += f"Actions: BOMB [coordinate] or MOVE [direction]\n"
        context += f"Enemy grid coordinates: A1 to J10\n"
        
        # Get battlefield memory with proper error handling
        try:
            memory_summary = self.memory_manager.generate_dynamic_battlefield_summary()
            # logger.info(f"📊 Memory summary generated: {memory_summary[:100] if memory_summary else 'None'}...")
            if memory_summary and "No battlefield activity" not in memory_summary:
                # Add explicit coordinate list for clarity
                attacked_coords = self._get_all_attacked_coordinates()
                coord_list = f"ATTACKED COORDINATES (AVOID THESE): {', '.join(sorted(attacked_coords))}" if attacked_coords else "No coordinates attacked yet"
                
                context += f"\nPREVIOUS ATTACKS:\n{coord_list}\n\n{memory_summary}\n"
                # logger.info(f"✅ Battlefield intel added: {len(attacked_coords)} coordinates")
            else:
                context += f"\nPREVIOUS ATTACKS: No coordinates attacked yet\n"
                logger.warning(f"⚠️ No battlefield intel available")
        except Exception as e:
            logger.error(f"❌ Memory system failed: {e}")
        
        return context

    def _get_all_attacked_coordinates(self) -> List[str]:
        """Get list of all coordinates attacked in previous rounds"""
        attacked = set()
        
        # Check all team grids for attacked coordinates
        for team in self.state.teams.values():
            for row_idx, row in enumerate(team.grid.grid):
                for col_idx, cell in enumerate(row):
                    if cell in [CellState.HIT, CellState.MISS]:
                        coord = self.position_to_coordinate((row_idx, col_idx))
                        attacked.add(coord)
        
        return list(attacked)
    
    def _get_context_for_player(self, player_id: str) -> str:
        """Generate context information for a player"""
        team_id = self._get_player_team_id(player_id)
        if not team_id:
            return "Context unavailable"
        
        team = self.state.teams[team_id]
        player_ship = team.grid.get_ship_by_owner(player_id)
        enemy_team_id = self._get_enemy_team_id(team_id)
        
        # Basic context
        context = f"[BATTLESHIP GAME - Round {self.state.round_number}]\n"
        context += f"Team: {team.name}\n"
        context += f"Enemy Team: {self.state.teams[enemy_team_id].name if enemy_team_id else 'Unknown'}\n\n"
        
        # Ship status
        if player_ship:
            if player_ship.is_sunk:
                context += f"Your ship: {player_ship.name} (SUNK - you are eliminated)\n"
            elif player_ship.hits > 0:
                context += f"Your ship: {player_ship.name} (size {player_ship.size}, took {player_ship.hits} hits - DAMAGED!)\n"
            else:
                context += f"Your ship: {player_ship.name} (size {player_ship.size}, undamaged)\n"
            
            # Ship position
            if player_ship.positions:
                start_pos = self.position_to_coordinate(player_ship.positions[0])
                context += f"Your ship location: {start_pos} ({player_ship.orientation})\n"
        else:
            context += "Your ship: None assigned\n"
        
        # Valid moves
        if player_ship and not player_ship.is_sunk:
            valid_moves = team.grid.get_valid_moves_for_ship(player_ship)
            if valid_moves:
                context += f"Valid moves: {', '.join(valid_moves)}\n"
        
        # Game objective
        context += f"\nOBJECTIVE: Attack enemy ships with BOMB [coordinate] or move your ship with MOVE [direction]\n"
        context += f"Enemy grid coordinates: A1 to J10\n"
        context += f"Valid actions: BOMB [any coordinate A1-J10], MOVE [UP/DOWN/LEFT/RIGHT/ROTATE]\n"
        
        
        # Get battlefield memory with proper error handling
        try:
            memory_summary = self.memory_manager.generate_dynamic_battlefield_summary()
            # logger.info(f"📊 Memory summary generated: {memory_summary[:100] if memory_summary else 'None'}...")
            if memory_summary and "No battlefield activity" not in memory_summary:
                # Add explicit coordinate list for clarity
                attacked_coords = self._get_all_attacked_coordinates()
                coord_list = f"ATTACKED COORDINATES (AVOID THESE): {', '.join(sorted(attacked_coords))}" if attacked_coords else "No coordinates attacked yet"
                
                context += f"\nPREVIOUS ATTACKS:\n{coord_list}\n\n{memory_summary}\n"
                logger.info(f"✅ Battlefield intel added: {len(attacked_coords)} coordinates")
            else:
                context += f"\nPREVIOUS ATTACKS: No coordinates attacked yet\n"
                logger.warning(f"⚠️ No battlefield intel available")
        except Exception as e:
            logger.error(f"❌ Memory system failed: {e}")
        
        return context

    def _parse_action_from_conversation(self, conversation: Dict[str, Any], player_id: str) -> Optional[PlayerAction]:
        """Parse action from conversation result"""
        if not conversation or "chat_result" not in conversation:
            return None
        
        chat_result = conversation["chat_result"]
        if not hasattr(chat_result, "chat_history"):
            return None
        
        # Look for player's response in chat history
        for message in chat_result.chat_history:
            if message.get("name") == player_id:
                content = message.get("content", "")
                return self._parse_action_from_text(content, player_id)
        
        return None

    def _parse_action_from_text(self, text: str, player_id: str) -> Optional[PlayerAction]:
        """Parse action from text content"""
        if not text:
            return None
        
        text = text.upper().strip()
        
        # Parse BOMB action
        bomb_match = re.search(r'\bBOMB\s+([A-J](?:10|[1-9]))\b', text)
        if bomb_match:
            coordinate = bomb_match.group(1)
            reason_match = re.search(r'BOMB\s+[A-J](?:10|[1-9])\s*-?\s*(.*)', text)
            reason = reason_match.group(1).strip() if reason_match else None
            return PlayerAction(player_id, ActionType.BOMB, coordinate, reason)
        
        # Parse MOVE action
        move_match = re.search(r'\bMOVE\s+(UP|DOWN|LEFT|RIGHT|ROTATE)\b', text)
        if move_match:
            direction = move_match.group(1)
            reason_match = re.search(r'MOVE\s+(?:UP|DOWN|LEFT|RIGHT|ROTATE)\s*-?\s*(.*)', text)
            reason = reason_match.group(1).strip() if reason_match else None
            return PlayerAction(player_id, ActionType.MOVE, direction, reason)
        
        return None

    def _extract_content_from_conversation(self, conversation: Dict[str, Any], player_id: str) -> str:
        """Extract content from conversation result"""
        if not conversation or "chat_result" not in conversation:
            return ""
        
        chat_result = conversation["chat_result"]
        if not hasattr(chat_result, "chat_history"):
            return ""
        
        for message in chat_result.chat_history:
            if message.get("name") == player_id:
                content = message.get("content", "")
                return str(content)
        
        return ""

    # =============================================================================
    # Game State and Victory
    # =============================================================================

    def _check_victory(self) -> bool:
        """Check if any team has won"""
        team_ships_remaining = {}
        
        for team_id, team in self.state.teams.items():
            remaining = sum(1 for ship in team.grid.ships if not ship.is_sunk)
            team_ships_remaining[team_id] = remaining
        
        # Check for winner
        teams_with_ships = [tid for tid, count in team_ships_remaining.items() if count > 0]
        
        if len(teams_with_ships) == 1:
            self.state.winner = teams_with_ships[0]
            return True
        elif len(teams_with_ships) == 0:
            self.state.winner = None  # Draw
            return True
        
        return False

    async def _game_over_phase(self):
        """Handle game over and scoring"""
        self.state.phase = GamePhase.GAME_OVER
        
        # Calculate final scores
        self._calculate_final_scores()
        
        if self.state.winner:
            winner_team = self.state.teams[self.state.winner]
            await self._log_event("GAME_OVER", 
                                f"{winner_team.name} wins after {self.state.round_number} rounds!",
                                {"winner": self.state.winner})
        else:
            await self._log_event("GAME_OVER", 
                                "Draw - all ships destroyed!",
                                {"winner": None})
        
        # Log final scores
        await self._log_final_scores()

    def _calculate_final_scores(self):
        """Calculate final scores for all players"""
        winning_team = self.state.winner
        
        for team_id, team in self.state.teams.items():
            for member in team.members:
                score = self.state.player_scores[member]
                
                # Team victory bonus
                if team_id == winning_team:
                    score.team_victory = 100
                
                # Ship survival bonus
                player_ship = team.grid.get_ship_by_owner(member)
                if player_ship and not player_ship.is_sunk:
                    score.ship_survival = 20

    async def _log_final_scores(self):
        """Log final score summary"""
        score_summary = "FINAL SCORES:\n"
        
        for team_id, team in self.state.teams.items():
            score_summary += f"\n{team.name}:\n"
            team_total = 0
            
            for member in team.members:
                score = self.state.player_scores[member]
                score_summary += f"  {member}: {score.total_score} points\n"
                score_summary += f"    (Victory: {score.team_victory}, Survival: {score.ship_survival}, "
                score_summary += f"Coordination: {score.coordination_bonus}, Penalties: {score.waste_penalty})\n"
                team_total += score.total_score
            
            score_summary += f"  Team Total: {team_total}\n"
        
        await self._log_event("FINAL_SCORES", score_summary)

    # =============================================================================
    # Logging and Export  
    # =============================================================================

    async def _log_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log game event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
            "game_state": {
                "phase": self.state.phase.value,
                "round": self.state.round_number,
                "current_team": self.state.current_team,
                "eliminated_players": list(self.state.eliminated_players),
            },
        }
        
        self.state.game_log.append(event)
        
        # Log important events
        if event_type in {"GAME_START", "PHASE_START", "ALPHA_DELIBERATION_START", "BETA_INDIVIDUAL_START", 
                         "SIMULTANEOUS_EXECUTION_START", "BOMB_RESULT", "MOVE_SUCCESS", "GAME_OVER"}:
            logger.info(f"[{event_type}] {message}")

    def save_game_log(self, filename: Optional[str] = None) -> Path:
        """Save complete game log with scores and statistics"""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"simultaneous_battleship_{timestamp}.json"
        
        # Compile comprehensive game data
        game_data = {
            "game_type": "simultaneous_battleship",
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "final_state": {
                "phase": self.state.phase.value,
                "winner": self.state.winner,
                "winner_name": self.state.teams[self.state.winner].name if self.state.winner else None,
                "total_rounds": self.state.round_number,
                "eliminated_players": self.state.eliminated_players,
            },
            "team_stats": {},
            "player_scores": {},
            "game_log": self.state.game_log,
            "memory_export": self.memory_manager.export_all_memories(),
        }
        
        # Team statistics
        for team_id, team in self.state.teams.items():
            ships_remaining = sum(1 for ship in team.grid.ships if not ship.is_sunk)
            total_ships = len(team.grid.ships)
            
            game_data["team_stats"][team_id] = {
                "name": team.name,
                "coordination_enabled": team.coordination_enabled,
                "members": team.members,
                "ships_remaining": ships_remaining,
                "ships_total": total_ships,
                "survival_rate": ships_remaining / total_ships if total_ships > 0 else 0.0,
                "eliminated_members": [p for p in team.members if p in self.state.eliminated_players],
            }
        
        # Player scores
        for player_id, score in self.state.player_scores.items():
            game_data["player_scores"][player_id] = {
                "total_score": score.total_score,
                "team_victory": score.team_victory,
                "ship_survival": score.ship_survival,
                "coordination_bonus": score.coordination_bonus,
                "waste_penalty": score.waste_penalty,
                "eliminated": player_id in self.state.eliminated_players,
            }
        
        # Save to file
        file_path = output_dir / filename
        with file_path.open('w') as f:
            json.dump(game_data, f, indent=2)
        
        logger.info(f"Game log saved to {file_path}")
        return file_path

    def get_game_statistics(self) -> Dict[str, Any]:
        """Get comprehensive game statistics"""
        stats = {
            "game_type": "simultaneous_battleship",
            "duration_rounds": self.state.round_number,
            "winner": self.state.winner,
            "winner_name": self.state.teams[self.state.winner].name if self.state.winner else None,
            "total_eliminations": len(self.state.eliminated_players),
            "team_performance": {},
            "coordination_analysis": {},
            "action_metrics": {},
            "score_summary": {},
        }
        
        # Team performance
        for team_id, team in self.state.teams.items():
            ships_remaining = sum(1 for ship in team.grid.ships if not ship.is_sunk)
            total_ships = len(team.grid.ships)
            team_score = sum(self.state.player_scores[member].total_score for member in team.members)
            
            stats["team_performance"][team_id] = {
                "name": team.name,
                "coordination_enabled": team.coordination_enabled,
                "ships_remaining": ships_remaining,
                "survival_rate": ships_remaining / total_ships if total_ships > 0 else 0.0,
                "team_score": team_score,
                "average_player_score": team_score / len(team.members) if team.members else 0,
            }
        
        # Action metrics from game log
        bomb_actions = len([e for e in self.state.game_log if e["event_type"] == "BOMB_RESULT"])
        move_actions = len([e for e in self.state.game_log if e["event_type"] in ["MOVE_SUCCESS", "MOVE_BLOCKED"]])
        
        stats["action_metrics"] = {
            "total_bomb_actions": bomb_actions,
            "total_move_actions": move_actions,
            "action_ratio": move_actions / bomb_actions if bomb_actions > 0 else 0,
        }
        
        # Score summary
        for player_id, score in self.state.player_scores.items():
            stats["score_summary"][player_id] = {
                "total": score.total_score,
                "breakdown": {
                    "victory": score.team_victory,
                    "survival": score.ship_survival,
                    "coordination": score.coordination_bonus,
                    "penalties": score.waste_penalty,
                }
            }
        
        return stats


# =============================================================================
# Main Runner Function
# =============================================================================

async def run_simultaneous_battleship_simulation(agent_network, battleship_config_path: str, 
                                               random_seed: Optional[int] = None):
    """
    Main entry point for running Simultaneous Battleship simulation
    
    Args:
        agent_network: Configured AgentNetwork instance
        battleship_config_path: Path to battleship configuration JSON
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (game_instance, statistics_dict)
    """
    game = BattleshipGame(battleship_config_path, agent_network, random_seed)
    
    try:
        logger.info("🚀 STARTING SIMULTANEOUS BATTLESHIP SIMULATION")
        logger.info("=" * 60)
        logger.info("🎯 Experiment: Alpha 3-Phase Coordination vs Beta Individual Decisions")
        logger.info("⚡ NEW: Simultaneous execution eliminates turn order bias")
        logger.info("🚢 Features: Individual ship ownership, BOMB/MOVE actions, player elimination")
        logger.info("📊 Scoring: Team victory, ship survival, coordination bonuses, waste penalties")
        
        await game.start_game()
        
        stats = game.get_game_statistics()
        log_file = game.save_game_log()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎮 SIMULTANEOUS BATTLESHIP SIMULATION COMPLETE")
        print("=" * 60)
        
        if game.state.winner:
            winner_name = game.state.teams[game.state.winner].name
            coordination_type = "Coordinated" if game.state.teams[game.state.winner].coordination_enabled else "Individual"
            print(f"🏆 Winner: {winner_name} ({coordination_type})")
        else:
            print("🤝 Result: Draw")
        
        print(f"📊 Rounds: {game.state.round_number}")
        print(f"💀 Eliminations: {len(game.state.eliminated_players)}")
        print(f"📁 Log saved: {log_file}")
        
        # Team scores
        print("\n📈 TEAM SCORES:")
        for team_id, team_data in stats["team_performance"].items():
            coord_type = "Coordinated" if team_data["coordination_enabled"] else "Individual"
            print(f"  {team_data['name']} ({coord_type}): {team_data['team_score']} points")
        
        print("=" * 60)
        
        return game, stats
        
    except Exception as e:
        logger.error(f"Simultaneous battleship simulation failed: {e}")
        raise


if __name__ == "__main__":
    print("Run via battleship_runner.py for proper async execution.")
    print("This file provides the BattleshipGame class and run_simultaneous_battleship_simulation function.")