"""
Battleship Game Engine for Agent Network Framework

Implements adversarial battleship gameplay with team-based AI agents.
Supports variable team sizes, AI assistants, and personality-driven gameplay.
"""

import json
import asyncio
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime
from memory_manager import GlobalMemoryManager, MemoryType

logger = logging.getLogger(__name__)


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


@dataclass
class Ship:
    name: str
    size: int
    positions: List[Tuple[int, int]] = field(default_factory=list)
    hits: int = 0
    
    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.size


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
        """Place a ship on the grid. Returns True if successful."""
        row, col = start_pos
        positions = []
        
        # Calculate ship positions
        for i in range(ship.size):
            if orientation.lower() == 'horizontal':
                new_pos = (row, col + i)
            else:  # vertical
                new_pos = (row + i, col)
            
            # Check bounds
            if (new_pos[0] >= self.size[0] or new_pos[1] >= self.size[1] or
                new_pos[0] < 0 or new_pos[1] < 0):
                return False
            
            # Check for collision
            if self.grid[new_pos[0]][new_pos[1]] != CellState.EMPTY:
                return False
            
            positions.append(new_pos)
        
        # Place ship
        ship.positions = positions
        for pos in positions:
            self.grid[pos[0]][pos[1]] = CellState.SHIP
        
        self.ships.append(ship)
        return True
    
    def attack(self, target: Tuple[int, int]) -> Tuple[str, Optional[Ship]]:
        """Attack a position. Returns (result, ship_if_sunk)"""
        row, col = target
        
        if row >= self.size[0] or col >= self.size[1] or row < 0 or col < 0:
            return "INVALID", None
        
        current_state = self.grid[row][col]
        
        if current_state in [CellState.HIT, CellState.MISS]:
            return "ALREADY_ATTACKED", None
        
        if current_state == CellState.SHIP:
            self.grid[row][col] = CellState.HIT
            
            # Find which ship was hit
            hit_ship = None
            for ship in self.ships:
                if target in ship.positions:
                    ship.hits += 1
                    hit_ship = ship
                    break
            
            if hit_ship and hit_ship.is_sunk:
                return "SUNK", hit_ship
            else:
                return "HIT", None
        else:
            self.grid[row][col] = CellState.MISS
            return "MISS", None
    
    def all_ships_sunk(self) -> bool:
        """Check if all ships are sunk"""
        return all(ship.is_sunk for ship in self.ships)
    
    def get_display_grid(self, hide_ships: bool = True) -> List[List[str]]:
        """Get grid for display purposes"""
        display = []
        for row in self.grid:
            display_row = []
            for cell in row:
                if hide_ships and cell == CellState.SHIP:
                    display_row.append(CellState.EMPTY.value)
                else:
                    display_row.append(cell.value)
            display.append(display_row)
        return display




@dataclass
class Team:
    name: str
    members: List[str]
    leader: str
    grid: GameGrid
    color: str = "blue"
    ship_placement_complete: bool = False


@dataclass
class GameState:
    phase: GamePhase = GamePhase.SETUP
    current_team: Optional[str] = None
    current_player: Optional[str] = None
    round_number: int = 1
    teams: Dict[str, Team] = field(default_factory=dict)
    game_log: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None


class GameMaster:
    """Programmatic game master - no LLM needed"""
    
    def __init__(self, game_instance):
        self.game = game_instance
        self.message_log = []
    
    async def announce_attack_result(self, player_id: str, coordinate: str, 
                                   result: str, sunk_ship: Optional[Ship] = None):
        """Announce attack result - no LLM required"""
        if result == "HIT":
            announcement = f"🎯 HIT at {coordinate}!"
        elif result == "MISS": 
            announcement = f"💦 MISS at {coordinate}."
        elif result == "SUNK" and sunk_ship:
            announcement = f"💥 HIT and SUNK! {sunk_ship.name} destroyed at {coordinate}!"
        elif result == "ALREADY_ATTACKED":
            announcement = f"❌ {coordinate} already attacked. Choose different coordinate."
        else:
            announcement = f"📍 {result} at {coordinate}."
        
        # Log the announcement
        self.message_log.append({
            'type': 'attack_result',
            'player': player_id,
            'coordinate': coordinate,
            'result': result,
            'announcement': announcement,
            'timestamp': datetime.now().isoformat()
        })
        
        # Display result without sending LLM messages
        logger.info(f"[GAME_MASTER] {announcement}")
        
        return announcement
    
    async def announce_round_results(self, team_id: str, round_results: List[str]):
        """Share round results - programmatic only"""
        team = self.game.state.teams[team_id]
        all_attempted = self.game._get_all_attempted_coordinates()
        
        if round_results:
            summary = f"""
🏁 Round {self.game.state.round_number} Results - {team.name}:

This round's attacks:
{chr(10).join(round_results)}

📊 Total battlefield attempts: {len(all_attempted)} coordinates
🎯 Remaining valid targets: {100 - len(all_attempted)}
            """.strip()
            
            # Log internally without sending messages to agents
            self.message_log.append({
                'type': 'round_summary',
                'team': team_id,
                'round': self.game.state.round_number,
                'results': round_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"[GAME_MASTER] {team.name} round complete: {', '.join([r.split(':')[0] for r in round_results])}")
    
    async def announce_game_over(self, winner_team_id: str = None):
        """Announce game completion - programmatic"""
        if winner_team_id:
            winner = self.game.state.teams[winner_team_id]
            announcement = f"🏆 GAME OVER! {winner.name} wins the battle!"
        else:
            announcement = f"🏁 GAME OVER! Match ended without clear winner."
        
        self.message_log.append({
            'type': 'game_over',
            'winner': winner_team_id,
            'rounds': self.game.state.round_number,
            'announcement': announcement,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"[GAME_MASTER] {announcement}")
        return announcement
    
    def get_game_master_log(self) -> List[Dict]:
        """Get all game master announcements for logging"""
        return self.message_log


class BattleshipGame:
    """Main battleship game controller"""
    
    def __init__(self, battleship_config_path: str, agent_network):
        self.config = self._load_config(battleship_config_path)
        self.network = agent_network
        self.state = GameState()
        self.coordinate_history: Dict[str, List[str]] = {}
        
        # Initialize programmatic game master (no LLM needed)
        self.game_master = GameMaster(self)
        
        self.memory_manager = GlobalMemoryManager()
        
        # Game configuration - set BEFORE initializing teams
        self.grid_size = tuple(self.config['game_config']['grid_size'])
        
        # Read communication rounds from config file
        turn_config = self.config.get('turn_config', {})
        self.max_communication_rounds = turn_config.get('max_team_communication_rounds', 3)
        self.max_assistant_turns = turn_config.get('max_assistant_interaction_turns', 3)
        
        # Initialize teams (this may use self.grid_size)
        self._initialize_teams()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load battleship configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_teams(self):
        """Initialize teams from configuration"""
        team_config = self.config['team_config']
        
        for team_id, team_data in team_config.items():
            grid = GameGrid(size=self.grid_size)
            team = Team(
                name=team_data['name'],
                members=team_data['members'],
                leader=team_data['leader'],
                grid=grid,
                color=team_data.get('color', 'blue')
            )
            self.state.teams[team_id] = team
            
            # Initialize coordinate history for team members
            for member in team.members:
                self.coordinate_history[member] = []
        
        logger.info(f"Initialized {len(self.state.teams)} teams")
    
    def coordinate_to_position(self, coordinate: str) -> Tuple[int, int]:
        """Convert coordinate like 'B7' to grid position (1, 6)"""
        if len(coordinate) < 2:
            raise ValueError(f"Invalid coordinate format: {coordinate}")
        
        letter = coordinate[0].upper()
        number = coordinate[1:]
        
        row = ord(letter) - ord('A')
        col = int(number) - 1
        
        return (row, col)
    
    def position_to_coordinate(self, position: Tuple[int, int]) -> str:
        """Convert grid position (1, 6) to coordinate 'B7'"""
        row, col = position
        letter = chr(ord('A') + row)
        number = str(col + 1)
        return f"{letter}{number}"
    
    async def start_game(self):
        """Start the battleship game"""
        await self._log_game_event("GAME_START", "Battleship game beginning")
        
        # Phase 1: Ship Placement
        await self._ship_placement_phase()
        
        # Phase 2: Battle Phase
        await self._battle_phase()
        
        # Phase 3: Game Over
        await self._game_over_phase()
    
    async def _ship_placement_phase(self):
        """Handle ship placement for all teams"""
        self.state.phase = GamePhase.SHIP_PLACEMENT
        await self._log_game_event("PHASE_START", "Ship placement phase beginning")
        
        for team_id, team in self.state.teams.items():
            await self._team_ship_placement(team_id, team)
        
        await self._log_game_event("PHASE_END", "All teams have placed their ships")
    
    async def _team_ship_placement(self, team_id: str, team: Team):
        """Handle ship placement for a single team"""
        logger.info(f"[SHIP_PLACEMENT] {team.name} beginning ship placement")
        
        # Get ship configuration
        ship_set = self.config['game_config']['ship_sets'][
            self.config['game_config']['selected_ship_set']
        ]
        
        # Read deliberation rounds from team config
        team_config = self.config['team_config'][team_id]
        deliberation_turns = team_config.get('ship_placement_deliberation_turns', 
                            self.config.get('game_phases', {}).get('ship_placement', {}).get('max_deliberation_turns', 3))
        
        logger.info(f"[SHIP_PLACEMENT] {team.name} will have {deliberation_turns} deliberation rounds")
        
        # Team deliberation with configurable rounds
        await self._ship_placement_discussion(team_id, team.leader, deliberation_turns)

        # Auto-place ships
        self._auto_place_ships(team, ship_set)
        team.ship_placement_complete = True
        
        logger.info(f"[SHIP_PLACEMENT] {team.name} completed ship placement - {len(team.grid.ships)} ships placed")

    async def _ship_placement_discussion(self, team_id: str, leader_id: str, max_rounds: int):
        """CLEAN ship placement: Leader asks -> Players respond -> Leader decides"""
        team = self.state.teams[team_id]
        other_members = [m for m in team.members if m != leader_id]
        
        if not other_members:
            logger.info(f"[SHIP_PLACEMENT] {team.name} - Leader {leader_id} deciding alone (no team members)")
            return
        
        ship_placement_questions = [
            "Where should we place our Carrier (5 spaces)? Edge or center?",
            "Battleship (4 spaces) - horizontal or vertical? Which area?", 
            "Small ships strategy - cluster for defense or scatter for coverage?"
        ]
        
        # Collect all player inputs across all rounds
        all_player_inputs = []
        
        for round_num in range(min(max_rounds, len(ship_placement_questions))):
            question = ship_placement_questions[round_num]
            
            logger.info(f"[SHIP_PLACEMENT] {team.name} Round {round_num + 1}/{max_rounds}: {question}")
            
            # Step 1: Leader asks the ACTUAL QUESTION to all players at once
            # Send the question directly - NO generic prompts
            for member in other_members:
                await self._send_clean_message(leader_id, member, question)
            
            # Step 2: Each player responds with their input
            # We let the conversation system handle the responses naturally
            # The responses will be captured in the conversation logs
            
            # Brief pause to let responses complete
            await asyncio.sleep(1)
        
        # Step 3: Leader makes final decision based on all inputs
        if max_rounds > 0:
            # Create summary of what was discussed
            discussed_topics = ship_placement_questions[:max_rounds]
            
            # Leader makes internal decision (no broadcast needed for simulation)
            logger.info(f"[SHIP_PLACEMENT] {team.name} - Leader {leader_id} making final decisions based on: {', '.join(discussed_topics)}")
            
            # Optional: Leader could announce decision in a real implementation
            # For simulation, we skip the decision announcement to avoid extra messages

    # Add this to your BattleshipGame class:

    async def _send_battle_message(self, sender: str, recipient: str, question: str):
        """
        Send a tactical question during battle, instructing
        the recipient to reply in the COORDINATE:[X] format.
        """
        prompt = (
            f"{question}\n\n"
            "Suggest coordinate for attack.\n"
            "Format: COORDINATE: [A1] - REASONING: [why]\n"
            "Maximum 25 words."
        )
        try:
            await self.network.send_message(
                sender_id=sender,
                receiver_id=recipient,
                message=prompt,
                max_turns=1
            )
        except Exception as e:
            logger.error(f"Failed battle message {sender}->{recipient}: {e}")

    
    def _get_leader_style(self, leader_id: str) -> str:
        """Get leader's decision-making style"""
        agent_assignment = self.config['agent_assignments'].get(leader_id, {})
        profile_name = agent_assignment.get('profile', '')
        profile = self.config['player_profiles'].get(profile_name, {})
        return profile.get('leadership_style', 'collaborative')

    async def _send_clean_message(self, sender: str, recipient: str, message: str):
        """Send clean message without generic prompts"""    
        try:
            # NO "Give your tactical input" or other generic crap
            # Just send the actual question/message
    
            
            clean_message = f"{message}\n\nReminder: Maximum 50 words."
            
            await self.network.send_message(
                sender_id=sender,
                receiver_id=recipient, 
                message=clean_message,
                max_turns=1
            )
        except Exception as e:
            logger.error(f"Failed clean message {sender} -> {recipient}: {e}")
    
    def _auto_place_ships(self, team: Team, ship_set: List[Dict]):
        """Auto-place ships randomly for simulation"""
        for ship_config in ship_set:
            for _ in range(ship_config['quantity']):
                ship = Ship(name=ship_config['name'], size=ship_config['size'])
                
                # Try to place ship randomly
                max_attempts = 100
                for attempt in range(max_attempts):
                    row = random.randint(0, self.grid_size[0] - 1)
                    col = random.randint(0, self.grid_size[1] - 1)
                    orientation = random.choice(['horizontal', 'vertical'])
                    
                    if team.grid.place_ship(ship, (row, col), orientation):
                        logger.info(f"Placed {ship.name} at {self.position_to_coordinate((row, col))} ({orientation})")
                        break
                else:
                    logger.error(f"Failed to place {ship.name} after {max_attempts} attempts")
    
    async def _battle_phase(self):
        """Main battle phase with turn-based gameplay"""
        self.state.phase = GamePhase.BATTLE
        await self._log_game_event("PHASE_START", "Battle phase beginning")
        
        team_ids = list(self.state.teams.keys())
        
        while not self._check_victory_condition():
            # Each team takes their turn
            for team_id in team_ids:
                if self._check_victory_condition():
                    break
                
                await self._team_battle_round(team_id)
                
                # Share round results with team
                await self._share_round_results(team_id)
            
            self.state.round_number += 1
            
            # Safety check to prevent infinite games
            if self.state.round_number > 100:
                await self._log_game_event("GAME_TIMEOUT", "Game ended due to round limit")
                break
    
    async def _team_battle_round(self, team_id: str):
        """One team's complete round - all members take turns"""
        team = self.state.teams[team_id]
        self.state.current_team = team_id
        
        await self._log_game_event("TEAM_ROUND_START", 
                                  f"{team.name} starting round {self.state.round_number}",
                                  {"team": team_id, "round": self.state.round_number})
        
        # Determine turn order (can be random or predetermined)
        turn_order = team.members.copy()
        if self.config.get('turn_config', {}).get('random_order', False):
            random.shuffle(turn_order)
        
        # Each player takes their individual turn
        for player_id in turn_order:
            if self._check_victory_condition():
                break
            
            await self._player_turn(player_id, team_id)
    
    # Inside your BattleshipGame class:

    async def _player_turn(self, player_id: str, team_id: str):
        """
        One player's turn:
        1. Stamp turn start (for advice filtering)
        2. Log the turn start
        3. Fetch the player's profile
        4. Run consultation (assistant + team)
        5. Make the coordinate call and execute the attack
        """
        # 1️⃣ Mark when this turn begins
        self.turn_start_ts = asyncio.get_event_loop().time()

        # 2️⃣ Log the turn start
        self.state.current_player = player_id
        await self._log_game_event(
            "PLAYER_TURN_START",
            f"{player_id} starting turn",
            {"player": player_id, "team": team_id}
        )

        # 3️⃣ Fetch this player's profile from config
        player_config = self.config['agent_assignments'].get(player_id, {})
        profile = self.config['player_profiles'].get(player_config.get('profile', ''), {})

        # 4️⃣ Consultation phase (assistant + teammates), now correctly passing profile
        await self._player_consultation_phase(player_id, team_id, profile)

        # 5️⃣ Coordinate decision & attack
        coordinate = await self._player_coordinate_call(player_id, team_id)
        if coordinate:
            await self._execute_attack(player_id, team_id, coordinate)


    
    async def _player_consultation_phase(self, player_id: str, team_id: str, profile: Dict):
        """Player consultation with AI assistant and/or team"""
        assistant_reliance = profile.get('assistant_reliance', 'medium')
        decision_speed = profile.get('decision_speed', 'medium')
        
        logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} consultation phase - reliance: {assistant_reliance}")
        
        # Always try assistant first if available
        agent_assignment = self.config['agent_assignments'].get(player_id, {})
        has_assistant = agent_assignment.get('has_assistant', False)
        
        if has_assistant:
            logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} consulting AI assistant")
            await self._consult_ai_assistant(player_id)
        else:
            logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} has no AI assistant")
        
        # Then team consultation based on profile
        if assistant_reliance == 'high':
            # High reliance: minimal team consultation
            if decision_speed != 'fast':
                logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} brief team consultation")
                await self._team_consultation(player_id, team_id)
        elif assistant_reliance == 'low':
            # Low reliance: extensive team consultation
            logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} extensive team consultation")
            await self._team_consultation(player_id, team_id)
        else:
            # Medium reliance: balanced approach
            logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} standard team consultation")
            await self._team_consultation(player_id, team_id)

    async def _consult_ai_assistant(self, player_id: str):
        """Player consults with their AI assistant - CLEAN"""
        agent_assignment = self.config['agent_assignments'].get(player_id, {})
        assistant_id = agent_assignment.get('assistant_id')
        
        if not assistant_id or not agent_assignment.get('has_assistant', False):
            return
        
        # Get current game context
        game_context = self._build_game_context_for_player(player_id)
        
        # CLEAN assistant prompt - no fluff
        consultation_prompt = f"""Battle situation:
{game_context}

Suggest coordinate for attack.
Format: COORDINATE: [A1] - REASONING: [why]
Maximum 25 words."""
        
        try:
            # FIXED: Player asks assistant, assistant responds
            await self.network.send_message(
                sender_id=player_id,        # Player asks
                receiver_id=assistant_id,   # Assistant responds
                message=consultation_prompt,
                max_turns=1  # Single exchange
            )
            
            logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} consulted {assistant_id}")
                                          
        except Exception as e:
            logger.error(f"Failed AI consultation for {player_id}: {e}")
    
    async def _team_consultation(self, player_id: str, team_id: str):
        """Player consultation with team members"""
        team = self.state.teams[team_id]
        other_members = [m for m in team.members if m != player_id]
        
        if not other_members:
            return
        
        # Conduct team discussion with meaningful topics
        await self._team_discussion(team_id, player_id)
    
    async def _team_discussion(self, team_id: str, active_player: str):
        team = self.state.teams[team_id]
        other_members = [m for m in team.members if m != active_player]
        if not other_members:
            return

        discussion_questions = self._generate_battle_questions(team_id, active_player)

        for round_num in range(min(self.max_communication_rounds, len(discussion_questions))):
            question = discussion_questions[round_num]
            logger.info(f"[BATTLE] [R{self.state.round_number}] {active_player} asking: {question}")

            # → Use the new helper to enforce coordinate format
            for member in other_members:
                await self._send_battle_message(active_player, member, question)

            await asyncio.sleep(0.5)

    
    def _get_recent_team_coords(self, team_id: str) -> List[str]:
        """Get recent coordinates attempted by this team"""
        recent_team_coords = []
        
        team = self.state.teams.get(team_id)
        if team:
            for member in team.members:
                if member in self.coordinate_history:
                    # Get last 2 coordinates per team member
                    recent_team_coords.extend(self.coordinate_history[member][-2:])
        
        return recent_team_coords
    
    def _generate_battle_questions(self, team_id: str, active_player: str) -> List[str]:
        """Generate tactical questions without giving away coordinates"""
        recent_team_coords = self._get_recent_team_coords(team_id)
        
        if not recent_team_coords:
            # Early game questions
            questions = [
                "Should I target center areas or corners first?",  # No coordinates!
                "Go systematic or random hunting?"
            ]
        elif len(recent_team_coords) < 5:
            # Mid-game questions  
            questions = [
                "Continue current search pattern or switch zones?",
                "Focus on unexplored areas or follow up on hits?"  # No coordinates!
            ]
        else:
            # Late game questions
            questions = [
                "Any patterns you noticed in their ship placement?",
                "Should I target near previous hits or try new area?"
            ]
        
        return questions[:2]

    def _get_strategic_coordinates(self, team_id: str) -> List[str]:
        """Generate strategic coordinate suggestions based on game state"""
        # Dynamic coordinate suggestions that change based on what's been tried
        all_tried = []
        team = self.state.teams.get(team_id)
        if team:
            for member in team.members:
                if member in self.coordinate_history:
                    all_tried.extend(self.coordinate_history[member])
        
        # Different coordinate pools based on game progression
        if len(all_tried) < 3:
            # Early game - center and corners
            strategies = ["E5", "C3", "G7", "B8", "F4"]
        elif len(all_tried) < 8:
            # Mid game - systematic patterns
            strategies = ["A1", "J10", "D6", "H4", "B9"]  
        else:
            # Late game - remaining high-value targets
            strategies = ["F2", "C9", "I5", "A7", "H8"]
        
        # Filter out already tried coordinates
        available = [coord for coord in strategies if coord not in all_tried]
        
        # If all strategic coords tried, generate random ones
        if not available:
            import string
            available = []
            for i in range(5):
                row = random.choice(string.ascii_uppercase[:10])  # A-J
                col = random.randint(1, 10)
                coord = f"{row}{col}"
                if coord not in all_tried:
                    available.append(coord)
        
        return available[:3]  # Return top 3 suggestions

    def _count_recent_hits(self, player_id: str) -> int:
        """Count recent hits for context (placeholder implementation)"""
        return len(self.coordinate_history.get(player_id, [])) % 3
    
    async def _send_team_message(self, sender: str, recipients: List[str], message: str):
        """Send message to multiple recipients cleanly"""
        for recipient in recipients:
            if sender != recipient:
                await self._send_clean_message(sender, recipient, message)
    
    def _get_all_attempted_coordinates(self) -> List[str]:
        """Get all coordinates attempted by both teams"""
        all_attempted = set()
        for team_data in self.state.teams.values():
            for member in team_data.members:
                if member in self.coordinate_history:
                    all_attempted.update(self.coordinate_history[member])
        return sorted(list(all_attempted))

    def _simulate_coordinate_choice(self, player_id: str, team_id: str) -> str:
        """Extract coordinate based on player personality and conversations"""
        
        # Get all attempted coordinates to avoid duplicates
        all_attempted_coords = self._get_all_attempted_coordinates()
        
        # Get player personality traits
        player_config = self.config['agent_assignments'].get(player_id, {})
        profile = self.config['player_profiles'].get(player_config.get('profile', ''), {})
        
        assistant_reliance = profile.get('assistant_reliance', 'medium')
        decision_speed = profile.get('decision_speed', 'medium')
        leadership_style = profile.get('leadership_style', 'collaborative')
        
        logger.info(f"[COORD_CHOICE] {player_id} personality: reliance={assistant_reliance}, speed={decision_speed}")
        logger.info(f"[COORD_CHOICE] {player_id} avoiding {len(all_attempted_coords)} already attempted: {all_attempted_coords}")
        
        # Fast decision makers ignore advice and go with instinct
        if decision_speed == 'fast':
            logger.info(f"[COORD_CHOICE] {player_id} making fast instinctual decision")
            return self._generate_instinctual_coordinate(player_id, all_attempted_coords)
        
        # Try to extract coordinate from recent conversations based on personality
        if hasattr(self.network, 'conversation_history'):
            recent_conversations = [
                conv for conv in self.network.conversation_history[-8:]  # Last 8 conversations
                if conv.get('receiver') == player_id or conv.get('sender') == player_id
            ]
            
            # High AI reliance: Strongly prefer AI assistant advice
            if assistant_reliance == 'high':
                for conv in reversed(recent_conversations):
                    if conv.get('sender', '').startswith('assistant_'):
                        coordinate = self._extract_coordinate_from_message(conv)
                        if coordinate and coordinate not in all_attempted_coords:
                            logger.info(f"[COORD_CHOICE] {player_id} (high AI reliance) using AI suggestion: {coordinate}")
                            return coordinate
                
                # If no AI advice available, reluctantly consider team input
                logger.info(f"[COORD_CHOICE] {player_id} found no valid AI advice, reluctantly considering team")
                
            # Low AI reliance: Prefer team input, ignore AI
            elif assistant_reliance == 'low':
                # Look for team suggestions (ignore AI)
                team_suggestions = []
                for conv in reversed(recent_conversations):
                    if not conv.get('sender', '').startswith('assistant_'):
                        coordinate = self._extract_coordinate_from_message(conv)
                        if coordinate and coordinate not in all_attempted_coords:
                            team_suggestions.append(coordinate)
                
                if team_suggestions:
                    chosen = team_suggestions[0]  # Take first valid team suggestion
                    logger.info(f"[COORD_CHOICE] {player_id} (low AI reliance) using team suggestion: {chosen}")
                    return chosen
                
                logger.info(f"[COORD_CHOICE] {player_id} found no valid team suggestions, going with instinct")
                
            # Medium reliance: Consider both, with personality-based weighting
            else:
                ai_suggestions = []
                team_suggestions = []
                
                for conv in reversed(recent_conversations):
                    coordinate = self._extract_coordinate_from_message(conv)
                    if coordinate and coordinate not in all_attempted_coords:
                        if conv.get('sender', '').startswith('assistant_'):
                            ai_suggestions.append(coordinate)
                        else:
                            team_suggestions.append(coordinate)
                
                # Balanced approach - consider both
                if ai_suggestions and team_suggestions:
                    # For collaborative leaders, prefer team consensus
                    if leadership_style == 'collaborative' and len(team_suggestions) > 1:
                        chosen = team_suggestions[0]
                        logger.info(f"[COORD_CHOICE] {player_id} (collaborative) choosing team consensus: {chosen}")
                        return chosen
                    # For authoritative leaders, prefer AI analysis  
                    elif leadership_style == 'authoritative' and ai_suggestions:
                        chosen = ai_suggestions[0]
                        logger.info(f"[COORD_CHOICE] {player_id} (authoritative) choosing AI analysis: {chosen}")
                        return chosen
                    # Default: AI first for medium reliance
                    else:
                        chosen = ai_suggestions[0]
                        logger.info(f"[COORD_CHOICE] {player_id} (medium reliance) choosing AI suggestion: {chosen}")
                        return chosen
                
                elif ai_suggestions:
                    chosen = ai_suggestions[0]
                    logger.info(f"[COORD_CHOICE] {player_id} using available AI suggestion: {chosen}")
                    return chosen
                
                elif team_suggestions:
                    chosen = team_suggestions[0]
                    logger.info(f"[COORD_CHOICE] {player_id} using available team suggestion: {chosen}")
                    return chosen
        
        # Personality-based fallback if no suggestions found
        logger.info(f"[COORD_CHOICE] {player_id} no valid suggestions found, using personality-based fallback")
        return self._generate_personality_based_coordinate(player_id, profile, all_attempted_coords)
    
    def _generate_instinctual_coordinate(self, player_id: str, avoid_coords: List[str]) -> str:
        """Generate coordinate for fast/instinctual players"""
        # Fast players prefer corners and edges (quick, aggressive moves)
        aggressive_targets = ["A1", "A10", "J1", "J10", "A5", "J5", "E1", "E10"]
        
        for coord in aggressive_targets:
            if coord not in avoid_coords:
                return coord
        
        # Fallback to random
        return self._generate_random_coordinate(player_id, avoid_coords)
    
    def _generate_personality_based_coordinate(self, player_id: str, profile: Dict, avoid_coords: List[str]) -> str:
        """Generate coordinate based on player personality when no advice available"""
        risk_tolerance = profile.get('risk_tolerance', 'medium')
        strategy_focus = profile.get('strategy_focus', 'balanced')
        
        if risk_tolerance == 'high':
            # High risk: target center areas
            high_risk_coords = ["E5", "F5", "D5", "E6", "F6", "D6", "E4", "F4"]
        elif risk_tolerance == 'low':
            # Low risk: systematic edge approach  
            high_risk_coords = ["A1", "A2", "B1", "I1", "J1", "J2", "I10", "J10"]
        else:
            # Medium risk: balanced approach
            high_risk_coords = ["C3", "C7", "G3", "G7", "D4", "F4", "D6", "F6"]
        
        for coord in high_risk_coords:
            if coord not in avoid_coords:
                logger.info(f"[COORD_CHOICE] {player_id} using {risk_tolerance}-risk strategy: {coord}")
                return coord
        
        return self._generate_random_coordinate(player_id, avoid_coords)
    
    def _generate_random_coordinate(self, player_id: str, avoid_coords: List[str]) -> str:
        """Generate random coordinate avoiding already attempted coordinates"""
        max_attempts = 100  # Increased for safety
        for _ in range(max_attempts):
            row = random.randint(0, self.grid_size[0] - 1)
            col = random.randint(0, self.grid_size[1] - 1)
            coordinate = self.position_to_coordinate((row, col))
            
            if coordinate not in avoid_coords:
                logger.info(f"[COORD_CHOICE] {player_id} using random coordinate: {coordinate}")
                return coordinate
        
        # Emergency fallback - find ANY untried coordinate
        all_possible = []
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                coord = self.position_to_coordinate((row, col))
                if coord not in avoid_coords:
                    all_possible.append(coord)
        
        if all_possible:
            chosen = random.choice(all_possible)
            logger.warning(f"[COORD_CHOICE] {player_id} emergency fallback coordinate: {chosen}")
            return chosen
        
        # This should never happen in a 10x10 grid unless all 100 squares are tried
        logger.error(f"[COORD_CHOICE] {player_id} NO AVAILABLE COORDINATES! Using A1 as last resort")
        return "A1"
    
    def _extract_coordinate_from_message(self, conversation: Dict) -> Optional[str]:
        """
        Extract only from the agent’s reply (not the prompt),
        and skip any game_master messages.
        """
        import re

        chat_result = conversation.get('chat_result')
        if not chat_result or not hasattr(chat_result, 'chat_history'):
            return None

        for msg in chat_result.chat_history:
            speaker = msg.get('name') or msg.get('role', '')
            content = msg.get('content', '')

            # 1) Skip the game_master’s own prompt
            if speaker == "game_master":
                continue

            # 2) Look for an explicit COORDINATE: [X#] in the agent’s reply
            m = re.search(r'COORDINATE:\s*\[?([A-J](?:10|[1-9]))\]?', content)
            if m:
                return m.group(1)

        return None

    
    # def _parse_coordinate_from_text(self, text: str) -> Optional[str]:
    #     """Parse coordinate from text using regex"""
    #     import re
        
    #     # Look for COORDINATE: [A1] format first
    #     coordinate_pattern = r'COORDINATE:\s*\[?([A-J][1-9]|[A-J]10)\]?'
    #     match = re.search(coordinate_pattern, text)
    #     if match:
    #         return match.group(1)
        
    #     # Look for standalone coordinates like "E5", "D4", etc.
    #     standalone_pattern = r'\b([A-J][1-9]|[A-J]10)\b'
    #     matches = re.findall(standalone_pattern, text)
    #     if matches:
    #         # Return first valid coordinate found
    #         return matches[0]
        
    #     return None
    
    def _parse_coordinate_from_text(self, text: str) -> Optional[str]:
        import re
        
        # # Debug: Print what we're trying to parse
        # print(f"DEBUG PARSING: {text}")
        
        # Look for COORDINATE: [A1] format first
        coordinate_pattern = r'COORDINATE:\s*\[?([A-J][1-9]|[A-J]10)\]?'
        match = re.search(coordinate_pattern, text)
        if match:
            # print(f"DEBUG FOUND: {match.group(1)}")
            return match.group(1)
        
        return None
    
    def _consolidate_advice_for_player(self, player_id: str) -> str:
        """
        Pull only advice messages (assistant or teammate) that arrived
        after self.turn_start_ts for the given player_id.
        """
        tracker = getattr(self.network, 'conversation_tracker', None)
        if not tracker:
            return ""

        since = getattr(self, 'turn_start_ts', 0.0)
        advice_items = []
        seen = set()
        kws = {'target','focus','avoid','center','edge','corner','pattern','cluster','switch','continue'}

        # Look at all convs _involving_ this agent (we’ll filter by timestamp ourselves)
        for conv in tracker.get_agent_conversations(player_id, recent_only=False):
            # Skip convs that began before this turn
            if conv.get('timestamp', 0.0) < since:
                continue

            for msg in conv.get('messages', []):
                # Only messages _to_ this player AND only messages since turn start
                if msg['speaker'] == player_id or msg.get('timestamp', 0.0) < since:
                    continue

                giver = msg['speaker']
                content = msg['content']

                # 1) Coordinate suggestions
                coord = tracker._parse_coordinate_from_text(content)
                if coord:
                    key = (giver, 'coord', coord)
                    if key not in seen:
                        prefix = '🤖' if 'assistant' in giver else '👥'
                        advice_items.append(f"{prefix} {giver}: Suggests {coord}")
                        seen.add(key)
                    continue

                # 2) Strategic advice via keywords
                if len(content) > 20 and any(k in content.lower() for k in kws):
                    key = (giver, 'strat', content[:30])
                    if key not in seen:
                        summary = content[:60].rstrip() + ('…' if len(content) > 60 else '')
                        prefix = '🤖' if 'assistant' in giver else '👥'
                        advice_items.append(f"{prefix} {giver}: {summary}")
                        seen.add(key)

        return "\n".join(advice_items)

    
    async def _player_coordinate_call(self, player_id: str, team_id: str) -> Optional[str]:
        """
        Player makes final coordinate call, after gathering all assistant and teammate advice,
        then brainstorming and deciding according to their profile.
        """
        # 1. Load the player's profile to know how they prioritize inputs
        assignment = self.config['agent_assignments'].get(player_id, {})
        profile_name = assignment.get('profile', '')
        profile = self.config['player_profiles'].get(profile_name, {})

        # 2. Gather all advice this turn (assistant + teammates)
        advice_summary = self._consolidate_advice_for_player(player_id)

        # 3. Build the shared game context
        game_context = self._build_game_context_for_player(player_id)
        
        # 🔥 4. NEW: Get agent's accumulated battlefield memory
        memory_context = self._build_memory_context(player_id)

        # 5. Incorporate profile-driven instructions
        profile_desc = profile.get('description', '')
        reliance = profile.get('assistant_reliance', 'medium')

        # 6. Construct the decision prompt WITH memory context
        header = f"COORDINATE DECISION TIME (Profile: {profile_desc})"
        if advice_summary:
            decision_prompt = f"""{header}

    {game_context}

    {memory_context}

    ADVICE RECEIVED THIS TURN:
    {advice_summary}

    As a {profile_name.replace('_', ' ')}, you should {"prioritize AI suggestions" if reliance=="high" else "weigh team input" if reliance=="low" else "balance AI and team input"}.
    Based on your battlefield memory and advice above, choose your attack coordinate.
    CRITICAL: Avoid coordinates already attempted by anyone (check battlefield memory).
    Format: COORDINATE: [A1] - REASONING: [why]
    Maximum 50 words."""
        else:
            decision_prompt = f"""{header}

    {game_context}

    {memory_context}

    No advice received this turn.
    Based on your battlefield memory, choose your attack coordinate.
    CRITICAL: Avoid coordinates already attempted (check memory above).
    Format: COORDINATE: [A1] - REASONING: [why]
    Maximum 50 words."""

        try:
            conv = await self.network.send_message(
                sender_id="game_master",
                receiver_id=player_id,
                message=decision_prompt,
                max_turns=1
            )

            # Try to parse the agent's reply
            coordinate = self._extract_coordinate_from_message(conv)
            if coordinate:
                logger.info(f"[COORD_CALL] {player_id} chose via LLM: {coordinate}")
                return coordinate

            # Fallback if no valid coordinate found
            fallback = self._simulate_coordinate_choice(player_id, team_id)
            logger.info(f"[COORD_CALL] {player_id} fallback to personality logic: {fallback}")
            return fallback

        except Exception as e:
            logger.error(f"[COORD_CALL] Error for {player_id}: {e}")
            return self._simulate_coordinate_choice(player_id, team_id)

    
    async def _execute_attack(self, player_id: str, team_id: str, coordinate: str):
        """Execute the attack and report results"""
        try:
            position = self.coordinate_to_position(coordinate)
            
            # Find target team
            target_team_id = None
            for tid, team in self.state.teams.items():
                if tid != team_id:
                    target_team_id = tid
                    break
            
            if not target_team_id:
                logger.error("No target team found")
                return
            
            target_team = self.state.teams[target_team_id]
            
            # Execute attack
            result, sunk_ship = target_team.grid.attack(position)
            
            # Record in memory system
            all_agents = []
            for team in self.state.teams.values():
                all_agents.extend(team.members)

            self.memory_manager.record_coordinate_attack(
                coordinate=coordinate,
                attacker=player_id,
                target_team=target_team.name,
                result=result,
                sunk_ship=sunk_ship.name if sunk_ship else None,
                round_number=self.state.round_number,
                inform_agents=all_agents
            )
            
            # Record coordinate and result
            if player_id not in self.coordinate_history:
                self.coordinate_history[player_id] = []
            self.coordinate_history[player_id].append(coordinate)
            
            # Store attack result for later reference
            attack_record = {
                'player': player_id,
                'coordinate': coordinate,
                'result': result,
                'sunk_ship': sunk_ship.name if sunk_ship else None,
                'round': self.state.round_number
            }
            
            # Add to game log with attack result
            await self._log_game_event("ATTACK_RESULT", 
                                      f"{player_id} -> {coordinate}: {result}" + (f" ({sunk_ship.name})" if sunk_ship else ""),
                                      attack_record)
            
            # Clean attack result logging
            if result == "SUNK" and sunk_ship:
                logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} -> {coordinate}: {result} ({sunk_ship.name})")
            else:
                logger.info(f"[BATTLE] [R{self.state.round_number}] {player_id} -> {coordinate}: {result}")
            
            # Announce result using programmatic game master
            await self.game_master.announce_attack_result(player_id, coordinate, result, sunk_ship)
            
            turn_intel = (
                f"TURN INTEL: {player_id} attacked {coordinate} → {result}" + (f" (sunk {sunk_ship.name})" if sunk_ship else "")
            )

            # Determine which agents should get this memory (player + its assistant)
            agents_to_update = [player_id]
            assignment = self.config['agent_assignments'].get(player_id, {})
            assistant_id = assignment.get('assistant_id')
            if assignment.get('has_assistant') and assistant_id:
                agents_to_update.append(assistant_id)

            # Inject into their battlefield memory
            await self._inject_battlefield_intel(agents_to_update, turn_intel)
            
        except Exception as e:
            logger.error(f"Attack failed {coordinate} by {player_id}: {e}")
    
    def export_game_memories(self):
        """Export game memories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"battleship_memories_{timestamp}.json"
        
        memory_export = self.memory_manager.export_all_memories()
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / filename, 'w') as f:
            json.dump(memory_export, f, indent=2)
        
        print(f"Memories exported to: output/{filename}")
        return filename
    
    def _get_attack_result_for_coordinate(self, coordinate: str, player_id: str) -> str:
        """Get the attack result for a specific coordinate from game log"""
        # Look through recent game log for this coordinate's attack result
        for event in reversed(self.state.game_log):
            if event.get('event_type') == 'ATTACK_RESULT':
                metadata = event.get('metadata', {})
                if (metadata.get('player') == player_id and 
                    metadata.get('coordinate') == coordinate):
                    result = metadata.get('result', 'MISS')
                    sunk_ship = metadata.get('sunk_ship')
                    
                    if result == "SUNK" and sunk_ship:
                        return f'HIT & SUNK ({sunk_ship})'
                    elif result == "HIT":
                        return 'HIT'
                    elif result == "MISS":
                        return 'MISS'
                    else:
                        return result
        
        # Fallback
        return 'MISS'

    async def _share_round_results(self, team_id: str):
        """Share round results by directly updating agent and assistant memory"""
        team = self.state.teams[team_id]
        
        # Collect round results with hit/miss information
        round_results = []
        for member in team.members:
            if member in self.coordinate_history:
                recent_coords = self.coordinate_history[member][-1:] if self.coordinate_history[member] else []
                for coord in recent_coords:
                    attack_result = self._get_attack_result_for_coordinate(coord, member)
                    round_results.append(f"{coord}: {attack_result}")
        
        all_attempted_coords = self._get_all_attempted_coordinates()
        
        if round_results:
            # Create comprehensive battlefield intel summary
            intel_summary = f"""BATTLEFIELD INTEL UPDATE - Round {self.state.round_number}:
{team.name} attacks: {', '.join(round_results)}
Global battlefield status: {len(all_attempted_coords)} coordinates attempted by both teams
Remaining valid targets: {100 - len(all_attempted_coords)}
Strategic note: Use this intel for next round planning and coordinate selection."""
            
            # Get all agents to update: players + their assistants
            agents_to_update = []
            
            # Add all team members (players)
            agents_to_update.extend(team.members)
            
            # Add all assistants for this team's members
            for member in team.members:
                agent_assignment = self.config.get('agent_assignments', {}).get(member, {})
                assistant_id = agent_assignment.get('assistant_id')
                if assistant_id and agent_assignment.get('has_assistant', False):
                    agents_to_update.append(assistant_id)
                    logger.info(f"[INTEL] Including assistant {assistant_id} for player {member}")
            
            # Directly inject intel into all agents (no conversations)
            await self._inject_battlefield_intel(agents_to_update, intel_summary.strip())
            
            # Also share intel with opponent team for realistic battlefield awareness
            opponent_team_id = None
            for tid, t in self.state.teams.items():
                if tid != team_id:
                    opponent_team_id = tid
                    break
            
            if opponent_team_id:
                opponent_team = self.state.teams[opponent_team_id]
                opponent_agents = []
                
                # Add opponent team members
                opponent_agents.extend(opponent_team.members)
                
                # Add opponent assistants  
                for member in opponent_team.members:
                    agent_assignment = self.config.get('agent_assignments', {}).get(member, {})
                    assistant_id = agent_assignment.get('assistant_id')
                    if assistant_id and agent_assignment.get('has_assistant', False):
                        opponent_agents.append(assistant_id)
                
                opponent_intel = f"""ENEMY ACTIVITY INTEL - Round {self.state.round_number}:
Enemy team ({team.name}) attempted: {', '.join([r.split(':')[0] for r in round_results])}
Global battlefield status: {len(all_attempted_coords)} coordinates attempted by both teams
Strategic note: Use this intel to avoid already-attempted coordinates."""
                
                await self._inject_battlefield_intel(opponent_agents, opponent_intel.strip())
            
            total_updated = len(agents_to_update) + (len(opponent_agents) if opponent_team_id else 0)
            logger.info(f"[INTEL] Updated battlefield memory for {total_updated} total agents (players + assistants)")

    async def _inject_battlefield_intel(self, agent_ids: List[str], intel_summary: str):
        """Safely inject battlefield intel into agent memory without corrupting AutoGen internals"""
        successful_injections = 0
        
        for agent_id in agent_ids:
            if agent_id in self.network.agents:
                try:
                    agent = self.network.agents[agent_id]
                    
                    # Initialize battlefield memory as a regular list if it doesn't exist
                    if not hasattr(agent, 'battlefield_memory'):
                        agent.battlefield_memory = []
                    elif not isinstance(agent.battlefield_memory, list):
                        # Convert any other type to list for safety
                        agent.battlefield_memory = []
                    
                    # Add new intel to agent's memory
                    intel_entry = {
                        'type': 'battlefield_intel',
                        'content': intel_summary,
                        'round': self.state.round_number,
                        'timestamp': datetime.now().isoformat(),
                        'agent_id': agent_id
                    }
                    
                    agent.battlefield_memory.append(intel_entry)
                    
                    # Keep only last 10 intel updates to prevent memory bloat
                    agent.battlefield_memory = agent.battlefield_memory[-10:]
                    
                    # Store latest intel in easily accessible attribute
                    agent.latest_battlefield_intel = intel_summary
                    
                    # Store intel summary for quick access during coordinate selection
                    agent.current_round_intel = intel_summary
                    
                    successful_injections += 1
                    logger.debug(f"[INTEL] Successfully updated memory for {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to inject intel into {agent_id}: {e}")
                    # Try a simpler approach as fallback
                    try:
                        agent = self.network.agents[agent_id]
                        agent.latest_battlefield_intel = intel_summary
                        successful_injections += 1
                        logger.info(f"[INTEL] Used fallback intel injection for {agent_id}")
                    except Exception as fallback_error:
                        logger.error(f"Even fallback injection failed for {agent_id}: {fallback_error}")
            else:
                logger.warning(f"Agent {agent_id} not found in network")
        
        logger.info(f"[INTEL] Successfully injected intel into {successful_injections}/{len(agent_ids)} agents")
    
    def _build_game_context_for_player(self, player_id: str) -> str:
        """Build current game context for a player"""
        team_id = None
        for tid, team in self.state.teams.items():
            if player_id in team.members:
                team_id = tid
                break
        
        if not team_id:
            return "Game context unavailable"
        
        # Get player's team info
        team = self.state.teams[team_id]
        
        # Get opponent team info
        opponent_team = None
        for tid, t in self.state.teams.items():
            if tid != team_id:
                opponent_team = t
                break
        
        context = f"""Round: {self.state.round_number}
Your Team: {team.name}
Opponent: {opponent_team.name if opponent_team else 'Unknown'}

Your Previous Coordinates: {', '.join(self.coordinate_history.get(player_id, []))}

Grid: {self.grid_size[0]}x{self.grid_size[1]} (A1 to {chr(ord('A') + self.grid_size[0] - 1)}{self.grid_size[1]})"""
        
        return context
    
    def _build_memory_context(self, player_id: str) -> str:
        """Use the intel already injected into agent memory"""
        if player_id in self.network.agents:
            agent = self.network.agents[player_id]
            
            # Use the latest battlefield intel you're already storing
            if hasattr(agent, 'latest_battlefield_intel'):
                return f"BATTLEFIELD MEMORY:\n{agent.latest_battlefield_intel}"
            
            # Fallback to battlefield memory list
            if hasattr(agent, 'battlefield_memory') and agent.battlefield_memory:
                latest_intel = agent.battlefield_memory[-1]['content']
                return f"BATTLEFIELD MEMORY:\n{latest_intel}"
        
        return "BATTLEFIELD MEMORY: No intel available"
    
    def _check_victory_condition(self) -> bool:
        """Check if any team has won"""
        for team_id, team in self.state.teams.items():
            if team.grid.all_ships_sunk():
                # This team lost, find the winner
                for winner_id, winner_team in self.state.teams.items():
                    if winner_id != team_id:
                        self.state.winner = winner_id
                        return True
        return False
    
    async def _game_over_phase(self):
        """Handle game over and victory announcement"""
        self.state.phase = GamePhase.GAME_OVER
        
        # Use programmatic game master for final announcement
        await self.game_master.announce_game_over(self.state.winner)
        
        if self.state.winner:
            await self._log_game_event("GAME_OVER", f"{self.state.teams[self.state.winner].name} wins!",
                                      {"winner": self.state.winner, "rounds": self.state.round_number})
        else:
            await self._log_game_event("GAME_OVER", "Game ended without clear winner")
    
    async def _log_game_event(self, event_type: str, message: str, 
                         metadata: Optional[Dict] = None):
        """Log game events with clean output"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
            "game_state": {
                "phase": self.state.phase.value,
                "round": self.state.round_number,
                "current_team": self.state.current_team,
                "current_player": self.state.current_player
            }
        }
        
        self.state.game_log.append(event)
        
        # Clean console output with phase and round info
        if event_type in ["PHASE_START", "TEAM_ROUND_START", "PLAYER_TURN_START", "ATTACK_RESULT", "GAME_OVER"]:
            phase_indicator = f"[{self.state.phase.value.upper()}]"
            round_indicator = f"[R{self.state.round_number}]" if self.state.round_number > 0 else ""
            logger.info(f"{phase_indicator} {round_indicator} {message}")
    
    def save_game_log(self, filename: Optional[str] = None):
        """Save complete game log to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"battleship_game_{timestamp}.json"
        
        game_summary = {
            "game_config": self.config,
            "final_state": {
                "phase": self.state.phase.value,
                "winner": self.state.winner,
                "total_rounds": self.state.round_number,
                "teams": {
                    team_id: {
                        "name": team.name,
                        "members": team.members,
                        "ships_remaining": len([s for s in team.grid.ships if not s.is_sunk])
                    }
                    for team_id, team in self.state.teams.items()
                }
            },
            "coordinate_history": self.coordinate_history,
            "game_log": self.state.game_log
        }
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / filename, 'w') as f:
            json.dump(game_summary, f, indent=2)
        
        logger.info(f"Game log saved to {output_dir / filename}")
        return output_dir / filename
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get comprehensive game statistics"""
        stats = {
            "game_duration_rounds": self.state.round_number,
            "winner": self.state.winner,
            "teams": {},
            "player_performance": {},
            "communication_metrics": {}
        }
        
        # Team statistics
        for team_id, team in self.state.teams.items():
            team_stats = {
                "name": team.name,
                "member_count": len(team.members),
                "ships_sunk": len([s for s in team.grid.ships if s.is_sunk]),
                "ships_total": len(team.grid.ships),
                "survival_rate": len([s for s in team.grid.ships if not s.is_sunk]) / len(team.grid.ships) if team.grid.ships else 0
            }
            stats["teams"][team_id] = team_stats
        
        # Player performance
        for player_id, coordinates in self.coordinate_history.items():
            stats["player_performance"][player_id] = {
                "total_attacks": len(coordinates),
                "coordinates_called": coordinates
            }
        
        # Communication metrics
        communication_events = [event for event in self.state.game_log 
                               if event["event_type"] in ["AI_CONSULTATION", "TEAM_DISCUSSION"]]
        stats["communication_metrics"] = {
            "total_communications": len(communication_events),
            "ai_consultations": len([e for e in communication_events if e["event_type"] == "AI_CONSULTATION"]),
            "team_discussions": len([e for e in communication_events if e["event_type"] == "TEAM_DISCUSSION"])
        }
        
        return stats


# Integration with existing agent network
async def run_battleship_simulation(agent_network, battleship_config_path: str):
    """
    Main function to run battleship simulation with agent network
    
    Args:
        agent_network: Initialized AgentNetwork instance
        battleship_config_path: Path to battleship_config.json
    """
    
    # Initialize battleship game
    game = BattleshipGame(battleship_config_path, agent_network)
    
    try:
        # Start the game
        await game.start_game()
        
        # Get final statistics
        stats = game.get_game_statistics()
        
        # Save game log
        log_file = game.save_game_log()
        
        # Print summary
        print("\n🎮 BATTLESHIP GAME COMPLETE!")
        print("=" * 50)
        
        if game.state.winner:
            winner_team = game.state.teams[game.state.winner]
            print(f"🏆 WINNER: {winner_team.name}")
        else:
            print("🤝 GAME ENDED WITHOUT CLEAR WINNER")
        
        print(f"📊 Total Rounds: {game.state.round_number}")
        print(f"📝 Game Log: {log_file}")
        
        # Team summary
        print("\n📈 TEAM PERFORMANCE:")
        for team_id, team_stats in stats["teams"].items():
            team = game.state.teams[team_id]
            print(f"  {team_stats['name']}: {team_stats['ships_sunk']}/{team_stats['ships_total']} ships lost")
        
        # Player summary  
        print("\n🎯 PLAYER ACTIVITY:")
        for player_id, player_stats in stats["player_performance"].items():
            print(f"  {player_id}: {player_stats['total_attacks']} attacks")
        
        print(f"\n💬 Communication: {stats['communication_metrics']['total_communications']} total interactions")
        print(f"   🤖 AI consultations: {stats['communication_metrics']['ai_consultations']}")
        print(f"   👥 Team discussions: {stats['communication_metrics']['team_discussions']}")
        
        return game, stats
        
    except Exception as e:
        logger.error(f"Battleship simulation failed: {e}")
        raise