# 🚀 LLM Agent Network Framework

Multi-agent AI battleship with **persistent memory** and **authentic decision-making**. Teams of AI agents strategize, communicate, and adapt using accumulated battlefield intelligence.

## 🎯 What it does

AI agents play adversarial battleship with **100% authentic decision-making** - no predefined moves or scripted strategies. Agents remember past attacks, learn enemy patterns, and make strategic decisions through genuine AI reasoning.

**Key Features:**
- **Authentic AI decisions** - all coordinates chosen by AI reasoning, not fallbacks
- **Persistent memory** - agents remember attacks, patterns, and strategic insights  
- **Personality-driven behavior** - decision-making based on risk tolerance, team reliance, leadership style
- **Multi-model support** - mix local (Ollama) and cloud (OpenAI/Claude/Gemini) models
- **Team strategy** - leaders coordinate, players consult AI assistants and teammates
- **Real-time adaptation** - strategies evolve based on battlefield intelligence

## 📁 Project Structure

```
LLM-Agent-Network/
├── agent_network.py          # Core framework
├── battleship_game.py        # Battleship game engine with memory
├── battleship_runner.py      # Main battleship runner
├── memory_manager.py         # Persistent memory system for agents
├── battleship_config.json   # Game settings & teams
├── LLM_config.json          # Agent configurations
├── networks/
│   ├── battleship.txt       # Battleship network topology
│   ├── full_mesh.txt        # Full connectivity testing
│   └── simple.txt           # Basic 2-agent testing
├── output/                  # Generated logs & results
└── README.md
```

## 🎮 Game Overview

**Standard battleship** with AI teams that think, strategize, and adapt:
- **10x10 grid** (A1-J10), **5 ships per team** (Carrier to Patrol Boat)
- **Teams of 2-4 agents**: Leaders coordinate, players make attack decisions  
- **AI assistants** provide strategic analysis and coordinate suggestions
- **Win condition**: First team to sink all enemy ships

## 🧠 How Agents Think

**Memory-Driven Decisions**: Every agent maintains battlefield intelligence - past attacks, enemy patterns, team strategies. No duplicates, pure strategic reasoning.

**Authentic AI Process**:
```
1. Consultation → AI assistant + team discussion
2. Memory Context → Past attacks, enemy patterns, strategic insights  
3. Personality Analysis → Risk tolerance, team reliance, decision speed
4. AI Reasoning → Genuine strategic coordinate selection
5. Execution → Attack, update memories, share intelligence
```

## 🎯 Game Flow & Phases

### **Phase 1: Ship Placement** 
```
1. Team Leader Discussion (configurable rounds)
   - Leader asks: "Where should we place our Carrier (5 spaces)? Edge or center?"
   - Team members provide input on ship placement strategy
   - Leader makes final decisions based on team consensus

2. Automatic Ship Placement
   - Ships are randomly placed for simulation efficiency
   - Real implementation could use AI-driven placement
```

### **Phase 2: Battle Rounds**
```
Round Structure:
├── Team A Turn
│   ├── Player A1 Individual Turn
│   │   ├── 1. Consultation Phase
│   │   │   ├── AI Assistant Consultation (if available)
│   │   │   └── Team Discussion (based on personality)
│   │   ├── 2. Advice Consolidation
│   │   │   └── Gather all AI and team suggestions
│   │   ├── 3. Coordinate Decision (Authentic AI)
│   │   │   ├── Memory Context: Previous attacks, battlefield intel
│   │   │   ├── Personality Profile: Risk tolerance, assistant reliance
│   │   │   └── Strategic Reasoning: AI chooses coordinate
│   │   └── 4. Attack Execution & Memory Update
│   ├── Player A2 Individual Turn
│   └── [... all team members take turns]
├── Intel Sharing Phase
│   ├── Battlefield intel updated for all agents
│   └── Round results shared with teams
└── Team B Turn (same structure)
```

### **Phase 3: Game Over**
```
- Victory announcement when all ships sunk
- Statistics generation and memory export
- Game log saved for analysis
```

## 💭 Communication Tactics & Strategies

### **AI Assistant Consultation**
```python
# Example AI Assistant Prompt
"""Battle situation:
Round: 3
Your Team: Alpha Fleet
Previous Coordinates: A1, B2, C3

Suggest coordinate for attack.
Format: COORDINATE: [A1] - REASONING: [why]
Maximum 25 words."""
```

### **Team Discussion Questions** (Dynamic Based on Game State)
**Early Game (< 3 attempts):**
- "Should I target center areas or edges first?"
- "Go systematic or random hunting?"

**Mid Game (3-8 attempts):**
- "Continue current search pattern or switch zones?"
- "Focus on unexplored areas or follow up on hits?"

**Late Game (8+ attempts):**
- "Any patterns you noticed in their ship placement?"
- "Should I target near previous hits or try new area?"

### **Communication Flow**
```
Player Turn Communication:
1. AI Assistant Consultation (if has_assistant = true)
   └── Strategic analysis and coordinate suggestion

2. Team Discussion (based on assistant_reliance level)
   ├── HIGH reliance: Minimal team consultation
   ├── MEDIUM reliance: Balanced AI + team input
   └── LOW reliance: Extensive team discussion

3. Advice Consolidation
   └── AI weighs all suggestions based on personality profile
```

## 🎭 Personality Profiles & Decision Making

### **Personality Dimensions**
- **`assistant_reliance`**: `high` (follows AI advice), `medium` (balanced), `low` (prefers team input)
- **`decision_speed`**: `fast` (instinctual), `medium` (considers input), `slow` (careful analysis)
- **`risk_tolerance`**: `high` (center attacks), `medium` (balanced), `low` (systematic edges)
- **`leadership_style`**: `collaborative` (team consensus), `authoritative` (AI analysis preferred)

### **Authentic Decision Logic**
```python
# AI Decision Prompt (Profile-Driven)
COORDINATE SELECTION - analytical player

FORBIDDEN: A1, B2, C3, D4  # Already attempted coordinates

BATTLEFIELD MEMORY:
Round 3 intel: Enemy attacks at F7, G3
Global status: 8 coordinates attempted
Remaining targets: 92

ADVICE RECEIVED:
🤖 assistant_a1: Suggests H9 - Strategic center targeting
👥 teammate_1: Focus on unexplored quadrants

PROFILE: analytical_player
- Assistant reliance: medium (balance all inputs)
- Risk tolerance: high
- Decision speed: medium

Choose your attack coordinate. Avoid forbidden coordinates.
REQUIRED FORMAT: COORDINATE: [X#]
```

## 🔄 Memory-Driven Adaptation

### **Battlefield Intel Updates** (After Each Attack)
```json
{
  "type": "battlefield_intel", 
  "content": "TURN INTEL: player_a1 attacked E5 → HIT (sunk Destroyer)",
  "round": 3,
  "agents_updated": ["player_a1", "assistant_a1", "all_opponents"]
}
```

### **Cross-Team Intelligence Sharing**
- **Own Team**: Full attack results and strategic insights
- **Enemy Team**: Limited intel about opponent activity
- **Assistants**: Updated with their player's memory context

### **Memory Context in Decisions**
```
BATTLEFIELD MEMORY:
Enemy team (Bravo Fleet) attempted: F7, F4, D5
Global battlefield status: 13 coordinates attempted by both teams
Strategic note: Use this intel to avoid already-attempted coordinates.
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install ag2[ollama,openai]
```

### 2. Configure API Keys
Create or edit `LLM_config.json` and add your API keys:
```json
{
  "global_config": {
    "api_keys": {
      "openai": "your-openai-key",
      "anthropic": "your-claude-key"
    }
  }
}
```

### 3. Setup Local Models (Optional)
For local Llama models via Ollama:
```bash
# Install and start Ollama
export OLLAMA_HOST=0.0.0.0:11435
ollama serve &
ollama pull llama3.1:8b
```

### 4. Run Battleship Game
```bash
# Create sample configuration and run game
python battleship_runner.py --create-sample
python battleship_runner.py

# Advanced options
python battleship_runner.py --communication-rounds 5 --verbose
```

### 5. Test Basic Network (Optional)
```bash
python3 main.py --rounds 1 --max-turns 2 --edges networks/simple.txt
```

## 📊 Memory Export & Analytics

```bash
# Export all agent memories after game
game.export_game_memories()  # → battleship_memories_TIMESTAMP.json

# View battlefield intelligence
agent_memory = game.memory_manager.get_agent_memory("player_a1")
print(agent_memory.generate_battlefield_summary())
```

## 🎯 Example Authentic AI Decision

```
COORDINATE DECISION TIME (Profile: analytical player)

BATTLEFIELD MEMORY:
Enemy team (Bravo Fleet) attempted: F7, F4, D5
Global battlefield status: 13 coordinates attempted by both teams
Strategic note: Use this intel to avoid already-attempted coordinates.

ADVICE RECEIVED THIS TURN:
👥 leader_alpha: Suggests center targeting strategy
🤖 assistant_a1: Suggests H9 - HIGH PROBABILITY ZONE

PROFILE ANALYSIS:
- Medium AI reliance: Balancing AI suggestion with team strategy
- High risk tolerance: Center attacks preferred
- Medium decision speed: Considering all inputs

AI REASONING: "H9 aligns with assistant analysis and hasn't been attempted. 
Center positioning matches my risk profile. Avoiding all forbidden coordinates."

FINAL DECISION: COORDINATE: [H9]
```

## ⚙️ Configuration

**LLM Config** (`LLM_config.json`): Agent definitions and API keys
**Battleship Config** (`battleship_config.json`): Teams, personalities, game settings  
**Network Topology** (`networks/battleship.txt`): Communication patterns

### Sample Team Configuration
```json
{
  "team_alpha": {
    "name": "Alpha Fleet",
    "members": ["player_a1", "player_a2"],
    "leader": "player_a1",
    "color": "blue"
  }
}
```

### Sample Personality Profile
```json
{
  "analytical_player": {
    "description": "analytical player",
    "assistant_reliance": "medium",
    "decision_speed": "medium", 
    "risk_tolerance": "high",
    "leadership_style": "authoritative"
  }
}
```

## 🚀 Advanced Features

### Multi-Model Support
- **GPT-4** for strategic leaders
- **Claude** for analytical players  
- **Local Llama** for assistants
- **Gemini** for specific roles

### Authentic AI Decision Pipeline
```
Consultation → Advice Consolidation → Memory Context → Personality Analysis → AI Reasoning → Coordinate Selection
```

### Error Handling & Reliability
- **Retry Logic**: If AI gives invalid coordinate, single retry with clearer prompts
- **Emergency Fallback**: Random selection from valid coordinates if AI fails
- **Memory Validation**: Prevents duplicate attacks through battlefield intel

### Memory Analytics
```python
# Analyze player behavior patterns
behavior_analysis = memory_manager.analyze_player_behavior(
    player_id="player_a1", 
    recent_coordinates=["A1", "B2", "C3"]
)
# → {"coordinate_preference": "Prefers edge attacks", "attack_strategy": "Systematic approach"}
```

## 🎮 Game Statistics & Logging

### Exported Game Data
- **Complete coordinate history** for each player
- **Communication logs** between all agents
- **Memory snapshots** at each decision point
- **Performance metrics** by team and individual
- **Personality-driven decision analysis**

### Sample Game Output
```
🎮 BATTLESHIP GAME COMPLETE!
🏆 WINNER: Alpha Fleet
📊 Total Rounds: 8
📈 TEAM PERFORMANCE:
  Alpha Fleet: 2/5 ships lost
  Bravo Fleet: 5/5 ships lost  
🎯 PLAYER ACTIVITY:
  player_a1: 4 attacks (3 hits, 1 sunk)
  player_a2: 4 attacks (2 hits)
💬 Communication: 24 total interactions
   🤖 AI consultations: 8
   👥 Team discussions: 16
```

---

**🎯 Ready for authentic AI battleship tournaments with persistent memory and adaptive learning!** 

*Agents that think, strategize, remember, and evolve their tactics through genuine AI reasoning.*