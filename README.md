# 🚀 LLM Agent Network Framework

A framework for creating networks of AI agents using AG2 (AutoGen 2) with **persistent memory** and **adaptive learning**. Main showcase: **Adversarial Battleship** with team strategy, AI assistants, and personality-driven gameplay.

## 🎯 What it does

Create networks where AI agents communicate with **memory-driven decision making**. Agents learn from battlefield history, avoid duplicate moves, and adapt strategies based on accumulated intelligence.

**Key Features:**
- Mix local (Ollama) and cloud (OpenAI/Claude/Gemini) models
- **Persistent memory system** - agents remember past attacks, player patterns, and strategic insights  
- Role-based agent definitions (leader, player, assistant, game_master)
- **Battleship tournament simulation** with team vs team gameplay
- **Personality-driven behavior** - agents make decisions based on assistant reliance, risk tolerance, and leadership style
- Comprehensive logging and analytics

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

## 🧠 Memory System

**Global Memory Manager** tracks:
- **Battlefield Memory**: All coordinate attacks, hits/misses, sunk ships
- **Player Behavior**: Observed patterns (edge vs center preference, systematic vs random)
- **Strategic Insights**: Team tactics, enemy patterns, successful strategies
- **Conversation History**: AI advice quality, team discussions

**Memory-Informed Decisions**: Agents see accumulated battlefield intel when making coordinate calls, preventing duplicate attacks and enabling adaptive strategy.

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install ag2[ollama,openai]
```

### 2. Setup Ollama (for local models)
```bash
export OLLAMA_HOST=0.0.0.0:11435
ollama serve &
ollama pull llama3.1:8b
```

### 3. Configure API Keys
Add your API keys to `LLM_config.json` in the `global_config.api_keys` section.

## 🚀 Quick Start

### ⚔️ Run Battleship Game (Main Feature)
```bash
# Create battleship network
python battleship_runner.py --create-sample

# Run tournament with memory-driven agents
python battleship_runner.py

# Advanced game with more communication
python battleship_runner.py --communication-rounds 5 --verbose
```

### Test Basic Connectivity
```bash
python3 main.py --rounds 1 --max-turns 2 --edges networks/simple.txt
```

## 🎮 Battleship Features

### Memory-Driven Gameplay
- **No Duplicate Attacks**: Agents remember all attempted coordinates
- **Pattern Recognition**: Learn enemy attack patterns (edge vs center, systematic vs random)
- **Strategic Learning**: Build insights about ship placement and team tactics
- **Cross-Team Intelligence**: Teams get intel about enemy activity each round

### Personality Profiles
- **`assistant_reliance`**: `high` (follows AI advice), `medium` (balanced), `low` (prefers team input)
- **`decision_speed`**: `fast` (instinctual), `medium` (considers input), `slow` (careful analysis)
- **`risk_tolerance`**: `high` (center attacks), `medium` (balanced), `low` (systematic edges)
- **`leadership_style`**: `collaborative` (team consensus), `authoritative` (AI analysis preferred)

### Game Flow
1. **Ship Placement**: Team leaders coordinate with member input
2. **Battle Rounds**: Teams alternate, players take individual turns
3. **Memory Updates**: After each attack, all agents receive battlefield intel
4. **Adaptive Strategy**: Agents use accumulated memory to avoid duplicates and recognize patterns

## 📊 Memory Export & Analytics

```bash
# Export all agent memories after game
game.export_game_memories()  # → battleship_memories_TIMESTAMP.json

# View battlefield intelligence
agent_memory = game.memory_manager.get_agent_memory("player_a1")
print(agent_memory.generate_battlefield_summary())
```

## 🎯 Example Memory-Informed Decision

```
COORDINATE DECISION TIME (Profile: analytical player)

BATTLEFIELD MEMORY:
Enemy team (Bravo Fleet) attempted: F7, F4, D5
Global battlefield status: 13 coordinates attempted by both teams
Strategic note: Use this intel to avoid already-attempted coordinates.

ADVICE RECEIVED THIS TURN:
👥 leader_alpha: Suggests D5
🤖 assistant_a1: Suggests H9

CRITICAL: Avoid coordinates already attempted (check battlefield memory).
```

**Agent Response**: *"H9 has not been previously attempted by either side per memory. D5 already hit by both teams, so avoiding..."*

## ⚙️ Configuration

**LLM Config** (`LLM_config.json`): Agent definitions and API keys
**Battleship Config** (`battleship_config.json`): Teams, personalities, game settings  
**Network Topology** (`networks/battleship.txt`): Communication patterns

## 🚀 Advanced Features

### Multi-Model Support
- **GPT-4** for strategic leaders
- **Claude** for analytical players  
- **Local Llama** for assistants
- **Gemini** for specific roles

### Memory Analytics
```python
# Analyze player behavior patterns
behavior_analysis = memory_manager.analyze_player_behavior(
    player_id="player_a1", 
    recent_coordinates=["A1", "B2", "C3"]
)
# → {"coordinate_preference": "Prefers edge attacks", "attack_strategy": "Systematic approach"}
```

---

**🎯 Ready for AI battleship tournaments with persistent memory and adaptive learning!** 

*Agents that learn, remember, and evolve their strategies over time.*