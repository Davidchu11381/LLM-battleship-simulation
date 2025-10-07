# 🚀 LLM Agent Network Framework — Dynamic Battleship

Multi-agent Battleship experiment comparing coordinated team deliberation vs. individual decision-making with simultaneous execution to eliminate turn order bias.

## 🎯 Experiment Objective

**Hypothesis:** Team Alpha (coordinated deliberation) outperforms Team Beta (individual decisions) when both teams execute actions simultaneously.

**Controls:** Same board, ships, LLM configs, and random seeds. Only coordination method differs.

## 📦 Project Structure

```
LLM-Agent-Network/
├── agent_network.py          # Agent messaging and communication
├── battleship_game.py        # Game engine and rules
├── battleship_runner.py      # Main launcher
├── memory_manager.py         # Team vs. individual memory management
├── battleship_config.json    # Game settings
├── LLM_config.json           # Agent configurations
├── networks/battleship.txt   # Team communication topology
└── output/                   # Results and logs
```

## 🕹️ Game Rules

- **Board:** 10×10 grid (A1–J10)
- **Teams:** Alpha Squadron vs. Beta Fleet (3 agents each)
- **Ships:** Each agent controls one ship (Battleship-4, Cruiser-3, or Destroyer-2)
- **Actions:** Each turn choose `BOMB <coord>` or `MOVE <direction>`
- **Execution:** Simultaneous execution after both teams complete decision phase
- **Victory:** First team to sink all enemy ships wins

## 🧠 Team Differences

**Alpha Squadron (Coordinated Deliberation):**
- 3-step team deliberation process before action execution
- Full team strategy proposals with reasoning
- Democratic voting on complete team plans
- Shared memory of all battlefield events

**Beta Fleet (Individual Decisions):**
- No team communication or deliberation
- Individual memory and decision making
- Same information access but no coordination

## 🗣️ Game Flow (Eliminating Turn Order Bias)

### Phase 1: Team Alpha Deliberation (3 Steps)

While Team Beta waits on standby:

#### Step 1: Team Strategy Proposals (Broadcast)

- All 3 Alpha agents simultaneously propose complete team strategies
- Each proposal includes specific actions for ALL team members
- Format: `PROPOSAL: player_a1 BOMB E5, player_a2 MOVE LEFT, player_a3 BOMB F6 - [50 word reasoning]`
- Reasoning should consider: game rules, strategy, individual ship safety, team coordination, maximizing firepower

#### Step 2: Democratic Voting

- Each Alpha agent votes for ONE complete team plan (can vote for their own)
- Agents consider both personal ship safety AND team victory
- Format: `VOTE: [player_name]'s plan - [brief reasoning]`

#### Step 3: Plan Selection

- Majority vote wins (2/3 or 3/3)
- Tie-breaking: Random selection between tied plans
- Selected plan becomes Alpha's actions for simultaneous execution

### Phase 2: Team Beta Individual Decisions

- Each Beta agent makes independent decisions (no communication)
- Same battlefield information as Alpha but no coordination
- Individual memory only

### Phase 3: Simultaneous Execution

- Both teams execute their chosen actions at exactly the same time
- No turn order bias - all BOMB and MOVE actions processed simultaneously
- Results applied and new round begins

## ⚙️ Implementation Specifications

### Team Alpha Communication Architecture

- **Step 1:** GroupChat broadcast - all agents propose simultaneously
- **Step 2:** Sequential voting - each agent votes for one plan
- **Step 3:** System determines majority winner, random tie-breaking
- **Timing:** Beta team waits until Alpha completes all 3 steps

### Team Beta Decision Architecture

- **Individual prompts:** Each agent receives context and chooses action independently
- **No communication:** Pure individual decision-making
- **Parallel processing:** All Beta decisions can be made simultaneously

### Simultaneous Execution System

- **Action Collection:** Gather final actions from both teams
- **Conflict Resolution:** Handle duplicate bombing coordinates (both teams can bomb same spot)
- **Synchronous Processing:** Apply all BOMB and MOVE actions at once
- **Result Broadcasting:** Inform all agents of round results

### Agent Incentive Structure

- Team Victory: +100 points per agent
- Individual Ship Survival: +20 points
- Team Coordination Bonus (Alpha only): +5 points for effective proposals/votes
- Natural consequences for poor coordination (wasted attacks, missed opportunities)

### Memory Management

- **Alpha Team:** Shared memory of all deliberations, proposals, votes, and battlefield events
- **Beta Team:** Individual memory only - each agent tracks their own observations
- **Global Events:** All BOMB results, MOVE results, eliminations recorded for analysis
- **Battlefield Intel:** Explicit lists of attacked coordinates provided to prevent wasteful targeting

## 🛠️ Detailed Implementation Requirements

### Core Game Engine Updates

#### Remove Turn-Based Structure

- Replace alternating team turns with simultaneous execution
- Implement two-phase structure: Decision → Execution

#### Alpha Deliberation Phase

```python
async def alpha_deliberation_phase(team_members):
    # Step 1: Collect team strategy proposals
    proposals = await collect_team_proposals(team_members)
    
    # Step 2: Conduct voting
    votes = await conduct_voting(team_members, proposals)
    
    # Step 3: Determine winning plan
    winning_plan = determine_majority_winner(votes)
    
    return parse_actions_from_plan(winning_plan)
```

#### Beta Individual Phase

```python
async def beta_individual_phase(team_members):
    # Parallel individual decisions
    actions = await gather_individual_decisions(team_members)
    return actions
```

#### Simultaneous Execution

```python
async def execute_simultaneous_actions(alpha_actions, beta_actions):
    # Process all BOMB actions from both teams
    # Process all MOVE actions from both teams
    # Apply results and update game state
    # Broadcast results to all agents
```

### Communication Protocol Implementation

#### Alpha Step 1 - Team Proposals

- Prompt: "Propose a complete strategy for your entire team. Include specific BOMB/MOVE actions for each teammate and 50-word reasoning considering game rules, strategy, ship safety, and team coordination."
- Collection: Gather all 3 proposals simultaneously
- Validation: Ensure each proposal specifies actions for all team members

#### Alpha Step 2 - Voting

- Present all proposals to each agent
- Prompt: "Vote for ONE complete team plan. Consider ship survival, team firepower preservation, and strategic coordination for team victory."
- Collection: Gather votes sequentially or simultaneously
- Validation: Each agent votes for exactly one plan

#### Alpha Step 3 - Plan Selection

- Count votes for each proposal
- Implement majority rule (simple majority of 3 agents)
- Random tie-breaking for 3-way ties or 1-1-1 splits
- Parse winning plan into individual PlayerAction objects

### Agent Context and Prompting

#### Rich Context Generation

- Round number and game state
- Own ship status and position
- Team ship status and positions
- Battlefield memory (hits, misses, eliminations)
- Valid actions for own ship
- Enemy team status (what's known)

#### Alpha Proposal Prompts

- Include full team context
- Emphasize strategic thinking
- Require consideration of all team members
- 50-word reasoning requirement

#### Beta Individual Prompts

- Same battlefield information as Alpha
- Emphasize individual decision-making
- No team coordination information

## Expected Research Outcomes

If coordination hypothesis is correct:

- Alpha teams show higher win rates due to strategic coordination
- Alpha teams demonstrate more effective ship positioning and protection
- Alpha teams exhibit better target prioritization and focus fire
- Alpha proposals evolve in quality over multiple games
- Voting patterns reveal agent learning and adaptation

### Key Research Metrics

- **Win Rate:** Alpha vs Beta across multiple games
- **Coordination Effectiveness:** Quality of Alpha proposals and voting consensus
- **Survival Rate:** Individual ship survival by team
- **Tactical Efficiency:** Hit rates, target selection, resource utilization
- **Emergent Behavior:** Evolution of strategies over multiple rounds

### Experimental Controls

- Same LLM models and configurations for both teams
- Identical ship assignments and battlefield information
- Same random seeds for reproducible results
- Only difference: coordination method (deliberation vs individual)

This design eliminates turn order bias while providing a clean test of whether structured team deliberation outperforms individual decision-making in strategic gameplay.

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install ag2[ollama] networkx
```

### 2. Start Local LLM

```bash
export OLLAMA_HOST=0.0.0.0:11435
ollama serve &
ollama run gpt-oss:120b  # or your preferred model
```

### 3. Configure

- Edit `LLM_config.json` for your model settings
- Set Alpha `"team_coordination": true`, Beta `"team_coordination": false`

### 4. Run Experiment

```bash
# Single game
python battleship_runner.py --seed 123

# Multiple trials
for s in $(seq 1 50); do
  python battleship_runner.py --seed $s
done
```