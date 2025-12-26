# LLM Agent Network: Does Coordination Make Multi-Agent Teams Smarter?

## The Problem

Multi-agent LLM systems are increasingly deployed for complex tasks, but a fundamental question remains unanswered: **does structured coordination actually help, or do independent agents perform just as well?**

This matters because coordination has costs. Deliberation takes time and tokens. Voting requires consensus-building. Shared memory increases context length. If individual agents can match coordinated teams, the simpler architecture wins.

**What's missing:** Clean experimental comparisons are rare. Most multi-agent benchmarks conflate coordination benefits with other variables (different prompts, asymmetric information, turn-order advantages). We need a controlled environment where coordination is the *only* variable.

## Research Questions

1. **RQ1:** Does inter-agent coordination improve team performance in adversarial settings?
2. **RQ2:** What mechanisms explain coordination benefits (or lack thereof)?
3. **RQ3:** How does coordination affect offensive vs. defensive strategy allocation?

## Our Approach

We built a **Dynamic Battleship** testbed that isolates coordination effects:

- **Two teams, identical capabilities:** Alpha Squadron (coordinated) vs. Beta Fleet (independent)
- **Same information:** Both teams see the same battlefield state
- **Simultaneous execution:** No turn-order bias
- **Only difference:** Alpha deliberates (propose → critique → execute); Beta decides independently

### Alpha's 3-Phase Deliberation Protocol
1. **Proposal:** Each agent proposes a complete team strategy with reasoning
2. **Critique:** Agents evaluate and vote on proposals
3. **Execution:** Team executes the selected coordinated plan

### Beta's Independent Approach
- Each agent chooses actions independently
- No communication, no shared planning
- Parallel decision-making

## Results (n = 73)

Loaded 73 valid games, skipped 29 error/incomplete games.

### Summary Table

| Metric | Alpha (Coordinated) | Beta (Independent) | p-value |
|--------|---------------------|-------------------|---------|
| Win Rate | 54.8% (40/73) | 45.2% (33/73) | 0.483 |
| Ship Survival | **44.7%** | 27.9% | 0.014* |
| Repeated Targets/Game | **3.01** | 16.22 | <0.001*** |
| Offense Ratio | 85.9% | 94.7% | <0.001*** |
| Hit Rate | 13.3% | 11.5% | 0.072 |
| Kills/Game | 2.16 | 2.23 | - |
| First Elimination | 20 games | 53 games | - |

Effect size for win rate (Cohen's h): 0.19 (small)

![Coordination Analysis](plots/coordination_analysis.png)

### Key Findings

**RQ1: Does coordination improve performance?**

No. The coordinated team (Alpha) wins 55% of games compared to 45% for independent agents, but this difference is not statistically significant (p = 0.483, 95% CI: [42.7%, 66.5%]). The effect size is small (h = 0.19). **Coordination does not translate to more wins.**

However, coordination does improve *survival*: coordinated teams preserve 45% of ships vs. 28% for independent teams (p = 0.014). Teams that coordinate lose fewer agents even when they don't win.

**RQ2: What mechanisms drive the difference?**

The clearest finding is **targeting efficiency**. Independent agents waste over 5x more bombs on previously-targeted coordinates (16.2 vs 3.0 repeated targets per game, p < 0.001). This is coordination's primary measurable benefit: *collective memory* and *action deconfliction*.

Critically, hit rates are nearly identical (13.3% vs 11.5%, p = 0.072). When agents shoot at new targets, they perform equally well. Kill counts are also equivalent (2.16 vs 2.23 per game). **Coordination doesn't make agents smarter or more lethal; it only prevents them from duplicating effort.**

The efficiency gains don't convert to wins because Beta compensates through sheer volume. Despite wasting shots on repeated targets, Beta fires nearly as many total bombs (4421 vs 4522) and achieves comparable kills.

**RQ3: How does coordination affect strategy?**

Coordinated teams allocate more resources to defense. Alpha uses 789 moves vs. Beta's 314 (offense ratio: 86% vs 95%, p < 0.001). This defensive flexibility explains the survival gap but not a win rate advantage.

The first-elimination data is striking: Beta loses a player first in 53/73 games (73%). Coordination provides early-game protection. Yet this advantage dissipates over the course of the game.

### Interpretation

Coordination benefits are real but narrow:

1. **Deconfliction:** Deliberation prevents multiple agents from targeting the same coordinate (5x fewer wasted shots)
2. **Strategic balance:** Shared planning enables defense allocation that independent agents neglect
3. **Early protection:** Coordinated teams rarely lose the first player

However, these benefits don't compound into victories. Possible explanations:

- **Deliberation overhead:** The propose-critique-vote cycle may slow Alpha's tempo, allowing Beta to compensate
- **Diminishing returns:** Once deconfliction is achieved, additional coordination adds little value
- **Offense dominance:** In this game, raw aggression may be as effective as strategic balance

## Limitations and Future Work

- **Game-specific:** Results may not generalize beyond Battleship's mechanics
- **Single model:** Results are from one LLM configuration (generalization untested)
- **Fixed protocol:** Only one coordination mechanism (propose-critique-vote) has been evaluated
- **Invalid game rate:** 29/102 games (28%) were invalid due to errors or incompleteness

**Planned extensions:**
- Test alternative coordination protocols (hierarchical, emergent, minimal)
- Vary team sizes and ship configurations
- Cross-model experiments
- Investigate why efficiency gains don't convert to wins

---

## Quick Start

```bash
# Install dependencies
pip install ag2[ollama] networkx

# Start LLM server
CUDA_VISIBLE_DEVICES=0,1,2,3 OLLAMA_HOST=127.0.0.1:11436 ollama serve &

# Run single experiment
python battleship_runner.py --seed 123

# Run batch trials
for s in $(seq 1 50); do
  python battleship_runner.py --seed $((s*100))
done

# Analyze results
python analyze_results.py
```

## Project Structure

```
LLM-Agent-Network/
├── battleship_runner.py      # Main experiment launcher
├── battleship_game.py        # Game engine (simultaneous execution)
├── agent_network.py          # Agent communication layer
├── memory_manager.py         # Shared vs. individual memory systems
├── analyze_results.py        # Statistical analysis script
├── LLM_config.json           # Model configuration
├── battleship_config.json    # Game rules and team settings
├── output/                   # Game logs (JSON)
└── plots/                    # Generated figures
```

## Game Rules

- 10x10 grid, 3 ships per team (sizes: 4, 3, 2)
- Each agent controls one ship
- Dual action space: `BOMB <coord>` or `MOVE <direction>`
- Simultaneous execution each round
- Victory condition: Sink all enemy ships
- Player elimination: Agents are removed when their ship sinks

## Configuration

**LLM_config.json:** Model provider and parameters for each agent

**battleship_config.json:** 
- `team_coordination: true` enables deliberation (Alpha)
- `team_coordination: false` enables independent play (Beta)

---

## Citation

If you use this testbed, please cite:

```
@misc{llm-agent-network,
  title={LLM Agent Network: Coordination Effects in Multi-Agent Systems},
  year={2025},
  note={Work in progress}
}
```