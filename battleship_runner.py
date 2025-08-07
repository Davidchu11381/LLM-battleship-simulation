#!/usr/bin/env python3
"""
Battleship Game Runner

Main script to run battleship simulations using the agent network framework.
"""

import asyncio
import argparse
import sys
import json
import traceback
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

from agent_network import AgentNetwork, NetworkConfig
from battleship_game import run_battleship_simulation


def create_sample_battleship_network():
    """Create a sample network topology file for battleship"""
    sample_edges = [
        "# Battleship Network Topology",
        "# Team Alpha internal communication",
        "leader_alpha -> player_a1",
        "leader_alpha -> player_a2",
        "player_a1 -> leader_alpha",
        "player_a1 -> player_a2",
        "player_a2 -> leader_alpha",
        "player_a2 -> player_a1",
        "",
        "# Team Bravo internal communication",
        "leader_bravo -> player_b1",
        "player_b1 -> leader_bravo",
        "",
        "# AI Assistant connections",
        "assistant_alpha_leader -> leader_alpha",
        "assistant_a1 -> player_a1",
        "assistant_bravo_leader -> leader_bravo",
        "assistant_b1 -> player_b1",
        "",
        "# Game Master connections (can communicate with all players)",
        "game_master -> leader_alpha",
        "game_master -> player_a1",
        "game_master -> player_a2",
        "game_master -> leader_bravo",
        "game_master -> player_b1"
    ]
    
    networks_dir = Path("networks")
    networks_dir.mkdir(exist_ok=True)
    
    with open(networks_dir / "battleship.txt", "w") as f:
        f.write("\n".join(sample_edges))
    
    print(f"📝 Created sample battleship network: {networks_dir / 'battleship.txt'}")


async def main():
    """Main battleship game runner"""
    parser = argparse.ArgumentParser(description='Run Battleship Agent Network Simulation')
    parser.add_argument('--llm-config', type=str, default='LLM_config.json',
                        help='Path to LLM configuration file (default: LLM_config.json)')
    parser.add_argument('--battleship-config', type=str, default='battleship_config.json',
                        help='Path to battleship configuration file (default: battleship_config.json)')
    parser.add_argument('--edges', type=str, default=None,
                        help='Network topology file (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    print("⚔️  BATTLESHIP AGENT NETWORK SIMULATION")
    print("=" * 50)
    
    # Verify config files exist
    if not Path(args.llm_config).exists():
        print(f"❌ LLM config file '{args.llm_config}' not found!")
        return
    if not Path(args.battleship_config).exists():
        print(f"❌ Battleship config file '{args.battleship_config}' not found!")
        return
    
    try:
        # 1️⃣ Load both configs
        print(f"📄 Loading LLM configuration from {args.llm_config}...")
        with open(args.llm_config) as f:
            llm_cfg = json.load(f)
        
        print(f"📄 Loading battleship configuration from {args.battleship_config}...")
        with open(args.battleship_config) as f:
            game_cfg = json.load(f)
        
        # 2️⃣ Merge each agent’s battleship profile into its LLM specialization
        for agent_id, assignment in game_cfg["agent_assignments"].items():
            profile = game_cfg["player_profiles"][assignment["profile"]]
            node = llm_cfg.setdefault("nodes", {}).setdefault(agent_id, {})
            spec = node.setdefault("specialization", {})
            spec.update(profile)
            # also inject team_name
            for team in game_cfg["team_config"].values():
                if agent_id in team["members"]:
                    spec["team_name"] = team["name"]
                    break
        
        # 3️⃣ Parse edges (either provided or sample)
        if args.edges:
            edges = NetworkConfig._load_edges_from_file(args.edges)
        else:
            sample_edges_file = Path("networks/battleship.txt")
            if sample_edges_file.exists():
                edges = NetworkConfig._load_edges_from_file(str(sample_edges_file))
            else:
                print("⚠️  No network topology specified and battleship.txt not found!")
                print("💡 Run with --create-sample first, then try again.")
                return
        
        # 4️⃣ Initialize agent network from enriched config
        net_cfg = NetworkConfig(
            nodes=llm_cfg["nodes"],
            edges=edges,
            global_config=llm_cfg.get("global_config", {})
        )
        network = AgentNetwork(net_cfg)
        
        # 5️⃣ Display network summary
        print("\n⚔️  Battleship Network Configuration:")
        network.visualize_network()
        info = network.get_network_info()
        print(f"📊 Summary: {info['num_agents']} agents, {info['num_connections']} connections")
        print(f"🎮 Communication rounds: Read from config file")
        
        # 6️⃣ Start the simulation
        print("\n🚀 STARTING BATTLESHIP SIMULATION")
        print("=" * 50)
        game, stats = await run_battleship_simulation(
            agent_network=network,
            battleship_config_path=args.battleship_config
        )
        
        # 7️⃣ Detailed Summary & Analysis
        print("\n" + "=" * 50)
        print("📊 DETAILED GAME ANALYSIS")
        print("=" * 50)
        
        if args.verbose:
            print("\n🗺️  FINAL GRID STATES:")
            for team_id, team in game.state.teams.items():
                print(f"\n{team.name} ({team_id}):")
                grid_display = team.grid.get_display_grid(hide_ships=False)
                # Column headers
                print("   " + " ".join(str(i+1).rjust(2) for i in range(len(grid_display[0]))))
                # Rows
                for i, row in enumerate(grid_display):
                    label = chr(ord('A') + i)
                    print(f"{label:2} " + " ".join(cell.rjust(2) for cell in row))
        
        # Communication breakdown
        print(f"\n💬 COMMUNICATION BREAKDOWN:")
        for team_id, team in game.state.teams.items():
            print(f"\n{team.name}:")
            for member in team.members:
                events = [
                    e for e in game.state.game_log
                    if e.get("metadata", {}).get("player") == member
                    and e["event_type"] in ["AI_CONSULTATION", "TEAM_DISCUSSION"]
                ]
                ai_cnt = sum(1 for e in events if e["event_type"] == "AI_CONSULTATION")
                team_cnt = sum(1 for e in events if e["event_type"] == "TEAM_DISCUSSION")
                print(f"  {member}: {ai_cnt} AI consultations, {team_cnt} team discussions")
        
        # Coordinate call efficiency
        print(f"\n🎯 COORDINATE CALL EFFICIENCY:")
        for player_id, coords in game.coordinate_history.items():
            efficiency = (len(set(coords)) / len(coords)) if coords else 0
            print(f"  {player_id}: {len(coords)} calls, {efficiency:.1%} efficiency")
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📈 Total game events logged: {len(game.state.game_log)}")
    
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        if args.verbose:
            traceback.print_exc()
        print("\n💡 Common issues:")
        print("   - Check API keys in LLM config")
        print("   - Ensure Ollama is running: ollama serve")
        print("   - Verify agent IDs match between configs")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_battleship_network()
        print("✅ Sample battleship network created!")
        print("💡 Now run: python battleship_runner.py")
    else:
        asyncio.run(main())
