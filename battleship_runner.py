#!/usr/bin/env python3
"""
Simultaneous Battleship Game Runner

Main script to run simultaneous battleship simulations with individual ship ownership,
dual action space (BOMB/MOVE), and 3-phase Alpha coordination vs Beta individual decisions.
"""

import asyncio
import argparse
import sys
import json
import traceback
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

from agent_network import create_battleship_network
from battleship_game import run_simultaneous_battleship_simulation


async def main():
    """Main simultaneous battleship game runner"""
    parser = argparse.ArgumentParser(description='Run Simultaneous Battleship Agent Network Simulation')
    parser.add_argument('--llm-config', type=str, default='LLM_config.json',
                        help='Path to LLM configuration file (default: LLM_config.json)')
    parser.add_argument('--battleship-config', type=str, default='battleship_config.json',
                        help='Path to battleship configuration file (default: battleship_config.json)')
    parser.add_argument('--edges', type=str, default='networks/battleship.txt',
                        help='Network topology file (default: networks/battleship.txt)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible ship assignments')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--export-memories', action='store_true',
                        help='Export agent memories after game completion')
    args = parser.parse_args()
    
    print("🎮 SIMULTANEOUS BATTLESHIP AGENT NETWORK SIMULATION")
    print("=" * 60)
    
    # Verify config files exist
    if not Path(args.llm_config).exists():
        print(f"❌ LLM config file '{args.llm_config}' not found!")
        return
    if not Path(args.battleship_config).exists():
        print(f"❌ Battleship config file '{args.battleship_config}' not found!")
        return
    if not Path(args.edges).exists():
        print(f"❌ Network topology file '{args.edges}' not found!")
        print(f"💡 Create {args.edges} with proper Alpha/Beta communication rules")
        return
    
    try:
        # 1️⃣ Load configurations
        print(f"📄 Loading LLM configuration from {args.llm_config}...")
        print(f"📄 Loading battleship configuration from {args.battleship_config}...")
        print(f"📄 Loading network topology from {args.edges}...")
        
        with open(args.battleship_config) as f:
            game_cfg = json.load(f)
        
        # 2️⃣ Validate simultaneous battleship configuration
        team_config = game_cfg.get('team_config', {})
        alpha_members = team_config.get('team_a', {}).get('members', [])
        beta_members = team_config.get('team_b', {}).get('members', [])
        
        if not alpha_members or not beta_members:
            print("❌ Both Alpha Squadron and Beta Fleet must have members!")
            return
        
        print(f"🔵 Alpha Squadron: {len(alpha_members)} members - {', '.join(alpha_members)}")
        print(f"🔴 Beta Fleet: {len(beta_members)} members - {', '.join(beta_members)}")
        
        # Check team coordination settings
        alpha_coordination = team_config.get('team_a', {}).get('team_coordination', False)
        beta_coordination = team_config.get('team_b', {}).get('team_coordination', False)
        print(f"🤝 Alpha coordination: {'✅ 3-Phase Protocol' if alpha_coordination else '❌ Disabled'}")
        print(f"🤝 Beta coordination: {'✅ Enabled' if beta_coordination else '❌ Individual Decisions'}")
        
        if not alpha_coordination:
            print("⚠️  Warning: Alpha Squadron should have team_coordination=true for 3-phase deliberation experiment")
        if beta_coordination:
            print("⚠️  Warning: Beta Fleet should have team_coordination=false for individual decision experiment")
        
        # 3️⃣ Initialize simultaneous battleship network
        print(f"\n🕸️  Initializing agent network...")
        network = create_battleship_network(args.llm_config, args.edges)
        
        # 4️⃣ Display network summary
        print("\n🎮 Simultaneous Battleship Network Configuration:")
        network.visualize_network()
        info = network.get_network_info()
        print(f"📊 Network Summary: {info['total_agents']} agents, {info['total_connections']} connections")
        
        # 5️⃣ Game setup summary
        ship_set = game_cfg['game_config']['selected_ship_set']
        ships = game_cfg['game_config']['ship_sets'][ship_set]
        print(f"\n⚓ Ship Configuration: {ship_set} set")
        for ship in ships:
            print(f"   {ship['name']} (size {ship['size']}) x{ship['quantity']}")
        
        if args.seed:
            print(f"🎲 Random seed: {args.seed} (reproducible ship assignments)")
        else:
            print(f"🎲 Random seed: None (random ship assignments)")
        
        # 6️⃣ Start the simultaneous battleship simulation
        print(f"\n🚀 STARTING SIMULTANEOUS BATTLESHIP SIMULATION")
        print("=" * 60)
        print("🎯 Experiment: 3-Phase Alpha Coordination vs Beta Individual Decisions")
        print("⚡ NEW: Simultaneous execution eliminates turn order bias")
        print("🚢 Features: Individual ship ownership, dual action space (BOMB/MOVE), player elimination")
        print("")
        
        game, stats = await run_simultaneous_battleship_simulation(
            agent_network=network,
            battleship_config_path=args.battleship_config,
            random_seed=args.seed
        )
        
        # 7️⃣ Simultaneous Battleship Analysis
        print("\n" + "=" * 60)
        print("🎮 SIMULTANEOUS BATTLESHIP GAME ANALYSIS")
        print("=" * 60)
        
        # Game outcome
        if game.state.winner:
            winner_team = game.state.teams[game.state.winner]
            print(f"🏆 WINNER: {winner_team.name}")
            
            # Analyze coordination effect on victory
            winner_coordination = team_config.get(game.state.winner, {}).get('team_coordination', False)
            coordination_effect = "3-phase coordination" if winner_coordination else "individual decisions"
            print(f"🤝 Victory method: {coordination_effect}")
        else:
            print("🤝 GAME ENDED WITHOUT CLEAR WINNER")
        
        print(f"📊 Game duration: {game.state.round_number} rounds")
        
        # Individual ship ownership and elimination summary
        print(f"\n🚢 INDIVIDUAL SHIP OWNERSHIP & ELIMINATION:")
        total_eliminated = 0
        for team_id, team in game.state.teams.items():
            print(f"\n  {team.name}:")
            for ship in team.grid.ships:
                status = "💥 SUNK" if ship.is_sunk else "⚓ ALIVE"
                if ship.owner_id in game.state.eliminated_players:
                    status += " (❌ PLAYER ELIMINATED)"
                    total_eliminated += 1
                print(f"    {ship.owner_id}: {ship.name} (size {ship.size}) {status}")
        
        print(f"\n💀 Total players eliminated: {total_eliminated}/{len(alpha_members + beta_members)}")
        
        # Action analysis (if available)
        if 'action_metrics' in stats:
            bomb_actions = stats['action_metrics']['total_bomb_actions']
            move_actions = stats['action_metrics']['total_move_actions']
            print(f"\n⚔️ SIMULTANEOUS ACTION SPACE ANALYSIS:")
            print(f"  💣 Bomb actions: {bomb_actions}")
            print(f"  🚢 Move actions: {move_actions}")
            print(f"  📊 Total actions: {bomb_actions + move_actions}")
            
            if move_actions > 0:
                ratio = bomb_actions / move_actions
                print(f"  📈 Bomb:Move ratio: {ratio:.1f}:1")
            
            # Analyze action distribution by team
            print(f"\n📊 Action distribution analysis:")
            print(f"   Movement usage: {move_actions / (bomb_actions + move_actions) * 100:.1f}% of all actions")
            print(f"   Strategic balance: {'High mobility' if move_actions > bomb_actions * 0.3 else 'Low mobility'}")
        
        # Team coordination experiment results
        print(f"\n🧪 COORDINATION EXPERIMENT RESULTS:")
        alpha_config = team_config.get('team_a', {})
        beta_config = team_config.get('team_b', {})
        
        print(f"  🔵 Alpha Squadron (3-Phase Protocol):")
        print(f"     Strategy: Proposals → Voting → Plan Selection")
        print(f"     Coordination: {'✅ 3-Phase deliberation' if alpha_config.get('team_coordination') else '❌ No coordination'}")
        print(f"     Ships remaining: {len([s for s in game.state.teams['team_a'].grid.ships if not s.is_sunk])}")
        
        print(f"  🔴 Beta Fleet (Individual Decisions):")
        print(f"     Strategy: Parallel individual choices")
        print(f"     Coordination: {'✅ Team coordination' if beta_config.get('team_coordination') else '❌ Individual decisions'}")
        print(f"     Ships remaining: {len([s for s in game.state.teams['team_b'].grid.ships if not s.is_sunk])}")
        
        # Player performance breakdown
        print(f"\n🎯 PLAYER PERFORMANCE:")
        for player_id, score in game.state.player_scores.items():
            eliminated_status = " (❌ ELIMINATED)" if player_id in game.state.eliminated_players else ""
            print(f"  {player_id}: {score.total_score} points{eliminated_status}")
            print(f"    Victory: +{score.team_victory}, Survival: +{score.ship_survival}, Coordination: +{score.coordination_bonus}, Penalties: {score.waste_penalty}")
        
        # Verbose analysis
        if args.verbose:
            print(f"\n🗺️ FINAL GRID STATES:")
            for team_id, team in game.state.teams.items():
                print(f"\n{team.name} ({team_id}):")
                # Simple grid display
                print("   Grid visualization not implemented - see game log for details")
            
            print(f"\n📋 DETAILED EVENT LOG:")
            important_events = [e for e in game.state.game_log 
                               if e["event_type"] in ["BOMB_RESULT", "MOVE_SUCCESS", "GAME_OVER", "SIMULTANEOUS_EXECUTION_START"]]
            for event in important_events[-10:]:  # Last 10 important events
                timestamp = event["timestamp"][:19]  # Remove microseconds
                print(f"  [{timestamp}] {event['event_type']}: {event['message']}")
        
        # Export memories if requested
        if args.export_memories:
            try:
                memory_export = game.memory_manager.export_all_memories()
                memory_file = Path("output") / f"memories_{game.state.round_number}rounds.json"
                memory_file.parent.mkdir(exist_ok=True)
                with memory_file.open('w') as f:
                    json.dump(memory_export, f, indent=2)
                print(f"\n💾 Agent memories exported to: {memory_file}")
            except Exception as e:
                print(f"\n⚠️ Memory export failed: {e}")
        
        # Final summary
        print(f"\n✅ Simultaneous battleship simulation completed successfully!")
        print(f"📈 Total game events logged: {len(game.state.game_log)}")
        
        # Save detailed game log
        log_file = game.save_game_log()
        print(f"📝 Complete game log saved to: {log_file}")
        
        # Experiment insights
        print(f"\n🔬 EXPERIMENT INSIGHTS:")
        if game.state.winner and alpha_coordination != beta_coordination:
            winner_used_coordination = team_config.get(game.state.winner, {}).get('team_coordination', False)
            insight = "3-phase coordination" if winner_used_coordination else "Individual decisions"
            print(f"   💡 {insight} strategy achieved victory in this simulation")
        print(f"   📊 Individual ship ownership created {total_eliminated} eliminations")
        print(f"   ⚡ Simultaneous execution eliminated turn order bias")
        if 'action_metrics' in stats and stats['action_metrics']['total_move_actions'] > 0:
            print(f"   🚢 Dynamic movement added {stats['action_metrics']['total_move_actions']} tactical repositioning actions")
        
        # Team coordination analysis summary
        coordination_summary = game.memory_manager.get_experiment_summary()
        if coordination_summary:
            print(f"\n📊 COORDINATION EFFECTIVENESS SUMMARY:")
            coord_data = coordination_summary.get('coordination_effectiveness', {})
            for team_id, team_analysis in coord_data.items():
                if isinstance(team_analysis, dict):
                    coord_level = team_analysis.get('coordination_level', 'unknown')
                    consensus_rate = team_analysis.get('consensus_rate', 0)
                    print(f"   {team_id}: {coord_level} coordination ({consensus_rate:.1%} consensus rate)")
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        if args.verbose:
            traceback.print_exc()
        print("\n💡 Common troubleshooting:")
        print("   - Check API keys in LLM config")
        print("   - Ensure Ollama is running: ollama serve")
        print("   - Verify agent IDs match between configs")
        print("   - Check network topology file format")
        print("   - Validate team coordination settings")
        print("   - Ensure team_a has team_coordination=true, team_b has team_coordination=false")


if __name__ == "__main__":
    asyncio.run(main())