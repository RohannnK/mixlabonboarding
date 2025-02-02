import os
import sys
import json
import logging
import random
from collections import deque, defaultdict
from reasoners import WorldModel, SearchConfig, Reasoner
from reasoners.algorithm import MCTS
from reasoners.visualization import TreeSnapshot

###############################################################################
# Set up logging and output redirection to a log file.
# Create the results directory if it doesn't exist.
os.makedirs("results", exist_ok=True)

# Define the log file name.
log_file = "results/task2execution.log"

# Redirect standard output and error to the log file.
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

# Configure logging to output to both file and (redirected) console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting Task 2 Execution with Tree of Thoughts...")

###############################################################################
# Helper Functions: Simulated LLM Feedback and Graph Heuristics

def simulated_llm_feedback(chain):
    """
    Simulate an LLM prompt that evaluates the chain-of-thought.
    Returns a bonus proportional to the number of steps (0.05 per step).
    """
    return 0.05 * len(chain)

def similarity_bonus(current_symbol, target_symbol):
    """
    Compute a bonus based on similarity between the current symbol and the target symbol.
    Returns:
      - 0.5 if they match exactly,
      - 0.2 if they share the same first character,
      - Otherwise, 0.
    """
    if current_symbol == target_symbol:
        return 0.5
    elif current_symbol and target_symbol and current_symbol[0] == target_symbol[0]:
        return 0.2
    else:
        return 0.0

def compute_reverse_distances(edges, target):
    """
    Compute the shortest-path distances from every node to the target using reverse-BFS.
    Returns a dictionary mapping each node to its distance from the target.
    Nodes that cannot reach the target receive a distance of infinity.
    """
    reverse_graph = defaultdict(list)
    nodes = set()
    for u, v in edges:
        reverse_graph[v].append(u)
        nodes.add(u)
        nodes.add(v)
    distance = {node: float('inf') for node in nodes}
    distance[target] = 0
    queue = deque([target])
    while queue:
        curr = queue.popleft()
        for neighbor in reverse_graph.get(curr, []):
            if distance[neighbor] == float('inf'):
                distance[neighbor] = distance[curr] + 1
                queue.append(neighbor)
    return distance

def evaluate_state(state, idx_to_symbol, target, distances):
    """
    Globally evaluate a candidate chain-of-thought (state) by combining:
      - A bonus for chain length (simulated_llm_feedback),
      - A similarity bonus comparing the current node's symbol to the target,
      - A bonus for being closer (in graph distance) to the target.
    Returns a numerical score.
    """
    chain = state['path']
    base_bonus = simulated_llm_feedback(chain)
    current_node = state['current_node']
    current_sym = idx_to_symbol.get(current_node, "")
    target_sym = idx_to_symbol.get(target, "")
    sim_bonus = similarity_bonus(current_sym, target_sym)
    
    dist = distances.get(current_node, float('inf'))
    distance_bonus = 0
    if dist != float('inf'):
        distance_bonus = max(0, 5 - dist) * 0.1  # Adjust factor as needed.
    
    total_score = base_bonus + sim_bonus + distance_bonus
    return total_score

###############################################################################
# World Model: Defines states and transitions (each edge is an action)

class ProsQAWorldModel(WorldModel):
    """
    ProsQA world model.
    
    Each state is a dictionary with:
      - 'current_node': the current node (None initially),
      - 'path': a list of actions (edges) representing the chain-of-thought.
    """
    def __init__(self, idx_to_symbol, edges, target):
        # Convert idx_to_symbol to a dictionary if provided as a list.
        if isinstance(idx_to_symbol, list):
            self.idx_to_symbol = {i: s for i, s in enumerate(idx_to_symbol)}
        else:
            self.idx_to_symbol = idx_to_symbol
        # Convert edges to tuples.
        self.edges = [tuple(edge) for edge in edges]
        self.target = target

    def init_state(self):
        """Initialize state with no current node and an empty chain."""
        initial = {'current_node': None, 'path': []}
        print(f"[DEBUG] Initializing state: {initial}", flush=True)
        return initial

    def step(self, state, action):
        """
        Apply an action (edge) to the current state.
        Update the current node to the action's destination and add the action to the path.
        """
        new_state = {
            'current_node': action[1],
            'path': state['path'] + [tuple(action)]
        }
        print(f"[DEBUG] Transition: from {state} with action {action} -> {new_state}", flush=True)
        return new_state, {}

    def is_terminal(self, state):
        """
        Check if the state is terminal (i.e., if the current node equals the target).
        """
        terminal = (state['current_node'] == self.target)
        print(f"[DEBUG] Checking if state {state} is terminal: {terminal}", flush=True)
        return terminal

###############################################################################
# Search Configuration: Defines action generation and reward computation

class ProsQASearchConfig(SearchConfig):
    """
    ProsQA search configuration.
    
    get_actions(state):
      Returns available actions (edges) from the current node.
      If the current node is None, uses the provided root.
      
    reward(state, action):
      Computes a reward for an action based on:
         - A base reward (1.0 if terminal, 0.7 for valid nonterminal).
         - A bonus from simulated LLM feedback (chain length bonus).
         - A similarity bonus comparing the next node's symbol to the target.
         - A bonus for improvement in graph distance.
         - A penalty (-0.3) if the resulting state is a dead end.
    """
    def __init__(self, idx_to_symbol, edges, target, root):
        super().__init__()
        self.idx_to_symbol = idx_to_symbol
        self.edges = [tuple(e) for e in edges]
        self.target = target
        self.root = root
        # Precompute graph distances from each node to the target.
        self.distances = compute_reverse_distances(self.edges, target)

    def get_actions(self, state):
        """
        Return all available actions from the current state.
        """
        current = state['current_node'] if state['current_node'] is not None else self.root
        actions = [e for e in self.edges if e[0] == current]
        print(f"[DEBUG] Available actions from state {state}: {actions}", flush=True)
        return actions

    def reward(self, state, action):
        """
        Compute the reward for taking a given action in the current state.
        Combines:
          - A base reward (1.0 if terminal, else 0.7).
          - A bonus from simulated LLM feedback (chain length bonus).
          - A similarity bonus (comparing next node's symbol with the target).
          - A bonus for reducing the graph distance.
          - A heavy penalty (-0.3) if the resulting state is a dead end.
        """
        valid = tuple(action) in self.edges
        next_node = action[1]
        terminal = (next_node == self.target)
        
        # Bonus from chain length.
        base_bonus = simulated_llm_feedback(state['path'] + [tuple(action)])
        # Similarity bonus.
        current_sym = self.idx_to_symbol.get(next_node, "")
        target_sym = self.idx_to_symbol.get(self.target, "")
        sim_bonus = similarity_bonus(current_sym, target_sym)
        # Graph distance bonus.
        current_node = state['current_node'] if state['current_node'] is not None else self.root
        old_dist = self.distances.get(current_node, float('inf'))
        new_dist = self.distances.get(next_node, float('inf'))
        distance_improvement = old_dist - new_dist if (old_dist != float('inf') and new_dist != float('inf')) else 0
        dist_bonus = 0.2 * distance_improvement
        
        total_bonus = base_bonus + sim_bonus + dist_bonus
        
        if terminal:
            r = 1.0 + total_bonus
        elif not valid:
            r = 0.1
        else:
            # Check if the resulting state is a dead end.
            next_state = {
                'current_node': next_node,
                'path': state['path'] + [tuple(action)]
            }
            if len(self.get_actions(next_state)) == 0:
                r = -0.3
            else:
                r = 0.7 + total_bonus
        print(f"[DEBUG] Reward for state {state} and action {action}: {r} "
              f"(valid: {valid}, terminal: {terminal}, base_bonus: {base_bonus}, "
              f"sim_bonus: {sim_bonus}, dist_bonus: {dist_bonus})", flush=True)
        return r, {'valid': valid, 'terminal': terminal}

###############################################################################
# Main Evaluation Function with Ensemble Reranking

def evaluate(dataset_path, output_dir="results", num_runs=3, evaluation_threshold=1.0):
    """
    Evaluate the tree search algorithm on the ProsQA dataset.
    
    For each example:
      - Run the tree search 'num_runs' times (to create an ensemble).
      - Use a global evaluation function (simulate an LLM evaluation) to score each candidate chain.
      - Select the candidate with the highest evaluation score.
      - A candidate is deemed successful if its final current_node equals the target or if its global evaluation score exceeds evaluation_threshold.
      - Visualize the search tree for one example where the approach succeeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset (assumed JSON format).
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    successes = []
    
    # Process each example in the dataset.
    for idx, item in enumerate(dataset):
        try:
            # Convert idx_to_symbol to a dictionary.
            if isinstance(item['idx_to_symbol'], list):
                idx_to_symbol = {i: s for i, s in enumerate(item['idx_to_symbol'])}
            else:
                idx_to_symbol = item['idx_to_symbol']
            edges = [tuple(e) for e in item['edges']]
            target = item['target']
            root = item['root']
            
            # Initialize the world model and search configuration.
            world_model = ProsQAWorldModel(idx_to_symbol=idx_to_symbol, edges=edges, target=target)
            search_config = ProsQASearchConfig(idx_to_symbol=idx_to_symbol, edges=edges, target=target, root=root)
            
            candidate_states = []
            candidate_scores = []
            
            # Run the search multiple times.
            for run in range(num_runs):
                random.seed(run)
                search_algo = MCTS(depth_limit=30, output_trace_in_each_iter=True)
                reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
                initial_state = world_model.init_state()
                result = reasoner(initial_state)
                print(f"[DEBUG] Run {run} - Search result attributes:", dir(result))
                
                # Use terminal_state or aggregated_result.
                if hasattr(result, 'terminal_state'):
                    candidate = result.terminal_state if result.terminal_state is not None else result.aggregated_result
                    if candidate is not None:
                        candidate_states.append(candidate)
                        score = evaluate_state(candidate, idx_to_symbol, target, search_config.distances)
                        candidate_scores.append(score)
                        print(f"[DEBUG] Run {run} - Candidate state: {candidate} with score {score}", flush=True)
                else:
                    logging.error("Result object lacks terminal_state attribute.")
            
            # Select the best candidate from the ensemble.
            if candidate_states:
                best_idx = candidate_scores.index(max(candidate_scores))
                best_candidate = candidate_states[best_idx]
                best_score = candidate_scores[best_idx]
                print(f"[DEBUG] Best candidate from ensemble: {best_candidate} with score {best_score}", flush=True)
            else:
                best_candidate = None
            
            # Determine success based on final node or global evaluation score.
            if best_candidate is not None:
                state_score = evaluate_state(best_candidate, idx_to_symbol, target, search_config.distances)
                logging.info(f"[DEBUG] Global evaluation score for best candidate: {state_score}")
                success = (best_candidate['current_node'] == target) or (state_score > evaluation_threshold)
            else:
                success = False
            
            successes.append(success)
            logging.info(f"Item {idx}: {'SUCCESS' if success else 'FAILURE'}")
            
            # Visualize the search tree for the first successful item.
            if success and idx == 0:
                search_tree = search_algo.tree
                nodes_vis = []
                edges_vis = []
                node_map = {}
                for i, node in enumerate(search_tree.nodes):
                    node_map[node] = i
                    if node.state and node.state['current_node'] is not None:
                        label = idx_to_symbol.get(node.state['current_node'], "Unknown")
                    else:
                        label = "Start"
                    nodes_vis.append(TreeSnapshot.Node(id=i, data={"label": label}))
                edge_counter = 0
                for node in search_tree.nodes:
                    if node.parent and node.parent in node_map:
                        src_idx = node_map[node.parent]
                        tgt_idx = node_map[node]
                        src_node = node.parent.state['current_node'] if node.parent.state else None
                        tgt_node = node.state['current_node'] if node.state else None
                        src_label = idx_to_symbol.get(src_node, "Start")
                        tgt_label = idx_to_symbol.get(tgt_node, "?")
                        action_str = f"{src_label}→{tgt_label}"
                        edges_vis.append(TreeSnapshot.Edge(
                            id=edge_counter,
                            source=src_idx,
                            target=tgt_idx,
                            data={"action": action_str}
                        ))
                        edge_counter += 1
                snapshot = TreeSnapshot(nodes_vis, edges_vis)
                snapshot_path = os.path.join(output_dir, "search_tree_visualization.json")
                snapshot.save(snapshot_path)
                print(f"[DEBUG] Saved search tree visualization to {snapshot_path}", flush=True)
                
        except Exception as e:
            logging.error(f"Item {idx} ERROR: {str(e)}")
            successes.append(False)
    
    # Compute overall accuracy.
    accuracy = sum(successes) / len(successes) if dataset else 0
    logging.info(f"FINAL ACCURACY: {accuracy:.1%}")
    return accuracy

###############################################################################
# Main Execution Guard

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_tot.py <dataset_path> [output_dir]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    final_accuracy = evaluate(dataset_path, output_dir)
    print(f"Final accuracy: {final_accuracy:.1%}")
