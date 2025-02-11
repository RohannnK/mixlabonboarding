import os
import sys
import json
import logging
import random
from collections import deque, defaultdict
from copy import deepcopy
from reasoners import WorldModel, SearchConfig, Reasoner
from reasoners.algorithm import MCTS, MCTSResult, MCTSNode
from reasoners.visualization import TreeSnapshot

###############################################################################
# Set up logging and output redirection.
os.makedirs("results", exist_ok=True)
log_file = "results/task2execution.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout
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
# Helper Functions: LLM Feedback, Similarity, Graph Distances, Evaluation

def simulated_llm_feedback(chain):
    """Return bonus of 0.05 per step in the chain."""
    return 0.05 * len(chain)

def similarity_bonus(current_symbol, target_symbol):
    """
    Return 0.5 if symbols match; 0.2 if their first characters match; else 0.
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
    Returns a dictionary mapping each node to its distance.
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
    Compute an evaluation score for a candidate state from chain length, similarity, and graph distance.
    """
    chain = state['path']
    base_bonus = simulated_llm_feedback(chain)
    current_node = state['current_node']
    current_sym = idx_to_symbol.get(current_node, "")
    target_sym = idx_to_symbol.get(target, "")
    sim_bonus = similarity_bonus(current_sym, target_sym)
    dist = distances.get(current_node, float('inf'))
    distance_bonus = max(0, 5 - dist) * 0.1 if dist != float('inf') else 0
    return base_bonus + sim_bonus + distance_bonus

###############################################################################
# World Model

class ProsQAWorldModel(WorldModel):
    """
    Each state is a dictionary with:
      - 'current_node': current node (None initially)
      - 'path': list of (u, v) edges taken so far.
    """
    def __init__(self, idx_to_symbol, edges, target):
        self.idx_to_symbol = {i: s for i, s in enumerate(idx_to_symbol)} if isinstance(idx_to_symbol, list) else idx_to_symbol
        self.edges = [tuple(edge) for edge in edges]
        self.target = target

    def init_state(self):
        init_state = {'current_node': None, 'path': []}
        print(f"[DEBUG] Initializing state: {init_state}", flush=True)
        return init_state

    def step(self, state, action):
        new_state = {
            'current_node': action[1],
            'path': state['path'] + [tuple(action)]
        }
        print(f"[DEBUG] Transition: from {state} with action {action} -> {new_state}", flush=True)
        return new_state, {}

    def is_terminal(self, state):
        term = (state['current_node'] == self.target)
        print(f"[DEBUG] Checking if state {state} is terminal: {term}", flush=True)
        return term

###############################################################################
# Search Configuration

class ProsQASearchConfig(SearchConfig):
    """
    Defines available actions and computes reward.
    """
    def __init__(self, idx_to_symbol, edges, target, root):
        super().__init__()
        self.idx_to_symbol = idx_to_symbol
        self.edges = [tuple(e) for e in edges]
        self.target = target
        self.root = root
        self.distances = compute_reverse_distances(self.edges, target)

    def get_actions(self, state):
        current = state['current_node'] if state['current_node'] is not None else self.root
        actions = [edge for edge in self.edges if edge[0] == current]
        print(f"[DEBUG] Available actions from state {state}: {actions}", flush=True)
        return actions

    def reward(self, state, action):
        valid = (action in self.edges)
        next_node = action[1]
        terminal = (next_node == self.target)
        chain = state['path'] + [tuple(action)]
        base_bonus = simulated_llm_feedback(chain)
        current_sym = self.idx_to_symbol.get(next_node, "")
        target_sym = self.idx_to_symbol.get(self.target, "")
        sim_bonus = similarity_bonus(current_sym, target_sym)
        current_node = state['current_node'] if state['current_node'] is not None else self.root
        old_dist = self.distances.get(current_node, float('inf'))
        new_dist = self.distances.get(next_node, float('inf'))
        dist_improvement = old_dist - new_dist if old_dist != float('inf') and new_dist != float('inf') else 0
        dist_bonus = 0.2 * dist_improvement
        total_bonus = base_bonus + sim_bonus + dist_bonus
        if terminal:
            r = 1.0 + total_bonus
        elif not valid:
            r = 0.1
        else:
            next_state = {'current_node': next_node, 'path': chain}
            actions_next = self.get_actions(next_state)
            r = -0.3 if len(actions_next) == 0 else 0.7 + total_bonus
        print(f"[DEBUG] Reward for state {state} and action {action}: {r}", flush=True)
        return r, {'valid': valid, 'terminal': terminal}

###############################################################################
# Custom Save Function for TreeSnapshot

def save_snapshot(snapshot, file_path):
    """
    Save the TreeSnapshot to a JSON file using its custom __dict__() method.
    """
    # TreeSnapshot defines a __dict__() method that returns a dictionary
    data = snapshot.__dict__()
    with open(file_path, "w") as f:
        json.dump(data, f, default=str, indent=2)
    print(f"[DEBUG] Saved search tree visualization to {file_path}", flush=True)

###############################################################################
# Main Evaluation Function with Ensemble Reranking and Visualization

def evaluate(dataset_path, output_dir="results", num_runs=3, evaluation_threshold=1.0):
    """
    For each example:
      - Run the search num_runs times.
      - Collect candidate states along with their search trace (result.trace_of_nodes) and score.
      - Choose the candidate with the highest score (with a nonempty trace).
      - Build a visualization from that candidate's trace.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    successes = []
    visualization_candidates = []  # (candidate, trace_of_nodes, score, idx_to_symbol)

    for idx, item in enumerate(dataset):
        try:
            idx_to_symbol = {i: s for i, s in enumerate(item['idx_to_symbol'])} if isinstance(item['idx_to_symbol'], list) else item['idx_to_symbol']
            edges = [tuple(e) for e in item['edges']]
            target = item['target']
            root = item['root']
            world_model = ProsQAWorldModel(idx_to_symbol=idx_to_symbol, edges=edges, target=target)
            search_config = ProsQASearchConfig(idx_to_symbol=idx_to_symbol, edges=edges, target=target, root=root)
            candidate_states = []
            candidate_scores = []
            candidate_traces = []
            
            for run in range(num_runs):
                random.seed(run)
                search_algo = MCTS(depth_limit=30, output_trace_in_each_iter=True)
                print(f"[DEBUG] MCTS attributes: {dir(search_algo)}", flush=True)
                reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
                initial_state = world_model.init_state()
                result = reasoner(initial_state)
                print(f"[DEBUG] Run {run} - Search result attributes: {dir(result)}", flush=True)
                if hasattr(result, 'terminal_state') and result.terminal_state is not None:
                    candidate = result.terminal_state
                    score = evaluate_state(candidate, idx_to_symbol, target, search_config.distances)
                    candidate_states.append(candidate)
                    candidate_scores.append(score)
                    # Retrieve the trace from the result
                    trace_nodes = result.trace_of_nodes if hasattr(result, 'trace_of_nodes') and result.trace_of_nodes else []
                    candidate_traces.append(trace_nodes)
                    print(f"[DEBUG] Run {run} - Candidate state: {candidate} with score {score}", flush=True)
                else:
                    logging.error(f"Item {idx}, run {run}: No terminal_state in result.")
                    candidate_states.append(None)
                    candidate_scores.append(float('-inf'))
                    candidate_traces.append([])
            
            if candidate_states and any(s is not None for s in candidate_states):
                best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
                best_candidate = candidate_states[best_idx]
                best_score = candidate_scores[best_idx]
                best_trace = candidate_traces[best_idx]
                print(f"[DEBUG] Best candidate from ensemble: {best_candidate} with score {best_score}", flush=True)
            else:
                best_candidate = None
                best_trace = []
            
            if best_candidate is not None:
                final_eval = evaluate_state(best_candidate, idx_to_symbol, target, search_config.distances)
                logging.info(f"[DEBUG] Global evaluation score for best candidate: {final_eval}")
                success = (best_candidate['current_node'] == target) or (final_eval > evaluation_threshold)
            else:
                success = False
            successes.append(success)
            logging.info(f"Item {idx}: {'SUCCESS' if success else 'FAILURE'}")
            
            if best_trace and len(best_trace) > 0:
                visualization_candidates.append((best_candidate, best_trace, best_score, idx_to_symbol))
            
        except Exception as e:
            logging.error(f"Item {idx} ERROR: {str(e)}")
            successes.append(False)
    
    if visualization_candidates:
        best_overall = max(visualization_candidates, key=lambda tup: tup[2])
        candidate, trace, score, best_idx_to_symbol = best_overall
        logging.info(f"[INFO] Visualizing candidate with score: {score}")
        node_map = {}
        nodes_vis = []
        edges_vis = []
        for i, node in enumerate(trace):
            node_map[node] = i
            if node.state and node.state.get('current_node') is not None:
                label = best_idx_to_symbol.get(node.state['current_node'], "Unknown")
            else:
                label = "Start"
            nodes_vis.append(TreeSnapshot.Node(id=i, data={"label": label}))
        edge_id = 0
        for node in trace:
            if hasattr(node, 'parent') and node.parent in node_map:
                src_idx = node_map[node.parent]
                tgt_idx = node_map[node]
                src_node = node.parent.state.get('current_node') if node.parent.state else None
                tgt_node = node.state.get('current_node') if node.state else None
                src_label = best_idx_to_symbol.get(src_node, "Start")
                tgt_label = best_idx_to_symbol.get(tgt_node, "?")
                action_str = f"{src_label}→{tgt_label}"
                edges_vis.append(TreeSnapshot.Edge(
                    id=edge_id,
                    source=src_idx,
                    target=tgt_idx,
                    data={"action": action_str}
                ))
                edge_id += 1
        snapshot = TreeSnapshot(nodes_vis, edges_vis)
        snapshot_path = os.path.join(output_dir, "search_tree_visualization.json")
        try:
            save_snapshot(snapshot, snapshot_path)
        except Exception as e:
            logging.error(f"[ERROR] Could not save visualization: {str(e)}")
            print("[ERROR] Visualization saving failed.", flush=True)
    else:
        logging.error("[ERROR] No candidate with a nonempty search trace was found for visualization.")
        print("[ERROR] No candidate available for visualization.", flush=True)
    
    accuracy = sum(successes) / len(successes) if len(dataset) > 0 else 0
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
