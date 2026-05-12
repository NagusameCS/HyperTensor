#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
Tree Speculative Decoding: Medusa / EAGLE Infrastructure (Paper III gap 3)

Implements tree-structured speculative decoding primitives compatible with
the geodessical runtime. Supports:
  - Medusa-style: multiple draft heads predict tokens in parallel, tree-attn
    verifier accepts along the tree.
  - EAGLE-style: feature-level drafting where the drafter operates on the
    verifier's hidden states at the feature (not token) level.

Reference: Stewart, "Geodesic Speculative Decoding," HyperTensor Paper III, 2026.
          Cai et al., "Medusa: Simple LLM Inference Acceleration," ICML 2024.
          Li et al., "EAGLE: Speculative Decoding Requires Rethinking Feature
          Uncertainty," arXiv 2024.

Usage:
    from tree_spec import TreeSpecConfig, TreeDrafter, tree_accept
    cfg = TreeSpecConfig(num_heads=4, max_branch=3, max_depth=5)
    drafter = TreeDrafter(cfg)
    tree = drafter.draft(logits, hidden_state)
    accepted, bonus = tree_accept(tree, verifier_logits)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import namedtuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeSpecConfig:
    """Configuration for tree speculative decoding."""
    # Medusa parameters
    num_heads: int = 4          # Number of draft heads
    max_branch: int = 3         # Top-k tokens per position
    max_depth: int = 5          # Maximum tree depth (lookahead)

    # EAGLE parameters
    eagle_mode: bool = False    # Use feature-level drafting
    feature_dim: int = 4096     # Hidden state dimension (for EAGLE)

    # Acceptance
    temperature: float = 0.0    # Sampling temperature (0 = greedy)
    threshold: float = 0.45     # Acceptance probability threshold

    # Composition
    grc_rank: Optional[int] = None  # GRC compression rank (if combined)
    grc_skip_o: bool = True     # Skip output projection in GRC mode


# Tree node for draft structure
TreeNode = namedtuple('TreeNode', ['token_id', 'parent_idx', 'depth', 'logit'])


@dataclass
class DraftTree:
    """A tree of draft tokens with parent-child structure."""
    nodes: List[TreeNode]       # All nodes in BFS order
    paths: List[List[int]]      # All root-to-leaf paths (token IDs)

    @property
    def num_tokens(self) -> int:
        return len(self.nodes)

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    def get_path(self, leaf_idx: int) -> List[int]:
        """Get token ID path from root to leaf."""
        return self.paths[leaf_idx]

    def to_flat_tokens(self) -> List[int]:
        """Flatten tree to token sequence for batched verifier."""
        return [n.token_id for n in self.nodes]


class TreeDrafter:
    """
    Tree-structured drafter implementing Medusa-style multi-head drafting.

    At each position, num_heads independent draft heads predict the next
    token. The top max_branch candidates per head form a tree. The verifier
    evaluates all paths simultaneously via tree attention.

    In EAGLE mode, drafting operates on feature-level representations
    (hidden states) rather than token logits, enabling uncertainty-aware
    drafting at the representation level.
    """

    def __init__(self, config: TreeSpecConfig):
        self.cfg = config
        # Draft heads would be loaded from a trained checkpoint in production.
        # Here we provide the scaffolding with placeholder head functions.
        self._heads: List[callable] = []

    def register_head(self, head_fn: callable):
        """Register a draft head function: (logits, hidden) -> top-k (tokens, logits)."""
        self._heads.append(head_fn)

    def draft(
        self,
        logits: np.ndarray,
        hidden_state: Optional[np.ndarray] = None,
    ) -> DraftTree:
        """
        Build a draft tree from the current position.

        Args:
            logits: Current-step logits, shape (vocab_size,).
            hidden_state: Optional hidden state (for EAGLE mode), shape (d,).

        Returns:
            DraftTree with all draft paths.
        """
        cfg = self.cfg
        nodes: List[TreeNode] = []
        paths: List[List[int]] = []

        # Root: current position argmax (or sampled) token
        if cfg.temperature == 0.0:
            root_token = int(np.argmax(logits))
        else:
            probs = np.exp(logits / cfg.temperature)
            probs /= probs.sum()
            root_token = int(np.random.choice(len(logits), p=probs))

        root_logit = float(logits[root_token])
        nodes.append(TreeNode(root_token, -1, 0, root_logit))

        # BFS tree construction
        queue = [0]  # Node indices to expand
        leaf_paths: Dict[int, List[int]] = {0: [root_token]}

        while queue:
            node_idx = queue.pop(0)
            node = nodes[node_idx]
            if node.depth >= cfg.max_depth:
                continue

            # Collect top-k candidates from all heads
            candidates: Dict[int, float] = {}  # token_id → max logit

            for head_fn in self._heads:
                top_tokens, top_logits = head_fn(logits, hidden_state)
                for tok, logit in zip(top_tokens[:cfg.max_branch],
                                      top_logits[:cfg.max_branch]):
                    tok = int(tok)
                    if tok not in candidates or float(logit) > candidates[tok]:
                        candidates[tok] = float(logit)

            # Sort by logit and take top max_branch
            sorted_candidates = sorted(candidates.items(),
                                       key=lambda x: x[1], reverse=True)
            for tok, logit in sorted_candidates[:cfg.max_branch]:
                child_idx = len(nodes)
                nodes.append(TreeNode(tok, node_idx, node.depth + 1, logit))
                queue.append(child_idx)

                # Build path
                path = leaf_paths[node_idx] + [tok]
                leaf_paths[child_idx] = path

        # Collect all root-to-leaf paths
        for node_idx, node in enumerate(nodes):
            # A node is a leaf if no other node has it as parent
            is_leaf = True
            for other in nodes:
                if other.parent_idx == node_idx:
                    is_leaf = False
                    break
            if is_leaf and node_idx in leaf_paths:
                paths.append(leaf_paths[node_idx])

        return DraftTree(nodes=nodes, paths=paths)


def tree_accept(
    tree: DraftTree,
    verifier_logits: List[np.ndarray],
    temperature: float = 0.0,
) -> Tuple[List[int], Optional[int]]:
    """
    Tree-based acceptance via rejection sampling.

    The verifier evaluates all tree nodes in a single batched forward pass
    using tree attention (causal mask respecting tree structure). Acceptance
    follows the standard speculative rejection rule applied along each path.

    Args:
        tree: Draft tree from the drafter.
        verifier_logits: List of verifier logit vectors, one per tree position.
        temperature: Sampling temperature (0 = greedy match).

    Returns:
        (accepted_tokens, bonus_token):
        - accepted_tokens: List of accepted token IDs.
        - bonus_token: Additional token from verifier after rejection (or None).
    """
    accepted = []

    # Greedy mode: simple token match along each path
    if temperature == 0.0:
        # Find the longest path where all tokens match verifier argmax
        best_path = []
        for path in tree.paths:
            matches = True
            for depth, tok in enumerate(path):
                if depth >= len(verifier_logits):
                    break
                if tok != int(np.argmax(verifier_logits[depth])):
                    matches = False
                    break
            if matches and len(path) > len(best_path):
                best_path = path

        accepted = best_path

        # Bonus token: verifier's next token after accepted path
        bonus = None
        if len(accepted) < len(verifier_logits):
            bonus = int(np.argmax(verifier_logits[len(accepted)]))
    else:
        # Stochastic acceptance with rejection sampling
        for depth in range(min(len(tree.paths[0]) if tree.paths else 0,
                               len(verifier_logits))):
            # Find draft token at this depth (use first path's token)
            draft_tok = None
            for path in tree.paths:
                if depth < len(path):
                    draft_tok = path[depth]
                    break
            if draft_tok is None:
                break

            p_draft = 1.0  # Greedy drafter
            v_logits = verifier_logits[depth]
            v_probs = np.exp(v_logits / temperature)
            v_probs /= v_probs.sum()
            p_verify = v_probs[draft_tok]

            accept_prob = min(1.0, p_verify / max(p_draft, 1e-10))
            if np.random.random() < accept_prob:
                accepted.append(draft_tok)
            else:
                # Rejection: sample from verifier (truncated)
                v_probs[draft_tok] = 0
                v_probs /= v_probs.sum()
                bonus = int(np.random.choice(len(v_probs), p=v_probs))
                accepted.append(bonus)
                return accepted, None

        bonus = None
        if len(accepted) < len(verifier_logits):
            v_logits = verifier_logits[len(accepted)]
            bonus = int(np.argmax(v_logits))

    return accepted, bonus


def compute_acceptance_rate(
    tree: DraftTree,
    verifier_logits: List[np.ndarray],
    n_trials: int = 100,
) -> Tuple[float, float]:
    """
    Monte Carlo estimate of acceptance rate and mean accepted length.

    Returns:
        (acceptance_rate, mean_accepted_length)
    """
    total_accepted = 0
    total_length = 0
    for _ in range(n_trials):
        accepted, bonus = tree_accept(tree, verifier_logits)
        total_accepted += len(accepted)
        total_length += len(tree.nodes)
    return total_accepted / max(total_length, 1), total_accepted / n_trials


# ---------------------------------------------------------------------------
# EAGLE-style feature drafter
# ---------------------------------------------------------------------------
class EagleFeatureDrafter:
    """
    EAGLE-style drafter that operates on feature-level representations.

    Unlike token-level drafting (Medusa), EAGLE drafts at the hidden-state
    feature level, predicting the verifier's hidden state at future positions.
    Tokens are then decoded from these predicted features.

    This is a scaffold; a full implementation requires a trained feature
    predictor network aligned with the verifier's hidden states.
    """

    def __init__(self, d_model: int = 4096, max_depth: int = 4):
        self.d_model = d_model
        self.max_depth = max_depth

    def predict_features(
        self, h_current: np.ndarray, n_steps: int
    ) -> List[np.ndarray]:
        """
        Predict future hidden states from current.

        Args:
            h_current: Current hidden state, shape (d,).
            n_steps: Number of future steps to predict.

        Returns:
            List of predicted hidden states.
        """
        # Placeholder: in production this would be a trained predictor network.
        # Returns h_current repeated (identity prediction — trivial baseline).
        return [h_current.copy() for _ in range(n_steps)]

    def features_to_tokens(
        self, features: List[np.ndarray], lm_head: np.ndarray
    ) -> List[int]:
        """
        Decode predicted features to token IDs via the LM head.

        Args:
            features: List of predicted hidden states.
            lm_head: LM head weight matrix, shape (vocab_size, d).

        Returns:
            List of predicted token IDs.
        """
        tokens = []
        for h in features:
            logits = lm_head @ h
            tokens.append(int(np.argmax(logits)))
        return tokens


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Tree Speculative Decoding — Self-Test")
    print("=" * 60)

    # Toy vocabulary
    vocab_size = 1000
    rng = np.random.default_rng(42)

    # Create a simple draft head: always returns top-k of current logits
    def dummy_head(logits, hidden=None):
        order = np.argsort(-logits)
        return order[:5], logits[order[:5]]

    # Medusa-style test
    cfg = TreeSpecConfig(num_heads=3, max_branch=3, max_depth=4)
    drafter = TreeDrafter(cfg)
    drafter.register_head(dummy_head)
    drafter.register_head(dummy_head)
    drafter.register_head(dummy_head)

    logits = rng.normal(0, 1, vocab_size)
    tree = drafter.draft(logits)

    print(f"  Tree nodes: {tree.num_tokens}")
    print(f"  Tree paths: {tree.num_paths}")
    print(f"  Max depth: {max(n.depth for n in tree.nodes)}")

    # Simulate verifier
    verifier_logits = [rng.normal(0, 1, vocab_size) for _ in range(10)]
    accepted, bonus = tree_accept(tree, verifier_logits)
    print(f"  Accepted tokens: {len(accepted)}")
    print(f"  Bonus token: {bonus}")

    # EAGLE-style test
    eagle = EagleFeatureDrafter(d_model=256)  # Small for test
    h = rng.normal(0, 1, 256)
    features = eagle.predict_features(h, 3)
    print(f"  Eagle predicted features: {len(features)}")
    print(f"  Feature shape: {features[0].shape}")

    # Feature→token decode
    lm_head = rng.normal(0, 0.1, (vocab_size, 256))
    tokens = eagle.features_to_tokens(features, lm_head)
    print(f"  Eagle decoded tokens: {tokens}")

    print("\n  Tree Spec module: OK")
