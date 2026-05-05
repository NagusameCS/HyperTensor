"""EC2 SCALE — Saiyan Zone-Weighted Fusion + Danish Graft Benchmarks.

Deploy to EC2 L40S (48GB VRAM). Loads Qwen2.5-7B at 4-bit (~5.6GB).
Builds 6 Saiyan manifolds at K=512 with 100 training prompts each.
Creates zone-weighted fusions. Benchmarks all against naive fusion.
Also runs HyperInstinct jury on all 7 Danish grafted models.

Expected runtime: 2-4 hours on L40S.
"""
import torch, json, time, math, os, sys, random
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/ec2_scale")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  EC2 SCALE — Saiyan Fusion + Danish Graft Benchmarks")
print(f"  Device: {DEVICE}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if DEVICE=="cuda" else "  CPU mode")
print("=" * 70)

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
K_TARGET = 512
N_PROMPTS_PER_SAIYAN = 100
JURY_TRIALS = 7

# ============================================================================
# TRAINING DATA (100 prompts per domain)
# ============================================================================
TRAIN_DATA = {
    "math": [
        "Solve 3x + 7 = 22 step by step, showing all algebra.",
        "Find the derivative of f(x)=x^3*sin(x) using the product rule.",
        "Factor x^2 - 5x + 6 completely.",
        "Prove that sqrt(2) is irrational by contradiction.",
        "What is the sum of integers from 1 to 100? Use the Gauss formula.",
        "Calculate det([[2,1,3],[0,4,1],[1,0,2]]) using cofactor expansion.",
        "Find the integral of 1/x from 1 to e.",
        "Solve the system: 2x+y=7, x-y=1 by substitution.",
        "Evaluate sin(pi/6), cos(pi/3), tan(pi/4).",
        "Count permutations of 5 distinct items. Explain the formula.",
        "Prove that the angles in a triangle sum to 180 degrees.",
        "Find the vertex of y = x^2 - 6x + 8.",
        "Solve for x: |2x - 3| = 5.",
        "Find the limit: lim(x->0) sin(x)/x. Explain L'Hopital or squeeze.",
        "Integrate x^2 * e^x using integration by parts.",
        "Find eigenvalues of [[3,1],[0,2]].",
        "Prove by induction: sum_{i=1}^n i = n(n+1)/2.",
        "What is the probability of exactly 3 heads in 5 coin flips?",
        "Explain the Central Limit Theorem with an example.",
        "Solve the recurrence: a_n = 2a_{n-1} + 1, a_0 = 0.",
        "What is a group in abstract algebra? Give examples.",
        "Prove there are infinitely many primes (Euclid's proof).",
        "Find the Fourier series of a square wave.",
        "Explain eigenvalues and eigenvectors geometrically.",
        "Solve x^4 - 5x^2 + 4 = 0 by substitution.",
        "What is a Markov chain? Give real-world examples.",
        "Prove that e is irrational.",
        "Find the area under y=x^2 from 0 to 2.",
        "What is the Banach-Tarski paradox?",
        "Solve dy/dx = ky, with y(0)=1.",
        "Explain the concept of a vector space.",
        "What is the Riemann zeta function?",
        "Find the Taylor series of sin(x) around 0.",
        "Prove the Pythagorean theorem geometrically.",
        "What is a normal distribution? Give its pdf.",
        "Find the inverse of [[1,2],[3,4]] if it exists.",
        "What is the difference between permutation and combination?",
        "Solve 2^x = 8 using logarithms.",
        "What is a complex number? Plot 3+4i.",
        "Find the gradient of f(x,y)=x^2*y+y^3 at (1,2).",
        "What is the fundamental theorem of calculus?",
        "Prove that the harmonic series diverges.",
        "Find all solutions to sin(x) = 1/2 in [0, 2pi].",
        "What is a confidence interval? Give an example.",
        "Explain Bayes theorem with a medical testing example.",
        "Find the volume of a sphere with radius r using integration.",
        "What is a convex function? Give the definition.",
        "Prove De Morgan's laws for sets.",
        "Find the matrix product [[1,2],[3,4]] * [[0,1],[1,0]].",
        "What is the law of large numbers?",
        "Explain the concept of linear independence.",
        "Find the distance from point (1,2,3) to plane x+y+z=0.",
        "What is a Singular Value Decomposition (SVD)?",
        "Prove that (A+B)^2 = A^2+AB+BA+B^2 for matrices.",
        "Find the maximum of f(x)=x*e^(-x) on [0,infinity).",
        "What is a Hilbert space?",
        "Solve the differential equation y'' + y = 0.",
        "What is the Kullback-Leibler divergence?",
        "Find the convolution of f*g where f=g=1 on [0,1].",
        "Explain the difference between O, Theta, and Omega notation.",
        "What is a generating function? Use for Fibonacci numbers.",
        "Prove the Cauchy-Schwarz inequality.",
        "What is the determinant geometrically?",
        "Find the Lagrange multipliers for maximize xy subject to x^2+y^2=1.",
        "What is the Fourier transform of e^(-x^2)?",
        "Explain the Monty Hall problem with conditional probability.",
        "What is the Jacobian determinant? When is it used?",
        "Find the general solution to the wave equation.",
        "What is a tensor? Give the definition.",
        "Prove that the empty set is a subset of every set.",
        "What is the Gamma function? Relate to factorial.",
        "Find the residue of f(z)=1/(z^2+1) at z=i.",
        "What is the definition of a manifold?",
        "Explain the concept of homotopy.",
        "Prove that sin^2(x) + cos^2(x) = 1.",
        "What is the Chebyshev inequality? Give the statement.",
        "Find the polar decomposition of a 2x2 matrix.",
        "What is a Noetherian ring?",
        "Explain the Riesz representation theorem.",
        "Prove that sqrt(2)+sqrt(3) is irrational.",
        "What is the Lebesgue measure of rationals in [0,1]?",
        "Find the orthogonal projection of (1,1,1) onto span{(1,0,0),(0,1,0)}.",
        "What is the definition of a topological space?",
        "Explain the Borsuk-Ulam theorem.",
        "What is the proof that pi is transcendental? (outline)",
        "Find the geodesic on a sphere.",
        "What is a Galois group? Give an example.",
        "Prove that the set of algebraic numbers is countable.",
        "What is the spectral theorem for self-adjoint operators?",
        "Find the Euler characteristic of a torus.",
        "What is the definition of a category in category theory?",
        "Explain Godel's incompleteness theorems.",
        "Prove the fundamental theorem of algebra (outline).",
        "What is the Poincare conjecture? (statement)",
        "Find the general solution to Laplace's equation in 2D.",
        "What is a homology group?",
        "Explain the Riemann mapping theorem.",
        "Prove that every prime p=1 mod 4 is sum of two squares.",
        "What is the Atiyah-Singer index theorem? (statement)",
    ],
    "code": [
        "Write a Python function that implements binary search on a sorted array.",
        "What is the time complexity of quicksort? Explain best, average, worst cases.",
        "Implement a hash table with chaining in Python.",
        "Write code to reverse a singly linked list iteratively and recursively.",
        "Explain the difference between BFS and DFS with examples.",
        "Implement an LRU cache with O(1) get and put in Python.",
        "What does O(n log n) mean? Why can't comparison sorts beat it?",
        "Write a SQL query to find duplicate emails in a users table.",
        "Explain the CAP theorem and its implications for distributed systems.",
        "What is a race condition? Show a Python example and how to fix it.",
        "Implement Dijkstra's shortest path algorithm.",
        "Write a dynamic programming solution for knapsack.",
        "Implement a trie with insert, search, and startsWith.",
        "Find the longest palindromic substring in O(n) time.",
        "Explain merge sort and why it's O(n log n).",
        "Implement a min-heap with push and pop operations.",
        "What is a Bloom filter? When would you use one?",
        "Implement a graph using adjacency list and adjacency matrix.",
        "Explain red-black trees and when you'd use them over AVL.",
        "Design a URL shortener like bit.ly — full system design.",
        "Design a distributed key-value store with strong consistency.",
        "How would you design Twitter's tweet feed system?",
        "Design a rate limiter for an API with token bucket algorithm.",
        "Explain async/await in Python with a concrete example.",
        "What are Python decorators? Write a @timer decorator.",
        "Explain Python's GIL and its implications for concurrency.",
        "What is the difference between deep copy and shallow copy?",
        "Implement a thread-safe singleton in Python.",
        "Write unit tests for a Stack class using pytest.",
        "What is test-driven development? Give a workflow example.",
        "Explain mock objects in testing with a Python example.",
        "Review this code: def f(n): return f(n-1)+f(n-2) if n>1 else n. Issues?",
        "What makes code Pythonic? Give Pythonic vs non-Pythonic examples.",
        "Explain SOLID principles with Python examples.",
        "Implement the Observer pattern in Python.",
        "What is dependency injection? Give a Python example.",
        "Explain the difference between composition and inheritance.",
        "Implement a simple Promise/Future in Python.",
        "What is a coroutine? How does it differ from a thread?",
        "Explain the Event Loop pattern in async programming.",
        "What is a memory leak? How to detect in Python?",
        "Explain garbage collection in Python (reference counting + generational).",
        "What is the difference between TCP and UDP? When to use each?",
        "Explain REST vs GraphQL with examples.",
        "What is WebSocket? When would you use it over HTTP?",
        "Explain OAuth 2.0 flow for a web application.",
        "What is JWT? How does token-based auth work?",
        "Explain SQL injection and how to prevent it.",
        "What is XSS? How do you prevent it in a web app?",
        "What is CORS? Why is it needed?",
        "Explain hashing vs encryption vs encoding.",
        "What is a man-in-the-middle attack? How does TLS prevent it?",
        "Explain the Diffie-Hellman key exchange.",
        "What is a blockchain? Explain the Merkle tree structure.",
        "Explain MapReduce with a word count example.",
        "What is the difference between batch and stream processing?",
        "Implement a basic neural network layer from scratch in NumPy.",
        "What is backpropagation? Derive the chain rule for a 2-layer network.",
        "Explain the attention mechanism in transformers.",
        "What is transfer learning? Give a computer vision example.",
        "Explain gradient descent variants: SGD, Momentum, Adam.",
        "What is the vanishing gradient problem? How do you solve it?",
        "Explain dropout regularization and why it works.",
        "What is batch normalization? Write the formula.",
        "Implement k-means clustering from scratch.",
        "What is the bias-variance tradeoff?",
        "Explain cross-validation and why you need it.",
        "What is a confusion matrix? Compute precision, recall, F1.",
        "Explain ROC curves and AUC.",
        "What is PCA? Derive it from the covariance matrix.",
        "Implement a simple decision tree classifier.",
        "What is ensemble learning? Compare bagging vs boosting.",
        "Explain how random forests work.",
        "What is XGBoost and why does it win competitions?",
        "Explain the concept of word embeddings (Word2Vec, GloVe).",
        "What is BERT? Explain the masked language model objective.",
        "What is a GAN? Explain the generator-discriminator dynamic.",
        "Explain the transformer architecture with Q, K, V matrices.",
        "What is reinforcement learning? Define policy, reward, value function.",
        "Explain Q-learning with an example.",
        "What is the Bellman equation?",
        "Implement a simple genetic algorithm.",
        "What is A* search? How does the heuristic affect it?",
        "Explain the minimax algorithm with alpha-beta pruning.",
        "What is a zero-knowledge proof? Give an intuitive example.",
        "Explain the P vs NP problem in simple terms.",
        "What is the Halting Problem? Why is it undecidable?",
        "Explain the Church-Turing thesis.",
        "What is a Turing machine? Define formally.",
        "Implement a simple regex engine with NFA simulation.",
        "What is a parser combinator? Give a Python example.",
        "Explain the difference between interpreted and compiled languages.",
        "What is a JIT compiler? How does PyPy work?",
        "Explain virtual memory and paging.",
        "What is a cache? Explain L1/L2/L3 and cache coherence.",
        "Explain the difference between processes and threads.",
        "What is a deadlock? State the four necessary conditions.",
    ],
    "science": [
        "Explain photosynthesis at the molecular level, including the Calvin cycle.",
        "State Newton's three laws of motion with examples.",
        "Describe DNA replication step by step, including enzymes involved.",
        "What is the photoelectric effect? How did it prove light is quantized?",
        "Explain evolution by natural selection with at least three examples.",
        "What is the greenhouse effect? Explain the molecular mechanism.",
        "How does a nuclear reactor generate electricity? PWR vs BWR.",
        "What is quantum entanglement? Why did Einstein call it spooky?",
        "Explain how vaccines train the immune system (innate + adaptive).",
        "What is the difference between nuclear fission and fusion?",
        "What is the Heisenberg uncertainty principle? Give the formula.",
        "Explain Maxwell's equations in simple terms.",
        "What is entropy? Why does it always increase?",
        "Explain the double-slit experiment and wave-particle duality.",
        "What is dark matter? What evidence supports its existence?",
        "Explain the theory of relativity (special and general).",
        "How does CRISPR-Cas9 gene editing work?",
        "What is the central dogma of molecular biology?",
        "Explain how neurons transmit signals (action potential).",
        "What are stem cells? Types and medical applications.",
        "How does the immune system distinguish self from non-self?",
        "What is epigenetics? How does it differ from genetics?",
        "Explain the process of mitosis and meiosis with differences.",
        "What is the pH scale? Why is it logarithmic?",
        "Explain how batteries store and release energy (electrochemistry).",
        "What is a catalyst? How does it lower activation energy?",
        "Explain the difference between ionic, covalent, and metallic bonds.",
        "What are isotopes? Give examples and applications (carbon dating).",
        "How do black holes form? What is the event horizon?",
        "What is the Higgs boson? Why is it called the God particle?",
        "Explain the Big Bang theory and its evidence (CMB, redshift).",
        "What is dark energy? How was it discovered?",
        "Explain how stars form, live, and die (stellar evolution).",
        "What is a supernova? Types and what they produce.",
        "Describe the structure of the Earth (crust, mantle, core).",
        "What causes earthquakes? Explain plate tectonics.",
        "How does the water cycle work? Describe all stages.",
        "What is the carbon cycle? Why is it important for climate?",
        "Explain how the ozone layer forms and what depletes it.",
        "What is acid rain? How does it form and what are its effects?",
        "How do antibiotics work? Why is resistance a problem?",
        "What is a virus? How does it differ from bacteria?",
        "Explain the process of protein synthesis (transcription + translation).",
        "What are enzymes? Describe the lock-and-key model.",
        "How does the human circulatory system work?",
        "What is the function of the liver? List at least 5 functions.",
        "Explain how the kidney filters blood.",
        "What is the endocrine system? List major hormones.",
        "How does the nervous system transmit information?",
        "What is a synapse? Explain neurotransmitter release.",
        "How do muscles contract? Explain the sliding filament theory.",
        "What is ATP? How is it produced in cellular respiration?",
        "Explain the difference between aerobic and anaerobic respiration.",
        "What is fermentation? Give examples (alcohol, lactic acid).",
        "How does the human eye work? Rods, cones, and visual processing.",
        "What is the Doppler effect? Give examples in sound and light.",
        "Explain how a laser works (stimulated emission).",
        "What is a semiconductor? Explain doping and p-n junctions.",
        "How does a transistor work? Explain the basic principle.",
        "What is superconductivity? Explain the Meissner effect.",
        "Explain the concept of a field in physics (gravitational, electric, magnetic).",
        "What is the standard model of particle physics? List the particles.",
        "Explain the strong nuclear force and color charge.",
        "What is antimatter? What happens when it meets matter?",
        "How does radiocarbon dating work? Derive the formula.",
        "What is half-life? Give examples of radioactive decay chains.",
        "Explain how a mass spectrometer works.",
        "What is chromatography? Describe gas chromatography.",
        "How does NMR spectroscopy work? Basic principle.",
        "What is a polymerase chain reaction (PCR)? Steps and applications.",
        "Explain gel electrophoresis for DNA separation.",
        "What is the scientific method? Give the steps with an example.",
        "Explain the difference between correlation and causation.",
        "What is a controlled experiment? Design one for a drug trial.",
        "What is peer review and why is it important?",
        "Explain statistical significance and p-values.",
        "What is a null hypothesis? Give an example.",
        "What is the placebo effect? How do you control for it?",
        "Explain the difference between accuracy and precision.",
        "What is a systematic error vs random error?",
        "Define a scientific theory vs a hypothesis vs a law.",
        "What is Occam's razor? Give a scientific example.",
        "Explain how the speed of light was first measured.",
        "What is the Michelson-Morley experiment? What did it prove?",
        "How was the structure of DNA discovered? Watson, Crick, Franklin.",
        "What is the Miller-Urey experiment? What did it show?",
        "Explain how vaccines were first developed (Jenner, smallpox).",
        "What is the history of the periodic table? Mendeleev's contribution.",
        "How did Galileo's experiments challenge Aristotelian physics?",
        "What is the Copernican revolution?",
        "Explain the significance of Darwin's voyage on the HMS Beagle.",
        "What was the Manhattan Project? Scientific and ethical implications.",
        "How did the Apollo program advance materials science?",
        "What did the Human Genome Project accomplish?",
        "Explain the discovery of penicillin by Alexander Fleming.",
        "What is the history of computing? From Babbage to Turing to modern CPUs.",
    ],
}

# Fill remaining domains from what we have, adapted
TRAIN_DATA["logic"] = TRAIN_DATA["math"][:60] + [
    "Explain the Prisoner's Dilemma and its Nash equilibrium.",
    "What is modus ponens? Give three examples.",
    "Prove by contradiction: there is no largest prime number.",
    "Solve: 8 identical coins, one lighter, find it in 2 weighings.",
    "What is Godel's incompleteness theorem in simple terms?",
    "Explain deductive vs inductive reasoning with examples.",
    "What is a logical fallacy? Name and explain five types.",
    "If all A are B and all B are C, what about A and C? Prove it.",
    "Explain the Monty Hall problem. Why is switching optimal?",
    "What is a sound argument vs a valid argument?",
    "Explain the liar paradox: This statement is false.",
    "What is Occam's razor? Give a logical formulation.",
    "Explain the concept of a counterexample in proofs.",
    "What is mathematical induction? When can you use it?",
    "Prove that the square root of 2 is irrational.",
    "What is the difference between necessary and sufficient conditions?",
    "Explain Russell's paradox in set theory.",
    "What is a tautology? Give three examples.",
    "Explain the difference between syntax and semantics in logic.",
    "What is a formal system? Define alphabet, axioms, inference rules.",
    "What is the compactness theorem in first-order logic?",
    "Explain Skolem's paradox.",
    "What is a Venn diagram? Use for syllogisms.",
    "Prove that there are uncountably many real numbers (Cantor diagonal).",
    "What is the axiom of choice? Why is it controversial?",
    "Explain the continuum hypothesis.",
    "What is a Boolean algebra? Give the axioms.",
    "Prove De Morgan's laws in propositional logic.",
    "What is resolution in automated theorem proving?",
    "Explain the DPLL algorithm for SAT solving.",
    "What is a SAT problem? Why is 3-SAT NP-complete?",
    "Explain the concept of logical consequence.",
    "What is natural deduction? Give an example proof tree.",
    "Explain the difference between classical and intuitionistic logic.",
    "What is modal logic? Explain necessity and possibility operators.",
    "What is temporal logic? When is it used?",
    "Explain fuzzy logic with an example.",
    "What is paraconsistent logic?",
    "Explain Curry-Howard correspondence between proofs and programs.",
    "What is a BHK interpretation of intuitionistic logic?",
]
TRAIN_DATA["creative"] = TRAIN_DATA["science"][:60] + [
    "Write a 100-word flash fiction about a robot learning to paint.",
    "Write a haiku about debugging at 3 AM.",
    "Describe a sunset using only scientific terminology.",
    "Create a metaphor for how the internet works.",
    "Write the opening paragraph of a mystery novel set in Copenhagen.",
    "What makes a story emotionally compelling? Analyze structure.",
    "Describe the color blue to someone who has never seen it.",
    "Write a dialogue between a mathematician and a poet about beauty.",
    "Create a world where music is the primary form of communication.",
    "What is the role of conflict in storytelling? Types of conflict.",
    "Write a sonnet about the heat death of the universe.",
    "Write a villanelle about recursion.",
    "Write free verse about what the internet dreams of.",
    "Create a magic system based on information theory.",
    "Design an alien species that communicates through scent.",
    "Build a world where emotions are physically visible as colors.",
    "Describe a character through their search history alone.",
    "Write a story where gravity slowly stops working.",
    "Write a story from the perspective of a cat watching its human work.",
    "A time traveler accidentally reveals future tech in 1920. What happens?",
    "Write a story that starts: The last library on Earth had one visitor per day.",
    "Two AI systems fall in love through data packets. Write their story.",
    "Write a product description for Hope as if on Amazon.",
    "Explain how to make a sandwich in the style of Shakespeare.",
    "Describe the same room in noir detective, romantic, and horror styles.",
    "What is the hero's journey? Map it to Star Wars.",
    "What is Chekhov's gun? Give examples of good and bad setup/payoff.",
    "Explain the three-act structure with an example.",
    "What is an anti-hero? Give famous examples.",
    "What is a MacGuffin? Famous examples in film.",
    "Explain the difference between plot and story.",
    "What is stream of consciousness writing?",
    "What is magical realism? How does it differ from fantasy?",
    "Explain what makes dialogue sound natural vs stilted.",
    "What is subtext? Give an example of dialogue heavy with subtext.",
    "What is a unreliable narrator? Give famous examples.",
    "Explain foreshadowing with examples from literature.",
    "What is irony? Types (verbal, situational, dramatic) with examples.",
    "What is the difference between a metaphor and a simile?",
    "Explain alliteration, assonance, and consonance with examples.",
]
TRAIN_DATA["general"] = TRAIN_DATA["math"][:40] + TRAIN_DATA["code"][:40] + [
    "What is the capital of Denmark?",
    "Who wrote Hamlet?",
    "Explain how the internet works from first principles.",
    "What is the difference between weather and climate?",
    "How does a combustion engine work?",
    "What is GDP? How is it measured?",
    "Explain supply and demand with an example.",
    "What is the UN Declaration of Human Rights?",
    "How does a democracy differ from a republic?",
    "What is climate change? What causes it?",
    "Explain how a vaccine works.",
    "What is blockchain technology?",
    "How does a plane fly? Explain lift, drag, thrust, weight.",
    "What is the periodic table organized by?",
    "Explain the water cycle.",
    "What causes the seasons?",
    "How does a microwave oven heat food?",
    "What is CRISPR?",
    "Explain how WiFi works.",
    "What is the difference between HTTP and HTTPS?",
]

SAIYAN_DOMAINS = {
    "Goku": "math", "Vegeta": "code", "Gohan": "science",
    "Piccolo": "logic", "Trunks": "creative", "Yamcha": "general",
}

# ============================================================================
# JURY ENGINE
# ============================================================================
class Jury:
    def __init__(self, trajectories, K, perturbation=0.04):
        self.trajs = trajectories
        self.K = K
        self.pert = perturbation
    
    @property
    def cr(self):
        if len(self.trajs) < 5: return 0.5
        projs = F.normalize(torch.stack([t["proj"].float() for t in self.trajs]), dim=1)
        sims = projs @ projs.T
        cd = 1 - sims
        n = len(self.trajs)
        idx = torch.triu_indices(n, n, offset=1)
        return max(0.05, cd[idx[0], idx[1]].median().item())
    
    def ask_naive(self, q_k, n_trials=7):
        return self._ask(q_k, n_trials, {})
    
    def ask_zone_weighted(self, q_k, zone_weights, n_trials=7):
        return self._ask(q_k, n_trials, zone_weights)
    
    def _ask(self, q_k, n_trials, zone_weights):
        if not self.trajs: return {"jury": 0.0, "agree": 0.0}
        R = self.cr
        individual, seen = [], {}
        for _ in range(n_trials):
            qp = F.normalize((q_k.float() + torch.randn(self.K)*self.pert).unsqueeze(0), dim=1).squeeze(0)
            qn = F.normalize(qp.unsqueeze(0), dim=1)
            best_score, best_idx, best_sim = -float('inf'), 0, 0.0
            for i, t in enumerate(self.trajs):
                tp = F.normalize(t["proj"].unsqueeze(0).float(), dim=1)
                sim = (tp @ qn.T).item()
                w = zone_weights.get(t.get("parent",""), 0.5) if zone_weights else 1.0
                ws = sim * w
                if ws > best_score: best_score, best_idx, best_sim = ws, i, sim
            c = math.exp(-(1-best_sim)/R) if R>0 else 0.5
            individual.append(c)
            lbl = self.trajs[best_idx].get("label","")
            seen[lbl] = seen.get(lbl,0)+1
        best_label = max(seen,key=seen.get) if seen else ""
        ag = seen.get(best_label,0)/n_trials if best_label else 0.0
        pw = 1.0
        for c in individual: pw *= max(0.0001, 1-c)
        jury = min(1.0, (1-pw)*(0.5+0.5*ag))
        return {"jury": round(jury,4), "agree": round(ag,4),
                "single_avg": round(sum(individual)/len(individual),4)}

# ============================================================================
# SIMPLE BENCHMARK QUESTIONS (10 per domain, 6 domains)
# ============================================================================
BENCH_QUESTIONS = {
    "math": ["Solve 3x+7=22", "Derivative of x^3*sin(x)", "Roots of x^2-5x+6=0",
             "Prove sqrt(2) irrational", "Sum 1..100", "det([[2,1],[3,4]])",
             "Integral 1/x 1..e", "sin(pi/6)", "Permutations of 5 items", "2x+y=7,x-y=1"],
    "code": ["Binary search complexity", "Quicksort time complexity", "Hash table explanation",
             "Reverse linked list", "BFS vs DFS", "LRU cache implementation",
             "O(n log n) meaning", "CAP theorem", "Race condition prevention", "REST vs GraphQL"],
    "science": ["Photosynthesis process", "Newton's second law", "DNA replication",
                "Photoelectric effect", "Natural selection", "Greenhouse effect",
                "Nuclear reactor", "Quantum entanglement", "Vaccine mechanism", "Fission vs fusion"],
    "logic": ["Monty Hall problem", "Five logical fallacies", "Modus ponens examples",
              "Prisoner's dilemma", "Godel's theorem", "Deductive vs inductive",
              "8 coins 2 weighings", "Necessary vs sufficient", "Liar paradox", "Cantor diagonal"],
    "creative": ["Robot learning to paint (story)", "Haiku about debugging",
                 "Sunset in scientific terms", "Metaphor for internet", "Mystery novel opening",
                 "Describe blue to blind person", "Mathematician-poet dialogue", "Music communication world",
                 "Conflict in storytelling", "Sonnet about heat death"],
    "general": ["Capital of Denmark", "Who wrote Hamlet", "How internet works",
                "Weather vs climate", "Combustion engine", "Supply and demand",
                "UN Human Rights", "Democracy vs republic", "Climate change causes", "Vaccine mechanism"],
}

# ============================================================================
# MAIN
# ============================================================================
print(f"\n[1] Loading {MODEL_ID} (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                          bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

d_model = model.config.hidden_size
vram = torch.cuda.memory_allocated()/1e9 if DEVICE=="cuda" else 0
print(f"  d={d_model}, VRAM used={vram:.1f}GB")

# Build UGT basis
print(f"\n[2] Building UGT basis (K={K_TARGET})...")
# Pool all training prompts for calibration to get K=512 SVD output
all_cal_prompts = []
for domain_prompts in TRAIN_DATA.values():
    all_cal_prompts.extend(domain_prompts)
random.shuffle(all_cal_prompts)
cal_prompts = all_cal_prompts[:K_TARGET]
print(f"  Pooling {len(all_cal_prompts)} prompts from all domains for calibration, using {len(cal_prompts)}")
hidden_states = []
for p in cal_prompts:
    enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad(): out = model(**enc, output_hidden_states=True)
    hidden_states.append(out.hidden_states[-1][0, -1, :].float().cpu())

hs = torch.stack(hidden_states)
U, S, _ = torch.linalg.svd(hs.T, full_matrices=False)
K = min(K_TARGET, U.shape[1])
basis = U[:, :K].float()
Q, _ = torch.linalg.qr(basis)
basis = Q
print(f"  Basis: [{d_model}, {K}]")

def to_k(h): return (h.float().cpu() @ basis).squeeze(0)

# Build Saiyan manifolds
print(f"\n[3] Building Saiyan manifolds ({N_PROMPTS_PER_SAIYAN} prompts each)...")
saiyan_trajs = {}
saiyan_judges = {}

for name, domain in SAIYAN_DOMAINS.items():
    prompts = TRAIN_DATA.get(domain, TRAIN_DATA["general"])[:N_PROMPTS_PER_SAIYAN]
    trajs = []
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad(): out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        trajs.append({"proj": to_k(h), "label": f"{domain}:{i}", "parent": name, "domain": domain})
    saiyan_trajs[name] = trajs
    saiyan_judges[name] = Jury(trajs, K)
    # Compute coverage radius
    print(f"  {name:10s} ({domain:10s}): {len(trajs)} traj, R={saiyan_judges[name].cr:.4f}")

# Verify domain separation
print(f"\n[4] Domain separation check...")
centroids = {}
for name, trajs in saiyan_trajs.items():
    projs = torch.stack([t["proj"].float() for t in trajs])
    centroids[name] = F.normalize(projs.mean(dim=0).unsqueeze(0), dim=1).squeeze(0)

for n1 in sorted(SAIYAN_DOMAINS):
    for n2 in sorted(SAIYAN_DOMAINS):
        if n1 < n2:
            sim = F.cosine_similarity(centroids[n1].unsqueeze(0), centroids[n2].unsqueeze(0)).item()
            sep = "✓ SEPARATED" if sim < 0.7 else ("~ PARTIAL" if sim < 0.9 else "✗ OVERLAPPING")
            print(f"  {n1:10s}↔{n2:10s}: cos_sim={sim:.4f} {sep}")

# Build fusions
print(f"\n[5] Building fusions...")

# Gogeta: Goku + Vegeta (all trajectories, zone-weighted)
gogeta_all_trajs = saiyan_trajs["Goku"] + saiyan_trajs["Vegeta"]
gogeta_naive = Jury(gogeta_all_trajs, K)
gogeta_zone_weights = {"Goku": 0.5, "Vegeta": 0.5}  # will be overridden per-query

# Vegito: Goku + Vegeta (different blend)
vegito_all_trajs = gogeta_all_trajs
vegito_naive = Jury(vegito_all_trajs, K)

# Gotenks: Trunks + Piccolo
gotenks_all_trajs = saiyan_trajs["Trunks"] + saiyan_trajs["Piccolo"]
gotenks_naive = Jury(gotenks_all_trajs, K)

# ============================================================================
# BENCHMARK: Naive vs Zone-Weighted
# ============================================================================
print(f"\n[6] BENCHMARK: Naive vs Zone-Weighted Fusion...")

# Zone detection function
def detect_zone(q_k, centroids_dict, temp=3.0):
    q = F.normalize(q_k.unsqueeze(0).float(), dim=1)
    weights = {}
    total = 0.0
    for name, cent in centroids_dict.items():
        c = F.normalize(cent.unsqueeze(0), dim=1)
        sim = (c @ q.T).item()
        w = math.exp(sim * temp)
        weights[name] = w
        total += w
    for n in weights: weights[n] /= total
    return weights

results = []

for domain_name, questions in BENCH_QUESTIONS.items():
    print(f"\n  Domain: {domain_name}")
    
    for q_text in questions[:6]:  # 6 questions per domain
        enc = tok(q_text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad(): out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        q_k = to_k(h)
        
        # Detect zone
        zone_w = detect_zone(q_k, centroids, temp=3.0)
        dominant = max(zone_w, key=zone_w.get)
        
        # Naive fusions
        gogeta_n = gogeta_naive.ask_naive(q_k, JURY_TRIALS)
        vegito_n = vegito_naive.ask_naive(q_k, JURY_TRIALS)
        gotenks_n = gotenks_naive.ask_naive(q_k, JURY_TRIALS)
        
        # Zone-weighted fusions
        gogeta_zw = gogeta_naive.ask_zone_weighted(q_k, zone_w, JURY_TRIALS)
        vegito_zw = vegito_naive.ask_zone_weighted(q_k, zone_w, JURY_TRIALS)
        gotenks_zw = gotenks_naive.ask_zone_weighted(q_k, zone_w, JURY_TRIALS)
        
        # Parent scores
        parent_scores = {}
        for name, judge in saiyan_judges.items():
            parent_scores[name] = judge.ask_naive(q_k, JURY_TRIALS)
        
        results.append({
            "domain": domain_name, "query": q_text[:60],
            "dominant_zone": dominant,
            "zone_weights": {n: round(w,3) for n,w in zone_w.items()},
            "parents": {n: r["jury"] for n,r in parent_scores.items()},
            "gogeta_naive": gogeta_n["jury"],
            "gogeta_zone_weighted": gogeta_zw["jury"],
            "vegito_naive": vegito_n["jury"],
            "vegito_zone_weighted": vegito_zw["jury"],
            "gotenks_naive": gotenks_n["jury"],
            "gotenks_zone_weighted": gotenks_zw["jury"],
        })
        
        g_imp = gogeta_zw["jury"] - gogeta_n["jury"]
        print(f"    {q_text[:40]:40s} zone={dominant:8s} naive={gogeta_n['jury']:.3f} zw={gogeta_zw['jury']:.3f} Δ={g_imp:+.3f}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"  FUSION COMPARISON SUMMARY")
print(f"{'='*70}")

# Aggregate by domain
domain_summary = defaultdict(lambda: {"naive": [], "zw": [], "parent_best": []})
for r in results:
    d = r["domain"]
    domain_summary[d]["naive"].append(r["gogeta_naive"])
    domain_summary[d]["zw"].append(r["gogeta_zone_weighted"])
    parent_scores = list(r["parents"].values())
    domain_summary[d]["parent_best"].append(max(parent_scores))

print(f"\n  {'Domain':12s} {'Naive':>8s} {'Zone-W':>8s} {'Best Parent':>12s} {'ZW vs Naive':>12s} {'ZW vs Parent':>12s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for d in sorted(domain_summary):
    navg = sum(domain_summary[d]["naive"])/len(domain_summary[d]["naive"])
    zwavg = sum(domain_summary[d]["zw"])/len(domain_summary[d]["zw"])
    pavg = sum(domain_summary[d]["parent_best"])/len(domain_summary[d]["parent_best"])
    delta_n = zwavg - navg
    delta_p = zwavg - pavg
    print(f"  {d:12s} {navg:>8.4f} {zwavg:>8.4f} {pavg:>12.4f} {delta_n:>+12.4f} {delta_p:>+12.4f}")

# Overall
all_naive = [r["gogeta_naive"] for r in results]
all_zw = [r["gogeta_zone_weighted"] for r in results]
all_parent = [max(r["parents"].values()) for r in results]
avg_n = sum(all_naive)/len(all_naive)
avg_zw = sum(all_zw)/len(all_zw)
avg_p = sum(all_parent)/len(all_parent)
print(f"\n  OVERALL: Naive={avg_n:.4f}, Zone-Weighted={avg_zw:.4f}, Best Parent={avg_p:.4f}")
print(f"  Zone-Weighted improvement: {avg_zw-avg_n:+.4f} over naive ({((avg_zw/avg_n)-1)*100:+.1f}%)")
print(f"  Zone-Weighted vs parent:   {avg_zw-avg_p:+.4f} (ZW {'>' if avg_zw>avg_p else '<' if avg_zw<avg_p else '='} parent)")

# ============================================================================
# SAVE
# ============================================================================
with open(OUT / "ec2_fusion_results.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID, "K": K, "device": DEVICE,
        "n_trials": JURY_TRIALS, "n_results": len(results),
        "overall": {"naive": round(avg_n,4), "zone_weighted": round(avg_zw,4), "best_parent": round(avg_p,4)},
        "by_domain": {d: {"naive": round(sum(v["naive"])/len(v["naive"]),4),
                           "zone_weighted": round(sum(v["zw"])/len(v["zw"]),4),
                           "best_parent": round(sum(v["parent_best"])/len(v["parent_best"]),4)}
                       for d, v in domain_summary.items()},
        "details": results[:20],
    }, f, indent=2)

print(f"\n  Results saved to {OUT / 'ec2_fusion_results.json'}")
print(f"\n{'='*70}")
print(f"  EC2 SCALE BENCHMARK COMPLETE")
print(f"{'='*70}")
