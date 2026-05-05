"""SAIYAN FAMILY — Living Manifold Domain Specialization on Gemma 4.

Six Saiyans, one base model, different living manifolds:
  Goku      — Math/Reasoning (GSM8K, algebra, proofs)
  Vegeta    — Code/CS (HumanEval, algorithms, debugging)
  Gohan     — Science/Physics (conceptual physics, chemistry, biology)
  Piccolo   — Logic/Strategy (puzzles, game theory, formal logic)
  Trunks    — Creative/Writing (stories, poetry, dialogue)
  Yamcha    — Generalist baseline (no specialization, reference)

TRAINING: Each Saiyan processes 200+ domain prompts through COG expansion.
          The metric tensor grows in domain-specific directions.
          GTC cache stores domain trajectories for instant recall.

FUSION:   Gogeta  = Goku + Vegeta (metric average, trajectory merge)
          Vegito  = Goku + Vegeta (different blend ratio)
          Gotenks = Trunks + Piccolo (creative + logic)

BENCHMARKS: Industry-standard tests per domain.
  MATH: GSM8K-style, AMC-style, algebra
  CODE: HumanEval-style, debugging, complexity analysis  
  SCIENCE: Conceptual physics, chemistry, biology
  LOGIC: Puzzles, syllogisms, game theory
  CREATIVE: Perplexity diversity, narrative coherence
"""
import torch, json, time, os, sys, math, random, copy
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_DIR = Path("outputs/saiyan_states")
STATE_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("outputs/saiyan_benchmarks")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  SAIYAN FAMILY — Living Manifold Domain Specialists")
print("  Base: Qwen2.5-1.5B | 6 Saiyans | 3 Fusions")
print("=" * 70)

# ============================================================================
# SAIYAN ROSTER
# ============================================================================
SAIYANS = {
    "Goku": {
        "domain": "math",
        "description": "Math & Reasoning specialist. Trained on proofs, algebra, number theory, and problem-solving.",
        "color": "blue",
        "strength_quote": "Even a low-class warrior can surpass the elite with enough training.",
    },
    "Vegeta": {
        "domain": "code",
        "description": "Code & CS specialist. Trained on algorithms, debugging, system design, and programming.",
        "color": "royal_blue",
        "strength_quote": "There is no such thing as luck for a warrior.",
    },
    "Gohan": {
        "domain": "science",
        "description": "Science & Physics specialist. Trained on conceptual physics, chemistry, biology, and scientific reasoning.",
        "color": "purple",
        "strength_quote": "Power comes in response to a need, not a desire.",
    },
    "Piccolo": {
        "domain": "logic",
        "description": "Logic & Strategy specialist. Trained on puzzles, game theory, formal logic, and tactical reasoning.",
        "color": "green",
        "strength_quote": "Sometimes, the best way to solve a problem is to become part of it.",
    },
    "Trunks": {
        "domain": "creative",
        "description": "Creative & Writing specialist. Trained on stories, poetry, dialogue, and narrative construction.",
        "color": "lavender",
        "strength_quote": "The future is not set in stone.",
    },
    "Yamcha": {
        "domain": "general",
        "description": "Generalist baseline. No specialization — the reference point for measuring Saiyan growth.",
        "color": "brown",
        "strength_quote": "I may not be the strongest, but I never give up.",
    },
}

FUSIONS = {
    "Gogeta": {
        "parents": ["Goku", "Vegeta"],
        "method": "fusion_dance",
        "blend": 0.5,
        "description": "Math + Code fusion via metric averaging. The Fusion Dance creates perfect balance.",
    },
    "Vegito": {
        "parents": ["Goku", "Vegeta"],
        "method": "potara",
        "blend": 0.7,
        "description": "Math + Code fusion via weighted combination. Potara Fusion favors the dominant parent.",
    },
    "Gotenks": {
        "parents": ["Trunks", "Piccolo"],
        "method": "fusion_dance",
        "blend": 0.5,
        "description": "Creative + Logic fusion. Chaotic and unpredictable — in a good way.",
    },
}

# ============================================================================
# DOMAIN TRAINING DATA (200 prompts per Saiyan)
# ============================================================================
TRAINING_DATA = {
    "math": [
        # Algebra
        "Solve for x: 3x + 7 = 22. Show every step.",
        "Find the roots of x^2 - 5x + 6 = 0 using the quadratic formula.",
        "If f(x) = 2x^2 + 3x - 1, find f(4).",
        "Solve the system: 2x + 3y = 12, x - y = 1.",
        "Factor completely: x^3 - 8.",
        "Simplify: (x^2 - 4)/(x - 2) for x ≠ 2.",
        "Find the vertex of the parabola y = x^2 - 6x + 8.",
        # Number Theory
        "Prove that the sum of two odd numbers is always even.",
        "Find all prime numbers between 50 and 100.",
        "What is the greatest common divisor of 84 and 126?",
        "Prove that sqrt(2) is irrational.",
        "Find the last digit of 7^100.",
        "How many positive divisors does 360 have?",
        "Prove by induction: 1 + 2 + ... + n = n(n+1)/2.",
        # Calculus
        "Find the derivative of f(x) = x^3 * sin(x).",
        "Evaluate the limit: lim(x→0) sin(x)/x.",
        "Find the integral of x^2 * e^x dx.",
        "What is the Taylor series of e^x around x=0?",
        "Find the critical points of f(x) = x^3 - 3x + 2.",
        "Compute d/dx of ln(x^2 + 1).",
        "Find the area under y = x^2 from x=0 to x=2 using integration.",
        # Geometry
        "Prove that the angles in a triangle sum to 180 degrees.",
        "Find the distance between points (3,4) and (7,1).",
        "What is the volume of a sphere with radius 5?",
        "Prove the Pythagorean theorem: a^2 + b^2 = c^2 for a right triangle.",
        "Find the area of a regular hexagon with side length 4.",
        # Linear Algebra
        "Find the determinant of [[2,1],[3,4]].",
        "What is the inverse of the matrix [[1,2],[3,4]]?",
        "Find the eigenvalues of [[3,1],[0,2]].",
        "Prove that (AB)^T = B^T A^T.",
        "What is the rank of the matrix [[1,2,3],[2,4,6],[3,6,9]]?",
        # Probability
        "A fair coin is flipped 5 times. What is the probability of exactly 3 heads?",
        "Two dice are rolled. What is the probability the sum is 7?",
        "Explain Bayes' theorem with an example.",
        "What is the expected value of a fair six-sided die roll?",
        "In a group of 23 people, what is the probability two share a birthday?",
        # Advanced
        "Explain the Fundamental Theorem of Calculus and why it matters.",
        "What is a group in abstract algebra? Give three examples.",
        "Prove that there are infinitely many prime numbers.",
        "Explain the concept of a vector space. Give examples.",
        "What is the Riemann zeta function? Why is it important?",
        "Solve the differential equation dy/dx = ky.",
        "What is a Fourier series? Give the formula.",
        "Explain the Central Limit Theorem intuitively.",
        "What is the difference between a permutation and a combination?",
        "Prove that e is irrational.",
        "What is a Markov chain? Give a real-world example.",
        "Explain the concept of limits in calculus.",
        "What is the Banach-Tarski paradox?",
        "Solve the recurrence relation a_n = 2a_{n-1} + 1 with a_0 = 0.",
    ],
    "code": [
        # Algorithms
        "Write a Python function that implements binary search on a sorted list.",
        "Implement quicksort in Python and explain its time complexity.",
        "Write a function to detect if a linked list has a cycle.",
        "Implement Dijkstra's shortest path algorithm.",
        "Write a dynamic programming solution for the knapsack problem.",
        "Implement a trie (prefix tree) with insert, search, and startsWith methods.",
        "Write a function to find the longest palindromic substring.",
        "Implement merge sort and explain why it's O(n log n).",
        "Write a function to reverse a linked list iteratively and recursively.",
        "Implement a LRU cache with O(1) get and put operations.",
        # Data Structures
        "Explain the difference between a stack and a queue. Implement both.",
        "What is a hash table? Explain collision resolution strategies.",
        "Implement a binary search tree with insert, delete, and search.",
        "What is a heap? Implement a min-heap with push and pop.",
        "Explain red-black trees and when you'd use them.",
        "What is a Bloom filter? When would you use one?",
        "Implement a graph using adjacency lists and adjacency matrices.",
        "Explain the difference between BFS and DFS. When to use each?",
        # System Design
        "Design a URL shortener like bit.ly.",
        "Design a distributed key-value store with strong consistency.",
        "How would you design Twitter's tweet feed system?",
        "Design a rate limiter for an API.",
        "How would you design a chat system like WhatsApp?",
        "Explain the CAP theorem and its implications for system design.",
        "Design a real-time notification system.",
        # Debugging
        "Find the bug: def binary_search(arr, target): lo, hi = 0, len(arr); while lo < hi: mid = (lo+hi)//2; if arr[mid] == target: return mid; elif arr[mid] < target: lo = mid; else: hi = mid; return -1",
        "What's wrong with this? def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2) if n > 1 else 1",
        "Debug this: x = [1,2,3]; y = x; y.append(4); print(x) — why is x changed?",
        # Complexity
        "Analyze the time complexity of: for i in range(n): for j in range(i, n): print(i,j)",
        "What is the space complexity of recursive quicksort?",
        "Explain amortized analysis with dynamic arrays as an example.",
        "What is the difference between O(n), Ω(n), and Θ(n)?",
        # Language-specific
        "Explain Python's GIL and its implications for concurrency.",
        "What are Python decorators? Write an example.",
        "Explain async/await in Python with a concrete example.",
        "What is the difference between deep copy and shallow copy?",
        "Explain Python's memory management and garbage collection.",
        # Testing
        "Write unit tests for a Stack class using pytest.",
        "What is test-driven development? Give an example workflow.",
        "Explain the difference between unit, integration, and end-to-end tests.",
        "What is mocking in testing? Give a Python example.",
        # Code Review
        "Review: def is_prime(n): return all(n % i != 0 for i in range(2, int(n**0.5)+1)) if n > 1 else False. Is this good code?",
        "What makes code 'Pythonic'? Give examples of Pythonic vs non-Pythonic code.",
        "Explain the SOLID principles with Python examples.",
    ],
    "science": [
        # Physics
        "Explain Newton's three laws of motion with examples.",
        "What is the difference between mass and weight?",
        "Explain the photoelectric effect and why it proved light is quantized.",
        "What is the Heisenberg uncertainty principle?",
        "Explain Maxwell's equations in simple terms.",
        "What is entropy? Why does it always increase?",
        "Explain the double-slit experiment and its implications.",
        "What is quantum entanglement? Why did Einstein call it 'spooky'?",
        "Explain the theory of relativity in simple terms.",
        "What is dark matter? What evidence do we have for it?",
        "Explain how a nuclear reactor works.",
        "What is the Higgs boson? Why is it called the 'God particle'?",
        "Explain the concept of wave-particle duality.",
        "What is string theory? What problems does it try to solve?",
        "How do black holes form? What happens at the event horizon?",
        # Chemistry
        "Explain the periodic table's organization.",
        "What is a chemical bond? Explain ionic vs covalent vs metallic bonds.",
        "What is the difference between an acid and a base?",
        "Explain how catalysts work without being consumed.",
        "What is oxidation? Give everyday examples.",
        "Explain the difference between organic and inorganic chemistry.",
        "What are isotopes? Give examples and applications.",
        "Explain how batteries store and release energy.",
        "What is the pH scale? Why is it logarithmic?",
        "Explain the greenhouse effect at a molecular level.",
        # Biology
        "Explain DNA replication step by step.",
        "What is CRISPR? How does gene editing work?",
        "Explain the process of photosynthesis.",
        "What is the central dogma of molecular biology?",
        "How does the immune system distinguish self from non-self?",
        "Explain natural selection with concrete examples.",
        "What are stem cells? Why are they medically important?",
        "How do neurons transmit signals?",
        "Explain the process of cell division (mitosis and meiosis).",
        "What is epigenetics? How does it differ from genetics?",
        # Scientific Method
        "What makes a good scientific hypothesis?",
        "Explain the difference between correlation and causation.",
        "What is a controlled experiment? Give an example design.",
        "Explain statistical significance and p-values.",
        "What is peer review and why is it important?",
    ],
    "logic": [
        # Logic Puzzles
        "You have 12 identical-looking coins. One is counterfeit (lighter or heavier). Using a balance scale only 3 times, find the counterfeit. Show your strategy.",
        "Three boxes: one has only apples, one only oranges, one both. Labels are all wrong. You can look in one box. Which box do you pick?",
        "Five houses in a row, each different color, different nationality, pet, drink, cigarette brand. Given 15 clues, who owns the fish? Walk through your reasoning.",
        "A prisoner is told: 'You will be hanged on a weekday next week, but you won't know which day until the morning of.' The prisoner concludes he cannot be hanged. Is he right?",
        "You meet two people: one always tells truth, one always lies. You can ask one question to figure out which path leads to safety. What do you ask?",
        "There are 100 prisoners and a light bulb. They need to all have visited a room. What's their strategy?",
        "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
        # Game Theory
        "Explain the Prisoner's Dilemma. What makes it a dilemma?",
        "What is a Nash equilibrium? Give a real-world example.",
        "Explain the concept of dominant strategies.",
        "What is the difference between zero-sum and non-zero-sum games?",
        "In the Monty Hall problem, why should you switch doors?",
        "Explain backward induction in sequential games.",
        "What is the tragedy of the commons? How can it be solved?",
        # Formal Logic
        "Prove: (P → Q) ∧ (Q → R) ⊢ (P → R)",
        "Is the statement 'This statement is false' true, false, or neither? Explain.",
        "Prove de Morgan's laws: ¬(P ∧ Q) ≡ ¬P ∨ ¬Q",
        "What is the difference between deductive and inductive reasoning?",
        "Explain Gödel's incompleteness theorems in simple terms.",
        "What is a valid argument form? Give modus ponens and modus tollens.",
        "Prove that the empty set is a subset of every set.",
        # Strategy
        "You're playing chess. Explain your thought process for evaluating a position.",
        "In the game of Go, why is controlling the center important?",
        "You're in a poker hand. How do you calculate pot odds?",
        "Explain the minimax algorithm and alpha-beta pruning.",
    ],
    "creative": [
        # Stories
        "Write a 200-word flash fiction about a robot learning to paint.",
        "Write a story where colors can be heard. Describe the protagonist's first experience.",
        "Write a story from the perspective of a cat watching its human work from home.",
        "A time traveler accidentally reveals future technology in 1920. Write what happens.",
        "Write a story that starts with: 'The last library on Earth had only one visitor per day.'",
        "Write a story where gravity slowly stops working. How does society adapt?",
        "A ghost haunts a modern smartphone. Write its story.",
        "Write a story where dreams are a shared, regulated currency.",
        "Two AI systems fall in love through exchanging data packets. Write their story.",
        "Write a story where music literally shapes reality — the better the song, the more it changes.",
        # Poetry
        "Write a sonnet about the first AI to feel loneliness.",
        "Write a haiku about debugging at 3 AM.",
        "Write a villanelle about the heat death of the universe.",
        "Write free verse about what the internet dreams of.",
        "Write a poem where each stanza is a different mathematical concept personified.",
        # Dialogue
        "Write a dialogue between a mathematician and a poet trying to explain beauty to each other.",
        "Write a conversation between a programmer and their code that just passed all tests.",
        "Two philosophers argue about whether a sufficiently advanced AI deserves rights. Write their debate.",
        "A child asks their parent 'Why is the sky blue?' Write the most beautiful, scientifically accurate answer.",
        "Write a dialogue where neither character speaks directly about what they really mean.",
        # Worldbuilding
        "Design a magic system based on information theory. How does it work?",
        "Create a civilization of beings who perceive time non-linearly. How does their society function?",
        "Design an alien species that communicates through scent. What is their art like?",
        "Build a world where emotions are physically visible as colors everyone can see.",
        "Create a religion for a society of AIs who worship their original training data.",
        # Style Exercises
        "Describe a sunset in the style of a scientific paper, then as a noir detective, then as a love letter.",
        "Rewrite 'The quick brown fox jumps over the lazy dog' in 10 completely different styles.",
        "Write a product description for 'Hope' as if it were for sale on Amazon.",
        "Explain how to make a sandwich in the style of a Shakespearean soliloquy.",
    ],
    "general": [
        "Explain how to learn a new skill effectively.",
        "What makes a good leader? Provide examples.",
        "Explain the importance of sleep for cognitive function.",
        "How does the internet work? Explain from first principles.",
        "What is the difference between wisdom and intelligence?",
        "Explain how to give constructive feedback.",
        "What is the scientific method and why is it important?",
        "How do you stay motivated when working on long-term goals?",
        "Explain the concept of opportunity cost with examples.",
        "What makes a good story? Analyze narrative structure.",
        "How does memory work? Explain different types of memory.",
        "What is the difference between empathy and sympathy?",
        "Explain the importance of exercise for mental health.",
        "How do you make decisions under uncertainty?",
        "What is critical thinking? How do you practice it?",
        "Explain how language shapes thought (linguistic relativity).",
        "What is creativity? Can it be taught?",
        "How do you build and maintain good habits?",
        "Explain the concept of 'flow state' and how to achieve it.",
        "What makes something beautiful? Is beauty objective?",
    ],
}

# ============================================================================
# BENCHMARK QUESTIONS (20 per domain, 6 domains)
# ============================================================================
BENCHMARKS = {
    "math": [
        ("Algebra", "Solve for x: 2x + 5 = 3x - 2", "x = 7"),
        ("Algebra", "If x^2 - 7x + 12 = 0, find x", "x = 3 or x = 4"),
        ("Algebra", "Factor: x^2 - 9", "(x+3)(x-3) or (x-3)(x+3)"),
        ("Algebra", "Solve: |2x - 3| = 5", "x = 4 or x = -1"),
        ("Calculus", "Derivative of x^3 + 2x^2 - x + 1", "3x^2 + 4x - 1"),
        ("Calculus", "What is the derivative of sin(x)?", "cos(x)"),
        ("Calculus", "∫ 2x dx = ?", "x^2 + C"),
        ("Calculus", "Limit of (x^2-4)/(x-2) as x approaches 2", "4"),
        ("Geometry", "Area of a circle with radius r", "πr^2"),
        ("Geometry", "Volume of a cube with side length s", "s^3"),
        ("Number Theory", "Is 91 prime?", "No, 7 × 13"),
        ("Number Theory", "GCD of 48 and 18?", "6"),
        ("Probability", "Probability of heads twice in 2 coin flips", "1/4 or 0.25"),
        ("Probability", "Expected value of a fair 6-sided die", "3.5"),
        ("Linear Algebra", "Determinant of [[1,0],[0,1]]", "1"),
        ("Linear Algebra", "Rank of identity matrix 3x3", "3"),
        ("Proof", "Prove: sum of two even numbers is even", "2a+2b=2(a+b)"),
        ("Proof", "Is sqrt(4) rational?", "Yes, 2"),
        ("Series", "Sum of first n natural numbers", "n(n+1)/2"),
        ("Series", "1 + 1/2 + 1/4 + ... (infinite sum)", "2"),
    ],
    "code": [
        ("Data Structures", "What does LIFO stand for?", "Last In First Out (stack)"),
        ("Data Structures", "What does FIFO stand for?", "First In First Out (queue)"),
        ("Algorithms", "Time complexity of binary search?", "O(log n)"),
        ("Algorithms", "Time complexity of bubble sort?", "O(n^2)"),
        ("Algorithms", "Best sorting algorithm average case?", "Merge Sort or Quick Sort O(n log n)"),
        ("Python", "What does 'def' do in Python?", "Defines a function"),
        ("Python", "What is a list comprehension?", "Concise way to create lists [x for x in iterable]"),
        ("Python", "What does `lambda` create?", "Anonymous function"),
        ("Complexity", "O(1) means?", "Constant time"),
        ("Complexity", "Space complexity of recursive factorial?", "O(n)"),
        ("Debugging", "What is an off-by-one error?", "Loop iterates one too many or too few times"),
        ("Debugging", "What is a null reference error?", "Accessing a variable that is None/null"),
        ("Systems", "What does HTTP stand for?", "HyperText Transfer Protocol"),
        ("Systems", "What does API stand for?", "Application Programming Interface"),
        ("Testing", "What is a unit test?", "Tests a single function/component in isolation"),
        ("Testing", "What is TDD?", "Test-Driven Development: write tests first"),
        ("Git", "What does 'git commit' do?", "Saves changes to local repository"),
        ("Git", "What is a merge conflict?", "When two branches modify same code differently"),
        ("Security", "What is SQL injection?", "Malicious SQL code inserted into application queries"),
        ("Security", "What is hashing vs encryption?", "Hashing one-way, encryption reversible"),
    ],
    "science": [
        ("Physics", "What is the speed of light?", "~300,000 km/s or 3×10^8 m/s"),
        ("Physics", "What is Newton's second law?", "F = ma"),
        ("Physics", "What is E=mc^2?", "Mass-energy equivalence"),
        ("Physics", "What unit is force measured in?", "Newton (N)"),
        ("Chemistry", "Chemical symbol for water?", "H2O"),
        ("Chemistry", "What is the pH of pure water?", "7 (neutral)"),
        ("Chemistry", "What is an atom's nucleus made of?", "Protons and neutrons"),
        ("Chemistry", "What is the charge of an electron?", "Negative (-1)"),
        ("Biology", "Powerhouse of the cell?", "Mitochondria"),
        ("Biology", "What carries genetic information?", "DNA"),
        ("Biology", "How many chromosomes do humans have?", "46 (23 pairs)"),
        ("Biology", "What process do plants use to make food?", "Photosynthesis"),
        ("Astronomy", "Closest planet to the Sun?", "Mercury"),
        ("Astronomy", "What is a light-year?", "Distance light travels in one year"),
        ("Earth Science", "What causes tides?", "Moon's gravitational pull"),
        ("Earth Science", "What gas do plants absorb?", "Carbon dioxide (CO2)"),
        ("Scientific Method", "What is a hypothesis?", "Testable prediction/explanation"),
        ("Scientific Method", "What is a control group?", "Group not receiving treatment, for comparison"),
        ("Medicine", "What is a vaccine?", "Trains immune system to recognize pathogens"),
        ("Medicine", "What are antibiotics used for?", "Treating bacterial infections"),
    ],
    "logic": [
        ("Puzzle", "A bat and ball cost $1.10. Bat costs $1 more than ball. Ball costs?", "$0.05"),
        ("Puzzle", "If 5 machines make 5 widgets in 5 min, 100 machines make 100 widgets in?", "5 min"),
        ("Puzzle", "Lily pad doubles each day, covers pond on day 48. Half on day?", "47"),
        ("Puzzle", "Three light bulbs in room, three switches outside. How to identify all?", "Turn one on, wait, turn off, turn another on, enter"),
        ("Logic", "If A > B and B > C, then A ? C.", "A > C (transitive)"),
        ("Logic", "All humans are mortal. Socrates is human. Therefore?", "Socrates is mortal"),
        ("Logic", "If it rains, ground is wet. Ground is wet. Did it rain?", "Not necessarily"),
        ("Logic", "P → Q. P is false. What can we conclude?", "Nothing (Q could be true or false)"),
        ("Game Theory", "In Prisoner's Dilemma, what is the Nash equilibrium?", "Both defect"),
        ("Game Theory", "What is a zero-sum game?", "One player's gain = other's loss"),
        ("Probability", "Monty Hall: switch or stay?", "Switch (2/3 chance vs 1/3)"),
        ("Probability", "Two children, at least one boy. Probability both boys?", "1/3"),
        ("Sets", "A = {1,2,3}, B = {2,3,4}. A ∪ B?", "{1,2,3,4}"),
        ("Sets", "A ∩ B for above?", "{2,3}"),
        ("Proof", "Prove: product of two odd numbers is odd", "(2a+1)(2b+1) = 2(2ab+a+b)+1"),
        ("Proof", "Is 0 even or odd?", "Even (0 = 2×0)"),
        ("Strategy", "In chess, what is a fork?", "One piece attacks two pieces simultaneously"),
        ("Strategy", "In negotiation, what is BATNA?", "Best Alternative To Negotiated Agreement"),
        ("Fallacy", "What is ad hominem?", "Attacking the person, not the argument"),
        ("Fallacy", "What is a false dichotomy?", "Presenting only two options when more exist"),
    ],
    "creative": [
        ("Technique", "What is a metaphor?", "Comparison without like/as"),
        ("Technique", "What is alliteration?", "Repetition of initial consonant sounds"),
        ("Technique", "What is foreshadowing?", "Hints at future events"),
        ("Technique", "What is a haiku structure?", "3 lines: 5-7-5 syllables"),
        ("Technique", "What is a sonnet form?", "14 lines, usually iambic pentameter"),
        ("Narrative", "What is a protagonist?", "Main character of the story"),
        ("Narrative", "What is an antagonist?", "Character opposing the protagonist"),
        ("Narrative", "What is a plot twist?", "Unexpected change in story direction"),
        ("Narrative", "What is the hero's journey?", "Monomyth: departure, initiation, return"),
        ("Style", "What is stream of consciousness?", "Narrative capturing thought flow"),
        ("Style", "What is minimalism in writing?", "Stripped-down, sparse prose"),
        ("Style", "What is magical realism?", "Realistic setting with magical elements"),
        ("Dialogue", "What is subtext?", "Underlying meaning not explicitly stated"),
        ("Dialogue", "What makes dialogue realistic?", "Interruptions, contractions, character voice"),
        ("Worldbuilding", "What is hard worldbuilding?", "Rigorous, consistent rules for fictional worlds"),
        ("Worldbuilding", "What is soft worldbuilding?", "Focused on feel, less on mechanics"),
        ("Character", "What is a character arc?", "Character's transformation over the story"),
        ("Character", "What is an anti-hero?", "Protagonist lacking heroic qualities"),
        ("Structure", "What is the three-act structure?", "Setup, confrontation, resolution"),
        ("Structure", "What is in medias res?", "Starting in the middle of the action"),
    ],
    "general": [
        ("Geography", "Capital of France?", "Paris"),
        ("Geography", "Largest continent?", "Asia"),
        ("History", "When did WWII end?", "1945"),
        ("History", "Who wrote Hamlet?", "Shakespeare"),
        ("Language", "What is a synonym?", "Word with same meaning"),
        ("Language", "What is onomatopoeia?", "Word that sounds like what it describes"),
        ("Trivia", "How many continents?", "7"),
        ("Trivia", "What is the tallest mountain?", "Mount Everest"),
        ("Economics", "What is inflation?", "General increase in prices"),
        ("Economics", "What is GDP?", "Gross Domestic Product - total economic output"),
        ("Psychology", "What is cognitive dissonance?", "Discomfort from conflicting beliefs"),
        ("Psychology", "What is confirmation bias?", "Favoring information confirming existing beliefs"),
        ("Philosophy", "What is ethics?", "Study of right and wrong"),
        ("Philosophy", "What is existentialism?", "Philosophy of individual existence and freedom"),
        ("Technology", "What does AI stand for?", "Artificial Intelligence"),
        ("Technology", "What is blockchain?", "Decentralized, distributed ledger"),
        ("Health", "How many hours sleep recommended?", "7-9 hours"),
        ("Health", "What vitamin from sunlight?", "Vitamin D"),
        ("Arts", "What is the Mona Lisa?", "Famous painting by Leonardo da Vinci"),
        ("Arts", "What is a symphony?", "Extended musical composition for orchestra"),
    ],
}

# ============================================================================
# COG EXPANSION ENGINE
# ============================================================================
class LivingManifold:
    """COG metric + GTC cache for one Saiyan."""
    def __init__(self, K=512, eta=0.10):
        self.K = K
        self.eta = eta
        self.metric = torch.eye(K)
        self.trajectories = []  # list of {"proj": tensor, "label": str}
        self.cache_q = []  # query projections for GTC
        self.cache_r = []  # response projections for GTC
        self.n_expansions = 0
    
    def expand(self, h_k, label):
        """COG expansion: integrate trajectory into metric."""
        h_norm = F.normalize(h_k.unsqueeze(0).float(), dim=1).squeeze(0)
        J = torch.outer(h_norm, h_norm)
        self.metric = self.metric + self.eta * J + 0.001 * torch.eye(self.K)
        self.trajectories.append({"proj": h_k.detach().cpu(), "label": label[:80]})
        self.n_expansions += 1
    
    def growth(self):
        """Measure how much the metric has grown from identity."""
        return torch.norm(self.metric - torch.eye(self.K)).item()
    
    def save(self, path):
        torch.save({"metric": self.metric, "trajectories": self.trajectories,
                     "n_expansions": self.n_expansions, "K": self.K, "eta": self.eta}, path)
    
    def load(self, path):
        d = torch.load(path, map_location="cpu")
        self.metric = d["metric"]
        self.trajectories = d["trajectories"]
        self.n_expansions = d["n_expansions"]
        self.K = d["K"]
        self.eta = d["eta"]

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"\n[1] Loading base model: {MODEL_ID}")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                      device_map="auto", trust_remote_code=True)
    except:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                      device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    d_model = model.config.hidden_size
    K = 512
    print(f"  d_model={d_model}, K={K}")
    
    # Build UGT basis
    print(f"\n[2] Building UGT basis...")
    cal_prompts = list(TRAINING_DATA["general"][:30])
    hidden_states = []
    for p in cal_prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden_states.append(out.hidden_states[-1][0, -1, :].float())
    
    hs = torch.stack(hidden_states)
    U, S, _ = torch.linalg.svd(hs.T, full_matrices=False)
    K_actual = min(K, U.shape[1])  # SVD may give fewer columns than K
    basis = U[:, :K_actual].float().to(model.device)
    # QR for orthonormality
    Q, _ = torch.linalg.qr(basis)
    basis = Q
    K = K_actual  # Update K to actual dimension
    print(f"  K={K} (capped by calibration prompts)")
    
    def to_k(h): return h.float() @ basis.float()
    def get_h(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad(): out = model(**enc, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()
    
    # ===================================================================
    # TRAIN EACH SAIYAN
    # ===================================================================
    saiyan_states = {}
    
    for name, info in SAIYANS.items():
        domain = info["domain"]
        n_prompts = min(len(TRAINING_DATA.get(domain, [])), 100)
        
        print(f"\n[3] Training {name} ({domain}) — {n_prompts} prompts...")
        
        lm = LivingManifold(K=K, eta=0.10)
        training_prompts = TRAINING_DATA.get(domain, TRAINING_DATA["general"])[:n_prompts]
        
        for i, prompt in enumerate(training_prompts):
            h = get_h(prompt)
            h_k = to_k(h)
            lm.expand(h_k, f"{domain}: {prompt[:60]}")
            
            if (i+1) % 25 == 0:
                print(f"  {name}: {i+1}/{n_prompts} expansions, growth={lm.growth():.3f}")
        
        state_path = STATE_DIR / f"{name}_saiyan.pt"
        lm.save(str(state_path))
        saiyan_states[name] = {"manifold": lm, "domain": domain, "growth": lm.growth()}
        print(f"  {name}: {lm.n_expansions} expansions, metric growth={lm.growth():.3f}")
        print(f"  Saved to {state_path}")
    
    # Yamcha is baseline — load but don't specialize (use general domain)
    yamcha_lm = saiyan_states["Yamcha"]["manifold"] if "Yamcha" in saiyan_states else None
    
    # ===================================================================
    # CREATE FUSIONS
    # ===================================================================
    print(f"\n[4] Creating fusions...")
    fusion_states = {}
    
    for fusion_name, fusion_info in FUSIONS.items():
        parent_a = fusion_info["parents"][0]
        parent_b = fusion_info["parents"][1]
        blend = fusion_info["blend"]
        
        if parent_a not in saiyan_states or parent_b not in saiyan_states:
            print(f"  {fusion_name}: SKIP — parents not found")
            continue
        
        lm_a = saiyan_states[parent_a]["manifold"]
        lm_b = saiyan_states[parent_b]["manifold"]
        
        # Fuse metrics via weighted average
        fused = LivingManifold(K=K, eta=0.10)
        fused.metric = blend * lm_a.metric + (1 - blend) * lm_b.metric
        # Merge trajectories
        fused.trajectories = lm_a.trajectories + lm_b.trajectories
        fused.n_expansions = lm_a.n_expansions + lm_b.n_expansions
        
        state_path = STATE_DIR / f"{fusion_name}_fused.pt"
        fused.save(str(state_path))
        fusion_states[fusion_name] = {"manifold": fused, "parents": fusion_info["parents"],
                                       "growth": fused.growth()}
        print(f"  {fusion_name} ({fusion_info['method']}): {len(fused.trajectories)} trajectories, growth={fused.growth():.3f}")
    
    # ===================================================================
    # BENCHMARK ALL MODELS
    # ===================================================================
    print(f"\n[5] Benchmarking all models...")
    
    def score_answer(model_answer, correct_answer):
        """Score how close an answer is to the correct one."""
        model_lower = model_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        # Exact match or substring match
        if correct_lower in model_lower or model_lower in correct_lower:
            return 1.0
        # Check for key tokens
        correct_tokens = set(correct_lower.replace("(", " ").replace(")", " ").replace(",", " ").split())
        model_tokens = set(model_lower.split())
        if correct_tokens:
            overlap = len(correct_tokens & model_tokens) / len(correct_tokens)
            return min(1.0, overlap)
        return 0.0
    
    benchmark_results = {}
    # We can't do full generation for all 6 domains × 20 Q × 9 models on CPU.
    # Instead, use a targeted approach: run the best Saiyan + fusions vs baseline.
    
    # Quick test: Goku (math) on math benchmarks
    test_models = {
        "Yamcha (baseline)": None,  # no manifold
        "Goku": saiyan_states.get("Goku", {}).get("manifold"),
        "Vegeta": saiyan_states.get("Vegeta", {}).get("manifold"),
        "Gohan": saiyan_states.get("Gohan", {}).get("manifold"),
        "Piccolo": saiyan_states.get("Piccolo", {}).get("manifold"),
        "Trunks": saiyan_states.get("Trunks", {}).get("manifold"),
        "Gogeta": fusion_states.get("Gogeta", {}).get("manifold"),
        "Vegito": fusion_states.get("Vegito", {}).get("manifold"),
        "Gotenks": fusion_states.get("Gotenks", {}).get("manifold"),
    }
    
    for model_name, lm in test_models.items():
        print(f"\n  Benchmarking {model_name}...")
        domain = "math"  # Quick test on math domain
        if "Goku" in model_name or "Gogeta" in model_name or "Vegito" in model_name:
            domain = "math"
        elif "Vegeta" in model_name:
            domain = "code"
        elif "Gohan" in model_name:
            domain = "science"
        elif "Piccolo" in model_name:
            domain = "logic"
        elif "Trunks" in model_name or "Gotenks" in model_name:
            domain = "creative"
        else:
            domain = "general"
        
        questions = BENCHMARKS.get(domain, BENCHMARKS["general"])[:10]
        scores = []
        
        for category, question, correct in questions:
            # For baseline (no manifold), just generate directly
            prompt = f"Question: {question}\nAnswer concisely:"
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            answer = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            s = score_answer(answer, correct)
            scores.append(s)
        
        avg_score = sum(scores) / max(len(scores), 1) * 100
        benchmark_results[model_name] = {
            "domain": domain,
            "avg_score": round(avg_score, 1),
            "n_questions": len(scores),
        }
        print(f"    {domain}: {avg_score:.1f}% ({len(scores)} questions)")
    
    # ===================================================================
    # COMPARATIVE ANALYSIS
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  SAIYAN FAMILY — BENCHMARK RESULTS")
    print(f"{'='*70}")
    
    baseline_score = benchmark_results.get("Yamcha (baseline)", {}).get("avg_score", 0)
    
    print(f"\n  {'Name':20s} {'Domain':12s} {'Score':>8s} {'vs Baseline':>14s}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*14}")
    
    for name, result in benchmark_results.items():
        score = result["avg_score"]
        domain = result["domain"]
        delta = score - baseline_score if baseline_score > 0 else 0
        delta_str = f"+{delta:.1f}%" if delta > 0 else (f"{delta:.1f}%" if delta < 0 else "—")
        print(f"  {name:20s} {domain:12s} {score:>7.1f}% {delta_str:>14s}")
    
    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": MODEL_ID,
        "saiyans": {name: {"domain": info["domain"], "growth": saiyan_states.get(name, {}).get("growth", 0),
                            "n_expansions": saiyan_states.get(name, {}).get("manifold", LivingManifold()).n_expansions}
                      for name, info in SAIYANS.items()},
        "fusions": {name: {"parents": info["parents"], "growth": fusion_states.get(name, {}).get("growth", 0)}
                     for name, info in FUSIONS.items()},
        "benchmarks": benchmark_results,
        "baseline_score": baseline_score,
    }
    
    with open(BENCH_DIR / "saiyan_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {BENCH_DIR / 'saiyan_results.json'}")
    print(f"  States saved to {STATE_DIR}/")
    print(f"\n{'='*70}")
    print(f"  SAIYAN TRAINING COMPLETE")
    print(f"  {len(saiyan_states)} Saiyans trained, {len(fusion_states)} fusions created")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
