# IntroAI-Assignments
A list of projects that I worked on for my Intro to Ai class at Rutgers. 

"The Pac-Man projects were developed at UC Berkeley for the education purpose of AI, and adapted by our course staff for Rutgers CS440. The projects apply an array of AI techniques to playing Pac-Man. However, these projects don’t focus on building AI for video games. Instead, they teach foundational AI concepts, such as informed state-space search, probabilistic inference, and reinforcement learning. These concepts underly real-world application areas such as natural language processing, computer vision, and robotics."


Read more information about the projects here: https://xintongemilywang.github.io/projects/index.html

## Projects
Project 1 — Search (search_sol-1.zip)
search.py implements the four generic search algorithms, all using a (state, path) frontier pattern with a visited set:

DFS — util.Stack(), goal-tests on pop (not push), so it's a proper graph search.
BFS — util.Queue(), marks nodes visited when enqueued to avoid duplicate frontier entries.
UCS — util.PriorityQueue() keyed on path cost, with a best_cost dict to discard stale/dominated queue entries.
A* — same as UCS but priority = path_cost + heuristic(state), which is what actually directs search toward the goal instead of expanding uniformly outward.

searchAgents.py has the harder, problem-specific pieces:

CornersProblem — state is (position, visited_corners_tuple) rather than just position, since you need to remember which of the 4 corners you've already hit.
cornersHeuristic — relaxes the problem (ignores walls) and computes the best Manhattan-distance tour over the remaining corners via brute-force permutations — admissible because real maze distance ≥ Manhattan distance.
foodHeuristic — takes max(farthest food via maze distance, diameter between the two farthest food dots), using a memoized mazeDistance cache stored in problem.heuristicInfo so repeated calls don't recompute paths.
findPathToClosestDot — just runs BFS on an AnyFoodSearchProblem (goal = any food cell).

Project 2 — Reinforcement Learning (reinforcement_sol_zip.zip)

valueIterationAgents.py — textbook batch value iteration: for each iteration, computes a new values table from the old one (so updates don't leak within a sweep), with computeQValueFromValues doing the Bellman backup Σ P(s'|s,a)[R + γV(s')].
qlearningAgents.py — model-free Q-learning: getQValue reads from a Counter (defaults to 0), update applies the standard sample update (1-α)Q + α(reward + γ·maxQ(s')), and getAction does ε-greedy exploration via util.flipCoin. ApproximateQAgent swaps the Q-table for a linear weights · features model and updates weights via gradient-style correction.
analysis.py — the gridworld parameter-tuning answers (discount/noise/living-reward combos) that produce each of the 5 requested policies (e.g., risk the cliff for the close exit vs. play it safe for the far one).

Project 3 — Multi-Agent Games (multiagent_sol.zip)
All in multiAgents.py:

ReflexAgent.evaluationFunction — heuristic blend of score, distance to nearest food, food count, and ghost proximity (big penalty for non-scared ghosts within 2 steps, bonus for chasing scared ones).
MinimaxAgent / AlphaBetaAgent / ExpectimaxAgent — share the same recursion skeleton (cycle through agent indices, increment depth only after the last ghost moves), differing only in how non-Pacman levels combine child values: min, min-with-pruning, or probability-weighted average.
betterEvaluationFunction — extends the reflex heuristic with capsule-distance incentives and a tiered ghost-distance penalty (catastrophic at distance ≤1, steep at ≤2, mild beyond).

Project 4 — Machine Learning (machinelearning_sol.zip)
All in models.py (PyTorch-based):

PerceptronModel — single weight vector, trains with the classic mistake-driven update w += x*y until a full pass makes zero mistakes.
RegressionModel — a small 3-hidden-layer MLP (128→128→64→1) trained with Adam + MSE loss to fit sin(x).
DigitClassificationModel — 784→300→150→10 MLP with ReLU activations, cross-entropy loss, Adam optimizer, training until validation accuracy hits 98%.
