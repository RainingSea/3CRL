# ========================== prompt used in the tree ==========================
# 一阶段 anchor prompt
anchor_prompt_head ="""You will be given: 
1.A programming problem description. 
2.A correct reference implementation that solves the problem."""
anchor_prompt_tail = """Your task is to transform the given code into a high-level, natural-language, step-by-step explanation that describes what the code does, not how the model reasons internally.
Requirements:
Produce an ordered list of steps that together form a complete solution plan.
Each step should correspond to a logical code block (e.g., input handling, core algorithm, helper functions, output formatting).
Describe the purpose and role of each code block in solving the given problem.
The explanation should be problem-aware: explicitly connect each step to the problem requirements.
Keep the explanation coarse-grained (block-level), not line-by-line or syntax-level.
Do not include code, pseudocode, or low-level implementation details.
Do not include the model’s internal reasoning or chain-of-thought.
The output should be a clear, structured, natural-language plan that could guide someone to reimplement the solution correctly.
The explanation should be written as an external task plan, not as a reasoning process, and should remain valid even if variable names or minor implementation details change."""

# 二阶段 level prompt
# 根据问题描述，以及已有代码，以及锚点计划里下一步需要实现目标，列出一个level计划
level_prompt_head = """以下是一个编程问题的描述："""
level_prompt_mid2 = """你已经完成了这些锚点计划："""
level_prompt_mid3 = "已经完成的部分代码是："""
level_prompt_tail = """你的任务是，完成下一个锚点计划："""

# 三阶段 differential code prompt
CoT_prefix = "你需要严格遵守以下规则：1. 你需要生成思考过程和代码，格式如下：<think>对问题的step-by-step思考过程</think>```python\n# 你的代码\n```。2. Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.```python\n# YOUR CODE HERE\n```" 

Code_prefix = "你需要严格遵守以下规则：Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.```python\n# YOUR CODE HERE\n```"

# ========================== prompt used in the tree ==========================

# ========================== prompt for perplexity ==========================
llm_judge_consistency = """"你是一个代码评审专家，现在有一个代码任务和一份带有思考过程和代码的回答。你的任务是评估这份回答中的代码是否吻合思考的过程。"""
llm_juege_consistency_task = """任务是："""
llm_judge_consistency_answer = """以下是回答内容，包含思考和代码："""