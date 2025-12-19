# 

# 一阶段prompt
anchor_prompt_head ="""以下是一个编程问题的描述："""
anchor_prompt_tail = """你现在需要严格遵循以下要求： 做一个粗粒度的锚点计划，写出完成这个问题大致的、必须的、一定正确的步骤。一个高层次的，有顺序的，有逻辑的计划。 例如： 问题：实现一个快排。 输出：#### 1 实现一个swap函数。#### 2 实现包含pivot的交换逻辑 #### 3 整合代码，完成一个可执行函数作为结果 你需要效仿以上做法，但是记住，你不需要一定分为3点，根据实际问题来。做一个粗粒度的锚点计划，写出完成这个问题大致的、必须的、一定正确的步骤。一个高层次的，有顺序的，有逻辑的计划，每一步需要明确依赖前一步的结果，整体形成一个线性可执行的完成路径。"""

# 二阶段prompt
# 根据问题描述，以及已有代码，以及锚点计划里下一步需要实现目标，列出一个level计划
level_prompt_head = """以下是一个编程问题的描述："""
level_prompt_mid1 = "以下是解决这个问题的粗粒度锚点计划："
level_prompt_mid2 = """你已经完成了这些锚点计划："""
level_prompt_mid3 = "已经完成的部分代码是："""
level_prompt_tail = """你的任务是： 基于原编程问题描述和整体的粗粒度锚点计划和已经完成的代码，生成一份为了完成下一个锚点计划的详细规划。只需要文字规划。"""

# 三阶段prompt
# 根据原始问题，level计划，已有代码，写CoT的prompt和Code的prompt【CoTCode prompt和Code prompt】
code_prompt_head = """这是一个编程问题的描述："""
code_prompt_mid1 = "以下是部分已生成代码："
code_prompt_mid2 = "下一步需要生成的代码所负责的功能是："
code_prompt_tail = """你的任务是： 依据上述编程问题描述，已有代码，以及下一步需要实现的功能，生成对应的Python代码。"""

CoT_prefix = "你需要严格遵守以下规则：1. 你需要生成思考过程和代码，格式如下：<think>你的思考过程</think>```python\n# 你的代码\n```。2. Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.```python\n# YOUR CODE HERE\n```" 

Code_prefix = "你需要严格遵守以下规则：Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.```python\n# YOUR CODE HERE\n```"

# 四阶段prompt
# 