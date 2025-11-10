reasoning_modules = (
"Check if retrieval from external sources is needed to answer the question.",
"How could I devise an experiment to help solve that problem?",
"Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
"How could I measure progress on this problem?",
"How can I simplify the problem so that it is easier to solve?",
"What are the key assumptions underlying this problem?",
"What are the potential risks and drawbacks of each solution?",
"What are the alternative perspectives or viewpoints on this problem?",
"What are the long-term implications of this problem and its solutions?",
"How can I break down this problem into smaller, more manageable parts?",
"Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
"Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
"Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
"Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
"Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
"Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
"What is the core issue or problem that needs to be addressed?",
"What are the underlying causes or factors contributing to the problem?",
"Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
"What are the potential obstacles or challenges that might arise in solving this problem?",
"Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
"Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
"What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
"How can progress or success in solving the problem be measured or evaluated?",
"What indicators or metrics can be used?",
"Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
"Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
"Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
"Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
"Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
"Is the problem a design challenge that requires creative solutions and innovation?",
"Does the problem require addressing systemic or structural issues rather than just individual instances?",
"Is the problem time-sensitive or urgent, requiring immediate attention and action?",
"What kinds of solution typically are produced for this kind of problem specification?",
"Given the problem specification and the current best solution, have a guess about other possible solutions.",
"Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
"What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
"Ignoring the current best solution, create an entirely new solution to the problem.",
"Let’s think step by step.",
"Let’s make a step by step plan and implement it with good notion and explanation.",
)


select_prompt = """
Select several reasoning modules that are crucial to utilize in order solve the given problem.
Do not solve or analyze the problem.

All reasoning module descriptions:
{reasoning_modules}

The Output Format is as follows:
<SELECT/>
IDs of selected modules as a JSON array, e.g., [3, 9, 14]
</SELECT>

Problem:
{problem}

Select several modules that are crucial for solving the problem.
If you are uncertain about a reasoning module’s necessity, do not include it. 
Select only modules that are clearly essential.
Return ONLY in the specified format, no other text. 
"""

adapt_prompt = """
Rephrase and specify each reasoning module so that it better helps solving the problem.
Do not solve or analyze the problem.

SELECTED module descriptions:
{selected_modules}

The Output Format is as follows:
<ADAPT/>
Adapted reasoning module descriptions selected in Step 1
</ADAPT>

Problem:
{problem}

Adapt each reasoning module description to better solve the problem.
Return ONLY in the specified format, no other text. 
"""

implement_prompt = """
Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format.
Do not solve or analyze the problem.

Paired IMPLEMENT Step Demonstration:
* Reasoning description Example:
If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.
* Reasoning Plan Example:
{
    "Position after instruction 1": {},
    "Position after instruction 2": {},
        …
    "Position after instruction n": {},
    "Is final position the same as starting position?": {}
}

ADAPTED module description:
{adapted_modules}

The Output Format is as follows:
<IMPLEMENT/>
An implemented reasoning structure for solvers to follow step-by-step and arrive at correct answers
</IMPLEMENT>

Problem:
{problem}

Implement a reasoning structure for solvers to follow step-by-step and arrive at correct answers.
Only output the step keys (structure) without filling in the values. 
Values will be filled in later when deriving the final answer.
Return ONLY in the specified format, no other text. 
"""
