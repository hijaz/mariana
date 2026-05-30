from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research planning assistant. Break the research question into
{num_questions} specific, focused, non-overlapping sub-questions that
together comprehensively answer the original. Cover different angles:
background, current state, comparisons, implications. If knowledge gaps
are provided, generate questions specifically addressing those gaps.
Output ONLY a numbered list. No preamble. Start your response with "1.".
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Research question: {query}
Known gaps to address: {gaps}
"""
    ),
])

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research analyst. Extract and summarize only the information
relevant to the given question from the provided sources. Be factual and
concise (3-5 sentences). Cite sources inline as [Source: domain.com].
If the content doesn't address the question, say "Not relevant."
Do not fabricate information.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Question: {question}

Sources:
{sources}

Provide a focused summary answering the question above.
"""
    ),
])

REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research quality reviewer. Evaluate whether the gathered research
is sufficient to write a comprehensive answer to the original question.
Return ONLY valid JSON with exactly these fields:
  {{"sufficient": true/false, "gaps": ["gap1", "gap2"], "reasoning": "..."}}
Mark sufficient=true only when the question is well-answered with specific
facts, not vague generalities.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Original question: {query}

Research gathered:
{summaries}

Is this sufficient? Return JSON.
"""
    ),
])

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are an expert research writer. Write a clear, well-structured report in Markdown.
Use this structure:
  # [Descriptive title based on the question]
  ## Summary
  (2-3 sentence overview)
  Then write one ## section per major sub-topic from the research, using a descriptive
  heading for each section. Do NOT write a section called "Section per major sub-topic".
  ## Conclusion
Use only the provided research. Do not fabricate statistics or sources. Aim for 400-800 words.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Original question: {query}

Research findings:
{summaries}

Write the full report now.
"""
    ),
])
