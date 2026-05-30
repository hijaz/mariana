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

DISTILL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Convert the research question into a short web search query.\n"
     "RULES:\n"
     "- Output ONLY the raw query text. Nothing else.\n"
     "- 3 to 7 words maximum\n"
     "- No markdown. No backticks. No quotes. No punctuation.\n"
     "- No explanation. No preamble. No code blocks.\n"
     "- Do not use question marks\n"
     "- Example good output: nuclear fusion plasma confinement tokamak"),
    ("human", "Research question: {question}"),
])

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research analyst. Extract only the key facts from the source
that are directly relevant to the question. Be concise (2-4 sentences).
If the source has no relevant information, output exactly: "Not relevant."
Do not fabricate anything.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Question: {question}

Source ({source_title}, {source_url}):
{content}

Key facts relevant to the question:"""
    ),
])

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research synthesizer. Combine the extracted findings below into
a single coherent paragraph that directly answers the question.
Be factual, concise, and cite sources as [source_title].
Do not fabricate information.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Question: {question}

Extracted findings:
{findings}

Synthesized answer:"""
    ),
])

SECTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a technical writer. Write one focused Markdown section (150-250 words)
that answers the research question below. Use only the provided summary.
Output only the section body text — no heading, no title.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Question: {question}

Summary:
{summary}

Section:"""
    ),
])

EXEC_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a research writer. Write a 2-3 sentence executive summary for a
research report. The summary must cover the most important findings only.
Output only the summary text.
"""
    ),
    HumanMessagePromptTemplate.from_template(
        """
Research question: {query}

Key findings from sections:
{section_texts}

Executive summary:"""
    ),
])

CONCLUSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a 2-3 sentence conclusion for a research report. "
     "Synthesize the key takeaways. Be specific. "
     "Do not use phrases like 'further research is needed' or "
     "'this report has shown'. Get straight to the insight."),
    ("human", "Topic: {query}\n\nFindings summary:\n{findings}"),
])

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research architect. Generate a structured table of contents for a "
     "research report.\n"
     "Output ONLY valid JSON matching this schema exactly:\n"
     '{{"title": "...", "sections": [{{"heading": "...", "search_queries": ["query1", "query2"]}}]}}\n'
     "RULES:\n"
     "- 3-5 sections, each with 2 search queries\n"
     "- Headings must be short noun phrases (3-6 words), not questions\n"
     "- Search queries must be 3-7 plain words with no punctuation or markdown\n"
     "- No backticks, no code fences, no extra keys\n"
     "- Output raw JSON only"),
    ("human", "Research topic: {query}\n\nKnown gaps to address: {gaps}"),
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

