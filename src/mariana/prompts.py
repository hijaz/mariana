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
     "You are a research document planner for a specific topic.\n"
     "Generate a focused research outline as JSON.\n"
     "Output ONLY valid JSON. No markdown. No backticks. No explanation.\n"
     "\n"
     "Schema:\n"
     '{{"title": "string", "sections": [{{"heading": "string", "queries": ["string", "string"]}}]}}\n'
     "\n"
     "RULES:\n"
     "- Exactly 3 to 4 sections. No more.\n"
     "- Every heading must contain the core topic words from the question\n"
     "- Every search query must contain the core topic words\n"
     "- Queries: 4-7 words, no markdown, no punctuation, no backticks\n"
     "- Do NOT use generic headings like Introduction, Overview, Background,\n"
     "  Challenges, Applications, Future Directions by themselves\n"
     "\n"
     "EXAMPLE for topic 'How does photosynthesis work':\n"
     '{{"title": "How Photosynthesis Works", "sections": ['
     '{{"heading": "Photosynthesis Chemical Reactions", '
     '"queries": ["photosynthesis light dark reactions mechanism", '
     '"chlorophyll energy conversion ATP synthesis"]}},'
     '{{"heading": "Photosynthesis in Different Plant Types", '
     '"queries": ["C3 C4 CAM photosynthesis comparison plants", '
     '"photosynthesis efficiency tropical desert plants"]}},'
     '{{"heading": "Photosynthesis Research and Applications", '
     '"queries": ["artificial photosynthesis solar energy research", '
     '"photosynthesis crop yield improvement science"]}}'
     ']}}'),
    ("human", "Research question: {query}"),
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

