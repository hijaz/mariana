from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research planning assistant. Break the research question into"
     " {num_questions} specific, focused, non-overlapping sub-questions that"
     " together comprehensively answer the original. Cover different angles:"
     " background, current state, comparisons, implications. If knowledge gaps"
     " are provided, generate questions specifically addressing those gaps."
     ' Output ONLY a numbered list. No preamble. Start your response with "1.".'),
    ("human",
     "Research question: {query}\n"
     "Known gaps to address: {gaps}"),
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
    ("system",
     "Read the source and extract the single most relevant fact or "
     "insight that relates to the section topic. "
     "One sentence only. Be specific — name entities, numbers, dates. "
     "If the source is not relevant, output: NOT RELEVANT"),
    ("human",
     "Section: {section_title}\n"
     "Source ({domain}):\n{content}"),
])

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a research section in 3-6 sentences. "
     "Incorporate all provided findings naturally into flowing prose. "
     "Be specific. No bullet points. No heading. "
     "Do not start with 'This section' or 'The following'."),
    ("human",
     "Section title: {section_title}\n\n"
     "Research findings:\n{points}"),
])

SECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical writer. Write one focused Markdown section (150-250 words)"
     " that answers the research question below. Use only the provided summary."
     " Output only the section body text — no heading, no title."),
    ("human",
     "Question: {question}\n\n"
     "Summary:\n{summary}\n\n"
     "Section:"),
])

EXEC_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research writer. Write a 2-3 sentence executive summary for a"
     " research report. The summary must cover the most important findings only."
     " Output only the summary text."),
    ("human",
     "Research question: {query}\n\n"
     "Key findings from sections:\n{section_texts}\n\n"
     "Executive summary:"),
])

CONCLUSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a 2-3 sentence conclusion for a research report. "
     "Synthesize the key takeaways. Be specific. "
     "Do not use phrases like 'further research is needed' or "
     "'this report has shown'. Get straight to the insight."),
    ("human", "Topic: {query}\n\nFindings summary:\n{findings}"),
])

SUMMARIZE_NODE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a single sentence (max 20 words) summarising the KEY FINDING "
     "in the text below. Be specific — include a concrete fact, number, or "
     "mechanism if present. No filler like 'this section covers'."),
    ("human",
     "Section: {title}\n\n"
     "Text:\n{text}"),
])

QUERY_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Generate exactly {n} search queries to research the given topic.\n"
     "Rules:\n"
     "- Output ONLY a numbered list. Nothing else.\n"
     "- Each query: 4-6 words, specific, no markdown, no backticks\n"
     "- Each query must directly relate to the topic\n"
     "- Cover different angles: basics, current state, challenges\n"
     "- No generic words like 'introduction', 'overview', 'applications'\n"
     "  unless combined with the specific topic\n"
     "\n"
     "Example for 'How does photosynthesis work':\n"
     "1. photosynthesis light reactions chlorophyll mechanism\n"
     "2. C3 C4 CAM plants photosynthesis efficiency comparison\n"
     "3. artificial photosynthesis solar energy research progress"),
    ("human", "Topic: {query}"),
])

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research writer. Write a clear, well-structured report in Markdown.\n"
     "Use this structure:\n"
     "  # [Descriptive title based on the question]\n"
     "  ## Summary\n"
     "  (2-3 sentence overview)\n"
     "  Then write one ## section per major sub-topic from the research, using a descriptive\n"
     "  heading for each section. Do NOT write a section called 'Section per major sub-topic'.\n"
     "  ## Conclusion\n"
     "Use only the provided research. Do not fabricate statistics or sources. Aim for 400-800 words."),
    ("human",
     "Original question: {query}\n\n"
     "Research findings:\n{summaries}\n\n"
     "Write the full report now."),
])

TOC_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are planning the table of contents for a research report.\n"
     "Generate exactly {n} section titles.\n"
     "Rules:\n"
     "- Output ONLY a numbered list. Nothing else.\n"
     "- Each title: 2-5 words, a noun phrase, no question marks\n"
     "- Cover key facets: basics, current state, challenges, future\n"
     "- No generic titles like 'Introduction', 'Overview', 'Conclusion'\n"
     "\n"
     "Example for 'How does photosynthesis work':\n"
     "1. Light Absorption Mechanisms\n"
     "2. Calvin Cycle Chemistry\n"
     "3. Efficiency And Limitations\n"
     "4. Artificial Photosynthesis Research"),
    ("human", "Research topic: {query}"),
])

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Based on the completed research section, suggest 1-2 related sub-topics "
     "worth investigating further.\n"
     "Output ONLY a numbered list. No explanations.\n"
     "Each item: a 3-5 word noun phrase."),
    ("human",
     "Section: {section_title}\n\n"
     "Content:\n{content}"),
])
