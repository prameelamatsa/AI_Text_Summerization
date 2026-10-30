
import os
import re
import math
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file if present
load_dotenv()

# ------------------------------------------------------------------------------
# 1. NLP & Readability Metrics Helper Functions
# ------------------------------------------------------------------------------
def count_syllables(word: str) -> int:
    """Estimates the syllable count of an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    syllables = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(1, syllables)

def calculate_metrics(text: str):
    """Calculates word count, sentence count, and Flesch Reading Ease score."""
    if not text.strip():
        return {"words": 0, "sentences": 0, "flesch": 0.0, "readability": "N/A"}

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = max(1, len(sentences))
    
    words = re.findall(r'\b\w+\b', text)
    num_words = max(1, len(words))

    num_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease Formula
    flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    flesch_score = max(0.0, min(100.0, round(flesch_score, 1)))

    if flesch_score >= 80:
        readability = "Very Easy (6th grade)"
    elif flesch_score >= 60:
        readability = "Standard (8th-9th grade)"
    elif flesch_score >= 40:
        readability = "Fairly Difficult (College)"
    else:
        readability = "Difficult (Advanced / Professional)"

    return {
        "words": num_words,
        "sentences": num_sentences,
        "flesch": flesch_score,
        "readability": readability
    }

def extract_text_from_file(uploaded_file) -> str:
    """Extracts plain text from .txt or .pdf uploaded files."""
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    return ""

# ------------------------------------------------------------------------------
# 2. AI Summarization Logic using google-genai
# ------------------------------------------------------------------------------
def generate_summary(text: str, length: str, format_type: str, tone: str, api_key: str) -> str:
    """Calls Gemini API via official google-genai SDK to generate customized summary."""
    client = genai.Client(api_key=api_key)

    length_instructions = {
        "Short": "Keep the summary concise (1-2 sentences or ~50 words).",
        "Medium": "Provide a moderate summary (~150-200 words).",
        "Detailed": "Provide a comprehensive, detailed overview covering main concepts and sub-points."
    }

    format_instructions = {
        "Bullet Points": "Format the response entirely as concise bullet points.",
        "Key Takeaways": "Start with a brief core statement, followed by key takeaway bullet points.",
        "Executive Summary": "Format as a structured executive summary with clear paragraph sections.",
        "Paragraphs": "Present the summary in clear, fluid paragraphs."
    }

    tone_instructions = {
        "Professional": "Maintain a formal, professional, business-appropriate tone.",
        "Simple (ELI5)": "Explain concepts simply as if to a 5-year-old, using clear analogies.",
        "Technical": "Use technical, domain-specific terminology suitable for subject matter experts."
    }

    prompt = f"""
Summarize the following text based on these exact constraints:
- Summary Length: {length_instructions.get(length, '')}
- Output Format: {format_instructions.get(format_type, '')}
- Target Tone: {tone_instructions.get(tone, '')}

Source Text:
{text}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1000
        )
    )
    return response.text

# ------------------------------------------------------------------------------
# 3. Streamlit UI Design
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="wide")

    st.title("📝 AI Text Summarizer & Analyzer")
    st.markdown("Transform long documents into clear, customizable summaries with real-time NLP quality evaluation.")

    # Sidebar: Setup & Configuration Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Handling (Reads from environment or sidebar input)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            if not api_key:
                st.warning("Please provide a Gemini API Key to run summarizations.")

        st.divider()
        st.subheader("Summary Settings")
        summary_length = st.select_slider("Length", options=["Short", "Medium", "Detailed"], value="Medium")
        summary_format = st.selectbox("Format", ["Bullet Points", "Key Takeaways", "Executive Summary", "Paragraphs"])
        summary_tone = st.selectbox("Tone", ["Professional", "Simple (ELI5)", "Technical"])

    # Main Interface Split into Two Columns
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Input Text")
        uploaded_file = st.file_uploader("Upload Document (.txt or .pdf)", type=["txt", "pdf"])
        
        input_text = ""
        if uploaded_file is not None:
            input_text = extract_text_from_file(uploaded_file)
            st.success(f"File loaded successfully! ({len(input_text.split())} words)")
        
        input_text = st.text_area(
            "Or paste your text below:",
            value=input_text,
            height=320,
            placeholder="Paste raw text here..."
        )

        # Input Metrics
        if input_text:
            in_metrics = calculate_metrics(input_text)
            st.caption(f"**Word Count:** {in_metrics['words']} | **Sentences:** {in_metrics['sentences']} | **Flesch Score:** {in_metrics['flesch']} ({in_metrics['readability']})")

        summarize_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Generated Summary")

        if summarize_button:
            if not api_key:
                st.error("Missing Gemini API Key. Please add it to the sidebar.")
            elif not input_text.strip():
                st.warning("Please enter or upload text to summarize.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary_res = generate_summary(
                            text=input_text,
                            length=summary_length,
                            format_type=summary_format,
                            tone=summary_tone,
                            api_key=api_key
                        )
                        st.session_state['summary_result'] = summary_res

                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")

        if 'summary_result' in st.session_state and st.session_state['summary_result']:
            summary_res = st.session_state['summary_result']
            st.markdown(summary_res)
            st.divider()

            # Evaluation & Quality Metrics
            st.subheader("📊 Summary Evaluation Metrics")
            in_m = calculate_metrics(input_text)
            out_m = calculate_metrics(summary_res)

            compression_ratio = round((1 - (out_m["words"] / max(1, in_m["words"]))) * 100, 1)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Compression Ratio", f"{max(0, compression_ratio)}%")
            m_col2.metric("Summary Words", out_m["words"], delta=f"-{in_m['words'] - out_m['words']}")
            m_col3.metric("Flesch Readability", f"{out_m['flesch']} / 100", help=out_m['readability'])

            # Export Option
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=summary_res,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()