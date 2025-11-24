import gradio as gr

from agent import solve


class BasicAgent:
    """Wraps the LangGraph agent built in agent.py for simple Q&A."""

    def __init__(self):
        print("LangGraph-based multi-tool agent ready.")

    def __call__(self, question: str):
        """Return answer for a single question."""
        return solve(question)


agent = BasicAgent()


def chat_fn(message: str, history):
    """Gradio chat function."""
    history = history or []

    if not message.strip():
        return history, ""

    try:
        answer, tools_used = agent(message)
        tools_str = "🛠 Tools used: " + ", ".join(tools_used) if tools_used else "🛠 Tools used: none"

        final_text = f"{answer}\n\n---\n{tools_str}"
    except Exception as e:
        final_text = f"Error: {e}"

    history = history + [[message, final_text]]
    return history, ""


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# 🔧 Multi-Tool AI Agent (LangGraph + OpenAI)")
    gr.Markdown(
        """
        This is a general-purpose AI agent built with **LangGraph**, **OpenAI**, and
        several tools:

        - `similar_question` → retrieves similar Q&A from a local FAISS index
        - `web_search` → web search via Tavily
        - `wiki_search` → Wikipedia search
        - `arxiv_search` → scientific paper search
        - `calculator` → arithmetic evaluations

        Type a question below and the agent will decide which tools to call.
        """
    )

    chatbot = gr.Chatbot(label="Agent Conversation")

    with gr.Row():
        msg = gr.Textbox(
            label="Your question",
            placeholder="Ask me anything...",
            lines=1,  # single line → Enter submits
        )
        send_btn = gr.Button("Send", variant="primary")

    clear_btn = gr.Button("Clear")

    # Enter key submits
    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
    # Clicking the button also submits
    send_btn.click(chat_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])

    # Clear button
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

    gr.Markdown("""Created with ❤️ by [Ezan](https://github.com/EnricoZanetti)""")

if __name__ == "__main__":
    print("\n------------------------------ App Starting ------------------------------\n")
    print("Launching Gradio Interface for the Multi-Tool Agent...")
    demo.launch(debug=True)
