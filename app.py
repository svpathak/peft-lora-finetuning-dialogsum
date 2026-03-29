import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "sanchitvp/flan-t5-base-finetuned-dialogsum"

# Load model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

MAX_INPUT_CHARS = 2000
examples = [["""Dr. Watson: Sherlock, what do you think? Who committed this robbery?
Sherlock: I see some evidence.
Dr. Watson: And the evidences lead to..?
Sherlock: to Moriarty!"""],
["""Interviewer: So why should we hire you?
Candidate: I have experience building scalable ML systems and deploying models to production.
Interviewer: Can you give an example?
Candidate: I fine-tuned FLAN-T5 using LoRA and deployed it as a public inference service.
Interviewer: Interesting."""],
["""Manager: Team, are we aligned for the product launch?
Engineer: Yes, deployment is complete. The inference service is working as expected.
Manager: Great, let's launch tomorrow morning."""],
["""Engineer A: The model accuracy improved after fine-tuning.
Engineer B: What method did you use?
Engineer A: LoRA-based parameter-efficient training.
Engineer B: How much of the model did you update?
Engineer A: Only 0.36% of parameters."""],
["""Romeo: Why must our families be enemies?
Juliet: It is not our names that define us.
Romeo: Then let us abandon them.
Juliet: If love is true, we shall find a way."""],
["""Harry: I saw something in the forest last night.
Ron: Was it dangerous?
Harry: It felt powerful, like dark magic.
Ron: Then Harry, we need to tell Dumbledore immediately."""]]

def summarize_dialogue(dialogue: str) -> str:
    if not dialogue or len(dialogue.strip()) == 0:
        return "Please enter a dialogue."

    if len(dialogue) > MAX_INPUT_CHARS:
        return f"Input too long. Please limit to {MAX_INPUT_CHARS} characters."

    prompt = f"Summarize the following dialogue:\n{dialogue}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=6,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo")) as demo:

    gr.Markdown(
        """
        # Dialogue Summarization  
        Fine-tuned FLAN-T5-Base **0.36% of total parameters (884K out of 250M parameters)**.  
        Paste a multi-turn dialogue below to generate a concise summary.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            dialogue_input = gr.Textbox(
                lines=8,
                placeholder="Enter dialogue here...",
                label="Dialogue Input"
            )

            summarize_btn = gr.Button("Generate Summary", variant="primary")

            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                lines=8,
                label="Generated Summary"
            )

    summarize_btn.click(
        fn=summarize_dialogue,
        inputs=dialogue_input,
        outputs=summary_output
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[dialogue_input, summary_output]
    )

    gr.Examples(
        examples=examples,
        inputs=dialogue_input
    )

    gr.Markdown(
        """
        ---
        This demo runs on CPU infrastructure.  
        Do not enter sensitive information.  
        Only for educational purposes. Just a demo.
        """
    )

demo.launch()