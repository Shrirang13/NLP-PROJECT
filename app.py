"""
Gradio GUI for Hinglish error classification + correction.

Workflow:
Input sentence -> preprocess -> classify error type -> run correction pipeline -> show outputs.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from modules.gui_utils import classify_and_correct


project_root = Path(__file__).resolve().parent
data_dir = project_root / "data"
models_dir = project_root / "models"


def predict(text: str):
    if not text.strip():
        return "", "", 0.0
    label, corrected, confidence = classify_and_correct(
        text=text,
        data_dir=data_dir,
        models_dir=models_dir,
    )
    return label, corrected, confidence


with gr.Blocks(title="Hinglish Error Detection and Correction") as demo:
    gr.Markdown("## Morphology-Aware Hinglish Error Classification and Correction")
    gr.Markdown(
        "Enter a Hinglish sentence. The system will classify the type of error and "
        "show the corrected output using the existing rule-based pipeline."
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Input Hinglish Sentence",
            placeholder="e.g. mai kal colleg ja rha hu",
            lines=2,
        )
    with gr.Row():
        label_out = gr.Textbox(label="Predicted Error Class (spelling / grammar / repetition / normalization / clean)")
    with gr.Row():
        corrected_out = gr.Textbox(label="Corrected Sentence")
    with gr.Row():
        confidence_out = gr.Number(label="Confidence Score (if available)", precision=4)

    run_btn = gr.Button("Classify and Correct")
    run_btn.click(
        predict,
        inputs=[inp],
        outputs=[label_out, corrected_out, confidence_out],
    )

if __name__ == "__main__":
    demo.launch()

