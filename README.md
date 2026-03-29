# Fine-tuning FLAN-T5 on DialogSum with PEFT-LoRA

Fine-tuned `google/flan-t5-base` for dialogue summarization on the DialogSum dataset using Parameter-Efficient Fine-Tuning (LoRA). Only 0.36% of model parameters were updated during training (884K out of 250M). The LoRA adapters were merged into the base model after training for standard inference without any adapter dependency.

Live demo: [HuggingFace Spaces](https://sanchitvp-finetuned-flan-t5-base-dialogsum.hf.space/)  
Model: [sanchitvp/flan-t5-base-finetuned-dialogsum](https://huggingface.co/sanchitvp/flan-t5-base-finetuned-dialogsum)  
Notebook: [Kaggle](https://www.kaggle.com/code/sanchitpathak/finetuning-flan-t5-on-dialogsum-with-peft-lora)

---

## Project Structure

```
.
├── notebook/
│   └── finetuning-flan-t5-on-dialogsum-with-peft-lora.ipynb
├── app/
│   └── app.py          # Gradio inference app (HuggingFace Spaces)
└── README.md
```

---

## Task and Dataset

**Task:** Abstractive dialogue summarization  
**Dataset:** [DialogSum](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum) - a benchmark dataset of multi-speaker conversations paired with human-written summaries.

---

## Approach

Standard full fine-tuning of a 250M parameter model is compute-intensive and updates all weights. This project uses LoRA (Low-Rank Adaptation), which injects small trainable rank-decomposition matrices into the transformer's attention layers while keeping the original weights frozen. This reduces the trainable parameter count from 250M to 884K.

After training, the LoRA adapter weights are merged back into the base model using `merge_and_unload()`. The final checkpoint behaves as a standard Seq2Seq model with no PEFT dependency at inference time.

**LoRA configuration:**
- Rank (r): 8
- Alpha: 32
- Target modules: query and value projection layers
- Trainable parameters: 884,736 (0.36% of total)

---

## Training Setup

| Parameter         | Value                  |
|-------------------|------------------------|
| Base model        | google/flan-t5-base    |
| Dataset           | DialogSum              |
| Framework         | Transformers + PEFT    |
| Backend           | PyTorch                |
| LoRA rank         | 8                      |
| LoRA alpha        | 32                     |
| Trainable params  | 884K / 250M (0.36%)    |
| Environment       | Kaggle Notebook (GPU)  |

---

## Evaluation

Evaluated on the DialogSum test split using ROUGE metrics.

| Metric  | Score |
|---------|-------|
| ROUGE-1 | 0.484 |
| ROUGE-2 | 0.236 |
| ROUGE-L | 0.399 |

---

## Installation

```bash
pip install transformers torch
```

## Inference

Since adapters are merged, the model loads like any standard HuggingFace Seq2Seq checkpoint.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sanchitvp/flan-t5-base-finetuned-dialogsum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dialogue = """
Person A: Hey, are we still meeting tomorrow?
Person B: Yes, at 10 AM. Don't forget the documents.
Person A: Got it. See you then.
"""

input_text = "Summarize the following dialogue:\n" + dialogue

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
)

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

## Deployment

The inference app is built with Gradio and hosted on HuggingFace Spaces. It runs on CPU infrastructure and accepts multi-turn dialogue text as input. Six example dialogues are included to demonstrate generalization across conversation types (detective fiction, job interviews, engineering discussions, and others).

---

## Limitations

- Trained specifically on DialogSum; may not generalize well to highly domain-specific or very long conversations.
- No safety fine-tuning applied.
- Runs on CPU in the live demo; latency will be higher than GPU inference.

---

## License

Apache-2.0, inherited from `google/flan-t5-base`.

---

## Author

Sanchit Pathak  
[HuggingFace](https://huggingface.co/sanchitvp) | [Kaggle](https://www.kaggle.com/sanchitpathak)
