# Failed attempts: seven LLM fine-tuning runs

Seven attempts to fine-tune Gemma 2 9B and Qwen3.5-2B to predict exam marks.
None produced a usable prediction. The scripts are kept here as a record of what
was run.

## What these runs establish

They establish that these seven configurations, built this way, on this data,
did not work.

They do not establish that language models cannot classify. The earlier version
of this document concluded that "LLMs are built to generate text tokens, not
predict numeric scores" and that the task was a fundamental mismatch. That
conclusion is wrong, and the section below says why.

## Summary

| # | Model | Method | Data | Loss | Output | Accuracy |
|---|---|---|---|---|---|---|
| 1 | Gemma 2 9B | causal LM | synthetic 400 | 9.78 | constant 0.0 | 0% |
| 2 | Gemma 2 9B | instruction SFT | synthetic 400 | 6.98 | constant 0.0 | 0% |
| 3 | Gemma 2 9B | masked training | synthetic 400 | 6.38 | constant 0.0 | 0% |
| 4 | Qwen3.5-2B | base trainer | real 122 | n/a | crash in the vision processor | n/a |
| 5 | Qwen3.5-2B | custom collator | real 122 | n/a | `Qwen3VLProcessor` has no `.pad` | n/a |
| 6 | Qwen3.5-2B | categorical A/B/C/D | synthetic 400 | 35.88 | punctuation tokens | 13% |
| 7 | Gemma 2 9B | minimal test | synthetic 400 | 20.62 | empty strings | 0% |

Four of the seven trained on the 400 synthetic answers, which no model in
`results/` uses and which `DATA_CARD.md` recommends against. Two never trained
at all.

## Why the diagnosis was wrong

Fine-tuning a decoder to emit a label is a standard and well-supported thing to
do. Three routes were available and none of the seven runs took any of them.

**A classification head.** Load the base model under
`AutoModelForSequenceClassification` with `num_labels=3` and train with
cross-entropy over the three bands. This is the direct form of the task. It
needs no generation at all.

**Constrained decoding.** Keep the generative setup and restrict the output
distribution to the label tokens, then take the argmax over those. This removes
the failure mode seen in runs 1, 2, 3 and 7, where the model minimized loss by
emitting a constant string.

**Scoring the candidates.** Compute the log-likelihood the model assigns to each
of the three labels given the prompt, and pick the highest. This works with no
fine-tuning at all and would have made a useful zero-shot baseline, which the
project never established.

Loss did converge in runs 1, 2 and 3. Converged loss on a free-form generation
objective says the model found a low-entropy output. Emitting a constant is a
low-entropy output. Nothing about that constitutes evidence about the task.

## The specific failures

**Runs 1, 2, 3, 7.** Free-form generation fine-tunes where the model settled on
a constant output. The loss was computed over the whole sequence, so the label
token contributed a small fraction of it. The signal that mattered was diluted
into the prompt tokens.

**Runs 4 and 5.** Ordinary API bugs. A vision-language checkpoint was loaded for
a text-only task, so the processor expected image inputs and then lacked the
`.pad` method the collator called. Neither run trained. Loading a text
checkpoint would have avoided both.

**Run 6.** Marks were mapped to the letters A, B, C and D, and the script
asserted specific token ids for them. The model emitted punctuation instead.
The 13% figure is below the majority-class rate of 37.7% and carries no
information.

Across all seven, no hyperparameter sweep was run. Learning rate, LoRA rank,
sequence length and epoch count were each set once.

## What the comparison with the sklearn models is worth

The earlier version of this document closed by scoring the sklearn models at
56% and 68% against the LLM runs at 0%, and read that as the simple approach
winning.

The 68% has since been withdrawn. `results/RESULTS.md` puts the best
cross-validated model at 63.2%, against a text-free per-question majority rule
at 63.1%. The sklearn side of that comparison does not clear its own baseline,
so the comparison decides nothing.

## What would test the question properly

An honest LLM comparison on this dataset would run a zero-shot likelihood
baseline first, since it costs nothing and needs no training. Then a
classification head on a text encoder, cross-validated the same way as
`scripts/evaluate.py`, and reported against the same per-question majority
floor. On 122 answers, the likely outcome is that it also fails to clear the
floor. That would be a result.

## Files

| file | attempt |
|---|---|
| `finetune_gemma2_quick_improved.py` | 1 |
| `finetune_gemma2_slow_production_improved.py` | 2 |
| `finetune_gemma2_masked.py` | 3 |
| `finetune_qwen35_2b_base_trainer.py` | 4 |
| `finetune_qwen35_2b_categorical.py` | 5 and 6 |
| `gemma2_fast_test.py` | 7 |
| `debug_gemma2_output.py` | output inspection for the Gemma runs |
| `debug_raw_output.py` | output inspection for the Qwen runs |
| `train_xgboost_on_synthetic.py` | an XGBoost regressor trained on the synthetic file, kept here for the same reason |

These scripts need `transformers`, `torch`, `unsloth` and `xgboost`, none of
which are in `requirements.txt`. Nothing in `results/` depends on them.
