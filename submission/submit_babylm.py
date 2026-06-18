#!/usr/bin/env python3
"""Submit a Prabhasa entry to the BabyLM 2026 leaderboard via the Gradio API.
Usage:  python submit_babylm.py strict
        python submit_babylm.py strict-small
Run with the PSALM-integration .venv python (has gradio_client + huggingface_hub).
"""
import sys
from gradio_client import Client, handle_file

SPACE = "BabyLM-community/BabyLM-Leaderboard-2026"
PRED = {
    "strict": "/home/sharaths/babylm_submission/prabhasa-b_s_strict_preds.json",
    "strict-small": "/home/sharaths/babylm_submission/prabhasa-b_ss-0.1_strict-small_preds.json",
}

# Shared fields (identical for both entries)
APPROACHES = ["Architectural innovations", "Training objective innovations",
              "Data augmentation", "Linguistic bias", "Controlled experiments"]
GENRE = ("Child-directed speech, Wikipedia, subtitles, books (BabyLM) "
         "+ classical Sanskrit (DCS/GRETIL)")
PREPROC = ("Vidyut morpheme + karaka role annotation; SentencePiece 20k joint "
           "English-Sanskrit; N-hot role vectors precomputed; synthetic Paninian "
           "dose generated via Vidyut-Prakriya.")

PER_TRACK = {
    "strict": dict(
        model="prabhasa-b_s", repo="qbz506/prabhasa-b_s",
        training_data="BabyLM strict", datasize=100, gpu_train=13.7, gpu_dev=50,
        flops=4.7e17,
        desc=("Prabhasa (Paninian Structured Pretraining): a masked-LM encoder "
              "(14L/768d/12h, RoPE, Muon) with three Paninian mechanisms - Vidyut "
              "N-hot morpheme/role embeddings added to token embeddings, karaka "
              "role-stratified budget-matched masking, and a sabdabodha auxiliary "
              "role-prediction objective (discarded at inference). Trained pure-MLM "
              "on the BabyLM Strict English corpus plus a small in-budget Sanskrit "
              "component (DCS/GRETIL) and a synthetic Paninian dose. Submitted "
              "checkpoint is the validated seed-0 model (the recipe shows "
              "training-seed sensitivity at 100M; this is the best validated seed)."),
    ),
    "strict-small": dict(
        model="prabhasa-b_ss-0.1", repo="qbz506/prabhasa-b_ss-0.1",
        training_data="BabyLM strict-small", datasize=10, gpu_train=1.4, gpu_dev=30,
        flops=7.8e16,
        desc=("Prabhasa (Paninian Structured Pretraining), Strict-Small: same "
              "architecture (14L/768d/12h, RoPE, Muon) with Vidyut N-hot morpheme/"
              "role embeddings, karaka role-stratified masking, and a sabdabodha "
              "auxiliary objective. Trained with a hybrid MLM+CLM objective on the "
              "BabyLM Strict-Small English corpus plus a small in-budget Sanskrit "
              "component (DCS/GRETIL). 3-seed model."),
    ),
}


def main():
    track = sys.argv[1] if len(sys.argv) > 1 else "strict"
    assert track in PER_TRACK, f"track must be strict|strict-small, got {track}"
    t = PER_TRACK[track]
    print(f"Connecting to {SPACE} ...")
    c = Client(SPACE)
    print(f"Submitting {t['model']} ({track}) with {PRED[track]} ...")
    res = c.predict(
        t["model"],                 # 0  model name
        "main",                     # 1  revision
        t["repo"],                  # 2  hf repo
        track,                      # 3  track
        handle_file(PRED[track]),   # 4  results_file (our predictions JSON)
        None,                       # 5  predictions_file (multilingual only)
        "Encoder only",             # 6  model_type
        APPROACHES,                 # 7  approaches (list)
        "LTG-BERT",                 # 8  base_model (ELC-BERT lineage)
        "cosine",                   # 9  lr scheduler
        10,                         # 10 epochs
        "SentencePiece (BPE)",      # 11 tokenizer
        "0",                        # 12 random_seed
        12,                         # 13 num_heads
        192,                        # 14 max_seq_len
        t["gpu_dev"],               # 15 gpu_dev (approx)
        t["training_data"],         # 16 training_data
        t["datasize"],              # 17 datasize (M words)
        GENRE,                      # 18 data_genre
        0.001,                      # 19 learning_rate
        "Muon",                     # 20 optimizer
        256,                        # 21 batch_size
        20000,                      # 22 token_set_size
        14,                         # 23 num_layers
        130.3,                      # 24 total_parameters
        t["flops"],                 # 25 flops (approx)
        t["gpu_train"],             # 26 gpu_train
        "No",                       # 27 data_human (no human annotation)
        PREPROC,                    # 28 data_preprocessing
        "No",                       # 29 data_aug (dropdown; synthetic noted in desc/approaches)
        t["desc"],                  # 30 description
        None,                       # 31 other_hyp
        "",                         # 32 teacher_models
        False, False, False,        # 33-35 lang_en/nld/zho
        0, 0, 0,                    # 36-38 num_words_en/nld/zho
        api_name="/submit_and_refresh",
    )
    msg = res[0] if isinstance(res, (list, tuple)) else res
    print("===== SUBMISSION RESULT MESSAGE =====")
    print(msg)


if __name__ == "__main__":
    main()
