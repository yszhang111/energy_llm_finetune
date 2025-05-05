import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from loguru import logger
import time

# --- CONFIGURATION SECTION ---
# !! EDIT THESE VALUES BEFORE RUNNING !!

# 1. Paths and Model ID
TOKENIZER_NAME_OR_PATH = "deepseek-ai/DeepSeek-R1" # Change to your exact model ID (e.g., "deepseek-ai/deepseek-coder-7b-instruct-v1.5")
JSONL_INPUT_PATH = "./data/datasets-1746398605127-alpaca-2025-05-05.jsonl" # <--- SET Your input dataset file path here
OUTPUT_DIR = "./data/tokenized_data1" # <--- SET Where to save the tokenized output

# 2. Tokenization Settings
MAX_SEQ_LENGTH = 2048 # Maximum sequence length for truncation

# 3. JSONL Data Keys (Based on your sample: {"instruction": ..., "input": ..., "output": ...})
#    If your keys are different (e.g., "prompt", "completion"), change these!
INSTRUCTION_KEY = "instruction"
INPUT_KEY = "input"
OUTPUT_KEY = "output" # The key containing the desired model response

# 4. Processing Parameters
NUM_PROC = 1 # Use half the CPU cores (or at least 1)
BATCH_SIZE = 1000 # How many rows to process at once in .map()

# --- END CONFIGURATION SECTION ---


# Global tokenizer to avoid reloading in each process
tokenizer = None

# --- Tokenization Function ---
def tokenize_and_format(batch, instruction_key, input_key, output_key, max_seq_length):
    """
    Tokenizes a batch of examples and prepares them for causal LM SFT.
    Assumes instruction/input/output format based on hardcoded keys.
    Masks prompt tokens in labels.
    """
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. This should not happen in map.")

    full_prompts = []
    texts_to_tokenize = []

    # Construct prompt and full text based on hardcoded keys
    for i in range(len(batch[output_key])):
        instruction = batch[instruction_key][i] if instruction_key in batch and batch[instruction_key][i] else ""
        input_ctx = batch[input_key][i] if input_key in batch and batch[input_key][i] else ""
        completion = batch[output_key][i]

        # Construct prompt (Alpaca-style common)
        if instruction and input_ctx:
            prompt_text = f"Instruction:\n{instruction}\n\nInput:\n{input_ctx}\n\nOutput:\n"
        elif instruction:
            prompt_text = f"Instruction:\n{instruction}\n\nOutput:\n"
        elif input_ctx: # Less common, but possible
            prompt_text = f"Input:\n{input_ctx}\n\nOutput:\n"
        else: # Minimal prompt if only output exists
            prompt_text = "Output:\n"
            logger.warning(f"Example missing instruction/input keys.")

        full_prompts.append(prompt_text)
        # Add EOS token to the completion for Causal LM training
        texts_to_tokenize.append(prompt_text + completion + tokenizer.eos_token)

    # Tokenize the combined texts
    model_inputs = tokenizer(
        texts_to_tokenize,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=True, # Add BOS if configured in tokenizer
    )

    # Tokenize prompts separately *without* EOS to find length for masking
    prompt_tokens = tokenizer(
        full_prompts,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=True, # Add BOS if configured in tokenizer
    )

    # Prepare labels - mask prompt tokens
    labels = []
    for i in range(len(model_inputs["input_ids"])):
        prompt_len = len(prompt_tokens["input_ids"][i])
        label_ids = list(model_inputs["input_ids"][i]) # Make a mutable copy
        # Mask tokens belonging to the prompt
        label_ids[:prompt_len] = [-100] * prompt_len
        labels.append(label_ids)

    model_inputs["labels"] = labels

    return model_inputs


# --- Main Execution ---
if __name__ == "__main__":
    # Setup logging (Logs will go to the specified OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output dir exists for logging
    log_file_path = os.path.join(OUTPUT_DIR, "tokenization.log")
    logger.add(log_file_path, rotation="10 MB")
    logger.info("--- Starting Tokenization Script (Hardcoded Config) ---")
    logger.info(f"Tokenizer: {TOKENIZER_NAME_OR_PATH}")
    logger.info(f"Input File: {JSONL_INPUT_PATH}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
    logger.info(f"Instruction Key: '{INSTRUCTION_KEY}', Input Key: '{INPUT_KEY}', Output Key: '{OUTPUT_KEY}'")
    logger.info(f"Num Processes: {NUM_PROC}, Batch Size: {BATCH_SIZE}")

    start_time = time.time()

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME_OR_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME_OR_PATH,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token ({tokenizer.eos_token})")
        globals()['tokenizer'] = tokenizer # Make available to map function
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{TOKENIZER_NAME_OR_PATH}': {e}")
        exit(1)

    # 2. Load Dataset
    logger.info(f"Loading dataset from: {JSONL_INPUT_PATH}")
    try:
        raw_dataset = load_dataset("json", data_files=JSONL_INPUT_PATH, split="train")
        logger.info(f"Raw dataset loaded: {raw_dataset}")
        # Validate expected keys
        required_keys = {INSTRUCTION_KEY, INPUT_KEY, OUTPUT_KEY} - {None, ""} # Remove None/empty keys from check if not used
        missing_keys = required_keys - set(raw_dataset.column_names)
        if missing_keys:
             logger.warning(f"Dataset columns {raw_dataset.column_names} might be missing expected keys based on config: {missing_keys}. Check CONFIGURATION SECTION.")
        # Ensure output key exists
        if OUTPUT_KEY not in raw_dataset.column_names:
            logger.error(f"Output key '{OUTPUT_KEY}' not found in dataset columns: {raw_dataset.column_names}. Cannot proceed.")
            exit(1)

    except Exception as e:
        logger.error(f"Failed to load dataset from '{JSONL_INPUT_PATH}': {e}")
        exit(1)

    # 3. Tokenize Dataset
    logger.info(f"Tokenizing dataset with max_seq_length={MAX_SEQ_LENGTH} using {NUM_PROC} processes...")
    tokenization_start_time = time.time()

    fn_kwargs = {
        "instruction_key": INSTRUCTION_KEY,
        "input_key": INPUT_KEY,
        "output_key": OUTPUT_KEY,
        "max_seq_length": MAX_SEQ_LENGTH
    }

    tokenized_dataset = raw_dataset.map(
        tokenize_and_format,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=raw_dataset.column_names
    )
    tokenization_end_time = time.time()
    logger.info(f"Tokenization finished in {tokenization_end_time - tokenization_start_time:.2f} seconds.")
    logger.info(f"Tokenized dataset info: {tokenized_dataset}")
    if len(tokenized_dataset) > 0:
        logger.info(f"Example tokenized sample [0]: {tokenized_dataset[0]}")
    else:
        logger.warning("Tokenized dataset appears to be empty.")


    # 4. Save Tokenized Dataset
    logger.info(f"Saving tokenized dataset to: {OUTPUT_DIR}")
    try:
        # save_to_disk automatically creates the directory if needed
        tokenized_dataset.save_to_disk(OUTPUT_DIR)
        logger.info("Dataset saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save tokenized dataset to '{OUTPUT_DIR}': {e}")
        exit(1)

    end_time = time.time()
    logger.info(f"--- Script finished successfully in {end_time - start_time:.2f} seconds. ---")