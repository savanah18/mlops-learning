from transformers import AutoTokenizer


# jcblaise/roberta-tagalog-base
# jcblaise/bert-tagalog-base-cased
# GKLMIP/roberta-tagalog-base

if __name__ == "__main__":    
    from configs import DatasetParams, TokenizerParams
    tp = TokenizerParams(tokenizer_name="jcblaise/roberta-tagalog-base")
    # tokenizer = AutoTokenizer.from_pretrained(tp.tokenizer_name, **tp.tokenizer_args, **tp.tokenizer_kwargs)
    # tokenizer.save_pretrained(f"./tokenizers/{tp.tokenizer_name}")

    src_tokenizer = AutoTokenizer.from_pretrained(tp.tokenizer_name, **tp.tokenizer_args, **tp.tokenizer_kwargs)
    src_tokenizer.save_pretrained(f"./tokenizers/{tp.tokenizer_name}")
    # load 
    # src_tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{tp.tokenizer_name}")

    tgt_tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")



    src_sentences = [
        "Hmmm, San ka punta?",
        "May pasok kaya bukas?"
    ]

    tgt_sentences = [
        "Hmm, where are you going?",
        "Is there class tomorrow?"
    ]

    inputs = src_tokenizer(
        text=src_sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    print(inputs)
    # TODO , Use a tokenizer that is bilingual or multilingual, or train a custom tokenizer on the Tagalog-English dataset. This will help in better tokenization and understanding of both languages.

    tp = TokenizerParams(tokenizer_name="Helsinki-NLP/opus-mt-tl-en")
    tokenizer = AutoTokenizer.from_pretrained(tp.tokenizer_name, **tp.tokenizer_args, **tp.tokenizer_kwargs)
    src_tokenizer.save_pretrained(f"./tokenizers/{tp.tokenizer_name}")
    tagalog_texts = ["May pasok kaya bukas?", "Magandang umaga sa inyo."]
    english_targets = ["Is there work tomorrow?", "Good morning to you."]

    inputs = tokenizer(
        text=tagalog_texts,
        text_target=english_targets,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    print(inputs.keys())
    # dict_keys(['input_ids', 'attention_mask', 'labels'])

    print("Source Token IDs:", inputs["input_ids"])
    print("Target Label IDs:", inputs["labels"])