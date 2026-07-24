import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk

# data
from data_modules.configs import DatasetParams, TokenizerParams

# model
from modules.transformer import Translator
from modules.configs import ModelParams

# training
from configs import TrainingParams
from utils import load_dataset, load_tokenizer


def evaluate():
    pass


def train(train_params: TrainingParams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset, dataloder, and tokenizer
    dp = DatasetParams(dataset_name="rhyliieee/tagalog-filipino-english-translation")
    ds = load_dataset(dp, online=False, path="./data_modules/data", split="train")
    train_dataloader = DataLoader(dataset=ds, batch_size=train_params.batch_size, shuffle=True)

    tp = tokenizer_params = TokenizerParams(tokenizer_name="Helsinki-NLP/opus-mt-tl-en")
    tokenizer = load_tokenizer(tp, online=False, path="./data_modules/tokenizers")

    #print(dir(tokenizer))
    #exit()

    # load the model and training loop
    model_params = ModelParams(
        padding_idx=tokenizer.pad_token_id,
        in_vocab_size=tokenizer.vocab_size,
        out_vocab_size=tokenizer.vocab_size
    )
    translator = Translator(model_params=model_params)
    translator.to(device)

    # training loop
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # ignore padding tokens in the loss computation
    optimizer = torch.optim.Adam(translator.parameters(), lr=train_params.learning_rate) # lets optimize learning rate at some point, make it schedulable

    for epoch in range(train_params.max_epochs): # just one epoch for now
        translator.train()
        print(f"Epoch {epoch+1}/{train_params.max_epochs}")
        for batch,data in enumerate(train_dataloader): # just one batch for now
            src_texts, tgt_texts = data[dp.source_language], data[dp.target_language]
            #print(src_texts, tgt_texts)

            inputs = tokenizer(
                text=src_texts,
                text_target=tgt_texts,
                padding=True,
                truncation=True,
                max_length=model_params.max_len,
                return_tensors="pt"
            )

            # print(inputs.keys())

            #print(tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id)
            X, Y = inputs["input_ids"], inputs["labels"]
            #print(X.shape, Y.shape)

            decoder_input = Y[:, :-1].to(device)  # Remove the last token for decoder input
            decoder_target = Y[:, 1:].to(device)  # Remove the first token for decoder target


            logits = translator(X.to(device), decoder_input) 
            #print(logits.shape, decoder_target.shape)

            # compute loss
            optimizer.zero_grad()
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),  # BxLxV -> (BxL)xV
                decoder_target.reshape(-1)  # BxL -> (BxL)
            )
            #print(loss)
            loss.backward() #propagate
            optimizer.step() #update

            if batch % 100 == 0:
                print(f"Batch {batch}, Loss: {loss.item()}")





if __name__ == "__main__":
    train_params = TrainingParams(
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=1
    )
    print(f"Training parameters: {train_params}")
    train(train_params)

