import torch
import whisper
from whisper.tokenizer import get_tokenizer

def predict_text(embeddings, text, maxLen=25):
    print("Predicting text from embeddings...")
    print(text)

    model = whisper.load_model("base")
    model.eval()

    tokenizer = get_tokenizer(multilingual=model.is_multilingual)

    # Start Of Transcript token
    tokens = torch.tensor([[tokenizer.sot]]).to(model.device)

    embeddings = embeddings.to(model.device)
    embeddings = embeddings.unsqueeze(0)
    with torch.no_grad():
        for _ in range(maxLen):
            logits = model.decoder(tokens, embeddings)

            # Greedy decoding
            nextToken = torch.argmax(logits[:, -1, :], dim=-1)

            # Append token (fix dimension issue)
            tokens = torch.cat([tokens, nextToken.unsqueeze(1)], dim=1)

            if nextToken.item() == tokenizer.eot:
                break

    text = tokenizer.decode(tokens[0].tolist())
    print(text)

    return text