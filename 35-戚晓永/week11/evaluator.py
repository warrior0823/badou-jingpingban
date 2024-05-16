import torch


def eval(model, tokenizer, prompt):
    prompt = tokenizer.encode(prompt)
    answer = []
    model.eval()
    with torch.no_grad():
        while True:
            pred = model(torch.LongTensor([prompt + answer]))
            answer += [pred.item()]
            if len(prompt + answer) > 511 or pred == tokenizer.sep_token_id:
                return tokenizer.decode(answer)