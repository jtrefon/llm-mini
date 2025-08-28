import argparse
import torch
from transformers import AutoTokenizer
from model import GPTMini, GPTConfig
import pytorch_lightning as pl


def load_from_lightning_ckpt(ckpt_path: str, tokenizer_name: str, device: str = 'mps'):
    # Rebuild config from tokenizer + default model hyperparams used in training
    # For simplicity, we infer dims by reading the hyperparams saved by Lightning
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    mcfg = GPTConfig(
        vocab_size=len(tokenizer),
        n_layers=hparams['model']['n_layers'],
        d_model=hparams['model']['d_model'],
        n_heads=hparams['model']['n_heads'],
        n_kv_heads=hparams['model']['n_kv_heads'],
        d_ff=hparams['model']['d_ff'],
        dropout=hparams['model']['dropout'],
        rope_theta=hparams['model']['rope_theta'],
        tie_embeddings=hparams['model']['tie_embeddings'],
        swa_window=hparams['model']['swa_window']
    )
    model = GPTMini(mcfg)
    model.load_state_dict(ckpt['state_dict'], strict=False)

    dev = torch.device('mps' if device == 'mps' and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(dev)
    model.eval()
    return model, tokenizer, dev


def generate_text(model, tokenizer, device, prompt: str, max_new_tokens=128, temperature=0.8, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out_ids = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to Lightning .ckpt file')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='mps', choices=['mps','cuda','cpu'])
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    model, tok, dev = load_from_lightning_ckpt(args.ckpt, tokenizer_name=args.tokenizer, device=args.device)
    text = generate_text(model, tok, dev, args.prompt, args.max_new_tokens, args.temperature, args.top_p)
    print('\n=== Generation ===\n')
    print(text)
