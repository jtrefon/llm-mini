import argparse
import torch
from transformers import AutoTokenizer
from model import GPTMini, GPTConfig
import pytorch_lightning as pl


def load_from_lightning_ckpt(ckpt_path: str, tokenizer_name: str, device: str = 'mps'):
    # Rebuild config from tokenizer + hyperparams saved in Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    # Prefer tokenizer from checkpoint hparams if requested
    tk_from_ckpt = None
    try:
        tk_from_ckpt = hparams.get('data', {}).get('tokenizer_name') if isinstance(hparams, dict) else None
    except Exception:
        tk_from_ckpt = None
    effective_tok_name = tokenizer_name
    if tokenizer_name in ('auto', 'infer', None) and tk_from_ckpt:
        effective_tok_name = tk_from_ckpt
        print(f"[infer] Using tokenizer from checkpoint: {effective_tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(effective_tok_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    try:
        tokenizer.model_max_length = int(1e9)
    except Exception:
        pass

    # Pull model dims from saved hyperparameters; fall back to safe defaults if needed
    model_cfg = hparams.get('model', {}) if isinstance(hparams, dict) else {}
    mcfg = GPTConfig(
        vocab_size=len(tokenizer),
        n_layers=int(model_cfg.get('n_layers', 16)),
        d_model=int(model_cfg.get('d_model', 768)),
        n_heads=int(model_cfg.get('n_heads', 12)),
        n_kv_heads=int(model_cfg.get('n_kv_heads', 2)),
        d_ff=int(model_cfg.get('d_ff', 3072)),
        dropout=float(model_cfg.get('dropout', 0.0)),
        rope_theta=float(model_cfg.get('rope_theta', 1000000.0)),
        tie_embeddings=bool(model_cfg.get('tie_embeddings', True)),
        swa_window=int(model_cfg.get('swa_window', 0)),
    )
    model = GPTMini(mcfg)

    # Load state dict with compatibility for Lightning's 'net.' prefix
    state = ckpt.get('state_dict', ckpt)
    if any(isinstance(k, str) and k.startswith('net.') for k in state.keys()):
        state = {k.split('net.', 1)[1]: v for k, v in state.items() if k.startswith('net.')}
    load_compat = model.load_state_dict(state, strict=False)
    try:
        missing = getattr(load_compat, 'missing_keys', [])
        unexpected = getattr(load_compat, 'unexpected_keys', [])
        if missing or unexpected:
            print(f"[infer] Warning: missing={len(missing)} unexpected={len(unexpected)} state dict keys; generation may be affected.")
    except Exception:
        pass

    preferred = (device or "auto").lower()
    mps_available = bool(
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    cuda_available = torch.cuda.is_available()
    if preferred == "cuda":
        dev = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
    elif preferred == "mps":
        dev = torch.device("mps" if mps_available else ("cuda" if cuda_available else "cpu"))
    elif preferred == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if cuda_available else ("mps" if mps_available else "cpu"))
    model.to(dev)
    model.eval()
    return model, tokenizer, dev

def compute_ppl(model, tokenizer, device, text):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(inputs['input_ids'])
    labels = inputs['input_ids']
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()


def generate_text(model, tokenizer, device, prompt: str, max_new_tokens=128, temperature=0.2, top_p=0.9, top_k: int = 0, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0, beam_width=1):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out_ids = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    # Only return generated continuation (exclude the prompt prefix)
    gen_ids = out_ids[0, inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Trim leading whitespace/newlines to avoid "empty page" effect
    cleaned = text.lstrip()
    if cleaned.strip() == "":
        # Fallback: retry with broader sampling and disabled early EOS to force some content
        with torch.no_grad():
            out_ids2 = model.generate(
                inputs['input_ids'],
                max_new_tokens=max(max_new_tokens, 64),
                temperature=max(1.0, temperature),
                top_p=1.0,
                top_k=0,
                repetition_penalty=max(1.0, repetition_penalty),
                eos_token_id=None,
                no_repeat_ngram_size=max(3, int(no_repeat_ngram_size or 0)),
            )
        gen_ids2 = out_ids2[0, inputs['input_ids'].shape[1]:]
        cleaned2 = tokenizer.decode(gen_ids2, skip_special_tokens=True).lstrip()
        if cleaned2.strip() != "":
            return cleaned2
        # As a last resort, return a placeholder rather than blank output
        return "(no non-whitespace output)"
    return cleaned


def is_alpaca_enveloped(text: str) -> bool:
    """Detect if text already contains the Alpaca-style instruction/response envelope."""
    t = (text or "").strip()
    return ("### Instruction:" in t) and ("### Response:" in t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to Lightning .ckpt file')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='mps', choices=['mps','cuda','cpu'])
    parser.add_argument('--prompt', type=str, default='Once upon a time')
    parser.add_argument('--instruct', action='store_true', help='Force-wrap the prompt in an Alpaca-style template (auto-detected otherwise).')
    parser.add_argument('--no_wrap', action='store_true', help='Disable Alpaca-style prompt wrapping entirely (raw completion mode).')
    parser.add_argument('--disable_eos', action='store_true', help='Disable EOS stopping (useful if generations end too early).')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help='Prevent repeating any n-gram of this size (0 to disable).')
    parser.add_argument('--beam_width', type=int, default=1, help='Beam search width (1 for greedy)')
    parser.add_argument('--compute_ppl', action='store_true', help='Compute perplexity on generated text')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        try:
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # Auto-wrap prompt in Alpaca-style envelope unless it's already enveloped.
    if not args.no_wrap:
        if args.instruct or not is_alpaca_enveloped(args.prompt):
            if not is_alpaca_enveloped(args.prompt):
                user = args.prompt.strip()
                args.prompt = f"### Instruction:\n{user}\n\n### Response:\n"

    model, tok, dev = load_from_lightning_ckpt(args.ckpt, tokenizer_name=args.tokenizer, device=args.device)
    if args.disable_eos:
        try:
            tok.eos_token_id = None
        except Exception:
            pass
    text = generate_text(
        model,
        tok,
        dev,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=max(0, args.no_repeat_ngram_size),
        beam_width=args.beam_width,
    )
    print('\n=== Generation ===\n')
    print(text)
    if args.compute_ppl:
        ppl = compute_ppl(model, tok, dev, args.prompt + text)
        print(f'PPL: {ppl:.2f}')
