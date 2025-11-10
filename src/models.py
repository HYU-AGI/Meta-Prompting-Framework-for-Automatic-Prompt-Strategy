from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName  # type: ignore
    _HARMONY = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    _HARMONY_EOS: Optional[List[int]] = _HARMONY.stop_tokens_for_assistant_actions()
except Exception:
    _HARMONY = None
    _HARMONY_EOS = None


SUPPORTED_MODELS = {
    "gpt-oss-20b",
    "qwen3-8b",
    "qwen3-14b",
    "deepseek-v2-16b",
    "llama-3b",
    "llama-8b",
    "mistral-7b",
}


@dataclass
class Args:
    model_name: str
    model_id: str
    cache_dir: Optional[str] = None
    max_new_tokens: int = 512
    do_sample: bool = True
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    temperature: float = 0.7


def _inject_reasoning_low(messages):
    has_low = any(m.get("role") == "system" and "Reasoning:" in m.get("content", "") for m in messages)
    if not has_low:
        return [{"role": "system", "content": "Reasoning: low"}] + messages
    return messages


def _extract_harmony_final(decoded: str) -> Optional[str]:
    start_tag = "<|channel|>final<|message|>"
    end_tag = "<|end|>"
    if start_tag in decoded:
        part = decoded.split(start_tag, 1)[-1]
        if end_tag in part:
            part = part.split(end_tag, 1)[0]
        return part.strip()
    start_alt = "<|final|>"
    if start_alt in decoded:
        part = decoded.split(start_alt, 1)[-1]
        if end_tag in part:
            part = part.split(end_tag, 1)[0]
        return part.strip()
    return None


def load_model(args: Args) -> Tuple[Any, Any]:
    if args.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Local model {args.model_name} is not supported.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_kernels=False
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("use_kernels", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    return model, tokenizer


def generate_local_response(
    model: Any,
    tokenizer: Any,
    messages,
    args: Args,
    forced_prefix: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    try:
        model.generation_config = GenerationConfig.from_pretrained(args.model_id)
    except Exception:
        pass

    if args.model_name == "deepseek-v2-16b":
        try:
            model.config.use_cache = False
        except Exception:
            pass

    inputs = _apply_chat_template(tokenizer, args.model_name, messages).to(model.device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else getattr(model.generation_config, "eos_token_id", None)
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    if pad_id is not None:
        model.generation_config.pad_token_id = pad_id

    prefix_len = 0
    if forced_prefix:
        if args.model_name.startswith("gpt-oss"):
            forced = "<|channel|>final<|message|>" + forced_prefix
        else:
            forced = forced_prefix
        pref_ids = tokenizer(forced, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        prefix_len = pref_ids.shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], pref_ids], dim=1)
        if "attention_mask" in inputs:
            pref_mask = torch.ones_like(pref_ids)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pref_mask], dim=1)

    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=args.max_new_tokens)
    if args.do_sample:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_p=args.top_p if args.top_p is not None else 0.95,
                top_k=args.top_k if args.top_k is not None else 50,
                temperature=None if args.temperature == 0 else args.temperature,
            )
        )
    else:
        gen_kwargs.update(dict(do_sample=False))

    if args.model_name.startswith("gpt-oss") and _HARMONY_EOS:
        gen_kwargs["eos_token_id"] = _HARMONY_EOS

    outputs = model.generate(**inputs, **gen_kwargs)

    if args.model_name.startswith("gpt-oss"):
        decoded_full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        final_only = _extract_harmony_final(decoded_full)
        if final_only is None:
            start = prompt_len - prefix_len if forced_prefix else prompt_len
            generated = tokenizer.decode(outputs[0][start:], skip_special_tokens=True).strip()
        else:
            generated = final_only.strip()
    else:
        start = prompt_len - prefix_len if forced_prefix else prompt_len
        generated = tokenizer.decode(outputs[0][start:], skip_special_tokens=True).strip()

    stats = {
        "input_len": prompt_len,
        "output_len": int(outputs[0].shape[0] - prompt_len),
        "prefix_len": prefix_len,
    }
    return generated, stats

def _apply_chat_template(tokenizer, model_name: str, messages, *, as_text: bool = False):
    if model_name.startswith("gpt-oss"):
        messages = _inject_reasoning_low(messages)
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )
    elif model_name.startswith("qwen3"):
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, enable_thinking=False, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=not as_text, add_generation_prompt=True,
            return_tensors=None if as_text else "pt",
            return_dict=not as_text
        )


def _force_eager_attn(model):
    try:
        if hasattr(model, "set_default_attn_implementation"):
            model.set_default_attn_implementation("eager")
    except Exception:
        pass
    try:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    except Exception:
        pass


def _prepare_sdp_env(disable_flash: bool):
    try:
        if disable_flash:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


@torch.no_grad()
def step_decode_with_bias(
    model: Any,
    tokenizer: Any,
    messages,
    args: Args,
    *,
    forced_prefix: Optional[str] = None,
    alpha: float = 0.0,
    stop_at: str = "</IMPLEMENT>",
    
    marker_m1_only: bool = False,
    marker_open_str: str = "[M1]",
    marker_close_str: str = "[/M1]",
    keyword_str: Optional[str] = None,
    keyword_casefold: bool = True,
    
    marker_gain_coef: float = 0.30,
    marker_gain_cap: float = 0.50,
    marker_dropout: float = 0.40,
    marker_taper: bool = False,
    
    marker_gain_neg_cap: Optional[float] = None,
    
    delta_gain_coef: float = 1.0,
    delta_gain_cap: float = 1.0,
    delta_gain_neg_cap: Optional[float] = None,
    
    # gpt-oss 호환
    force_eager_attn: bool = False,
    disable_sdp_flash: bool = False,
) -> Tuple[str, Dict[str, int]]:

    device = next(model.parameters()).device
    model.eval()

    if force_eager_attn:
        _force_eager_attn(model)
    if disable_sdp_flash:
        _prepare_sdp_env(disable_flash=True)

    if marker_gain_neg_cap is None:
        marker_gain_neg_cap = float(getattr(args, "marker_gain_neg_cap", marker_gain_cap))
    if delta_gain_neg_cap is None:
        delta_gain_neg_cap = float(getattr(args, "delta_gain_neg_cap", delta_gain_cap))

    rendered: str = _apply_chat_template(tokenizer, args.model_name, messages, as_text=True)
    if forced_prefix:
        if args.model_name.startswith("gpt-oss"):
            rendered += "<|channel|>final<|message|>" + forced_prefix
        else:
            rendered += forced_prefix

    enc_all = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc_all["input_ids"].to(device)
    offsets = enc_all["offset_mapping"][0].tolist()
    prompt_len = input_ids.size(1)

    m1_found = False
    m1_token_idxs: List[int] = []


    if marker_m1_only:
        start_char = rendered.find(marker_open_str)
        end_char_tag = rendered.find(marker_close_str)
        if start_char >= 0 and end_char_tag > start_char:
            open_end = start_char + len(marker_open_str)
            inner_start = open_end
            inner_end = end_char_tag
            for ti, (s, e) in enumerate(offsets):
                if not (e <= inner_start or s >= inner_end):
                    m1_token_idxs.append(ti)
            m1_found = len(m1_token_idxs) > 0

    if (not marker_m1_only) and keyword_str:
        kw_start = rendered.find(keyword_str)
        if kw_start < 0 and keyword_casefold:
            kw_start = rendered.lower().find(keyword_str.lower())
        if kw_start >= 0:
            inner_start = kw_start
            inner_end = kw_start + len(keyword_str)
            for ti, (s, e) in enumerate(offsets):
                if not (e <= inner_start or s >= inner_end):
                    m1_token_idxs.append(ti)
            m1_found = len(m1_token_idxs) > 0

    scale_gain = max(-float(marker_gain_neg_cap),
                     min(float(alpha) * float(marker_gain_coef), float(marker_gain_cap)))
    delta_gain = max(-float(delta_gain_neg_cap),
                     min(float(alpha) * float(delta_gain_coef), float(delta_gain_cap)))
    used_scaled_embed = False
    scaled_tokens = 0

    if m1_found and (scale_gain != 0.0 or delta_gain != 0.0):
        embed = model.get_input_embeddings()(input_ids)
        kw_ids = None
        kw_vec = None
        if keyword_str:
            tok = tokenizer(keyword_str, add_special_tokens=False, return_tensors="pt")
            kw_ids = tok["input_ids"][0].to(device)
            weight = model.get_input_embeddings().weight  # [V, d]
            try:
                kw_vec = weight.index_select(0, kw_ids).mean(dim=0)  # [d]
            except Exception:
                kw_vec = None

        taper_weights = None
        if marker_taper and len(m1_token_idxs) >= 3:
            n = len(m1_token_idxs)
            center = (n - 1) / 2.0
            taper_weights = []
            for k in range(n):
                d = abs(k - center) / max(1.0, center)
                w = 1.0 - 0.5 * d  # [0.5, 1.0]
                taper_weights.append(w)

        rng = torch.Generator(device=device)
        drop_p = float(marker_dropout)

        eps = 1e-6
        for idx_pos, tok_idx in enumerate(m1_token_idxs):
            if drop_p > 0.0 and torch.rand(1, generator=rng, device=device).item() < drop_p:
                continue

            g_scale = scale_gain
            g_delta = delta_gain
            if taper_weights is not None:
                w = float(taper_weights[idx_pos])
                g_scale = scale_gain * w
                g_delta = delta_gain * w
            h = embed[0, tok_idx, :]

            if g_scale != 0.0:
                factor = 1.0 + g_scale
                if factor < 0.1:
                    factor = 0.1
                h = factor * h

            if g_delta != 0.0:
                if kw_vec is None:
                    r = torch.randn_like(h)
                    r = r - (torch.dot(r, h) / (h.norm() ** 2 + eps)) * h
                    if torch.norm(r) > eps:
                        u = r / (torch.norm(r) + eps)
                    else:
                        u = torch.randn_like(h)
                        u = u / (torch.norm(u) + eps)
                else:
                    proj = (torch.dot(kw_vec, h) / (h.norm() ** 2 + eps)) * h
                    u = kw_vec - proj
                    if torch.norm(u) > eps:
                        u = u / (torch.norm(u) + eps)
                    else:
                        r = torch.randn_like(h)
                        r = r - (torch.dot(r, h) / (h.norm() ** 2 + eps)) * h
                        u = r / (torch.norm(r) + eps)
                h = h + g_delta * u

            embed[0, tok_idx, :] = h
            scaled_tokens += 1

        used_scaled_embed = scaled_tokens > 0

        generated = input_ids
        past_key_values = None
        stop_token = getattr(tokenizer, "eos_token_id", None)

        model_inputs = {"inputs_embeds": embed.to(model.dtype), "use_cache": True, "return_dict": True}
        out = model(**model_inputs)
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        for step in range(args.max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True) if not getattr(args, "do_sample", True) else None
            if next_token is None:
                g = logits
                temperature = getattr(args, "temperature", 0.7)
                top_p = getattr(args, "top_p", 0.95)
                top_k = getattr(args, "top_k", 50)
                if temperature and temperature != 1.0:
                    g = g / temperature
                probs = torch.softmax(g, dim=-1)
                if top_p and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cumsum > top_p).float().argmax(dim=-1, keepdim=True)
                    cutoff = torch.clamp(cutoff, min=1)
                    mask = torch.arange(sorted_probs.size(-1), device=device).unsqueeze(0) <= cutoff
                    filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                    filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)
                    next_sorted = torch.multinomial(filtered, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_sorted)
                elif isinstance(top_k, int) and top_k > 0:
                    topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                    topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)
                    sample_idx = torch.multinomial(topk_vals, num_samples=1)
                    next_token = topk_idx.gather(-1, sample_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if stop_token is not None and next_token.item() == stop_token:
                break

            partial_plain = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
            if stop_at and (stop_at in partial_plain):
                break

            out = model(input_ids=generated[:, -1:], use_cache=True, return_dict=True, past_key_values=past_key_values)
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values

        full_decoded = tokenizer.decode(generated[0], skip_special_tokens=False)
        if args.model_name.startswith("gpt-oss"):
            final_only = _extract_harmony_final(full_decoded)
            if final_only is not None:
                text = final_only
            else:
                text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        else:
            text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

        stats = {
            "prompt_len": int(prompt_len),
            "gen_tokens": int(generated[0, prompt_len:].shape[0]),
            "used_scaled_embed": bool(used_scaled_embed),
            "m1_found": bool(m1_found),
            "m1_span_len": int(len(m1_token_idxs)),
            "scaled_tokens": int(scaled_tokens),
            "marker_eff_gain": float(scale_gain),
            "delta_eff_gain": float(delta_gain),
        }
        return text, stats

    generated = input_ids
    past_key_values = None
    stop_token = getattr(tokenizer, "eos_token_id", None)

    for step in range(args.max_new_tokens):
        model_inputs = {"input_ids": generated if past_key_values is None else generated[:, -1:],
                        "use_cache": True, "return_dict": True}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        out = model(**model_inputs)
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        next_token = torch.argmax(logits, dim=-1, keepdim=True) if not getattr(args, "do_sample", True) else None
        if next_token is None:
            g = logits
            temperature = getattr(args, "temperature", 0.7)
            top_p = getattr(args, "top_p", 0.95)
            top_k = getattr(args, "top_k", 50)
            if temperature and temperature != 1.0:
                g = g / temperature
            probs = torch.softmax(g, dim=-1)
            if top_p and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumsum > top_p).float().argmax(dim=-1, keepdim=True)
                cutoff = torch.clamp(cutoff, min=1)
                mask = torch.arange(sorted_probs.size(-1), device=device).unsqueeze(0) <= cutoff
                filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)
                next_sorted = torch.multinomial(filtered, num_samples=1)
                next_token = sorted_idx.gather(-1, next_sorted)
            elif isinstance(top_k, int) and top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)
                sample_idx = torch.multinomial(topk_vals, num_samples=1)
                next_token = topk_idx.gather(-1, sample_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if stop_token is not None and next_token.item() == stop_token:
            break

        partial_plain = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        if stop_at and (stop_at in partial_plain):
            break

    full_decoded = tokenizer.decode(generated[0], skip_special_tokens=False)
    if args.model_name.startswith("gpt-oss"):
        final_only = _extract_harmony_final(full_decoded)
        if final_only is not None:
            text = final_only
        else:
            text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
    else:
        text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

    stats = {
        "prompt_len": int(prompt_len),
        "gen_tokens": int(generated[0, prompt_len:].shape[0]),
        "used_scaled_embed": False,
        "m1_found": bool(m1_found),
        "m1_span_len": int(len(m1_token_idxs)),
        "scaled_tokens": 0,
        "marker_eff_gain": 0.0,
        "delta_eff_gain": 0.0,
    }
    return text, stats
