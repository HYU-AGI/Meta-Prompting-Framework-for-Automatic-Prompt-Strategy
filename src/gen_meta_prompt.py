import json
from prompts import *
import re
import torch
import math
from typing import Optional
from models import (
    Args as GenArgs,
    step_decode_with_bias,
)


class SelfDiscover:
    def __init__(self, llm, actions, problem, logger, args):
        self.llm = llm
        self.actions = actions
        self.problem = problem
        self.logger = logger
        self.args = args
        self.max_retries = 3

        self.rag_module = f"1. {reasoning_modules[0]}"
        self.using_rag = False
        self.next_step = True
        self.selected_modules = None
        self.adapted_modules = None
        self.reasoning_structure = None


        self.spp_nll: Optional[float] = None
        self.spp_ppl: Optional[float] = None
        self.alpha: float = 0.0 

        self.used_scaled_embed = None
        self.m1_found = None
        self.m1_span_len = None
        self.scaled_tokens = None
        self.marker_eff_gain = None
        self.delta_eff_gain = None


    @torch.no_grad()
    def _compute_spp(self, text: str):
        model = self.llm.model
        tokenizer = self.llm.tokenizer
        device = next(model.parameters()).device

        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        if input_ids.shape[1] < 2:
            self.spp_nll, self.spp_ppl, self.alpha = 0.0, 1.0, 0.0
            return

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attn_mask[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        nll = -(token_lp * mask).sum() / (mask.sum() + 1e-9)

        ppl = float(torch.exp(nll))
        self.spp_nll = float(nll)
        self.spp_ppl = ppl

    
        alpha_cap = float(self.args.alpha_cap)
        ppl_neg_on = max(1.01, float(getattr(self.args, "ppl_neg_on", 5.0))) 
        ppl_off    = max(ppl_neg_on + 1e-3, float(self.args.ppl_off)) 
        ppl_on     = max(ppl_off + 1e-3, float(self.args.ppl_on)) 

        def smoothstep(u: float) -> float:
            return u * u * (3.0 - 2.0 * u)

        if ppl <= ppl_neg_on:
            t = -1.0
        elif ppl < ppl_off:
            ln_ppl = math.log(ppl); ln_neg = math.log(ppl_neg_on); ln_off = math.log(ppl_off)
            u = (ln_ppl - ln_neg) / max(1e-6, (ln_off - ln_neg))
            t = -1.0 + smoothstep(u)   # -1 → 0
        elif ppl >= ppl_on:
            t = 1.0
        else:
            ln_ppl = math.log(ppl); ln_off = math.log(ppl_off); ln_on = math.log(ppl_on)
            u = (ln_ppl - ln_off) / max(1e-6, (ln_on - ln_off))
            t = smoothstep(u)          # 0 → 1

        alpha_cap = float(self.args.alpha_cap)
        alpha_pos_cap = float(self.args.alpha_pos_cap) if getattr(self.args, "alpha_pos_cap", None) is not None else alpha_cap
        alpha_neg_cap = float(self.args.alpha_neg_cap) if getattr(self.args, "alpha_neg_cap", None) is not None else alpha_cap
        self.alpha = float((alpha_pos_cap if t >= 0.0 else alpha_neg_cap) * t)


    def extract_tags(self, response, step='select'):
        if step == "select":
            pattern = r"<SELECT/>\s*(\[.*?\])\s*</SELECT>"
            step_name = "Step 1. SELECT"
        elif step == "adapt":
            pattern = r"<ADAPT/>\s*(.*?)\s*</ADAPT>"
            step_name = "Step 2. ADAPT"
        elif step == "implement":
            pattern = r"<IMPLEMENT/>\s*(.*?)\s*</IMPLEMENT>"
            step_name = "Step 3. IMPLEMENT"
        else:
            raise ValueError(f"Unknown step: {step}")

        matches = re.findall(pattern, response, re.S)
        if not matches and step == "implement":
            matches = re.findall(r"<IMPLEMENT/>\s*(.*?)\s*}", response, re.S)
        if not matches and step == "select":
            arr = re.findall(r"\[(?:\s*\d+\s*,?)+\s*\]", response)
            if arr:
                matches = [arr[0]]
        if not matches:
            raise ValueError(f"{step_name} - No valid block found")

        block = matches[0].strip()
        if step == "implement" and block and block[-1] != "}":
            block += "\n}"

        if step == "select":
            try:
                ids = json.loads(block)
                if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                    raise ValueError
                if ids and max(ids) > len(reasoning_modules):
                    raise ValueError
                return [f"{i}. " + reasoning_modules[i-1] for i in sorted(set(ids))]
            except Exception:
                raise ValueError(f"{step_name} - Invalid JSON array: {block}")
        else:
            return block


    # action별 response 생성 (adapt, implement)
    def _return_response_vanilla(self, prompt, action):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response, _ = self.llm.generate_response(prompt)
                return response
            except Exception:
                retry_count += 1
                print(f"retry {retry_count}")
        raise RuntimeError(f"{action} step failed after retries")

    
    
    def _select_with_marker_bias(self, prompt) -> str:
        # Self-Perplexity 값 계산
        self._compute_spp(self.problem)

        messages = [{"role": "user", "content": prompt}]
        gen_args = GenArgs(
            model_name=self.args.model_name,
            model_id=self.args.model_id,
            cache_dir=self.args.cache_dir,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.do_sample,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            temperature=self.args.temperature,
        )
        # 모듈 선택시 Self-Perplexity 값 반영해서 외부 지식 보강 필요한 경우 관련된 모듈을 선택하도록 유도
        try:
            text, stats = step_decode_with_bias(
                self.llm.model, self.llm.tokenizer, messages, gen_args,
                forced_prefix="<SELECT/>\n",
                alpha=self.alpha,
                stop_at="</SELECT>",

                marker_m1_only=False,
                keyword_str="Check if retrieval from external sources is needed to answer the question.",

                marker_gain_coef=self.args.marker_gain_coef,
                marker_gain_cap=self.args.marker_gain_cap,
                marker_dropout=self.args.marker_dropout,
                marker_taper=self.args.marker_taper,

                delta_gain_coef=self.args.delta_gain_coef,
                delta_gain_cap=self.args.delta_gain_cap,
                keyword_casefold=True,
                
                # gpt-oss 호환
                force_eager_attn=True if self.args.model_name.startswith("gpt-oss") else self.args.force_eager_attn,
                disable_sdp_flash=True if self.args.model_name.startswith("gpt-oss") else self.args.disable_sdp_flash,
            )
            
            self.used_scaled_embed = stats.get("used_scaled_embed")
            self.m1_found        = stats.get("m1_found")
            self.m1_span_len     = stats.get("m1_span_len")
            self.scaled_tokens   = stats.get("scaled_tokens")
            self.marker_eff_gain = stats.get("marker_eff_gain")
            self.delta_eff_gain  = stats.get("delta_eff_gain")
            
            return text
        except Exception as e:
            print(f"[SELECT/bias] fallback: {e}")
            return self._return_response_vanilla(prompt, "SELECT")


    def __call__(self):
        for action in self.actions:
            # 모듈 선택
            if action == "SELECT":
                prompt = select_prompt.replace("{problem}", self.problem)
                prompt = prompt.replace("{reasoning_modules}", "\n".join([f"{i+1}. {m}" for i, m in enumerate(reasoning_modules)]))
                try:
                    response = self._select_with_marker_bias(prompt)
                    self.selected_modules = self.extract_tags(response, step='select')
                except Exception as e:
                    self.selected_modules = response
                    self.next_step = False
                    print(e)

            # 선택된 모듈을 문제에 적합하게 패러프레이징
            elif action == "ADAPT":
                if self.next_step:
                    if self.rag_module in self.selected_modules:
                        # model이 외부 지식 보강 필요하다고 판단한 경우
                        self.using_rag = True
                    prompt = adapt_prompt.replace("{problem}", self.problem)
                    prompt = prompt.replace("{selected_modules}", "\n".join(self.selected_modules))
                    try:
                        response = self._return_response_vanilla(prompt, "ADAPT")
                        self.adapted_modules = self.extract_tags(response, step='adapt')
                    except Exception as e:
                        self.adapted_modules = response
                        self.next_step = False
                        print(e)

            # 마지막으로 reasoning 과정을 {"reasoing step1" : [], "reasoning step2": [], ...} 형태로 meta-prompt 생성
            elif action == "IMPLEMENT":
                if self.next_step:
                    prompt = implement_prompt.replace("{problem}", self.problem)
                    prompt = prompt.replace("{adapted_modules}", self.adapted_modules)
                    try:
                        response = self._return_response_vanilla(prompt, "IMPLEMENT")
                        self.reasoning_structure = self.extract_tags(response, step='implement')
                    except Exception as e:
                        self.reasoning_structure = response
                        print(e)

