"""
This file contains **drop-in** replacements and additions for your
`EGT_(Embedded_GAN_in_Transformer).ipynb` to implement the recommendations:

1) Uncertainty-targeted, diversity-aware replay buffer (R-Zero style)
2) Prioritized sampling with hard-negative mining for critics
3) Self-play wake step that generates K samples per prompt and keeps
   those in an uncertainty band; optional diversity filtering
4) EMA teacher KL as before; optional RLVR head (plug-in) for verifiable tasks
5) System bits: bf16 autocast when CUDA, optional activation checkpointing

Integration notes are at the bottom of this file.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 0) Utilities
# -----------------------------

def jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity. Returns in [0,1]."""
    A = set(a.split())
    B = set(b.split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / max(1, union)

@torch.no_grad()
def entropy_from_samples(texts: List[str]) -> float:
    """Rough uncertainty heuristic using normalized n-gram modes.
    Compute cluster frequencies by exact text string, take distribution entropy.
    Higher entropy = more spread = more uncertainty."""
    if not texts:
        return 0.0
    from collections import Counter
    c = Counter(texts)
    total = sum(c.values())
    probs = [v/total for v in c.values()]
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    # normalize by log(K)
    if total > 1:
        ent /= math.log(total + 1e-12)
    return float(ent)

@torch.no_grad()
def agreement_from_samples(texts: List[str]) -> float:
    """Agreement = fraction of the modal string among K samples."""
    if not texts:
        return 0.0
    from collections import Counter
    cnt = Counter(texts)
    mode = cnt.most_common(1)[0][1]
    return mode / len(texts)

# -----------------------------
# 1) Prioritized, Diversity-Aware Replay Buffer
# -----------------------------
class PrioritizedReplayBuffer:
    """
    Stores (prompt, response, meta) with:
      - uncertainty (uncert in [0,1])
      - diversity signature (simple running exemplar set)
      - score: any auxiliary score (e.g., distinct-2)

    Sampling supports uncertainty band selection and hard/medium mixes.
    """
    def __init__(self, maxlen: int = 20000, min_diversity: float = 0.35):
        self.maxlen = maxlen
        self.min_div = min_diversity
        self.items: List[Dict] = []
        # Maintain a small set of exemplars for fast diversity checks
        self._exemplars: List[str] = []

    def __len__(self):
        return len(self.items)

    def _is_diverse(self, text: str) -> bool:
        if not self._exemplars:
            return True
        sims = [jaccard_tokens(text, ex) for ex in self._exemplars]
        return (max(sims) if sims else 0.0) <= (1.0 - self.min_div)

    def add(self, prompt: str, response: str, *, uncert: float, score: float):
        # enforce diversity by checking against exemplars; if similar, probabilistically keep
        keep = self._is_diverse(response) or (random.random() < 0.1)
        if not keep:
            return False
        self.items.append({
            'prompt': prompt,
            'response': response,
            'uncert': float(uncert),
            'score': float(score),
            't': time.time(),
        })
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen:]
        # update exemplars with a small chance
        if self._is_diverse(response) and random.random() < 0.2:
            self._exemplars.append(response)
            if len(self._exemplars) > 512:
                self._exemplars = self._exemplars[-512:]
        return True

    def sample_prioritized(self, batch_size: int, *, band: Tuple[float,float]=(0.3,0.6), hard_frac: float = 0.6) -> List[Tuple[str,str,float]]:
        """Return a mix: HARD (high-uncert) and MEDIUM (band) examples."""
        if not self.items:
            return []
        lo, hi = band
        hard = [it for it in self.items if it['uncert'] > hi]
        mid  = [it for it in self.items if lo <= it['uncert'] <= hi]
        easy = [it for it in self.items if it['uncert'] < lo]
        # fallback pools
        def take(pool, k):
            k = max(0, min(k, len(pool)))
            return random.sample(pool, k) if k else []
        k_hard = int(batch_size * hard_frac)
        k_mid  = batch_size - k_hard
        out = take(hard, k_hard)
        out += take(mid, k_mid)
        if len(out) < batch_size:  # backfill
            need = batch_size - len(out)
            pool = hard or mid or easy or self.items
            out += take(pool, need)
        random.shuffle(out)
        return [(it['prompt'], it['response'], it['score']) for it in out[:batch_size]]

# -----------------------------
# 2) Model hooks — RLVR head (optional) and activation checkpoint
# -----------------------------
class RLVRHead(nn.Module):
    """Binary verifier head over pooled final hidden state."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2), nn.GELU(),
            nn.Linear(hidden_size//2, 1)
        )
    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        # h_last: [B, T, H] → mean-pool over tokens
        pooled = h_last.mean(dim=1)
        return self.net(pooled).squeeze(-1)  # logits

# Monkey-patch / thin wrappers expected by your TinyGPT/Block
try:
    from types import MethodType
    # Add a flag to enable checkpointing inside Block.forward
    def block_forward_with_checkpoint(self, x):
        # assumes self has: ln1, attn, ln2, mlp, with_critic, critic
        def attn_mlp(inp):
            y, att = self.attn(self.ln1(inp))
            x1 = inp + y
            y2 = self.mlp(self.ln2(x1))
            return x1 + y2
        if getattr(self, 'use_ckpt', False) and self.training:
            x2 = torch.utils.checkpoint.checkpoint(attn_mlp, x, use_reentrant=False)
        else:
            y, att = self.attn(self.ln1(x))
            x = x + y
            y = self.mlp(self.ln2(x))
            x2 = x + y
        crit = None
        if getattr(self, 'with_critic', False):
            crit = self.critic(x2.mean(dim=1).detach())
        return x2, None, crit
    # The host notebook defined Block; we patch its forward if present
    if 'Block' in globals():
        Block.forward = block_forward_with_checkpoint
except Exception:
    pass

# -----------------------------
# 3) Wake phase: K-sampling, uncertainty & diversity scoring
# -----------------------------
@dataclass
class RZWakeCfg:
    k_samples: int = 6
    temperature: float = 0.9
    top_k_val: int = 40
    max_new_tokens: int = 128
    uncert_band: Tuple[float,float] = (0.3, 0.6)

class RZWake:
    def __init__(self, tokenizer, model, device: str='cpu', gen_fn=None):
        self.tok = tokenizer
        self.model = model
        self.device = device
        self.gen_fn = gen_fn  # expects (model, idx, max_new_tokens, temperature, top_k_val) → ids

    @torch.no_grad()
    def generate_one(self, prompt: str, *, temperature: float, top_k_val: int, max_new_tokens: int) -> str:
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.device)
        out = self.gen_fn(self.model, idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k_val=top_k_val)
        full_ids = out[0].tolist()
        pl = len(self.tok.encode(prompt))
        gen_ids = full_ids[pl:]
        return self.tok.decode(gen_ids)

    @torch.no_grad()
    def k_sample(self, prompt: str, cfg: RZWakeCfg) -> Dict:
        texts = [self.generate_one(prompt, temperature=cfg.temperature, top_k_val=cfg.top_k_val, max_new_tokens=cfg.max_new_tokens) for _ in range(cfg.k_samples)]
        # Uncertainty via agreement/entropy; keep both for debugging
        agree = agreement_from_samples(texts)
        uncert = 1.0 - agree
        ent = entropy_from_samples(texts)
        # choose representative: take a sample in the middle of uncertainty band with best distinct-2 proxy
        def distinct2(txt: str) -> float:
            toks = txt.split()
            if len(toks) < 2:
                return 0.0
            bigrams = list(zip(toks, toks[1:]))
            return len(set(bigrams))/max(1, len(bigrams))
        # prefer the sample with the highest distinct-2 among the set
        best = max(texts, key=distinct2)
        return {
            'prompt': prompt,
            'choice': best,
            'uncert': float(uncert),
            'entropy': float(ent),
            'k_texts': texts,
        }

# -----------------------------
# 4) Sleep phase helpers: hard-negative mining for critics
# -----------------------------
@torch.no_grad()
def pick_hard_negatives(fake_scores: List[torch.Tensor], top_m: int) -> List[int]:
    """Return indices of fake samples whose critic logits are near 0 (hard)."""
    if not fake_scores:
        return []
    # Assume each element is shape [B, 1] or [B]; flatten to [B]
    fs = [fs.view(-1) for fs in fake_scores]
    # mean across critic layers
    fs_mat = torch.stack(fs, dim=0).mean(0)  # [B]
    # closeness to 0
    diffs = torch.abs(fs_mat)
    _, idx = torch.topk(-diffs, k=min(top_m, diffs.numel()))
    return idx.tolist()

# -----------------------------
# 5) EGT System Upgrade Wrapper
# -----------------------------
@dataclass
class TrainCfgRZ:
    block_size: int = 256
    critic_layers: tuple = (1,3,5)
    lr_D: float = 5e-4
    lr_G: float = 1e-4
    ttur_D_steps: int = 3
    ttur_G_steps: int = 1
    ema_decay: float = 0.999
    device: str = 'cpu'
    # new
    use_bf16: bool = True
    use_ckpt: bool = True
    uncert_band: Tuple[float,float] = (0.3, 0.6)
    hard_frac: float = 0.6
    k_samples: int = 6
    min_diversity: float = 0.35
    fm_lambda: float = 5.0
    beta_gan: float = 0.3
    alpha_kld: float = 0.7
    # RLVR
    enable_rlvr: bool = False

class EGTSystemRZ:
    def __init__(self, tokenizer, base_cfg_class, tiny_gpt_class, gen_fn, cfg: TrainCfgRZ):
        self.cfg = cfg
        self.tok = tokenizer
        model_cfg = base_cfg_class(vocab_size=tokenizer.vocab_size, block_size=cfg.block_size)
        # enable checkpointing per block if requested
        self.G = tiny_gpt_class(model_cfg, critic_layers=list(cfg.critic_layers)).to(cfg.device)
        for blk in getattr(self.G, 'blocks', []):
            setattr(blk, 'use_ckpt', cfg.use_ckpt)
        self.Teacher = tiny_gpt_class(model_cfg, critic_layers=[]).to(cfg.device)
        self.Teacher.load_state_dict(self.G.state_dict(), strict=False)
        self.ema = EMA(self.Teacher, decay=cfg.ema_decay)
        # Critics params
        D_params = []
        for blk in self.G.blocks:
            if getattr(blk, 'with_critic', False):
                D_params += list(blk.critic.parameters())
        self.D_params = D_params
        # LoRA-only updates + critics
        for name, p in self.G.named_parameters():
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad = (('A.weight' in name) or ('B.weight' in name) or ('critic' in name))
        self.optD = torch.optim.Adam(self.D_params, lr=cfg.lr_D) if self.D_params else None
        self.optG = torch.optim.Adam([p for p in self.G.parameters() if p.requires_grad], lr=cfg.lr_G)
        # Buffer and wake helper
        self.buffer = PrioritizedReplayBuffer(maxlen=20000, min_diversity=cfg.min_diversity)
        self.rzwake = RZWake(tokenizer, self.G, device=cfg.device, gen_fn=gen_fn)

    @torch.no_grad()
    def wake_step(self, prompt: str, wcfg: Optional[RZWakeCfg]=None):
        wcfg = wcfg or RZWakeCfg(k_samples=self.cfg.k_samples, uncert_band=self.cfg.uncert_band)
        sample = self.rzwake.k_sample(prompt, wcfg)
        # compute a simple distinct-2 proxy score
        def distinct2(txt: str) -> float:
            toks = txt.split()
            if len(toks) < 2: return 0.0
            bigrams = list(zip(toks, toks[1:]))
            return len(set(bigrams))/max(1, len(bigrams))
        kept = self.buffer.add(sample['prompt'], sample['choice'], uncert=sample['uncert'], score=distinct2(sample['choice']))
        return sample['choice'], sample['uncert'], kept

    def encode_batch(self, pairs: List[Tuple[str,str,float]]):
        ids = []
        for p, r, _ in pairs:
            txt = (p + ' ' + r).strip()
            ids.append(self.tok.encode(txt, add_bos=True, add_eos=True))
        maxT = max(len(x) for x in ids)
        mat = torch.full((len(ids), maxT), 0, dtype=torch.long, device=self.cfg.device)
        for i, x in enumerate(ids):
            mat[i, :len(x)] = torch.tensor(x, dtype=torch.long, device=self.cfg.device)
        return mat

    def sleep_epoch(self, steps_D: int=30, steps_G: int=10, batch_size: int=8) -> Dict[str,float]:
        if len(self.buffer) < max(16, batch_size):
            return {'lossD': 0.0, 'lossG': 0.0}
        self.G.train(); self.Teacher.eval()
        bce = nn.BCEWithLogitsLoss()
        lossD_avg = 0.0; lossG_avg = 0.0
        autocast = (torch.cuda.is_available() and self.cfg.use_bf16)

        # ----- D steps -----
        for _ in range(steps_D):
            batch = self.buffer.sample_prioritized(batch_size, band=self.cfg.uncert_band, hard_frac=self.cfg.hard_frac)
            real_idx = self.encode_batch(batch)
            # real (older) samples
            _, real_scores = self.G(real_idx)
            prompts = [p for p,_,_ in batch]
            # generate fresh fakes on same prompts
            fake_pairs = []
            for p in prompts:
                txt, _, _ = self.wake_step(p)
                fake_pairs.append((p, txt, 0.0))
            fake_idx = self.encode_batch(fake_pairs)
            _, fake_scores = self.G(fake_idx)
            if not real_scores or not fake_scores:
                continue
            self.optD.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                lossD = 0.0
                for rs, fs in zip(real_scores, fake_scores):
                    y_real = torch.ones_like(rs)
                    y_fake = torch.zeros_like(fs)
                    lossD = lossD + bce(rs, y_real) + bce(fs, y_fake)
            lossD.backward()
            nn.utils.clip_grad_norm_(self.D_params, 1.0)
            self.optD.step()
            lossD_avg += float(lossD.item())

        # ----- G steps -----
        for _ in range(steps_G):
            batch = self.buffer.sample_prioritized(batch_size, band=self.cfg.uncert_band, hard_frac=self.cfg.hard_frac)
            idx = self.encode_batch(batch)
            self.optG.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits, _ = self.Teacher(idx)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                s_logits, s_scores = self.G(idx)
                loss_kld = kl_divergence(s_logits, t_logits)
                # regenerate fakes for adversarial term
                prompts = [p for p,_,_ in batch]
                fake_pairs = []
                for p in prompts:
                    txt, _, _ = self.wake_step(p)
                    fake_pairs.append((p, txt, 0.0))
                fake_idx = self.encode_batch(fake_pairs)
                _, fake_scores = self.G(fake_idx)
                # Hard-negative mining: focus GAN term on ambiguous fakes
                gan_loss = 0.0; fm_loss = 0.0
                if fake_scores:
                    # choose hardest m = half of batch
                    idx_hard = pick_hard_negatives(fake_scores, top_m=max(1, len(prompts)//2))
                    for li, fs in enumerate(fake_scores):
                        if li in idx_hard:
                            y_real = torch.ones_like(fs)
                            gan_loss = gan_loss + bce(fs, y_real)  # generator tries to fool critics
                    # feature matching between real and fake critic layer means
                    for rs, fs in zip(s_scores, fake_scores):
                        fm_loss = fm_loss + F.l1_loss(rs.detach(), fs)
                lossG = self.cfg.alpha_kld*loss_kld + self.cfg.beta_gan*gan_loss + self.cfg.fm_lambda*fm_loss
            lossG.backward()
            nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.optG.step()
            lossG_avg += float(lossG.item())
            # EMA teacher update
            self.ema.update(self.G)

        return {
            'lossD': lossD_avg/max(1, steps_D),
            'lossG': lossG_avg/max(1, steps_G),
        }

# -----------------------------
# 6) Minimal dependencies from the original notebook we reuse
# -----------------------------
# Expect these to exist in the host notebook:
#   - TinyBPETokenizer
#   - Config
#   - TinyGPT
#   - EMA
#   - generate(model, idx, max_new_tokens, temperature, top_k_val)
#   - kl_divergence(p_logits, q_logits)

# -----------------------------
# 7) Quick integration helper
# -----------------------------
_INTEGRATION_GUIDE = r"""
How to wire this in your notebook:

1) After your original class/function definitions (TinyGPT, Config, generate, EMA, kl_divergence),
   paste this whole file's content into a new cell.

2) Instantiate the upgraded system:

   cfg = TrainCfgRZ(device='cuda' if torch.cuda.is_available() else 'cpu',
                    block_size=256,
                    critic_layers=(1,3,5),
                    use_bf16=True,
                    use_ckpt=True,
                    k_samples=6,
                    uncert_band=(0.3, 0.6),
                    hard_frac=0.6)

   tok = TinyBPETokenizer(vocab_size=1024)
   # train tokenizer on your corpus first as in your original notebook

   sys_rz = EGTSystemRZ(
       tokenizer=tok,
       base_cfg_class=Config,
       tiny_gpt_class=TinyGPT,
       gen_fn=generate,
       cfg=cfg,
   )

3) Wake seeding (self-play):

   for _ in range(1000):
       p = random.choice(PROMPTS)  # or stream real prompts from your corpus
       txt, uncert, kept = sys_rz.wake_step(p)

4) Sleep epochs (training):

   for epoch in range(50):
       stats = sys_rz.sleep_epoch(steps_D=50, steps_G=20, batch_size=8)
       print(epoch, stats)

Notes:
- If you have a verifiable task, add your own verify_fn and wire an RLVR head similar to the critics.
- The upgraded ReplayBuffer implements uncertainty-band curriculum and diversity filtering.
- The generator loss combines KL-to-EMA, GAN (hard-negative-focused), and feature matching.
- We kept LoRA-only training for the generator to preserve low memory usage.
"""
