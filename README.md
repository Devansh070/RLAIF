# RLAIF

<img width="1440" height="1840" alt="image" src="https://github.com/user-attachments/assets/05685a5a-2602-4f34-b626-5979caec5f41" />


A complete pipeline for aligning a small language model (SmolLM2-135M-Instruct) using Reinforcement Learning from AI Feedback (RLAIF). Replaces manual reward model training with a pre-trained DeBERTa-based reward signal, keeping the full PPO actor-critic loop intact.

---

## Overview

Standard RLHF pipelines require three sequential stages: supervised fine-tuning (SFT), reward model (RM) training on human preference data, and PPO-based policy optimisation. This project collapses stage 2 by substituting a pre-trained open-source reward model (`OpenAssistant/reward-model-deberta-v3-large-v2`), making the full alignment loop accessible without human annotation or a second training phase.

**Target hardware:** Google Colab T4 (15 GB VRAM)

---

## Pipeline

```
Raw DPO Dataset  →  ChatML Formatting  →  SFT  →  RLAIF Reward Scoring  →  PPO  →  Aligned Model
```

### Stage 1 — Supervised Fine-Tuning (SFT)

The base model is first fine-tuned on the `chosen` completions from `HumanLLMs/Human-Like-DPO-Dataset`. This establishes a reasonable starting policy before RL begins.

**Loss:** Cross-entropy over the full sequence (prompt + response). The SFT collator shifts inputs and labels by one position (standard next-token prediction), zeroing out the last real token's input to prevent training past the EOS boundary.

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{T} \sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_{<t})$$

where $\pi_\theta$ is the model, $a_t$ the token at position $t$, and $T$ the sequence length.

**Optimiser:** AdamW, lr = 9e-6, cosine decay with min_lr = 0.3 × initial, no warmup.

**Result (Image 2):** SFT loss decays from ~8.0 to ~2.6 over ~2,700 iterations, confirming the model is learning the target distribution.
<img width="755" height="555" alt="image" src="https://github.com/user-attachments/assets/2d46e478-0d1d-4383-b96d-b8f195c3673d" />

---

### Stage 2 — RLAIF Reward Scoring

Instead of training a reward model, inference is run on `OpenAssistant/reward-model-deberta-v3-large-v2`:

- Architecture: DeBERTa-v3-large with a scalar regression head
- Parameters: ~304M (exact count printed at runtime)
- Training data: WebGPT comparisons, SummarizeFeedback, synthetic-instruct-gptj-pairwise
- Input: tokenised `(question, answer)` pair via its own tokeniser (max 512 tokens, truncation applied)
- Output: single scalar reward score $r \in \mathbb{R}$

The generated sequence is decoded back to text, split on ChatML markers to isolate the question and assistant answer, then re-tokenised with the DeBERTa tokeniser before scoring.

The reward is injected only at the **final generated token** position:

$$r_t = \begin{cases} R_\phi(x, y) & \text{if } t = T_{\text{last}} \\ 0 & \text{otherwise} \end{cases}$$

This places the full sequence-level reward at the end of the trajectory, delegating temporal credit assignment entirely to GAE.

---

### Stage 3 — Proximal Policy Optimisation (PPO)

Full actor-critic PPO is run on top of the SFT checkpoint.

#### Token Generation (Rollout)

During rollout, tokens are sampled via **multinomial sampling** from the softmax distribution — not greedy decoding. This preserves stochasticity, which is necessary for the policy to explore the response space during training.

#### Value Model

A copy of the SFT model with `lm_head` replaced by a linear scalar projector:

$$V_\psi(s_t) = W \cdot h_t \in \mathbb{R}, \quad W \in \mathbb{R}^{1 \times 576}$$

where 576 is the hidden size of SmolLM2-135M.

#### TD Residual and GAE

Temporal difference residual at each step:

$$\delta_t = r_t + \gamma \cdot V_\psi(s_{t+1}) - V_\psi(s_t)$$

Generalised Advantage Estimation (Schulman et al., 2016):

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}$$

- Value model uses $\lambda = 1.0$ (actual Monte Carlo returns, no bootstrapping bias)
- Policy model uses $\lambda = 0.95$ (slight bias to reduce variance in advantage estimates)

#### Value Loss

The value target $V_\text{actual}$ is computed as $\hat{A}_t^{\text{GAE}(1.0)} + V_\text{old}$, where $V_\text{old}$ is a detached snapshot of the current batch's value estimates. The loss is normalised by the empirical variance of $(V_\text{old} - V_\text{actual})$ and clipped symmetrically:

$$\mathcal{L}_V = \frac{1}{2\sigma_V^2} \cdot \max\!\left[(V_\text{new} - V_\text{actual})^2,\; (\text{clip}(V_\text{new},\, V_\text{old} \pm \epsilon_v \sigma_V) - V_\text{actual})^2\right]$$

where $\sigma_V = \sqrt{\mathbb{E}[(V_\text{old} - V_\text{actual})^2]}$ and $\epsilon_v = 0.2$.

#### Policy Loss (Clipped Surrogate)

Probability ratio between updated and reference logits:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_{<t})}{\pi_{\theta_\text{ref}}(a_t \mid s_{<t})}$$

**Implementation note:** Within each update step, $\pi_{\theta_\text{ref}}$ is obtained by detaching the logits from the same forward pass as $\pi_\theta$. This means $\rho_t = 1$ at initialisation per batch, making this equivalent to **single-step PPO** — rollout data is not reused across multiple gradient updates. The clipping therefore acts as a local gradient norm constraint rather than an importance-sampling correction across replay epochs.

Clipped surrogate objective:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\!\left[\max\!\left(-\hat{A}_t \cdot \rho_t,\; -\hat{A}_t \cdot \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\right)\right]$$

with $\epsilon = 0.2$.

#### KL Divergence Regularisation

Prevents the policy from deviating too far from the SFT checkpoint. The KL is computed in the **forward direction** $D_{\text{KL}}(\pi_\text{old} \| \pi_\text{new})$:

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(\pi_\text{old} \| \pi_\text{new}) = \sum_v \pi_\text{old}(v) \left[\log \pi_\text{old}(v) - \log \pi_\text{new}(v)\right]$$

Both log-softmax terms are computed from scratch each step (no pre-stored reference policy), using numerically stable log-sum-exp normalisation.

#### Total Policy Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \beta \cdot \mathcal{L}_{\text{KL}}, \quad \beta = 0.05$$

**Result (Image 1):** Mean reward grows from ~1.0 to ~5.1 (absolute, on the DeBERTa reward scale) over ~14,000 PPO iterations, exhibiting a sigmoid-shaped rise between iterations 3,000–6,000 before plateauing. The plateau reflects the reward model's scoring ceiling rather than policy collapse.
<img width="750" height="569" alt="image" src="https://github.com/user-attachments/assets/a626ab95-c5ca-48eb-a01e-a5701fb521cb" />

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | SmolLM2-135M-Instruct |
| Reward model | DeBERTa-v3-large-v2 (~304M params) |
| Batch size | 8 |
| Gradient accumulation steps | 8
| Max sequence length | 768 tokens |
| SFT learning rate | 9e-6 |
| PPO learning rate (actor + critic) | 3e-5 |
| LR schedule (SFT) | Cosine, no warmup, min = 0.3× initial |
| LR schedule (PPO) | Cosine, 10% linear warmup |
| Discount factor γ | 1.0 |
| GAE λ (policy) | 0.95 |
| GAE λ (value) | 1.0 |
| PPO clip ε | 0.2 |
| Value clip ε | 0.2 |
| KL coefficient β | 0.05 |
| SFT epochs | 1 |
| PPO epochs | 5 |

---

## Training Results

### SFT Loss (Image 2)
Starts at ~8.0, decays monotonically to ~2.6. The steep initial drop indicates rapid adaptation from the base model's pretraining distribution to the instruction-following target format. Plotted as a 50-iteration moving average of per-step cross-entropy.

### PPO Reward (Image 1)
Starts at ~1.0, rises sharply between iterations 3,000–6,000, then plateaus at ~5.1. These are raw scalar outputs from the DeBERTa reward model's regression head and are not normalised. The plateau reflects reward model ceiling effects. Plotted as a 120-iteration moving average of per-step mean batch reward.

---

## Key Design Decisions

**Pre-trained RM instead of training from scratch.** Training a reward model requires a large pool of human preference annotations, GPU time, and careful hyperparameter tuning. Using `reward-model-deberta-v3-large-v2` eliminates all three requirements while providing a reward signal trained on diverse open-source comparison data. The tradeoff is that the reward model is fixed and cannot be adapted to the specific task distribution.

**Reward only on the final token.** The reward model scores the entire completion as a single scalar. Placing this scalar only at the final token position respects the causal structure of the trajectory and delegates temporal credit assignment entirely to GAE. Intermediate tokens receive a gradient signal only through the advantage propagation.

**Separate GAE lambdas for actor and critic.** λ = 1.0 for the value model gives unbiased Monte Carlo return estimates, which is important for stable critic learning in sparse-reward settings. λ = 0.95 for the policy introduces a small bias to reduce variance in the advantage estimates, producing more stable policy gradient steps.

**Single-step PPO (no rollout replay).** The reference logits are detached within the same forward pass rather than stored from a separate rollout phase. This is a deliberate simplification: it avoids stale policy gradients from replayed trajectories at the cost of forgoing the data-efficiency benefits of multi-epoch PPO update loops. The clipping still prevents excessively large gradient steps within a batch.

**Left-padding for generation, right-padding for SFT.** Generation with a KV cache requires the last real token to be at a predictable position — left padding ensures this (`attention_mask.sum() - 1` is always the last real index). SFT loss is computed over the full sequence, where right padding keeps labels and inputs aligned without index arithmetic.

**Sequence realignment after generation.** After rollout, sequences are right-aligned (padding shifted to the left) and leading padding columns shared across the whole batch are trimmed. This ensures the value and policy models receive compact, consistently formatted inputs during the update phases.
