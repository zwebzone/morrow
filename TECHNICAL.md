# Technical specification

## 1. State and session boundary

Let \(\mathcal X_\tau\) be the tokenized context of session \(\tau\). For each
GatedDeltaNet layer \(\ell\) and head \(h\), its recurrent state is a matrix
\(\mathbf S^{\ell,h}\in\mathbb R^{d_k\times d_v}\). The complete persistent
state is the product-space tensor

\[
\mathbf P_\tau=\{\mathbf P_\tau^{\ell,h}\}_{\ell,h}.
\]

At every session boundary, attention KV entries and tokens are discarded. The
next forward pass starts with fresh attention KV and is initialized only by
\(\mathbf P_{\tau-1}\). In code, persistent states use the layout
`[recurrent_layer, head, key_dim, value_dim]`.

## 2. Writing a session

Morrow appends \(m\) learned write-token embeddings
\(\mathbf Z\in\mathbb R^{m\times d}\) to the current session. A write-only
LoRA \(\omega\) is enabled while the frozen backbone processes

\[
[\mathcal X_\tau;\mathbf Z]\quad\text{from}\quad\mathbf P_{\tau-1}.
\]

The LoRA is disabled for every QA and preservation forward. The terminal native
recurrent states from this write pass form \(\mathbf T_\tau\). The first
projection of each LoRA residual is randomly initialized and its second
projection is zero initialized, so the initial adapter is an exact no-op.

## 3. Low-rank carrier

For recurrent layer \(\ell\) and head \(h\), the carrier computes

\[
\mathbf C_\tau^{\ell,h}
=g_{0}^{\ell,h}\mathbf T_\tau^{\ell,h}
+g_{K}^{\ell,h}\mathbf B_K^\ell\mathbf A_K^\ell
 \mathbf T_\tau^{\ell,h}
+g_{V}^{\ell,h}\mathbf T_\tau^{\ell,h}
 \mathbf B_V^\ell\mathbf A_V^\ell.
\]

The key- and value-side low-rank factors are shared across heads within a
layer; the three scalar gates are layer- and head-specific. Shapes are
\(\mathbf A_K^\ell\in\mathbb R^{r\times d_k}\),
\(\mathbf B_K^\ell\in\mathbb R^{d_k\times r}\),
\(\mathbf B_V^\ell\in\mathbb R^{d_v\times r}\), and
\(\mathbf A_V^\ell\in\mathbb R^{r\times d_v}\). Zero initialization of
\(\mathbf B_K\) and \(\mathbf A_V\), together with \(g_0=1\), initializes the
carrier to the identity map.

## 4. Stable recurrent carry

The first update is \(\mathbf P_1=\mathbf C_1\). For \(\tau\ge2\), flattening
all recurrent layers and heads only for the inner product and norm, define
\(\widehat{\mathbf P}=\mathbf P_{\tau-1}/\|\mathbf P_{\tau-1}\|_F\) and
\(\widehat{\mathbf C}=\mathbf C_\tau/\|\mathbf C_\tau\|_F\). With
\(\theta=\arccos\langle\widehat{\mathbf P},\widehat{\mathbf C}\rangle\),

\[
\mathbf P_\tau=\|\mathbf P_{\tau-1}\|_F
\left[
\frac{\sin((1-\lambda)\theta)}{\sin\theta}\widehat{\mathbf P}
+\frac{\sin(\lambda\theta)}{\sin\theta}\widehat{\mathbf C}
\right].
\]

Parallel, antipodal, and zero-norm proposals do not define a unique tangent;
the implementation therefore keeps the previous state in those degenerate
cases. This is a global product-space operation, not a per-head normalization.

## 5. Training objectives

Each session has a QA set \(\mathcal D_\tau\). At terminal depth \(\tau\), the
buffer \(\mathcal B_\tau\) mixes QA pairs from the current session with replay
from earlier sessions.

The post-transition path removes the written context and uses fresh attention
KV:

\[
\mathcal L_{\mathrm{post}}^{(\tau)}=
\mathbb E_{(q,a)\sim\mathcal B_\tau}
\operatorname{NLL}(a\mid q,\mathbf P_\tau).
\]

The pre-transition path keeps the current context visible while initializing
the model with the preceding persistent state:

\[
\mathcal L_{\mathrm{pre}}^{(\tau)}=
\mathbb E_{(q,a)\sim\mathcal B_\tau}
\operatorname{NLL}(a\mid\mathcal X_\tau,q,\mathbf P_{\tau-1}).
\]

It is omitted at depth one because \(\mathbf P_0\) is fixed and therefore
cannot train the first transition.

For a source-disjoint general-capability prompt and gold continuation, the
preservation term distills the memory-conditioned distribution from the exact
frozen memory-off model:

\[
\mathcal L_{\mathrm{pres}}^{(\tau)}=
\mathbb E_{q\sim\mathcal G}
D_{\mathrm{KL}}\!\left(
\operatorname{sg}[p_\phi^{\mathrm{off}}(\cdot\mid q)]\,\|\,
p_\phi(\cdot\mid q,\mathbf P_\tau)
\right).
\]

The token-level KL is evaluated only at completion positions. Teacher and
student use the same prompt, continuation, tokenizer, frozen backbone, and
fresh attention KV; their only difference is the persistent state. The total
terminal-depth loss is

\[
\mathcal L^{(\tau)}=\mathcal L_{\mathrm{post}}^{(\tau)}
+\mathbb 1[\tau>1]\lambda_{\mathrm{pre}}
 \mathcal L_{\mathrm{pre}}^{(\tau)}
+\lambda_{\mathrm{pres}}\mathcal L_{\mathrm{pres}}^{(\tau)}.
\]

## 6. Depth curriculum and optimization

At optimization step \(s\), an unroll depth \(U\) is sampled uniformly from the
currently allowed support. The maximum allowed depth grows linearly from one
to ten:

\[
\mathcal L_{\mathrm{train}}(s)=
\mathbb E_{U\sim\pi_s}[\mathcal L^{(U)}].
\]

All \(U\) transitions execute in chronological order, but supervision is
applied only at the terminal depth. Truncated backpropagation retains gradient
through the two most recent transitions; earlier recurrent states are computed
normally and detached at the truncation boundary. This permits the pre loss to
reach the writer that created \(\mathbf P_{U-1}\).

Only \(\Theta=\{\mathbf Z,\omega,\psi\}\)—write tokens, write-only LoRA, and
carrier parameters—is passed to AdamW. The code raises an error if a backbone
parameter is trainable. Checkpoints likewise contain only \(\Theta\).

The supplied configurations use eight write tokens and a rank-32 carrier at
both scales. The write-only LoRA uses rank 8 with scale 16 for the 4B backbone,
and rank 16 with scale 32 for the 27B backbone. They set
\(\lambda_{\mathrm{pre}}=0.25\),
\(\lambda_{\mathrm{pres}}=0.05\), temperature one, and a two-transition
gradient horizon, matching the settings specified in the paper.
