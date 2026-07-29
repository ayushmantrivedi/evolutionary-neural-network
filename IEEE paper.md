Accessibility Collapse, Recoverability Compression, and Storage–Access Decoupling: Towards a Geometric Theory of Information Persistence in Continual Learning

line 1: 1st Given Name Surname
line 2: *dept. name of organization
(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCID

line 1: 4th Given Name Surname
line 2: *dept. name of organization*
*(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCIDline 1: 2nd Given Name Surname
line 2: *dept. name of organization
(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCID

line 1: 5th Given Name Surname
line 2: *dept. name of organization
(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCIDline 1: 3rd Given Name Surname
line 2: *dept. name of organization
(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCID

line 1: 6th Given Name Surname
line 2: *dept. name of organization
(of Affiliation)*
line 3: *name of organization
(of Affiliation)*line 4: City, Country
line 5: email address or ORCID

*Abstract*— **Catastrophic forgetting in continual learning is conventionally interpreted as the irreversible destruction of previously acquired task information. We challenge this interpretation across eighteen controlled experiments on Split CIFAR-100 using ResNet-18 and ResNet-34 under SGD, Elastic Weight Consolidation (EWC), experience replay, and a novel Accessibility Anchor method. Our central empirical finding is a consistent and large Accessibility Gap (AG = LP − ACC ≈ 0.48): linear probes trained on frozen representations recover substantial task information despite near-complete task accuracy collapse (ACC ≈ 0.08). We introduce three novel findings that together constitute a geometric theory of information persistence. First, recoverability is massively compressible: 64× compression of the probe weight matrix via SVD retains 96% of full recoverability (8 of 512 dimensions). Second, EWC and replay affect storage and accessibility through fundamentally different mechanisms — EWC primarily preserves storage while replay primarily restores accessibility — establishing empirically that storage and accessibility are decoupled properties. Third, a lightweight Accessibility Anchor mechanism (feature anchoring loss) raises task accuracy from 0.074 to 0.196 while leaving linear probe accuracy unchanged, confirming that accessibility can be manipulated independently of storage. All findings are validated across multiple architectures, random seeds, and continual learning paradigms. We propose Dynamic Information Accessibility Theory (DIAT) as a unified framework, reframing catastrophic forgetting as geometric accessibility failure rather than information destruction, and introduce the Accessibility Gap as a principled diagnostic metric for continual learning systems.**

Keywords—Continual Learning, Catastrophic Forgetting, Representational Geometry, Accessibility Collapse, Recovery Manifold, Linear Probing, Storage-Access Decoupling, Elastic Weight Consolidation, Experience Replay, Information Persistence

# Introduction

Continual learning — the ability of a neural network to acquire knowledge sequentially from a series of tasks without catastrophic degradation of previously learned capabilities — remains one of the central open challenges in machine learning [1], [2]. The dominant phenomenon obstructing this goal, catastrophic forgetting [3], [4], describes the sharp degradation of performance on earlier tasks as the network adapts to new ones.

The prevailing assumption underlying two decades of continual learning research is captured by a simple implication:

*Low Accuracy => Knowledge Destroyed*

This assumption has motivated a field organized almost entirely around prevention: regularization methods constrain parameter changes [5], [6], [7]; replay methods rehearse historical data [8], [9]; architectural methods partition capacity [10], [11]. All of these approaches implicitly accept that information is being lost and attempt to slow or prevent that loss.

In this work, we challenge this foundational assumption empirically and systematically. We ask a different question:

***Does catastrophic forgetting destroy information, or does it destroy access to information?***

Our answer, supported by eighteen controlled experiments across architectures, methods, and random seeds, is that forgetting overwhelmingly destroys access rather than information. Even when task accuracy collapses to near-random performance (ACC ≈ 0.08), linear probes trained on frozen representations recover 0.56 accuracy — an Accessibility Gap of 0.48. This gap is robust, reproducible, and governed by measurable geometric properties of the representation space.

***A. Core Contributions***

The primary contributions of this work are:

1. Accessibility Collapse Framework. We introduce the Accessibility Gap (AG = LP − ACC) as a principled diagnostic metric and demonstrate that accessibility collapse — not information destruction — is the dominant failure mode in catastrophic forgetting. We validate this across two architectures, three learning paradigms, and three random seeds.
2. Recoverability Compression. We show that 64× compression of the probe weight matrix (8 of 512 dimensions) retains 96% of full recoverability. Information required for task recovery occupies a remarkably compact geometric structure that is orders of magnitude smaller than the ambient representation space.
3. Storage–Access Decoupling. We demonstrate empirically that EWC and experience replay operate through fundamentally different mechanisms: EWC primarily preserves storage (LP), while replay primarily restores accessibility (ACC). This establishes storage and accessibility as independently controllable properties.
4. Accessibility Anchor. We propose a lightweight intervention — feature anchoring — that raises task accuracy from 0.074 to 0.196 while leaving linear probe accuracy unchanged (0.542 → 0.543), demonstrating that accessibility can be selectively preserved without modifying what information is stored.
5. Dynamic Information Accessibility Theory (DIAT). We propose a unified theoretical framework that reframes catastrophic forgetting as geometric accessibility failure, introduces a three-level model (Storage → Representation → Accessibility), and provides falsifiable predictions for future investigation.

This work builds directly on two prior studies from our group. Trivedi and Melwani [28] introduced accessibility collapse and the three-level framework (storage, representation, accessibility) with static endpoint measurements. Trivedi and Melwani [29] extended this to dynamic trajectories, establishing the Compact Recovery Manifold (k\_t ≈ 8) and demonstrating R² = 0.82 geometric predictability of recoverability. The present work extends both: validating across methods and architectures, introducing storage–access decoupling, and proposing the Accessibility Anchor.

# Related Work

***A. Catastrophic Forgetting and Continual Learning***

Catastrophic forgetting was first systematically characterised by McCloskey and Cohen [3] and French [4], establishing that overlapping distributed representations in gradient-based networks lead to sequential interference. Two decades of mitigation strategies can be broadly categorised as regularization-based [5], [6], [7], replay-based [8], [9], and architectural [10], [11]. Comprehensive surveys are provided in De Lange et al. [2] and Hadsell et al. [12]. All existing methods implicitly accept information loss and attempt prevention. Our work proposes a complementary diagnostic and repair paradigm.

***B. Understanding Forgetting Internally***

Ramasesh et al. [13] investigated the anatomy of catastrophic forgetting in transformers, finding intermediate layers most susceptible. Nguyen et al. [14] studied theoretical conditions for forgetting, and Goodfellow et al. [15] provided early empirical characterisation. Most relevant to our work is Davari et al. [16], who used linear probing to show that representational forgetting is substantially smaller than functional forgetting — a finding our work extends, quantifies geometrically, and validates across methods. Zheng et al. [17] demonstrated spurious forgetting in language models, finding that forgetting is often more recoverable than accuracy metrics suggest, which aligns with our accessibility collapse hypothesis.

***C. Probing and Representational Geometry***

Linear probing as a tool for understanding intermediate representations was introduced by Alain and Bengio [18] and has been widely applied in transfer learning and self-supervised learning evaluation. Kornblith et al. [19] introduced CKA for rotation-invariant representational similarity, and Raghu et al. [20] introduced SVCCA. The application of these geometric tools to continual learning trajectories, as performed here and in our prior work [28], [29], represents a relatively unexplored direction. Papyan et al. [21] introduced Neural Collapse — the convergence of class representations to geometric structures during training — which provides theoretical grounding for why task-specific geometric structure persists in representations even after functional collapse.

***D. Subspace Methods in Continual Learning***

Saha et al. [22] proposed Gradient Projection Memory (GPM), projecting gradients onto subspaces orthogonal to prior task representations. This approach implicitly assumes task-specific knowledge occupies low-dimensional subspaces — an assumption our recoverability compression results directly quantify (k ≈ 8 out of 512 dimensions). Our findings provide direct empirical support for low-rank projection methods and suggest that even smaller projections than those used in GPM may be sufficient.

***E. Positioning of This Work***

Unlike prior approaches, we do not aim to prevent forgetting, propose a new training algorithm as our primary contribution, or require modification of the training process. Instead, we characterize what survives forgetting geometrically, demonstrate that survival is governed by accessible structure, and show that this structure can be manipulated independently of stored information. Our Accessibility Anchor is a proof-of-concept that accessibility can be preserved as a property distinct from storage — motivating a new class of lightweight, geometry-aware continual learning methods.

# Background and Notation

We consider a standard continual learning setting in which a neural network is trained sequentially on a series of tasks T₀, T₁, ..., T\_{N−1} without access to past task data. Let f\_θ : X → Z denote the backbone network mapping inputs to a d-dimensional representation space (d = 512 for ResNet-18 penultimate layer). Let θ\_t denote parameters after completing training on task

| **Symbol** | **Definition** |
| --- | --- |
| *ACC\_t* | Task accuracy of the active network θ\_t on Task 0. Measures functional accessibility. |
| *LP\_t* | Linear probe accuracy: a linear classifier trained on frozen features f\_θ\_t(x), measuring stored representational information. |
| *AG\_t* | Accessibility Gap: \_t. Quantifies the discrepancy between storage and accessibility. |
| *RS* | Recovery Score: accuracy of a classifier reset — frozen backbone, reinitialized and retrained linear head. |
| *RR* | Route Redundancy: number of independent recovery subspaces achieving ≥ 90% of full probe accuracy. |
| *k\_t* | Recovery Subspace Dimensionality: minimum rank of probe weight SVD preserving 90% of full probe accuracy. |
| *W\_t* | Probe weight matrix where C is the number of classes. |
| *U\_k* | Top-k right singular vectors of W\_t. Defines the Recovery Manifold. |
| *P\_k* | Projection operator: |

*TABLE I. Notation used throughout this paper*

***A. Accessibility Collapse — Formal Definition***

We define Accessibility Collapse for task T\_i as the condition:

where θ\_N are the parameters after learning all N tasks. Equivalently, AG = LP − ACC >> 0. A large positive AG indicates that the network retains substantial representational information about T\_i that its active classifier cannot access through its native decision pathway.

***B. Recovery Subspace***

Given a probe weight matrix W\_t trained on frozen features, we perform Singular Value Decomposition:

The top-k right singular vectors U\_k define the Recovery Manifold — the k-dimensional subspace of representation space that concentrates most of the probe's discriminative capacity. The projection of features h onto this manifold is:

The Accessibility Gap Compression Experiment (E13) measures how probe accuracy varies as a function of k, tracing the recoverability–dimensionality frontier.

# Dynamic Information Accessibility Theory

***A. Core Claim***

Dynamic Information Accessibility Theory (DIAT) provides a unified account of our empirical findings across three papers. The central claim is:

Catastrophic forgetting is not primarily the destruction of stored information. It is the progressive geometric reorganisation of representation space that renders previously stored task information inaccessible through the network's active decision pathways, while the information itself remains recoverable through alternative geometric access routes.

***B. Three-Level Knowledge Model***

DIAT introduces a three-level model of knowledge in neural networks, each operationalised by a distinct measurement:

Level 1 — Storage: Task-specific solutions remain valid in checkpoint form. Measured by checkpoint persistence: evaluating θ\*\_t on task T\_t after all subsequent training. Our experiments confirm storage persists at near-original accuracy across all tasks.

Level 2 — Representation: Task-relevant information remains encoded in frozen features. Measured by linear probe accuracy LP\_t. Our results demonstrate LP\_t ≈ 0.56 even when ACC\_t ≈ 0.08.

Level 3 — Accessibility: Information can be retrieved through the active decision pathway. Measured by ACC\_t. This is the level at which catastrophic forgetting manifests — and the only level that collapses completely.

***C. The Decoupling Mechanism***

The 4:1 ratio observed in our experiments (ΔPE / ΔR ≈ 3.93) — information concentration deteriorates four times faster than recoverability — establishes that information storage and accessibility are not merely correlated properties. They are independently regulated. EWC and replay target different levels: EWC operates primarily at the representation level (storage), while replay operates primarily at the accessibility level. The Accessibility Anchor confirms this decoupling through direct intervention.

***D. Falsifiable Predictions***

DIAT generates four testable predictions beyond the scope of the present experiments:

1. Destroying the recovery manifold (projecting features onto its complement) should cause recoverability to collapse toward random-subspace performance.
2. Realigning a drifted recovery manifold via Procrustes rotation should restore recoverability without any retraining.
3. Gradient energy directed into the recovery manifold should predict recoverability decline more strongly than gradient energy directed elsewhere.
4. A lighter anchor targeting only the 8 critical singular directions should preserve recoverability more efficiently than full-representation anchoring.

#### These predictions define the experimental agenda for the next phase of this research programme

# Experimental setup

***A. Dataset and Task Structure***

All experiments use Split CIFAR-100 [23]: CIFAR-100 partitioned into 10 sequential tasks of 10 classes each. Tasks are disjoint; no data from prior tasks is available during subsequent training. We use standard train/test splits (500/100 images per class). No replay buffer, exemplar memory, or generative replay is used except in dedicated replay experiments (E11).

***B. Architectures***

Two architectures are evaluated. ResNet-18 [24] is the primary architecture, adapted for CIFAR-scale inputs (32×32) with four residual blocks producing d = 512-dimensional penultimate representations. ResNet-34 is used for architecture validation (E9), with d = 512-dimensional output. Both are trained from random initialisation with no pretrained weights.

***C. Continual Learning Methods***

Four conditions are evaluated:

* SGD: pure sequential training with no continual learning mechanism. Baseline condition.
* EWC (Elastic Weight Consolidation) [5]: regularisation-based method penalising changes to Fisher-important parameters. λ = 5000.
* Replay: experience replay with a buffer of 200 samples per task, uniformly sampled at each training step.
* Accessibility Anchor: our proposed method (Section VIII). Feature anchoring loss applied to penultimate representations.

***D. Evaluation Metrics***

All metrics are computed on Task 0 test data unless otherwise noted:

* ACC: standard classification accuracy using the active classifier.
* LP: linear probe accuracy. Logistic regression trained on frozen penultimate features for 500 epochs, L2 regularisation C = 1.0.
* AG = LP − ACC: Accessibility Gap.
* RS: Recovery Score from classifier reset — frozen backbone, freshly initialised and retrained linear head, 50 epochs.
* k: Recovery Subspace Dimensionality — minimum SVD rank preserving 90% of probe accuracy.
* RR: Route Redundancy — number of orthogonal 8-dimensional subspaces each independently achieving RS ≥ 0.90 × LP.

***E. Optimisation Details***

| **Hyperparameter** | **Value** | **Notes** |
| --- | --- | --- |
| Optimiser | SGD with momentum | All methods |
| Momentum | 0.9 |  |
| Weight Decay | 5 × 10⁻⁴ |  |
| LR Schedule | Cosine Annealing | T\_max = epochs per task |
| Batch Size | 128 |  |
| Loss | Cross-Entropy |  |
| Epochs per Task | 50 | All tasks |
| EWC λ | 5000 | EWC condition only |
| Replay Buffer | 200 samples/task | Replay condition only |
| Anchor λ | 0.1 | Accessibility Anchor only |
| Random Seeds | 42, 123, 999 | Multi-seed validation |

*TABLE II. Optimisation hyperparameters held constant across experiments.*

# Phase 1: Accessibility Collapse

Phase I establishes the core empirical finding: catastrophic forgetting produces complete functional collapse while leaving substantial representational information intact.

***E1. Accessibility Collapse***

We compare final task accuracy against linear probe recoverability after sequential training across all 10 tasks with no continual learning mechanism.

| **Metric** | **Value** | **Interpretation** |
| --- | --- | --- |
| ACC (Task 0) | 0.081 | Near-random functional performance |
| LP (Task 0) | 0.562 | Substantial representational retention |
| **AG = LP − ACC** | **0.481** | Large accessibility gap — information survives |

*TABLE III. E1 results. Large AG demonstrates accessibility collapse rather than information destruction*

![](data:image/png;base64...)

*Accessibility Collapse: ACC vs. LP across all 10 tasks, demonstrating persistent large AG.*

*Fig. 1*

Observation: LP >> ACC. The network retains over 0.56 probe accuracy despite near-complete functional collapse to 0.08. The Accessibility Gap of 0.481 represents the difference between what the network can currently do and what it still knows. This is the primary empirical foundation of this work.[28]

***E2. Low-Rank Recoverability***

We perform SVD analysis of probe weights to determine the dimensionality of the recovery subspace.

Result: Recoverability is concentrated within approximately k = 8 principal directions out of 512 total dimensions (1.56%). This finding, first reported in [29] and reproduced here, establishes that the decodable structure of forgotten information is remarkably compact.

***E3. Recovery Geometry***

Principal angle analysis confirms that recovery geometry remains coherent despite continual learning — representations drift substantially but retain organized recoverable structure. Mean principal angle between checkpoint and final model representations: 65° (consistent with [29]).

Interpretation: The recovery manifold rotates substantially during continual learning but does not dissolve. Its compactness (k ≈ 8) is preserved even as its orientation changes, consistent with the Dynamic Compact Recovery Manifold hypothesis established in [29].

# Phase 2: Structure Of Recoveribility

Phase II investigates the geometric properties of the recovery manifold — whether recovery routes are structured, whether they are redundant, and how recoverability evolves temporally.

***E4. Random Subspace Controls***

To verify that the recovery manifold is genuinely structured rather than an artifact of dimensionality, we compare recovery accuracy using the learned manifold U\_final against random d-dimensional projections.

| **Subspace Type** | **Accuracy** | **vs. U\_final** |
| --- | --- | --- |
| Random 8D | 0.328 ± 0.021 | −0.236 (z = 11.4) |
| Random 16D | 0.401 ± 0.015 | −0.163 |
| Random 32D | 0.457 ± 0.013 | −0.107 |
| Random 64D | 0.500 ± 0.012 | −0.064 |
| **U\_final (8D)** | **0.564** | — |

*TABLE IV. E4 results. U\_final (8D) outperforms Random 64D, confirming the manifold is information-rich, not merely low-dimensional (z-score = 11.4 vs. Random 8D).*

The structured 8-dimensional manifold substantially outperforms 64-dimensional random projections (z-score = 11.4, p << 0.001). This rules out any dimensionality-only explanation and establishes that the recovery manifold is a genuinely structured, information-rich geometric object.

***E5. Causal Perturbation Studies***

We perform three interventions to determine what properties of the manifold enable recovery:

Basis scrambling: randomly permuting the columns of U\_k while preserving the column space (Span(U\_k)). Result: recovery unchanged. Interpretation: recovery depends on Span(U\_k), not on the specific basis vectors chosen. The manifold as a subspace is the causal object, not its particular coordinate representation.

Procrustes alignment: rotating U\_final toward U\_0 via the Procrustes solution R\* = argmin‖U\_0 − R·U\_final‖\_F. Result: structural coherence determines recovery; orientation alone is insufficient. Recoverability is not restored by aligning to the original manifold orientation when the underlying representations have changed.

Noise injection: adding Gaussian noise to U\_k of increasing magnitude. Result: recovery degrades gracefully with noise magnitude, confirming the geometric structure is the information-bearing object.

Combined interpretation: recovery depends on subspace structure rather than specific orientation or basis. This finding motivates the Accessibility Anchor design (Section VIII).

***E6. Redundant Recovery Routes***

We investigate whether task information is accessible through a unique recovery pathway or through multiple independent routes.

Method: after identifying U\_final as the primary recovery manifold, we extract its orthogonal complement (I − P\_k), then perform SVD of the probe weights restricted to this complement to identify secondary recovery routes. We repeat iteratively, removing each identified route before searching for the next.

![](data:image/png;base64...)

*Redundant Recovery Routes: accuracy across successive route removal iterations, demonstrating distributed recoverability*

*Fig. 2*

Figure 2 shows recoverability under successive removal of recovery routes. Removing one or two routes causes only minor degradation (LP: 0.532→0.523), indicating that recoverability is not concentrated in a single dominant direction. Instead, recovery information is distributed across multiple routes. Performance degrades gradually as routes are removed, reaching LP=0.171 when only one route remains. Since this value remains substantially above random chance (0.10), even a single route retains meaningful task information. These results support the Recovery Route Redundancy hypothesis and suggest that neural representations contain multiple partially independent pathways for information retrieval.

Result: Multiple independent recovery routes exist. Removing U\_final does not eliminate recovery — secondary routes retain meaningful probe accuracy. Recovery is distributed across orthogonal subspaces of the representation space. Route Redundancy (RR) quantifies the number of independent routes achieving ≥ 90% of full probe accuracy.

This is a novel finding with significant implications: it suggests that the representation space contains a distributed encoding of task information across multiple geometric structures, making accessibility collapse harder to achieve through any single intervention.

***E7. Recoverability Trajectory***

We track LP\_t, ACC\_t, and AG\_t across all 10 task checkpoints to characterise the temporal dynamics of accessibility collapse.

| **Tas--k t** | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *LP\_t* | 0.689 | 0.539 | 0.546 | 0.534 | 0.566 | 0.542 | 0.511 | 0.531 | 0.525 | 0.562 |

*TABLE V. E7: LP trajectory across task checkpoints. Recoverability stabilises after initial decline.*

![](data:image/png;base64...)

*Recoverability Trajectory: ACC\_t, LP\_t, and AG\_t across t = 0,...,9.*

*Fig. 3*

Observation: LP\_t declines sharply after Task 0 (0.689 → 0.539) then stabilises, fluctuating between 0.511 and 0.566 for tasks 1–9. The final checkpoint (Task 9) recovers to 0.562. Accessibility collapse — ACC dropping to near-zero — occurs rapidly, while representational recoverability enters a stable regime. This is consistent with the DIAT claim that forgetting destroys access before it destroys storage.

***E8. Cross-Task Validation***

We repeat the accessibility collapse analysis for all 10 tasks to verify the phenomenon is not a Task 0 artifact.

Result: Tasks 0–8 all exhibit LP >> ACC (accessibility collapse). Task 9 is the only task for which LP ≈ ACC, as expected — it is the most recently learned task and has not been subject to subsequent interference. Accessibility Collapse is a systematic property of continual learning, not a Task 0 artifact.

# Phase 3: Recoverability Compression

Phase III investigates how efficiently task information is encoded in the recovery manifold, and characterises the compressibility of recoverability.

***E9. Recovery Compression***

We measure probe accuracy as a function of the number of singular directions k retained in the probe weight SVD.

| **Dimensions (k)** | **Probe Accuracy** | **Retention** | **Compression Ratio** |
| --- | --- | --- | --- |
| 512 (full) | 0.561 | 100% | 1× |
| 64 | 0.558 | 99.5% | 8× |
| 32 | 0.549 | 97.9% | 16× |
| 16 | 0.541 | 96.4% | 32× |
| **8** | **0.539** | **96.1%** | **64×** |

*TABLE VI. E19: Recovery compression. 64× compression retains 96% of full probe accuracy.*

![](data:image/png;base64...)

*Recovery Efficiency Frontier: probe accuracy vs. subspace dimensionality k.*

*Fig. 4*

The 64× compression result is striking. Eight of 512 dimensions capture 96% of the probe's discriminative capacity. The information required for task recovery is not distributed uniformly across the representation space — it is concentrated in a compact, highly efficient geometric structure. This is the strongest quantitative characterisation of recoverability compression in the continual learning literature.

***E10. Accessibility Gap Compression***

We track how AG varies with the number of dimensions used to compute the probe:

Result: AG increases smoothly and monotonically with k, reaching its plateau by approximately k = 32. Accessibility emerges hierarchically as more dimensions are incorporated. This establishes that the accessibility gap is a graded property of the representation geometry, not a binary phenomenon.

***E11. Recovery Efficiency Frontier***

The 64× compression result (8 dimensions, 96% retention) defines the operating point of the recovery efficiency frontier. Beyond k = 8, diminishing returns are observed — each additional dimension contributes less than 0.1% additional recoverability. The frontier is sharp: most information is captured by the top-8 singular directions, and the remaining 504 directions contribute the remaining 4%.

Practical implication: storing U\_k (8 × 512 = 4,096 parameters per task) is sufficient to maintain near-full recoverability throughout a 10-task sequence. This is substantially more memory-efficient than storing full checkpoints (11M+ parameters) or full replay buffers.

# Phase 6: Storage-Access Decoupling

Phase IV is the most conceptually significant phase of the study. We investigate whether storage and accessibility are independently controllable properties by comparing methods that target different levels of the DIAT hierarchy.

***E12. Architecture Validation***

| **Architecture** | **ACC** | **LP** | **AG** |
| --- | --- | --- | --- |
| ResNet-18 | 0.081 | 0.562 | 0.481 |
| ResNet-34 | 0.090 | 0.507 | 0.417 |

*TABLE VII. E12: Architecture validation. Accessibility Collapse persists in ResNet-34.*

![](data:image/png;base64...)

*Architecture validation. Accessibility Collapse persists in ResNet-34.*

*Fig 5*

Accessibility Collapse survives architecture scaling. The reduced AG in ResNet-34 (0.417 vs 0.481) is noteworthy and warrants investigation — larger networks may have lower accessibility gaps, potentially due to greater representational capacity allowing more task-specific structure to be maintained.

***E13. EWC Validation***

| **Method** | **ACC** | **LP** | **AG** |
| --- | --- | --- | --- |
| SGD | 0.081 | 0.562 | 0.481 |
| EWC | 0.103 | 0.617 | **0.514** |

*TABLE VIII. E13: EWC results. EWC increases both LP and ACC, but LP increases more — storage improves more than accessibility.*

Critically, EWC increases LP from 0.562 to 0.617 (+0.055) but increases ACC from 0.081 to 0.103 (+0.022). The AG actually increases from 0.481 to 0.514. EWC primarily preserves storage — it prevents representational information from diffusing — but does not proportionally restore accessibility. This is consistent with EWC's mechanism: penalising parameter drift preserves representational structure but does not realign the active classifier.

***E14. Replay Validation***

| **Method** | **ACC** | **LP** | **AG** | **Primary Effect** |
| --- | --- | --- | --- | --- |
| SGD | 0.081 | 0.562 | 0.481 | Baseline |
| Replay | **0.188** | 0.515 | 0.327 | Accessibility ↑↑, storage stable |

*TABLE IX. E14: Replay results. Replay substantially improves ACC (+0.107) while LP remains similar, reducing AG.*

Replay increases ACC from 0.081 to 0.188 (+0.107) — a much larger gain than EWC (+0.022) — while LP remains similar (0.515 vs 0.562). The AG falls from 0.481 to 0.327. Replay primarily restores accessibility, not storage. This is mechanistically coherent: rehearsing historical data trains the active classifier to access existing representations, not to preserve representational structure.

***E15. Storage–Access Decoupling — Unified Analysis***

*![](data:image/png;base64...)*

*Storage–Access Decoupling: ACC vs. LP for SGD, EWC, Replay, and Anchor methods, illustrating distinct method trajectories.*

*Fig. 6*

The combined evidence from E10 and E11 establishes a fundamental finding:

**EWC and experience replay are not simply more or less effective versions of the same approach. They operate through fundamentally different mechanisms: EWC primarily preserves information storage (LP), while replay primarily restores information accessibility (ACC). Storage and accessibility are independently controllable properties of the representation.**

This decoupling has direct implications for method design: a hybrid approach combining EWC-style storage preservation with replay-style accessibility restoration could achieve substantially better performance than either method alone, targeting both levels of the DIAT hierarchy simultaneously.

***E16. Storage–Accessibility Landscape***

| **Method** | **ACC** | **LP** | **AG** | **Interpretat-ion** |
| --- | --- | --- | --- | --- |
| SGD | 0.074 | 0.542 | 0.468 | Low storage, low access |
| EWC | 0.103 | 0.617 | 0.514 | High storage, low access |
| Replay | 0.188 | 0.515 | 0.327 | Moderate storage, high access |
| Accessibility Anchor | **0.196** | 0.543 | 0.347 | Moderate storage, high access |

*TABLE X. E16: Storage–Accessibility Landscape. Each method occupies a distinct region. Accessibility Anchor achieves replay-level accessibility without full data replay*

# The Accessibility Anchor

Having established that accessibility and storage are decoupled and independently controllable, we propose the Accessibility Anchor — a lightweight method that explicitly targets accessibility preservation during continual learning.

***A. Motivation***

Standard continual learning methods either constrain parameter changes (preserving storage at the cost of plasticity) or rehearse historical data (restoring accessibility at the cost of memory and privacy). Neither approach directly targets accessibility as a property of the representation geometry. The Accessibility Anchor directly penalises representational drift in the subspace that governs accessibility.

***B. Method Description***

The Accessibility Anchor adds a feature anchoring loss during training on each task T\_t. After training task T₀, we extract the anchor representation A₀ — the set of penultimate features from a small held-out set (we use 10% of Task 0 training data, stored as feature vectors rather than raw images to preserve privacy and reduce memory).

For each subsequent task T\_t, we compute the current features f\_θ\_t(x\_anchor) and penalise deviation from the stored anchors:

The total training loss becomes:

where L\_CE is the standard cross-entropy loss on the current task. The anchor loss encourages the backbone to maintain representations compatible with Task 0's accessibility structure while remaining free to adapt for new tasks.

***C. Algorithm***

**Algorithm 1: Accessibility Anchor Training**

*Input: task sequence anchor ratio r, anchor weight λ*

*Output: with preserved accessibility*

1. Train on T₀ normally → θ₀

2. Extract anchor features: A

3. Store , not

4. For

a. For each batch (

i

5. Return θ\_N

***D. E17 — Accessibility Anchor Results***

| **Metric** | **SGD** | **Anchor** | **Change** |
| --- | --- | --- | --- |
| ACC | 0.074 | **0.196** | +0.122 (+165%) |
| LP | 0.542 | 0.543 | +0.001 (stable) |
| AG | 0.468 | 0.347 | −0.121 (reduced) |

*TABLE XI. E17: Accessibility Anchor results. ACC improves 165% while LP remains essentially unchanged, demonstrating that accessibility was preserved independently of storage.*

The Accessibility Anchor raises ACC from 0.074 to 0.196 — a 165% improvement in functional task performance — while LP remains essentially unchanged (0.542 → 0.543). This is the key proof-of-concept result: accessibility was selectively preserved while storage remained constant. The AG drops from 0.468 to 0.347, confirming that the gap between what the network knows and what it can access has been substantially reduced.

Crucially, the Anchor achieves comparable ACC to Replay (0.196 vs 0.188) without storing raw data, using only O(d) memory for the anchor feature vectors rather than O(|D|) for replay buffers. This makes it particularly suitable for privacy-sensitive and memory-constrained deployments.

# Robustness Validation

***E18. Multi-Seed Validation***

| **Seed** | **ACC** | **LP** | **AG** |
| --- | --- | --- | --- |
| 42 | 0.118 | 0.524 | 0.406 |
| 123 | 0.109 | 0.551 | 0.442 |
| 999 | 0.106 | 0.533 | 0.427 |
| **Mean** | **0.111** | **0.536** | **0.425** |

*TABLE XII. E18: Multi-seed validation. Accessibility Collapse is robust to random initialisation (σ\_AG < 0.02).*

![](data:image/png;base64...)

*Multi-seed validation. Accessibility Collapse is robust to random initialisation (σ\_AG < 0.02).*

*Fig 7*

Accessibility Collapse is not an initialisation artifact. The AG remains substantial (0.425 ± 0.018) across all three seeds. The standard deviation of AG across seeds is less than 0.02, confirming high reproducibility.

# Discussion

***A. Forgetting as Accessibility Failure***

The most important observation from this study is not any single number but the consistency of the 4:1 ratio across conditions: information concentration deteriorates roughly four times faster than recoverability (ΔPE / ΔR ≈ 3.93). This ratio — first established in the DIAT unified log and confirmed across all experimental conditions — is the signature of an active compensatory mechanism. The Dynamic Recovery Manifold continuously reorganises to preserve access to information even as the surrounding representational geometry drifts.

This finding challenges the field's implicit assumption that preventing parameter drift is the right objective. If recoverability is maintained by geometric reorganisation rather than structural preservation, then interventions targeting accessibility directly — like our Anchor — may be more efficient than interventions targeting parameter stability.

***B. Implications for Existing Methods***

Our storage–access decoupling result has concrete implications for method design. EWC preserves storage but does not restore accessibility. Replay restores accessibility but requires data retention. A principled combination — EWC for storage plus a lightweight anchor for accessibility — could achieve stronger retention than either method alone while using less memory than replay. This is the natural next experiment in this research programme.

The k ≈ 8 result also has implications for subspace methods like GPM [22]. If the recovery manifold occupies only 8 of 512 dimensions, then gradient projection onto an 8-dimensional manifold complement is sufficient to preserve recoverability — substantially smaller than the projections typically used. This could reduce the memory and computational cost of subspace methods significantly.

***C. The Redundant Routes Finding***

The existence of multiple independent recovery routes is a novel and underexplored finding. It suggests that neural representations in continual learning are more robust than they appear — not only does information survive forgetting, but it survives in multiple orthogonal structures simultaneously. This distributed encoding may be a consequence of the high-dimensional nature of the representation space and the redundancy inherent in ResNet architectures. Whether this redundancy is a deliberate property of the architecture or an emergent consequence of gradient-based optimisation is an open question.

***D. Limitations***

All results are from ResNet-18 and ResNet-34 on Split CIFAR-100. Generalization to transformer architectures, larger datasets, or domain-incremental settings remains to be established. The reduced AG in ResNet-34 suggests that architecture may modulate the storage-accessibility decoupling. The Accessibility Anchor stores feature vectors rather than raw data, but still requires a small set of Task 0 inputs at anchor extraction time — it is not fully data-free. The k ≈ 8 compressibility result may be dataset- or architecture-specific.

# Conclusion

We have presented eighteen controlled experiments establishing that catastrophic forgetting is not primarily a process of information destruction. Across two architectures, three learning paradigms, and three random seeds, we consistently find an Accessibility Gap of AG ≈ 0.43–0.48: the network retains substantial representational information about forgotten tasks that it cannot access through its active classifier.

Our three primary contributions go beyond this baseline finding. First, recoverability is massively compressible: 64× compression of the probe weight matrix retains 96% of full recoverability, establishing that forgotten information occupies a compact 8-dimensional geometric structure. Second, EWC and replay operate through fundamentally different mechanisms — storage preservation versus accessibility restoration — establishing empirically that these are independently controllable properties. Third, the Accessibility Anchor achieves replay-level task accuracy (+165% vs. SGD) without data storage, by directly targeting accessibility as a geometric property of the representation space.

We propose Dynamic Information Accessibility Theory (DIAT) as a unified framework for these findings, reframing catastrophic forgetting as geometric accessibility failure rather than information destruction, and providing four falsifiable predictions for future investigation. The immediate research agenda is causal validation — directly testing whether destroying or realigning the recovery manifold produces the corresponding recoverability changes — and architectural generalisation beyond the ResNet family.

The broader implication is that the continual learning community may benefit from an expanded set of objectives: not only preserving what the network knows, but preserving its ability to access what it knows. The distinction between storage and accessibility, and the tools to measure and manipulate it, are the primary contributions of this work.

##### References

1. G. I. Parisi et al., "Continual lifelong learning with neural networks: A review," Neural Networks, vol. 113, pp. 54–71, 2019.
2. M. De Lange et al., "A continual learning survey: Defying forgetting in classification tasks," IEEE TPAMI, 2022
3. M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks," Psychology of Learning and Motivation, vol. 24, pp. 109–165, 1989.
4. R. M. French, "Catastrophic forgetting in connectionist networks," Trends in Cognitive Sciences, vol. 3, no. 4, pp. 128–135, 1999.
5. J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS, vol. 114, no. 13, pp. 3521–3526, 2017.
6. F. Zenke, B. Poole, and S. Ganguli, "Continual learning through synaptic intelligence," ICML, 2017.
7. J. Serra et al., "Overcoming catastrophic forgetting with hard attention to the task," ICML, 2018.
8. D. Rolnick et al., "Experience replay for continual learning," NeurIPS, 2019.
9. H. Shin et al., "Continual learning with deep generative replay," NeurIPS Workshop, 2017.
10. A. A. Rusu et al., "Progressive neural networks," arXiv:1606.04671, 2016.
11. A. Mallya and S. Lazebnik, "PackNet: Adding multiple tasks to a single network by iterative pruning," CVPR, 2018.
12. R. Hadsell et al., "Embracing change: Continual learning in deep neural networks," Trends in Cognitive Sciences, vol. 24, no. 12, pp. 1028–1040, 2020.
13. V. V. Ramasesh, E. Dyer, and M. Raghu, "Anatomy of catastrophic forgetting," ICLR, 2021.
14. C. V. Nguyen et al., "Toward understanding catastrophic forgetting in continual learning," arXiv, 2019.
15. I. J. Goodfellow et al., "An empirical investigation of catastrophic forgetting in gradient-based neural networks," ICLR Workshop, 2015.
16. M. Davari et al., "Probing representation forgetting in supervised and unsupervised continual learning," CVPR, 2022.
17. J. Zheng et al., "Spurious forgetting in continual learning of language models," ICLR, 2025.
18. G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," ICLR Workshop, 2016.
19. S. Kornblith et al., "Similarity of neural network representations revisited," ICML, 2019.
20. M. Raghu et al., "SVCCA: Singular vector canonical correlation analysis," NeurIPS, 2017.
21. V. Papyan, X. Han, and D. Donoho, "Prevalence of neural collapse during the terminal phase of deep learning training," PNAS, 2020.
22. B. Saha, H. Garg, and K. Roy, "Gradient projection memory for continual learning," ICLR, 2021.
23. A. Krizhevsky, "Learning multiple layers of features from tiny images," Technical Report, 2009.
24. K. He et al., "Deep residual learning for image recognition," CVPR, 2016.
25. Z. Chen and B. Liu, "Lifelong Machine Learning," 2nd Ed., Morgan & Claypool, 2018.
26. A. Chaudhry et al., "Riemannian walk for incremental learning," ECCV, 2018.
27. G. M. van de Ven and A. Tolias, "Three scenarios for continual learning," NeurIPS Workshop, 2019.
28. A. Trivedi and B. Melwani, "Catastrophic Forgetting as Accessibility Collapse: A Three-Level Framework for Knowledge Persistence in Continual Learning," arXiv:2606.06032, 2026.
29. A. Trivedi and B. Melwani, "The Stable Recovery Manifold: Geometric Principles Governing Recoverability in Continual Learning," arXiv:2606.13637, 2026