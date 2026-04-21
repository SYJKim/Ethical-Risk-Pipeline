# Topic-to-Category Mapping Codebook

This codebook reproduces the operational coding guidelines used in the
topic-to-category mapping stage of the paper. It is intended to enable
independent coders to replicate the mapping procedure that assigned the 33
BERTopic-derived sub-topics to the five higher-level ethical risk categories.

This file mirrors **Appendix B** of the paper (Kim, Kim, & Lee, 2026).

## 1. Purpose

Provide operational criteria for post-hoc alignment of non-outlier topics
identified through BERTopic with the five predefined ethical risk categories.

## 2. Unit of Coding

- The unit of coding is **one non-outlier topic**.
- Each topic is assigned to **only one primary category** (single-label
  assignment), to maintain mutual exclusivity across categories and prevent
  distortion of category-level shares.
- Documents classified as Topic = −1 (outliers) by HDBSCAN are **excluded**
  from the category-mapping process.

## 3. Mapping Criteria

Category assignment for each topic is conducted by jointly reviewing:

1. the top keywords produced by c-TF-IDF,
2. the **top three representative documents** ranked by topic membership
   probability, and
3. the topic label.

Where category assignment remains uncertain from three documents alone, up
to **two additional documents** may be reviewed as supplementary evidence.

## 4. Decision Rule

Each topic is assigned to a single category based on the **most direct harm
mechanism** expressed in its representative documents. Decisions prioritize
the semantic core that repeatedly appears in the representative documents
rather than the mere presence of individual keywords.

If a topic appears to relate to more than one category, determine the
primary category based on whether the central risk arises from:

- the failure of an individual system → **1. Technical safety**
- a data processing issue → **2. Privacy and data misuse**
- unfairness or rights violations → **3. Fairness and discrimination**
- intentional malicious misuse → **4. Malicious misuse**
- structural harms at the societal or institutional level → **5. Societal
  and democratic risks**

## 5. Category Definitions and Boundary Rules

The five ethical risk categories and their decision boundaries are defined
as follows (paper Table 4).

### 1. Technical safety

**Operational definition.** Cases dealing with safety issues inherent to the
system itself, such as model errors, hallucinations, instability, unsafe
outputs, or lack of explainability.

**Indicative examples.** Hallucinations; incorrect answers; calculation
errors; fabricated information; unsafe advice related to medical, legal, or
safety contexts; abnormal outputs; reliability issues caused by unstable
model behavior.

**Decision boundary.**
- If the core issue involves **intentional attacks, fraud, or circumvention**,
  prioritize **4. Malicious misuse**.
- If the key issue is **data leakage**, prioritize **2. Privacy and data
  misuse**.

### 2. Privacy and data misuse

**Operational definition.** Cases where the core issue concerns the
collection, leakage, exposure, or unauthorized use of personal data,
sensitive information, conversation records, or user data.

**Indicative examples.** Exposure of personal data; leakage of sensitive
information; unauthorized collection of training data; exposure of
conversation logs; breach of confidentiality; issues in processing
sensitive data.

**Decision boundary.**
- If the data exposure **directly facilitates attacks, fraud, or criminal
  activities**, prioritize **4. Malicious misuse**.

### 3. Fairness and discrimination

**Operational definition.** Cases where distributive or normative unfairness
is central, such as bias or discrimination against specific individuals or
groups, unfair outcomes, or violations of creators' rights.

**Indicative examples.** Bias related to race, gender, or religion; biased
outputs; training-data bias; copyright infringement; plagiarism; violations
of creators' rights.

**Decision boundary.**
- If the central issue concerns **macro-level institutional harms** such as
  democracy, labor markets, education, or censorship, prioritize **5.
  Societal and democratic risks**.

### 4. Malicious misuse

**Operational definition.** Cases where generative AI is intentionally
exploited for harmful purposes such as attacks, fraud, misinformation
campaigns, security circumvention, or criminal activities.

**Indicative examples.** Jailbreaking; malware generation; phishing; fraud;
misinformation/disinformation; criminal use.

**Decision boundary.**
- If the core issue is an **unintended model error or hallucination**,
  prioritize **1. Technical safety**.
- If the issue is **simple data leakage**, prioritize **2. Privacy and data
  misuse**.

### 5. Societal and democratic risks

**Operational definition.** Cases where the central concern involves
macro-level or institutional harms such as impacts on labor markets,
education systems, democratic processes, human rights, censorship, digital
divides, or social inequality.

**Indicative examples.** Job displacement; political bias; censorship;
educational disruption; human-rights concerns; distortions in the public
sphere.

**Decision boundary.**
- If the issue centers on **direct discrimination against a specific group**,
  prioritize **3. Fairness and discrimination**.
- If the issue primarily concerns **system malfunction**, prioritize **1.
  Technical safety**.

## 6. Reliability Protocol

- Two doctoral researchers in Management Information Systems (MIS),
  independent of the coders involved in the human annotation stage,
  performed the mapping independently for all 33 non-outlier topics.
- Inter-rater reliability on the initial independent mapping was assessed
  using Cohen's kappa.
- Disagreements were resolved through discussion to finalize category
  assignments. The final reconciled assignments are recorded in
  `data/topic_category_mapping.csv` under the `Gold_label` column.

## Reference

Kim, S., Kim, S. H., & Lee, B. G. (2026). A Reproducible Framework for
LLM-Based Classification and Structural Analysis of Ethical Risk Discourse
in Social Media. *PeerJ Computer Science* (under review).
