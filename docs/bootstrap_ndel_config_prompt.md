# NDEL Bootstrap Config Prompt

This document contains a **ready-to-use prompt** you can paste into an LLM (such as GPT-5.1 or GPT-5.1-codex) that has access to *your* data science / ML codebase.

The model will:
- Inspect your repository,
- Infer datasets, models, features, and privacy concerns,
- Generate a project-local `ndel_profile.py` with a `make_ndel_config()` function that returns an `NdelConfig`.

> **Recommended workflow**
>
> 1. Ensure the LLM can see your repo (via GitHub integration, local tooling, etc.).
> 2. Paste the prompt block below into the LLM.
> 3. Ask it to produce `ndel_profile.py` at the root of your project.
> 4. Review and edit the file (especially privacy assumptions).
> 5. Commit `ndel_profile.py` to version control and import `make_ndel_config()` when using NDEL.

---

## Prompt to paste into your LLM

Copy everything inside this fenced block into your LLM:

```text
You are an AI assistant helping configure **NDEL**, a descriptive DSL for data science and machine learning code.

NDEL is:
- A post-facto, read-only description layer over existing Python DS/ML code (and, later, SQL).
- A Python **library**, not a CLI.
- Used to turn internal pipelines into human-readable descriptions, without executing or generating code.

Your job is to inspect THIS repository and generate a Python module called **`ndel_profile.py`** that defines a single function:

    def make_ndel_config() -> NdelConfig: ...

This function should return an `NdelConfig` instance tailored to this codebase.

You must NOT execute any code. Work purely from static signals: filenames, imports, function/variable names, comments, and directory structure.


------------------------------------------------------------
1. NDEL mental model and available types
------------------------------------------------------------
NDEL exposes config types from the `ndel` package. You can assume the following approximate shapes (do NOT redefine these; just import and use them):

    from ndel import (
        NdelConfig,
        PrivacyConfig,
        DomainConfig,
        AbstractionLevel,
    )

Conceptually:

- PrivacyConfig controls what MUST NOT leak into NDEL descriptions.
- DomainConfig maps code-level names (e.g. df_users, model) to human- and domain-level aliases.
- AbstractionLevel controls how detailed descriptions can be.
- NdelConfig bundles the above.

The types look roughly like this:

    class AbstractionLevel(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    @dataclass
    class PrivacyConfig:
        hide_table_names: bool = False
        hide_file_paths: bool = False
        redact_identifiers: List[str] = field(default_factory=list)
        max_literal_length: int = 200

    @dataclass
    class DomainConfig:
        dataset_aliases: Dict[str, str] = field(default_factory=dict)
        model_aliases: Dict[str, str] = field(default_factory=dict)
        feature_aliases: Dict[str, str] = field(default_factory=dict)
        pipeline_name: Optional[str] = None

    @dataclass
    class NdelConfig:
        privacy: Optional[PrivacyConfig] = None
        domain: Optional[DomainConfig] = None
        abstraction: AbstractionLevel = AbstractionLevel.MEDIUM


------------------------------------------------------------
2. How to analyze this repo
------------------------------------------------------------
Scan the repository with these priorities:

(1) Data / datasets
- Look for where data is loaded, e.g.:
  - pandas IO: pd.read_csv, read_parquet, read_table, read_sql
  - Spark or other readers: spark.read.*, custom load_* helpers, etc.
- Infer:
  - Common dataset variables (df_users, events, sessions, etc.).
  - High-level meanings (e.g. “clickstream events over time”, “daily user aggregates”).
  - Obvious PII / sensitive fields (columns or variables related to email, IP, addresses, payment details, etc.).

(2) Models
- Detect where models are constructed and trained:
  - sklearn, XGBoost, LightGBM, CatBoost, PyTorch, Keras, or project-specific estimators.
  - Patterns like: model = SomeEstimator(...); model.fit(X, y)
- Infer:
  - Typical model variable names (churn_model, clf, recommender, etc.).
  - What they predict, if you can infer from target names or comments (e.g. churn, risk, click-through).

(3) Features / transformations
- Detect feature engineering code:
  - New columns built on DataFrames (df["is_power_user"] = ...).
  - sklearn Pipeline objects.
  - Functions named like make_features / build_features / transform_data.
- Capture feature names and, where possible, their domain intuition.

(4) Metrics / evaluation
- Identify evaluation patterns (AUC, accuracy, RMSE, F1, etc.).
- Note which datasets are used for train/validation/test if that is clear.

(5) Privacy / sensitivity
- Identify any information that should almost certainly be hidden in NDEL descriptions:
  - Raw S3 bucket names, internal schema names, internal hostnames.
  - PII-related columns: email, ip, address, phone, ssn, card, etc.
  - Secrets or tokens (DO NOT show actual values; only recognize the presence of such fields).

You MUST NOT execute code. Everything is static analysis.


------------------------------------------------------------
3. What to put into NdelConfig
------------------------------------------------------------
You will produce a module `ndel_profile.py` with:

    from ndel import (
        NdelConfig,
        PrivacyConfig,
        DomainConfig,
        AbstractionLevel,
    )

    def make_ndel_config() -> NdelConfig:
        ...

Inside `make_ndel_config()`, construct:

1) PrivacyConfig
   - Set hide_table_names=True if you see raw internal table/schema names or cloud paths (e.g., internal S3 buckets) that shouldn’t appear in public descriptions.
   - Set hide_file_paths=True if there are local or network paths with internal structure.
   - Populate redact_identifiers with **names**, not values, for fields that are likely PII or highly sensitive, e.g.:
     - "email", "ip", "ip_address", "phone", "address", "ssn", "card", "customer_id", etc.
   - Use a conservative max_literal_length (e.g. 200) to avoid dumping huge literal blobs (SQL strings, long descriptions) into NDEL output.
   - Add a short comment near PrivacyConfig explaining your choices and include a TODO for human review, for example:
     - # TODO: review and adjust privacy assumptions for this project.

2) DomainConfig.dataset_aliases
   - Keys: dataset variable names you observe (df_users, events, sessions, aggregates, etc.).
   - Values: human-readable names inferred from context, focused on business meaning, e.g.:
     - "users_activity_30d", "clickstream_events", "transactions_daily", "user_profiles_snapshot".
   - Prefer stable domain concepts over internal table or file names.

3) DomainConfig.model_aliases
   - Keys: model variable names (churn_model, model, clf, recommender, etc.).
   - Values: clear conceptual labels, e.g.:
     - "user_churn_classifier", "product_recommender", "fraud_risk_model".
   - Use comments when unsure, e.g.:
     - # TODO: confirm whether this model is really for churn or something else.

4) DomainConfig.feature_aliases
   - Keys: feature or column names (is_power_user, sessions_per_day, risk_score, etc.).
   - Values: more descriptive or business-friendly names, e.g.:
     - "power_user_flag", "avg_sessions_per_active_day", "credit_risk_score".
   - Do not attempt to enumerate every feature in a very large codebase; focus on the most central features that appear in training pipelines or are widely referenced.

5) DomainConfig.pipeline_name
   - Set a meaningful default pipeline name such as:
     - "churn_prediction_pipeline", "recommendation_training_pipeline", "fraud_detection_batch_job".
   - If there are multiple major pipelines, pick the most central one, and leave a TODO comment to split or extend config later if needed.

6) AbstractionLevel
   - Choose AbstractionLevel.MEDIUM unless:
     - HIGH is clearly more appropriate (only high-level descriptions should be exposed), or
     - LOW is clearly appropriate (very detailed feature-level descriptions are desired).
   - Add a comment explaining your choice.

7) NdelConfig
   - Construct and return:

     NdelConfig(
         privacy=PrivacyConfig(...),
         domain=DomainConfig(...),
         abstraction=AbstractionLevel.MEDIUM,  # or HIGH/LOW as justified
     )


------------------------------------------------------------
4. Output format and style
------------------------------------------------------------
Your **entire final answer** must be valid Python code for `ndel_profile.py`, containing:

- Exactly one public function: make_ndel_config() -> NdelConfig
- Any necessary imports from ndel:
  
  from ndel import (
      NdelConfig,
      PrivacyConfig,
      DomainConfig,
      AbstractionLevel,
  )

- No redefinitions of NdelConfig, PrivacyConfig, DomainConfig, or AbstractionLevel.
- Helpful inline comments explaining:
  - where each alias or setting came from (e.g., file/function names),
  - where you are uncertain (use clear `TODO` comments),
  - any assumptions that a human reviewer should double-check.

Do NOT:
- Print raw secrets, access tokens, or actual PII examples.
- Execute repository code.
- Add extra text outside the Python module content. The answer should be directly saveable as `ndel_profile.py`.

Be conservative with privacy:
- When in doubt, add more to redact_identifiers or hide_table_names and flag it with a TODO for human review.
- Prefer underspecified, safe descriptions over leaking internal details.
```
