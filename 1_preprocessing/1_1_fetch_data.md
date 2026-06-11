## Fetch Data Workflow

```python
# ── 0. Imports ────────────────────────────────────────────────
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ── 1a. Load from Database ────────────────────────────────────
load_dotenv()

DB_STRING = os.getenv("DB_STRING")

if DB_STRING is None:
    raise ValueError("DB_STRING environment variable is not set")

db = create_engine(DB_STRING)

query_string = """
SELECT
    x,
    y,
    z
FROM ...
JOIN m ON n
"""

df = pd.read_sql(query_string, db)

# Save to CSV for offline use (avoid unnamed index column on re-import)
df.to_csv("data.csv", index=False)

# ── 1b. Load from CSV (alternative) ──────────────────────────
df = pd.read_csv("data.csv")

df.head()
```