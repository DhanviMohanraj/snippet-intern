# Electronics Recommendation Agent

An intelligent, multi-category electronics recommendation engine powered by **Local MongoDB**, **FastAPI**, and **Groq Cloud API**.

> **Architecture Principle:**  
> *"The LLM understands and words; deterministic Python code decides and enforces."*

---

## 1. Project Overview

The **Electronics Recommendation Agent** assists consumers in finding the right electronics across five primary categories:
- **Laptops**
- **Smartphones**
- **Headphones**
- **Tablets**
- **Monitors**

Rather than allowing an LLM to hallucinate specifications or prices, the agent utilizes a hybrid architecture:
1. **Natural Language Understanding (NLU):** Groq Cloud API parses conversational user requests into structured constraint filters (e.g., budget limits, performance targets, form factors).
2. **Deterministic Matching & Scoring:** Pure Python domain logic queries a local MongoDB catalogue, applies hard constraints, ranks candidate products, and deterministically selects the top two matching products.
3. **Conversational Synthesis:** Groq formats the final recommendation into natural, helpful conversational responses justifying the choice.

---

## 2. Technology Stack

- **Backend Framework:** FastAPI 0.110+ (Python 3.12)
- **ASGI Server:** Uvicorn with standard extras
- **Data Validation & Settings:** Pydantic v2 & Pydantic-Settings
- **Database Driver:** Motor (asyncio driver for MongoDB)
- **Database Engine:** Local MongoDB instance (`mongodb://localhost:27017`)
- **LLM Provider:** Groq Cloud API (`https://api.groq.com/openai/v1`)
- **Testing & Verification:** Pytest, Pytest-Asyncio, HTTPX
- **Frontend (Future):** React + Vite (planned for later module)

---

## 3. Project Modules & Scope

### Module 1: Project Scaffold and Configuration
- Standardized directory layout for backend, frontend placeholder, and documentation.
- Typed settings configuration loaded from `.env` and system environment variables.
- Asynchronous connection manager for local MongoDB with admin ping verification.
- Centralized logging with credential masking and duplicate handler prevention.
- Reusable `AppError` and custom exception handlers with `X-Request-ID` tracking.
- Automated test suite validating configuration, lifecycle, and API endpoints without external network calls.

### Module 2: Data Layer and Product Catalogue
- Validated Pydantic v2 `Product` model supporting 5 categories with category-specific specification validation.
- MongoDB persistence via `ProductRepository` providing `get()`, `find()`, `count()`, and `categories()`.
- Synthetic product catalogue (`backend/data/products.json`) containing exactly 120 products with diverse specs and edge cases.
- Idempotent bulk upsert seed script (`backend/scripts/seed.py`) with automatic index creation.
- 53 automated tests covering models, repository operations, idempotency, and application lifecycle.

---

## 4. Repository Structure

```text
snippet-intern/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Application package root
│   │   ├── main.py              # FastAPI app factory and lifecycle management
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py        # Pydantic Settings & environment validation
│   │   │   ├── logging.py       # Structured logging & secret scrubbing filter
│   │   │   └── errors.py        # AppError hierarchy, request IDs & handlers
│   │   ├── api/
│   │   │   └── __init__.py      # API routers placeholder (/api/v1)
│   │   ├── models/
│   │   │   ├── __init__.py      # Export Product, Category, CategorySummary
│   │   │   └── product.py       # Pydantic Product model & category spec schemas
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── mongo.py         # Async Motor connection manager & ping check
│   │   │   └── repositories.py  # Asynchronous ProductRepository
│   │   ├── llm/
│   │   │   └── __init__.py      # Groq client integration (Module 4)
│   │   └── agent/
│   │       └── __init__.py      # Recommendation engine logic (Module 5)
│   ├── data/
│   │   └── products.json        # 120-item synthetic product catalogue
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── seed.py              # Idempotent MongoDB catalogue seed script
│   ├── tests/
│   │   ├── test_scaffold.py     # Module 1 scaffold and lifecycle tests
│   │   ├── test_product_model.py# Product model validation & spec tests
│   │   ├── test_product_repository.py # Async repository queries & aggregation tests
│   │   └── test_seed.py         # Catalogue integrity & seed idempotency tests
│   ├── pytest.ini               # Pytest configuration
│   └── requirements.txt         # Bounded backend Python dependencies
├── frontend/
│   └── .gitkeep                 # React frontend placeholder
├── docs/
│   └── .gitkeep                 # Project documentation placeholder
├── .env                         # Local environment file (git-ignored)
├── .env.example                 # Example configuration template
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

---

## 5. Prerequisites

1. **Python 3.12+**
   - Verify Python: `python --version` (or `py -3.12 --version` on Windows)
2. **Local MongoDB**
   - MongoDB Community Server must be installed and running locally on port `27017`.
   - The backend connects directly to your existing local MongoDB service; it does not install, start, stop, or manage MongoDB services.
   - **No Docker or Docker Compose is used.**
3. **Groq API Key (for future modules)**
   - Groq API credentials are configured via the root `.env` file.
   - Placeholder keys (such as `replace_me`) allow the backend to boot and run local routes without failing.

---

## 6. Environment Configuration

Copy `.env.example` to create `.env` in the root directory:

```bash
cp .env.example .env
```

### Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `MONGO_URI` | MongoDB connection URI | `mongodb://localhost:27017` |
| `MONGO_DB` | MongoDB database name | `electronics_agent` |
| `LLM_BASE_URL` | Base URL for Groq API | `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | Private Groq API key | `replace_me` |
| `LLM_MODEL` | Default inference model identifier | `openai/gpt-oss-20b` |
| `LLM_FALLBACK_BASE_URL` | Optional secondary LLM endpoint | *None* |
| `LLM_FALLBACK_MODEL` | Optional secondary model name | *None* |
| `LLM_TIMEOUT_S` | Request timeout in seconds | `30` |
| `LLM_MAX_RETRIES` | Max retries for failed requests | `2` |
| `CORS_ORIGINS` | JSON array of permitted web client origins | `["http://localhost:5173"]` |

---

## 7. Installation and Setup

### Windows (PowerShell)

```powershell
# 1. Navigate to the backend directory
cd backend

# 2. Create Python 3.12 virtual environment
py -3.12 -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install backend dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux (Bash / Zsh)

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create Python 3.12 virtual environment
python3.12 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install backend dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 8. Product Data Layer (Module 2)

### Product Schema

Every product is validated through Pydantic v2:
- `product_id`: Unique string ID (e.g., `LAP-0001`, `PHN-0001`, `HPH-0001`, `TAB-0001`, `MON-0001`).
- `name`: Product display name.
- `brand`: Fictional manufacturer / brand.
- `category`: Exactly one of `laptop`, `phone`, `headphones`, `tablet`, `monitor`.
- `price_inr`: Positive integer (rupees). Booleans and floats are strictly rejected.
- `specs`: Validated dictionary containing only permitted category-specific keys. Missing specifications remain omitted rather than fabricated.
- `intended_use`: List of non-empty strings defining target personas / use cases.
- `features`: List of key selling points / features.
- `in_stock`: Boolean stock status.
- `rating`: Optional float between `0.0` and `5.0`, or `None` if unrated.

### Category-Specific Specifications

| Category | Allowed Specification Keys |
|---|---|
| **Laptop** | `cpu`, `ram_gb`, `storage_gb`, `gpu`, `vram_gb` (>=0), `weight_kg`, `battery_hr`, `screen_in`, `os` |
| **Phone** | `chipset`, `ram_gb`, `storage_gb`, `camera_mp`, `battery_mah`, `screen_in`, `weight_g`, `os` |
| **Headphones** | `type` (`over-ear` or `in-ear`), `anc` (bool), `battery_hr`, `wireless` (bool), `weight_g`, `codec` |
| **Tablet** | `chipset`, `ram_gb`, `storage_gb`, `screen_in`, `stylus_support` (bool), `battery_hr`, `weight_g` |
| **Monitor** | `size_in`, `resolution`, `refresh_hz`, `panel` (`IPS`, `VA`, `OLED`), `ports` (list of strings) |

*Unknown specification keys or specifications from another category are strictly rejected.*

### Synthetic Catalogue Dataset

The catalogue (`backend/data/products.json`) contains **120 synthetic products**:

| Category | Count | Price Range (INR) |
|---|---:|---|
| Laptops | 40 | ₹31,999 – ₹2,49,000 |
| Phones | 30 | ₹9,999 – ₹1,49,999 |
| Headphones | 20 | ₹1,299 – ₹34,999 |
| Tablets | 15 | ₹9,999 – ₹89,999 |
| Monitors | 15 | ₹7,999 – ₹72,000 |
| **Total** | **120** | |

> **Synthetic Data Disclaimer:**  
> All brands (e.g. Zenit, Nexa, Voltix, Lumina, Aether, Apex, Titan, Sonix, Ocular) and product specifications in this dataset are entirely fictional and designed exclusively for demonstration, benchmarking, and recommendation evaluation.

### Deliberate Edge Cases Included:
1. **Budget Thresholds:** Products priced immediately above common budget barriers (e.g., `LAP-0011` at ₹80,500).
2. **Weight vs. Performance:** Heavy workstation/gaming laptops (e.g., 2.95kg with RTX 4090) vs. featherweights (e.g., 1.05kg with integrated graphics).
3. **Out-of-Stock Products:** 8 out-of-stock items distributed across categories.
4. **Missing Specifications:** Real-world items where non-critical specs (like `vram_gb`, `battery_hr`, or `weight_g`) are legitimately omitted.
5. **Rating Diversity:** Top-rated items (4.8–4.9), budget items (3.7–4.0), and unrated items (`rating: null`).

### MongoDB Indexes

The following indexes are ensured on the `products` collection:
- `idx_product_id_unique`: Unique index on `product_id`.
- `idx_category_price_inr`: Compound index on `category` (ASC) and `price_inr` (ASC).
- `idx_intended_use`: Index on `intended_use` (ASC) for fast tag filtering.

### Running the Seed Script

From the `backend` directory with the virtual environment activated:

```powershell
python -m scripts.seed
```

Example Output:
```text
==================================================
ELECTRONICS RECOMMENDATION AGENT — CATALOGUE SEED
==================================================
Products Loaded from JSON: 120
New Records Inserted:     120
Existing Records Updated:  0
Records Unchanged:         0
Total Products in Database:120
==================================================
```

Running the seed script a second time is **100% idempotent**:
```text
Products Loaded from JSON: 120
New Records Inserted:     0
Existing Records Updated:  0
Records Unchanged:         120
Total Products in Database:120
```

### Verifying Seeded Records in MongoDB

Run a quick query from Python to verify catalogue aggregation:

```powershell
python -c "import asyncio; from app.db.mongo import AsyncIOMotorClient, get_settings; from app.db.repositories import ProductRepository; async def check(): c = AsyncIOMotorClient(get_settings().mongo_uri); repo = ProductRepository(c[get_settings().mongo_db]['products']); print([(s.category.value, s.product_count) for s in await repo.categories()]); c.close(); asyncio.run(check())"
```

---

## 9. Running the Application

Ensure local MongoDB is running, then start Uvicorn from the `backend` directory:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Interactive API Documentation
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Core Endpoints
- `GET /`: Returns service metadata, version, and running state.
- `GET /live`: Liveness probe for uptime monitors.

---

## 10. Running Automated Tests

Run the full automated test suite with pytest from the `backend` directory:

```powershell
python -m pytest -v
```

All 53 unit and integration tests run cleanly:
- **`test_scaffold.py` (19 tests):** Module 1 settings validation, CORS, error formatting, lifespan startup/shutdown, secret masking.
- **`test_product_model.py` (14 tests):** Product schema validation, price integer constraints, rating boundaries, category specification enforcement, contradictory spec rejection.
- **`test_product_repository.py` (14 tests):** Async MongoDB queries, pagination, deterministic sorting, category/price/use-case filtering, category summary aggregation.
- **`test_seed.py` (6 tests):** Catalogue counts (40/30/20/15/15), edge case verification, bulk upsert idempotency, document modification, and index verification.

---

## 11. Local MongoDB Troubleshooting

If startup or seeding fails with `DatabaseConnectionError`:
1. **Verify service status:** Ensure MongoDB Community service is active:
   - **Windows:** Check Services (`services.msc`) for `MongoDB Server (MongoDB)` and confirm its status is `Running`.
   - Alternatively, test port connectivity in PowerShell:
     ```powershell
     Test-NetConnection -ComputerName 127.0.0.1 -Port 27017
     ```
2. **Check URI:** Ensure `MONGO_URI=mongodb://localhost:27017` in your `.env` matches your local configuration.
3. **Firewall / Port conflict:** Verify that port 27017 is not blocked by another process.

---

## 12. Security and Git Hygiene

- **Root `.env` is Git-Ignored:** Sensitive keys, connection strings, and local configuration must never be committed. Verify with `git status --ignored`.
- **`.env.example` is Trackable:** Kept up to date as a configuration contract for all team members.
- **Log Masking:** Passwords in MongoDB URIs and sensitive authorization headers are automatically scrubbed by `SensitiveDataFilter`.
- **No Cloud LLM Calls on Startup:** Placeholder API keys will never trigger outbound requests during application boot.

---

## 13. Roadmap & Upcoming Modules

- [x] **Module 1:** Project Scaffold, Async MongoDB Lifespan & Configuration
- [x] **Module 2:** Electronics Product Catalogue & MongoDB Repository Layer
- [ ] **Module 3:** Health Check & Readiness Endpoints (`/api/v1/health` verifying MongoDB and Groq status)
- [ ] **Module 4:** Groq LLM Client & Natural Language Requirement Extraction
- [ ] **Module 5:** Deterministic Scoring, Filtering, & Recommendation Agent
- [ ] **Module 6:** React + Vite Frontend & Conversational Chat Interface
