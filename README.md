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
- Automated scaffold tests verifying lifecycle, config, and offline operation.

### Module 2: Data Layer and Product Catalogue
- Validated Pydantic v2 `Product` model supporting 5 categories with category-specific specification validation.
- MongoDB persistence via `ProductRepository` providing `get()`, `find()`, `count()`, and `categories()`.
- Synthetic product catalogue (`backend/data/products.json`) containing exactly 120 products with diverse specs and edge cases.
- Idempotent bulk upsert seed script (`backend/scripts/seed.py`) with automatic index creation.

### Module 3: Product and Health API
- Read-only public REST API routes mounted under `/api/v1`.
- `GET /api/v1/products`: Filterable, paginated product search with deterministic sorting.
- `GET /api/v1/products/{product_id}`: Single product retrieval with 404 error envelope handling.
- `GET /api/v1/products/categories`: Category summary aggregation endpoint with counts and price bounds.
- `GET /api/v1/health`: Dependency health check reporting MongoDB and Groq status independently without text generation.
- 81 automated tests verifying models, repository operations, API responses, and health diagnostics.

### Module 4: Groq LLM Client & Structured Output Infrastructure
- Reusable, asynchronous Groq Cloud API client implementing the `LLMClient` protocol.
- Unstructured text generation (`complete_text`) with token limits and temperature controls.
- Structured JSON output (`complete_json`) validated directly against Pydantic v2 schemas.
- Resilient retry policy with bounded exponential backoff and `Retry-After` header parsing.
- Sanitized exception hierarchy protecting secrets and API credentials.
- Deterministic `FakeLLMClient` for offline testing and downstream module development.
- 27 new tests (108 total suite) verifying completions, errors, retries, and offline isolation.


---

## 4. Repository Structure

```text
snippet-intern/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Application package root
│   │   ├── main.py              # FastAPI app factory, lifespan, and router mounting
│   │   ├── api/
│   │   │   ├── __init__.py      # Export health_router, products_router
│   │   │   ├── products.py      # Product listing, filtering, detail, & categories
│   │   │   └── health.py        # Independent MongoDB & Groq health check
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py        # Pydantic Settings & environment validation
│   │   │   ├── logging.py       # Structured logging & secret scrubbing filter
│   │   │   └── errors.py        # AppError hierarchy, request IDs & handlers
│   │   ├── models/
│   │   │   ├── __init__.py      # Export domain and API response models
│   │   │   ├── product.py       # Pydantic Product model & category spec schemas
│   │   │   └── api.py           # ProductListResponse, HealthResponse, ErrorResponse
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── mongo.py         # Async Motor connection manager & ping check
│   │   │   └── repositories.py  # Asynchronous ProductRepository
│   │   ├── llm/
│   │   │   ├── __init__.py      # Module 4 exports (LLMClient, GroqClient, FakeLLMClient, errors)
│   │   │   ├── client.py        # GroqClient implementation & retry logic
│   │   │   ├── errors.py        # Sanitized LLM exception hierarchy
│   │   │   └── fake.py          # Deterministic offline FakeLLMClient
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
│   │   ├── test_seed.py         # Catalogue integrity & seed idempotency tests
│   │   ├── test_api_products.py # Product listing, filter, detail & categories tests
│   │   ├── test_api_health.py   # Health check states & credential safety tests
│   │   └── test_llm_client.py   # Module 4 Groq LLM client, retries, and offline fake tests

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
3. **Groq API Key (for future modules & health check)**
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

## 8. Database Seeding (Module 2)

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

Running the seed script a second time is **100% idempotent** and produces 0 duplicates.

---

## 9. Product & Health API Routes (Module 3)

All application endpoints are mounted under `/api/v1`.

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/products` | Filterable, paginated search over catalogue products. |
| `GET` | `/api/v1/products/{product_id}` | Retrieve single product by unique ID (e.g. `LAP-0001`). |
| `GET` | `/api/v1/products/categories` | Summary aggregation across all 5 supported categories. |
| `GET` | `/api/v1/health` | Independent status check for local MongoDB and Groq Cloud API. |

---

### Endpoint Details & Query Parameters

#### 1. `GET /api/v1/products`

Retrieve paginated product records matching multiple optional query filters:

| Parameter | Type | Validation / Constraints | Default |
|---|---|---|---|
| `category` | `string` | One of: `laptop`, `phone`, `headphones`, `tablet`, `monitor` | *None* |
| `min_price` | `integer` | Positive integer `>= 0` (inclusive) | *None* |
| `max_price` | `integer` | Positive integer `>= 0` (inclusive). Must be `>= min_price` | *None* |
| `use_case` | `string` | Filter by persona tag in `intended_use` (e.g. `programming`, `gaming`) | *None* |
| `brand` | `string` | Case-insensitive brand name match (e.g. `Zenit`, `Velo`) | *None* |
| `in_stock` | `boolean` | `true` for available, `false` for out of stock | *None* |
| `sort` | `string` | `price_asc`, `price_inr_asc`, `price_desc`, `price_inr_desc`, `rating_desc`, `rating_asc`, `name_asc`, `name_desc` | `price_inr_asc` |
| `page` | `integer` | Minimum: `1` | `1` |
| `page_size` | `integer` | Minimum: `1`, Maximum: `50` | `20` |

**Example Request:**
```http
GET /api/v1/products?category=laptop&min_price=40000&max_price=90000&use_case=programming&page=1&page_size=2
```

**Example Response (HTTP 200):**
```json
{
  "items": [
    {
      "product_id": "LAP-0001",
      "name": "Zenit Air 14",
      "brand": "Zenit",
      "category": "laptop",
      "price_inr": 54999,
      "specs": {
        "cpu": "Intel Core i5-13420H",
        "ram_gb": 16,
        "storage_gb": 512,
        "gpu": "Intel Iris Xe Graphics",
        "vram_gb": 0,
        "weight_kg": 1.35,
        "battery_hr": 10.5,
        "screen_in": 14.0,
        "os": "Windows 11 Home"
      },
      "intended_use": ["programming", "student", "general"],
      "features": ["Backlit Keyboard", "Fingerprint Reader", "FHD IPS Anti-Glare"],
      "in_stock": true,
      "rating": 4.4
    },
    {
      "product_id": "LAP-0026",
      "name": "Nexa CodeBook 14",
      "brand": "Nexa",
      "category": "laptop",
      "price_inr": 49999,
      "specs": {
        "cpu": "AMD Ryzen 5 5600H",
        "ram_gb": 16,
        "storage_gb": 512,
        "gpu": "AMD Radeon Graphics",
        "vram_gb": 0,
        "weight_kg": 1.45,
        "battery_hr": 8.0,
        "screen_in": 14.0,
        "os": "Windows 11 Home"
      },
      "intended_use": ["programming", "student"],
      "features": ["Tactile Keyboard", "Dual Heat Pipes"],
      "in_stock": true,
      "rating": 4.2
    }
  ],
  "total": 13,
  "page": 1,
  "page_size": 2,
  "total_pages": 7
}
```

---

#### 2. `GET /api/v1/products/{product_id}`

Retrieve single product record:

**Example Request:**
```http
GET /api/v1/products/LAP-0001
```

**Example Response (HTTP 200):**
```json
{
  "product_id": "LAP-0001",
  "name": "Zenit Air 14",
  "brand": "Zenit",
  "category": "laptop",
  "price_inr": 54999,
  "specs": {
    "cpu": "Intel Core i5-13420H",
    "ram_gb": 16,
    "storage_gb": 512,
    "gpu": "Intel Iris Xe Graphics",
    "vram_gb": 0,
    "weight_kg": 1.35,
    "battery_hr": 10.5,
    "screen_in": 14.0,
    "os": "Windows 11 Home"
  },
  "intended_use": ["programming", "student", "general"],
  "features": ["Backlit Keyboard", "Fingerprint Reader", "FHD IPS Anti-Glare"],
  "in_stock": true,
  "rating": 4.4
}
```

**Nonexistent Product (HTTP 404):**
```json
{
  "error": {
    "code": "PRODUCT_NOT_FOUND",
    "message": "Product with ID 'NONEXISTENT-999' was not found in the catalogue.",
    "request_id": "5f8a004eb76f44ecb74d4d7eb8090b8f"
  }
}
```

---

#### 3. `GET /api/v1/products/categories`

Retrieve category summary statistics:

**Example Request:**
```http
GET /api/v1/products/categories
```

**Example Response (HTTP 200):**
```json
[
  {
    "category": "laptop",
    "product_count": 40,
    "min_price_inr": 31999,
    "max_price_inr": 249000
  },
  {
    "category": "phone",
    "product_count": 30,
    "min_price_inr": 9999,
    "max_price_inr": 149999
  },
  {
    "category": "headphones",
    "product_count": 20,
    "min_price_inr": 1299,
    "max_price_inr": 34999
  },
  {
    "category": "tablet",
    "product_count": 15,
    "min_price_inr": 9999,
    "max_price_inr": 89999
  },
  {
    "category": "monitor",
    "product_count": 15,
    "min_price_inr": 7999,
    "max_price_inr": 72000
  }
]
```

---

#### 4. `GET /api/v1/health`

Diagnostic check reporting availability of local MongoDB and Groq Cloud API independently:

**Example Response (HTTP 200):**
```json
{
  "status": "degraded",
  "db": "up",
  "llm": "down"
}
```

- `status`: `"ok"` if both `db == "up"` and `llm == "up"`. Otherwise `"degraded"`.
- `db`: `"up"` if local MongoDB responds to an admin ping command; otherwise `"down"`.
- `llm`: `"up"` if Groq Cloud responds with HTTP 200 to an authenticated `/models` request; otherwise `"down"`.
- **Safe Offline Operation:** If `LLM_API_KEY` is empty or set to `"replace_me"`, the endpoint immediately marks `llm: "down"` without making outbound network requests.
- **No Completions Triggered:** The health check only queries provider model availability; it never invokes completion or text generation logic.

---

### Standard Error Handling

Error responses follow a uniform envelope with request correlation tracing:

```json
{
  "error": {
    "code": "PRODUCT_NOT_FOUND",
    "message": "The requested resource was not found.",
    "request_id": "848bb241413a48e7ba453a9e1e2bc00b"
  }
}
```

HTTP Status Codes:
- `200 OK`: Request succeeded.
- `404 Not Found`: Target product ID does not exist.
- `422 Unprocessable Entity`: Input validation error (e.g. `min_price > max_price`, `page_size > 50`, invalid sort or category).
- `500 Internal Server Error`: Unhandled server exception (internal stack traces are never exposed to clients).
- `503 Service Unavailable`: Core database connection failure during startup.

---

## 10. Groq LLM Client & Structured Output Infrastructure (Module 4)

Module 4 provides a reusable, asynchronous LLM infrastructure connecting to the **Groq Cloud API** using an OpenAI-compatible interface. It enforces clean separation between natural-language generation and deterministic data validation.

### Responsibilities
1. **Chat Completion Requests:** Asynchronously dispatches chat messages to Groq Cloud API (`https://api.groq.com/openai/v1`).
2. **Unstructured Text Output:** Returns generated text via `complete_text()` with caller-controlled token bounds.
3. **Structured Pydantic Validation:** Guarantees that structured responses returned by `complete_json()` strictly validate against caller-supplied Pydantic v2 schemas.
4. **Transient Resilience:** Bounded exponential backoff retries for network drops, timeouts, 5xx server errors, and 429 rate limits.
5. **Rate Limit Awareness:** Parses and adheres to HTTP `Retry-After` response headers.
6. **Credential Privacy:** Redacts API keys and Bearer tokens from exceptions, error messages, and logs using regex sanitization.
7. **Offline Testing:** Features a deterministic `FakeLLMClient` for offline test suites and downstream module development without network access.

### Client Interface (`LLMClient` Protocol)

```python
from typing import Protocol, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class LLMClient(Protocol):
    async def complete_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """Send chat messages and return the assistant response as text."""
        ...

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> T:
        """Send chat messages and return a validated Pydantic v2 model instance."""
        ...

    async def ping(self) -> bool:
        """Perform a lightweight provider connectivity check without generating text."""
        ...

    async def close(self) -> None:
        """Close underlying HTTP client transport resources."""
        ...
```

### Configuration Settings

| Setting | Environment Variable | Default | Description |
|---|---|---|---|
| `llm_base_url` | `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible base URL |
| `llm_api_key` | `LLM_API_KEY` | `replace_me` | Private Groq API key |
| `llm_model` | `LLM_MODEL` | `openai/gpt-oss-20b` | Target inference model ID |
| `llm_timeout_s` | `LLM_TIMEOUT_S` | `30.0` | Request timeout in seconds |
| `llm_max_retries` | `LLM_MAX_RETRIES` | `2` | Maximum retry attempts (bounded 0–5) |

### Structured Output & Pydantic Validation

Structured output generation in `complete_json()` follows a 6-step lifecycle:
1. **Schema Extraction:** Generates standard JSON Schema from `schema.model_json_schema()`.
2. **Prompt Injection:** Appends a strict schema instruction to the conversation system prompt.
3. **JSON Mode Activation:** Requests `response_format={"type": "json_object"}` from Groq.
4. **Code Block Stripping:** Safely extracts payload if wrapped in markdown code blocks (` ```json ... ``` `).
5. **Python Validation:** Parses JSON and executes `schema.model_validate(data)` in Python (never trusts LLM business logic).
6. **Bounded Correction Retries:** If JSON is malformed or validation fails, executes up to 2 correction attempts supplying the LLM with specific error context. If errors persist, raises `LLMSchemaValidationError`.

### Timeout & Retry Policy

- **Retry Definition:** `LLM_MAX_RETRIES` specifies retries *after* the initial attempt (total attempts = `1 + max_retries`, capped between 0 and 5).
- **Exponential Backoff:** Delays calculate as `min(0.5 * (2.0 ** attempt), 10.0)` seconds (0.5s, 1.0s, 2.0s...).
- **Rate-Limit Handling:** When HTTP 429 includes a `Retry-After` header, the client waits for the specified duration (capped at 30s).
- **Retryable Errors:** HTTP 429, 500, 502, 503, 504, `httpx.TimeoutException`, `httpx.NetworkError`.
- **Non-Retryable Errors:** HTTP 401 (`LLMAuthenticationError`), HTTP 403 (`LLMPermissionError`), configuration errors (`LLMConfigurationError`).
- **Injectable Sleeper:** The client accepts `sleep_func` (defaulting to `asyncio.sleep`), allowing tests to inject mock sleepers for instantaneous 0ms test runs.

### Deterministic Offline `FakeLLMClient`

`FakeLLMClient` satisfies the `LLMClient` protocol without opening network sockets:
- Return pre-configured canned text queues (`text_responses`).
- Return pre-configured canned Pydantic models or dictionaries (`json_responses`).
- Simulate malformed JSON decoding failures (`simulate_invalid_json=True`).
- Simulate Pydantic schema validation failures (`simulate_schema_error=True`).
- Simulate provider errors and timeouts (`error_to_raise=LLMTimeoutError(...)`).
- Record call history for assertions (`calls`, `call_count`, `last_messages`).

---

## 11. Running the Application

Ensure local MongoDB is running, then start Uvicorn from the `backend` directory:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Interactive API Documentation
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Useful Example URLs
- [http://127.0.0.1:8000/api/v1/products](http://127.0.0.1:8000/api/v1/products)
- [http://127.0.0.1:8000/api/v1/products?category=laptop&page=1&page_size=10](http://127.0.0.1:8000/api/v1/products?category=laptop&page=1&page_size=10)
- [http://127.0.0.1:8000/api/v1/products/LAP-0001](http://127.0.0.1:8000/api/v1/products/LAP-0001)
- [http://127.0.0.1:8000/api/v1/products/categories](http://127.0.0.1:8000/api/v1/products/categories)
- [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## 12. Running Automated Tests

Run the full automated test suite with pytest from the `backend` directory:

```powershell
python -m pytest -v
```

All 108 unit and integration tests run cleanly offline without network calls or token consumption:
- **`test_scaffold.py` (19 tests):** Module 1 settings validation, CORS, error formatting, lifespan startup/shutdown, secret masking.
- **`test_product_model.py` (14 tests):** Product schema validation, price integer constraints, rating boundaries, category specification enforcement, contradictory spec rejection.
- **`test_product_repository.py` (14 tests):** Async MongoDB queries, pagination, deterministic sorting, category/price/use-case filtering, category summary aggregation.
- **`test_seed.py` (6 tests):** Catalogue counts (40/30/20/15/15), edge case verification, bulk upsert idempotency, document modification, and index verification.
- **`test_api_products.py` (14 tests):** Product listing, category/price/use-case/brand/stock filtering, pagination boundaries, sorting, 404 handling, category aggregation.
- **`test_api_health.py` (14 tests):** Independent DB/Groq states, simulated auth failures, network timeouts, placeholder key skipping, and credential suppression.
- **`test_llm_client.py` (27 tests):** Module 4 text generation, Pydantic JSON validation, markdown stripping, malformed JSON recovery, schema validation recovery, empty choices handling, HTTP 401/403 failure, HTTP 429/5xx exponential backoff, Retry-After header parsing, connection errors, timeouts, retry exhaustion, config validation, credential leak prevention, and FakeLLMClient.

### Optional Manual Live Groq Check

> [!NOTE]
> Automated tests mock all network calls. To verify your real Groq Cloud API credentials manually, execute the following script from the `backend` directory:

```powershell
python -c "import asyncio; from pydantic import BaseModel; from app.llm.client import GroqClient; class Demo(BaseModel): status: str; async def main(): c = GroqClient(); print('Ping:', await c.ping()); print('Text:', await c.complete_text([{'role': 'user', 'content': 'Hello'}], max_tokens=100)); print('JSON:', (await c.complete_json([{'role': 'user', 'content': 'output status=ok'}], schema=Demo)).status); await c.close(); asyncio.run(main())"
```

---

## 13. Local MongoDB Troubleshooting

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

## 14. Security and Git Hygiene

- **Root `.env` is Git-Ignored:** Sensitive keys, connection strings, and local configuration must never be committed. Verify with `git status --ignored`.
- **`.env.example` is Trackable:** Kept up to date as a configuration contract for all team members.
- **Log Masking:** Passwords in MongoDB URIs and sensitive authorization headers are automatically scrubbed by `SensitiveDataFilter`.
- **No Cloud LLM Calls on Startup:** Placeholder API keys will never trigger outbound requests during application boot.
- **No Secret Exposure in Errors:** Exception messages, error logs, and repr representations are scrubbed via `sanitize_error_message()`.

---

## 15. Roadmap & Upcoming Modules

- [x] **Module 1:** Project Scaffold, Async MongoDB Lifespan & Configuration
- [x] **Module 2:** Electronics Product Catalogue & MongoDB Repository Layer
- [x] **Module 3:** Product and Health API Endpoints (`/api/v1/products`, `/api/v1/health`)
- [x] **Module 4:** Groq LLM Client & Structured Output Infrastructure
- [ ] **Module 5:** Deterministic Scoring, Filtering, & Recommendation Agent
- [ ] **Module 6:** React + Vite Frontend & Conversational Chat Interface

