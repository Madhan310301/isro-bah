# Bharatiya-Antariksh-Code-Nest-7.o
Team Nest 7.0 — a squad of builders driven by curiosity and execution. We turn ideas into working prototypes fast, blending diverse skills in AI/ML, web, and IoT to tackle real-world problems with creative, scalable solutions. Ready to innovate, collaborate, and bring something impactful to Bharatiya Antariksh Hackathon 2026.

## SatBridge — Cross-Modal Satellite Image Retrieval

ISRO Bharatiya Antariksha Hackathon 2026 — Challenge 11 (BAH2026).

Cross-modal satellite image retrieval system combining a **Node.js/TypeScript API server** with a **Python/PyTorch ML backend**.

## Stack

| Layer | Tech |
|---|---|
| Monorepo | pnpm workspaces |
| API Server | Express 5, TypeScript, Drizzle ORM |
| Database | PostgreSQL |
| Frontend Sandbox | Vite + React + Tailwind CSS v4 + shadcn/ui |
| ML Backend | Python 3.11, PyTorch, FastAPI, FAISS |
| Auth | JWT (optional) |
| Validation | Zod, drizzle-zod |
| API Codegen | Zod schemas for API validation |
| API Codegen | Orval (from OpenAPI spec) |

## Prerequisites

- **Node.js** ≥ 24
- **pnpm** ≥ 9
- **Python** ≥ 3.11 (for the ML backend)
- **PostgreSQL** (local or remote)

## Getting Started

### 1. Clone & install

```bash
git clone <your-repo-url>
cd SatBridge

# Install Node.js dependencies
pnpm install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual values (DATABASE_URL is required)
```

### 3. Push DB schema (first time only)

```bash
pnpm --filter @workspace/db run push
```

### 4. Run the API server

```bash
pnpm --filter @workspace/api-server run dev
# → Runs on http://localhost:5000
```

### 5. Run the mockup sandbox (frontend)

```bash
PORT=8081 BASE_PATH=/ pnpm --filter @workspace/mockup-sandbox run dev
# → Runs on http://localhost:8081
```

### 6. Set up the ML backend (optional)

```bash
cd bah2026-challenge11
python -m venv venv
venv\Scripts\activate      # macOS/Linux: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the ML API server
uvicorn app.main:app --reload --port 8000
```

## Workspace Structure

```
SatBridge/
├── artifacts/
│   ├── api-server/          # Express API server
│   └── mockup-sandbox/      # Vite + React frontend sandbox
├── lib/
│   ├── api-client-react/    # Generated React Query hooks
│   ├── api-spec/            # OpenAPI spec + Orval codegen
│   ├── api-zod/             # Zod schemas for API validation
│   └── db/                  # Drizzle ORM schema + migrations
├── bah2026-challenge11/     # Python ML backend (FastAPI + PyTorch)
├── scripts/                 # Build/CI helper scripts
├── package.json             # Root workspace config
└── pnpm-workspace.yaml      # pnpm workspace + catalog definitions
```

## Useful Commands

```bash
# Full typecheck across all packages
pnpm run typecheck

# Typecheck + build everything
pnpm run build

# Regenerate API hooks and Zod schemas from the OpenAPI spec
pnpm --filter @workspace/api-spec run codegen

# Push DB schema changes (dev only — never in prod)
pnpm --filter @workspace/db run push
```

## License

MIT

