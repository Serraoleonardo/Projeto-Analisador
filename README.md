# 🧠 Resume Analyzer

Aplicação web que analisa a compatibilidade de currículos com vagas usando IA (Claude).

## Stack
- **Backend:** FastAPI + pdfplumber + Anthropic SDK
- **Frontend:** HTML/CSS/JS puro (sem framework)
- **IA:** Claude Haiku (rápido e barato)

## Como rodar

### 1. Clone e instale dependências
```bash
pip install -r requirements.txt
```

### 2. Configure a chave da API
```bash
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

### 3. Suba o servidor
```bash
cd backend
uvicorn main:app --reload
```

### 4. Acesse
Abra http://localhost:8000 no navegador.

## Estrutura
```
resume-analyzer/
├── backend/
│   └── main.py          # FastAPI + pipeline de análise
├── frontend/
│   └── index.html       # UI completa (HTML/CSS/JS)
├── requirements.txt
└── README.md
```

## Como funciona

1. **Upload do PDF** → `pdfplumber` extrai o texto página por página
2. **Montagem do prompt** → combina texto do currículo + descrição da vaga
3. **Chamada à API** → Claude analisa e retorna JSON estruturado
4. **Renderização** → frontend exibe score animado + cards de resultado

## Deploy (Railway)

```bash
# Procfile
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```
