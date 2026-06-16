from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pdfplumber
import anthropic
import httpx
import json
import io
import os

load_dotenv()

app = FastAPI(title="Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extrai texto puro do PDF página por página.
    pdfplumber abre em memória — nenhum arquivo é salvo no disco.
    """
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


async def extract_text_from_url(url: str) -> str:
    """
    Faz o scraping do link da vaga e extrai o texto relevante.
    Usamos httpx (cliente HTTP assíncrono) + BeautifulSoup para parsear o HTML.
    Removemos scripts, estilos e menus para ficar só com o conteúdo da vaga.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=422, detail=f"Não foi possível acessar o link da vaga (status {response.status_code}).")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove tags que não têm conteúdo útil
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Limpa linhas em branco excessivas
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if not text:
        raise HTTPException(status_code=422, detail="Não foi possível extrair texto do link informado.")

    # Limita a 8000 caracteres para não estourar o prompt
    return text[:8000]


def build_prompt(resume_text: str, job_description: str) -> str:
    return f"""Você é um especialista em recrutamento e análise de currículos.

Analise o currículo abaixo em relação à vaga informada e responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem backticks.

CURRÍCULO:
{resume_text}

DESCRIÇÃO DA VAGA:
{job_description}

Retorne exatamente este JSON:
{{
  "score": <número de 0 a 100>,
  "nivel_compatibilidade": "<Baixo | Médio | Alto | Excelente>",
  "pontos_fortes": [
    "<ponto forte 1>",
    "<ponto forte 2>",
    "<ponto forte 3>"
  ],
  "lacunas": [
    "<lacuna 1>",
    "<lacuna 2>",
    "<lacuna 3>"
  ],
  "sugestoes": [
    "<sugestão concreta 1>",
    "<sugestão concreta 2>",
    "<sugestão concreta 3>"
  ],
  "resumo": "<parágrafo curto com avaliação geral do candidato para essa vaga>"
}}"""


def analyze_with_claude(resume_text: str, job_description: str) -> dict:
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": build_prompt(resume_text, job_description)
            }
        ]
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


@app.post("/fetch-job")
async def fetch_job(url: str = Form(...)):
    """
    Endpoint separado que o frontend chama ao colar um link.
    Retorna o texto extraído para o usuário revisar antes de analisar.
    """
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Informe uma URL válida começando com http.")

    text = await extract_text_from_url(url)
    return {"text": text}


@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    job_url: str = Form(default="")
):
    """
    Endpoint principal.
    Se job_url for informado, faz scraping do link e usa como descrição da vaga.
    Se não, usa o job_description digitado manualmente.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    # Resolve descrição da vaga: link tem prioridade sobre texto manual
    if job_url.strip():
        final_job_description = await extract_text_from_url(job_url.strip())
    elif job_description.strip():
        final_job_description = job_description.strip()
    else:
        raise HTTPException(status_code=400, detail="Informe o link ou o texto da vaga.")

    file_bytes = await file.read()

    resume_text = extract_text_from_pdf(file_bytes)
    if not resume_text:
        raise HTTPException(status_code=422, detail="Não foi possível extrair texto do PDF. Verifique se não é um PDF escaneado.")

    result = analyze_with_claude(resume_text, final_job_description)
    return result
