from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pdfplumber
import anthropic
import pathlib
import httpx
import json
import io
import os

load_dotenv()

BASE_DIR = pathlib.Path(__file__).parent.parent

app = FastAPI(title="Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


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


def build_resume_optimization_prompt(resume_text: str) -> str:
    """
    Prompt para o Modo 2 - aba Currículo.
    Diferente do build_prompt(), aqui não há vaga — a análise é sobre
    qualidade geral do currículo: clareza, formatação, impacto, palavras-chave.
    """
    return f"""Você é um especialista em RH e carreira, com foco em otimização de currículos para o mercado brasileiro.

Analise o currículo abaixo e responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem backticks.

CURRÍCULO:
{resume_text}

Avalie clareza, impacto das descrições (uso de verbos de ação e resultados quantificáveis), formatação, e presença de palavras-chave relevantes para a área do candidato.

Retorne exatamente este JSON:
{{
  "score": <número de 0 a 100, qualidade geral do currículo>,
  "nivel": "<Precisa melhorar | Razoável | Bom | Excelente>",
  "pontos_fortes": ["<ponto forte 1>", "<ponto forte 2>", "<ponto forte 3>"],
  "pontos_fracos": ["<ponto fraco 1>", "<ponto fraco 2>", "<ponto fraco 3>"],
  "sugestoes": ["<sugestão concreta e específica 1>", "<sugestão concreta e específica 2>", "<sugestão concreta e específica 3>", "<sugestão concreta e específica 4>"],
  "resumo": "<parágrafo curto com avaliação geral do currículo>"
}}"""


def build_linkedin_optimization_prompt(profile_text: str) -> str:
    """
    Prompt para o Modo 2 - aba LinkedIn.
    Foco em headline, seção sobre, experiências e como o perfil
    se posiciona para recrutadores e para o algoritmo de busca do LinkedIn.
    """
    return f"""Você é um especialista em personal branding e otimização de perfis do LinkedIn para o mercado brasileiro.

Analise o perfil do LinkedIn abaixo e responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem backticks.

PERFIL DO LINKEDIN:
{profile_text}

Avalie a headline (título), a seção "Sobre", a descrição das experiências profissionais, e o uso de palavras-chave que ajudam recrutadores a encontrar o perfil.

Retorne exatamente este JSON:
{{
  "score": <número de 0 a 100, qualidade geral do perfil>,
  "nivel": "<Precisa melhorar | Razoável | Bom | Excelente>",
  "pontos_fortes": ["<ponto forte 1>", "<ponto forte 2>", "<ponto forte 3>"],
  "pontos_fracos": ["<ponto fraco 1>", "<ponto fraco 2>", "<ponto fraco 3>"],
  "sugestoes": ["<sugestão concreta e específica 1>", "<sugestão concreta e específica 2>", "<sugestão concreta e específica 3>", "<sugestão concreta e específica 4>"],
  "resumo": "<parágrafo curto com avaliação geral do perfil>"
}}"""


def analyze_generic_with_claude(prompt: str, max_tokens: int = 2000) -> dict:
    """
    Versão genérica de analyze_with_claude() que aceita um prompt já montado.
    Reutilizada pelos endpoints de otimização de currículo, LinkedIn e
    geração de perfil completo.

    max_tokens é configurável porque o gerador de perfil LinkedIn produz
    um JSON bem maior (headline + sobre + experiências + habilidades +
    dicas) do que os outros modos, que só retornam feedback resumido.
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Resposta veio cortada (provavelmente atingiu o limite de tokens).
        # Erro mais claro para o usuário em vez de um JSONDecodeError cru.
        raise HTTPException(
            status_code=502,
            detail="A IA gerou uma resposta incompleta. Tente novamente — geralmente funciona na segunda tentativa."
        )


@app.post("/optimize-resume")
async def optimize_resume(file: UploadFile = File(...)):
    """
    Modo 2 - aba Currículo.
    Recebe só o PDF (sem vaga) e devolve feedback geral de qualidade.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text:
        raise HTTPException(status_code=422, detail="Não foi possível extrair texto do PDF. Verifique se não é um PDF escaneado.")

    prompt = build_resume_optimization_prompt(resume_text)
    return analyze_generic_with_claude(prompt)


@app.post("/optimize-linkedin")
async def optimize_linkedin(
    profile_text: str = Form(default=""),
    profile_url: str = Form(default="")
):
    """
    Modo 2 - aba LinkedIn.
    Aceita texto colado manualmente OU link do perfil.
    LinkedIn bloqueia scraping na maioria dos casos — por isso o texto manual
    é a via mais confiável, mas tentamos o link também.
    """
    if profile_url.strip():
        final_text = await extract_text_from_url(profile_url.strip())
    elif profile_text.strip():
        final_text = profile_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Cole o texto do perfil ou informe o link.")

    prompt = build_linkedin_optimization_prompt(final_text)
    return analyze_generic_with_claude(prompt)


def build_linkedin_generator_prompt(resume_text: str, current_profile_text: str = "") -> str:
    """
    Prompt para o gerador de perfil LinkedIn completo.
    Diferente do build_linkedin_optimization_prompt() (que só dá feedback),
    este prompt instrui a IA a ESCREVER o conteúdo pronto para copiar.

    Se current_profile_text vier preenchido, a IA usa o perfil atual como
    base e melhora; se vier vazio, a IA cria o perfil do zero a partir
    apenas do currículo.
    """
    contexto_perfil_atual = ""
    if current_profile_text.strip():
        contexto_perfil_atual = f"""
PERFIL ATUAL DO LINKEDIN (use como base, mas reescreva e melhore):
{current_profile_text}
"""

    return f"""Você é um especialista em personal branding e redação para LinkedIn, focado no mercado brasileiro.

Sua tarefa é ESCREVER o conteúdo completo e pronto para a pessoa copiar e colar diretamente no LinkedIn. Não dê apenas dicas — escreva o texto final, profissional e persuasivo.

CURRÍCULO (fonte de informações sobre experiências, formação e habilidades):
{resume_text}
{contexto_perfil_atual}

Responda APENAS com um JSON válido, sem texto adicional, sem markdown, sem backticks.

Regras importantes:
- A headline deve ter no máximo 220 caracteres, indo direto ao ponto, com cargo + especialidade + valor que entrega.
- O "sobre" deve ter 3-4 parágrafos curtos, em primeira pessoa, com tom humano e profissional (não robótico), contando a trajetória e o diferencial do candidato.
- Cada experiência deve ter de 2 a 4 bullets reescritos com verbos de ação e resultados (use números/impacto sempre que o currículo permitir inferir).
- As habilidades devem ser termos exatos de busca usados por recrutadores na área do candidato.

Retorne exatamente este JSON:
{{
  "headline": "<headline pronta para colar, até 220 caracteres>",
  "sobre": "<texto completo da seção Sobre, com parágrafos separados por \\n\\n>",
  "experiencias": [
    {{
      "cargo_empresa": "<nome do cargo na empresa, conforme currículo>",
      "bullets": ["<bullet reescrito 1>", "<bullet reescrito 2>", "<bullet reescrito 3>"]
    }}
  ],
  "habilidades": ["<habilidade 1>", "<habilidade 2>", "<habilidade 3>", "<habilidade 4>", "<habilidade 5>", "<habilidade 6>", "<habilidade 7>", "<habilidade 8>"],
  "dicas_extras": ["<dica sobre foto de perfil>", "<dica sobre banner>", "<dica sobre featured/destaques>", "<dica de frequência de postagem ou interação>"]
}}"""


@app.post("/build-linkedin-profile")
async def build_linkedin_profile(
    file: UploadFile = File(...),
    current_profile_text: str = Form(default="")
):
    """
    Modo 3 - Criar perfil LinkedIn completo.
    Sempre exige o currículo em PDF (fonte de informações confiáveis).
    current_profile_text é opcional: se vier preenchido, a IA usa como
    base para melhorar em vez de criar do zero.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)

    if not resume_text:
        raise HTTPException(status_code=422, detail="Não foi possível extrair texto do PDF. Verifique se não é um PDF escaneado.")

    prompt = build_linkedin_generator_prompt(resume_text, current_profile_text.strip())
    return analyze_generic_with_claude(prompt, max_tokens=3000)


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
