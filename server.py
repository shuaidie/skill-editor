"""
Hermes Skill Editor - FastAPI backend
Serves skill list (with Chinese translation on demand), file CRUD, and LLM chat.
"""

import json
import os
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "~/.hermes/skills")).expanduser()
LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1")
TRANSLATION_CACHE = Path(__file__).parent / "translations.json"

SYSTEM_PROMPT = (
    "你是一个技能编辑器助手。帮助用户改进 Hermes Agent 的 SKILL.md 文件。\n\n"
    "规则：\n"
    "1. 当需要修改内容时，输出完整的更新后 SKILL.md（包括 YAML frontmatter）。\n"
    "2. 用 ```markdown ... ``` 代码块包裹文件内容。\n"
    "3. 在代码块之前说明你做了什么修改及原因。\n"
    "4. 保持 YAML frontmatter 有效（name, description 字段）。\n"
    "5. 除非用户要求，否则保持现有结构不变。\n"
    "6. 使用用户相同的语言回复。"
)

app = FastAPI(title="Skill Editor")

# Allow all origins — user accesses from Tailscale IPs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    skill_path: Optional[str] = None

class SaveRequest(BaseModel):
    path: str
    content: str

class NewSkillRequest(BaseModel):
    name: str
    category: str
    description: str
    content: Optional[str] = ""

class TranslateRequest(BaseModel):
    full_path: str
    text: str

class DeleteRequest(BaseModel):
    full_path: str


# ── Translation cache ────────────────────────────────────

def load_translations() -> dict:
    if TRANSLATION_CACHE.exists():
        try:
            return json.loads(TRANSLATION_CACHE.read_text())
        except Exception:
            return {}
    return {}

def save_translations(cache: dict):
    TRANSLATION_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


async def translate_to_zh(text: str) -> str:
    """Translate a short description to Chinese via llama-server.
    Uses /completions endpoint (not /chat/completions) because Qwen3 model
    returns empty content in chat mode, while completions works reliably."""
    prompt = f"你是一个翻译。把以下英文翻译成简洁自然的中文。只返回中文翻译，不要任何其他内容：\n\n{text}"
    payload = {
        "prompt": prompt,
        "temperature": 0.3,
        "max_tokens": 200,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{LLAMA_URL}/completions", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                result = data["choices"][0]["text"].strip()
                # Strip all Qwen3 thinking tags (reasoning, think, etc.)
                result = result.split("</think>")[-1].strip()
                result = result.split("<reasoning>")[-1].split("</reasoning>")[-1].strip()
                result = result.split("</think>")[-1].strip()
                result = result.split("<answer>")[-1].split("</answer>")[-1].strip()
                # Remove any remaining XML-like tags
                import re
                result = re.sub(r'<[^>]+>', '', result).strip()
                # Validate: a valid translation should be Chinese-heavy or match
                # basic expectations (not contain English thinking patterns)
                if result and re.search(r'[\u4e00-\u9fff]', result):
                    # Must not be suspiciously short or contain thinking artifacts
                    if len(result) > 5 and "Here's a" not in result and "thought" not in result.lower():
                        return result
    except Exception:
        pass
    return text


# ── Skill discovery ──────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = {}
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip()
            return fm, parts[2].strip()
    return {}, text.strip()


def skill_rel_path(fp: Path) -> str:
    return str(fp.parent.relative_to(SKILLS_DIR))


def get_categories() -> list[str]:
    cats = set()
    if not SKILLS_DIR.exists():
        return sorted(cats)
    for fp in SKILLS_DIR.rglob("SKILL.md"):
        parts = skill_rel_path(fp).split("/")
        if len(parts) > 1:
            cats.add(parts[0])
    return sorted(cats)


def discover_skills() -> list[dict]:
    skills = []
    if not SKILLS_DIR.exists():
        return skills
    for fp in sorted(SKILLS_DIR.rglob("SKILL.md")):
        rel = skill_rel_path(fp)
        name = fp.parent.name
        parts = rel.split("/")
        category = parts[0] if len(parts) > 1 else "root"
        with open(fp) as f:
            text = f.read()
        fm, body = parse_frontmatter(text)
        desc = fm.get("description", body[:120]).strip()
        skills.append({
            "name": name,
            "category": category,
            "rel_path": rel,
            "full_path": str(fp),
            "description": desc,
            "frontmatter": fm,
        })
    return skills


# ── Routes ───────────────────────────────────────────────

@app.get("/api/skills")
def list_skills():
    """Return all skills. Chinese translations are loaded from cache (fast)."""
    skills = discover_skills()
    cache = load_translations()
    for s in skills:
        key = s["full_path"]
        s["description_zh"] = cache.get(key, None)  # None = not yet translated
    return JSONResponse(
        content=skills,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/categories")
def list_categories():
    return get_categories()


@app.get("/api/skill/{rel_path:path}")
def get_skill(rel_path: str):
    full = SKILLS_DIR / rel_path / "SKILL.md"
    if not full.exists():
        raise HTTPException(404, "Skill not found")
    with open(full) as f:
        return {"full_path": str(full), "content": f.read()}


@app.post("/api/skill/save")
def save_skill(req: SaveRequest):
    p = Path(req.path)
    if not str(p).startswith(str(SKILLS_DIR)):
        raise HTTPException(403, "Path outside skills directory")
    if not p.exists():
        raise HTTPException(404, "File not found")
    with open(p, "w") as f:
        f.write(req.content)
    return {"ok": True, "path": str(p)}


@app.post("/api/skill/new")
def new_skill(req: NewSkillRequest):
    if not req.name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "Invalid skill name (use lowercase letters, numbers, hyphens)")
    
    if req.category and req.category != "root":
        dir_path = SKILLS_DIR / req.category / req.name
    else:
        dir_path = SKILLS_DIR / req.name
    
    if dir_path.exists():
        raise HTTPException(409, f"Skill '{req.name}' already exists in '{req.category}'")
    
    dir_path.mkdir(parents=True, exist_ok=True)
    skill_file = dir_path / "SKILL.md"
    desc = req.description or ""
    if req.content:
        content = req.content
    else:
        content = (
            f"---\nname: {req.name}\ndescription: {desc}\n---\n"
            f"# {req.name}\n\n{desc}\n\n"
            f"## Overview\n\nTODO: Describe what this skill does.\n\n"
            f"## Workflow\n\n1. Step 1\n2. Step 2\n\n"
            f"## Pitfalls\n\n- Note any gotchas here\n"
        )
    skill_file.write_text(content)
    cache = load_translations()
    cache.pop(str(skill_file), None)
    save_translations(cache)
    return {
        "ok": True,
        "full_path": str(skill_file),
        "rel_path": str(dir_path.relative_to(SKILLS_DIR)),
    }


@app.post("/api/skill/delete")
def delete_skill(req: DeleteRequest):
    p = Path(req.full_path)
    if not str(p).startswith(str(SKILLS_DIR)):
        raise HTTPException(403, "Path outside skills directory")
    skill_dir = p.parent
    if not skill_dir.exists():
        raise HTTPException(404, "Skill directory not found")
    import shutil
    shutil.rmtree(skill_dir)
    # Clean translation cache
    cache = load_translations()
    for key in list(cache.keys()):
        if key.startswith(str(skill_dir)):
            cache.pop(key, None)
    save_translations(cache)
    return {"ok": True, "deleted": str(skill_dir)}


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    """Translate a skill description to Chinese. Caches the result."""
    cache = load_translations()
    if req.full_path in cache:
        return {"translated": cache[req.full_path]}
    
    translated = await translate_to_zh(req.text)
    cache[req.full_path] = translated
    save_translations(cache)
    return {"translated": translated}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    system_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    skill_context = ""
    if req.skill_path:
        p = Path(req.skill_path)
        if p.exists():
            with open(p) as f:
                content = f.read()
            skill_context = (
                f"\n\n--- CURRENT SKILL CONTENT ({p.name}) ---\n"
                f"{content}\n"
                f"--- END SKILL CONTENT ---\n"
            )
    
    user_msgs = [dict(m) for m in req.messages]
    if skill_context and user_msgs:
        user_msgs[-1]["content"] = user_msgs[-1]["content"] + skill_context
    
    payload = {
        "messages": system_msgs + user_msgs,
        "temperature": 0.3,
        "max_tokens": 8192,
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{LLAMA_URL}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise HTTPException(502, f"llama-server error: {resp.text[:300]}")
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        # Qwen3 puts answer in reasoning_content, not content
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
        }


@app.get("/")
def index():
    html_path = str(Path(__file__).parent / "index.html")
    # Read HTML and add cache-busting timestamp to force browser refresh
    content = Path(html_path).read_text()
    # Add version query to any relative resources
    from datetime import datetime
    ver = datetime.now().timestamp()
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
