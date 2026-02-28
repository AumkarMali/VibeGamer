"""
Vision chat via Ollama (qwen3-vl:30b or other vision model).
Run Ollama locally first: ollama pull qwen3-vl:30b
Then use this module for fast local inference with no API key.
"""
import tempfile
import os

# Default vision model; user can run: ollama pull qwen3-vl:30b
OLLAMA_VISION_MODEL = "qwen3-vl:30b"


def call_ollama(
    system: str,
    user_text: str,
    pil_image,
    conversation_messages: list = None,
    max_tokens: int = 4096,
    model: str = None,
) -> str:
    """
    Send system + user text + image to Ollama. Returns assistant text.
    conversation_messages: list of {"role": "user"|"assistant", "content": str} for prior turns (text only).
    """
    from ollama import chat

    model = model or OLLAMA_VISION_MODEL
    path = None
    try:
        if hasattr(pil_image, "save"):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil_image.save(path)
            image_input = path
        else:
            image_input = pil_image

        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        if conversation_messages:
            for m in conversation_messages:
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                    content = "\n".join(text_parts) if text_parts else str(content)
                if role and content is not None:
                    messages.append({"role": role, "content": str(content)})
        messages.append({
            "role": "user",
            "content": user_text,
            "images": [image_input],
        })

        response = chat(model=model, messages=messages)
        out = (response.message.content or "").strip()
        return out
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def is_ollama_ready(model: str = None) -> bool:
    """Return True if Ollama is running and the vision model is available."""
    try:
        from ollama import list as ollama_list
        model = model or OLLAMA_VISION_MODEL
        out = ollama_list()
        # API can return {"models": [...]} or a list
        models = out.get("models", out) if isinstance(out, dict) else out
        if not isinstance(models, list):
            models = []
        names = [m.get("name", "") if isinstance(m, dict) else str(m) for m in models]
        return any(model in n or n in model for n in names)
    except Exception:
        return False
