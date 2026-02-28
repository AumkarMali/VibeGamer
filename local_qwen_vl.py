"""
Local inference with Qwen3-VL-8B-Instruct (Hugging Face).
Free, no API key. Requires ~16GB+ RAM or GPU.
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""
import os
import tempfile
import threading

_model = None
_processor = None
_load_lock = threading.Lock()
_LOADING = False


def load_model():
    """Load model and processor once. Call from background thread on first use."""
    global _model, _processor, _LOADING
    with _load_lock:
        if _model is not None and _processor is not None:
            return True
        if _LOADING:
            return False
        _LOADING = True
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch
        with _load_lock:
            if _model is not None:
                _LOADING = False
                return True
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        with _load_lock:
            _model = model
            _processor = processor
            _LOADING = False
        return True
    except Exception:
        with _load_lock:
            _LOADING = False
        raise


def call_local_qwen(system: str, user_text: str, pil_image, conversation_messages: list = None, max_tokens: int = 4096) -> str:
    """
    Run inference with Qwen3-VL-8B-Instruct. Returns response text.
    conversation_messages: list of {"role": "user"|"assistant", "content": str} for prior turns (text only).
    """
    if _model is None or _processor is None:
        load_model()
    import torch
    # Build messages: system + prior turns + current user (image + text)
    messages = [{"role": "system", "content": system}]
    if conversation_messages:
        for m in conversation_messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                content = "\n".join(text_parts) if text_parts else str(content)
            if role and content:
                messages.append({"role": role, "content": content})
    # Current user message with image and text (processor often expects URL/path; save PIL to temp file)
    tmp_path = None
    try:
        if hasattr(pil_image, "save"):
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil_image.save(tmp_path)
            image_input = tmp_path
        else:
            image_input = pil_image
        content = [
            {"type": "image", "image": image_input},
            {"type": "text", "text": user_text},
        ]
        messages.append({"role": "user", "content": content})
        # Apply chat template and generate
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(inputs, dict):
            raise TypeError("apply_chat_template did not return a dict")
        # Move to model device; remove token_type_ids if present (not used by generate)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        dev = next(_model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=_processor.tokenizer.pad_token_id or _processor.tokenizer.eos_token_id,
        )
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated[0][input_len:]
        output_text = _processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text.strip()
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
