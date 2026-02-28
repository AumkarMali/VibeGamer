"""
AI Agent - Unified GUI.

Task input + screenshot → Qwen VL decides:
  - Chess task + board visible → start chess bot (YOLO + Stockfish)
  - Feedback/correction → store for learning
  - Other tasks → pywinauto + keyboard screen control (click, type, key press)

Uses Qwen VL (DashScope) for vision/routing.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import base64
import io
import json
import re
import pyautogui
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from chess_agent import ChessEngine, is_chess_task

try:
    from pywinauto_actions import get_ui_elements, execute_action as _execute_pywinauto
    _PYWINAUTO_OK = True
except ImportError:
    _PYWINAUTO_OK = False

# pynput for keyboard (sending keys to focused window)
try:
    from pynput.keyboard import Controller as PynputController, Key as PynputKey
    _PYNPUT_AVAILABLE = True
except ImportError:
    PynputController = PynputKey = None
    _PYNPUT_AVAILABLE = False

# ── config (persisted API key) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def _load_config():
    """Load config.json. Returns dict with api key, model, use_local_model."""
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_config(api_key: str, model: str, use_local: bool = True):
    """Save API key, model, and use_local_model to config.json."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "claude_api_key": api_key,
                "claude_model": model,
                "use_local_model": use_local,
            }, f, indent=2)
    except OSError:
        pass


# ── prompt history (learning from mistakes) ──
PROMPT_HISTORY_PATH = os.path.join(BASE_DIR, "prompt_history.json")
MAX_HISTORY_ENTRIES = 100


def _load_prompt_history():
    """Load prompt history for learning. Returns list of entries."""
    try:
        if os.path.isfile(PROMPT_HISTORY_PATH):
            with open(PROMPT_HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_prompt_entry(entry: dict):
    """Append one entry to prompt history. Trims to MAX_HISTORY_ENTRIES."""
    history = _load_prompt_history()
    history.append(entry)
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]
    try:
        with open(PROMPT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


def _parse_claude_json(text: str):
    """Parse JSON from Claude. Tolerates leading/trailing text and single quotes. Returns dict or None."""
    text = text.strip()
    # If Claude wrote reasoning before the JSON, find the JSON object (prefer last { so we get the real payload)
    start_candidates = [i for i, c in enumerate(text) if c == "{"]
    if not start_candidates:
        return None
    # Try from last { first (Claude often writes reasoning then JSON)
    for start in reversed(start_candidates):
        depth = 0
        chunk = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    break
        if chunk is None:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)", r'\1"\2"\3', chunk)
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', chunk)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            continue
    # Fallback: original first-{ extraction and fixes
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                text = text[start : i + 1]
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fix common LLM output: unquoted keys (only after { or ,), trailing commas
    fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)", r'\1"\2"\3', text)
    fixed = re.sub(r",\s*}", "}", fixed)  # trailing comma before }
    fixed = re.sub(r",\s*]", "]", fixed)  # trailing comma before ]
    # Fix single-quoted keys/values (simple: replace ' with " where it looks like JSON)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Last resort: try replacing ' with " for keys (key':  -> "key":)
    fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', text)
    fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _build_learning_context(max_entries: int = 15):
    """Build a string of past user feedback for Claude to learn from."""
    history = _load_prompt_history()
    # Use feedback entries (Claude-detected corrections)
    feedback_entries = [e for e in history if e.get("type") == "feedback"][-max_entries:]
    if not feedback_entries:
        return ""
    lines = [
        "",
        "LEARNING FROM USER FEEDBACK (the user corrected you—apply these lessons):",
        "- Use CLICK or DOUBLE_CLICK with element names from the UI list. Use KEYS for shortcuts (e.g. win), TYPE_TEXT to type. One message = one command.",
    ]
    for e in feedback_entries:
        user_msg = (e.get("user_message") or e.get("task") or "?")[:50]
        fb = e.get("feedback", {})
        if isinstance(fb, dict):
            parts = []
            for k, v in fb.items():
                if v:
                    parts.append(f"{k}: {str(v)[:60]}")
            fb_str = "; ".join(parts) if parts else json.dumps(fb)[:100]
        else:
            fb_str = str(fb)[:100]
        lines.append(f"- User said: \"{user_msg}\" → Feedback stored: {fb_str}")
    return "\n".join(lines)


# ── action executor (mouse + keyboard) ──
def _pynput_key(name):
    """Map key name to pynput Key or single-char string. Returns None if unknown."""
    if not _PYNPUT_AVAILABLE or PynputKey is None:
        return None
    n = str(name).lower()
    m = {
        "win": PynputKey.cmd, "winleft": PynputKey.cmd, "winright": PynputKey.cmd,
        "ctrl": PynputKey.ctrl, "control": PynputKey.ctrl,
        "alt": PynputKey.alt, "shift": PynputKey.shift,
        "enter": PynputKey.enter, "return": PynputKey.enter,
        "tab": PynputKey.tab, "space": PynputKey.space,
        "backspace": PynputKey.backspace, "esc": PynputKey.esc, "escape": PynputKey.esc,
    }
    for i in range(1, 13):
        m[f"f{i}"] = getattr(PynputKey, f"f{i}", None)
    m = {k: v for k, v in m.items() if v is not None}
    if n in m:
        return m[n]
    if len(n) == 1:
        return n
    return None


def _execute_action(action_dict: dict, screen_w: int, screen_h: int) -> tuple:
    """Execute pywinauto actions (CLICK, DOUBLE_CLICK, TYPE_IN, MENU) and pynput (KEYS, TYPE_TEXT). Returns (success, message)."""
    try:
        action = (action_dict.get("action") or "").upper().replace(" ", "_")
        params = action_dict.get("parameters") or action_dict
        if isinstance(params, dict) and "action" in params:
            params = params.get("parameters", params)

        # pywinauto actions (element-based)
        if action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "TASK_COMPLETE") and _PYWINAUTO_OK:
            return _execute_pywinauto(action_dict)

        if action in ("KEYS", "KEY_PRESS"):
            keys = params.get("keys") or params.get("keys_list") or []
            if isinstance(keys, str):
                keys = [keys]
            if not keys:
                return False, "No keys specified"
            key_map = {"control": "ctrl", "windows": "win", "command": "win", "win": "winleft", "super": "winleft"}
            keys = [key_map.get(str(k).lower(), k) for k in keys]
            # On Windows use pynput for reliable key delivery to foreground window
            if sys.platform == "win32" and _PYNPUT_AVAILABLE:
                pynput_keys = []
                for k in keys:
                    pk = _pynput_key(k)
                    if pk is None and len(str(k)) == 1:
                        pk = str(k).lower()
                    pynput_keys.append(pk)
                if all(pk is not None for pk in pynput_keys):
                    kb = PynputController()
                    for pk in pynput_keys:
                        kb.press(pk)
                    for pk in reversed(pynput_keys):
                        kb.release(pk)
                    return True, f"Pressed {'+'.join(str(k) for k in keys)}"
            # Fallback: pyautogui
            mods = {"winleft", "winright", "ctrl", "alt", "shift"}
            if len(keys) == 1 and str(keys[0]).lower() in mods:
                pyautogui.keyDown(keys[0])
                pyautogui.keyUp(keys[0])
            elif len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            return True, f"Pressed {'+'.join(str(k) for k in keys)}"

        if action == "TYPE_TEXT":
            text = params.get("text", "")
            if sys.platform == "win32" and _PYNPUT_AVAILABLE:
                kb = PynputController()
                kb.type(text)
            else:
                pyautogui.write(text, interval=params.get("interval", 0.05))
            return True, f"Typed: {text[:50]}..."

        if action == "TASK_COMPLETE":
            return True, params.get("message", "Task complete")

        if action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU") and not _PYWINAUTO_OK:
            return False, "pywinauto not available (Windows + pip install pywinauto)"

        return False, f"Unknown action: {action}"
    except Exception as e:
        import traceback
        return False, f"{e}\n{traceback.format_exc()}"


# ── theme ──
BG      = "#1a1a2e"
BG_DARK = "#0f0f1a"
FG      = "#ffffff"
ACCENT  = "#e94560"

# Qwen VL (vision) models via DashScope - best OCR/screen understanding
QWEN_MODELS = [
    "qwen-vl-max",           # best vision/OCR
    "qwen-vl-plus",
    "qwen2-vl-72b-instruct",
    "qwen2-vl-7b-instruct",
]
DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # Singapore; or dashscope-us.aliyuncs.com for US


def _call_qwen(api_key: str, model: str, system: str, user_content: list, messages: list = None, max_tokens: int = 4096) -> str:
    """Call Qwen VL via DashScope (OpenAI-compatible). user_content = [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]. Returns response text."""
    from openai import OpenAI
    client = OpenAI(base_url=DASHSCOPE_BASE, api_key=api_key, timeout=90.0)
    if messages is None:
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = list(messages)
        messages.append({"role": "user", "content": user_content})
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    # Defensive: support both object and dict-style response (avoid "string indices must be integers")
    choices = resp.get("choices", []) if isinstance(resp, dict) else getattr(resp, "choices", None) or []
    if not choices:
        err = resp.get("error", None) if isinstance(resp, dict) else getattr(resp, "error", None)
        raise RuntimeError(f"API returned no choices: {err or resp}")
    first = choices[0]
    msg = first.get("message", first) if isinstance(first, dict) else getattr(first, "message", first)
    content = msg.get("content", None) if isinstance(msg, dict) else getattr(msg, "content", None)
    if content is None:
        raise RuntimeError("API response had no message content")
    return (content if isinstance(content, str) else str(content)).strip()


class AgentGUI:
    """Single-window AI agent with chess and screen-control modes."""

    def __init__(self, root):
        self.root = root
        root.title("AI Agent")
        root.geometry("1400x900")
        root.configure(bg=BG)

        # ── settings variables (shared with popup) ──
        cfg = _load_config()
        default_key = (cfg.get("claude_api_key") or cfg.get("api_key") or "").strip() or os.environ.get("DASHSCOPE_API_KEY", "")
        default_model = cfg.get("claude_model") or cfg.get("model") or QWEN_MODELS[0]
        if default_model not in QWEN_MODELS:
            default_model = QWEN_MODELS[0]
        self.claude_api_key_var = tk.StringVar(value=default_key)
        self.claude_model_var   = tk.StringVar(value=default_model)
        self.use_local_var = tk.BooleanVar(value=cfg.get("use_local_model", True))
        # chess
        self.turn_var        = tk.StringVar(value="white")
        self.conf_var        = tk.DoubleVar(value=0.30)
        self.depth_var       = tk.IntVar(value=18)
        self.interval_var    = tk.DoubleVar(value=3.0)
        self.click_delay_var = tk.DoubleVar(value=0.15)

        # ── state ──
        self._running = False
        self._mode = None          # "chess", "screen_control", or None
        self._thread = None
        self._screen_action_count = 0
        self._hide_event = threading.Event()
        self._show_event = threading.Event()

        # ── chess engine (headless) ──
        self.chess = ChessEngine(log_fn=self.log)

        self._apply_styles()
        self._build_ui()

        # start loading chess models in background
        self.log("Loading chess engine...", "warning")
        threading.Thread(target=self._load_chess, daemon=True).start()
        # If using local Qwen3-VL, start loading it in background so first run is faster
        if self.use_local_var.get():
            def _load_local_vl():
                try:
                    from local_qwen_vl import load_model
                    load_model()
                    self.root.after(0, lambda: self.log("Local Qwen3-VL-8B ready.", "info"))
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"Local model load failed: {e}", "error"))
            threading.Thread(target=_load_local_vl, daemon=True).start()

    # ==================================================================
    #  Styles
    # ==================================================================
    def _apply_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",           background=BG)
        s.configure("TLabel",           background=BG, foreground=FG)
        s.configure("TLabelframe",      background=BG, foreground=FG)
        s.configure("TLabelframe.Label", background=BG, foreground=FG)

    # ==================================================================
    #  UI
    # ==================================================================
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding="10")
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── left panel ──
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # title
        tk.Label(left, text="AI Agent", font=("Arial", 24, "bold"),
                 bg=BG, fg=ACCENT).grid(row=0, column=0, pady=(0, 2))
        tk.Label(left, text="Type a task and press Start",
                 font=("Arial", 10, "italic"),
                 bg=BG, fg="#9e9e9e").grid(row=1, column=0, pady=(0, 8))

        # status
        self.status_label = tk.Label(left, text="Loading...",
                                     font=("Arial", 12), bg=BG, fg="#FF9800")
        self.status_label.grid(row=2, column=0, pady=(0, 10))

        # task input
        task_frame = ttk.LabelFrame(left, text="Task", padding="6")
        task_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.task_text = tk.Text(
            task_frame, height=3, width=32, font=("Consolas", 10),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff",
            wrap=tk.WORD, relief=tk.FLAT,
            highlightthickness=1, highlightcolor="#444")
        self.task_text.pack(fill=tk.X)
        self.task_text.insert("1.0", "play chess")

        # buttons row
        btn_row = tk.Frame(left, bg=BG)
        btn_row.grid(row=4, column=0, pady=(0, 8))

        self.start_btn = tk.Button(
            btn_row, text="Start", font=("Arial", 14, "bold"),
            bg="#4CAF50", fg="#fff", activebackground="#388E3C",
            relief=tk.FLAT, padx=25, pady=10, width=10,
            command=self._toggle, cursor="hand2")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row, text="Settings", font=("Arial", 11),
            bg="#555", fg="#fff", activebackground="#666",
            relief=tk.FLAT, padx=15, pady=10,
            command=self._open_settings, cursor="hand2",
        ).pack(side=tk.LEFT)

        tk.Button(
            left, text="Clear Log", font=("Arial", 10),
            bg="#444", fg="#fff", relief=tk.FLAT, padx=15, pady=5,
            command=lambda: self.log_text.delete("1.0", tk.END),
            cursor="hand2",
        ).grid(row=5, column=0, pady=(0, 10))

        # last action display
        act_frame = ttk.LabelFrame(left, text="Last Action", padding="8")
        act_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        self.action_label = tk.Label(
            act_frame, text="--", font=("Consolas", 24, "bold"),
            bg=BG, fg="#FFD700")
        self.action_label.pack()
        self.eval_label = tk.Label(
            act_frame, text="", font=("Consolas", 11), bg=BG, fg="#aaa")
        self.eval_label.pack()

        # stats
        stats_frame = ttk.LabelFrame(left, text="Stats", padding="8")
        stats_frame.grid(row=7, column=0, sticky="ew")
        self.stats_label = tk.Label(
            stats_frame,
            text="Mode: --\nCycles: 0\nActions: 0\nStatus: idle",
            font=("Consolas", 10), bg=BG, fg="#fff", justify=tk.LEFT)
        self.stats_label.pack(anchor="w")

        # ── right panel ──
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        img_frame = ttk.LabelFrame(right, text="Screen", padding="8")
        img_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        right.rowconfigure(0, weight=2)
        right.columnconfigure(0, weight=1)
        self.screenshot_label = tk.Label(
            img_frame,
            text="No screenshot yet\n\nPosition this window so\nit doesn't cover the target.",
            bg="#2a2a3e", fg="#888", font=("Arial", 12))
        self.screenshot_label.pack(expand=True, fill=tk.BOTH)

        log_frame = ttk.LabelFrame(right, text="Log", padding="8")
        log_frame.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(1, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=12, font=("Consolas", 9),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff", wrap=tk.WORD)
        self.log_text.pack(expand=True, fill=tk.BOTH)

        for tag, color in [("info", "#4CAF50"), ("error", "#f44336"),
                           ("action", "#2196F3"), ("warning", "#FF9800"),
                           ("header", "#e94560"), ("piece", "#80CBC4"),
                           ("move", "#FFD700"), ("board", "#B0BEC5"),
                           ("dim", "#666666"), ("thought", "#80CBC4"),
                           ("result", "#FFD700")]:
            self.log_text.tag_config(tag, foreground=color)

    # ==================================================================
    #  Settings Popup
    # ==================================================================
    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("640x1020")
        win.configure(bg=BG)
        win.transient(self.root)
        win.grab_set()

        px = 10
        py = 2

        # ── LLM (routing + screen comments) ──
        tk.Label(win, text="Vision model (task routing & screen)",
                 font=("Arial", 12, "bold"), bg=BG, fg=ACCENT).pack(
            anchor="w", padx=px, pady=(12, 4))
        use_local_cb = tk.Checkbutton(
            win, text="Use local Qwen3-VL-8B (free, no API key) – ~16GB RAM or GPU",
            variable=self.use_local_var, bg=BG, fg=FG, selectcolor=BG_DARK,
            activebackground=BG, activeforeground=FG, font=("Arial", 10),
        )
        use_local_cb.pack(anchor="w", padx=px, pady=py)
        tk.Label(win, text="Or DashScope API Key (if not using local):", bg=BG, fg=FG).pack(
            anchor="w", padx=px, pady=py)
        ckey_f = tk.Frame(win, bg=BG)
        ckey_f.pack(anchor="w", fill=tk.X, padx=px, pady=py)
        self._claude_key_entry = tk.Entry(
            ckey_f, textvariable=self.claude_api_key_var, show="*",
            font=("Consolas", 9), bg=BG_DARK, fg="#d4d4d4",
            insertbackground="#fff", relief=tk.FLAT)
        self._claude_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._claude_key_vis = False
        tk.Button(ckey_f, text="Show", font=("Arial", 8), bg="#444",
                  fg="#fff", relief=tk.FLAT, padx=6,
                  command=self._toggle_claude_key).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Label(win, text="Model:", bg=BG, fg=FG).pack(
            anchor="w", padx=px, pady=py)
        ttk.Combobox(win, textvariable=self.claude_model_var,
                     values=QWEN_MODELS, state="readonly",
                     width=40).pack(anchor="w", padx=px, pady=py)

        # ── Chess section ──
        tk.Label(win, text="Chess Agent", font=("Arial", 12, "bold"),
                 bg=BG, fg=ACCENT).pack(anchor="w", padx=px, pady=(16, 4))
        tk.Label(win, text="Playing as: auto-detected from the board",
                 bg=BG, fg="#9e9e9e", font=("Arial", 9)).pack(anchor="w", padx=px, pady=py)

        for label, var, lo, hi, res in [
            ("YOLO confidence:",  self.conf_var,        0.10, 0.95, 0.05),
            ("Stockfish depth:",  self.depth_var,       5,    25,   1),
            ("Scan interval (s):",self.interval_var,    1.0,  15.0, 0.5),
            ("Click delay (s):",  self.click_delay_var, 0.05, 1.0,  0.05),
        ]:
            tk.Label(win, text=label, bg=BG, fg=FG).pack(anchor="w", padx=px, pady=py)
            tk.Scale(win, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, bg=BG, fg=FG,
                     highlightthickness=0, troughcolor="#333",
                     length=300).pack(anchor="w", padx=px, pady=py)

        # close button (saves config)
        def _close_settings():
            _save_config(
                self.claude_api_key_var.get().strip(),
                self.claude_model_var.get(),
                self.use_local_var.get(),
            )
            win.destroy()
        tk.Button(win, text="Close", font=("Arial", 11, "bold"),
                  bg=ACCENT, fg="#fff", relief=tk.FLAT, padx=30, pady=8,
                  command=_close_settings, cursor="hand2",
                  ).pack(pady=(16, 10))

    def _toggle_claude_key(self):
        if hasattr(self, "_claude_key_entry"):
            self._claude_key_vis = getattr(
                self, "_claude_key_vis", False)
            self._claude_key_vis = not self._claude_key_vis
            self._claude_key_entry.config(
                show="" if self._claude_key_vis else "*")

    # ==================================================================
    #  Logging (thread-safe)
    # ==================================================================
    def log(self, msg, tag=""):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.root.after(0, self._log_insert, line, tag)

    def _log_insert(self, line, tag):
        self.log_text.insert(tk.END, line, tag)
        self.log_text.see(tk.END)

    # ==================================================================
    #  Image display
    # ==================================================================
    def _show_image(self, img):
        try:
            w = 640
            r = w / img.width
            h = int(img.height * r)
            resized = img.resize((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self.screenshot_label.config(image=photo, text="")
            self.screenshot_label.image = photo
        except Exception:
            pass

    # ==================================================================
    #  Claude (screenshot + task -> start_chess or comment)
    # ==================================================================
    def _ask_claude(self, task, screenshot, ui_elements=None):
        """Send screenshot + task to vision model (local Qwen3-VL or DashScope). Returns ('start_chess', reason) or ('comment', message)."""
        use_local = self.use_local_var.get()
        if not use_local:
            api_key = self.claude_api_key_var.get().strip()
            if not api_key:
                return ("comment", "Enable 'Use local Qwen3-VL' or set DashScope API key in Settings.")

        img = screenshot.convert("RGB")
        w, h = img.size
        scale = 1024 / max(w, h) if max(w, h) > 1024 else 1.0
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        learning = _build_learning_context()
        system = """You are a router for an AI agent. The user will send a task and a screenshot of their screen.
"""
        if learning:
            system += learning + "\n\n"
        system += """Prequisite knowledge:

 CRITICAL - How to determine playing_as: The human player's pieces are ALWAYS at the BOTTOM of the board (closer to bottom of screen). Look ONLY at the bottom two rows—which color of pieces are there?
  - DARK/GREY/BROWN pieces at the bottom → playing_as: "black"
  - LIGHT/WHITE/CREAM pieces at the bottom → playing_as: "white"
  IGNORE which piece moved. Do NOT infer from "a white piece moved" that the user plays white. Look at the bottom two rows only. Also look for UI text like "You play as Black", "Playing Black", "Black to move" near the bottom—that confirms black.

Respond with exactly one JSON object, no markdown, no extra text:

FIRST: Check if the user is giving FEEDBACK or a CORRECTION. If so: {"action": "store_feedback", "feedback": {"user_correction": "...", "what_was_wrong": "...", "correct_approach": "..."}}

ELSE:
- Chess task + board visible: {"action": "start_chess", "reason": "...", "playing_as": "white" or "black"}
- Chess task, NO board: {"action": "comment", "message": "No chess board visible."}
- Task requires CONTROLLING the screen: you MUST return ONE execute action. Use the UI elements list to click by name. One message = one command.
  * CLICK (click button/link by name): {"action": "CLICK", "parameters": {"element": "Open"}}
  * DOUBLE_CLICK (e.g. desktop icons): {"action": "DOUBLE_CLICK", "parameters": {"element": "Google Chrome"}}
  * TYPE_IN (type into a text field): {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "chess.com"}}
  * MENU (select menu path): {"action": "MENU", "parameters": {"path": "File->Open"}}
  * KEYS (keyboard shortcut): {"action": "KEYS", "parameters": {"keys": ["win"]}}
  * TYPE_TEXT (type into focused field): {"action": "TYPE_TEXT", "parameters": {"text": "notepad"}}
  * {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}
- Observational only (user asks "what's on screen?" or "describe this", no action wanted): {"action": "comment", "message": "your observation"}

Response format: You SHOULD output <thinking>...</thinking> first. In <thinking>: (1) What do you see? (2) What does the user want? (3) Which UI element from the list will you use (or use KEYS/TYPE_TEXT)? Then on the next line exactly one JSON object. Example:
<thinking>
User wants to open VSCode. I see the desktop/taskbar. I'll press the Win key to open Start so they can search for VSCode.
</thinking>
{"action": "KEY_PRESS", "parameters": {"keys": ["win"]}}"""

        user_text = f"Task: {task}\n\nLook at the screenshot and respond with JSON."
        if ui_elements:
            user_text += "\n\nClickable UI elements (use 'element' name in CLICK/DOUBLE_CLICK/TYPE_IN):\n"
            for item in ui_elements[:60]:
                name = (item.get("name") or "").strip()
                ctype = item.get("control_type", "")
                if name:
                    user_text += f"  \"{name}\" ({ctype})\n"
        try:
            if use_local:
                from local_qwen_vl import call_local_qwen
                raw_response = call_local_qwen(system, user_text, img, conversation_messages=None, max_tokens=4096)
            else:
                user_content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
                raw_response = _call_qwen(
                    self.claude_api_key_var.get().strip(),
                    self.claude_model_var.get(), system, user_content, max_tokens=4096
                )
            text = raw_response
            # If router included <thinking>, log it and parse only the part after
            think_match = re.search(r"<thinking\s*>.*?</thinking\s*>", text, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking = think_match.group(0)
                inner = re.search(r"<thinking\s*>(.*?)</thinking\s*>", thinking, re.DOTALL | re.IGNORECASE)
                thinking = inner.group(1).strip() if inner else thinking
                self.log("  [Router] thinking:", "header")
                for line in thinking.split("\n"):
                    self.log(f"  {line.strip()}", "thought")
                self.log("  [Router] → parsing action", "dim")
                text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1)
            data = _parse_claude_json(text)
            # If parse failed but response clearly contains store_feedback, try parsing full response
            if data is None and "store_feedback" in raw_response:
                data = _parse_claude_json(raw_response)
            if data is None:
                raw_preview = (text.strip() or "(empty)")[:120]
                return ("comment", f"Model did not return valid JSON (reply was: '{raw_preview}'). Reply with a single JSON object only, e.g. {{\"action\": \"KEYS\", \"parameters\": {{\"keys\": [\"win\"]}}}}.")
            raw_action = (data.get("action") or "comment")
            action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
            a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
            if a_norm in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                params = data.get("parameters", data)
                act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
            # Claude sometimes wraps the real action in comment
            if action == "comment":
                msg = data.get("message") or data.get("reason") or ""
                if isinstance(msg, str) and msg.strip().startswith("{"):
                    inner = _parse_claude_json(msg.strip())
                    if isinstance(inner, dict):
                        ia = (inner.get("action") or "").lower().strip().replace(" ", "_").replace("-", "_")
                        if ia == "store_feedback":
                            fb = inner.get("feedback", {})
                            return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
                        if ia.upper().replace(" ", "_") in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                            params = inner.get("parameters", inner)
                            return ("execute", {"action": ia.upper().replace(" ", "_"), "parameters": params if isinstance(params, dict) else {}})
            # Aliases
            if action == "storefeedback":
                action = "store_feedback"
            if action == "keypress":
                action = "keys"
            elif action == "typetext":
                action = "type_text"
            elif action == "taskcomplete":
                action = "task_complete"
            if action == "start_chess":
                reason = data.get("reason", "Chess board detected.")
                playing_as = (data.get("playing_as") or "white").lower()
                if playing_as not in ("white", "black"):
                    playing_as = "white"
                return ("start_chess", {"reason": reason, "playing_as": playing_as})
            if action == "store_feedback":
                feedback = data.get("feedback", {})
                if not isinstance(feedback, dict):
                    feedback = {"raw": str(feedback)}
                return ("store_feedback", feedback)
            # Screen control: exact match
            if action in ("click", "double_click", "type_in", "menu", "keys", "type_text", "task_complete"):
                params = data.get("parameters", data)
                act = "KEYS" if action == "keys" else action.upper().replace(" ", "_")
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
            # Fallback: if it looks like a screen action
            params = data.get("parameters")
            if isinstance(params, dict) or "parameters" in data:
                a_upper = str(raw_action).upper().replace(" ", "_").replace("-", "_")
                if a_upper in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                    act = "KEYS" if a_upper == "KEY_PRESS" else a_upper
                    return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}})
            # Last resort: scan raw text for any JSON object that is a screen action (handles extra text / wrong parse)
            for start in [i for i, c in enumerate(text) if c == "{"]:
                depth, end = 0, None
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    continue
                try:
                    obj = json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
                a = (obj.get("action") or "").upper().replace(" ", "_").replace("-", "_")
                if a in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                    p = obj.get("parameters", obj)
                    act = "KEYS" if a == "KEY_PRESS" else a
                    return ("execute", {"action": act, "parameters": p if isinstance(p, dict) else {}})
                if a == "STORE_FEEDBACK":
                    fb = obj.get("feedback", {})
                    return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
            # Last resort: raw text might be the store_feedback JSON (e.g. parser returned wrong object)
            for scan_text in (text, raw_response):
                if "store_feedback" not in scan_text and "store_feedback" not in str(data):
                    continue
                for start in [i for i, c in enumerate(scan_text) if c == "{"]:
                    depth, end = 0, None
                    for i in range(start, len(scan_text)):
                        if scan_text[i] == "{":
                            depth += 1
                        elif scan_text[i] == "}":
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end is None:
                        continue
                    try:
                        obj = json.loads(scan_text[start:end])
                    except json.JSONDecodeError:
                        continue
                    if (obj.get("action") or "").lower().replace(" ", "_") == "store_feedback":
                        fb = obj.get("feedback", {})
                        return ("store_feedback", fb if isinstance(fb, dict) else {"raw": str(fb)})
            msg = data.get("message", data.get("reason", text))
            self.log(f"  [router] action={repr(raw_action)} normalized={repr(action)} -> comment", "dim")
            return ("comment", msg or "No message from Claude.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"Router error: {e}", "error")
            self.log(tb, "dim")
            return ("comment", f"API error: {e}")

    def _ask_claude_next_action(self, task, screenshot, action_history: list, conversation_messages: list = None, thought_history: list = None, ui_elements=None):
        """Ask vision model for the next screen-control action. Uses conversation_messages and thought_history for context.
        Returns (action, payload, updated_conversation_messages, updated_thought_history)."""
        use_local = self.use_local_var.get()
        if not use_local and not self.claude_api_key_var.get().strip():
            return ("comment", "Enable 'Use local Qwen3-VL' or set DashScope API key.", conversation_messages or [], thought_history or [])
        thought_history = list(thought_history) if thought_history else []

        img = screenshot.convert("RGB")
        w, h = img.size
        scale = 1024 / max(w, h) if max(w, h) > 1024 else 1.0
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        history_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_history[-12:]))
        learning = _build_learning_context()

        system = "You control the user's computer. You have the full conversation history above: the user's task and every action you've already taken. Use it to remember the goal and what you've done."
        system += " Use the UI elements list to click by name. Output one action per message."
        if thought_history:
            system += "\n\nYour previous thinking (use this to stay consistent and build on your reasoning):\n"
            for i, prev in enumerate(thought_history[-3:]):
                snippet = prev[:800] + "..." if len(prev) > 800 else prev
                ago = len(thought_history[-3:]) - i
                system += f"\n[Your thinking from {ago} turn(s) ago]\n{snippet}\n"
        if learning:
            system += "\n" + learning
        system += """

You have two phases in your response:

PHASE 1 - THINKING (required): First output a <thinking> block. State: (1) What you see in the screenshot and what the user's task is. (2) What you have already done. (3) Which UI element or action you will use next and why.

PHASE 2 - OUTPUT (required): Right after </thinking>, output exactly one JSON object. Valid JSON only.

Actions (use element names from the UI list):
- CLICK: {"action": "CLICK", "parameters": {"element": "Open"}}
- DOUBLE_CLICK: {"action": "DOUBLE_CLICK", "parameters": {"element": "Google Chrome"}}
- TYPE_IN: {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "chess.com"}}
- MENU: {"action": "MENU", "parameters": {"path": "File->Open"}}
- KEYS: {"action": "KEYS", "parameters": {"keys": ["win", "r"]}}
- TYPE_TEXT: {"action": "TYPE_TEXT", "parameters": {"text": "notepad"}}
- TASK_COMPLETE: {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}

Example:
<thinking>
I see the Run dialog. The user wanted to open Notepad. I already pressed Win+R. I need to type "notepad" in the Open field and press Enter.
</thinking>
{"action": "TYPE_TEXT", "parameters": {"text": "notepad"}}"""

        user_text = f"Task: {task}\n\nActions so far:\n{history_str or '  (none yet)'}\n\nWhat's the next action? Look at the screenshot."
        if ui_elements:
            user_text += "\n\nClickable UI elements (use 'element' name in CLICK/DOUBLE_CLICK/TYPE_IN):\n"
            for item in ui_elements[:60]:
                name = (item.get("name") or "").strip()
                ctype = item.get("control_type", "")
                if name:
                    user_text += f"  \"{name}\" ({ctype})\n"
        messages = list(conversation_messages) if conversation_messages else []

        try:
            if use_local:
                from local_qwen_vl import call_local_qwen
                # Build text-only history for local model
                local_history = []
                for m in messages:
                    role, content = m.get("role"), m.get("content")
                    if isinstance(content, list):
                        text = next((c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"), "")
                    else:
                        text = content or ""
                    if role and (text or role == "assistant"):
                        local_history.append({"role": role, "content": text})
                text = call_local_qwen(system, user_text, img, conversation_messages=local_history, max_tokens=16384)
                messages.append({"role": "user", "content": user_text})
            else:
                user_content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
                text = _call_qwen(
                    self.claude_api_key_var.get().strip(),
                    self.claude_model_var.get(), system, user_content,
                    messages=messages, max_tokens=16384
                )
                messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": text})
            if len(messages) > 8:
                messages = messages[-8:]

            # Extract and log thinking phase; append to thought history for next turn
            think_match = re.search(r"<thinking\s*>.*?</thinking\s*>", text, re.DOTALL | re.IGNORECASE)
            if think_match:
                thinking = think_match.group(0)
                inner = re.search(r"<thinking\s*>(.*?)</thinking\s*>", thinking, re.DOTALL | re.IGNORECASE)
                thinking = inner.group(1).strip() if inner else thinking
                thought_history.append(thinking)
                if len(thought_history) > 10:
                    thought_history = thought_history[-10:]
                self.log("  Model thinking (reasoning; then it will output one action JSON):", "header")
                for line in thinking.split("\n"):
                    self.log(f"  {line.strip()}", "thought")
                self.log("  (end thinking → parsing action below)", "dim")
                # Parse only the part AFTER </thinking> so we don't treat thinking text as JSON
                text = re.sub(r"<thinking\s*>.*?</thinking\s*>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            else:
                self.log("  --- Model thinking: (no block) ---", "dim")

            if not text or "{" not in text:
                return ("comment", "Model only sent thinking, no action JSON. Output JSON right after </thinking>.", messages, thought_history)
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1)
            data = _parse_claude_json(text)
            if data is None:
                return ("comment", f"Model returned invalid JSON. Ask again or rephrase. Raw: {text[:120]}...", messages, thought_history)
            raw_action = data.get("action") or "comment"
            action = str(raw_action).lower().strip().replace(" ", "_").replace("-", "_")
            a_norm = str(raw_action).upper().strip().replace(" ", "_").replace("-", "_")
            if a_norm in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                params = data.get("parameters", data)
                act = "KEYS" if a_norm == "KEY_PRESS" else a_norm
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history)
            if action == "comment":
                msg = data.get("message") or data.get("reason") or ""
                if isinstance(msg, str) and msg.strip().startswith("{"):
                    inner = _parse_claude_json(msg.strip())
                    if isinstance(inner, dict):
                        ia = (inner.get("action") or "").upper().replace(" ", "_").replace("-", "_")
                        if ia in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                            params = inner.get("parameters", inner)
                            act = "KEYS" if ia == "KEY_PRESS" else ia
                            return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history)
            if action == "keypress":
                action = "keys"
            elif action == "typetext":
                action = "type_text"
            elif action == "taskcomplete":
                action = "task_complete"
            if action in ("click", "double_click", "type_in", "menu", "keys", "type_text", "task_complete"):
                params = data.get("parameters", data)
                act = "KEYS" if action == "keys" else action.upper().replace(" ", "_")
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history)
            a_upper = str(raw_action).upper().replace(" ", "_").replace("-", "_")
            if a_upper in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                params = data.get("parameters", data)
                act = "KEYS" if a_upper == "KEY_PRESS" else a_upper
                return ("execute", {"action": act, "parameters": params if isinstance(params, dict) else {}}, messages, thought_history)
            # Last resort: text might be the raw action JSON (e.g. parser returned wrong object). Find any action JSON and run it.
            for start in [i for i, c in enumerate(text) if c == "{"]:
                depth, end = 0, None
                for i in range(start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    continue
                try:
                    obj = json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
                a = (obj.get("action") or "").upper().replace(" ", "_").replace("-", "_")
                if a in ("CLICK", "DOUBLE_CLICK", "TYPE_IN", "MENU", "KEYS", "KEY_PRESS", "TYPE_TEXT", "TASK_COMPLETE"):
                    p = obj.get("parameters", obj)
                    act = "KEYS" if a == "KEY_PRESS" else a
                    return ("execute", {"action": act, "parameters": p if isinstance(p, dict) else {}}, messages, thought_history)
            return ("comment", data.get("message", text), messages, thought_history)
        except Exception as e:
            return ("comment", f"API error: {e}", conversation_messages or [], thought_history)

    # ==================================================================
    #  Stats
    # ==================================================================
    def _update_stats(self, status):
        mode = self._mode or "--"
        cycles = self.chess.cycle_count if self._mode == "chess" else 0
        actions = self.chess.move_count if self._mode == "chess" else self._screen_action_count
        self.root.after(0, lambda: self.stats_label.config(
            text=f"Mode: {mode}\n"
                 f"Cycles: {cycles}\n"
                 f"Actions: {actions}\n"
                 f"Status: {status}"))

    # ==================================================================
    #  Chess model loading
    # ==================================================================
    def _load_chess(self):
        ok = self.chess.load_models()

        def _done():
            if ok:
                self.status_label.config(text="Ready", fg="#4CAF50")
                self.log("Chess engine ready.", "info")
                self.log("Type a task and press Start.\n", "info")
            else:
                self.status_label.config(text="Chess load error", fg="#f44336")

        self.root.after(0, _done)

    # ==================================================================
    #  Start / Stop
    # ==================================================================
    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        task = self.task_text.get("1.0", tk.END).strip()
        if not task:
            self.log("Enter a task first!", "error")
            return

        self._running = True
        self.start_btn.config(text="Stop", bg="#f44336",
                              activebackground="#d32f2f")
        self.log("Workflow: Screenshot → Router (chess/execute/comment) → if execute: loop [run action → screenshot → Claude next] until done.", "dim")

        if self.use_local_var.get() or self.claude_api_key_var.get().strip():
            # Vision path: screenshot -> ask model -> start_chess or log comment
            self._thread = threading.Thread(
                target=self._start_with_claude, daemon=True, args=(task,))
            self._thread.start()
        else:
            # No Claude key: only chess via keyword
            if is_chess_task(task):
                self._mode = "chess"
                self._start_chess()
            else:
                self.log("Enable 'Use local Qwen3-VL' or add DashScope API key in Settings.", "warning")
                self._running = False
                self.start_btn.config(text="Start", bg="#4CAF50",
                                      activebackground="#388E3C")

    def _stop(self):
        self._running = False
        self.start_btn.config(text="Start", bg="#4CAF50",
                              activebackground="#388E3C")
        self.status_label.config(text="Stopped", fg="#FF9800")
        self.log("\nAGENT STOPPED", "header")
        if self._mode == "chess":
            self.log(f"  Moves played: {self.chess.move_count}", "info")
            self.log(f"  Scans: {self.chess.cycle_count}\n", "info")
        elif self._mode == "screen_control":
            self.log(f"  Screen actions: {self._screen_action_count}\n", "info")

    def _start_with_claude(self, task):
        """Take screenshot, ask Claude, then start_chess or log comment.
        Workflow: (1) Screenshot → (2) Router decides: chess | execute first action | comment.
        If execute → screen control loop: execute action → screenshot → Claude next action → repeat until TASK_COMPLETE or comment."""
        try:
            self.log("\n--- PHASE 1: Router (screenshot + first decision) ---", "header")
            self.log("Taking screenshot for Claude...", "action")
            self._hide_for_screenshot()
            time.sleep(0.3)
            screenshot = pyautogui.screenshot()
            ui_elements = get_ui_elements() if _PYWINAUTO_OK else []  # get while GUI hidden
            self._show_after_screenshot()
            self.root.after(0, self._show_image, screenshot)

            self.log("Asking Claude (router): chess / execute one action / comment?", "action")
            action, payload = self._ask_claude(task, screenshot, ui_elements)

            if action == "store_feedback":
                # Claude detected user feedback - store the feedback json block
                entry = {
                    "type": "feedback",
                    "user_message": task,
                    "feedback": payload if isinstance(payload, dict) else {"raw": str(payload)},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                _save_prompt_entry(entry)
                self.log("  Feedback stored for learning.", "info")
                fb_str = json.dumps(payload, indent=2)
                self.log(f"  {fb_str[:300]}{'...' if len(fb_str) > 300 else ''}", "dim")
                self._running = False
                self.root.after(0, lambda: self.start_btn.config(
                    text="Start", bg="#4CAF50", activebackground="#388E3C"))
                self.root.after(0, lambda: self.status_label.config(
                    text="Ready", fg="#4CAF50"))
                return

            if action == "start_chess":
                reason = payload.get("reason", "Chess board detected.") if isinstance(payload, dict) else payload
                playing_as = payload.get("playing_as", "white") if isinstance(payload, dict) else "white"
                self.log(f"  Claude: {reason}", "info")
                self.log(f"  Claude detected: playing as {playing_as}", "info")
                self.turn_var.set(playing_as)  # use Claude's detection for chess bot
                self._mode = "chess"
                self.root.after(0, self._start_chess)
                return

            if action == "execute":
                act = payload.get("action", "") if isinstance(payload, dict) else ""
                self.log(f"  Router → execute first action: {act}", "info")
                self.log("--- PHASE 2: Screen control loop (execute → screenshot → next action) ---", "header")
                self._mode = "screen_control"
                self.root.after(0, lambda: self._start_screen_control(task, screenshot, payload))
                return

            # comment path
            msg = payload.get("message", payload) if isinstance(payload, dict) else payload
            msg_str = str(msg).strip() if msg else "(no message)"
            self.log(f"[Claude] {msg_str}", "thought")
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(
                text="Ready", fg="#4CAF50"))
            # Short UI line (no raw JSON); big label stays small
            if msg_str.startswith("{"):
                display = "Claude replied (see log)"
            else:
                display = (msg_str[:28] + "…") if len(msg_str) > 28 else msg_str
            self.root.after(0, lambda d=display: self.action_label.config(text=d))
            self.root.after(0, lambda d=display: self._update_stats(d))
        except Exception as e:
            self.log(f"Claude flow error: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(
                text="Ready", fg="#4CAF50"))

    # ------------------------------------------------------------------
    #  Chess mode
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Screen control mode (mouse + keyboard)
    # ------------------------------------------------------------------
    def _start_screen_control(self, task, initial_screenshot, first_action):
        """Start screen control loop: execute first action, then loop for more."""
        self._screen_action_count = 0
        self.status_label.config(text="Screen control - running", fg="#4CAF50")
        self.log("Loop: will execute action → take screenshot → ask Claude for next → repeat.", "info")
        self.log(f"  Task: {task}", "info")
        self.log("", "info")
        self._thread = threading.Thread(
            target=self._screen_control_loop,
            daemon=True,
            args=(task, first_action,),
        )
        self._thread.start()

    def _screen_control_loop(self, task, first_action=None):
        """Loop: execute current action -> screenshot -> ask Claude for next -> repeat until TASK_COMPLETE or stop."""
        try:
            sw, sh = pyautogui.size()
            action_history = []
            conversation_messages = []
            thought_history = []
            current_action = first_action
            turn = 0

            while self._running:
                # 1) Execute current action (if we have one and it's not TASK_COMPLETE)
                if current_action:
                    act = current_action.get("action", "")
                    params = current_action.get("parameters", current_action)
                    action_dict = {"action": act, "parameters": params}

                    if act == "TASK_COMPLETE":
                        msg = params.get("message", "Task complete")
                        self.log(f"  Done: {msg}", "info")
                        display = (str(msg)[:36] + "…") if len(str(msg)) > 36 else str(msg)
                        self.root.after(0, lambda d=display: self.action_label.config(text=d))
                        self.root.after(0, lambda m=msg: self._update_stats(m))
                        break

                    self.log(f"  Executing: {act} {params}", "action")
                    self._hide_for_screenshot()
                    time.sleep(0.1)
                    ok, result = _execute_action(action_dict, sw, sh)
                    time.sleep(1.0)
                    self._show_after_screenshot()

                    action_history.append(f"{act}: {result}")
                    self._screen_action_count += 1
                    self.root.after(0, lambda r=result: self.action_label.config(text=r[:30]))
                    self.root.after(0, lambda: self._update_stats(f"last: {act}"))
                    if ok:
                        self.log(f"  -> OK: {result}", "info")
                    else:
                        self.log(f"  -> Failed: {result}", "error")

                if not self._running:
                    break

                # 2) Screenshot again (so Claude sees the result of the action)
                self.log("  Taking screenshot...", "action")
                self._hide_for_screenshot()
                time.sleep(0.3)
                screenshot = pyautogui.screenshot()
                ui_elements = get_ui_elements() if _PYWINAUTO_OK else []  # get while GUI hidden
                self._show_after_screenshot()
                self.root.after(0, self._show_image, screenshot)

                if not self._running:
                    break

                # 3) Ask Claude for next action
                turn += 1
                self.log(f"  --- Turn {turn}: asking Claude for next action ---", "header")
                self.log("  Asking Claude for next action...", "action")
                action, payload, conversation_messages, thought_history = self._ask_claude_next_action(
                    task, screenshot, action_history, conversation_messages, thought_history, ui_elements
                )

                if action == "execute":
                    current_action = payload
                    next_act = payload.get("action", "") if isinstance(payload, dict) else ""
                    self.log(f"  → Will execute: {next_act} (loop continues)", "info")
                else:
                    # Claude returned a comment instead of an action - log and stop
                    payload_str = str(payload).strip() if payload else "(empty)"
                    self.log(f"  Claude replied with comment (stopping): {payload_str[:200]}", "warning")
                    if not payload_str or payload_str.lower() in ("action", "ok", "done"):
                        self.log("  (Claude may have returned non-JSON or vague text; check model output.)", "dim")
                    # Update UI: short line only (no raw JSON)
                    if payload_str.startswith("{"):
                        display = "Stopped (see log)"
                    else:
                        display = (payload_str[:28] + "…") if len(payload_str) > 28 else payload_str or "Stopped"
                    self.root.after(0, lambda d=display: self.action_label.config(text=d))
                    self.root.after(0, lambda d=display: self._update_stats(f"Stopped: {d}"))
                    break

        except Exception as e:
            self.log(f"SCREEN CONTROL ERROR: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            err_msg = str(e)[:40]
            self.root.after(0, lambda m=err_msg: self.action_label.config(text=m))
            self.root.after(0, lambda m=err_msg: self._update_stats(f"Error: {m}"))
        finally:
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(
                text="Stopped", fg="#FF9800"))

    # ------------------------------------------------------------------
    #  Chess mode
    # ------------------------------------------------------------------
    def _start_chess(self):
        if not self.chess.ready:
            self.log("Chess engine not loaded yet!", "error")
            self._running = False
            self.start_btn.config(text="Start", bg="#4CAF50",
                                  activebackground="#388E3C")
            return

        self.chess.reset()
        self.status_label.config(text="Chess - running", fg="#4CAF50")

        self.log(f"\n{'='*55}", "header")
        self.log("CHESS AGENT STARTED", "header")
        self.log(f"  Playing as: {self.turn_var.get()}", "info")
        self.log(f"  Scan interval: {self.interval_var.get()}s", "info")
        self.log(f"  Stockfish depth: {self.depth_var.get()}", "info")
        self.log(f"{'='*55}\n", "header")

        self._thread = threading.Thread(target=self._chess_loop, daemon=True)
        self._thread.start()

    def _chess_loop(self):
        """Background thread for chess auto-play."""
        try:
            while self._running:
                self.log(f"-- Scan #{self.chess.cycle_count + 1} --", "action")

                # screenshot
                self._hide_for_screenshot()
                t0 = time.time()
                ss = pyautogui.screenshot()
                cap_t = time.time() - t0
                self._show_after_screenshot()

                # analyse
                r = self.chess.analyze(
                    ss,
                    conf=self.conf_var.get(),
                    depth=self.depth_var.get(),
                    turn=self.turn_var.get(),
                    force_move=False)

                if r["annotated"]:
                    self.root.after(0, self._show_image, r["annotated"])

                # --- act on result ---
                if r["status"] == "move":
                    best = r["best_move"]
                    fsq, tsq = best[:2], best[2:4]
                    promo = f"={best[4:].upper()}" if len(best) > 4 else ""
                    disp = f"{fsq} -> {tsq}{promo}"
                    self.root.after(0, lambda d=disp: self.action_label.config(text=d))
                    self.root.after(0, lambda e=r["eval"]:
                                   self.eval_label.config(text=f"Eval: {e}"))

                    # execute click
                    self._hide_for_screenshot()
                    self.chess.execute_move(
                        best, r["board_box"], r["orientation"],
                        click_delay=self.click_delay_var.get())
                    self._show_after_screenshot()

                    self.log(f"  Move #{self.chess.move_count} played!", "info")
                    self._update_stats(f"played {fsq}->{tsq}{promo}")

                    # re-scan to track board state
                    self._hide_for_screenshot()
                    ss2 = pyautogui.screenshot()
                    self._show_after_screenshot()
                    self.chess.capture_post_move_fen(ss2, self.conf_var.get())

                elif r["status"] == "game_over":
                    self.log(f"  GAME OVER: {r.get('message', '')}", "header")
                    self._update_stats(f"game over: {r.get('message','')}")
                    break

                elif r["status"] == "waiting":
                    self._update_stats("waiting for opponent")

                elif r["status"] == "no_board":
                    self._update_stats("no board detected")

                else:
                    self._update_stats("detection error")

                # interval wait with early-exit
                end = time.time() + self.interval_var.get()
                while time.time() < end and self._running:
                    time.sleep(0.05)

        except Exception as e:
            self.log(f"CHESS LOOP ERROR: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
        finally:
            self._running = False
            self.root.after(0, lambda: self.start_btn.config(
                text="Start", bg="#4CAF50", activebackground="#388E3C"))
            self.root.after(0, lambda: self.status_label.config(
                text="Stopped", fg="#FF9800"))

    # ==================================================================
    #  Thread-safe hide / show (for screenshots & clicks)
    # ==================================================================
    def _hide_and_signal(self):
        self.root.withdraw()
        self.root.update()
        self._hide_event.set()

    def _show_and_signal(self):
        self.root.deiconify()
        self.root.update()
        self._show_event.set()

    def _hide_for_screenshot(self):
        self._hide_event.clear()
        self.root.after(0, self._hide_and_signal)
        self._hide_event.wait(timeout=2.0)

    def _show_after_screenshot(self):
        self._show_event.clear()
        self.root.after(0, self._show_and_signal)
        self._show_event.wait(timeout=2.0)

    def _minimize_for_execution(self):
        """Minimize window (alternative to full hide). Screen control now uses _hide_for_screenshot so GUI is fully closed during actions."""
        self._hide_event.clear()
        self.root.after(0, lambda: (self.root.iconify(), self.root.update(), self._hide_event.set()))
        self._hide_event.wait(timeout=2.0)

    def _restore_after_execution(self):
        self._show_event.clear()
        self.root.after(0, lambda: (self.root.deiconify(), self.root.update(), self._show_event.set()))
        self._show_event.wait(timeout=2.0)


# ======================================================================
def main():
    root = tk.Tk()
    AgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
