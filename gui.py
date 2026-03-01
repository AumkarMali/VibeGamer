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
import math
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

# optional voice activation ("vibe gamer" hotword)
try:
    import speech_recognition as sr  # type: ignore[import]
    _VOICE_AVAILABLE = True
except ImportError:
    sr = None
    _VOICE_AVAILABLE = False

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


def _save_config(api_key: str, model: str, use_local: bool = True, **extra):
    """Save API key, model, use_local_model, and any extra keys to config.json."""
    cfg = _load_config()
    cfg.update({
        "claude_api_key": api_key,
        "claude_model": model,
        "use_local_model": use_local,
    })
    cfg.update(extra)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
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
OLLAMA_BASE = "https://app.ollama.com"  # use app.ollama, not bare ollama


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
    """Single-window AI agent with chess and screen-control modes. When task_only=True, only task panel (no hub)."""

    def __init__(self, root, task_only=False):
        self.root = root
        self.task_only = task_only
        root.title("Vibe Gamer")
        root.geometry("1100x800")
        root.configure(bg=BG)
        root.minsize(320, 320)

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

        # voice / hotword state
        self.voice_enabled = False
        self._voice_thread = None
        self._voice_stop_event = threading.Event()

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
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Task panel (old GUI: task / log / screen) — visible by default; hub overlay in corner
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.main_frame, padding="10")
        outer.grid(row=0, column=0, sticky="nsew")

        # ── left panel ──
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Back to hub (hidden when task_only; Kivy dot is the hub)
        self.back_to_hub_btn = tk.Button(
            left, text="← Back to hub", font=("Arial", 11),
            bg="#444", fg="#fff", relief=tk.FLAT, padx=12, pady=6,
            command=self._show_hub, cursor="hand2",
        )
        self.back_to_hub_btn.grid(row=0, column=0, pady=(0, 8))
        if self.task_only:
            self.back_to_hub_btn.grid_remove()

        # title (old GUI)
        tk.Label(left, text="AI Agent", font=("Arial", 24, "bold"),
                 bg=BG, fg=ACCENT).grid(row=1, column=0, pady=(0, 2))
        tk.Label(left, text="Type a task and press Start",
                 font=("Arial", 10, "italic"),
                 bg=BG, fg="#9e9e9e").grid(row=2, column=0, pady=(0, 8))

        # status
        self.status_label = tk.Label(left, text="Loading...",
                                     font=("Arial", 12), bg=BG, fg="#FF9800")
        self.status_label.grid(row=3, column=0, pady=(0, 10))

        # task input
        task_frame = ttk.LabelFrame(left, text="Task", padding="6")
        task_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.task_text = tk.Text(
            task_frame, height=3, width=32, font=("Consolas", 10),
            bg=BG_DARK, fg="#d4d4d4", insertbackground="#fff",
            wrap=tk.WORD, relief=tk.FLAT,
            highlightthickness=1, highlightcolor="#444")
        self.task_text.pack(fill=tk.X)
        self.task_text.insert("1.0", "play chess")

        # buttons row
        btn_row = tk.Frame(left, bg=BG)
        btn_row.grid(row=5, column=0, pady=(0, 8))

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
        ).grid(row=6, column=0, pady=(0, 10))

        # last action display
        act_frame = ttk.LabelFrame(left, text="Last Action", padding="8")
        act_frame.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        self.action_label = tk.Label(
            act_frame, text="--", font=("Consolas", 24, "bold"),
            bg=BG, fg="#FFD700")
        self.action_label.pack()
        self.eval_label = tk.Label(
            act_frame, text="", font=("Consolas", 11), bg=BG, fg="#aaa")
        self.eval_label.pack()

        # stats
        stats_frame = ttk.LabelFrame(left, text="Stats", padding="8")
        stats_frame.grid(row=8, column=0, sticky="ew")
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

        # Glass-dot hub: only when not task_only (Kivy runs the dot as on-screen overlay)
        if not self.task_only:
            self._build_hub_ui(self.root)

    def _build_hub_ui(self, parent):
        self._hub_expanded = False
        # Larger canvas so expanded radial menu (6 buttons) fits fully
        size = 380
        self.hub_canvas = tk.Canvas(parent, bg=BG, highlightthickness=0, bd=0)
        self.hub_canvas.config(cursor="hand2")
        self.hub_canvas.place(relx=1.0, rely=1.0, x=-12, y=-12, anchor="se", width=size, height=size)
        self.hub_canvas.bind("<Configure>", self._redraw_hub)

    def _show_hub(self):
        """Switch to small hub-only window (glass dot, expand to 6 dots). No-op when task_only."""
        if not hasattr(self, "hub_canvas"):
            return
        self.root.geometry("420x420")
        self.main_frame.grid_remove()
        self.hub_canvas.place_forget()
        self.hub_canvas.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._redraw_hub()

    def _show_task_panel(self):
        """Show full task panel (old GUI) with hub overlay in corner. No-op when task_only."""
        if not hasattr(self, "hub_canvas"):
            return
        self.root.geometry("1100x800")
        self.hub_canvas.grid_remove()
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        size = 380
        self.hub_canvas.place(relx=1.0, rely=1.0, x=-12, y=-12, anchor="se", width=size, height=size)
        if hasattr(self, "task_text"):
            self.root.after(50, lambda: self.task_text.focus_set())

    def _open_hub(self, expand=True):
        """Show hub (small window with glass dot) and optionally expand radial menu."""
        self._show_hub()
        if expand:
            self._hub_expanded = True
            self._redraw_hub()

    def _redraw_hub(self, event=None):
        """Apple-style glass orb: frosted layers, then radial menu when expanded."""
        if not hasattr(self, "hub_canvas"):
            return
        c = self.hub_canvas
        c.delete("all")
        w = c.winfo_width() or 1
        h = c.winfo_height() or 1
        cx, cy = w // 2, h // 2
        expanded = getattr(self, "_hub_expanded", False)

        # Apple-style glass colors: frosted dark with soft highlights
        glass_dark = "#3c3c4a"
        glass_mid = "#4a4a5c"
        glass_highlight = "#6e6e82"
        glass_edge = "#8a8a9e"
        glass_inner_light = "#5c5c6e"
        sub_glass = "#454555"
        sub_edge = "#7a7a8e"
        label_fg = "#e8e8ee"
        hint_fg = "#9898a8"

        # Smaller center orb; radius chosen so expanded ring fits in canvas
        radius = min(28, int(min(w, h) * 0.10))
        if radius < 18:
            radius = 18

        # —— Main glass orb (layered like Apple) ——
        # 1) Outer soft edge
        c.create_oval(
            cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2,
            fill="", outline=glass_edge, width=1, tags=("hub_main",)
        )
        # 2) Main fill
        c.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=glass_dark, outline=glass_mid, width=1, tags=("hub_main",)
        )
        # 3) Inner “frost” layer
        inner_r = max(radius - 8, 12)
        c.create_oval(
            cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
            fill=glass_mid, outline="", width=0, tags=("hub_main",)
        )
        # 4) Top-left highlight (glass reflection)
        spot_r = radius // 3
        spot_cx = cx - radius // 3
        spot_cy = cy - radius // 3
        c.create_oval(
            spot_cx - spot_r, spot_cy - spot_r, spot_cx + spot_r, spot_cy + spot_r,
            fill=glass_inner_light, outline="", width=0, tags=("hub_main",)
        )
        # 5) Center label
        c.create_text(
            cx, cy, text="Vibe\nGamer",
            fill=label_fg, font=("Segoe UI", 11, "bold"), tags=("hub_main",)
        )
        c.tag_bind("hub_main", "<Button-1>", lambda _e: self._toggle_hub_expansion())

        if not expanded:
            # Hint only when collapsed
            c.create_text(
                cx, cy + radius + 36,
                text="Click to open",
                fill=hint_fg, font=("Segoe UI", 9, "normal")
            )
            return

        # —— Expanded: Apple-style radial menu ——
        ring_r = radius * 2.4
        sub_radius = int(radius * 0.48)
        # Frosted “sheet” behind the ring (subtle circle)
        c.create_oval(
            cx - ring_r - sub_radius - 12, cy - ring_r - sub_radius - 12,
            cx + ring_r + sub_radius + 12, cy + ring_r + sub_radius + 12,
            fill=glass_dark, outline=glass_edge, width=1
        )
        # Ring line (optional, subtle)
        c.create_oval(
            cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
            fill="", outline=glass_mid, width=1
        )

        items = [
            ("Settings", "settings"),
            ("Task", "task"),
            ("Mic", "mic"),
            ("Screen", "screen"),
            ("History", "history"),
        ]
        for idx, (label, key) in enumerate(items):
            angle = (2 * math.pi * idx / len(items)) - math.pi / 2
            sx = cx + ring_r * math.cos(angle)
            sy = cy + ring_r * math.sin(angle)
            # Sub-dot: same glass treatment (fill + edge + small highlight)
            c.create_oval(
                sx - sub_radius - 1, sy - sub_radius - 1,
                sx + sub_radius + 1, sy + sub_radius + 1,
                fill="", outline=sub_edge, width=1, tags=(f"sub_{key}", "subdot"),
            )
            c.create_oval(
                sx - sub_radius, sy - sub_radius,
                sx + sub_radius, sy + sub_radius,
                fill=sub_glass, outline=glass_highlight, width=1,
                tags=(f"sub_{key}", "subdot"),
            )
            spot2_r = sub_radius // 2
            c.create_oval(
                sx - spot2_r, sy - spot2_r - 2, sx + spot2_r, sy - 2 + spot2_r,
                fill=glass_inner_light, outline="", width=0, tags=(f"sub_{key}", "subdot"),
            )
            txt = c.create_text(
                sx, sy + sub_radius + 14,
                text=label, fill=label_fg, font=("Segoe UI", 9),
                tags=(f"sub_{key}", "subdot"),
            )
            c.tag_bind(f"sub_{key}", "<Button-1>", lambda _e, k=key: self._handle_subdot_click(k))
            c.tag_bind(txt, "<Button-1>", lambda _e, k=key: self._handle_subdot_click(k))

        # Hint when expanded
        c.create_text(
            cx, cy + ring_r + sub_radius + 28,
            text="Click center to close",
            fill=hint_fg, font=("Segoe UI", 8, "normal")
        )

    def _toggle_hub_expansion(self):
        self._hub_expanded = not getattr(self, "_hub_expanded", False)
        self._redraw_hub()

    def _handle_subdot_click(self, key: str):
        """Handle clicks on radial dots: settings / task / mic / etc."""
        if key == "settings":
            self._open_settings()
        elif key == "task":
            self._show_task_panel()
        elif key == "mic":
            self._toggle_voice_activation()
        elif key == "screen":
            self._show_task_panel()
            self.log("Screen control: enter a task; the agent will click/type as needed.", "info")
        elif key == "history":
            self._show_task_panel()
            self.log("Prompt history is in prompt_history.json (learning from feedback).", "info")

    def _toggle_voice_activation(self):
        """Start/stop background hotword listening for \"vibe gamer\"."""
        if not _VOICE_AVAILABLE or sr is None:
            self.log(
                "Voice activation requires 'speech_recognition' (and microphone drivers). "
                "Install with: pip install SpeechRecognition pyaudio",
                "warning",
            )
            return

        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.log('Voice hotword listening enabled – say "vibe gamer" near your mic.', "info")
            self._voice_stop_event.clear()
            if self._voice_thread is None or not self._voice_thread.is_alive():
                self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
                self._voice_thread.start()
        else:
            self.log("Voice hotword listening disabled.", "info")
            self._voice_stop_event.set()

    def _voice_loop(self):
        """Background loop that listens for the phrase \"vibe gamer\"."""
        if not _VOICE_AVAILABLE or sr is None:
            return
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Voice mic error: {e}", "error"))
            self.voice_enabled = False
            return

        while self.voice_enabled and not self._voice_stop_event.is_set():
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                try:
                    text = recognizer.recognize_google(audio).lower()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    # network or API issue; stay quiet instead of spamming errors
                    continue

                cleaned = text.replace(" ", "")
                if "vibegamer" in cleaned or "vibe gamer" in text:
                    self.root.after(0, lambda: self._open_hub(expand=True))
                    self.root.after(0, lambda: self.log('Heard hotword "vibe gamer" – opening hub.', "info"))
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Voice loop error: {e}", "error"))
                break

        self.voice_enabled = False

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
# Kivy floating glass-dot (on-screen overlay, not in a normal window).
# Install Kivy for the dot:  pip install kivy>=2.2.0
# ======================================================================
def _run_voice_listener(callback_on_hotword, stop_event):
    """Run in a background thread; calls callback_on_hotword() when "vibe gamer" is heard. Stops when stop_event is set."""
    if not _VOICE_AVAILABLE or sr is None:
        return
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
    except Exception:
        return
    while not stop_event.is_set():
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            try:
                text = recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                continue
            cleaned = text.replace(" ", "")
            if "vibegamer" in cleaned or "vibe gamer" in text:
                callback_on_hotword()
        except sr.WaitTimeoutError:
            continue
        except Exception:
            break


def _launch_task_panel():
    """Launch the tkinter task panel in a new process."""
    import subprocess
    exe = sys.executable
    script = os.path.abspath(__file__)
    subprocess.Popen([exe, script, "--task"], cwd=os.path.dirname(script))


def _open_settings_window(parent):
    """Minimal, modern settings window launched from the floating dot."""
    TRANS_S = "#030303"
    WIN_W, WIN_H = 440, 520

    win = tk.Toplevel(parent)
    win.title("")
    win.overrideredirect(1)
    win.attributes("-topmost", True)
    win.configure(bg=TRANS_S)
    try:
        win.attributes("-transparentcolor", TRANS_S)
    except tk.TclError:
        try:
            win.attributes("-alpha", 0.96)
        except tk.TclError:
            pass

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - WIN_W) // 2)
    y = max(0, (sh - WIN_H) // 2 - 60)
    win.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
    win.resizable(False, False)

    # Glass pill background
    bg_c = tk.Canvas(win, bg=TRANS_S, highlightthickness=0, bd=0, width=WIN_W, height=WIN_H)
    bg_c.pack(fill=tk.BOTH, expand=True)
    pd, cr = 4, 24
    for corner_x, corner_y in [(pd, pd), (WIN_W - pd - 2*cr, pd), (pd, WIN_H - pd - 2*cr), (WIN_W - pd - 2*cr, WIN_H - pd - 2*cr)]:
        bg_c.create_oval(corner_x, corner_y, corner_x + 2*cr, corner_y + 2*cr, fill="#2e2e3e", outline="#505068", width=1)
    bg_c.create_rectangle(pd + cr, pd, WIN_W - pd - cr, WIN_H - pd, fill="#2e2e3e", outline="")
    bg_c.create_rectangle(pd, pd + cr, WIN_W - pd, WIN_H - pd - cr, fill="#2e2e3e", outline="")

    cfg = _load_config()
    api_key_var = tk.StringVar(value=(cfg.get("claude_api_key") or cfg.get("api_key") or "").strip() or os.environ.get("DASHSCOPE_API_KEY", ""))
    model_var = tk.StringVar(value=cfg.get("claude_model") or cfg.get("model") or QWEN_MODELS[0])
    use_local_var = tk.BooleanVar(value=cfg.get("use_local_model", True))
    if model_var.get() not in QWEN_MODELS:
        model_var.set(QWEN_MODELS[0])

    y_pos = 24
    # Title + close
    tk.Label(win, text="Settings", font=("Segoe UI", 16, "bold"), bg="#2e2e3e", fg="#e0e0ee").place(x=28, y=y_pos)
    tk.Button(win, text="X", font=("Segoe UI", 10, "bold"), bg="#2e2e3e", fg="#888", activebackground="#444", relief=tk.FLAT, bd=0, cursor="hand2", command=win.destroy).place(x=WIN_W - 38, y=y_pos + 2, width=24, height=24)
    y_pos += 50

    # Local model toggle
    tk.Checkbutton(win, text="Use local model (free, no key)", variable=use_local_var, bg="#2e2e3e", fg="#c0c0d0", selectcolor="#1e1e2e", activebackground="#2e2e3e", activeforeground="#c0c0d0", font=("Segoe UI", 10)).place(x=28, y=y_pos)
    y_pos += 40

    # API key
    tk.Label(win, text="API Key", font=("Segoe UI", 10), bg="#2e2e3e", fg="#9090a0").place(x=28, y=y_pos)
    y_pos += 24
    tk.Entry(win, textvariable=api_key_var, show="*", font=("Consolas", 11), bg="#1e1e2e", fg="#d0d0e0", insertbackground="#fff", relief=tk.FLAT, highlightthickness=1, highlightcolor="#606078", highlightbackground="#3a3a4c").place(x=28, y=y_pos, width=WIN_W - 56, height=34)
    y_pos += 48

    # Model
    tk.Label(win, text="Model", font=("Segoe UI", 10), bg="#2e2e3e", fg="#9090a0").place(x=28, y=y_pos)
    y_pos += 24
    model_combo = ttk.Combobox(win, textvariable=model_var, values=QWEN_MODELS, state="readonly", width=36)
    model_combo.place(x=28, y=y_pos, width=WIN_W - 56, height=32)
    y_pos += 50

    # ── Accessibility: Display Contrast ──
    tk.Label(win, text="Accessibility", font=("Segoe UI", 12, "bold"), bg="#2e2e3e", fg="#e0e0ee").place(x=28, y=y_pos)
    y_pos += 28
    tk.Label(win, text="Display Contrast", font=("Segoe UI", 10), bg="#2e2e3e", fg="#9090a0").place(x=28, y=y_pos)

    saved_contrast = cfg.get("display_contrast", 50)
    contrast_var = tk.IntVar(value=saved_contrast)

    contrast_val_lbl = tk.Label(win, text=f"{saved_contrast}%", font=("Segoe UI", 10, "bold"), bg="#2e2e3e", fg="#c0c0d0")
    contrast_val_lbl.place(x=WIN_W - 70, y=y_pos)
    y_pos += 26

    preview_cv = tk.Canvas(win, bg="#2e2e3e", highlightthickness=0, bd=0, height=30)
    preview_cv.place(x=28, y=y_pos + 34, width=WIN_W - 56, height=30)

    def _update_contrast_preview(val=None):
        v = contrast_var.get()
        contrast_val_lbl.config(text=f"{v}%")
        # Map 0-100 to a range of greys for preview: 0% = #000, 100% = #fff
        lo = int(30 + (v / 100) * 80)
        hi = int(80 + (v / 100) * 175)
        bg_hex = f"#{lo:02x}{lo:02x}{lo:02x}"
        fg_hex = f"#{hi:02x}{hi:02x}{hi:02x}"
        preview_cv.delete("all")
        preview_cv.create_rectangle(0, 0, (WIN_W - 56) // 2, 30, fill=bg_hex, outline="")
        preview_cv.create_rectangle((WIN_W - 56) // 2, 0, WIN_W - 56, 30, fill=fg_hex, outline="")
        preview_cv.create_text((WIN_W - 56) // 4, 15, text="Aa Dark", fill=fg_hex, font=("Segoe UI", 10))
        preview_cv.create_text(3 * (WIN_W - 56) // 4, 15, text="Aa Light", fill=bg_hex, font=("Segoe UI", 10))

    contrast_slider = tk.Scale(win, from_=0, to=100, orient=tk.HORIZONTAL, variable=contrast_var,
                               bg="#2e2e3e", fg="#c0c0d0", troughcolor="#1e1e2e", highlightthickness=0,
                               activebackground="#6a6a80", sliderrelief=tk.FLAT, bd=0, showvalue=False,
                               command=_update_contrast_preview)
    contrast_slider.place(x=28, y=y_pos, width=WIN_W - 56)
    y_pos += 70

    _update_contrast_preview()

    # Save button
    def _save_and_close():
        _save_config(api_key_var.get().strip(), model_var.get(), use_local_var.get(),
                     display_contrast=contrast_var.get())
        win.destroy()

    tk.Button(win, text="Save", font=("Segoe UI", 12, "bold"), bg="#4CAF50", fg="#fff", activebackground="#388E3C", relief=tk.FLAT, cursor="hand2", command=_save_and_close).place(x=28, y=y_pos, width=WIN_W - 56, height=40)

    # Drag support
    drag_data = {"x": 0, "y": 0}
    def _start_drag(e):
        drag_data["x"] = e.x
        drag_data["y"] = e.y
    def _do_drag(e):
        nx = win.winfo_x() + e.x - drag_data["x"]
        ny = win.winfo_y() + e.y - drag_data["y"]
        win.geometry(f"+{nx}+{ny}")
    bg_c.bind("<Button-1>", _start_drag)
    bg_c.bind("<B1-Motion>", _do_drag)


def _open_task_window(parent):
    """Glass-style floating task bar: type a task, hit Go, see status. Hooks into agent backend."""
    TRANS_TASK = "#020202"
    BAR_W, BAR_H = 600, 200

    win = tk.Toplevel(parent)
    win.title("")
    win.overrideredirect(1)
    win.attributes("-topmost", True)
    win.configure(bg=TRANS_TASK)
    try:
        win.attributes("-transparentcolor", TRANS_TASK)
    except tk.TclError:
        try:
            win.attributes("-alpha", 0.95)
        except tk.TclError:
            pass

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - BAR_W) // 2)
    y = max(0, (sh - BAR_H) // 2 - 100)
    win.geometry(f"{BAR_W}x{BAR_H}+{x}+{y}")
    win.resizable(False, False)

    # Glass pill background
    pill = tk.Canvas(win, bg=TRANS_TASK, highlightthickness=0, bd=0, width=BAR_W, height=BAR_H)
    pill.pack(fill=tk.BOTH, expand=True)
    # Rounded-rect pill (approximated with oval + rect)
    pad = 6
    pr = 30  # corner radius
    pill.create_oval(pad, pad, pad + 2 * pr, pad + 2 * pr, fill="#3a3a4c", outline="#6a6a80", width=1)
    pill.create_oval(BAR_W - pad - 2 * pr, pad, BAR_W - pad, pad + 2 * pr, fill="#3a3a4c", outline="#6a6a80", width=1)
    pill.create_oval(pad, BAR_H - pad - 2 * pr, pad + 2 * pr, BAR_H - pad, fill="#3a3a4c", outline="#6a6a80", width=1)
    pill.create_oval(BAR_W - pad - 2 * pr, BAR_H - pad - 2 * pr, BAR_W - pad, BAR_H - pad, fill="#3a3a4c", outline="#6a6a80", width=1)
    pill.create_rectangle(pad + pr, pad, BAR_W - pad - pr, BAR_H - pad, fill="#3a3a4c", outline="")
    pill.create_rectangle(pad, pad + pr, BAR_W - pad, BAR_H - pad - pr, fill="#3a3a4c", outline="")
    # Top border line
    pill.create_line(pad + pr, pad, BAR_W - pad - pr, pad, fill="#6a6a80")
    # Bottom border line
    pill.create_line(pad + pr, BAR_H - pad, BAR_W - pad - pr, BAR_H - pad, fill="#6a6a80")

    # --- Widgets inside the pill ---
    # Status label
    status_var = tk.StringVar(value="Ready")
    status_lbl = tk.Label(win, textvariable=status_var, font=("Segoe UI", 10), bg="#3a3a4c", fg="#aaaabc")
    status_lbl.place(x=30, y=18, width=BAR_W - 60, height=22)

    # Task text entry
    task_entry = tk.Entry(
        win, font=("Segoe UI", 14), bg="#2a2a3a", fg="#e8e8f0",
        insertbackground="#fff", relief=tk.FLAT,
        highlightthickness=1, highlightcolor="#8888a0", highlightbackground="#4a4a5c")
    task_entry.place(x=30, y=48, width=BAR_W - 140, height=42)
    task_entry.insert(0, "Type a task...")
    task_entry.bind("<FocusIn>", lambda e: task_entry.delete(0, tk.END) if task_entry.get() == "Type a task..." else None)

    # Go / Stop button
    running = [False]
    agent_ref = [None]

    go_btn = tk.Button(
        win, text="Go", font=("Segoe UI", 13, "bold"),
        bg="#4CAF50", fg="#fff", activebackground="#388E3C",
        relief=tk.FLAT, cursor="hand2", width=6)
    go_btn.place(x=BAR_W - 100, y=48, width=60, height=42)

    # Status line at bottom
    action_var = tk.StringVar(value="")
    action_lbl = tk.Label(win, textvariable=action_var, font=("Consolas", 10), bg="#3a3a4c", fg="#FFD700", anchor="w")
    action_lbl.place(x=30, y=100, width=BAR_W - 60, height=22)

    # Close button (X)
    close_btn = tk.Button(
        win, text="X", font=("Segoe UI", 10, "bold"),
        bg="#3a3a4c", fg="#888", activebackground="#555",
        relief=tk.FLAT, cursor="hand2", command=win.destroy, bd=0)
    close_btn.place(x=BAR_W - 36, y=14, width=24, height=24)

    # Hidden log + screenshot (invisible but needed by AgentGUI methods)
    hidden_frame = tk.Frame(win, bg=TRANS_TASK)

    class BarAgent(AgentGUI):
        """Minimal agent that runs behind the glass task bar."""
        def __init__(self):
            self.root = hidden_frame
            self.task_only = True

            cfg = _load_config()
            default_key = (cfg.get("claude_api_key") or cfg.get("api_key") or "").strip() or os.environ.get("DASHSCOPE_API_KEY", "")
            default_model = cfg.get("claude_model") or cfg.get("model") or QWEN_MODELS[0]
            if default_model not in QWEN_MODELS:
                default_model = QWEN_MODELS[0]
            self.claude_api_key_var = tk.StringVar(value=default_key)
            self.claude_model_var = tk.StringVar(value=default_model)
            self.use_local_var = tk.BooleanVar(value=cfg.get("use_local_model", True))
            self.turn_var = tk.StringVar(value="white")
            self.conf_var = tk.DoubleVar(value=0.30)
            self.depth_var = tk.IntVar(value=18)
            self.interval_var = tk.DoubleVar(value=3.0)
            self.click_delay_var = tk.DoubleVar(value=0.15)

            self._running = False
            self._mode = None
            self._thread = None
            self._screen_action_count = 0
            self._hide_event = threading.Event()
            self._show_event = threading.Event()
            self.voice_enabled = False
            self._voice_thread = None
            self._voice_stop_event = threading.Event()

            self.chess = ChessEngine(log_fn=self.log)

            # Invisible widgets that the inherited methods reference
            self.main_frame = tk.Frame(hidden_frame)
            self.task_text = tk.Text(hidden_frame, height=1, width=1)
            self.start_btn = tk.Button(hidden_frame)
            self.status_label = tk.Label(hidden_frame)
            self.action_label = tk.Label(hidden_frame)
            self.eval_label = tk.Label(hidden_frame)
            self.stats_label = tk.Label(hidden_frame)
            self.screenshot_label = tk.Label(hidden_frame)
            self.log_text = scrolledtext.ScrolledText(hidden_frame, height=1, width=1)
            for tag_name, color in [("info", "#4CAF50"), ("error", "#f44336"), ("action", "#2196F3"), ("warning", "#FF9800"), ("header", "#e94560"), ("piece", "#80CBC4"), ("move", "#FFD700"), ("board", "#B0BEC5"), ("dim", "#666666"), ("thought", "#80CBC4"), ("result", "#FFD700")]:
                self.log_text.tag_config(tag_name, foreground=color)

            threading.Thread(target=self._load_chess, daemon=True).start()
            if self.use_local_var.get():
                def _load_local_vl():
                    try:
                        from local_qwen_vl import load_model
                        load_model()
                    except Exception:
                        pass
                threading.Thread(target=_load_local_vl, daemon=True).start()

        def log(self, msg, tag=""):
            ts = time.strftime("%H:%M:%S")
            # Show last log line in the status bar
            short = msg.strip()[:80]
            win.after(0, lambda: action_var.set(f"[{ts}] {short}"))

        def _hide_and_signal(self):
            win.withdraw()
            win.update()
            self._hide_event.set()

        def _show_and_signal(self):
            win.deiconify()
            win.update()
            self._show_event.set()

        def _hide_for_screenshot(self):
            self._hide_event.clear()
            win.after(0, self._hide_and_signal)
            self._hide_event.wait(timeout=2.0)

        def _show_after_screenshot(self):
            self._show_event.clear()
            win.after(0, self._show_and_signal)
            self._show_event.wait(timeout=2.0)

    agent = BarAgent()
    agent_ref[0] = agent

    def _go_or_stop():
        if running[0]:
            running[0] = False
            agent._running = False
            go_btn.config(text="Go", bg="#4CAF50", activebackground="#388E3C")
            status_var.set("Stopped")
        else:
            task = task_entry.get().strip()
            if not task or task == "Type a task...":
                status_var.set("Enter a task first")
                return
            running[0] = True
            go_btn.config(text="Stop", bg="#e94560", activebackground="#c0384d")
            status_var.set("Running...")
            agent.task_text.delete("1.0", tk.END)
            agent.task_text.insert("1.0", task)
            agent._running = True
            agent.start_btn.config(text="Stop")
            threading.Thread(target=_run_agent_task, args=(task,), daemon=True).start()

    def _run_agent_task(task):
        try:
            agent._start_with_claude(task)
        except Exception as e:
            win.after(0, lambda: action_var.set(f"Error: {e}"))
        finally:
            running[0] = False
            agent._running = False
            win.after(0, lambda: go_btn.config(text="Go", bg="#4CAF50", activebackground="#388E3C"))
            win.after(0, lambda: status_var.set("Done"))

    go_btn.config(command=_go_or_stop)
    task_entry.bind("<Return>", lambda e: _go_or_stop())

    # Drag support (since borderless)
    drag_data = {"x": 0, "y": 0}
    def _start_drag(e):
        drag_data["x"] = e.x
        drag_data["y"] = e.y
    def _do_drag(e):
        dx = e.x - drag_data["x"]
        dy = e.y - drag_data["y"]
        nx = win.winfo_x() + dx
        ny = win.winfo_y() + dy
        win.geometry(f"+{nx}+{ny}")
    pill.bind("<Button-1>", _start_drag)
    pill.bind("<B1-Motion>", _do_drag)

    task_entry.focus_set()


def _run_tkinter_floating_dot():
    """Apple Intelligence-style floating glass dot. See-through, expands to 5 glass dots."""
    TRANS = "#010101"
    DOT_SIZE = 160
    EXPANDED_SIZE = 500
    RGB_THICKNESS = 6
    RGB_SPEED_MS = 50

    root = tk.Tk()
    root.title("")
    root.overrideredirect(1)
    root.attributes("-topmost", True)
    root.configure(bg=TRANS)
    try:
        root.attributes("-transparentcolor", TRANS)
    except tk.TclError:
        try:
            root.attributes("-alpha", 0.93)
        except tk.TclError:
            pass

    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    expanded = [False]
    voice_stop = threading.Event()
    voice_thread = [None]

    # ── RGB screen-border glow overlay ──
    import colorsys

    rgb_win = tk.Toplevel(root)
    rgb_win.title("")
    rgb_win.overrideredirect(1)
    rgb_win.attributes("-topmost", True)
    rgb_win.geometry(f"{sw}x{sh}+0+0")
    rgb_trans = "#020202"
    rgb_win.configure(bg=rgb_trans)
    try:
        rgb_win.attributes("-transparentcolor", rgb_trans)
        rgb_win.attributes("-alpha", 0.4)
    except tk.TclError:
        try:
            rgb_win.attributes("-alpha", 0.4)
        except tk.TclError:
            pass
    rgb_cv = tk.Canvas(rgb_win, bg=rgb_trans, highlightthickness=0, bd=0,
                       width=sw, height=sh)
    rgb_cv.pack(fill=tk.BOTH, expand=True)
    rgb_cv.config(cursor="arrow")
    rgb_win.attributes("-disabled", True)
    rgb_win.withdraw()

    rgb_hue = [0.0]
    rgb_active = [False]
    rgb_fading = [None]
    rgb_opacity = [0.0]
    rgb_job = [None]
    rgb_photo = [None]
    RGB_FADE_STEP = 0.12
    RGB_GLOW_DEPTH = 50

    NEON_COLORS = [
        (255, 0, 102),    # hot pink
        (0, 255, 255),    # cyan
        (57, 255, 20),    # neon green
        (255, 0, 255),    # magenta
        (0, 191, 255),    # deep sky blue
        (255, 255, 0),    # yellow
        (255, 105, 180),  # pink
        (0, 255, 128),    # spring green
    ]

    RGB_RENDER_SCALE = 2
    rw, rh = sw // RGB_RENDER_SCALE, sh // RGB_RENDER_SCALE

    _yy, _xx = np.mgrid[0:rh, 0:rw]
    _dt = _yy.astype(np.float32)
    _db = (rh - 1 - _yy).astype(np.float32)
    _dl = _xx.astype(np.float32)
    _dr = (rw - 1 - _xx).astype(np.float32)
    _dist = np.minimum(np.minimum(_dt, _db), np.minimum(_dl, _dr))
    _glow_d = RGB_GLOW_DEPTH / RGB_RENDER_SCALE
    _mask = _dist < _glow_d
    _fade_raw = np.where(_mask, ((_glow_d - _dist) / _glow_d) ** 2.0, 0.0).astype(np.float32)
    _is_top = (_dt <= _db) & (_dt <= _dl) & (_dt <= _dr)
    _is_bot = (~_is_top) & (_db <= _dl) & (_db <= _dr)
    _is_left = (~_is_top) & (~_is_bot) & (_dl <= _dr)
    _pos_base = np.zeros((rh, rw), dtype=np.float32)
    _pos_base[_is_top] = (_xx[_is_top] / rw)
    _pos_base[_is_bot] = (0.5 + _xx[_is_bot] / rw)
    _pos_base[_is_left] = (0.75 + _yy[_is_left] / rh)
    _is_right = ~(_is_top | _is_bot | _is_left)
    _pos_base[_is_right] = (0.25 + _yy[_is_right] / rh)

    _NEON_LUT = np.zeros((256, 3), dtype=np.float32)
    n_colors = len(NEON_COLORS)
    for i in range(256):
        t_val = i / 256.0 * n_colors
        i0 = int(t_val) % n_colors
        i1 = (i0 + 1) % n_colors
        f = t_val - int(t_val)
        for ch in range(3):
            _NEON_LUT[i, ch] = NEON_COLORS[i0][ch] + (NEON_COLORS[i1][ch] - NEON_COLORS[i0][ch]) * f

    _bg_arr = np.full((rh, rw, 3), 2, dtype=np.float32)
    _fade3 = _fade_raw[:, :, np.newaxis]

    def _build_rgb_array(hue_offset, opacity):
        """Render neon border with numpy -- returns uint8 array at half-res."""
        fade3 = _fade3 * opacity
        pos = (_pos_base + hue_offset) % 1.0
        lut_idx = np.clip((pos * 255).astype(np.int32), 0, 255)
        neon = _NEON_LUT[lut_idx]
        out = neon * fade3 + _bg_arr * (1.0 - fade3)
        result = np.clip(out, 3, 255).astype(np.uint8)
        result[~_mask] = 2
        return result

    rgb_img_id = [None]

    def _draw_rgb_border():
        op = rgb_opacity[0]
        if op <= 0.01:
            if rgb_img_id[0]:
                rgb_cv.delete(rgb_img_id[0])
                rgb_img_id[0] = None
            return
        arr = _build_rgb_array(rgb_hue[0], op)
        img = Image.fromarray(arr, "RGB").resize((sw, sh), Image.NEAREST)
        rgb_photo[0] = ImageTk.PhotoImage(img)
        if rgb_img_id[0]:
            rgb_cv.itemconfigure(rgb_img_id[0], image=rgb_photo[0])
        else:
            rgb_img_id[0] = rgb_cv.create_image(0, 0, anchor="nw", image=rgb_photo[0])

    def _rgb_tick():
        if not rgb_active[0] and rgb_fading[0] != "out":
            return

        # Handle fade transitions
        if rgb_fading[0] == "in":
            rgb_opacity[0] = min(1.0, rgb_opacity[0] + RGB_FADE_STEP)
            if rgb_opacity[0] >= 1.0:
                rgb_fading[0] = None
        elif rgb_fading[0] == "out":
            rgb_opacity[0] = max(0.0, rgb_opacity[0] - RGB_FADE_STEP)
            if rgb_opacity[0] <= 0.0:
                rgb_fading[0] = None
                if rgb_img_id[0]:
                    rgb_cv.delete(rgb_img_id[0])
                    rgb_img_id[0] = None
                rgb_win.withdraw()
                return

        rgb_hue[0] = (rgb_hue[0] + 0.012) % 1.0
        _draw_rgb_border()
        rgb_job[0] = root.after(RGB_SPEED_MS, _rgb_tick)

    def _start_rgb():
        if rgb_active[0] and rgb_fading[0] != "out":
            return
        rgb_active[0] = True
        rgb_fading[0] = "in"
        rgb_opacity[0] = 0.0
        rgb_win.deiconify()
        rgb_win.lift()
        root.lift()
        if rgb_job[0]:
            root.after_cancel(rgb_job[0])
        _rgb_tick()

    def _stop_rgb():
        rgb_active[0] = False
        if rgb_fading[0] == "out":
            return
        rgb_fading[0] = "out"
        if not rgb_job[0]:
            _rgb_tick()

    canvas = tk.Canvas(root, bg=TRANS, highlightthickness=0, bd=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.config(cursor="hand2")

    # ── Drag support (screen-coords based, no winfo during drag) ──
    drag = {"sx": 0, "sy": 0, "wx": 0, "wy": 0, "dragging": False}

    def _drag_start(e):
        drag["sx"] = e.x_root
        drag["sy"] = e.y_root
        drag["wx"] = root.winfo_x()
        drag["wy"] = root.winfo_y()
        drag["dragging"] = False

    def _drag_motion(e):
        dx = abs(e.x_root - drag["sx"])
        dy = abs(e.y_root - drag["sy"])
        if dx > 8 or dy > 8:
            drag["dragging"] = True
        if drag["dragging"]:
            nx = drag["wx"] + (e.x_root - drag["sx"])
            ny = drag["wy"] + (e.y_root - drag["sy"])
            root.geometry(f"+{nx}+{ny}")

    canvas.bind("<Button-1>", _drag_start)
    canvas.bind("<B1-Motion>", _drag_motion)

    def _position(size, center=True):
        if center:
            x = max(0, (sw - size) // 2)
            y = max(0, (sh - size) // 2)
            root.geometry(f"{size}x{size}+{x}+{y}")
        else:
            root.geometry(f"{size}x{size}")

    # No animation tick for dots -- they're pre-rendered and static

    def _cleanup_and_quit(e=None):
        rgb_active[0] = False
        rgb_fading[0] = None
        if rgb_job[0]:
            root.after_cancel(rgb_job[0])
            rgb_job[0] = None
        root.destroy()

    dot_photos = {}
    DOT_SS = 2

    def _render_dot_image(r, bright):
        """Render a smooth anti-aliased glass dot using PIL. Called once at startup."""
        trans = (1, 1, 1)
        ss = DOT_SS
        sz = int(r * 2 + 20)
        big = sz * ss
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c = big // 2

        ring_defs = [
            (0, 4, (16, 16, 24), 255),
            (4, 4, (106, 176, 230) if bright else (72, 136, 184), 255),
            (9, 3, (224, 232, 240) if bright else (176, 184, 196), 240),
            (13, 2, (26, 26, 40), 180),
            (16, 3, (112, 168, 216) if bright else (80, 144, 192), 220),
            (20, 2, (240, 244, 255) if bright else (224, 232, 240), 200),
        ]

        for inset, w, color_rgb, alpha in ring_defs:
            ri = max(int((r - inset) * ss), 2)
            draw.ellipse([c - ri, c - ri, c + ri, c + ri],
                        fill=None, outline=color_rgb + (alpha,), width=w * ss)

        ci = max(int((r - 24) * ss), 3)
        fc = (208, 220, 232, 220) if bright else (80, 104, 120, 220)
        oc = (112, 168, 216, 180) if bright else (56, 80, 104, 180)
        draw.ellipse([c - ci, c - ci, c + ci, c + ci], fill=fc, outline=oc, width=2 * ss)

        small = img.resize((sz, sz), Image.LANCZOS)
        bg = Image.new("RGB", (sz, sz), trans)
        bg.paste(small, mask=small.split()[3])
        return bg

    def _prerender_dots():
        """Pre-render all dot sizes at startup. Returns dict of tag -> PhotoImage."""
        for tag, r, bright in [("hub_main", 52, True), ("sub_dim", 78, False)]:
            img = _render_dot_image(r, bright)
            dot_photos[tag] = ImageTk.PhotoImage(img)
            dot_photos[f"{tag}_sz"] = img.width

    _prerender_dots()

    def _draw_glass_circle(cx, cy, r, tag, bright=False):
        """Place a pre-rendered dot image on the canvas (instant)."""
        key = "hub_main" if bright else "sub_dim"
        photo = dot_photos[key]
        sz = dot_photos[f"{key}_sz"]
        canvas.create_image(cx - sz // 2, cy - sz // 2, anchor="nw", image=photo, tags=(tag,))

    def redraw():
        canvas.delete("all")
        exp = expanded[0]
        size = EXPANDED_SIZE if exp else DOT_SIZE
        cx, cy = size // 2, size // 2
        orb_r = 52

        def _click_if_not_drag(callback):
            """Only fire callback if the user clicked, not dragged."""
            def handler(e):
                if not drag["dragging"]:
                    callback()
            return handler

        def _outlined_text(x, y, text, size, tag):
            """Text with dark outline so it reads on any background."""
            f = ("Segoe UI", size, "bold")
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                canvas.create_text(x+dx, y+dy, text=text, fill="#000000", font=f, tags=(tag,))
            canvas.create_text(x, y, text=text, fill="#e8e8e8", font=f, tags=(tag,))

        if not exp:
            _draw_glass_circle(cx, cy, orb_r, "hub_main", bright=True)
            return

        # --- Expanded: center orb + 3 radial glass dots ---
        ring_r = 155
        sub_r = 78
        items = [
            ("Settings", "settings"),
            ("Task", "task"),
            ("Mic", "mic"),
        ]

        for idx, (label, key) in enumerate(items):
            angle = (2 * math.pi * idx / len(items)) - math.pi / 2
            sx = int(cx + ring_r * math.cos(angle))
            sy = int(cy + ring_r * math.sin(angle))
            tag = f"sub_{key}"
            _draw_glass_circle(sx, sy, sub_r, tag)
            _outlined_text(sx, sy, label, 12, tag)

        # Center orb (on top)
        _draw_glass_circle(cx, cy, orb_r, "hub_main", bright=True)

    def do_expand():
        expanded[0] = True
        old_size = DOT_SIZE
        new_size = EXPANDED_SIZE
        cx = root.winfo_x() + old_size // 2
        cy = root.winfo_y() + old_size // 2
        nx = max(0, min(cx - new_size // 2, sw - new_size))
        ny = max(0, min(cy - new_size // 2, sh - new_size))
        root.geometry(f"{new_size}x{new_size}+{nx}+{ny}")
        root.update_idletasks()
        redraw()
        _start_rgb()

    def do_collapse():
        expanded[0] = False
        _stop_rgb()
        old_size = EXPANDED_SIZE
        new_size = DOT_SIZE
        cx = root.winfo_x() + old_size // 2
        cy = root.winfo_y() + old_size // 2
        nx = max(0, min(cx - new_size // 2, sw - new_size))
        ny = max(0, min(cy - new_size // 2, sh - new_size))
        root.geometry(f"{new_size}x{new_size}+{nx}+{ny}")
        root.update_idletasks()
        redraw()

    def on_choice(key):
        do_collapse()
        if key == "settings":
            _open_settings_window(root)
        elif key == "task":
            _open_task_window(root)
        elif key == "mic":
            if voice_thread[0] and voice_thread[0].is_alive():
                voice_stop.set()
            else:
                voice_stop.clear()
                voice_thread[0] = threading.Thread(
                    target=_run_voice_listener,
                    args=(lambda: root.after(0, do_expand), voice_stop),
                    daemon=True)
                voice_thread[0].start()

    def _on_canvas_release(e):
        """Handle all clicks by position -- works even when items are redrawn mid-click."""
        if drag["dragging"]:
            return
        size = EXPANDED_SIZE if expanded[0] else DOT_SIZE
        cx, cy = size // 2, size // 2
        orb_r = 52

        # Center dot
        dist_sq = (e.x - cx) ** 2 + (e.y - cy) ** 2
        if dist_sq <= (orb_r + 20) ** 2:
            if expanded[0]:
                do_collapse()
            else:
                do_expand()
            return

        # Sub-dots (only when expanded)
        if expanded[0]:
            ring_r = 155
            sub_r = 78
            items = [
                ("Settings", "settings"),
                ("Task", "task"),
                ("Mic", "mic"),
            ]
            for idx, (label, key) in enumerate(items):
                angle = (2 * math.pi * idx / len(items)) - math.pi / 2
                sx = cx + ring_r * math.cos(angle)
                sy = cy + ring_r * math.sin(angle)
                d = (e.x - sx) ** 2 + (e.y - sy) ** 2
                if d <= (sub_r + 15) ** 2:
                    on_choice(key)
                    return

    def _on_canvas_rclick(e):
        _cleanup_and_quit()

    canvas.bind("<ButtonRelease-1>", _on_canvas_release)
    canvas.bind("<Button-3>", _on_canvas_rclick)

    root.bind("<Escape>", _cleanup_and_quit)

    _position(DOT_SIZE)
    redraw()
    root.mainloop()


# Kivy glass-dot app (borderless, always-on-top, on-screen overlay)
def _run_kivy_dot_app():
    # Set before any Window is created (required for borderless / position)
    try:
        from kivy.config import Config
        Config.set("graphics", "borderless", "1")
        Config.set("graphics", "width", "420")
        Config.set("graphics", "height", "420")
    except Exception:
        pass
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.uix.widget import Widget
    from kivy.uix.floatlayout import FloatLayout
    from kivy.uix.label import Label
    from kivy.graphics import Color, Ellipse
    from kivy.clock import Clock
    from kivy.uix.button import Button

    class GlassDotWidget(Widget):
        def __init__(self, on_choice, **kwargs):
            super().__init__(**kwargs)
            self.expanded = False
            self.on_choice = on_choice
            self.radius = 28
            self._voice_stop = threading.Event()
            self._voice_thread = None
            self.voice_enabled = False
            self.bind(size=self._redraw, pos=self._redraw)
            # Always-visible center label so the dot is easy to spot
            self._label = Label(
                text="Vibe\nGamer",
                font_size="14sp",
                bold=True,
                color=(0.95, 0.95, 0.98, 1),
                halign="center",
                valign="middle",
                size_hint=(None, None),
                size=(120, 60),
                text_size=(120, 60),
            )
            self.add_widget(self._label)

        def _redraw(self, *args):
            self.canvas.clear()
            w, h = self.size
            if w <= 0 or h <= 0:
                return
            cx, cy = w / 2, h / 2
            r = self.radius
            glass_dark = (0.235, 0.235, 0.29, 1)
            glass_mid = (0.29, 0.29, 0.36, 1)
            glass_hl = (0.43, 0.43, 0.51, 1)
            glass_edge = (0.54, 0.54, 0.62, 1)
            with self.canvas:
                # Expanded: frosted sheet and ring first (behind orb)
                if self.expanded:
                    ring_r = r * 2.4
                    Color(*glass_dark)
                    Ellipse(pos=(cx - ring_r - r - 12, cy - ring_r - r - 12),
                            size=(2 * (ring_r + r + 12), 2 * (ring_r + r + 12)))
                    Color(*glass_mid)
                    Ellipse(pos=(cx - ring_r, cy - ring_r), size=(2 * ring_r, 2 * ring_r))
                # Center orb (on top)
                Color(*glass_edge)
                Ellipse(pos=(cx - r - 2, cy - r - 2), size=(2 * (r + 2), 2 * (r + 2)))
                Color(*glass_dark)
                Ellipse(pos=(cx - r, cy - r), size=(2 * r, 2 * r))
                Color(*glass_mid)
                Ellipse(pos=(cx - r + 8, cy - r + 8), size=(2 * (r - 8), 2 * (r - 8)))
                Color(*glass_hl)
                Ellipse(pos=(cx - r / 3 - r / 3, cy - r / 3 - r / 3), size=(2 * r / 3, 2 * r / 3))
            # Keep label centered on orb
            self._label.pos = (cx - 60, cy - 30)
            self._label.text = "Vibe\nGamer" if not self.expanded else "Click to close"

        def _clear_children(self):
            for c in list(self.children):
                self.remove_widget(c)

        def _build_buttons(self):
            self._clear_children()
            if not self.expanded:
                return
            w, h = self.size
            cx, cy = w / 2, h / 2
            ring_r = self.radius * 2.4
            items = [
                ("Settings", "settings"), ("Task", "task"), ("Mic", "mic"),
                ("Screen", "screen"), ("History", "history"),
            ]
            for idx, (label, key) in enumerate(items):
                angle = (2 * math.pi * idx / len(items)) - math.pi / 2
                sx = cx + ring_r * math.cos(angle)
                sy = cy + ring_r * math.sin(angle)
                btn = Button(
                    text=label, size_hint=(None, None), size=(80, 36),
                    pos=(sx - 40, sy - 18), background_color=(0.27, 0.27, 0.33, 0.95),
                    color=(0.9, 0.9, 0.93, 1),
                )
                btn.key = key
                btn.bind(on_press=lambda b, k=key: self._on_sub(b, k))
                self.add_widget(btn)

        def _on_sub(self, btn, key):
            self.on_choice(key)

        def on_touch_down(self, touch):
            if not self.collide_point(*touch.pos):
                return False
            w, h = self.size
            cx, cy = w / 2, h / 2
            dx, dy = touch.pos[0] - cx, touch.pos[1] - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= self.radius + 4:
                self.expanded = not self.expanded
                self._redraw()
                self._build_buttons()
                return True
            return super().on_touch_down(touch)

    class VibeGamerDotApp(App):
        def build(self):
            Window.borderless = True
            Window.always_on_top = True
            Window.size = (420, 420)
            Window.clearcolor = (0.1, 0.1, 0.12, 1)  # match dark hub
            Clock.schedule_once(self._position_window, 0.1)
            root = FloatLayout(size=(420, 420))
            self.dot = GlassDotWidget(size_hint=(1, 1), size=(420, 420), on_choice=self._on_choice)
            root.add_widget(self.dot)
            # Redraw after layout so size is correct
            Clock.schedule_once(lambda dt: self.dot._redraw(), 0.05)
            return root

        def _position_window(self, dt):
            # Place floating dot where it's always visible: center of primary screen
            try:
                sw, sh = pyautogui.size()
                # Center of screen so you can't miss it (then move to corner if you prefer)
                win_w, win_h = 420, 420
                Window.left = max(0, (sw - win_w) // 2)
                Window.top = max(0, (sh - win_h) // 2)
                # Bring to front so it's not behind other windows
                if hasattr(Window, "raise_window"):
                    Window.raise_window()
            except Exception:
                Window.left = 100
                Window.top = 100

        def _on_choice(self, key):
            if key == "task" or key == "settings" or key == "screen" or key == "history":
                _launch_task_panel()
            elif key == "mic":
                self._toggle_voice()

        def _toggle_voice(self):
            self.dot.voice_enabled = not self.dot.voice_enabled
            if self.dot.voice_enabled:
                self.dot._voice_stop.clear()
                self.dot._voice_thread = threading.Thread(
                    target=_run_voice_listener,
                    args=(lambda: Clock.schedule_once(self._expand_dot), self.dot._voice_stop),
                    daemon=True,
                )
                self.dot._voice_thread.start()
            else:
                self.dot._voice_stop.set()

        def _expand_dot(self, dt):
            self.dot.expanded = True
            self.dot._redraw()
            self.dot._build_buttons()

    VibeGamerDotApp().run()


def main():
    _run_tkinter_floating_dot()


if __name__ == "__main__":
    main()
