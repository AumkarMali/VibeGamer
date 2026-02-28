"""
pywinauto-based screen control.

Exposes UI elements to the agent and executes element-based instructions
(CLICK, DOUBLE_CLICK, TYPE_IN, MENU, KEYS, TYPE_TEXT) instead of raw mouse/keyboard.
Windows only.
"""
import re
import sys

if sys.platform != "win32":
    _PYWINAUTO_AVAILABLE = False
    Application = Desktop = None
else:
    try:
        from pywinauto import Application, Desktop
        from pywinauto.findwindows import ElementNotFoundError
        import ctypes
        _PYWINAUTO_AVAILABLE = True
    except ImportError:
        _PYWINAUTO_AVAILABLE = False
        Application = Desktop = None


# Control types that are typically clickable/actionable
CLICKABLE_TYPES = {"Button", "ListItem", "MenuItem", "Hyperlink", "DataItem", "TreeItem", "TabItem"}


def get_ui_elements(max_items: int = 80) -> list:
    """
    Get list of actionable UI elements from the foreground window and taskbar.
    Returns [{"name": str, "control_type": str}, ...] for the agent to choose from.
    """
    if not _PYWINAUTO_AVAILABLE:
        return []
    out = []
    seen = set()
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            app = Application(backend="uia").connect(handle=hwnd)
            win = app.window(handle=hwnd)
            for ctrl in win.descendants():
                try:
                    name = (ctrl.window_text() or "").strip()
                    if not name or len(name) > 60:
                        continue
                    ctype = getattr(ctrl.element_info, "control_type", None) or "Unknown"
                    key = (name, ctype)
                    if key in seen:
                        continue
                    seen.add(key)
                    if ctype in CLICKABLE_TYPES or "Edit" in ctype or "Combo" in ctype:
                        out.append({"name": name, "control_type": str(ctype)})
                        if len(out) >= max_items:
                            return out
                except Exception:
                    continue
        # Also try taskbar / desktop for Start, taskbar icons
        try:
            desktop = Desktop(backend="uia")
            taskbar = desktop.child_window(title="Taskbar", control_type="ToolBar")
            if taskbar.exists(timeout=1):
                for ctrl in taskbar.descendants():
                    try:
                        name = (ctrl.window_text() or "").strip()
                        if not name or len(name) > 60:
                            continue
                        ctype = getattr(ctrl.element_info, "control_type", None) or "Unknown"
                        key = (name, ctype)
                        if key in seen:
                            continue
                        seen.add(key)
                        if ctype in CLICKABLE_TYPES or "Button" in ctype:
                            out.append({"name": name, "control_type": str(ctype)})
                            if len(out) >= max_items:
                                return out
                    except Exception:
                        continue
        except Exception:
            pass
    except Exception:
        pass
    return out


def execute_action(action_dict: dict) -> tuple:
    """
    Execute a pywinauto-compatible action. Returns (success: bool, message: str).

    Supported actions:
    - CLICK: {"action": "CLICK", "parameters": {"element": "Open"}}
    - DOUBLE_CLICK: {"action": "DOUBLE_CLICK", "parameters": {"element": "Google Chrome"}}
    - TYPE_IN: {"action": "TYPE_IN", "parameters": {"element": "Search", "text": "chess.com"}}
    - MENU: {"action": "MENU", "parameters": {"path": "File->Open"}}
    - TASK_COMPLETE: {"action": "TASK_COMPLETE", "parameters": {"message": "Done"}}

    KEYS and TYPE_TEXT are handled by the caller (pynput) - not pywinauto.
    """
    if not _PYWINAUTO_AVAILABLE and sys.platform == "win32":
        return False, "pywinauto not available"
    action = (action_dict.get("action") or "").upper().replace(" ", "_")
    params = action_dict.get("parameters") or action_dict
    if isinstance(params, dict) and "action" in params:
        params = params.get("parameters", params) or params

    if action == "TASK_COMPLETE":
        return True, params.get("message", "Task complete")

    # pywinauto actions need Windows
    if sys.platform != "win32":
        return False, "pywinauto actions require Windows"

    element = params.get("element") or params.get("name")
    if not element and action in ("CLICK", "DOUBLE_CLICK", "TYPE_IN"):
        return False, f"Missing 'element' for action {action}"

    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return False, "No foreground window"
        app = Application(backend="uia").connect(handle=hwnd)
        win = app.window(handle=hwnd)

        def _find_and_click(container, double=False):
            # Try full match first, then shorter variants (Start menu items often have long suffixes)
            candidates = [element]
            if len(element) > 30:
                candidates.append(element[:30].strip())
            if "," in element:
                candidates.append(element.split(",")[0].strip())
            for search_text in candidates:
                if not search_text:
                    continue
                ctrl = container.child_window(title_re=f".*{_re_escape(search_text)}.*")
                if not ctrl.exists(timeout=1):
                    ctrl = container.child_window(title=search_text)
                if ctrl.exists(timeout=1):
                    if double:
                        ctrl.double_click_input()
                    else:
                        # Use click_input() - works on ListItem, DataItem where click() (Invoke) fails
                        ctrl.click_input()
                    return True
            return False

        if action == "CLICK":
            if _find_and_click(win, double=False):
                return True, f"Clicked '{element}'"
            # Fallback: try Desktop (for desktop icons, taskbar)
            try:
                desktop = Desktop(backend="uia")
                if _find_and_click(desktop, double=False):
                    return True, f"Clicked '{element}'"
            except Exception:
                pass
            return False, f"Element '{element}' not found"

        if action == "DOUBLE_CLICK":
            if _find_and_click(win, double=True):
                return True, f"Double-clicked '{element}'"
            try:
                desktop = Desktop(backend="uia")
                if _find_and_click(desktop, double=True):
                    return True, f"Double-clicked '{element}'"
            except Exception:
                pass
            return False, f"Element '{element}' not found"

        if action == "TYPE_IN":
            text = params.get("text", "")
            if not text:
                return False, "TYPE_IN requires 'text' parameter"
            ctrl = win.child_window(title_re=f".*{_re_escape(element)}.*", control_type="Edit")
            if not ctrl.exists(timeout=2):
                ctrl = win.child_window(title=element, control_type="Edit")
            if not ctrl.exists(timeout=2):
                ctrl = win.child_window(title_re=f".*{_re_escape(element)}.*")
            if not ctrl.exists(timeout=2):
                return False, f"Edit element '{element}' not found"
            ctrl.set_focus()
            ctrl.set_edit_text(text)
            return True, f"Typed into '{element}': {text[:30]}..."

        if action == "MENU":
            path = params.get("path") or params.get("menu") or ""
            if not path:
                return False, "MENU requires 'path' (e.g. 'File->Open')"
            win.menu_select(path)
            return True, f"Menu: {path}"

        return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _re_escape(s: str) -> str:
    """Escape string for use in regex."""
    return re.escape(str(s))
