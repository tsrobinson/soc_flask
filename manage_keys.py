"""Interactive CLI for managing API keys in the OS keyring.

Keys stored here are picked up by soc_flask/app.py via the same _load_key()
helper. Precedence at load time is env var > keyring > (missing — error).
Stored under service 'socbot' with the standard env-var name as the account.

Usage:
    python manage_keys.py              # show menu
    python manage_keys.py list         # show which keys are set (redacted)
    python manage_keys.py set          # prompt for a key and store it
    python manage_keys.py set OPENAI_API_KEY
    python manage_keys.py delete OPENAI_API_KEY
    python manage_keys.py get OPENAI_API_KEY    # print redacted value

macOS stores these in Keychain; Linux uses Secret Service (requires a running
keyring daemon like gnome-keyring); Windows uses Credential Manager.
"""

from __future__ import annotations

import argparse
import getpass
import sys

KEYRING_SERVICE = "socbot"
KNOWN_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY",
              "TOGETHER_API_KEY", "PINECONE_API_KEY")


def _keyring():
    try:
        import keyring
    except ImportError:
        print("error: the 'keyring' package is not installed.", file=sys.stderr)
        print("       pip install keyring", file=sys.stderr)
        sys.exit(2)
    return keyring


def _redact(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]} (len={len(value)})"


def cmd_list() -> int:
    kr = _keyring()
    print(f"Keyring service: {KEYRING_SERVICE}")
    print()
    for name in KNOWN_KEYS:
        try:
            v = kr.get_password(KEYRING_SERVICE, name)
        except Exception as exc:
            print(f"  {name:24s}  (error: {exc})")
            continue
        status = _redact(v) if v else "— not set —"
        print(f"  {name:24s}  {status}")
    return 0


def cmd_set(name: str | None) -> int:
    kr = _keyring()
    if not name:
        print("Which key?")
        for i, k in enumerate(KNOWN_KEYS, 1):
            print(f"  {i}. {k}")
        choice = input("Enter number or full name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(KNOWN_KEYS):
            name = KNOWN_KEYS[int(choice) - 1]
        else:
            name = choice
    if not name:
        print("error: no key name given", file=sys.stderr)
        return 1
    try:
        value = getpass.getpass(f"Paste value for {name} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 130
    if not value:
        print("error: empty value — not storing", file=sys.stderr)
        return 1
    try:
        kr.set_password(KEYRING_SERVICE, name, value)
    except Exception as exc:
        print(f"error: could not store in keyring: {exc}", file=sys.stderr)
        return 1
    print(f"stored {name} in OS keyring ({_redact(value)})")
    return 0


def cmd_get(name: str) -> int:
    kr = _keyring()
    try:
        v = kr.get_password(KEYRING_SERVICE, name)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not v:
        print(f"{name}: not set")
        return 1
    print(f"{name}: {_redact(v)}")
    return 0


def cmd_delete(name: str) -> int:
    kr = _keyring()
    try:
        kr.delete_password(KEYRING_SERVICE, name)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"deleted {name} from OS keyring")
    return 0


def interactive_menu() -> int:
    while True:
        print()
        print("socbot key manager")
        print("  1. list keys")
        print("  2. set a key")
        print("  3. delete a key")
        print("  q. quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "1":
            cmd_list()
        elif choice == "2":
            cmd_set(None)
        elif choice == "3":
            name = input("Which key to delete? ").strip()
            if name:
                cmd_delete(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="show which keys are set (redacted)")
    sub_set = sub.add_parser("set", help="store a key in keyring")
    sub_set.add_argument("name", nargs="?")
    sub_get = sub.add_parser("get", help="show a stored key (redacted)")
    sub_get.add_argument("name")
    sub_del = sub.add_parser("delete", help="remove a key from keyring")
    sub_del.add_argument("name")
    args = ap.parse_args()

    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "set":
        return cmd_set(args.name)
    if args.cmd == "get":
        return cmd_get(args.name)
    if args.cmd == "delete":
        return cmd_delete(args.name)
    return interactive_menu()


if __name__ == "__main__":
    sys.exit(main())
