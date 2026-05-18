#!/usr/bin/env python3
"""
Teste manual da TikTok API (OAuth + creator_info).

SEGURANÇA
---------
- Não commite segredos. Use variáveis de ambiente ou um ficheiro .env local (ignorado pelo git).
- Se colou o Client Secret em chats ou fóruns, regenere-o em developers.tiktok.com.

PowerShell (exemplo):
  $env:TIKTOK_CLIENT_KEY="sua_client_key"
  $env:TIKTOK_CLIENT_SECRET="seu_client_secret"
  $env:TIKTOK_REDIRECT_URI="http://127.0.0.1:8080/callback/"
  python scripts/test_tiktok_api.py auth-url --desktop

Depois abra a URL no browser, autorize, copie o `code` da query do redirect e:
  $env:TIKTOK_CODE="o_codigo"
  python scripts/test_tiktok_api.py exchange --code $env:TIKTOK_CODE --code-verifier "o_verifier_impresso"

Com access token:
  $env:TIKTOK_ACCESS_TOKEN="act...."
  python scripts/test_tiktok_api.py creator

Requisito: pip/ projeto já com httpx (dependência do ClipMaster).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import string
import sys
from urllib.parse import urlencode

import httpx

AUTH_PAGE = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"


def _load_dotenv_file(path: str) -> None:
    """Carrega KEY=VAL num .env simples (sem dependência python-dotenv)."""
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def client_key_secret() -> tuple[str, str]:
    key = (os.environ.get("TIKTOK_CLIENT_KEY") or "").strip()
    secret = (os.environ.get("TIKTOK_CLIENT_SECRET") or "").strip()
    if not key or not secret:
        print(
            "Defina TIKTOK_CLIENT_KEY e TIKTOK_CLIENT_SECRET (ou use --env-file).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key, secret


def generate_code_verifier(length: int = 64) -> str:
    """PKCE: 43–128 chars, conjunto não reservado RFC 7636."""
    length = max(43, min(length, 128))
    alphabet = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def code_challenge_s256(verifier: str) -> str:
    """TikTok Desktop: code_challenge = hex(SHA256(code_verifier)), método S256."""
    return hashlib.sha256(verifier.encode("ascii")).hexdigest()


def cmd_auth_url(args: argparse.Namespace) -> None:
    key, _ = client_key_secret()
    redirect = (args.redirect_uri or os.environ.get("TIKTOK_REDIRECT_URI") or "").strip()
    if not redirect:
        print("Indique --redirect-uri ou TIKTOK_REDIRECT_URI (deve coincidir com o portal).", file=sys.stderr)
        sys.exit(1)
    state = secrets.token_urlsafe(24)
    params: dict[str, str] = {
        "client_key": key,
        "response_type": "code",
        "scope": args.scope,
        "redirect_uri": redirect,
        "state": state,
    }
    if args.desktop:
        verifier = generate_code_verifier()
        challenge = code_challenge_s256(verifier)
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"
        print("--- Guarde o code_verifier até trocar o código pelo token ---", file=sys.stderr)
        print(verifier, file=sys.stderr)
        print("--- state (confira no callback) ---", file=sys.stderr)
        print(state, file=sys.stderr)
        print(file=sys.stderr)
    q = urlencode(params, safe="")
    url = f"{AUTH_PAGE}?{q}"
    print(url)


def cmd_exchange(args: argparse.Namespace) -> None:
    key, secret = client_key_secret()
    redirect = (args.redirect_uri or os.environ.get("TIKTOK_REDIRECT_URI") or "").strip()
    if not redirect:
        print("Indique --redirect-uri ou TIKTOK_REDIRECT_URI.", file=sys.stderr)
        sys.exit(1)
    code = (args.code or os.environ.get("TIKTOK_CODE") or "").strip()
    if not code:
        print("Indique --code ou TIKTOK_CODE.", file=sys.stderr)
        sys.exit(1)

    data = {
        "client_key": key,
        "client_secret": secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect,
    }
    verifier = (args.code_verifier or os.environ.get("TIKTOK_CODE_VERIFIER") or "").strip()
    if verifier:
        data["code_verifier"] = verifier

    with httpx.Client() as client:
        r = client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=60.0,
        )
    print(f"HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception:
        print(r.text)
        sys.exit(1)
    print(json.dumps(body, indent=2, ensure_ascii=False))
    if r.is_success and "access_token" in body:
        print("\nPara testar creator_info:", file=sys.stderr)
        print(f'  set TIKTOK_ACCESS_TOKEN="{body["access_token"]}"', file=sys.stderr)
        print("  python scripts/test_tiktok_api.py creator", file=sys.stderr)


def cmd_refresh(args: argparse.Namespace) -> None:
    key, secret = client_key_secret()
    rt = (args.refresh_token or os.environ.get("TIKTOK_REFRESH_TOKEN") or "").strip()
    if not rt:
        print("Indique --refresh-token ou TIKTOK_REFRESH_TOKEN.", file=sys.stderr)
        sys.exit(1)
    data = {
        "client_key": key,
        "client_secret": secret,
        "grant_type": "refresh_token",
        "refresh_token": rt,
    }
    with httpx.Client() as client:
        r = client.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=60.0,
        )
    print(f"HTTP {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def cmd_creator(_args: argparse.Namespace) -> None:
    token = (os.environ.get("TIKTOK_ACCESS_TOKEN") or "").strip()
    if not token:
        print("Defina TIKTOK_ACCESS_TOKEN.", file=sys.stderr)
        sys.exit(1)
    with httpx.Client() as client:
        r = client.post(
            CREATOR_INFO_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={},
            timeout=60.0,
        )
    print(f"HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Testes manuais TikTok OAuth / Content Posting")
    parser.add_argument(
        "--env-file",
        default="",
        help="Ficheiro .env opcional (ex.: scripts/tiktok.local.env) com TIKTOK_CLIENT_KEY=...",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_auth = sub.add_parser("auth-url", help="Imprime URL para abrir no browser (OAuth)")
    p_auth.add_argument(
        "--redirect-uri",
        default="",
        help="URI registada no portal (ex.: http://127.0.0.1:8080/callback/)",
    )
    p_auth.add_argument(
        "--scope",
        default=os.environ.get("TIKTOK_SCOPE", "user.info.basic,video.publish"),
        help="Scopes separados por vírgula",
    )
    p_auth.add_argument(
        "--desktop",
        action="store_true",
        help="Login Kit desktop: gera PKCE e imprime code_verifier no stderr",
    )
    p_auth.set_defaults(func=cmd_auth_url)

    p_ex = sub.add_parser("exchange", help="Troca authorization code por tokens")
    p_ex.add_argument("--code", default="", help="Código devolvido no redirect (ou TIKTOK_CODE)")
    p_ex.add_argument("--redirect-uri", default="", help="O mesmo usado em auth-url")
    p_ex.add_argument(
        "--code-verifier",
        default="",
        help="Obrigatório se usou auth-url --desktop (ou TIKTOK_CODE_VERIFIER)",
    )
    p_ex.set_defaults(func=cmd_exchange)

    p_rf = sub.add_parser("refresh", help="Renovar access_token com refresh_token")
    p_rf.add_argument("--refresh-token", default="", help="Ou TIKTOK_REFRESH_TOKEN")
    p_rf.set_defaults(func=cmd_refresh)

    p_cr = sub.add_parser("creator", help="POST creator_info/query (testa video.publish / token)")
    p_cr.set_defaults(func=cmd_creator)

    args = parser.parse_args()
    if args.env_file:
        _load_dotenv_file(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
