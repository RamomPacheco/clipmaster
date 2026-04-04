from __future__ import annotations

from pathlib import Path

import yt_dlp


def _ydl_base_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }


def list_video_heights(url: str) -> list[int]:
    """
    Lista alturas (px) de vídeo disponíveis no URL, da maior para a menor.
    """
    opts = _ydl_base_opts()
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    formatos = info.get("formats") or []
    resolucoes: set[int] = set()
    for f in formatos:
        altura = f.get("height")
        vcodec = f.get("vcodec")
        if vcodec != "none" and altura is not None:
            resolucoes.add(int(altura))
    return sorted(resolucoes, reverse=True)


def _safe_stem_from_id(video_id: str) -> str:
    raw = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(video_id))
    return raw[:80] if raw else "video"


def _resolved_output_path(output_dir: Path, base: str) -> Path:
    for ext in ("mp4", "webm", "mkv", "m4a"):
        p = output_dir / f"{base}.{ext}"
        if p.exists():
            return p
    matches = list(output_dir.glob(f"{base}.*"))
    if not matches:
        raise FileNotFoundError(
            f"Download concluído mas não foi encontrado ficheiro com prefixo {base!r} em {output_dir}"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def download_youtube_video(url: str, height: int, output_dir: str | Path) -> Path:
    """
    Descarrega o vídeo na altura indicada (melhor vídeo + melhor áudio, merge MP4 quando possível).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with yt_dlp.YoutubeDL(_ydl_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
    vid = info.get("id") or "video"
    base = _safe_stem_from_id(str(vid))
    outtmpl = str(output_dir / f"{base}.%(ext)s")

    ydl_opts_dl = {
        "format": (
            f"bestvideo[height={height}]+bestaudio/best[height<={height}]"
        ),
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "windowsfilenames": True,
        "noplaylist": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
        ydl.download([url])

    return _resolved_output_path(output_dir, base)


def baixar_video_interativo() -> None:
    """Interface de linha de comando (equivalente ao script original)."""
    print("-" * 50)
    print("🎬 Baixador do YouTube (Menu de Qualidade Dinâmico)")
    print("-" * 50)

    url = input("Cole a URL do vídeo do YouTube aqui: ")

    print("\n⏳ Lendo o vídeo e buscando as resoluções disponíveis. Aguarde...")

    try:
        lista_resolucoes = list_video_heights(url)

        if not lista_resolucoes:
            print("❌ Nenhuma resolução de vídeo encontrada para este link.")
            return

        print("\n✅ Qualidades encontradas para este vídeo:")
        for i, res in enumerate(lista_resolucoes):
            print(f"{i + 1} - {res}p")

        escolha = input(
            f"\nDigite o número da qualidade desejada (1 a {len(lista_resolucoes)}): "
        )

        try:
            indice_escolhido = int(escolha) - 1
            if indice_escolhido < 0 or indice_escolhido >= len(lista_resolucoes):
                raise ValueError
        except ValueError:
            print("⚠️ Opção inválida. Operação cancelada.")
            return

        resolucao_escolhida = lista_resolucoes[indice_escolhido]

        print(
            f"\n🚀 Iniciando o download em {resolucao_escolhida}p. Por favor, aguarde..."
        )
        download_youtube_video(url, resolucao_escolhida, Path("."))
        print("\n✅ Download concluído com sucesso!")

    except yt_dlp.utils.DownloadError:
        print("\n❌ Erro: Não foi possível acessar o vídeo. Verifique a URL.")
    except Exception as e:
        print(f"\n❌ Ocorreu um erro inesperado: {e}")
