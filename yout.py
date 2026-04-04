"""Ponto de entrada CLI para o baixador YouTube (usa o mesmo código da app ClipMaster)."""

from app.services.youtube_download import baixar_video_interativo

if __name__ == "__main__":
    baixar_video_interativo()
