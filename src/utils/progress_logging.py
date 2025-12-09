"""Handler de logging personnalisé pour afficher la progression en temps réel."""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.utils.progress_ui import Color


class ProgressLogHandler(logging.StreamHandler):
    """Handler de logging qui affiche les messages avec progression visuelle."""

    # Mapping des étapes du pipeline aux emojis
    EMOJI_MAP = {
        "Audio": "📊",
        "Features": "🎯",
        "Train": "🤖",
        "Entraînement": "🤖",
        "Validation": "✅",
        "Eval": "📈",
        "Predict": "🎲",
        "Prédiction": "🎲",
        "Preprocessing": "🔧",
        "Preprocessing": "🔧",
    }

    def emit(self, record: logging.LogRecord) -> None:
        """Affiche le message de log avec couleur et progression.
        
        Args:
            record: Le record de log à afficher
        """
        try:
            msg = record.getMessage()

            # Ignorer les messages vides ou les messages de debug trop verbeux
            if not msg or msg.strip() == "":
                return

            # Colorier basé sur le niveau de log
            if record.levelno == logging.ERROR:
                color = Color.RED
                prefix = "❌"
            elif record.levelno == logging.WARNING:
                color = Color.YELLOW
                prefix = "⚠️"
            elif record.levelno == logging.INFO:
                color = Color.CYAN
                prefix = "ℹ️"
            else:
                color = Color.ENDC
                prefix = "•"

            # Ajouter un emoji contextuel basé sur le contenu du message
            emoji = self._extract_emoji(msg)

            # Formater le message
            formatted_msg = f"{color}{prefix} {emoji} {msg}{Color.ENDC}"

            # Écrire sur le stream
            self.stream.write(formatted_msg + "\n")
            self.stream.flush()

        except Exception:
            self.handleError(record)

    @staticmethod
    def _extract_emoji(msg: str) -> str:
        """Extrait un emoji approprié du message.
        
        Args:
            msg: Le message à analyser
            
        Returns:
            Un emoji basé sur le contenu du message
        """
        msg_lower = msg.lower()

        if "audio" in msg_lower or "fichier" in msg_lower:
            return "🎵"
        elif "features" in msg_lower or "extraction" in msg_lower:
            return "🎯"
        elif "train" in msg_lower or "entraîn" in msg_lower or "fitting" in msg_lower:
            return "🤖"
        elif "validation" in msg_lower or "accuracy" in msg_lower:
            return "✅"
        elif "pred" in msg_lower or "prédiction" in msg_lower:
            return "🎲"
        elif "svm" in msg_lower or "quantum" in msg_lower:
            return "⚡"
        elif "qsvm" in msg_lower:
            return "⚛️"
        elif "eval" in msg_lower or "metric" in msg_lower:
            return "📈"
        elif "sauvegardé" in msg_lower or "save" in msg_lower or "saved" in msg_lower:
            return "💾"
        elif "erreur" in msg_lower or "error" in msg_lower or "exception" in msg_lower:
            return "❌"
        elif "warning" in msg_lower or "avertissement" in msg_lower:
            return "⚠️"
        else:
            return "→"


def configure_live_logging(level: int = logging.INFO) -> None:
    """Configure le logging pour afficher la progression en temps réel.
    
    Args:
        level: Niveau de log à afficher (par défaut INFO)
    """
    # Obtenir le logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Supprimer les handlers existants pour éviter les doublons
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Ajouter notre handler personnalisé
    handler = ProgressLogHandler()
    handler.setLevel(level)

    # Format simplifié (le message contient déjà l'info contextuelle)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
