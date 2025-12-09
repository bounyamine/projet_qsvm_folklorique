"""Utilitaires pour l'affichage visuel de la progression dans le terminal.

Provides:
- Colored banners and messages
- Progress bars with tqdm
- Status indicators
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


class Color:
    """ANSI color codes pour le terminal."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ProgressStage(Enum):
    """Étapes du pipeline avec leurs couleurs respectives."""

    START = (Color.CYAN, "▶")
    DATA = (Color.BLUE, "📊")
    PREPROCESSING = (Color.BLUE, "🔧")
    FEATURE_EXTRACTION = (Color.BLUE, "🎯")
    TRAINING = (Color.YELLOW, "🤖")
    VALIDATION = (Color.YELLOW, "✓")
    PREDICTION = (Color.GREEN, "🎲")
    EVALUATION = (Color.CYAN, "📈")
    COMPLETE = (Color.GREEN, "✅")
    ERROR = (Color.RED, "❌")


def print_banner(text: str, stage: ProgressStage = ProgressStage.START) -> None:
    """Affiche une bannière colorée avec étape.

    Args:
        text: Texte à afficher
        stage: Étape du pipeline (pour les couleurs et symboles)
    """
    color, symbol = stage.value
    banner = f"\n{color}{symbol} {text}{Color.ENDC}"
    print(banner)


def print_section(title: str, subtitle: Optional[str] = None) -> None:
    """Affiche une section avec titre et sous-titre.

    Args:
        title: Titre de la section
        subtitle: Sous-titre optionnel
    """
    print(f"\n{Color.BOLD}{Color.CYAN}{'=' * 70}{Color.ENDC}")
    print(f"{Color.BOLD}{Color.CYAN}{title:^70}{Color.ENDC}")
    if subtitle:
        print(f"{Color.CYAN}{subtitle:^70}{Color.ENDC}")
    print(f"{Color.BOLD}{Color.CYAN}{'=' * 70}{Color.ENDC}\n")


def print_step(step_num: int, total_steps: int, description: str) -> None:
    """Affiche une étape numérotée.

    Args:
        step_num: Numéro de l'étape
        total_steps: Nombre total d'étapes
        description: Description de l'étape
    """
    print(f"{Color.BOLD}{Color.YELLOW}[{step_num}/{total_steps}]{Color.ENDC} {description}")


def create_progress_bar(
    iterable,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    color: str = Color.CYAN,
) -> tqdm: # type: ignore
    """Crée une barre de progression colorée.

    Args:
        iterable: L'itérable à parcourir
        desc: Description de la barre
        total: Nombre total d'éléments (si itérable n'a pas de len)
        unit: Unité affichée ("it", "sample", "file", etc.)
        color: Couleur ANSI pour la barre

    Returns:
        Barre de progression tqdm
    """
    if tqdm is None:
        return iterable  # type: ignore

    return tqdm(
        iterable,
        desc=f"{color}{desc}{Color.ENDC}",
        total=total,
        unit=unit,
        ncols=80,
        bar_format="{desc} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
    )


def print_result(key: str, value, color: str = Color.GREEN) -> None:
    """Affiche un résultat final.

    Args:
        key: Clé du résultat
        value: Valeur du résultat
        color: Couleur ANSI
    """
    print(f"  {Color.BOLD}{key}:{Color.ENDC} {color}{value}{Color.ENDC}")


def print_success(message: str) -> None:
    """Affiche un message de succès."""
    print(f"\n{Color.GREEN}{Color.BOLD}✅ {message}{Color.ENDC}\n")


def print_warning(message: str) -> None:
    """Affiche un message d'avertissement."""
    print(f"\n{Color.YELLOW}{Color.BOLD}⚠️  {message}{Color.ENDC}\n")


def print_error(message: str) -> None:
    """Affiche un message d'erreur."""
    print(f"\n{Color.RED}{Color.BOLD}❌ Erreur: {message}{Color.ENDC}\n")


def print_info(message: str) -> None:
    """Affiche un message d'information."""
    print(f"{Color.CYAN}ℹ️  {message}{Color.ENDC}")
