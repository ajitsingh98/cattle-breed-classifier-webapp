"""
Breed information service.
Provides breed metadata lookup from CSV.
"""

import csv
from pathlib import Path
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import logger


class BreedInfoService:
    """Service for breed metadata lookup."""

    def __init__(self):
        self._data: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load breed metadata from CSV."""
        if self._loaded:
            return

        settings = get_settings()
        csv_path = Path(settings.metadata_path)

        if not csv_path.exists():
            logger.warning(f"Breed metadata file not found: {csv_path}")
            return

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    breed_name = row['breed_name'].strip()
                    self._data[breed_name] = {
                        'breed_id': int(row['breed_id']),
                        'breed_name': breed_name,
                        'animal_type': row['animal_type'].strip(),
                        'region': row['region'].strip(),
                        'avg_milk_liters_per_day': row['avg_milk_liters_per_day'].strip(),
                        'lifespan_years': row['lifespan_years'].strip(),
                        'primary_use': row['primary_use'].strip(),
                        'coat_color_notes': row['coat_color_notes'].strip(),
                        'horn_notes': row['horn_notes'].strip(),
                        'description': row['description'].strip(),
                        'source_reference': row['source_reference'].strip(),
                    }

            self._loaded = True
            logger.info(f"Loaded metadata for {len(self._data)} breeds")

        except Exception as e:
            logger.error(f"Failed to load breed metadata: {e}")

    def get_breed(self, breed_name: str) -> Optional[dict]:
        """Get breed info by exact name."""
        return self._data.get(breed_name)

    def get_breed_summary(self, breed_name: str) -> Optional[dict]:
        """Get condensed breed info for prediction responses."""
        info = self._data.get(breed_name)
        if info is None:
            return None
        return {
            'breed_name': info['breed_name'],
            'animal_type': info['animal_type'],
            'region': info['region'],
            'avg_milk_liters_per_day': info['avg_milk_liters_per_day'],
            'lifespan_years': info['lifespan_years'],
            'primary_use': info['primary_use'],
            'description': info['description'],
        }

    def get_all(self) -> list[dict]:
        """Return all breeds."""
        return list(self._data.values())

    def search(self, query: str) -> list[dict]:
        """Case-insensitive search."""
        q = query.lower()
        return [b for b in self._data.values() if q in b['breed_name'].lower()]

    @property
    def num_breeds(self) -> int:
        return len(self._data)


# Singleton
breed_info_service = BreedInfoService()
