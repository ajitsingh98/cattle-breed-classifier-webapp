"""
Breed metadata loader and lookup utility.
Reads breed_metadata.csv and provides enriched breed info for inference.
"""

import csv
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = PROJECT_ROOT / 'data' / 'breed_metadata.csv'


class BreedMetadataStore:
    """Loads and provides lookup for breed metadata."""

    def __init__(self, csv_path: Optional[str | Path] = None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_METADATA_PATH
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load CSV into an in-memory dict keyed by breed_name."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Breed metadata file not found: {self.csv_path}")

        with open(self.csv_path, 'r', encoding='utf-8') as f:
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

    def get(self, breed_name: str) -> Optional[dict]:
        """Look up breed info by exact name."""
        return self._data.get(breed_name)

    def search(self, query: str) -> list[dict]:
        """Case-insensitive partial match search."""
        query_lower = query.lower()
        return [
            info for name, info in self._data.items()
            if query_lower in name.lower()
        ]

    def get_all(self) -> list[dict]:
        """Return all breed records."""
        return list(self._data.values())

    def get_breed_names(self) -> list[str]:
        """Return sorted list of breed names."""
        return sorted(self._data.keys())

    @property
    def num_breeds(self) -> int:
        return len(self._data)

    def get_summary_for_prediction(self, breed_name: str) -> Optional[dict]:
        """
        Return a condensed breed info dict suitable for API responses.
        """
        info = self.get(breed_name)
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
