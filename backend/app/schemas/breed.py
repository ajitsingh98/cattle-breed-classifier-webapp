"""
Pydantic schemas for breed metadata endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class BreedDetail(BaseModel):
    """Full breed detail."""
    breed_id: int
    breed_name: str
    animal_type: str
    region: str
    avg_milk_liters_per_day: str
    lifespan_years: str
    primary_use: str
    coat_color_notes: str
    horn_notes: str
    description: str
    source_reference: str


class BreedListItem(BaseModel):
    """Summary breed item for list endpoints."""
    breed_id: int
    breed_name: str
    animal_type: str
    region: str
    primary_use: str


class BreedsListResponse(BaseModel):
    """Response for listing all breeds."""
    total: int
    breeds: list[BreedListItem]
