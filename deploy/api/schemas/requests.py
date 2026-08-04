"""Modelos de request da API (contratos de entrada), fortemente tipados.

Os campos e limites espelham as colunas físicas de imóveis usadas no treino
(`data/raw/kc_house_data.csv` / `data/raw/future_unseen_examples.csv`,
ver `src/data.py`). Mantidos em sincronia manualmente: se o conjunto de
features do modelo mudar, este schema deve ser atualizado junto.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

# Bounds de latitude/longitude de King County (região coberta pelos dados de
# treino), com uma margem de tolerância. Fora dessa região o modelo
# extrapola sem nenhuma garantia de qualidade.
_KING_COUNTY_LAT_RANGE: tuple[float, float] = (46.5, 48.5)
_KING_COUNTY_LONG_RANGE: tuple[float, float] = (-123.5, -120.5)


class HouseFeatures(BaseModel):
    """Características físicas de um único imóvel, para previsão de preço.

    Todos os campos são obrigatórios exceto `yr_renovated` (0 = nunca
    reformado, convenção do dataset original).
    """

    bedrooms: int = Field(..., ge=0, le=15, description="Número de quartos.")
    bathrooms: float = Field(..., ge=0, le=8, description="Número de banheiros (frações contam meio-banheiro).")
    sqft_living: int = Field(..., gt=0, le=15000, description="Área construída interna, em pés quadrados.")
    sqft_lot: int = Field(..., gt=0, le=2_000_000, description="Área do lote, em pés quadrados.")
    floors: float = Field(..., ge=1, le=4, description="Número de andares (aceita meios-andares, ex.: 1.5).")
    waterfront: int = Field(..., ge=0, le=1, description="1 se o imóvel tem frente para a água, 0 caso contrário.")
    view: int = Field(..., ge=0, le=4, description="Índice de qualidade da vista (0 a 4).")
    condition: int = Field(..., ge=1, le=5, description="Estado de conservação do imóvel (1 a 5).")
    grade: int = Field(..., ge=1, le=13, description="Nota de qualidade de construção/acabamento (1 a 13).")
    sqft_above: int = Field(..., ge=0, description="Área construída acima do nível do solo, em pés quadrados.")
    sqft_basement: int = Field(..., ge=0, description="Área do porão, em pés quadrados (0 se não houver).")
    yr_built: int = Field(..., ge=1800, le=2100, description="Ano de construção.")
    yr_renovated: int = Field(0, ge=0, le=2100, description="Ano da última reforma, ou 0 se nunca reformado.")
    zipcode: int = Field(..., ge=10000, le=99999, description="CEP (zipcode) dos EUA, 5 dígitos.")
    lat: float = Field(..., description="Latitude do imóvel.")
    long: float = Field(..., description="Longitude do imóvel.")
    sqft_living15: int = Field(..., gt=0, description="Área construída média das 15 casas mais próximas.")
    sqft_lot15: int = Field(..., gt=0, description="Área do lote média das 15 casas mais próximas.")

    @model_validator(mode="after")
    def _validate_structural_consistency(self) -> "HouseFeatures":
        """Garante consistência interna dos campos, replicando invariantes
        observadas nos dados de treino (100% dos registros satisfazem isso)."""
        if self.sqft_above + self.sqft_basement != self.sqft_living:
            raise ValueError(
                "sqft_above + sqft_basement deve ser igual a sqft_living "
                f"(recebido: {self.sqft_above} + {self.sqft_basement} != {self.sqft_living})"
            )
        if self.yr_renovated != 0 and self.yr_renovated < self.yr_built:
            raise ValueError("yr_renovated não pode ser anterior a yr_built")
        lat_min, lat_max = _KING_COUNTY_LAT_RANGE
        long_min, long_max = _KING_COUNTY_LONG_RANGE
        if not (lat_min <= self.lat <= lat_max) or not (long_min <= self.long <= long_max):
            raise ValueError(
                "lat/long fora da região coberta pelo modelo "
                f"(esperado lat em [{lat_min}, {lat_max}] e long em [{long_min}, {long_max}])"
            )
        return self


class PredictionRequest(BaseModel):
    """Payload de uma solicitação de previsão, em lote (1 a N imóveis)."""

    houses: list[HouseFeatures] = Field(
        ..., min_length=1, description="Lista de imóveis a precificar (tamanho máximo definido em Settings.max_batch_size)."
    )
    request_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Identificador de correlação, propagado para logs/tracing. Gerado automaticamente se omitido.",
    )
