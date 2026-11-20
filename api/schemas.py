from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class WineInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    wine_id: Optional[str] = Field(default="WINE_0001", alias="wine_id")
    fixed_acidity: float = Field(..., ge=0, alias="fixed acidity")
    volatile_acidity: float = Field(..., ge=0, alias="volatile acidity")
    citric_acid: float = Field(..., ge=0, alias="citric acid")
    residual_sugar: float = Field(..., ge=0, alias="residual sugar")
    chlorides: float = Field(..., ge=0, alias="chlorides")
    free_sulfur_dioxide: float = Field(..., ge=0, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., ge=0, alias="total sulfur dioxide")
    density: float = Field(..., ge=0, alias="density")
    pH: float = Field(..., ge=0, alias="pH")
    sulphates: float = Field(..., ge=0, alias="sulphates")
    alcohol: float = Field(..., ge=0, alias="alcohol")


class PredictionResponse(BaseModel):
    prediction: int
    quality_probability: float
    churn_probability: Optional[float] = None


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    wines: list[WineInput] = Field(default=[], min_length=1)
    # alias or alternative list name for compatibility
    customers: Optional[list[WineInput]] = None

    def get_items(self) -> list[WineInput]:
        if self.wines:
            return self.wines
        if self.customers:
            return self.customers
        return []


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
