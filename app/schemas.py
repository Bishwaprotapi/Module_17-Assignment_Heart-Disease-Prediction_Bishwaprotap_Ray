from pydantic import BaseModel, Field


class HeartFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, examples=[52])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=0, le=3, examples=[0])
    trestbps: int = Field(..., ge=50, le=300, examples=[125])
    chol: int = Field(..., ge=50, le=700, examples=[212])
    fbs: int = Field(..., ge=0, le=1, examples=[0])
    restecg: int = Field(..., ge=0, le=2, examples=[1])
    thalach: int = Field(..., ge=50, le=250, examples=[168])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[1.0])
    slope: int = Field(..., ge=0, le=2, examples=[2])
    ca: int = Field(..., ge=0, le=4, examples=[2])
    thal: int = Field(..., ge=0, le=3, examples=[3])


class PredictionResponse(BaseModel):
    heart_disease: bool
