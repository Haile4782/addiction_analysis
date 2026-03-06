from pydantic import BaseModel


class PredictionInput(BaseModel):
    age: int
    gender: str
    annual_income_usd: float
    bmi: float
    smokes_per_day: int
    drinks_per_week: int
    mental_health_status: str
    exercise_frequency: str
    sleep_hours: int