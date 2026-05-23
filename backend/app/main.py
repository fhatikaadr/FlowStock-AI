from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routes import forecast_routes

# Create DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FlowStock AI Forecasting API",
    description="Backend AI engine for inventory management and sales prediction.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast_routes.router, tags=["Forecasting"])

@app.get("/")
def root():
    return {"message": "Welcome to FlowStock AI Forecasting Engine API"}
