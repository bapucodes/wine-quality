import uvicorn

if __name__ == "__main__":
    print("[INFO] Starting Wine Quality Prediction API on http://localhost:8000 ...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
