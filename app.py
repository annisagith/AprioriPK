from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load('model.pkl')

# Pydantic model untuk validasi input data
class MenuItem(BaseModel):
    id: int
    menu: str

class AprioriRequest(BaseModel):
    data: list[MenuItem]
    support: float = 0.01
    min_threshold: float = 1

# Fungsi untuk preprocessing data
def preprocess_data(df):
    item_count = df.groupby(["id", "menu"])["menu"].count().reset_index(name="count")
    item_count_pivot = item_count.pivot_table(index='id', columns='menu', values='count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.astype("int32")
    item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)
    return item_count_pivot

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/apriori")
async def apriori_api(request: AprioriRequest):
    try:
        # Ambil data dari permintaan
        data = request.data
        df = pd.DataFrame([item.dict() for item in data])
        
        # Pra-pemrosesan data
        item_count_pivot = preprocess_data(df)
        
        # Menggunakan model untuk membuat prediksi atau menghasilkan output
        support = request.support
        min_threshold = request.min_threshold
        
        filtered_model = model[(model['support'] >= support) & (model['lift'] >= min_threshold)]
        
        return filtered_model.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
