from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from mlxtend.frequent_patterns import apriori, association_rules

app = FastAPI()

# Load model
# model = joblib.load('amodel.pkl')
rules = joblib.load('rmodel.pkl')    



class AprioriRequest(BaseModel):
   
    input_item : str


# Fungsi untuk preprocessing data
# def preprocess_data(df):
#     item_count = df.groupby(["id", "menu"])["menu"].count().reset_index(name="count")
#     item_count_pivot = item_count.pivot_table(index='id', columns='menu', values='count', aggfunc='sum').fillna(0)
#     item_count_pivot = item_count_pivot.astype("int32")
#     item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)
#     return item_count_pivot

# def build_apriori_model(item_count_pivot, support=0.01, metric="lift", min_threshold=1):
#     frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
#     rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
#     return rules

def recommend_menu(rules, input_item):
    filtered_rules = rules[rules['antecedents'].apply(lambda x: input_item in x)]
    if not filtered_rules.empty:
        max_confidence_rule = filtered_rules.loc[filtered_rules['confidence'].idxmax()]
        recommendations = max_confidence_rule[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict()
        return recommendations
    else:
        return {"message": "No recommendations found."}
    # filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
    # recommendations = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    # return recommendations

@app.get("/")
def read_root():
    return {"message": "rekomendasi API is running!"}

@app.post("/apriori")
async def apriori_api( request: AprioriRequest):
    
    input_item = request.input_item
    recomendasi = recommend_menu(rules, input_item)
    return recomendasi
     
   

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)

