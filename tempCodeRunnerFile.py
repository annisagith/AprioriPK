def build_apriori_model(item_count_pivot, support=0.01, metric="lift", min_threshold=1):
#     frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
#     rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
#     return rules