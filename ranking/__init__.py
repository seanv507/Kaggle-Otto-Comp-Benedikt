"""Benny's rankers (02_Clicks, 03_Carts, 04_Orders), one implementation per stage.

split    candidates and features per fold (01_Split)
train    XGBoost: local CV and the submission models (03_XGB)
combine  bags and folds into the top 20 per session (04_Combine-Bags)
"""
