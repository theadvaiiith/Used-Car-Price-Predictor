"""
used_car_price_gui_fixed.py
Tkinter GUI app with fixed feature importance chart and working retrain button.
"""

import tkinter as tk
from tkinter import ttk, font, messagebox
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageTk

# -----------------------------
# 1. Synthetic dataset
# -----------------------------
def create_synthetic_dataset(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    models = ["i10", "Swift", "Baleno", "Ertiga", "Creta", "City", "Verna"]
    vehicle_types = ["Hatchback", "Sedan", "SUV", "MUV"]
    fuels = ["Petrol", "Diesel", "CNG", "Hybrid"]
    transmissions = ["Manual", "Automatic"]

    data = []
    for _ in range(n):
        year = rng.randint(2005, 2024)
        model = rng.choice(models)
        vtype = rng.choice(vehicle_types)
        fuel = rng.choice(fuels)
        trans = rng.choice(transmissions)
        owners = rng.choice([1,2,3], p=[0.75,0.2,0.05])
        kms = int(abs(rng.normal(loc=60000, scale=30000)))
        kms = max(1000, min(kms, 300000))
        base_price = {"i10":4.5,"Swift":5.5,"Baleno":7.0,"Ertiga":8.5,
                      "Creta":13.0,"City":9.0,"Verna":11.0}[model]
        vtype_adj = {"Hatchback":0.9,"Sedan":1.0,"SUV":1.3,"MUV":1.1}[vtype]
        fuel_adj = {"Petrol":1.0,"Diesel":1.05,"CNG":0.9,"Hybrid":1.2}[fuel]
        trans_adj = {"Manual":1.0,"Automatic":1.1}[trans]
        age = 2025-year
        price_lakhs = base_price*vtype_adj*fuel_adj*trans_adj
        price_lakhs *= max(0.2,1-0.12*age-(kms/200000)*0.25-0.05*(owners-1))
        price_lakhs *= rng.uniform(0.9,1.08)
        price_rupees = int(price_lakhs*100000)
        data.append({
            "year_of_mfd":year,
            "model":model,
            "vehicle_type":vtype,
            "fuel_type":fuel,
            "transmission":trans,
            "no_of_owners":owners,
            "kms_driven":kms,
            "price_rupees":price_rupees
        })
    return pd.DataFrame(data)

# -----------------------------
# 2. ML Pipeline
# -----------------------------
def build_and_train_model(df, random_state=42):
    X = df.drop(columns=["price_rupees"])
    y = df["price_rupees"].astype(float)
    categorical_cols = ["model","vehicle_type","fuel_type","transmission"]
    numeric_cols = ["year_of_mfd","no_of_owners","kms_driven"]
    cat_proc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_proc = StandardScaler()
    preprocessor = ColumnTransformer([
        ("num", num_proc, numeric_cols),
        ("cat", cat_proc, categorical_cols)
    ])
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=120, random_state=random_state))
    ])
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    mae = mean_absolute_error(y,preds)
    r2 = r2_score(y,preds)
    return pipeline, (mae,r2)

# -----------------------------
# 3. GUI
# -----------------------------
class UsedCarPriceApp:
    def __init__(self, master):
        self.master = master
        master.title("Used Car Price Predictor")
        master.geometry("900x600")
        master.minsize(750,500)
        master.columnconfigure(0,weight=1)
        master.rowconfigure(0,weight=1)

        # Fonts
        self.font_big = font.Font(family="Helvetica", size=20, weight="bold")
        self.font_med = font.Font(family="Helvetica", size=12)
        self.font_small = font.Font(family="Helvetica", size=10)

        # Themes
        self.theme_var = tk.StringVar(value="Dark")
        self.themes = {
            "Dark":{"bg":"#0f1724","panel":"#121827","accent":"#00C2A8","text":"#E6EEF6","sub":"#9FB3C8","bar":"#0032c7"},
            "Light":{"bg":"#f0f4f8","panel":"#e2e8f0","accent":"#0288d1","text":"#042528","sub":"#475569","bar":"#0288d1"}
        }
        self._set_theme("Dark")

        # Header
        self.header = tk.Label(master, text="Used Car Price Predictor", font=self.font_big, bg=self.bg_color, fg=self.accent)
        self.header.pack(pady=(10,6))

        # Theme toggle
        theme_frame = tk.Frame(master, bg=self.bg_color)
        theme_frame.pack(pady=(0,8))
        tk.Label(theme_frame,text="Theme: ", bg=self.bg_color, fg=self.sub_text,font=self.font_small).pack(side="left")
        tk.Radiobutton(theme_frame,text="Dark",variable=self.theme_var,value="Dark",command=self._switch_theme,bg=self.bg_color,fg=self.text_color,selectcolor=self.panel_color,activebackground=self.bg_color).pack(side="left")
        tk.Radiobutton(theme_frame,text="Light",variable=self.theme_var,value="Light",command=self._switch_theme,bg=self.bg_color,fg=self.text_color,selectcolor=self.panel_color,activebackground=self.bg_color).pack(side="left")

        # Main frame
        self.main_frame = tk.Frame(master,bg=self.panel_color)
        self.main_frame.pack(fill="both",expand=True,padx=12,pady=6)
        self.main_frame.columnconfigure(0,weight=1)
        self.main_frame.columnconfigure(1,weight=1)
        self.main_frame.rowconfigure(0,weight=1)

        self.left = tk.Frame(self.main_frame,bg=self.panel_color)
        self.left.grid(row=0,column=0,sticky="nsew",padx=12,pady=6)
        self.right = tk.Frame(self.main_frame,bg=self.panel_color)
        self.right.grid(row=0,column=1,sticky="nsew",padx=12,pady=6)

        # Dataset & model
        self.df = create_synthetic_dataset()
        self.pipeline, (self.mae,self.r2) = build_and_train_model(self.df)

        # Inputs
        self.models = sorted(self.df["model"].unique())
        self.vehicle_types = sorted(self.df["vehicle_type"].unique())
        self.fuels = sorted(self.df["fuel_type"].unique())
        self.trans = sorted(self.df["transmission"].unique())

        # --- Input widgets ---
        # Year
        self.year_var = tk.IntVar(value=2016)
        yframe = self._labeled_frame(self.left,"Year of Manufacture")
        self.year_entry = ttk.Spinbox(yframe,from_=1995,to=2025,textvariable=self.year_var,width=10,font=self.font_med)
        self.year_entry.pack(side="left",padx=8,pady=4)

        # Model
        self.model_var = tk.StringVar(value=self.models[0])
        mframe = self._labeled_frame(self.left,"Model")
        self.model_menu = ttk.Combobox(mframe,values=self.models,textvariable=self.model_var,state="readonly",width=18,font=self.font_med)
        self.model_menu.pack(side="left",padx=8,pady=4)

        # Vehicle type
        self.vtype_var = tk.StringVar(value=self.vehicle_types[0])
        vframe = self._labeled_frame(self.left,"Vehicle Type")
        self.vtype_menu = ttk.Combobox(vframe,values=self.vehicle_types,textvariable=self.vtype_var,state="readonly",width=18,font=self.font_med)
        self.vtype_menu.pack(side="left",padx=8,pady=4)

        # Fuel type
        self.fuel_var = tk.StringVar(value=self.fuels[0])
        fframe = self._labeled_frame(self.left,"Fuel Type")
        self.fuel_menu = ttk.Combobox(fframe,values=self.fuels,textvariable=self.fuel_var,state="readonly",width=18,font=self.font_med)
        self.fuel_menu.pack(side="left",padx=8,pady=4)

        # Transmission
        self.trans_var = tk.StringVar(value=self.trans[0])
        tframe = self._labeled_frame(self.left,"Transmission")
        self.trans_menu = ttk.Combobox(tframe,values=self.trans,textvariable=self.trans_var,state="readonly",width=18,font=self.font_med)
        self.trans_menu.pack(side="left",padx=8,pady=4)

        # Owners
        self.owners_var = tk.IntVar(value=1)
        oframe = self._labeled_frame(self.left,"Number of Owners")
        self.owners_spin = ttk.Spinbox(oframe,from_=1,to=5,textvariable=self.owners_var,width=6,font=self.font_med)
        self.owners_spin.pack(side="left",padx=4,pady=4)

        # Kilometres
        self.kms_var = tk.IntVar(value=50000)
        kframe = self._labeled_frame(self.left,"Kilometres Driven")
        self.kms_entry = ttk.Entry(kframe,textvariable=self.kms_var,width=12,font=self.font_med)
        self.kms_entry.pack(side="left",padx=4,pady=4)

        # Buttons frame
        btn_frame = tk.Frame(self.left,bg=self.panel_color)
        btn_frame.pack(pady=10)

        # Predict button
        self.predict_btn = tk.Button(btn_frame,text="PREDICT PRICE 💰",command=self.predict_price,bg=self.accent,fg="#FFFFFF",font=self.font_med,bd=0,padx=16,pady=8,relief="flat",cursor="hand2")
        self.predict_btn.pack(side="left",padx=(0,6))

        # Retrain button
        self.retrain_btn = tk.Button(btn_frame,text="RE-TRAIN 🔄",command=self.retrain_model,bg="#374151",fg=self.text_color,font=self.font_med,bd=0,padx=16,pady=8,relief="flat",cursor="hand2")
        self.retrain_btn.pack(side="left",padx=6)

        # Result panel
        self.result_label = tk.Label(self.right,text="—",bg=self.panel_color,fg=self.accent,font=self.font_big)
        self.result_label.pack(pady=10)

        # Metrics
        self.mae_label = tk.Label(self.right,text=f"MAE: ₹{int(self.mae):,}",bg=self.panel_color,fg=self.sub_text,font=self.font_small)
        self.mae_label.pack(anchor="w")
        self.r2_label = tk.Label(self.right,text=f"R²: {self.r2:.3f}",bg=self.panel_color,fg=self.sub_text,font=self.font_small)
        self.r2_label.pack(anchor="w")

        # Feature importance
        self.canvas_label = tk.Label(self.right,bg=self.panel_color)
        self.canvas_label.pack(pady=12)
        self._render_feature_importance()

    # -----------------------------
    # Helper functions
    # -----------------------------
    def _labeled_frame(self,parent,text):
        frame = tk.Frame(parent,bg=self.panel_color)
        frame.pack(fill="x",pady=4)
        tk.Label(frame,text=text,bg=self.panel_color,fg=self.text_color,font=self.font_small).pack(anchor="w")
        inner = tk.Frame(frame,bg=self.panel_color)
        inner.pack(anchor="w")
        return inner

    def _set_theme(self,name):
        t = self.themes[name]
        self.bg_color = t["bg"]
        self.panel_color = t["panel"]
        self.accent = t["accent"]
        self.text_color = t["text"]
        self.sub_text = t["sub"]
        self.bar_color = t["bar"]

    def _switch_theme(self):
        self._set_theme(self.theme_var.get())
        self.master.configure(bg=self.bg_color)
        self.header.configure(bg=self.bg_color,fg=self.accent)
        self.left.configure(bg=self.panel_color)
        self.right.configure(bg=self.panel_color)
        self.canvas_label.configure(bg=self.panel_color)
        self.mae_label.configure(bg=self.panel_color,fg=self.sub_text)
        self.r2_label.configure(bg=self.panel_color,fg=self.sub_text)

    # -----------------------------
    # Predict
    # -----------------------------
    def predict_price(self):
        try:
            year = int(self.year_var.get())
            model = self.model_var.get()
            vtype = self.vtype_var.get()
            fuel = self.fuel_var.get()
            trans = self.trans_var.get()
            owners = int(self.owners_var.get())
            kms = int(self.kms_var.get())
        except Exception:
            messagebox.showerror("Invalid input","Please enter valid numbers.")
            return
        X_new = pd.DataFrame([{
            "year_of_mfd":year,
            "model":model,
            "vehicle_type":vtype,
            "fuel_type":fuel,
            "transmission":trans,
            "no_of_owners":owners,
            "kms_driven":kms
        }])
        pred = self.pipeline.predict(X_new)[0]
        self.result_label.configure(text=f"₹ {int(pred):,}",fg=self.accent)

    # -----------------------------
    # Retrain
    # -----------------------------
    def retrain_model(self):
        self.df = create_synthetic_dataset()
        self.pipeline, (self.mae,self.r2) = build_and_train_model(self.df)

        self.mae_label.configure(text=f"MAE: ₹{int(self.mae):,}")
        self.r2_label.configure(text=f"R²: {self.r2:.3f}")
        self.result_label.configure(text="—")

        # Update comboboxes
        self.models = sorted(self.df["model"].unique())
        self.vehicle_types = sorted(self.df["vehicle_type"].unique())
        self.fuels = sorted(self.df["fuel_type"].unique())
        self.trans = sorted(self.df["transmission"].unique())

        self.model_menu.configure(values=self.models)
        self.vtype_menu.configure(values=self.vehicle_types)
        self.fuel_menu.configure(values=self.fuels)
        self.trans_menu.configure(values=self.trans)

        # Reset selections
        self.model_var.set(self.models[0])
        self.vtype_var.set(self.vehicle_types[0])
        self.fuel_var.set(self.fuels[0])
        self.trans_var.set(self.trans[0])
        self._render_feature_importance()
        messagebox.showinfo("Retrain","Model retrained on new synthetic data!")

    # -----------------------------
    # Feature importance
    # -----------------------------
    def _render_feature_importance(self):
        try:
            rf = self.pipeline.named_steps["model"]
            pre = self.pipeline.named_steps["pre"]
            num_cols = pre.transformers_[0][2]
            ohe: OneHotEncoder = pre.transformers_[1][1]
            cat_cols = pre.transformers_[1][2]
            cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names = list(num_cols)+cat_feature_names
            importances = rf.feature_importances_
            imp_df = pd.DataFrame({"feature":feature_names,"importance":importances})
            imp_df = imp_df.sort_values("importance",ascending=False).head(8)

            # Plot fixed size
            plt.figure(figsize=(4,3))
            plt.barh(imp_df["feature"][::-1],imp_df["importance"][::-1],color=self.bar_color)
            plt.xlabel("Importance",color=self.text_color)
            plt.yticks(color=self.text_color,fontsize=9)
            plt.xticks(color=self.text_color)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf,format="png",dpi=120,bbox_inches="tight",transparent=True)
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            img = img.resize((320,180),Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas_label.configure(image=photo)
            self.canvas_label.image = photo
        except Exception as e:
            self.canvas_label.configure(text="(Feature importance unavailable)",fg=self.sub_text,font=self.font_small)

# -----------------------------
# Run
# -----------------------------
def main():
    root = tk.Tk()
    app = UsedCarPriceApp(root)
    root.mainloop()

if __name__=="__main__":
    main()
