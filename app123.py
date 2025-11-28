import analisi  # Importa il tuo file analisi.py
from flask import Flask, render_template, abort, request # <--- AGGIUNTO request per leggere il form
import matplotlib
# MODIFICA: Imposta il backend 'Agg' SUBITO, prima di usare pyplot, altrimenti il server crasha o si blocca.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from scipy import stats
import seaborn as sns
import os
from datetime import datetime, timedelta # <--- AGGIUNTO per gestire le date

app = Flask(__name__)

# ---------------------------------------------------------
# CARICAMENTO DATI ALL'AVVIO
# ---------------------------------------------------------
# Chiamiamo la funzione che abbiamo creato nel file analisi.py
print("Avvio server: sto scaricando i dati...")

# MODIFICA: Chiamo main con esegui_plot=False per impedire che si aprano finestre pop-up sul server.
# Inoltre, ora recuperiamo anche 'df_riepilogo' che ci serve per la tabella HTML.
df_prezzi_totali, df_solo_azioni, lista_tickers, market_ticker, df_riepilogo = analisi.main(esegui_plot=False)

# Assicuriamoci che l'indice sia in formato data corretto per poterlo filtrare
df_solo_azioni.index = pd.to_datetime(df_solo_azioni.index)

print("Dati scaricati e pronti.")

# ---------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    # MODIFICA: La pagina HTML 'index1.html' si aspetta 'dati_finanziari' come dizionario, non 'titoli' come lista.
    # Convertiamo il dataframe di riepilogo in un dizionario.
    dati_finanziari = df_riepilogo.to_dict(orient='index')
    
    return render_template("index1.html", dati_finanziari=dati_finanziari)

# MODIFICA: Aggiunto methods=['GET', 'POST'] per gestire la richiesta del form date
@app.route("/grafico1/<ticker>", methods=['GET', 'POST']) 
def grafico_ticker(ticker):
    # 1. Controlla se il titolo esiste nel dataframe scaricato
    if ticker not in df_solo_azioni.columns:
        return f"<h1>Errore</h1><p>Titolo {ticker} non trovato o dati insufficienti.</p>", 404

    # Recuperiamo la serie completa
    prezzi_azione_full = df_solo_azioni[ticker]

    # --- GESTIONE DATE ---
    # Calcolo date default (ultimo anno)
    max_date = prezzi_azione_full.index.max()
    default_start = (max_date - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end = max_date.strftime('%Y-%m-%d')

    if request.method == 'POST':
        # Se l'utente clicca aggiorna, prendiamo le date dal form
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
    else:
        # Altrimenti usiamo i default
        start_date = default_start
        end_date = default_end

    # 2. Prepara i dati specifici (FILTRATI PER DATA)
    # Usiamo .loc per tagliare il dataframe in base alle date scelte
    try:
        prezzi_azione = prezzi_azione_full.loc[start_date:end_date]
    except:
        prezzi_azione = prezzi_azione_full # Fallback in caso di errore date

    # Controllo dati insufficienti dopo il filtro
    if len(prezzi_azione) < 2:
        return f"<h1>Errore</h1><p>Dati insufficienti per il periodo selezionato.</p>"

    rendimento_log = np.log(prezzi_azione / prezzi_azione.shift(1)).dropna()

    # ==========================================
    # GRAFICO 1: TEMPORALE (Prezzo + Rendimento + Volatilità)
    # ==========================================
    fig1, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True) 
    
    # Prezzo
    axes[0].plot(prezzi_azione, color="steelblue", label='Prezzo')
    axes[0].set_title(f'{ticker} - Analisi Temporale', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Rendimenti Log
    axes[1].plot(rendimento_log, color='seagreen', lw=0.8, label='Rendimenti Log')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Volatilità Rolling 30gg
    volatilita = rendimento_log.rolling(30).std() * np.sqrt(252)
    axes[2].plot(volatilita, color='lightsalmon', label='Volatilità (30gg)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Conversione in Base64
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    plt.close(fig1) # Importante chiudere la figura per liberare memoria
    plot_temporale_b64 = base64.b64encode(img1.getvalue()).decode()

    # ==========================================
    # GRAFICO 2: DISTRIBUZIONE
    # ==========================================
    fig2 = plt.figure(figsize=(10, 6))
    
    # Istogramma
    # Controllo per evitare errori se i dati sono troppo pochi per la KDE
    if len(rendimento_log) > 10:
        # 1. Disegna solo l'Istogramma (senza KDE integrato)
        sns.histplot(rendimento_log, kde=False, stat="density", color="skyblue", 
                     element="step", alpha=0.5, label="Distribuzione Rendimenti")
        
        # 2. Disegna il KDE separatamente per avere la sua label
        sns.kdeplot(rendimento_log, color="blue", linewidth=1, label="Distribuzione Kemel Density Estimation)")
        
        # 3. Normale Teorica (codice invariato)
        mu, std = stats.norm.fit(rendimento_log)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=1, label="Distribuzione Normale")
        
    plt.title(f"{ticker} - Distribuzione dei Rendimenti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Conversione in Base64
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    plt.close(fig2)
    plot_distribuzione_b64 = base64.b64encode(img2.getvalue()).decode()

    # ==========================================
    # RETURN TEMPLATE
    # ==========================================
    return render_template(
        "grafico1.html", 
        ticker=ticker, 
        plot_temporale=plot_temporale_b64, 
        plot_distribuzione=plot_distribuzione_b64,
        start_date=start_date, # Passiamo le date al template per mantenerle nel box
        end_date=end_date
    )

if __name__ == "__main__":
    print("Avvio server Flask...")
    app.run(debug=True, use_reloader=False)