import os
import re
import json
import time
from google import genai
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY secret not set in Hugging Face Space settings.")

client = genai.Client(api_key=GEMINI_API_KEY)

LABEL_COLORS = {'NEGATIVE': '#e74c3c', 'POSITIVE': '#2ecc71', 'NEUTRAL': '#3498db'}
LABEL_EMOJI  = {'NEGATIVE': '😠', 'POSITIVE': '😊', 'NEUTRAL': '😐'}

SYSTEM_PROMPT = """You are an expert sentiment analysis system specialized in Cebuano (Bisaya/Bisakol) language text from the Philippines.

Analyze the sentiment of the given text and respond ONLY with a valid JSON object in this exact format:
{
  "prediction": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
  "confidence": <number 0-100>,
  "POSITIVE": <number 0-100>,
  "NEGATIVE": <number 0-100>,
  "NEUTRAL": <number 0-100>,
  "reason": "<brief one-sentence explanation>"
}

Rules:
- prediction must be exactly one of: POSITIVE, NEGATIVE, NEUTRAL
- The three scores (POSITIVE, NEGATIVE, NEUTRAL) must sum to 100
- confidence should match the score of the predicted class
- Understand Cebuano slang, abbreviations, and mixed Cebuano-English text
- Consider negators like: dili, wala, ayaw, di, wa
- Consider intensifiers like: kaayo, gyud, jud, grabe, sobra
- Do NOT output anything other than the JSON object"""


def _infer(text: str, retries: int = 3) -> dict:
    if not isinstance(text, str) or not text.strip():
        return None
    prompt = f'{SYSTEM_PROMPT}\n\nText to analyze: "{text.strip()}"'
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model="gemini-1.5-flash-8b", contents=prompt)
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            data = json.loads(raw)
            if data.get('prediction') not in ('POSITIVE', 'NEGATIVE', 'NEUTRAL'):
                continue
            return {
                'text':       text,
                'prediction': data['prediction'],
                'confidence': round(float(data.get('confidence', 0)), 1),
                'POSITIVE':   round(float(data.get('POSITIVE', 0)), 1),
                'NEGATIVE':   round(float(data.get('NEGATIVE', 0)), 1),
                'NEUTRAL':    round(float(data.get('NEUTRAL',  0)), 1),
                'reason':     data.get('reason', ''),
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"[_infer error] {e} | text: {text[:60]}")
    return None


def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 2]


def analyze_single(text):
    if not text or not text.strip():
        return "Please enter some Cebuano text.", None
    result = _infer(text.strip())
    if not result:
        return "Could not analyze the text. Please try again.", None

    label = result['prediction']
    color = LABEL_COLORS[label]
    emoji = LABEL_EMOJI[label]

    html = f"""
    <div style="font-family:sans-serif; padding:16px;">
        <div style="background:{color}22; border-left:5px solid {color};
                    border-radius:8px; padding:16px 20px; margin-bottom:14px;">
            <div style="font-size:26px; font-weight:bold; color:{color};">{emoji} {label}</div>
            <div style="color:#555; margin-top:6px; font-size:14px;">
                Confidence: <strong>{result['confidence']:.1f}%</strong>
            </div>
            <div style="color:#777; margin-top:8px; font-size:13px; font-style:italic;">
                {result['reason']}
            </div>
        </div>
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <div style="flex:1;background:#fde8e8;border-radius:8px;padding:10px 14px;text-align:center;">
                <div style="font-size:20px;font-weight:bold;color:#e74c3c;">{result['NEGATIVE']:.1f}%</div>
                <div style="font-size:11px;color:#888;">NEGATIVE</div>
            </div>
            <div style="flex:1;background:#e8fdf0;border-radius:8px;padding:10px 14px;text-align:center;">
                <div style="font-size:20px;font-weight:bold;color:#2ecc71;">{result['POSITIVE']:.1f}%</div>
                <div style="font-size:11px;color:#888;">POSITIVE</div>
            </div>
            <div style="flex:1;background:#e8f4fd;border-radius:8px;padding:10px 14px;text-align:center;">
                <div style="font-size:20px;font-weight:bold;color:#3498db;">{result['NEUTRAL']:.1f}%</div>
                <div style="font-size:11px;color:#888;">NEUTRAL</div>
            </div>
        </div>
    </div>"""

    fig, ax = plt.subplots(figsize=(5, 2.5))
    classes = ['NEGATIVE', 'POSITIVE', 'NEUTRAL']
    values  = [result['NEGATIVE'], result['POSITIVE'], result['NEUTRAL']]
    bars = ax.barh(classes, values, color=[LABEL_COLORS[c] for c in classes],
                   edgecolor='white', height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Score Breakdown')
    ax.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val+1, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    return html, fig


def _process_texts(raw_texts):
    all_texts = []
    for t in raw_texts:
        t = str(t).strip()
        if not t: continue
        if len(t.split()) > 20:
            all_texts.extend(split_into_sentences(t))
        else:
            all_texts.append(t)
    results = []
    for t in all_texts:
        r = _infer(t)
        if r:
            results.append(r)
        time.sleep(0.3)
    return results


def _build_output(results):
    if not results:
        return "No valid results returned.", None, None

    df    = pd.DataFrame(results)
    total = len(df)
    counts   = df['prediction'].value_counts()
    neg_c    = counts.get('NEGATIVE', 0)
    pos_c    = counts.get('POSITIVE', 0)
    neu_c    = counts.get('NEUTRAL',  0)
    avg_conf = df['confidence'].mean()

    summary_html = f"""
    <div style="font-family:sans-serif;">
        <div style="background:#1a1a2e;color:#eee;border-radius:10px;padding:16px 20px;margin-bottom:14px;">
            <h3 style="margin:0 0 4px 0;color:#f0c040;">Analysis Complete</h3>
            <p style="margin:0;color:#aaa;font-size:13px;">
                {total} texts analyzed | Avg confidence: {avg_conf:.1f}%
            </p>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
            <div style="flex:1;background:#fde8e8;border-left:4px solid #e74c3c;border-radius:8px;padding:12px 16px;">
                <div style="font-size:24px;font-weight:bold;color:#e74c3c;">{neg_c}</div>
                <div style="font-size:12px;color:#888;">NEGATIVE ({neg_c/total*100:.1f}%)</div>
            </div>
            <div style="flex:1;background:#e8fdf0;border-left:4px solid #2ecc71;border-radius:8px;padding:12px 16px;">
                <div style="font-size:24px;font-weight:bold;color:#2ecc71;">{pos_c}</div>
                <div style="font-size:12px;color:#888;">POSITIVE ({pos_c/total*100:.1f}%)</div>
            </div>
            <div style="flex:1;background:#e8f4fd;border-left:4px solid #3498db;border-radius:8px;padding:12px 16px;">
                <div style="font-size:24px;font-weight:bold;color:#3498db;">{neu_c}</div>
                <div style="font-size:12px;color:#888;">NEUTRAL ({neu_c/total*100:.1f}%)</div>
            </div>
        </div>
    </div>"""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#f8f9fa')
    labels_plot = [k for k in ['NEGATIVE','POSITIVE','NEUTRAL'] if counts.get(k,0) > 0]
    axes[0].pie([counts.get(k,0) for k in labels_plot],
                labels=labels_plot, colors=[LABEL_COLORS[k] for k in labels_plot],
                autopct='%1.1f%%', startangle=140,
                wedgeprops={'linewidth':2,'edgecolor':'white'}, textprops={'fontsize':11})
    axes[0].set_title('Sentiment Distribution', fontsize=12, fontweight='bold')

    class_conf = {lbl: df[df['prediction']==lbl]['confidence'].mean()
                  for lbl in ['NEGATIVE','POSITIVE','NEUTRAL']
                  if len(df[df['prediction']==lbl]) > 0}
    if class_conf:
        bars = axes[1].bar(list(class_conf.keys()), list(class_conf.values()),
                           color=[LABEL_COLORS[k] for k in class_conf],
                           edgecolor='white', linewidth=1.5, width=0.5)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel('Avg Confidence (%)')
        axes[1].set_title('Avg Confidence per Class', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('#f8f9fa')
        axes[1].spines[['top','right']].set_visible(False)
        for bar, val in zip(bars, class_conf.values()):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                         f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()

    display_df = df[['text','prediction','confidence','NEGATIVE','POSITIVE','NEUTRAL','reason']].head(50).copy()
    display_df.columns = ['Text','Prediction','Confidence%','NEG%','POS%','NEU%','Reason']
    return summary_html, fig, display_df


def analyze_bulk(text_block):
    if not text_block or not text_block.strip():
        return "Please paste some comments first.", None, None
    lines = [l.strip() for l in text_block.strip().split('\n') if l.strip()]
    return _build_output(_process_texts(lines))


def analyze_file(file, csv_column):
    if file is None:
        return "Please upload a file.", None, None
    fname = file.name
    texts = []
    try:
        if fname.endswith('.txt'):
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                texts = [l.strip() for l in f.readlines() if l.strip()]
        elif fname.endswith('.csv'):
            df_up = pd.read_csv(fname)
            col   = csv_column.strip() if csv_column else 'text'
            if col not in df_up.columns:
                return f"Column '{col}' not found. Available: {list(df_up.columns)}", None, None
            texts = df_up[col].dropna().astype(str).tolist()
        elif fname.endswith('.json'):
            with open(fname, 'r', encoding='utf-8') as f:
                data_loaded = json.load(f)
            if isinstance(data_loaded, list) and len(data_loaded) > 0:
                if isinstance(data_loaded[0], str):
                    texts = data_loaded
                elif isinstance(data_loaded[0], dict):
                    for key in ['text','comment','content','message','original']:
                        if key in data_loaded[0]:
                            texts = [str(item.get(key,'')) for item in data_loaded]
                            break
                    if not texts:
                        return f"No text field found. Keys: {list(data_loaded[0].keys())}", None, None
        else:
            return "Unsupported format. Use .csv, .txt, or .json", None, None
    except Exception as e:
        return f"Error reading file: {e}", None, None

    if not texts:
        return "No text found in file.", None, None
    return _build_output(_process_texts(texts))


with gr.Blocks(title="Bisakol Sentiment Analyzer") as demo:

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
                padding:20px 24px;margin-bottom:20px;">
        <h1 style="margin:0 0 6px 0;color:#f0c040;font-size:22px;letter-spacing:2px;">
            BISAKOL SENTIMENT ANALYZER
        </h1>
        <p style="margin:0;color:#aaa;font-size:13px;">
            Powered by Google Gemini AI | Specialized for Cebuano / Bisaya / Bisakol text
        </p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Single Text"):
            gr.Markdown("Type or paste a single Cebuano comment to analyze its sentiment.")
            single_input = gr.Textbox(label="Input Text",
                                      placeholder="e.g. Nindot kaayo ang programa!", lines=3)
            single_btn   = gr.Button("Analyze", variant="primary")
            with gr.Row():
                single_html  = gr.HTML(label="Result")
                single_chart = gr.Plot(label="Score Breakdown")
            single_btn.click(fn=analyze_single, inputs=[single_input],
                             outputs=[single_html, single_chart])

        with gr.Tab("Bulk Text"):
            gr.Markdown("Paste multiple comments, one per line.")
            bulk_input = gr.Textbox(label="Comments (one per line)",
                                    placeholder="Paste comments here...", lines=8)
            bulk_btn   = gr.Button("Analyze All", variant="primary")
            bulk_html  = gr.HTML(label="Summary")
            bulk_chart = gr.Plot(label="Charts")
            bulk_table = gr.Dataframe(label="Results (first 50)", wrap=True)
            bulk_btn.click(fn=analyze_bulk, inputs=[bulk_input],
                           outputs=[bulk_html, bulk_chart, bulk_table])

        with gr.Tab("Upload File"):
            gr.Markdown("Upload .txt (one per line), .csv (specify column), or .json file.")
            with gr.Row():
                file_input = gr.File(label="Upload File", file_types=['.csv','.txt','.json'])
                col_input  = gr.Textbox(label="CSV Column Name", value="text", scale=1)
            file_btn   = gr.Button("Analyze File", variant="primary")
            file_html  = gr.HTML(label="Summary")
            file_chart = gr.Plot(label="Charts")
            file_table = gr.Dataframe(label="Results (first 50)", wrap=True)
            file_btn.click(fn=analyze_file, inputs=[file_input, col_input],
                           outputs=[file_html, file_chart, file_table])

demo.launch()