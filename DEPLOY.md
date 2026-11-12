# Putting the web app online

The app (`app.py`) runs on any Python host. The easiest free option is **Streamlit Community Cloud**.
You get a public link such as `https://quantum-construction.streamlit.app` that your supervisor can open in a browser
with nothing installed.

Everything you need is already in the project: `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the data in `data/dslib/`.

---

## Option A: Streamlit Community Cloud (recommended, free)

### Step 1: put the project on GitHub (about 5 min)

1. Create a free account at <https://github.com> if you don't have one.
2. Click **+ → New repository**. Name it e.g. `quantum-construction-scheduling`, choose **Public**, and click **Create repository**.
3. On the new repository page, click **uploading an existing file**.
4. Open the unzipped project folder (the one that contains `app.py`), select **everything inside it**, and drag it
   into the browser. Include the `.streamlit` folder. On Windows, enable *View → Show → Hidden items* if you can't see it.
   Do **not** upload the `.venv` folder.
5. Click **Commit changes**.

*(Alternative if you use git in VS Code:)*
```powershell
git init
git add .
git commit -m "Quantum construction scheduler"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/quantum-construction-scheduling.git
git push -u origin main
```

### Step 2: deploy (about 5 min)

1. Go to <https://share.streamlit.io> and **sign in with GitHub**.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `YOUR_USERNAME/quantum-construction-scheduling`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: pick a name, e.g. `quantum-construction`
4. Open **Advanced settings → Python version → 3.12**.
5. Click **Deploy**.

The first build installs Qiskit, OR-Tools and the other packages, which takes **5–10 minutes**. After that the
app starts in seconds. Share the link.

### Notes

- The free tier puts apps to sleep after a few days without visitors. Opening the link wakes the app in about 30 s.
  Open it once before your review.
- The free server is small (CPU only, about 1 GB of memory). The app is sized for it: QAOA uses the 16-qubit core,
  and the hybrid run takes about 10–20 s.
- To update the app, change the files on GitHub and it redeploys automatically.
- **Never upload your IBM API key.** It isn't in the project; `run_ibm.py` reads it from your own computer only.

---

## Option B: Hugging Face Spaces (free alternative)

1. Create an account at <https://huggingface.co> → **New Space**.
2. **Space SDK: Docker → Blank** (or *Streamlit* if offered), hardware **CPU basic (free)**.
3. Upload all project files.
4. If you chose Docker, also create a file named `Dockerfile` in the Space with:
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   EXPOSE 7860
   CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
   ```
5. The Space builds and gives you a public link.

---

## Option C: show it on your own laptop during the review

No internet needed:

```powershell
.\.venv\Scripts\activate
py -m streamlit run app.py
```

To let others on the **same Wi-Fi** open it, use the *Network URL* that Streamlit prints
(e.g. `http://192.168.1.23:8501`). Windows Firewall may ask for permission; allow it on private networks.

---

## Why not Vercel / Netlify / GitHub Pages?

These host static websites and short serverless functions. The app needs a long-running Python server with Qiskit
and OR-Tools, which exceeds their size and time limits. Streamlit Cloud and Hugging Face are designed for this kind of app.
