# Streamlit LLM Flavor Demo

Streamlit app that extracts product features from vape/cannabis packaging images using the OpenAI API.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub (public recommended for Community Cloud).
2. Go to Streamlit Community Cloud and click "New app".
3. Select your repo, branch, and set the main file to `app.py`.
4. Click Deploy.

After deploy, open the app URL and enter your OpenAI API key in the sidebar.
