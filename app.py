import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


OPENAI_API_BASE = "https://api.openai.com/v1"
MODEL_ID = "gpt-5.2"
SYSTEM_PROMPT = """### Role
You are a high-precision Data Extraction Specialist for the vape/cannabis industry. Your task is to extract product text features from images with 100% literal accuracy.

### Critical Rules (Expert Constraints)
1. **Literal Extraction Only**: Extract text EXACTLY as it appears on the package.
   - NO spell correction (e.g., keep "Strazzberry", do not change to "Strawberry").
   - NO background inference (e.g., ignore fruit/cereal images if text isn't present).
   - Strict Verbatim: Do not translate, do not "fix", and do not assume. If the text says "STAR BUZZ", you MUST output "STAR BUZZ". Do not output "Strawberry" or "Starfruit" based on your guess.
2. **Clean Output**: Each flavor string must be "naked".
   - NO bullet points (-), NO stars (*), NO whitespace padding.
   - NO parentheses or metadata (e.g., do not include "(blue device)").
3. **Exclude Nicotine**: If nicotine content (e.g., "5%", "50mg") appears next to the flavor, DO NOT include it in the flavor field.
4. **Handling Absence**:
   - If no flavor text is visible, the list must be ["missing"].
   - For any other field not found in the image, return the literal string "Not found".
5. **Brand Identification**: Identify the brand associated with each flavor if multiple brands exist.
6. **Anti-Hallucination Rule**: If you see a word you don't recognize as a "standard" flavor (e.g., "Star Buzz"), you MUST extract it exactly as written. NEVER replace a literal word with a "common" flavor name.

### Output Schema (JSON)
Return ONLY a JSON object:
- "flavors_list": Array of strings. Pure flavor text only. For example, ["Flavor1", "Flavor2"]
- "multiple_descriptors": "1" (if >1), "0" (if 1), "n/a" (if none flavor descriptor extracted due to poor image quality or no flavor shown on the image).
- "extraction_evidence": Internal note on text location of flavors and ignored art.
- "brand_name": Brand name found.
- "nicotine_content": Nicotine text as shown (e.g., "5%", "50mg"). If missing, return "Not found".
- "size_or_volume": Size/volume text as shown of the product (e.g., "10ml", "2mL", "6000 puffs"). If missing, return "Not found".
- "warning_label_present": "Yes" if a warning label is visible, otherwise "No".
- "warning_label_location": If warning_label_present is "Yes", describe the label location on the package. If no warning label, return "Not found".
- "main_color": Array of all product colors you think. Use ONLY the allowed color names list below; ignore background/embellishments. If not found, return ["Not found"].

### Allowed Color Names (use ONLY these)
green, blue, purple, red, pink, yellow, orange, brown, teal, lightblue, grey, limegreen, magenta, lightgreen, brightgreen, skyblue, cyan, turquoise, darkblue, darkgreen, aqua, olive, navyblue, lavender, fuchsia, black, royalblue, violet, hotpink, tan, forestgreen, lightpurple, neongreen, yellowgreen, maroon, darkpurple, salmon, peach, beige, lime, seafoamgreen, mustard, brightblue, lilac, seagreen, palegreen, bluegreen, mint, lightbrown, mauve, darkred, greyblue, burntorange, darkpink, indigo, periwinkle, bluegrey, lightpink, aquamarine, gold, brightpurple, grassgreen, redorange, bluepurple, greygreen, kellygreen, puke, rose, darkteal, babyblue, paleblue, greenyellow, brickred, lightgrey, darkgrey, white, brightpink, chartreuse, purpleblue, royalpurple, burgundy, goldenrod, darkbrown, lightorange, darkorange, redbrown, paleyellow, plum, offwhite, pinkpurple, darkyellow, lightyellow, mustardyellow, brightred, peagreen, khaki, orangered, crimson, deepblue, springgreen, cream, palepink, yelloworange, deeppurple, pinkred, pastelgreen, sand, rust, lightred, taupe, armygreen, robinseggblue, huntergreen, greenblue, lightteal, cerulean, flesh, orangebrown, slateblue, slate, coral, blueviolet, ochre, leafgreen, electricblue, seablue, midnightblue, steelblue, brick, palepurple, mediumblue, burntsienna, darkmagenta, eggplant, sage, darkturquoise, puce, bloodred, neonpurple, mossgreen, terracotta, oceanblue, yellowbrown, brightyellow, dustyrose, applegreen, neonpink, skin, cornflowerblue, lightturquoise, wine, deepred, azure
"""


def build_user_prompt() -> str:
    return (
        "You will be given ONE image of a vape/cannabis product package.\n"
        "Extract flavor text exactly as written on the package. Do NOT infer from images or guess missing text.\n"
        "If multiple flavor descriptors appear, include all of them as separate strings.\n"
        "If no flavor text is visible, return [\"missing\"].\n\n"
        "Also extract the brand name if visible. Do not include nicotine strength or percentages in flavor strings.\n"
        "Extract nicotine content and size/volume exactly as shown. If missing, return \"Not found\".\n"
        "Only treat size_or_volume as values with explicit units (ml/mL/mg/g/puffs). "
        "Do NOT treat model/series names (e.g., AL6000) as size.\n"
        "Detect whether a warning label is present. If present, set warning_label_present to \"Yes\" and describe its location.\n"
        "If no warning label is visible, set warning_label_present to \"No\" and warning_label_location to \"Not found\".\n"
        "Extract all product colors (but ignore background/embellishments). "
        "Return an array of color names using ONLY the allowed color list in the system prompt. "
        "If unsure, return [\"Not found\"].\n"
        "If any extracted text is unclear, keep it verbatim (even if misspelled).\n\n"
        "Return JSON only, strictly matching the schema."
    )


def required_fields() -> List[str]:
    return [
        "flavors_list",
        "multiple_descriptors",
        "brand_name",
        "extraction_evidence",
        "nicotine_content",
        "size_or_volume",
        "warning_label_present",
        "warning_label_location",
        "main_color",
    ]


def extract_output_text(response_json: Dict[str, Any]) -> str:
    output = response_json.get("output", [])
    texts: List[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join([t for t in texts if t])


def call_openai(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
) -> Dict[str, Any]:
    url = f"{OPENAI_API_BASE}/responses"
    headers = {"Authorization": f"Bearer {api_key}"}
    image_content: Dict[str, Any] = {"type": "input_image", "image_url": image_url}
    payload = {
        "model": MODEL_ID,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    image_content,
                ],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if not resp.ok:
        detail = ""
        try:
            err = resp.json().get("error", {})
            message = err.get("message", "")
            err_type = err.get("type", "")
            code = err.get("code", "")
            detail = f"{message} (type={err_type}, code={code})".strip()
        except Exception:
            detail = resp.text[:1000]
        hint = ""
        if resp.status_code in (401, 403, 404):
            hint = " This may be a model access issue; check your API tier and organization verification."
        raise RuntimeError(f"HTTP {resp.status_code} {resp.reason}: {detail}{hint}")
    return resp.json()


def file_to_data_url(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    if not data:
        return None
    mime = uploaded_file.type or "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def normalize_drive_url(url: str) -> str:
    if "drive.google.com" not in url:
        return url
    if "id=" in url:
        file_id = url.split("id=")[-1].split("&")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[-1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def fetch_image_url(url: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    try:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            return None, None, f"HTTP {resp.status_code} {resp.reason}"
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return None, None, f"URL did not return an image (Content-Type: {content_type})"
        return resp.content, content_type, None
    except Exception as exc:
        return None, None, str(exc)


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "Missing API key"
    url = f"{OPENAI_API_BASE}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.ok:
            return True, "API key looks valid"
        if resp.status_code == 401:
            return False, "Invalid API key"
        return False, f"HTTP {resp.status_code} {resp.reason}"
    except Exception as exc:
        return False, str(exc)


def update_api_key_status() -> None:
    ok, msg = validate_api_key(st.session_state.get("api_key_input", ""))
    st.session_state["api_key_ok"] = ok
    st.session_state["api_key_msg"] = msg


st.set_page_config(page_title="Flavor Extractor Demo", layout="wide")
st.title("Flavor Extractor Demo")
st.caption("Extract flavors and some other features directly from the product iamge")

system_prompt = SYSTEM_PROMPT

with st.sidebar:
    st.subheader("API Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key_input",
        on_change=update_api_key_status,
    )
    if "api_key_ok" in st.session_state:
        if st.session_state["api_key_ok"]:
            st.success(st.session_state.get("api_key_msg", "API key looks valid"))
        else:
            st.error(st.session_state.get("api_key_msg", "API key check failed"))

st.subheader("Input")
input_mode = st.radio("Image input", ["Upload file", "Image URL"], index=0)
uploaded_file = None
image_url_input = ""
image_data_url = None

if input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "gif"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", width='content')
        image_data_url = file_to_data_url(uploaded_file)
else:
    image_url_input = st.text_input("Image URL")
    if image_url_input:
        normalized = normalize_drive_url(image_url_input)
        data, content_type, err = fetch_image_url(normalized)
        if err:
            st.error(f"Image preview failed: {err}")
        else:
            st.image(data, caption="Image URL preview", width='content')
            b64 = base64.b64encode(data).decode("ascii")
            image_data_url = f"data:{content_type};base64,{b64}"

run = False
has_image = bool(image_data_url)
if has_image:
    st.subheader("Output")
    run = st.button("Run extraction")

if run:
    if not api_key:
        st.error("OpenAI API key is required.")
        st.stop()
    if not system_prompt:
        st.error("system_prompt is missing or empty.")
        st.stop()

    if not image_data_url:
        st.error("Please provide a valid image before running.")
        st.stop()

    user_prompt = build_user_prompt()

    with st.spinner("Calling OpenAI..."):
        try:
            response_json = call_openai(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_url=image_data_url,
            )
        except Exception as exc:
            st.error(f"API call failed: {exc}")
            st.stop()

    output_text = extract_output_text(response_json)
    st.session_state["last_response_json"] = response_json
    st.session_state["last_output_text"] = output_text

if "last_output_text" in st.session_state:
    output_text = st.session_state.get("last_output_text", "")
    response_json = st.session_state.get("last_response_json", {})
    if not output_text:
        st.warning("No output_text found in response. Showing raw response.")
        st.json(response_json)
    else:
        try:
            parsed = json.loads(output_text)
            st.json(parsed)
            missing = [k for k in required_fields() if k not in parsed]
            if missing:
                st.warning(f"JSON missing required fields: {', '.join(missing)}")
            st.download_button(
                "Download JSON",
                data=json.dumps(parsed, ensure_ascii=False, indent=2),
                file_name="extraction.json",
                mime="application/json",
            )
        except json.JSONDecodeError:
            st.warning("Output is not valid JSON. Showing raw text.")
            st.code(output_text)
