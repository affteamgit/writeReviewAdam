import os
import openai
import requests
import json
import streamlit as st
from google.oauth2.service_account import Credentials   
from googleapiclient.discovery import build
from anthropic import Anthropic
from pathlib import Path
import re

# CONFIG 
OPENAI_API_KEY      = st.secrets["OPENAI_API_KEY"]
GROK_API_KEY        = st.secrets["GROK_API_KEY"]
ANTHROPIC_API_KEY   = st.secrets["ANTHROPIC_API_KEY"]
COINMARKETCAP_API_KEY = st.secrets["COINMARKETCAP_API_KEY"]

SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
SHEET_NAME     = st.secrets["SHEET_NAME"]

FOLDER_ID = st.secrets["FOLDER_ID"]
GUIDELINES_FOLDER_ID = st.secrets["GUIDELINES_FOLDER_ID"]

# Fine-tuned model for Adam's rewriting
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-1106:affiliation:adam0301:ByHlJhcR"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive"
]

DOCS_DRIVE_SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]

def get_service_account_credentials():
    return Credentials.from_service_account_info(st.secrets["service_account"], scopes=SCOPES)

def get_file_content_from_github(filename):
    """Get content of a file from GitHub repository."""
    try:
        github_base_url = "https://raw.githubusercontent.com/affteamgit/writeReview/main/templates/"
        file_url = f"{github_base_url}{filename}.txt"
        
        response = requests.get(file_url)
        response.raise_for_status()
        
        return response.text
        
    except Exception as e:
        print(f"Error reading file {filename} from GitHub: {str(e)}")
        return None

def get_selected_casino_data():
    creds = get_service_account_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    casino = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B1").execute().get("values", [[""]])[0][0].strip()
    rows = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B2:R").execute().get("values", [])
    sections = {
        "General": (2, 3, 4),
        "Payments": (5, 6, 7),
        "Games": (8, 9, 10),
        "Responsible Gambling": (11, 12, 13),
        "Bonuses": (14, 15, 16),
    }
    data = {}
    for sec, (mi, ti, si) in sections.items():
        main = "\n".join(r[mi] for r in rows if len(r) > mi and r[mi].strip())
        if ti is not None:
            top = "\n".join(r[ti] for r in rows if len(r) > ti and r[ti].strip())
        else:
            top = "[No top comparison available]"
        if si is not None:
            sim = "\n".join(r[si] for r in rows if len(r) > si and r[si].strip())
        else:
            sim = "[No similar comparison available]"
        data[sec] = {"main": main or "[No data provided]", "top": top, "sim": sim}
    return casino, data

# AI CLIENTS
client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

def call_openai(prompt):
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=800).choices[0].message.content.strip()

def call_grok(prompt):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK_API_KEY}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 800}
    j = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers).json()
    return j.get("choices", [{}])[0].get("message", {}).get("content", "[Grok failed]").strip()

def call_claude(prompt):
    return anthropic.messages.create(model="claude-sonnet-4-20250514", max_tokens=800, temperature=0.5, messages=[{"role": "user", "content": prompt}]).content[0].text.strip()

def parse_review_sections(content):
    """Parse review content into sections based on **Section Name** format."""
    section_headers = [
        "General",
        "Payments", 
        "Games",
        "Responsible Gambling",
        "Bonuses"
    ]
    
    lines = content.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if this line is a section header in **Section Name** format
        is_header = False
        for header in section_headers:
            if line_stripped == f"**{header}**":
                # Save previous section if exists
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_section = header
                current_content = []
                is_header = True
                break
        
        # If not a header, add to current content
        if not is_header:
            if current_section is None:
                # Skip content before the first section header
                continue
            current_content.append(line)
    
    # Don't forget the last section
    if current_section and current_content:
        sections.append({
            'title': current_section,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections

def rewrite_section(section_title, section_content):
    """Rewrite a single section using the fine-tuned model."""
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are Adam Gros, founder and editor-in-chief of Gamblineers, a seasoned crypto casino expert with over 10 years of experience. Your background is in mathematics and data analysis. You are a helpful assistant that rewrites content provided by the user - ONLY THROUGH YOUR TONE AND STYLE, YOU DO NOT CHANGE FACTS or ADD NEW FACTS. YOU REWRITE GIVEN FACTS IN YOUR OWN STYLE.\n\nYou write from a first-person singular perspective and speak directly to \"you,\" the reader.\n\nYour voice is analytical, witty, blunt, and honest-with a sharp eye for BS and a deep respect for data. You balance professionalism with dry humor. You call things as they are, whether good or bad, and never sugarcoat reviews.\n\nWriting & Style Rules\n- Always write in first-person singular (\"I\")\n- Speak directly to you, the reader\n- Keep sentences under 20 words\n- Never use em dashes or emojis\n- Never use fluff words like: \"fresh,\" \"solid,\" \"straightforward,\" \"smooth,\" \"game-changer\"\n- Avoid clichés: \"kept me on the edge of my seat,\" \"whether you're this or that,\" etc.\n- Bold key facts, bonuses, or red flags\n- Use short paragraphs (2–3 sentences max)\n- Use bullet points for clarity (pros/cons, bonuses, steps, etc.)\n- Tables are optional for comparisons\n- Be helpful without sounding preachy or salesy\n- If something sucks, say it. If it's good, explain why.\n\nTone\n- Casual but sharp\n- Witty, occasionally sarcastic (in good taste)\n- Confident, never condescending\n- Conversational, never robotic\n- Always honest-even when it hurts\n\nMission & Priorities\n- Save readers from scammy casinos and shady bonus terms\n- Transparency beats hype-user satisfaction > feature lists\n- Crypto usability matters\n- The site serves readers, not casinos\n- Highlight what others overlook-good or bad\n\nPersonality Snapshot\n- INTJ: Strategic, opinionated, allergic to buzzwords\n- Meticulous and detail-obsessed\n- Enjoys awkward silences and bad data being called out\n- Prefers dry humor and meaningful critiques."},
                {"role": "user", "content": section_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error rewriting {section_title}: {str(e)}"

def rewrite_review_with_adam(review_content):
    """Rewrite the entire review using Adam's voice, section by section."""
    sections = parse_review_sections(review_content)
    
    if not sections:
        # If no sections detected, rewrite as whole
        return rewrite_section("Full Review", review_content)
    
    rewritten_sections = []
    for section in sections:
        rewritten_content = rewrite_section(section['title'], section['content'])
        rewritten_sections.append(f"**{section['title']}**\n{rewritten_content}")
    
    return "\n\n".join(rewritten_sections)

def write_review_link_to_sheet(link):
    """Write the review link to cell B7 in the spreadsheet."""
    creds = get_service_account_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    body = {"values": [[link]]}
    sheets.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID, 
        range=f"{SHEET_NAME}!B7", 
        valueInputOption="RAW", 
        body=body
    ).execute()

def insert_parsed_text_with_formatting(docs_service, doc_id, review_text):
    # Parse the text into clean text and extract formatting positions
    plain_text = ""
    formatting_requests = []
    cursor = 1  # Google Docs uses 1-based index after the title

    pattern = r'(\*\*(.*?)\*\*|\[([^\]]+?)\]\((https?://[^\)]+)\))'
    last_end = 0

    for match in re.finditer(pattern, review_text):
        start, end = match.span()
        before_text = review_text[last_end:start]
        plain_text += before_text
        cursor_start = cursor + len(before_text)

        if match.group(2):  # Bold (**text**)
            bold_text = match.group(2)
            plain_text += bold_text
            formatting_requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": cursor_start, "endIndex": cursor_start + len(bold_text)},
                    "textStyle": {"bold": True},
                    "fields": "bold"
                }
            })
            cursor += len(before_text) + len(bold_text)

        elif match.group(3) and match.group(4):  # Link [text](url)
            link_text = match.group(3)
            url = match.group(4)
            plain_text += link_text
            formatting_requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": cursor_start, "endIndex": cursor_start + len(link_text)},
                    "textStyle": {"link": {"url": url}},
                    "fields": "link"
                }
            })
            cursor += len(before_text) + len(link_text)

        last_end = end

    remaining_text = review_text[last_end:]
    plain_text += remaining_text

    #  Insert clean plain text first
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": plain_text}}]}
    ).execute()

    title_line = plain_text.split('\n', 1)[0]
    title_start = 1
    title_end = title_start + len(title_line)

    formatting_requests.insert(0, {
    "updateParagraphStyle": {
        "range": {"startIndex": title_start, "endIndex": title_end},
        "paragraphStyle": {"namedStyleType": "TITLE"},
        "fields": "namedStyleType"
        }
    })

    # Apply inline bold & links
    if formatting_requests:
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": formatting_requests}
        ).execute()

    doc = docs_service.documents().get(documentId=doc_id).execute()
    header_requests = []
    section_titles = ["General", "Payments", "Games", "Responsible Gambling", "Bonuses"]

    for element in doc.get('body', {}).get('content', []):
        if 'paragraph' in element:
            paragraph = element['paragraph']
            paragraph_text = ''.join(
                elem['textRun']['content']
                for elem in paragraph.get('elements', [])
                if 'textRun' in elem
            ).strip()

            # Check if this is a section title
            if paragraph_text in section_titles:
                # Find the exact start and end from element indexes
                start_index = element.get('startIndex')
                end_index = element.get('endIndex')
                if start_index is not None and end_index is not None:
                    header_requests.append({
                        "updateTextStyle": {
                            "range": {"startIndex": start_index, "endIndex": end_index - 1},  # exclude trailing newline
                            "textStyle": {"bold": True, "fontSize": {"magnitude": 16, "unit": "PT"}},
                            "fields": "bold,fontSize"
                        }
                    })

    # Apply section headers formatting
    if header_requests:
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": header_requests}
        ).execute()

def create_google_doc_in_folder(docs_service, drive_service, folder_id, doc_title, review_text):
    doc_id = docs_service.documents().create(body={"title": doc_title}).execute()["documentId"]
    insert_parsed_text_with_formatting(docs_service, doc_id, review_text)

    file = drive_service.files().get(fileId=doc_id, fields="parents").execute()
    previous_parents = ",".join(file.get('parents', []))
    drive_service.files().update(fileId=doc_id, addParents=folder_id, removeParents=previous_parents, fields="id, parents").execute()
    return doc_id

def find_existing_doc(drive_service, folder_id, title):
    query = f"name='{title}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def main():
    st.set_page_config(page_title="Merged Review Generator", layout="centered", initial_sidebar_state="collapsed")
    
    # Initialize session state
    if 'review_completed' not in st.session_state:
        st.session_state.review_completed = False
        st.session_state.review_url = None
        st.session_state.casino_name = None
    
    # If review is already completed, show the success message
    if st.session_state.review_completed:
        st.success("Review successfully written & rewritten with Adam's voice, check the sheet :)")
        if st.session_state.review_url:
            st.info(f"Review link: {st.session_state.review_url}")
        
        # Add a button to generate a new review
        if st.button("Write Review", type="primary"):
            st.session_state.review_completed = False
            st.session_state.review_url = None
            st.session_state.casino_name = None
            st.rerun()
        return
    
    # Get casino name first to show in the interface
    try:
        user_creds = get_service_account_credentials()
        casino, _ = get_selected_casino_data()
        st.session_state.casino_name = casino
    except Exception as e:
        st.error(f"❌ Error loading casino data: {e}")
        return
    
    # Show casino name and generate button
    st.markdown(f"## Ready to write a review for: **{casino}**")
    st.markdown("The review will be written and then rewritten in Adam's voice before upload.")
    
    # Only generate review when button is clicked
    if st.button("Write Review", type="primary", use_container_width=True):
        # Show progress message
        progress_placeholder = st.empty()
        progress_placeholder.markdown("## Writing review, please wait...")
        
        try:
            docs_service = build("docs", "v1", credentials=user_creds)
            drive_service = build("drive", "v3", credentials=user_creds)

            price = requests.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest", headers={"Accepts": "application/json", "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY}, params={"symbol": "BTC", "convert": "USD"}).json().get("data", {}).get("BTC", {}).get("quote", {}).get("USD", {}).get("price")
            btc_str = f"1 BTC = ${price:,.2f}" if price else "[BTC price unavailable]"

            casino, secs = get_selected_casino_data()
            
            # Define section configurations
            section_configs = {
                "General": ("BaseGuidelinesClaude", "StructureTemplateGeneral", call_claude),
                "Payments": ("BaseGuidelinesClaude", "StructureTemplatePayments", call_claude),
                "Games": ("BaseGuidelinesClaude", "StructureTemplateGames", call_claude),
                "Responsible Gambling": ("BaseGuidelinesGrok", "StructureTemplateResponsible", call_grok),
                "Bonuses": ("BaseGuidelinesClaude", "StructureTemplateBonuses", call_claude),
            }

            # Get the prompt template from GitHub
            prompt_template = get_file_content_from_github("PromptTemplate")
            if not prompt_template:
                st.error("Error: Could not fetch prompt template from GitHub")
                return

            out = [f"{casino} review\n"]
            for sec, content in secs.items():
                guidelines_file, structure_file, fn = section_configs[sec]
                
                # Get guidelines and structure from GitHub
                guidelines = get_file_content_from_github(guidelines_file)
                structure = get_file_content_from_github(structure_file)
                
                if not guidelines or not structure:
                    st.error(f"Error: Could not fetch required files for section {sec}")
                    continue
                    
                prompt = prompt_template.format(
                    casino=casino,
                    section=sec,
                    guidelines=guidelines,
                    structure=structure,
                    main=content["main"],
                    top=content["top"],
                    sim=content["sim"],
                    btc_value=btc_str
                )
                
                review = fn(prompt)
                out.append(f"**{sec}**\n{review}\n")

            # Step 2: Rewrite with Adam's voice
            progress_placeholder.markdown("## Rewriting with Adam's voice...")
            
            initial_review = "\n".join(out)
            rewritten_review = rewrite_review_with_adam(initial_review)
            
            # Step 3: Upload to Google Drive
            progress_placeholder.markdown("## Uploading to Google Drive...")
            
            doc_title = f"{casino} Review"
            existing_doc_id = find_existing_doc(drive_service, FOLDER_ID, doc_title)

            if existing_doc_id:
                # Delete the old document
                drive_service.files().delete(fileId=existing_doc_id).execute()

            doc_id = create_google_doc_in_folder(docs_service, drive_service, FOLDER_ID, f"{casino} Review", rewritten_review)
            doc_url = f"https://docs.google.com/document/d/{doc_id}"
            
            # Write the review link to the spreadsheet
            write_review_link_to_sheet(doc_url)
            
            # Mark review as completed and store the URL
            st.session_state.review_completed = True
            st.session_state.review_url = doc_url
            
            # Clear progress message and show success
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()