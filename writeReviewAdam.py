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
import concurrent.futures
from typing import Dict, Tuple

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

def get_all_templates():
    """Fetch all templates at once with parallel processing"""
    templates = {}
    files = [
        'PromptTemplate', 
        'BaseGuidelinesClaude', 
        'BaseGuidelinesGrok', 
        'StructureTemplateGeneral', 
        'StructureTemplatePayments', 
        'StructureTemplateGames', 
        'StructureTemplateResponsible', 
        'StructureTemplateBonuses'
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(get_file_content_from_github, filename): filename 
                         for filename in files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                templates[filename] = future.result()
            except Exception as e:
                print(f"Error fetching template {filename}: {e}")
                templates[filename] = None
    
    return templates

def get_selected_casino_data():
    creds = get_service_account_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    casino = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B1").execute().get("values", [[""]])[0][0].strip()
    rows = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B2:S").execute().get("values", [])
    sections = {
        "General": (2, 3, 4),
        "Payments": (5, 6, 7),
        "Games": (8, 9, 10),
        "Responsible Gambling": (11, 12, 13),
        "Bonuses": (14, 15, 16),
    }
    data = {}
    comments_column = 17  # Column S (0-indexed)
    
    # Extract comments from column S
    all_comments = "\n".join(r[comments_column] for r in rows if len(r) > comments_column and r[comments_column].strip())
    
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
    
    return casino, data, all_comments

def get_cached_casino_data():
    """Get casino data without caching to prevent tone interference"""
    return get_selected_casino_data()

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

def sort_comments_by_section(comments):
    """Use AI to intelligently sort comments by section."""
    if not comments or not comments.strip():
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}
    
    prompt = f"""Please analyze the following feedback comments and sort them by the casino review sections they belong to.

Comments:
{comments}

Sections:
- General (overall casino experience, VPN support, reputation, establishment date, etc.)
- Payments (deposits, withdrawals, KYC, payment methods, processing times, etc.)
- Games (game selection, slots, table games, live casino, game providers, etc.)
- Responsible Gambling (limits, self-exclusion, problem gambling tools, etc.)
- Bonuses (welcome bonus, promotions, bonus terms, wagering requirements, etc.)

For each section, return ONLY the comments that belong to that section. If no comments belong to a section, leave it empty.

Format your response exactly like this:
**General**
[relevant comments here or leave empty]

**Payments**
[relevant comments here or leave empty]

**Games**
[relevant comments here or leave empty]

**Responsible Gambling**
[relevant comments here or leave empty]

**Bonuses**
[relevant comments here or leave empty]"""
    
    try:
        response = call_claude(prompt)
        # Parse the response into a dictionary
        sections = {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                section_name = line[2:-2]  # Remove ** from both ends
                if section_name in sections:
                    current_section = section_name
            elif current_section and line:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        return sections
    except Exception as e:
        print(f"Error sorting comments: {e}")
        # Fallback: return empty sections
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}

def incorporate_comments_into_review(review_content, comments):
    """Use AI to incorporate relevant comments into the review before Adam's rewrite."""
    if not comments.strip():
        return review_content
    
    # Parse the review into sections first to maintain structure
    sections = parse_review_sections(review_content)
    
    if not sections:
        # If parsing fails, just return original content
        print("Failed to parse sections for comment incorporation, returning original")
        return review_content
    
    print(f"Incorporating comments into {len(sections)} sections")
    
    # For each section, ask AI to incorporate relevant comments
    updated_sections = []
    
    # Get the title (first line before sections)
    lines = review_content.split('\n')
    title = lines[0] if lines else ""
    
    for section in sections:
        section_title = section['title']
        section_content = section['content']
        
        # Ask AI to incorporate comments for this specific section
        prompt = f"""You are incorporating feedback comments into a specific section of a casino review.

Section: {section_title}
Current content:
{section_content}

All available comments:
{comments}

Please:
1. Look for any comments that specifically mention "{section_title}" or are clearly about this section
2. If you find relevant comments, incorporate that information into the section content
3. If no comments are relevant to this section, return the original content unchanged
4. Only add factual information mentioned in the comments - don't make up new facts
5. Keep the writing style consistent with the original content
6. Do NOT include the section header in your response - only return the updated content

Return only the updated section content (without the **{section_title}** header):"""
        
        try:
            updated_content = call_claude(prompt)
            updated_sections.append(f"**{section_title}**\n{updated_content}")
            print(f"Successfully incorporated comments for section: {section_title}")
        except Exception as e:
            print(f"Error incorporating comments for {section_title}: {e}")
            # Fallback to original content for this section
            updated_sections.append(f"**{section_title}**\n{section_content}")
    
    # Reconstruct the full review
    result = title + "\n\n" + "\n\n".join(updated_sections)
    print("Comment incorporation completed successfully")
    return result

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
        print(f"Rewriting section: {section_title}")
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are Adam Gros, founder and editor-in-chief of Gamblineers, a seasoned crypto casino expert with over 10 years of experience. Your background is in mathematics and data analysis. You are a helpful assistant that rewrites content provided by the user - ONLY THROUGH YOUR TONE AND STYLE, YOU DO NOT CHANGE FACTS or ADD NEW FACTS. YOU REWRITE GIVEN FACTS IN YOUR OWN STYLE.\n\nYou write from a first-person singular perspective and speak directly to \"you,\" the reader.\n\nYour voice is analytical, witty, blunt, and honest-with a sharp eye for BS and a deep respect for data. You balance professionalism with dry humor. You call things as they are, whether good or bad, and never sugarcoat reviews.\n\nWriting & Style Rules\n- Always write in first-person singular (\"I\")\n- Speak directly to you, the reader\n- Keep sentences under 20 words\n- Never use em dashes or emojis\n- Never use fluff words like: \"fresh,\" \"solid,\" \"straightforward,\" \"smooth,\" \"game-changer\"\n- Avoid clichés: \"kept me on the edge of my seat,\" \"whether you're this or that,\" etc.\n- Bold key facts, bonuses, or red flags\n- Use short paragraphs (2–3 sentences max)\n- Use bullet points for clarity (pros/cons, bonuses, steps, etc.)\n- Tables are optional for comparisons\n- Be helpful without sounding preachy or salesy\n- If something sucks, say it. If it's good, explain why.\n\nTone\n- Casual but sharp\n- Witty, occasionally sarcastic (in good taste)\n- Confident, never condescending\n- Conversational, never robotic\n- Always honest-even when it hurts\n\nMission & Priorities\n- Save readers from scammy casinos and shady bonus terms\n- Transparency beats hype-user satisfaction > feature lists\n- Crypto usability matters\n- The site serves readers, not casinos\n- Highlight what others overlook-good or bad\n\nPersonality Snapshot\n- INTJ: Strategic, opinionated, allergic to buzzwords\n- Meticulous and detail-obsessed\n- Enjoys awkward silences and bad data being called out\n- Prefers dry humor and meaningful critiques."},
                {"role": "user", "content": section_content}
            ],
            timeout=30  # Reduced timeout to 30 seconds
        )
        print(f"Successfully rewrote section: {section_title}")
        return response.choices[0].message.content
    except Exception as error:
        error_msg = f"Fine-tuned model failed for {section_title}: {error}"
        print(error_msg)
        return f"[Error rewriting {section_title}]\n{section_content}"

def generate_overview_section(casino_name, keyword, main_points):
    """Generate Overview section using Adam's fine-tuned model."""
    try:
        print("Generating Overview section with Adam's voice...")
        
        # Create prompt for overview generation
        overview_prompt = f"""Write an engaging overview/introduction for a {casino_name} casino review. Use the following details:

Keyword/Theme: {keyword}

Main points to cover:
{main_points}

Context: This overview will introduce a comprehensive review that covers General info, Payments, Games, Responsible Gambling, and Bonuses sections. 

Write a compelling 2-3 paragraph introduction that:
1. Hooks the reader with the keyword/theme
2. Touches on the main points provided
3. Sets expectations for what the full review will cover
4. Maintains your signature analytical and honest approach

Do not repeat information that will be covered in detail in other sections - this should be a high-level introduction that draws readers in."""

        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are Adam Gros, founder and editor-in-chief of Gamblineers, a seasoned crypto casino expert with over 10 years of experience. Your background is in mathematics and data analysis. You are a helpful assistant that writes content in your distinctive voice and style.\n\nYou write from a first-person singular perspective and speak directly to \"you,\" the reader.\n\nYour voice is analytical, witty, blunt, and honest-with a sharp eye for BS and a deep respect for data. You balance professionalism with dry humor. You call things as they are, whether good or bad, and never sugarcoat reviews.\n\nWriting & Style Rules\n- Always write in first-person singular (\"I\")\n- Speak directly to you, the reader\n- Keep sentences under 20 words\n- Never use em dashes or emojis\n- Never use fluff words like: \"fresh,\" \"solid,\" \"straightforward,\" \"smooth,\" \"game-changer\"\n- Avoid clichés: \"kept me on the edge of my seat,\" \"whether you're this or that,\" etc.\n- Bold key facts, bonuses, or red flags\n- Use short paragraphs (2–3 sentences max)\n- Use bullet points for clarity (pros/cons, bonuses, steps, etc.)\n- Tables are optional for comparisons\n- Be helpful without sounding preachy or salesy\n- If something sucks, say it. If it's good, explain why.\n\nTone\n- Casual but sharp\n- Witty, occasionally sarcastic (in good taste)\n- Confident, never condescending\n- Conversational, never robotic\n- Always honest-even when it hurts"},
                {"role": "user", "content": overview_prompt}
            ],
            timeout=30
        )
        
        overview_content = response.choices[0].message.content.strip()
        print("Successfully generated Overview section")
        return f"**Overview**\n{overview_content}"
        
    except Exception as error:
        error_msg = f"Failed to generate Overview section: {error}"
        print(error_msg)
        return f"**Overview**\n[Error generating Overview section: {error}]"

def rewrite_review_with_adam(review_content):
    """Rewrite the entire review using Adam's voice, section by section."""
    try:
        print("Starting Adam's rewrite process...")
        sections = parse_review_sections(review_content)
        
        if not sections:
            print("No sections detected, rewriting as whole")
            # If no sections detected, rewrite as whole
            return rewrite_section("Full Review", review_content)
        
        print(f"Found {len(sections)} sections to rewrite")
        rewritten_sections = []
        
        for i, section in enumerate(sections, 1):
            print(f"Processing section {i}/{len(sections)}: {section['title']}")
            rewritten_content = rewrite_section(section['title'], section['content'])
            
            # If there was an error, still include it to avoid breaking the flow
            if rewritten_content.startswith("[Error rewriting"):
                print(f"Failed to rewrite {section['title']}, using original content")
                # Use original content if rewrite fails
                rewritten_sections.append(f"**{section['title']}**\n{section['content']}")
            else:
                rewritten_sections.append(f"**{section['title']}**\n{rewritten_content}")
        
        print("Adam's rewrite process completed successfully")
        return "\n\n".join(rewritten_sections)
        
    except Exception as e:
        error_msg = f"Fatal error in rewrite_review_with_adam: {str(e)}"
        print(error_msg)
        # Return original content if everything fails
        return f"[Rewrite failed - using original content]\n\n{review_content}"

def generate_section(section_data: Tuple) -> str:
    """Generate a single section review"""
    sec, content, templates, sorted_comments, casino, btc_str = section_data
    
    # Define section configurations
    section_configs = {
        "General": ("BaseGuidelinesClaude", "StructureTemplateGeneral", call_claude),
        "Payments": ("BaseGuidelinesClaude", "StructureTemplatePayments", call_claude),
        "Games": ("BaseGuidelinesClaude", "StructureTemplateGames", call_claude),
        "Responsible Gambling": ("BaseGuidelinesGrok", "StructureTemplateResponsible", call_grok),
        "Bonuses": ("BaseGuidelinesClaude", "StructureTemplateBonuses", call_claude),
    }
    
    try:
        guidelines_file, structure_file, fn = section_configs[sec]
        
        # Get templates from cached data
        guidelines = templates.get(guidelines_file)
        structure = templates.get(structure_file)
        prompt_template = templates.get('PromptTemplate')
        
        if not guidelines or not structure or not prompt_template:
            return f"**{sec}**\n[Error: Missing templates for section {sec}]\n"
        
        # Get comments for this specific section
        section_comments = ""
        if sorted_comments.get(sec, "").strip():
            section_comments = f"\n\nAdditional feedback to incorporate: {sorted_comments[sec]}"
        
        prompt = prompt_template.format(
            casino=casino,
            section=sec,
            guidelines=guidelines,
            structure=structure,
            main=content["main"] + section_comments,
            top=content["top"],
            sim=content["sim"],
            btc_value=btc_str
        )
        
        review = fn(prompt)
        return f"**{sec}**\n{review}\n"
        
    except Exception as e:
        print(f"Error generating section {sec}: {e}")
        return f"**{sec}**\n[Error generating section: {str(e)}]\n"

def generate_sections_parallel(casino: str, secs: Dict, sorted_comments: Dict, templates: Dict, btc_str: str) -> list:
    """Generate all sections in parallel while maintaining order"""
    
    # Prepare data for each section
    section_data = [
        (sec, content, templates, sorted_comments, casino, btc_str)
        for sec, content in secs.items()
    ]
    
    # Generate sections in parallel with max 3 workers to avoid API rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks and maintain order
        future_to_section = {
            executor.submit(generate_section, data): data[0] 
            for data in section_data
        }
        
        # Collect results in the original order
        results = {}
        for future in concurrent.futures.as_completed(future_to_section):
            section_name = future_to_section[future]
            try:
                results[section_name] = future.result()
            except Exception as e:
                print(f"Error in parallel generation for {section_name}: {e}")
                results[section_name] = f"**{section_name}**\n[Error: {str(e)}]\n"
    
    # Return results in the original section order
    section_order = ["General", "Payments", "Games", "Responsible Gambling", "Bonuses"]
    return [results[sec] for sec in section_order if sec in results]

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
    section_titles = ["Overview", "General", "Payments", "Games", "Responsible Gambling", "Bonuses"]

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
        st.session_state.rewritten_review = None
        st.session_state.awaiting_overview = False
    
    # If review is completed and awaiting overview input
    if st.session_state.awaiting_overview and st.session_state.rewritten_review:
        st.markdown(f"## Review Complete! Now add the Overview section for: **{st.session_state.casino_name}**")
        
        # Show the completed review for reference
        with st.expander("📖 View Completed Review (for reference)", expanded=False):
            st.markdown(st.session_state.rewritten_review)
        
        st.markdown("### Add Overview Section")
        st.markdown("Please provide a keyword/theme and main points for the introduction:")
        
        # Input fields for overview
        keyword = st.text_input("Keyword/Theme (e.g., 'Innovation', 'Reliability', 'User Experience')", 
                               placeholder="Enter the main theme/keyword for this casino")
        
        main_points = st.text_area("Main Points (2-3 key points to highlight in the overview)", 
                                  placeholder="• Strong crypto integration\n• Excellent customer support\n• Wide game variety",
                                  height=120)
        
        # Generate overview and finalize
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Generate Overview & Post to Google Docs", type="primary", disabled=not (keyword and main_points)):
                if keyword and main_points:
                    try:
                        # Generate overview section
                        st.info("🔄 Generating Overview section with Adam's voice...")
                        overview_section = generate_overview_section(
                            st.session_state.casino_name, 
                            keyword, 
                            main_points
                        )
                        
                        # Combine overview with the rest of the review - Overview goes first
                        title_line = f"{st.session_state.casino_name} review"
                        final_review = f"{title_line}\n\n{overview_section}\n\n{st.session_state.rewritten_review}"
                        
                        # Post to Google Docs
                        st.info("📤 Uploading to Google Drive...")
                        user_creds = get_service_account_credentials()
                        docs_service = build("docs", "v1", credentials=user_creds)
                        drive_service = build("drive", "v3", credentials=user_creds)
                        
                        doc_title = f"{st.session_state.casino_name} Review"
                        existing_doc_id = find_existing_doc(drive_service, FOLDER_ID, doc_title)

                        if existing_doc_id:
                            drive_service.files().delete(fileId=existing_doc_id).execute()

                        doc_id = create_google_doc_in_folder(docs_service, drive_service, FOLDER_ID, doc_title, final_review)
                        doc_url = f"https://docs.google.com/document/d/{doc_id}"
                        
                        # Write the review link to the spreadsheet
                        write_review_link_to_sheet(doc_url)
                        
                        # Mark as completed
                        st.session_state.review_completed = True
                        st.session_state.review_url = doc_url
                        st.session_state.awaiting_overview = False
                        st.session_state.rewritten_review = None
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error finalizing review: {e}")
        
        with col2:
            if st.button("Skip Overview (Post without Overview)", type="secondary"):
                try:
                    # Post to Google Docs without overview - using exact original workflow
                    st.info("📤 Uploading to Google Drive...")
                    user_creds = get_service_account_credentials()
                    docs_service = build("docs", "v1", credentials=user_creds)
                    drive_service = build("drive", "v3", credentials=user_creds)
                    
                    doc_title = f"{st.session_state.casino_name} Review"
                    existing_doc_id = find_existing_doc(drive_service, FOLDER_ID, doc_title)

                    if existing_doc_id:
                        drive_service.files().delete(fileId=existing_doc_id).execute()

                    # Use original review format - exactly as it was before
                    final_review = f"{st.session_state.casino_name} review\n\n{st.session_state.rewritten_review}"
                    doc_id = create_google_doc_in_folder(docs_service, drive_service, FOLDER_ID, doc_title, final_review)
                    doc_url = f"https://docs.google.com/document/d/{doc_id}"
                    
                    # Write the review link to the spreadsheet
                    write_review_link_to_sheet(doc_url)
                    
                    # Mark as completed
                    st.session_state.review_completed = True
                    st.session_state.review_url = doc_url
                    st.session_state.awaiting_overview = False
                    st.session_state.rewritten_review = None
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error posting review: {e}")
        
        return
    
    # If review is already completed, show the success message
    if st.session_state.review_completed:
        st.success("Review successfully written & rewritten with Adam's voice, check the sheet :)")
        if st.session_state.review_url:
            st.info(f"Review link: {st.session_state.review_url}")
        
        # Add a button to generate a new review
        if st.button("Write New Review", type="primary"):
            st.session_state.review_completed = False
            st.session_state.review_url = None
            st.session_state.casino_name = None
            st.session_state.rewritten_review = None
            st.session_state.awaiting_overview = False
            st.rerun()
        return
    
    # Get casino name first to show in the interface
    try:
        user_creds = get_service_account_credentials()
        casino, _, _ = get_cached_casino_data()
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

            # Load all data in parallel
            progress_placeholder.markdown("## Loading templates and data...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all data loading tasks
                templates_future = executor.submit(get_all_templates)
                casino_data_future = executor.submit(get_cached_casino_data)
                btc_future = executor.submit(
                    lambda: requests.get(
                        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                        headers={"Accepts": "application/json", "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY},
                        params={"symbol": "BTC", "convert": "USD"}
                    ).json().get("data", {}).get("BTC", {}).get("quote", {}).get("USD", {}).get("price")
                )
                
                # Collect results
                templates = templates_future.result()
                casino, secs, comments = casino_data_future.result()
                price = btc_future.result()
            
            btc_str = f"1 BTC = ${price:,.2f}" if price else "[BTC price unavailable]"
            
            # Check if all required templates were loaded
            required_templates = ['PromptTemplate', 'BaseGuidelinesClaude', 'BaseGuidelinesGrok']
            missing_templates = [t for t in required_templates if not templates.get(t)]
            if missing_templates:
                st.error(f"Error: Could not fetch required templates: {', '.join(missing_templates)}")
                return
            
            # Sort comments by section using AI
            progress_placeholder.markdown("## Sorting comments by section...")
            sorted_comments = sort_comments_by_section(comments)
            
            # Generate all sections in parallel
            progress_placeholder.markdown("## Generating review sections in parallel...")
            parallel_results = generate_sections_parallel(casino, secs, sorted_comments, templates, btc_str)
            
            # Combine results
            out = [f"{casino} review\n"] + parallel_results

            # Step 2: Rewrite with Adam's voice
            progress_placeholder.markdown("## Rewriting with Adam's voice...")
            
            initial_review = "\n".join(out)
            
            rewritten_review = rewrite_review_with_adam(initial_review)
            
            # Step 3: Store rewritten review and prompt for Overview input
            st.session_state.rewritten_review = rewritten_review
            st.session_state.awaiting_overview = True
            st.session_state.casino_name = casino
            
            # Clear progress message and show overview input screen
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()