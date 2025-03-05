import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import openai
import asyncio
import re
from sklearn.metrics.pairwise import cosine_similarity
import time

st.title("📊 obanai | Built at Grapevine")

# ----- Embedding Model Pricing Config -----
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_PRICING = {
    "text-embedding-3-small": {"pages_per_dollar": 62500, "tokens_per_page": 800},
    "text-embedding-3-large": {"pages_per_dollar": 9615, "tokens_per_page": 800},
    "text-embedding-ada-002": {"pages_per_dollar": 12500, "tokens_per_page": 800}
}
# Calculate tokens per dollar for the current model
tokens_per_dollar = EMBEDDING_MODEL_PRICING[EMBEDDING_MODEL]["pages_per_dollar"] * EMBEDDING_MODEL_PRICING[EMBEDDING_MODEL]["tokens_per_page"]

# ----- Define Synonyms for Robust Matching -----
YOE_SYNONYMS = ["yoe", "years", "experience"]
TITLE_SYNONYMS = ["job_title", "title", "role", "position", "designation"]
SALARY_SYNONYMS = ["total", "salary", "compensation", "ctc", "annual"]

def find_best_column_match(columns_list, synonyms):
    for col in columns_list:
        col_lower = col.lower().strip()
        for syn in synonyms:
            if syn in col_lower:
                return col
    return None

def detect_and_rename_columns(df):
    remaining_columns = list(df.columns)
    # Detect Year of Experience
    yoe_col = find_best_column_match(remaining_columns, YOE_SYNONYMS)
    if yoe_col:
        remaining_columns.remove(yoe_col)
    else:
        raise ValueError("Missing required column: Year of Experience")
    # Detect Job Title
    title_col = find_best_column_match(remaining_columns, TITLE_SYNONYMS)
    if title_col:
        remaining_columns.remove(title_col)
    else:
        raise ValueError("Missing required column: Job Title")
    # Detect Salary
    salary_col = find_best_column_match(remaining_columns, SALARY_SYNONYMS)
    if not salary_col:
        raise ValueError("Missing required column: Salary")
    remaining_columns.remove(salary_col)
    # Optionally detect Location column using synonyms
    location_synonyms = ["location", "city", "area"]
    location_col = find_best_column_match(remaining_columns, location_synonyms)
    new_names = {yoe_col: "YOE", title_col: "Job_Title", salary_col: "Salary"}
    if location_col:
        new_names[location_col] = "Location"
    # Additionally rename seniority_id and level_id if they exist
    if "seniority_id" in remaining_columns:
        new_names["seniority_id"] = "Seniority"
        remaining_columns.remove("seniority_id")
    if "level_id" in remaining_columns:
        new_names["level_id"] = "Level"
        remaining_columns.remove("level_id")
    return df.rename(columns=new_names)

def extract_yoe_from_string(job_title_string):
    if pd.isna(job_title_string):
        return None
    match = re.search(r'\((\d+\.?\d*)\s*yoe\)', job_title_string, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def clean_data(df, manual_yoe_column=None):
    if df.empty:
        raise ValueError("Uploaded CSV file is empty.")
    df = detect_and_rename_columns(df)
    if manual_yoe_column and manual_yoe_column in df.columns:
        df['YOE'] = df[manual_yoe_column]
    elif df['YOE'].isna().all():
        extracted = df['Job_Title'].apply(extract_yoe_from_string)
        if extracted.notna().any():
            df['YOE'] = extracted
    def convert_salary(val):
        try:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                cleaned_val = ''.join(c for c in val if c.isdigit() or c == '.')
                return float(cleaned_val)
        except (ValueError, TypeError):
            return np.nan
    df['Salary'] = df['Salary'].apply(convert_salary)
    df.dropna(subset=['Salary', 'Job_Title'], inplace=True)
    df['YOE'] = pd.to_numeric(df['YOE'], errors='coerce')
    df.dropna(subset=['YOE'], inplace=True)
    # Round YOE to whole number for display
    df['YOE'] = df['YOE'].round(0)
    return df

def clean_company_name(company):
    # Remove unwanted substring starting with " - query_result" (case-insensitive)
    return re.sub(r'\s*-\s*query_result.*$', '', company, flags=re.IGNORECASE)

def format_salary_entry(row):
    # For standard (viral) posts
    job_title = row['Job_Title']
    yoe = row['YOE']
    salary_in_lakhs = row['Salary'] / 100000
    if salary_in_lakhs >= 100:
        salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
    else:
        salary_str = f"{salary_in_lakhs:.1f} Lakhs"
    return f"{job_title} with {int(yoe)} YOE earns {salary_str}"

def format_salary_entry_custom(row):
    # For custom query posts (includes company info)
    job_title = row['Job_Title']
    job_title = re.sub(r'\s*-\s*query_result.*$', '', job_title, flags=re.IGNORECASE)
    yoe = row['YOE']
    salary_in_lakhs = row['Salary'] / 100000
    company = row["Company"] if "Company" in row else ""
    company = clean_company_name(company)
    if salary_in_lakhs >= 100:
        salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
    else:
        salary_str = f"{int(salary_in_lakhs)} LPA"
    return f"{job_title} at {company} with {int(yoe)} YOE earns {salary_str}"

def select_viral_entries(df):
    job_counts = df['Job_Title'].value_counts()
    top_titles = job_counts.head(5).index.tolist()
    selected_rows = []
    for title in top_titles:
        subset = df[df['Job_Title'] == title]
        rep_row = subset.sort_values('Salary', ascending=False).iloc[0]
        selected_rows.append(rep_row)
    return pd.DataFrame(selected_rows)

# ----- OPTIMIZED EMBEDDING FUNCTIONS -----

# Cache for storing computed embeddings (now returns token usage too)
@st.cache_data(ttl=3600)
def get_embeddings_batch(texts, api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        token_usage = response.usage.get("total_tokens", 0) if hasattr(response, 'usage') else 0
        return embeddings, token_usage, None
    except Exception as e:
        return None, 0, str(e)

@st.cache_data(ttl=3600)
def prepare_job_descriptions(df_json):
    """Prepare job descriptions for embedding"""
    df = pd.read_json(df_json)
    job_descriptions = []
    for _, row in df.iterrows():
        job_desc = f"{row['Job_Title']} with {int(row['YOE'])} YOE"
        if 'Location' in row and not pd.isna(row['Location']):
            job_desc += f" in {row['Location']}"
        job_descriptions.append(job_desc)
    return job_descriptions

def find_matching_jobs(query, df, embeddings, openai_api_key, top_n=5):
    """Find matching jobs using cosine similarity with precomputed embeddings"""
    start_time = time.time()
    
    # Get query embedding (single API call) and capture token usage
    query_embeddings, query_tokens, error = get_embeddings_batch([query], openai_api_key)
    if error:
        return None, None, None, f"Error getting query embedding: {error}"
    
    query_embedding = query_embeddings[0]
    query_embedding_np = np.array(query_embedding)
    job_embeddings_np = np.array(embeddings)
    
    similarities = cosine_similarity([query_embedding_np], job_embeddings_np)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    processing_time = time.time() - start_time
    return df.iloc[top_indices], similarities[top_indices], processing_time, query_tokens

def generate_custom_query_post(filtered_df, query, location=None, count=None):
    """
    Generates the custom query post text from embedding-filtered data.
    """
    if count is not None:
        selected_df = filtered_df.head(count)
    else:
        selected_df = filtered_df.head(5)
    
    companies = ", ".join([clean_company_name(comp) for comp in selected_df["Company"].unique()]) if "Company" in selected_df.columns else "Unknown Companies"
    n = len(selected_df)
    entries = [format_salary_entry_custom(row) for _, row in selected_df.iterrows()]

    header_info = f"in {location}" if location else f"for {companies}"
    post = (f"{n} Salaries matching '{query}' {header_info} -\n" +
            "\n".join(entries) +
            "\nWe share 5 new salaries from Grapevine every 12 hours.\nHit follow")
    return post

async def analyze_salaries(api_key, company_name, cleaned_df):
    # Standard (viral) post generation using Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    viral_df = select_viral_entries(cleaned_df)
    formatted_entries = [format_salary_entry(row) for _, row in viral_df.iterrows()]
    data_text = "\n".join(formatted_entries)
    try:
        final_prompt = f"""Create a social media post with EXACTLY 5 salaries.
STRICTLY adhere to this EXACT format. DO NOT DEVIATE.

5 Salaries of {company_name}

{data_text}

We share 5 new salaries from Grapevine every 12 hours.
Hit follow"""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, model.generate_content, final_prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

# ------------------ Streamlit UI ------------------

st.markdown("""
## About This App
This app helps you generate engaging social media posts about salary data. It offers two modes:
- Standard Post Generation: Creates viral-style posts using Gemini AI (requires Gemini API key)
- Custom Query Feature: Uses OpenAI embeddings to semantically search for relevant salary data (requires OpenAI API key)
""")

tab1, tab2 = st.tabs(["Standard Post Generation", "Custom Query"])

with tab1:
    st.header("Standard Post Generation (Viral Role Selection)")
    gemini_api_key = st.text_input("Enter your Gemini API key", type="password", key="gemini_key")
    uploaded_files = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True, key="standard_files")

    if uploaded_files and gemini_api_key:
        if st.button("1M+ Users .. Lessgo!"):
            all_results = {}
            for uploaded_file in uploaded_files:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    default_company_name = uploaded_file.name.split('.')[0]
                    cleaned_df = clean_data(df)
                    with st.spinner(f"Generating post for {default_company_name}..."):
                        result, error = asyncio.run(analyze_salaries(gemini_api_key, default_company_name, cleaned_df))
                    all_results[default_company_name] = (result, error)
                except Exception as e:
                    all_results[uploaded_file.name.split('.')[0]] = (None, str(e))
            for company_name, (result, error) in all_results.items():
                if error:
                    st.error(f"Error for {company_name}: {error}")
                else:
                    st.subheader(f"Generated Post for {company_name}")
                    st.code(result)
                    st.download_button(
                        f"Download Post for {company_name}",
                        result,
                        file_name=f"{company_name.lower()}_post.txt"
                    )
        
        # Individual File Processing:
        for uploaded_file in uploaded_files:
            try:
                st.markdown("---")
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                default_company_name = uploaded_file.name.split('.')[0]
                company_name = st.text_input(f"Company Name for {uploaded_file.name}",
                                            default_company_name,
                                            key=f"company_name_{uploaded_file.name}")
                st.write(f"Raw Data Preview for {uploaded_file.name}:")
                st.dataframe(df.head())
                column_options = df.columns.tolist()
                manual_yoe_column = st.selectbox("Select YOE column (optional)",
                                                options=["None"] + column_options,
                                                key=f"yoe_select_{uploaded_file.name}")
                manual_yoe_column = manual_yoe_column if manual_yoe_column != "None" else None
                cleaned_df = clean_data(df, manual_yoe_column)
                st.success(f"Cleaned {len(cleaned_df)} rows for {uploaded_file.name}!")
                if st.button(f"Generate Post for {company_name}", key=f"generate_{uploaded_file.name}"):
                    with st.spinner("Generating..."):
                        result, error = asyncio.run(analyze_salaries(gemini_api_key, company_name, cleaned_df))
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.subheader(f"Generated Post for {company_name}")
                        st.code(result)
                        st.download_button(
                            f"Download Post ({company_name})",
                            result,
                            file_name=f"{company_name.lower()}_post.txt"
                        )
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

    elif uploaded_files:
        st.warning("Please enter your Gemini API key for standard post generation.")
    elif gemini_api_key:
        st.info("Please upload at least one CSV file.")
    else:
        st.info("Enter your Gemini API key and upload CSV files to begin.")

with tab2:
    st.header("Custom Query Feature (Using OpenAI Embeddings)")
    st.markdown("""
    This feature uses OpenAI's text-embedding-3-small model to semantically match your query with relevant salary data.
    It understands natural language queries better than traditional filtering.
    """)
    
    openai_api_key = st.text_input("Enter your OpenAI API key (required for semantic search)", type="password", key="openai_key")
    uploaded_files_custom = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True, key="custom_files")
    
    # Process uploaded files and cache embeddings
    if uploaded_files_custom and openai_api_key:
        if "combined_df" not in st.session_state:
            st.session_state.combined_df = None
            st.session_state.job_embeddings = None
            st.session_state.job_descriptions = None
        
        file_names = [f.name for f in uploaded_files_custom]
        if ("processed_files" not in st.session_state or 
            st.session_state.processed_files != file_names):
            
            with st.spinner("Processing uploaded files..."):
                combined_data = []
                for uploaded_file in uploaded_files_custom:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)
                        default_company_name = uploaded_file.name.split('.')[0]
                        df["Company"] = default_company_name
                        cleaned_df = clean_data(df)
                        combined_data.append(cleaned_df)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    st.session_state.combined_df = combined_df
                    st.session_state.processed_files = file_names
                    
                    with st.spinner("Computing embeddings for all data (one-time operation)..."):
                        df_json = combined_df.to_json()
                        job_descriptions = prepare_job_descriptions(df_json)
                        
                        batch_size = 1000
                        all_embeddings = []
                        for i in range(0, len(job_descriptions), batch_size):
                            batch = job_descriptions[i:i+batch_size]
                            progress_text = f"Processing batch {i//batch_size + 1}/{(len(job_descriptions)-1)//batch_size + 1}"
                            with st.spinner(progress_text):
                                batch_embeddings, _, error = get_embeddings_batch(batch, openai_api_key)
                                if error:
                                    st.error(f"Error: {error}")
                                    break
                                all_embeddings.extend(batch_embeddings)
                        
                        if len(all_embeddings) == len(job_descriptions):
                            st.session_state.job_embeddings = all_embeddings
                            st.session_state.job_descriptions = job_descriptions
                            st.success(f"Successfully processed {len(combined_df)} entries and computed embeddings!")
        
        if st.session_state.combined_df is not None:
            st.write(f"Data ready: {len(st.session_state.combined_df)} salary entries")
            with st.expander("View Sample Data"):
                st.dataframe(st.session_state.combined_df.head())
                extra_cols = []
                if "Seniority" in st.session_state.combined_df.columns:
                    extra_cols.append("Seniority")
                if "Level" in st.session_state.combined_df.columns:
                    extra_cols.append("Level")
                if extra_cols:
                    st.write("Additional columns:")
                    st.dataframe(st.session_state.combined_df[extra_cols].head())
    
    query_text = st.text_area("Enter your natural language query (e.g., 'Senior software engineer with machine learning experience'):")
    count_slider = st.slider("Number of results to show", min_value=1, max_value=10, value=5)
    
    if query_text and st.session_state.get("combined_df") is not None and st.session_state.get("job_embeddings") is not None:
        if st.button("Run Semantic Search"):
            location_match = re.search(r'in\s+([A-Za-z ]+)', query_text, re.IGNORECASE)
            location = location_match.group(1).strip() if location_match else None
            
            with st.spinner("Searching for matching roles..."):
                filtered_df, similarities, processing_time, query_tokens = find_matching_jobs(
                    query_text, 
                    st.session_state.combined_df, 
                    st.session_state.job_embeddings, 
                    openai_api_key, 
                    top_n=count_slider
                )
            
            if filtered_df is None:
                st.error("Error processing query. Please try again.")
            elif filtered_df.empty:
                st.warning("No matching entries found for the query.")
            else:
                st.success(f"Found matching entries in {processing_time:.2f} seconds!")
                st.subheader("Matching Results")
                display_df = filtered_df.copy()
                display_df['Similarity'] = similarities
                display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_df[['Company', 'Job_Title', 'YOE', 'Salary', 'Similarity']])
                
                result = generate_custom_query_post(filtered_df, query_text, location, count_slider)
                st.subheader("Generated Post")
                st.code(result)
                st.download_button(
                    "Download Custom Query Post", 
                    result, 
                    file_name="custom_query_post.txt"
                )
                # Display token usage and cost information
                cost = query_tokens / tokens_per_dollar
                st.info(f"Embedding Query used {query_tokens} tokens, costing approximately ${cost:.8f}")
    elif not st.session_state.get("combined_df") and uploaded_files_custom:
        st.warning("Still processing data. Please wait...")
    elif not uploaded_files_custom:
        st.info("Please upload CSV files for the custom query feature.")
